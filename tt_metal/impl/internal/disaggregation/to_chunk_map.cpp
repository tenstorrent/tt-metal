// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/to_chunk_map.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

#include <internal/disaggregation/noc_addr.hpp>

namespace tt::tt_metal::internal::disaggregation {

using tt::constants::BFLOAT8_B_TILE_HW;
using tt::constants::TILE_WIDTH;

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_metal::distributed::MeshMapperConfig;

// The mesh geometry + tensor facts the addresser derives ONCE per cache (instead of reading them from
// bespoke KvLayoutSpec fields): the seq axis and feature width off the tensor's NdShardSpec, and the
// CP/TP mesh extents + head-shard axis off the tensor's TensorTopology.
struct Derived {
    uint32_t seq_axis = 0;
    uint64_t f = 1;                    // per-token feature width
    uint32_t tpc = TILE_WIDTH;         // migration granule == the shard's seq-block size
    uint32_t sp_dim = 1;              // extent of the seq-shard (CP) mesh axis
    uint32_t mesh_cols = 1;          // mesh_size / sp_dim (TP / replica fan-out)
    uint32_t mesh_rows = 1;
    std::optional<uint32_t> head_axis;  // tensor axis a non-seq mesh axis shards (GQA head)
    uint32_t n_heads = 1;
    uint32_t num_banks = 0;             // set from num_dram_banks(arch) in derive()
    std::vector<uint32_t> banks;        // resolved bank permutation (arch + bank_order), size num_banks
};

// The active bank ordering resolved for the cache's arch: the OPTIMAL permutation or the identity
// round-robin, sized to the arch's DRAM bank count (both arch facts, via num_dram_banks/optimal_bank_order).
std::vector<uint32_t> bank_table(tt::ARCH arch, BankOrder order) {
    const uint32_t n = num_dram_banks(arch);
    std::vector<uint32_t> banks(n);
    if (order == BankOrder::Optimal) {
        const auto opt = optimal_bank_order(arch);
        TT_FATAL(opt.size() >= n, "optimal_bank_order has fewer entries ({}) than DRAM banks ({})", opt.size(), n);
        for (uint32_t i = 0; i < n; ++i) {
            banks[i] = opt[i];
        }
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            banks[i] = i;
        }
    }
    return banks;
}

Derived derive(const CacheConfig& config) {
    Derived d;
    const KvLayoutSpec& spec = config.spec;
    d.num_banks = num_dram_banks(spec.arch);
    d.banks = bank_table(spec.arch, config.policy.bank_order);
    const auto& shape = spec.tensor.logical_shape();

    const std::optional<uint32_t> seq = spec.sequence_axis();
    d.seq_axis = seq.value_or(0);
    d.f = feature_width(spec.tensor);
    if (seq.has_value()) {
        const auto& nd = spec.tensor.memory_config().nd_shard_spec();
        d.tpc = nd.has_value() ? static_cast<uint32_t>(nd->shard_shape[static_cast<int>(*seq)]) : TILE_WIDTH;
    }

    const auto& ms = config.topology.distribution_shape();
    const auto& placements = config.topology.placements();
    uint32_t mesh_size = 1;
    for (size_t i = 0; i < ms.dims(); ++i) {
        mesh_size *= ms[static_cast<int>(i)];
    }
    for (size_t i = 0; i < placements.size(); ++i) {
        if (const auto* s = std::get_if<MeshMapperConfig::Shard>(&placements[i])) {
            if (seq.has_value() && static_cast<uint32_t>(s->dim) == *seq) {
                d.sp_dim = ms[static_cast<int>(i)];
            } else {
                d.head_axis = static_cast<uint32_t>(s->dim);
            }
        }
    }
    d.mesh_cols = std::max(1u, mesh_size / std::max(1u, d.sp_dim));
    d.mesh_rows = ms.dims() > 0 ? ms[0] : 1u;
    d.n_heads = d.head_axis.has_value() ? shape[static_cast<int>(*d.head_axis)] : 1u;
    return d;
}

// A single resolved chunk: physical bank + per-bank offset + the mesh (row, col) coordinates that
// hold its replicas.
struct Located {
    uint32_t bank_id = 0;
    uint64_t offset = 0;
    std::vector<std::pair<uint32_t, uint32_t>> coords;  // (row, col) mesh coordinates
};

// MLA CP ownership: round-robin over sp devices with a per-device chunk stride.
std::pair<uint32_t, uint32_t> cp_mla_stride(const Derived& d, const GenerationPolicy& policy, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = d.sp_dim;
    const uint32_t round_idx = position / (dcs * sp_dim);
    const uint32_t in_round = position % (dcs * sp_dim);
    const uint32_t sp_device_idx = in_round / dcs;
    const uint32_t in_chunk = in_round % dcs;
    const uint32_t owner = (sp_device_idx + policy.sp_origin.get()) % sp_dim;
    return {owner, round_idx * dcs + in_chunk};
}

// GLM CP ownership: owner = chunk_index % num_devices.
std::pair<uint32_t, uint32_t> cp_chunk_modulo(const Derived& d, const GenerationPolicy& policy, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = d.sp_dim;
    if (sp_dim <= 1 || dcs == 0) {
        return {0, position};
    }
    const uint32_t chunk_index = position / dcs;
    const uint32_t owner = (chunk_index + policy.sp_origin.get()) % sp_dim;
    const uint32_t local_pos = (chunk_index / sp_dim) * dcs + (position % dcs);
    return {owner, local_pos};
}

Located locate_one(
    const KvLayoutSpec& spec,
    const GenerationPolicy& policy,
    const Derived& d,
    std::optional<uint32_t> head,
    uint32_t position,
    uint32_t slot,
    uint64_t base,
    const MapGeometry& geom) {
    const uint32_t tpc = d.tpc;
    const uint32_t kcs = policy.k_chunk_size.get();
    const BankScheme scheme = policy.bank_scheme;
    const auto& banks = d.banks;

    const uint64_t f = d.f;
    const uint32_t sp_dim = d.sp_dim;
    const uint32_t mesh_cols = d.mesh_cols;
    const uint32_t per_dev_seq = geom.max_seq_len / sp_dim;

    uint32_t dcs = policy.device_chunk_size.has_value() ? policy.device_chunk_size->get() : kcs * d.num_banks;

    const auto& shape = spec.tensor.logical_shape();

    // ---- MLA_SHARD: CP stride + shard_id stacking + OPTIMAL perm over num_banks ----
    if (scheme == BankScheme::MlaShard) {
        const uint32_t csb = chunk_size_bytes(spec.tensor, tpc);
        const auto [owner, local_pos] = cp_mla_stride(d, policy, position, dcs);
        const uint32_t chunks_per_slot = per_dev_seq / kcs;
        const uint32_t local_chunk = local_pos / kcs;
        const uint32_t in_chunk = local_pos % kcs;
        // Slot folds into the physical shard index (paging, if any, is a separate block-table contract).
        const uint32_t shard_id = slot * chunks_per_slot + local_chunk;
        const uint32_t shard_size_b = (kcs / tpc) * csb;
        const uint32_t bank_id = banks[shard_id % d.num_banks];
        const uint64_t off =
            base + static_cast<uint64_t>(shard_id / d.num_banks) * shard_size_b + (in_chunk / tpc) * csb;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back(owner, c);
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK_CYCLIC: minimax index_k, column-split by idx_cp, round-robin blocks over banks ----
    if (scheme == BankScheme::BlockCyclic) {
        const uint32_t idx_cp = policy.idx_cp.get();
        const uint32_t block = kcs;
        const uint32_t idx_row_bytes = static_cast<uint32_t>((f / TILE_WIDTH) * BFLOAT8_B_TILE_HW);
        const uint32_t n_blocks = per_dev_seq / block;
        const uint32_t n_blocks_dev = (n_blocks + idx_cp - 1) / idx_cp;
        const uint32_t bph = policy.banks_per_head.get() ? policy.banks_per_head.get() : d.num_banks;
        const uint32_t num_banks = std::min(bph, n_blocks_dev);
        const uint32_t blocks_per_bank = n_blocks_dev / num_banks;
        const uint32_t gb = position / block;
        const uint32_t dev = gb % idx_cp;
        const uint32_t local_position = (gb / idx_cp) * block + position % block;
        const uint32_t global_block = slot * blocks_per_bank * num_banks + local_position / block;
        const uint32_t bank_id = banks[global_block % num_banks];
        const uint32_t local_block = global_block / num_banks;
        const uint32_t tile_row = (local_block * block + (local_position % block)) / TILE_WIDTH;
        const uint64_t off = base + static_cast<uint64_t>(tile_row) * idx_row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t r = 0; r < d.mesh_rows; ++r) {
            for (uint32_t c = 0; c < mesh_cols; ++c) {
                if (c % idx_cp == dev) {
                    coords.emplace_back(0u, r * mesh_cols + c);
                }
            }
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK with GQA group: minimax K/V height-sharded, group -> mesh row block ----
    if (scheme == BankScheme::Block && policy.group.has_value()) {
        const uint32_t bph = policy.banks_per_head.get() ? policy.banks_per_head.get() : d.num_banks;
        const uint32_t row_bytes = static_cast<uint32_t>((f / TILE_WIDTH) * BFLOAT8_B_TILE_HW);
        const uint32_t st_pb = (per_dev_seq / TILE_WIDTH) / bph;
        const uint32_t chunk = position / kcs;
        const uint32_t off_in_chunk = position % kcs;
        const uint32_t bank_slice = chunk % bph;
        const uint32_t within = (chunk / bph) * (kcs / TILE_WIDTH) + off_in_chunk / TILE_WIDTH;
        const uint32_t bank_id = banks[bank_slice];
        const uint64_t off =
            base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back(0u, policy.group->get() * mesh_cols + c);
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK / CYCLIC per-head GQA: head -> TP shard (gpt-oss) ----
    if ((scheme == BankScheme::Block || scheme == BankScheme::Cyclic) && d.head_axis.has_value()) {
        TT_FATAL(head.has_value(), "per-head scheme requires an enumerated head");
        const uint32_t n_heads = d.n_heads;
        const uint32_t n_heads_per_dev = std::max(1u, n_heads / (mesh_cols * std::max(1u, sp_dim)));
        const uint32_t bph = policy.banks_per_head.get() ? policy.banks_per_head.get() : d.num_banks;
        const uint64_t dht = f / TILE_WIDTH;
        const uint32_t row_bytes = static_cast<uint32_t>(dht * BFLOAT8_B_TILE_HW);
        const uint32_t chip = *head / n_heads_per_dev;
        const uint32_t local_head = *head % n_heads_per_dev;
        const uint32_t seq_extent = shape[static_cast<int>(d.seq_axis)];
        const uint32_t st_pb = (seq_extent / TILE_WIDTH) / bph;
        const uint32_t sk_chunk_t = kcs / TILE_WIDTH;
        const uint32_t tile_row = position / TILE_WIDTH;
        uint32_t bank_slice = 0;
        uint32_t within = 0;
        if (scheme == BankScheme::Cyclic) {
            const uint32_t chunk = tile_row / sk_chunk_t;
            bank_slice = chunk % bph;
            within = (chunk / bph) * sk_chunk_t + (tile_row % sk_chunk_t);
        } else {
            bank_slice = tile_row / st_pb;
            within = tile_row % st_pb;
        }
        const uint32_t bank_id = banks[local_head * bph + bank_slice];
        const uint64_t off =
            base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords = {{policy.sp_origin.get(), chip}};
        return {bank_id, off, std::move(coords)};
    }

    // ---- NATURAL: GLM-style paged CP; IDENTITY (page%banks) vs OPTIMAL-perm (perm[chunk%blocks]) ----
    const auto [owner, local_pos] = cp_chunk_modulo(d, policy, position, dcs);
    const uint32_t csb = chunk_size_bytes(spec.tensor, tpc);
    const uint32_t tpc_page = kcs;
    if (policy.bank_order == BankOrder::Identity) {
        const uint32_t pages_per_slot = geom.num_slots > 1 ? (per_dev_seq / tpc_page) : 0;
        const uint32_t eff_slot = pages_per_slot > 0 ? slot : 0;
        const uint32_t page_id = eff_slot * pages_per_slot + local_pos / tpc_page;
        const uint32_t in_page_offset = (local_pos % tpc_page) * (csb / tpc_page);
        const uint32_t bank_id = page_id % d.num_banks;
        const uint64_t off = base + static_cast<uint64_t>(page_id / d.num_banks) * csb + in_page_offset;
        return {bank_id, off, {{owner, 0u}}};
    }
    // OPTIMAL indexer: fixed permutation over num_blocks; per-bank slot stacking (ND-shard). `banks` is
    // the OPTIMAL perm here (this branch is the bank_order == Optimal case).
    const uint32_t nblk = policy.num_blocks ? policy.num_blocks : d.num_banks;
    const uint32_t chunk_idx = local_pos / tpc_page;
    const uint32_t bank_id = banks[chunk_idx % nblk];
    const uint32_t within_bank = chunk_idx / nblk;
    const uint32_t slot_size_b = (per_dev_seq / tpc_page * csb) / nblk;
    const uint32_t eff_slot = geom.num_slots > 1 ? slot : 0;
    const uint64_t off = base + static_cast<uint64_t>(within_bank) * csb + static_cast<uint64_t>(slot_size_b) * eff_slot;
    return {bank_id, off, {{owner, 0u}}};
}

// (row, col) mesh coordinate -> logical chip id (row-major within the mesh).
FabricNodeId to_fabric_node(tt::tt_fabric::MeshId mesh_id, uint32_t row, uint32_t col, uint32_t mesh_cols) {
    return FabricNodeId(mesh_id, row * std::max(1u, mesh_cols) + col);
}

}  // namespace

KvChunkAddressTable to_chunk_map(
    const std::vector<CacheConfig>& configs,
    tt::tt_fabric::MeshId mesh_id,
    const MapGeometry& geometry) {
    TT_FATAL(!configs.empty(), "to_chunk_map requires at least one cache config");

    std::vector<Derived> derived;
    derived.reserve(configs.size());
    std::vector<KvChunkAddressTableConfig> table_configs;
    for (const auto& config : configs) {
        const Derived d = derive(config);
        derived.push_back(d);
        table_configs.push_back(KvChunkAddressTableConfig{
            .num_layers = geometry.num_layers,
            .max_sequence_length = geometry.max_seq_len,
            .num_slots = geometry.num_slots * d.n_heads,
            .chunk_n_tokens = d.tpc,
            .chunk_size_bytes = chunk_size_bytes(config.spec.tensor, d.tpc),
        });
    }

    KvChunkAddressTable table{std::span<const KvChunkAddressTableConfig>(table_configs)};

    for (uint32_t cfg = 0; cfg < configs.size(); ++cfg) {
        const CacheConfig& config = configs[cfg];
        const KvLayoutSpec& spec = config.spec;
        const GenerationPolicy& policy = config.policy;
        const Derived& d = derived[cfg];
        const bool per_head = d.head_axis.has_value();

        uint32_t extent = geometry.max_seq_len;
        if (const auto* w = std::get_if<temporal::Window>(&spec.temporal)) {
            if (w->width.get() != 0) {
                extent = std::min(extent, w->width.get());
            }
        }

        for (uint32_t layer = 0; layer < geometry.num_layers; ++layer) {
            for (uint32_t h = 0; h < d.n_heads; ++h) {
                const std::optional<uint32_t> head = per_head ? std::optional<uint32_t>(h) : std::nullopt;
                for (uint32_t slot = 0; slot < geometry.num_slots; ++slot) {
                    for (uint32_t pos = 0; pos < extent && pos < geometry.max_seq_len; pos += geometry.position_step) {
                        const Located loc = locate_one(spec, policy, d, head, pos, slot, config.base_addr, geometry);
                        std::vector<FabricNodeId> nodes;
                        nodes.reserve(loc.coords.size());
                        for (const auto& [row, col] : loc.coords) {
                            nodes.push_back(to_fabric_node(mesh_id, row, col, d.mesh_cols));
                        }
                        const DeviceGroupIndex grp = table.add_device_group(std::move(nodes));
                        const uint64_t noc_addr = (static_cast<uint64_t>(loc.bank_id) << 32) | addr_local(loc.offset);
                        const uint32_t map_slot = per_head ? (h * geometry.num_slots + slot) : slot;
                        table.set(
                            layer,
                            pos,
                            map_slot,
                            KvCacheLocation{
                                .noc_addr = noc_addr,
                                .size_bytes = table_configs[cfg].chunk_size_bytes,
                                .device_group_index = grp},
                            cfg);
                    }
                }
            }
        }
    }

    return table;
}

}  // namespace tt::tt_metal::internal::disaggregation
