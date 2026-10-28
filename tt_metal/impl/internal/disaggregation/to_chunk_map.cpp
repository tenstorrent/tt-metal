// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/to_chunk_map.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

#include <internal/disaggregation/noc_addr.hpp>

namespace tt::tt_metal::internal::disaggregation {

constexpr uint32_t kBfp8TileBytes = tt::tile_size(tt::DataFormat::Bfp8_b);
using tt::constants::TILE_WIDTH;

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshMapperConfig;
using tt::tt_metal::distributed::MeshShape;

// The mesh geometry + tensor facts the addresser derives ONCE per cache (instead of reading them from
// bespoke KvLayoutSpec fields): the seq axis and feature width off the tensor's NdShardSpec, and the
// seq-shard and head-shard mesh axes off the tensor's TensorTopology.
struct Derived {
    uint32_t seq_axis = 0;
    uint64_t f = 1;                    // per-token feature width
    uint32_t tpc = TILE_WIDTH;         // migration granule == the shard's seq-block size
    uint32_t sp_dim = 1;               // devices the sequence is sharded over (product of seq_mesh_axes)
    std::vector<uint32_t> seq_mesh_axes;     // distribution axes that shard the sequence, outermost first
    std::optional<uint32_t> head_mesh_axis;  // distribution axis that shards the head
    std::optional<uint32_t> head_axis;  // tensor axis a non-seq mesh axis shards (GQA head)
    uint32_t n_heads = 1;
    uint32_t n_heads_per_dev = 1;
    uint32_t num_banks = 0;             // set from num_dram_banks(arch) in derive()
    std::vector<uint32_t> banks;        // resolved bank permutation (arch + bank_order), size num_banks
    uint32_t fold_layers = 0;           // layers folded into the batch axis, user-major; 0 => not folded
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
    TT_FATAL(
        config.topology.mesh_coords().size() == ms.mesh_size(),
        "TensorTopology lists {} mesh coords for a distribution of {} devices",
        config.topology.mesh_coords().size(),
        ms.mesh_size());
    for (size_t i = 0; i < placements.size(); ++i) {
        if (const auto* s = std::get_if<MeshMapperConfig::Shard>(&placements[i])) {
            if (seq.has_value() && static_cast<uint32_t>(s->dim) == *seq) {
                d.seq_mesh_axes.push_back(static_cast<uint32_t>(i));
                d.sp_dim *= ms[static_cast<int>(i)];
            } else {
                d.head_mesh_axis = static_cast<uint32_t>(i);
                d.head_axis = static_cast<uint32_t>(s->dim);
            }
        }
    }
    if (d.head_axis.has_value()) {
        d.n_heads = shape[static_cast<int>(*d.head_axis)];
        d.n_heads_per_dev = std::max(1u, d.n_heads / ms[static_cast<int>(*d.head_mesh_axis)]);
    }
    return d;
}

// A single resolved chunk: physical bank + per-bank offset + the seq/head shard indices that own it.
struct Located {
    uint32_t bank_id = 0;
    uint64_t offset = 0;
    uint32_t seq_shard = 0;
    uint32_t head_shard = 0;
};

// MLA CP ownership: round-robin over sp devices with a per-device chunk stride.
std::pair<uint32_t, uint32_t> cp_mla_stride(const Derived& d, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = d.sp_dim;
    const uint32_t round_idx = position / (dcs * sp_dim);
    const uint32_t in_round = position % (dcs * sp_dim);
    return {in_round / dcs, round_idx * dcs + in_round % dcs};
}

// Chunk-modulo CP ownership (GLM, M3 index_k): owner = chunk_index % sp_dim.
std::pair<uint32_t, uint32_t> cp_chunk_modulo(const Derived& d, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = d.sp_dim;
    if (sp_dim <= 1 || dcs == 0) {
        return {0, position};
    }
    const uint32_t chunk_index = position / dcs;
    const uint32_t owner = chunk_index % sp_dim;
    const uint32_t local_pos = (chunk_index / sp_dim) * dcs + (position % dcs);
    return {owner, local_pos};
}

Located locate_one(
    const KvLayoutSpec& spec,
    const GenerationPolicy& policy,
    const Derived& d,
    std::optional<uint32_t> head,
    uint32_t layer,
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
    const uint32_t per_dev_seq = geom.max_seq_len / sp_dim;

    uint32_t dcs = policy.device_chunk_size.has_value() ? policy.device_chunk_size->get() : kcs * d.num_banks;

    const auto& shape = spec.tensor.logical_shape();

    // ---- MLA_SHARD: CP stride + shard_id stacking + OPTIMAL perm over num_banks ----
    if (scheme == BankScheme::MlaShard) {
        const uint32_t csb = chunk_size_bytes(spec, tpc);
        const auto [owner, local_pos] = cp_mla_stride(d, position, dcs);
        const uint32_t chunks_per_slot = per_dev_seq / kcs;
        const uint32_t local_chunk = local_pos / kcs;
        const uint32_t in_chunk = local_pos % kcs;
        // Slot folds into the physical shard index (paging, if any, is a separate block-table contract).
        const uint32_t shard_id = slot * chunks_per_slot + local_chunk;
        const uint32_t shard_size_b = (kcs / tpc) * csb;
        const uint32_t bank_id = banks[shard_id % d.num_banks];
        const uint64_t off =
            base + static_cast<uint64_t>(shard_id / d.num_banks) * shard_size_b + (in_chunk / tpc) * csb;
        return {bank_id, off, owner, 0};
    }

    // ---- BLOCK_CYCLIC: minimax index_k, seq blocks split over the seq-shard axis, round-robin over banks ----
    if (scheme == BankScheme::BlockCyclic) {
        const auto [owner, local_position] = cp_chunk_modulo(d, position, dcs);
        const uint32_t block = kcs;
        const uint32_t idx_row_bytes = static_cast<uint32_t>((f / TILE_WIDTH) * kBfp8TileBytes);
        const uint32_t n_blocks_dev = per_dev_seq / block;
        const uint32_t bph = policy.banks_per_head.get() ? policy.banks_per_head.get() : d.num_banks;
        const uint32_t num_banks = std::min(bph, n_blocks_dev);
        const uint32_t blocks_per_bank = n_blocks_dev / num_banks;
        const uint32_t global_block = slot * blocks_per_bank * num_banks + local_position / block;
        const uint32_t bank_id = banks[global_block % num_banks];
        const uint32_t local_block = global_block / num_banks;
        const uint32_t tile_row = (local_block * block + (local_position % block)) / TILE_WIDTH;
        const uint64_t off = base + static_cast<uint64_t>(tile_row) * idx_row_bytes;
        return {bank_id, off, owner, 0};
    }

    // ---- BLOCK / CYCLIC height-sharded K/V: per-head GQA (gpt-oss) or one head (M3 group) ----
    if (scheme == BankScheme::Block || scheme == BankScheme::Cyclic) {
        const uint32_t h = head.value_or(0);
        const uint32_t bph = policy.banks_per_head.get() ? policy.banks_per_head.get() : d.num_banks;
        const uint64_t dht = f / TILE_WIDTH;
        const uint32_t row_bytes = static_cast<uint32_t>(dht * kBfp8TileBytes);
        const uint32_t local_head = h % d.n_heads_per_dev;
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
        return {bank_id, off, 0, h / d.n_heads_per_dev};
    }

    // ---- NATURAL: GLM-style paged CP; IDENTITY (page%banks) vs OPTIMAL-perm (perm[chunk%blocks]) ----
    const auto [owner, local_pos] = cp_chunk_modulo(d, position, dcs);
    const uint32_t csb = chunk_size_bytes(spec, tpc);
    const uint32_t tpc_page = kcs;
    if (policy.bank_order == BankOrder::Identity) {
        // The ND-shard ravel: the batch row (slot, or slot and layer when folded), then the chip's local head.
        const uint32_t batch_row = d.fold_layers > 0 ? slot * d.fold_layers + layer : slot;
        const uint32_t row = head.has_value() ? batch_row * d.n_heads_per_dev + *head % d.n_heads_per_dev : batch_row;
        const uint32_t page_id = row * (per_dev_seq / tpc_page) + local_pos / tpc_page;
        const uint32_t in_page_offset = (local_pos % tpc_page) * (csb / tpc_page);
        const uint32_t bank_id = page_id % d.num_banks;
        const uint64_t off = base + static_cast<uint64_t>(page_id / d.num_banks) * csb + in_page_offset;
        return {bank_id, off, owner, head.has_value() ? *head / d.n_heads_per_dev : 0};
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
    return {bank_id, off, owner, 0};
}

// Whether distribution coord `t` holds seq shard `seq_shard`: the shard index is row-major over the
// seq-shard axes, the first (outermost) axis varying slowest.
bool holds_seq_shard(const Derived& d, const MeshShape& dist, const MeshCoordinate& t, uint32_t seq_shard) {
    for (auto axis = d.seq_mesh_axes.rbegin(); axis != d.seq_mesh_axes.rend(); ++axis) {
        const uint32_t extent = dist[static_cast<int>(*axis)];
        if (t[static_cast<int32_t>(*axis)] != seq_shard % extent) {
            return false;
        }
        seq_shard /= extent;
    }
    return true;
}

// How many of the ascending `layers` fall in [first, end).
uint32_t published_layers_in(const std::vector<uint32_t>& layers, uint32_t first, uint32_t end) {
    return static_cast<uint32_t>(
        std::lower_bound(layers.begin(), layers.end(), end) - std::lower_bound(layers.begin(), layers.end(), first));
}

// The replica set holding a chunk: every device of the tensor's distribution whose seq/head shard
// index matches, mapped through the topology's mesh coords to chip ids on the physical mesh.
std::vector<FabricNodeId> device_group(
    const TensorTopology& topology,
    const Derived& d,
    const Located& loc,
    tt::tt_fabric::MeshId mesh_id,
    const MeshShape& mesh_shape) {
    const MeshShape& dist = topology.distribution_shape();
    std::vector<FabricNodeId> nodes;
    for (const auto& t : MeshCoordinateRange(dist)) {
        if (!holds_seq_shard(d, dist, t, loc.seq_shard) ||
            (d.head_mesh_axis.has_value() && t[static_cast<int32_t>(*d.head_mesh_axis)] != loc.head_shard)) {
            continue;
        }
        const auto& coord = topology.mesh_coords()[t.to_linear_index(dist)];
        nodes.emplace_back(mesh_id, static_cast<tt::tt_fabric::LogicalChipId>(coord.to_linear_index(mesh_shape)));
    }
    return nodes;
}

}  // namespace

KvChunkAddressTable to_chunk_map(
    const std::vector<CacheConfig>& configs,
    tt::tt_fabric::MeshId mesh_id,
    const MeshShape& mesh_shape,
    const MapGeometry& geometry) {
    TT_FATAL(!configs.empty(), "to_chunk_map requires at least one cache config");
    const bool named = !configs.front().name.empty();

    std::vector<Derived> derived;
    derived.reserve(configs.size());
    // One entry per table config; caches sharing a name share one.
    std::vector<KvChunkAddressTableConfig> table_configs;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> held_layers;
    std::map<std::string, uint32_t> table_config_of_name;
    std::vector<uint32_t> table_config_of(configs.size());
    for (uint32_t cfg = 0; cfg < configs.size(); ++cfg) {
        const CacheConfig& config = configs[cfg];
        TT_FATAL(config.name.empty() != named, "to_chunk_map: cache config names must be all empty or all set");
        TT_FATAL(
            config.first_layer < geometry.num_layers,
            "to_chunk_map: cache '{}' starts at layer {}, past the model's {} layers",
            config.name,
            config.first_layer,
            geometry.num_layers);
        const uint32_t cache_layers =
            config.num_layers > 0 ? config.num_layers : geometry.num_layers - config.first_layer;
        const uint32_t end_layer = config.first_layer + cache_layers;
        TT_FATAL(
            end_layer <= geometry.num_layers,
            "to_chunk_map: cache '{}' holds layers [{}, {}), past the model's {} layers",
            config.name,
            config.first_layer,
            end_layer,
            geometry.num_layers);
        TT_FATAL(
            std::is_sorted(config.layers.begin(), config.layers.end()) &&
                std::adjacent_find(config.layers.begin(), config.layers.end()) == config.layers.end(),
            "to_chunk_map: cache '{}' publishes layers that are not ascending and distinct",
            config.name);

        for (const auto& [alias, source] : config.layer_aliases) {
            const bool source_published =
                source >= config.first_layer && source < end_layer &&
                (config.layers.empty() || std::binary_search(config.layers.begin(), config.layers.end(), source));
            const bool alias_outside_cache = alias < config.first_layer || alias >= end_layer;
            TT_FATAL(
                source_published && alias_outside_cache && alias < geometry.num_layers,
                "to_chunk_map: cache '{}' aliases layer {} to layer {}, which needs a published layer of the cache "
                "and an alias outside it within the model's {} layers",
                config.name,
                alias,
                source,
                geometry.num_layers);
        }
        TT_FATAL(
            !config.holds_only_published_layers || !config.layers.empty(),
            "to_chunk_map: cache '{}' holds only its published layers but lists none",
            config.name);
        const uint32_t held_layers_count = config.holds_only_published_layers
                                               ? published_layers_in(config.layers, config.first_layer, end_layer)
                                               : cache_layers;
        Derived d = derive(config);
        const bool layers_folded =
            held_layers_count > 1 && config.spec.tensor.logical_shape()[0] == geometry.num_slots * held_layers_count;
        TT_FATAL(
            !config.holds_only_published_layers || held_layers_count <= 1 || layers_folded,
            "to_chunk_map: cache '{}' holds its {} published layers, but its batch axis is not {} slots x layers",
            config.name,
            held_layers_count,
            geometry.num_slots);
        d.fold_layers = layers_folded ? held_layers_count : 0;
        const bool identity_natural =
            config.policy.bank_scheme == BankScheme::Natural && config.policy.bank_order == BankOrder::Identity;
        TT_FATAL(
            !layers_folded || identity_natural,
            "to_chunk_map: cache '{}' folds its layers into the batch axis, which only the Natural scheme with "
            "the Identity bank order addresses",
            config.name);
        TT_FATAL(
            !config.head.has_value() || (d.head_axis.has_value() && *config.head < d.n_heads),
            "to_chunk_map: cache '{}' selects head {} of {} heads, but its topology shards no head axis or has fewer",
            config.name,
            config.head.value_or(0),
            d.n_heads);
        derived.push_back(d);

        const KvChunkAddressTableConfig table_config{
            .num_layers = geometry.num_layers,
            .max_sequence_length = geometry.max_seq_len,
            .num_slots = config.head.has_value() ? geometry.num_slots : geometry.num_slots * d.n_heads,
            .chunk_n_tokens = d.tpc,
            .chunk_size_bytes = chunk_size_bytes(config.spec, d.tpc),
        };
        uint32_t id = static_cast<uint32_t>(table_configs.size());
        if (named) {
            const auto [it, inserted] = table_config_of_name.emplace(config.name, id);
            id = it->second;
            if (!inserted) {
                const KvChunkAddressTableConfig& first = table_configs[id];
                TT_FATAL(
                    first.num_slots == table_config.num_slots && first.chunk_n_tokens == table_config.chunk_n_tokens &&
                        first.chunk_size_bytes == table_config.chunk_size_bytes,
                    "to_chunk_map: the caches named '{}' differ in slots or chunk size",
                    config.name);
            }
        }
        if (id == table_configs.size()) {
            table_configs.push_back(table_config);
            held_layers.emplace_back();
        }
        for (const auto& [first, end] : held_layers[id]) {
            TT_FATAL(
                end_layer <= first || config.first_layer >= end,
                "to_chunk_map: two caches named '{}' both hold layers in [{}, {})",
                config.name,
                std::max(first, config.first_layer),
                std::min(end, end_layer));
        }
        held_layers[id].emplace_back(config.first_layer, end_layer);
        table_config_of[cfg] = id;
    }

    std::map<std::string, KvChunkAddressTableConfig> by_name;
    for (const auto& [name, id] : table_config_of_name) {
        by_name.emplace(name, table_configs[id]);
    }
    KvChunkAddressTable table = named ? KvChunkAddressTable{by_name}
                                      : KvChunkAddressTable{std::span<const KvChunkAddressTableConfig>(table_configs)};

    for (uint32_t cfg = 0; cfg < configs.size(); ++cfg) {
        const CacheConfig& config = configs[cfg];
        const KvLayoutSpec& spec = config.spec;
        const GenerationPolicy& policy = config.policy;
        const Derived& d = derived[cfg];
        const bool per_head = d.head_axis.has_value();
        const uint32_t table_config = named ? table.config_id_of(config.name) : table_config_of[cfg];
        const uint32_t chunk_bytes = table_configs[table_config_of[cfg]].chunk_size_bytes;
        const tt::tt_fabric::MeshId cache_mesh = config.mesh_id.value_or(mesh_id);
        const uint32_t cache_layers =
            config.num_layers > 0 ? config.num_layers : geometry.num_layers - config.first_layer;

        uint32_t extent = geometry.max_seq_len;
        if (const auto* w = std::get_if<temporal::Window>(&spec.temporal)) {
            if (w->width.get() != 0) {
                extent = std::min(extent, w->width.get());
            }
        }

        uint32_t held_index = 0;
        for (uint32_t local_layer = 0; local_layer < cache_layers; ++local_layer) {
            const uint32_t layer = config.first_layer + local_layer;
            const bool published =
                config.layers.empty() || std::binary_search(config.layers.begin(), config.layers.end(), layer);
            if (!published) {
                continue;
            }
            const uint32_t cache_row_layer = config.holds_only_published_layers ? held_index++ : local_layer;
            std::vector<uint32_t> aliases;
            for (const auto& [alias, source] : config.layer_aliases) {
                if (source == layer) {
                    aliases.push_back(alias);
                }
            }
            const uint32_t first_head = config.head.value_or(0);
            const uint32_t end_head = config.head.has_value() ? *config.head + 1 : d.n_heads;
            for (uint32_t h = first_head; h < end_head; ++h) {
                const std::optional<uint32_t> head = per_head ? std::optional<uint32_t>(h) : std::nullopt;
                for (uint32_t slot = 0; slot < geometry.num_slots; ++slot) {
                    for (uint32_t pos = 0; pos < extent && pos < geometry.max_seq_len; pos += geometry.position_step) {
                        const Located loc =
                            locate_one(spec, policy, d, head, cache_row_layer, pos, slot, config.base_addr, geometry);
                        const DeviceGroupIndex grp =
                            table.add_device_group(device_group(config.topology, d, loc, cache_mesh, mesh_shape));
                        const uint64_t noc_addr = (static_cast<uint64_t>(loc.bank_id) << 32) | addr_local(loc.offset);
                        const bool fanned = per_head && !config.head.has_value();
                        const uint32_t map_slot = fanned ? (h * geometry.num_slots + slot) : slot;
                        const KvCacheLocation location{
                            .noc_addr = noc_addr, .size_bytes = chunk_bytes, .device_group_index = grp};
                        table.set(layer, pos, map_slot, location, table_config);
                        for (uint32_t alias : aliases) {
                            table.set(alias, pos, map_slot, location, table_config);
                        }
                    }
                }
            }
        }
    }

    return table;
}

KvChunkAddressTable to_chunk_map(const CacheLayoutModel& model) {
    return to_chunk_map(model.caches, model.mesh_id, model.mesh_shape, model.geometry);
}

namespace {

// The first way config `ac` of `actual` differs from config `ec` of `expected` (both named `name`).
std::optional<std::string> config_difference(
    const KvChunkAddressTable& expected,
    uint32_t ec,
    const KvChunkAddressTable& actual,
    uint32_t ac,
    const std::string& name) {
    const KvChunkAddressTableConfig& want = expected.config(ec);
    const KvChunkAddressTableConfig& got = actual.config(ac);
    const bool same_config = want.num_layers == got.num_layers && want.max_sequence_length == got.max_sequence_length &&
                             want.num_slots == got.num_slots && want.chunk_n_tokens == got.chunk_n_tokens &&
                             want.chunk_size_bytes == got.chunk_size_bytes && want.tensor_type == got.tensor_type;
    if (!same_config) {
        return fmt::format("config '{}' differs in its extents, chunk size or tensor type", name);
    }
    for (uint32_t layer = 0; layer < want.num_layers; ++layer) {
        for (uint32_t slot = 0; slot < want.num_slots; ++slot) {
            for (uint32_t pos = 0; pos < want.max_sequence_length; pos += want.chunk_n_tokens) {
                const KvCacheLocation a = expected.lookup(layer, pos, slot, ec);
                const KvCacheLocation b = actual.lookup(layer, pos, slot, ac);
                const auto where = [&] {
                    return fmt::format("config '{}' layer {} slot {} position {}", name, layer, slot, pos);
                };
                if (a.size_bytes != b.size_bytes) {
                    return fmt::format("{}: {} bytes, not {}", where(), b.size_bytes, a.size_bytes);
                }
                if (a.size_bytes == 0) {
                    continue;
                }
                if (a.noc_addr != b.noc_addr) {
                    return fmt::format("{}: noc_addr {:#x}, not {:#x}", where(), b.noc_addr, a.noc_addr);
                }
                const auto& want_nodes = expected.get_device_group(a.device_group_index).fabric_node_ids;
                const auto& got_nodes = actual.get_device_group(b.device_group_index).fabric_node_ids;
                if (want_nodes != got_nodes) {
                    return fmt::format("{}: a different device group", where());
                }
                for (const auto& node : want_nodes) {
                    const bool want_host = expected.has_host(node);
                    const bool same_host = want_host == actual.has_host(node) &&
                                           (!want_host || expected.get_host(node) == actual.get_host(node));
                    if (!same_host) {
                        return fmt::format(
                            "{}: chip {} of mesh {} has a different host", where(), node.chip_id, *node.mesh_id);
                    }
                }
            }
        }
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> first_table_difference(
    const KvChunkAddressTable& expected, const KvChunkAddressTable& actual, bool only_actual_configs) {
    if (only_actual_configs) {
        for (uint32_t ac = 0; ac < actual.num_configs(); ++ac) {
            const std::string& name = actual.config_name(ac);
            std::optional<uint32_t> ec;
            for (uint32_t c = 0; c < expected.num_configs() && !ec.has_value(); ++c) {
                if (expected.config_name(c) == name) {
                    ec = c;
                }
            }
            if (!ec.has_value()) {
                return fmt::format("config '{}' is not in the expected table", name);
            }
            if (auto difference = config_difference(expected, *ec, actual, ac, name)) {
                return difference;
            }
        }
        return std::nullopt;
    }
    if (expected.num_configs() != actual.num_configs()) {
        return fmt::format("{} configs, not {}", actual.num_configs(), expected.num_configs());
    }
    for (uint32_t c = 0; c < expected.num_configs(); ++c) {
        const std::string& name = expected.config_name(c);
        if (actual.config_name(c) != name) {
            return fmt::format("config {} is named '{}', not '{}'", c, actual.config_name(c), name);
        }
        if (auto difference = config_difference(expected, c, actual, c, name)) {
            return difference;
        }
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::internal::disaggregation
