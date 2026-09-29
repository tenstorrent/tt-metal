// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

#include "internal/disaggregation/kv_layout_spec.hpp"
#include "internal/disaggregation/noc_addr.hpp"
#include "internal/disaggregation/to_chunk_map.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshMapperConfig;
using tt::tt_metal::distributed::MeshShape;

// These tests drive the ONE `to_chunk_map` factory with REAL ND-sharded tt-metal TensorSpecs +
// host-side TensorTopology (no MeshDevice), and compare the resulting KvChunkAddressTable against a C++
// port of blaze's migration-path arithmetic (the same reference the device-free kv_layout_spec_smoke
// scripts use). The addresser reads the seq axis + feature width off the tensor's NdShardSpec and the
// mesh geometry + device coords off the TensorTopology; only the op/engine policy is passed explicitly.
// CPU-only; no device required.

constexpr uint32_t kTileLocal = 32;
constexpr uint32_t kBfp8TileBytesLocal = 1088;
constexpr uint32_t kNumBanks = 8;

// Build a real bfloat8_b, TILE-layout, DRAM ND-sharded TensorSpec. `shard_shape` tiles the seq axis at
// the 32-token DRAM block and leaves the feature axis full-width (the layout the addresser reads back).
TensorSpec make_bfp8_ndshard_spec(const Shape& shape, const Shape& shard_shape) {
    auto page_config = PageConfig(Layout::TILE);
    CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(kNumBanks - 1, 0)));
    NdShardSpec nd{shard_shape, grid, ShardOrientation::ROW_MAJOR, ShardDistributionStrategy::ROUND_ROBIN_1D};
    auto memory_config = MemoryConfig(BufferType::DRAM, nd);
    auto tensor_layout = TensorLayout(DataType::BFLOAT8_B, page_config, memory_config);
    return TensorSpec(shape, tensor_layout);
}

// Full row-major device-coordinate list for a 2D mesh (the addresser linearizes coords itself; this
// just satisfies the TensorTopology ctor).
std::vector<MeshCoordinate> row_major_coords(uint32_t rows, uint32_t cols) {
    std::vector<MeshCoordinate> coords;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            coords.emplace_back(r, c);
        }
    }
    return coords;
}

uint32_t bfp8_chunk_bytes(uint32_t tokens_per_chunk, uint64_t feature_dim) {
    return (tokens_per_chunk / kTileLocal) * static_cast<uint32_t>(feature_dim / kTileLocal) * kBfp8TileBytesLocal;
}

// --- Reference: blaze MLA migration arithmetic (deepseek_v3_kimi_k2_mla.py) ---
struct RefLoc {
    uint32_t bank_id;
    uint64_t offset;
    uint32_t owner;  // sp row
};

RefLoc mla_reference(
    uint32_t position,
    uint32_t slot,
    uint64_t base,
    uint64_t head_dim,
    uint32_t per_device_seq_len,
    uint32_t tokens_per_chunk,
    uint32_t sp_dim,
    uint32_t k_chunk_size,
    uint32_t device_chunk_size,
    uint32_t sp_origin) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t num_banks = 8;
    const uint32_t round_idx = position / (device_chunk_size * sp_dim);
    const uint32_t in_round = position % (device_chunk_size * sp_dim);
    const uint32_t sp_device_idx = in_round / device_chunk_size;
    const uint32_t in_chunk_dev = in_round % device_chunk_size;
    const uint32_t owner = (sp_device_idx + sp_origin) % sp_dim;
    const uint32_t local_pos = round_idx * device_chunk_size + in_chunk_dev;

    const uint32_t csb = bfp8_chunk_bytes(tokens_per_chunk, head_dim);
    const uint32_t chunks_per_slot = per_device_seq_len / k_chunk_size;
    const uint32_t local_chunk = local_pos / k_chunk_size;
    const uint32_t in_chunk = local_pos % k_chunk_size;
    const uint32_t shard_id = slot * chunks_per_slot + local_chunk;
    const uint32_t shard_size_b = (k_chunk_size / tokens_per_chunk) * csb;
    const uint32_t bank_id = kOrder[shard_id % num_banks];
    const uint64_t noc = base + static_cast<uint64_t>(shard_id / num_banks) * shard_size_b + (in_chunk / tokens_per_chunk) * csb;
    return {bank_id, noc, owner};
}

// --- Reference: gpt-oss GQA chunk_location (gpt_oss.py) ---
RefLoc gptoss_reference(
    uint32_t position,
    uint32_t slot,
    uint64_t base,
    uint32_t local_head,
    uint32_t bph,
    uint32_t st_pb,
    uint32_t row_bytes,
    bool cyclic,
    uint32_t sk_chunk_t) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t tile_row = position / kTileLocal;
    uint32_t bank_slice = 0;
    uint32_t within = 0;
    if (cyclic) {
        const uint32_t chunk = tile_row / sk_chunk_t;
        bank_slice = chunk % bph;
        within = (chunk / bph) * sk_chunk_t + (tile_row % sk_chunk_t);
    } else {
        bank_slice = tile_row / st_pb;
        within = tile_row % st_pb;
    }
    const uint32_t bank_id = kOrder[local_head * bph + bank_slice];
    const uint64_t noc = base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
    return {bank_id, noc, 0};
}

// --- MLA (DeepSeek-V3 / Kimi) ---
TEST(ToChunkMap, CPU_MlaShardMatchesMigrationReference) {
    const uint64_t F = 576;  // kv_lora_rank 512 + qk_rope 64
    const uint32_t sp_dim = 4;
    const uint32_t mesh_cols = 2;
    const uint32_t k_chunk = 128;
    const uint32_t chunk_n_tokens = kTileLocal;
    const uint32_t device_chunk_size = k_chunk * 8;  // 1024 (DeepSeek default)
    const uint32_t max_seq_len = device_chunk_size * sp_dim * 2;  // 8192, multiple of dcs*sp_dim
    const uint32_t num_slots = 3;
    const uint64_t base = 0x1000'0000ull;
    const uint32_t per_dev = max_seq_len / sp_dim;

    // Real ND-sharded TensorSpec: (slot, F, seq), seq axis = 2, tiled [1, F, 32].
    KvLayoutSpec spec{
        .tensor = make_bfp8_ndshard_spec(
            Shape{num_slots, static_cast<uint32_t>(F), max_seq_len}, Shape{1, static_cast<uint32_t>(F), kTileLocal})};
    spec.temporal = temporal::Dense{};
    spec.addressing = AddressingMode::Slot;
    spec.arch = tt::ARCH::BLACKHOLE;  // 8 DRAM banks + the OPTIMAL order derive from this

    // seq axis + feature width are derived from the tensor's NdShardSpec.
    EXPECT_EQ(spec.sequence_axis().value(), 2u);
    EXPECT_EQ(feature_width(spec.tensor), F);

    // Topology: 4x2 mesh, seq (tensor dim 2) sharded on mesh axis 0, replicated across mesh axis 1.
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Shard{.dim = 2}, MeshMapperConfig::Replicate{}};
    TensorTopology topology(MeshShape{sp_dim, mesh_cols}, placements, row_major_coords(sp_dim, mesh_cols));

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::MlaShard;
    policy.bank_order = BankOrder::Optimal;
    policy.k_chunk_size = KChunkSize{k_chunk};
    policy.device_chunk_size = DeviceChunkSize{device_chunk_size};

    CacheConfig config{.spec = spec, .topology = topology, .policy = policy, .base_addr = base};
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = chunk_n_tokens};

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, geom);

    uint32_t checked = 0;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (uint32_t pos = 0; pos < max_seq_len; pos += chunk_n_tokens) {
            const RefLoc ref = mla_reference(
                pos, slot, base, F, per_dev, chunk_n_tokens, sp_dim, k_chunk, device_chunk_size, 0);
            const KvCacheLocation loc = table.lookup(0, pos, slot, 0u);
            EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "pos=" << pos << " slot=" << slot;
            EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                << "pos=" << pos << " slot=" << slot;

            // Device group: MLA replicates the owning sp row across all mesh columns.
            const auto& grp = table.get_device_group(loc.device_group_index);
            ASSERT_EQ(grp.fabric_node_ids.size(), mesh_cols);
            for (uint32_t c = 0; c < mesh_cols; ++c) {
                EXPECT_EQ(grp.fabric_node_ids[c].chip_id, ref.owner * mesh_cols + c);
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

// --- gpt-oss GQA (CYCLIC dense layer) ---
TEST(ToChunkMap, CPU_GqaCyclicMatchesReference) {
    const uint32_t n_kv_heads = 8;
    const uint32_t tp = 8;
    const uint64_t head_dim = 64;
    const uint32_t sdpa_k_chunk = 512;
    const uint32_t max_seq_len = 4096;
    const uint32_t num_slots = 2;
    const uint64_t base = 0x10000ull;
    const uint32_t n_kv_heads_per_dev = std::max(1u, n_kv_heads / tp);  // 1
    const uint32_t bph = 8;  // banks_per_kv_head for head_dim 64, 1 head/dev
    const uint32_t row_bytes = static_cast<uint32_t>((head_dim / kTileLocal) * kBfp8TileBytesLocal);
    const uint32_t sk_chunk_t = sdpa_k_chunk / kTileLocal;
    const uint32_t st_pb = (max_seq_len / kTileLocal) / bph;

    // Real ND-sharded TensorSpec: (slot, head, seq, head_dim), seq axis = 2, tiled [1, 1, 32, head_dim].
    KvLayoutSpec spec{
        .tensor = make_bfp8_ndshard_spec(
            Shape{num_slots, n_kv_heads, max_seq_len, static_cast<uint32_t>(head_dim)},
            Shape{1, 1, kTileLocal, static_cast<uint32_t>(head_dim)})};
    spec.temporal = temporal::Dense{};
    spec.addressing = AddressingMode::Slot;
    spec.arch = tt::ARCH::BLACKHOLE;  // 8 DRAM banks + the OPTIMAL order derive from this

    EXPECT_EQ(spec.sequence_axis().value(), 2u);
    EXPECT_EQ(feature_width(spec.tensor), head_dim);

    // Topology: 1x8 mesh, head (tensor dim 1) sharded on mesh axis 1, seq replicated (sp_dim == 1).
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{.dim = 1}};
    TensorTopology topology(MeshShape{1, n_kv_heads}, placements, row_major_coords(1, n_kv_heads));

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Cyclic;
    policy.bank_order = BankOrder::Optimal;
    policy.k_chunk_size = KChunkSize{sdpa_k_chunk};
    policy.banks_per_head = BanksPerHead{bph};

    CacheConfig config{.spec = spec, .topology = topology, .policy = policy, .base_addr = base};
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = kTileLocal};

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, geom);

    const bool cyclic = true;
    uint32_t checked = 0;
    for (uint32_t head = 0; head < n_kv_heads; ++head) {
        const uint32_t local_head = head % n_kv_heads_per_dev;
        const uint32_t chip = head / n_kv_heads_per_dev;
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            const uint32_t map_slot = head * num_slots + slot;
            for (uint32_t pos = 0; pos < max_seq_len; pos += kTileLocal) {
                const RefLoc ref =
                    gptoss_reference(pos, slot, base, local_head, bph, st_pb, row_bytes, cyclic, sk_chunk_t);
                const KvCacheLocation loc = table.lookup(0, pos, map_slot, 0u);
                EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "head=" << head << " pos=" << pos;
                EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                    << "head=" << head << " pos=" << pos;
                const auto& grp = table.get_device_group(loc.device_group_index);
                ASSERT_EQ(grp.fabric_node_ids.size(), 1u);
                EXPECT_EQ(grp.fabric_node_ids[0].chip_id, chip);
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 0u);
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
