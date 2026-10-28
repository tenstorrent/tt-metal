// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

#include "internal/disaggregation/cache_tensor_layout_spec.hpp"
#include "internal/disaggregation/noc_addr.hpp"
#include "internal/disaggregation/to_chunk_map.hpp"
#include "impl/internal/disaggregation/cache_layout_model_protobuf.hpp"

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

// --- Reference: MiniMax-M3 K/V kv_chunk_location (minimax_m3.py) ---
RefLoc m3_kv_reference(
    uint32_t position, uint32_t slot, uint64_t base, uint32_t bph, uint32_t st_pb, uint32_t row_bytes) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t block_size = 128;
    const uint32_t chunk = position / block_size;
    const uint32_t bank_slice = chunk % bph;
    const uint32_t within = (chunk / bph) * (block_size / kTileLocal) + (position % block_size) / kTileLocal;
    const uint64_t noc =
        base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
    return {kOrder[bank_slice], noc, 0};
}

// --- Reference: MiniMax-M3 index_k idx_k_chunk_location (minimax_m3.py); owner = idx_cp column ---
RefLoc m3_index_reference(
    uint32_t position, uint32_t slot, uint64_t base, uint32_t per_dev_seq_kv, uint32_t idx_cp, uint32_t idx_row_bytes) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t block_size = 128;
    const uint32_t n_blocks = per_dev_seq_kv / block_size;
    const uint32_t n_blocks_dev = (n_blocks + idx_cp - 1) / idx_cp;
    const uint32_t num_banks = std::min(8u, n_blocks_dev);
    const uint32_t blocks_per_bank = n_blocks_dev / num_banks;
    const uint32_t gb = position / block_size;
    const uint32_t dev = gb % idx_cp;
    const uint32_t local_position = (gb / idx_cp) * block_size + position % block_size;
    const uint32_t global_block = slot * blocks_per_bank * num_banks + local_position / block_size;
    const uint32_t local_block = global_block / num_banks;
    const uint32_t tile_row = (local_block * block_size + local_position % block_size) / kTileLocal;
    return {kOrder[global_block % num_banks], base + static_cast<uint64_t>(tile_row) * idx_row_bytes, dev};
}

// --- MiniMax-M3 K/V: one config per (k|v, GQA group); group g lives on mesh row g, replicated over columns ---
TEST(ToChunkMap, CPU_M3KvMatchesReference) {
    const uint32_t mesh_rows = 4;
    const uint32_t mesh_cols = 2;
    const uint32_t head_dim = 128;
    const uint32_t max_seq_len = 2048;
    const uint32_t num_slots = 4;
    const uint32_t bph = 8;
    const uint32_t row_bytes = (head_dim / kTileLocal) * kBfp8TileBytesLocal;
    const uint32_t st_pb = (max_seq_len / kTileLocal) / bph;

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Cyclic;
    policy.bank_order = BankOrder::Optimal;
    policy.k_chunk_size = KChunkSize{128};
    policy.banks_per_head = BanksPerHead{bph};

    std::vector<CacheConfig> configs;
    for (uint32_t cfg = 0; cfg < 2 * mesh_rows; ++cfg) {
        const uint32_t group = cfg % mesh_rows;
        KvLayoutSpec spec{
            .tensor = make_bfp8_ndshard_spec(Shape{num_slots, max_seq_len, head_dim}, Shape{1, kTileLocal, head_dim}),
            .temporal = temporal::Dense{},
            .arch = tt::ARCH::BLACKHOLE};
        std::vector<MeshCoordinate> coords;
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back(group, c);
        }
        TensorTopology topology(MeshShape{mesh_cols}, {MeshMapperConfig::Replicate{}}, coords);
        configs.push_back(CacheConfig{
            .spec = spec, .topology = topology, .policy = policy, .base_addr = 0x20000ull + cfg * 0x100000ull});
    }
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = kTileLocal};

    auto table = to_chunk_map(configs, tt::tt_fabric::MeshId{0}, MeshShape{mesh_rows, mesh_cols}, geom);

    uint32_t checked = 0;
    for (uint32_t cfg = 0; cfg < configs.size(); ++cfg) {
        const uint32_t group = cfg % mesh_rows;
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            for (uint32_t pos = 0; pos < max_seq_len; pos += kTileLocal) {
                const RefLoc ref = m3_kv_reference(pos, slot, configs[cfg].base_addr, bph, st_pb, row_bytes);
                const KvCacheLocation loc = table.lookup(0, pos, slot, cfg);
                EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "cfg=" << cfg << " pos=" << pos;
                EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                    << "cfg=" << cfg << " pos=" << pos;
                const auto& grp = table.get_device_group(loc.device_group_index);
                ASSERT_EQ(grp.fabric_node_ids.size(), mesh_cols);
                for (uint32_t c = 0; c < mesh_cols; ++c) {
                    EXPECT_EQ(grp.fabric_node_ids[c].chip_id, group * mesh_cols + c);
                }
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 0u);
}

// --- MiniMax-M3 index_k: sequence blocks split over the 2 mesh columns (idx_cp == 2), replicated over rows ---
TEST(ToChunkMap, CPU_M3IndexKMatchesReference) {
    const uint32_t mesh_rows = 4;
    const uint32_t mesh_cols = 2;
    const uint32_t idx_dim = 128;
    const uint32_t max_seq_len = 2048;
    const uint32_t num_slots = 4;
    const uint64_t base = 0x20000ull;
    const uint32_t idx_row_bytes = (idx_dim / kTileLocal) * kBfp8TileBytesLocal;

    KvLayoutSpec spec{
        .tensor = make_bfp8_ndshard_spec(Shape{num_slots, max_seq_len, idx_dim}, Shape{1, kTileLocal, idx_dim}),
        .temporal = temporal::Dense{},
        .arch = tt::ARCH::BLACKHOLE};
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{.dim = 1}};
    TensorTopology topology(MeshShape{mesh_rows, mesh_cols}, placements, row_major_coords(mesh_rows, mesh_cols));

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::BlockCyclic;
    policy.bank_order = BankOrder::Optimal;
    policy.k_chunk_size = KChunkSize{128};
    policy.device_chunk_size = DeviceChunkSize{128};
    policy.banks_per_head = BanksPerHead{8};

    CacheConfig config{.spec = spec, .topology = topology, .policy = policy, .base_addr = base};
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = kTileLocal};

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, MeshShape{mesh_rows, mesh_cols}, geom);

    uint32_t checked = 0;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (uint32_t pos = 0; pos < max_seq_len; pos += kTileLocal) {
            const RefLoc ref = m3_index_reference(pos, slot, base, max_seq_len, mesh_cols, idx_row_bytes);
            const KvCacheLocation loc = table.lookup(0, pos, slot, 0u);
            EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "slot=" << slot << " pos=" << pos;
            EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                << "slot=" << slot << " pos=" << pos;
            const auto& grp = table.get_device_group(loc.device_group_index);
            ASSERT_EQ(grp.fabric_node_ids.size(), mesh_rows);
            for (uint32_t r = 0; r < mesh_rows; ++r) {
                EXPECT_EQ(grp.fabric_node_ids[r].chip_id, r * mesh_cols + ref.owner);
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

// --- MLA (DeepSeek-V3 / Kimi) ---
// `sp_origin` rotates which mesh row owns the first sequence shard; the topology lists the rows in shard order.
void check_mla_shard(uint32_t sp_origin) {
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
    spec.arch = tt::ARCH::BLACKHOLE;  // 8 DRAM banks + the OPTIMAL order derive from this

    // seq axis + feature width are derived from the tensor's NdShardSpec.
    EXPECT_EQ(spec.sequence_axis().value(), 2u);
    EXPECT_EQ(feature_width(spec.tensor), F);

    // Topology: 4x2 mesh, seq (tensor dim 2) sharded on mesh axis 0, replicated across mesh axis 1.
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Shard{.dim = 2}, MeshMapperConfig::Replicate{}};
    std::vector<MeshCoordinate> coords;
    for (uint32_t s = 0; s < sp_dim; ++s) {
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back((s + sp_origin) % sp_dim, c);
        }
    }
    TensorTopology topology(MeshShape{sp_dim, mesh_cols}, placements, coords);

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::MlaShard;
    policy.bank_order = BankOrder::Optimal;
    policy.k_chunk_size = KChunkSize{k_chunk};
    policy.device_chunk_size = DeviceChunkSize{device_chunk_size};

    CacheConfig config{.spec = spec, .topology = topology, .policy = policy, .base_addr = base};
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = chunk_n_tokens};

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, MeshShape{sp_dim, mesh_cols}, geom);

    uint32_t checked = 0;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (uint32_t pos = 0; pos < max_seq_len; pos += chunk_n_tokens) {
            const RefLoc ref = mla_reference(
                pos, slot, base, F, per_dev, chunk_n_tokens, sp_dim, k_chunk, device_chunk_size, sp_origin);
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

TEST(ToChunkMap, CPU_MlaShardMatchesMigrationReference) { check_mla_shard(0); }

TEST(ToChunkMap, CPU_MlaShardRotatedOriginMatchesMigrationReference) { check_mla_shard(1); }

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

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, MeshShape{1, n_kv_heads}, geom);

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

// --- MiniMax-M3 prefill write layout (models/demos/minimax_m3/tt/runners/kv_chunk_table.py) ---
// Each pipeline stage's caches are [num_users * stage_layers, 1, seq_len, head_dim] per chip, user-major, on
// the stage's own mesh at the stage's own base addresses. SP row r owns tokens [n * chunk + r * chunk_local,
// + chunk_local) of each prefill chunk n; K/V head h lives on TP column h, and index_k is replicated over
// the columns.
constexpr uint32_t kPrefillSp = 2;
constexpr uint32_t kPrefillCols = 4;  // == num_kv_heads
constexpr uint32_t kPrefillHeadDim = 128;
constexpr uint32_t kPrefillChunk = 128;  // chunk_local = 64 tokens: two DRAM blocks per row per chunk
constexpr uint32_t kPrefillSeqLen = 256;
constexpr uint32_t kPrefillUsers = 2;
constexpr uint32_t kPrefillConfigs = 2 * kPrefillCols + 1;
constexpr uint32_t kPrefillChunkBytes = (kPrefillHeadDim / kTileLocal) * kBfp8TileBytesLocal;

struct PrefillStage {
    uint32_t mesh = 0;
    uint32_t first_layer = 0;
    uint32_t num_layers = 0;
    std::array<uint64_t, 3> bases{};  // k, v, index_k
};

struct PrefillLayout {
    std::vector<PrefillStage> stages;
    std::vector<uint32_t> index_k_layers;  // model layers; empty => every layer
    uint32_t num_layers() const { return stages.back().first_layer + stages.back().num_layers; }
};

// One mesh holding all three layers; index_k on layers 0 and 2.
const PrefillLayout kOneMesh = {
    .stages = {{.mesh = 0, .first_layer = 0, .num_layers = 3, .bases = {0x10000, 0x80000, 0xF0000}}},
    .index_k_layers = {0, 2}};
// Two pipeline stages on meshes 0 and 1 holding layers [0, 2) and [2, 5); index_k on layers 1, 2 and 4.
const PrefillLayout kTwoStages = {
    .stages =
        {{.mesh = 0, .first_layer = 0, .num_layers = 2, .bases = {0x10000, 0x80000, 0xF0000}},
         {.mesh = 1, .first_layer = 2, .num_layers = 3, .bases = {0x20000, 0x90000, 0xE0000}}},
    .index_k_layers = {1, 2, 4}};

uint32_t cache_of(uint32_t cfg) { return cfg < kPrefillCols ? 0 : (cfg < 2 * kPrefillCols ? 1 : 2); }

CacheLayoutModel m3_prefill_model(const PrefillLayout& layout) {
    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Natural;
    policy.bank_order = BankOrder::Identity;
    policy.k_chunk_size = KChunkSize{kTileLocal};
    policy.device_chunk_size = DeviceChunkSize{kPrefillChunk / kPrefillSp};

    CacheLayoutModel model;
    for (const PrefillStage& stage : layout.stages) {
        const KvLayoutSpec spec{
            .tensor = make_bfp8_ndshard_spec(
                Shape{kPrefillUsers * stage.num_layers, 1, kPrefillSeqLen, kPrefillHeadDim},
                Shape{1, 1, kTileLocal, kPrefillHeadDim}),
            .temporal = temporal::Dense{},
            .arch = tt::ARCH::BLACKHOLE};
        for (uint32_t cfg = 0; cfg < kPrefillConfigs; ++cfg) {
            const bool index_k = cache_of(cfg) == 2;
            std::vector<MeshCoordinate> coords;
            for (uint32_t r = 0; r < kPrefillSp; ++r) {
                for (uint32_t c = 0; c < kPrefillCols; ++c) {
                    if (index_k || c == cfg % kPrefillCols) {
                        coords.emplace_back(r, c);
                    }
                }
            }
            const TensorTopology topology =
                index_k ? TensorTopology(
                              MeshShape{kPrefillSp, kPrefillCols},
                              {MeshMapperConfig::Shard{.dim = 2}, MeshMapperConfig::Replicate{}},
                              coords)
                        : TensorTopology(MeshShape{kPrefillSp}, {MeshMapperConfig::Shard{.dim = 2}}, coords);
            model.caches.push_back(CacheConfig{
                .spec = spec,
                .topology = topology,
                .policy = policy,
                .base_addr = stage.bases[cache_of(cfg)],
                .name = fmt::format("{:02d}", cfg),
                .layers = index_k ? layout.index_k_layers : std::vector<uint32_t>{},
                .first_layer = stage.first_layer,
                .num_layers = stage.num_layers,
                .mesh_id = tt::tt_fabric::MeshId{stage.mesh}});
        }
    }
    model.mesh_shape = MeshShape{kPrefillSp, kPrefillCols};
    model.geometry = MapGeometry{
        .num_layers = layout.num_layers(),
        .num_slots = kPrefillUsers,
        .max_seq_len = kPrefillSeqLen,
        .position_step = kTileLocal};
    return model;
}

// The prefill writer's walk for one stage's SP row: 32-token blocks round-robin over the banks, the
// per-bank offset advancing after each full sweep, in slot -> stage layer -> chunk -> block order. Keyed
// by (model layer, position, slot); the value is (bank, per-bank offset).
std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::pair<uint32_t, uint64_t>> m3_prefill_walk(
    const PrefillStage& stage, uint32_t row, uint64_t base) {
    const uint32_t chunk_local = kPrefillChunk / kPrefillSp;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::pair<uint32_t, uint64_t>> walk;
    uint32_t bank = 0;
    uint64_t offset = 0;
    for (uint32_t slot = 0; slot < kPrefillUsers; ++slot) {
        for (uint32_t local = 0; local < stage.num_layers; ++local) {
            for (uint32_t n = 0; n < kPrefillSeqLen / kPrefillChunk; ++n) {
                const uint32_t start = n * kPrefillChunk + row * chunk_local;
                for (uint32_t pos = start; pos < start + chunk_local; pos += kTileLocal) {
                    walk[{stage.first_layer + local, pos, slot}] = {bank, base + offset};
                    bank = (bank + 1) % kNumBanks;
                    if (bank == 0) {
                        offset += kPrefillChunkBytes;
                    }
                }
            }
        }
    }
    return walk;
}

void expect_m3_prefill_table(const PrefillLayout& layout, const KvChunkAddressTable& table) {
    ASSERT_EQ(table.num_configs(), kPrefillConfigs);
    for (uint32_t cfg = 0; cfg < kPrefillConfigs; ++cfg) {
        EXPECT_EQ(table.config_name(cfg), fmt::format("{:02d}", cfg));
        EXPECT_EQ(table.config(cfg).num_layers, layout.num_layers());
    }
    const auto publishes = [&](uint32_t layer) {
        return layout.index_k_layers.empty() ||
               std::binary_search(layout.index_k_layers.begin(), layout.index_k_layers.end(), layer);
    };

    uint32_t checked = 0;
    uint32_t published_index_k_layers = 0;
    for (uint32_t layer = 0; layer < layout.num_layers(); ++layer) {
        published_index_k_layers += publishes(layer) ? 1 : 0;
    }
    for (uint32_t cfg = 0; cfg < kPrefillConfigs; ++cfg) {
        const bool index_k = cache_of(cfg) == 2;
        const uint32_t head = cfg % kPrefillCols;
        for (const PrefillStage& stage : layout.stages) {
            for (uint32_t row = 0; row < kPrefillSp; ++row) {
                for (const auto& [key, expected] : m3_prefill_walk(stage, row, stage.bases[cache_of(cfg)])) {
                    const auto& [layer, pos, slot] = key;
                    const KvCacheLocation loc = table.lookup(layer, pos, slot, cfg);
                    if (index_k && !publishes(layer)) {
                        EXPECT_EQ(loc.size_bytes, 0u) << "unpublished index_k layer " << layer << " has a row";
                        continue;
                    }
                    EXPECT_EQ(loc.size_bytes, kPrefillChunkBytes);
                    EXPECT_EQ(addr_channel(loc.noc_addr), expected.first) << "cfg=" << cfg << " pos=" << pos;
                    EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(expected.second))
                        << "cfg=" << cfg << " layer=" << layer << " slot=" << slot << " pos=" << pos;
                    const auto& group = table.get_device_group(loc.device_group_index);
                    ASSERT_EQ(group.fabric_node_ids.size(), index_k ? kPrefillCols : 1u);
                    for (uint32_t i = 0; i < group.fabric_node_ids.size(); ++i) {
                        const uint32_t col = index_k ? i : head;
                        EXPECT_EQ(*group.fabric_node_ids[i].mesh_id, stage.mesh);
                        EXPECT_EQ(group.fabric_node_ids[i].chip_id, row * kPrefillCols + col);
                    }
                    ++checked;
                }
            }
        }
    }
    const uint32_t positions = kPrefillSeqLen / kTileLocal;
    EXPECT_EQ(checked, (2 * kPrefillCols * layout.num_layers() + published_index_k_layers) * kPrefillUsers * positions);
}

TEST(ToChunkMap, CPU_M3PrefillMatchesTheWritersBankWalk) {
    expect_m3_prefill_table(kOneMesh, to_chunk_map(m3_prefill_model(kOneMesh)));
}

TEST(ToChunkMap, CPU_M3PipelinedPrefillMatchesEachStagesBankWalk) {
    expect_m3_prefill_table(kTwoStages, to_chunk_map(m3_prefill_model(kTwoStages)));
}

TEST(ToChunkMap, CPU_FoldedLayersNeedTheNaturalIdentityLayout) {
    CacheLayoutModel model = m3_prefill_model(kOneMesh);
    model.caches.front().policy.bank_order = BankOrder::Optimal;
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_ConfigNamesAreAllSetOrAllEmpty) {
    CacheLayoutModel model = m3_prefill_model(kOneMesh);
    model.caches.back().name.clear();
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_PublishedLayersMustBeAscending) {
    CacheLayoutModel model = m3_prefill_model(kOneMesh);
    model.caches.back().layers = {2, 0};
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_StagesOfOneConfigMustNotOverlap) {
    CacheLayoutModel model = m3_prefill_model(kTwoStages);
    model.caches[kPrefillConfigs].first_layer = 1;  // stage 1's k head 0 now holds [1, 4)
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_ACacheMustEndWithinTheModel) {
    CacheLayoutModel model = m3_prefill_model(kTwoStages);
    model.caches[kPrefillConfigs].num_layers = 4;  // [2, 6) of 5 layers
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_StagesOfOneConfigMustShareAChunkSize) {
    CacheLayoutModel model = m3_prefill_model(kTwoStages);
    CacheConfig& stage_one = model.caches[kPrefillConfigs];
    stage_one.spec.tensor = make_bfp8_ndshard_spec(
        Shape{kPrefillUsers * 3, 1, kPrefillSeqLen, 2 * kPrefillHeadDim}, Shape{1, 1, kTileLocal, 2 * kPrefillHeadDim});
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(ToChunkMap, CPU_FirstTableDifferenceNamesTheFirstDifferingCell) {
    const KvChunkAddressTable table = to_chunk_map(m3_prefill_model(kTwoStages));
    EXPECT_EQ(first_table_difference(table, to_chunk_map(m3_prefill_model(kTwoStages))), std::nullopt);

    PrefillLayout moved = kTwoStages;
    moved.stages[1].bases[0] += 0x1000;
    const auto difference = first_table_difference(table, to_chunk_map(m3_prefill_model(moved)));
    ASSERT_TRUE(difference.has_value());
    EXPECT_NE(difference->find("config '00' layer 2 slot 0 position 0: noc_addr"), std::string::npos) << *difference;

    PrefillLayout fewer = kTwoStages;
    fewer.index_k_layers = {1, 2};
    const auto missing = first_table_difference(table, to_chunk_map(m3_prefill_model(fewer)));
    ASSERT_TRUE(missing.has_value());
    EXPECT_NE(missing->find("config '08' layer 4"), std::string::npos) << *missing;
}

// --- Gemma-4 sliding K/V (models/demos/gemma4_d_p/tt/runners/kv_chunk_table.py) ---
// One tensor per layer, per chip [num_users, heads_per_chip, seq_len / cp, 256]; head h on TP column
// h / heads_per_chip. Each head is its own named config. The writer's shard ravel is user-major, then the
// chip's local head: shard = (slot * heads_per_chip + local_head) * blocks + local_row / 32.
constexpr uint32_t kGemmaCp = 2;
constexpr uint32_t kGemmaTp = 2;
constexpr uint32_t kGemmaHeads = 4;  // two per chip
constexpr uint32_t kGemmaHeadsPerChip = kGemmaHeads / kGemmaTp;
constexpr uint32_t kGemmaHeadDim = 256;
constexpr uint32_t kGemmaSeqLen = 256;
constexpr uint32_t kGemmaChunk = 128;
constexpr uint32_t kGemmaUsers = 2;
constexpr uint32_t kGemmaLayers = 2;
constexpr uint32_t kGemmaChunkBytes = (kGemmaHeadDim / kTileLocal) * kBfp8TileBytesLocal;

uint64_t gemma_base(uint32_t layer) { return 0x100000ull * (layer + 1); }

CacheLayoutModel gemma_sliding_model() {
    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Natural;
    policy.bank_order = BankOrder::Identity;
    policy.k_chunk_size = KChunkSize{kTileLocal};
    policy.device_chunk_size = DeviceChunkSize{kGemmaChunk / kGemmaCp};
    const KvLayoutSpec spec{
        .tensor = make_bfp8_ndshard_spec(
            Shape{kGemmaUsers, kGemmaHeads, kGemmaSeqLen, kGemmaHeadDim}, Shape{1, 1, kTileLocal, kGemmaHeadDim}),
        .temporal = temporal::Dense{},
        .arch = tt::ARCH::BLACKHOLE};
    const TensorTopology topology(
        MeshShape{kGemmaCp, kGemmaTp},
        {MeshMapperConfig::Shard{.dim = 2}, MeshMapperConfig::Shard{.dim = 1}},
        row_major_coords(kGemmaCp, kGemmaTp));

    CacheLayoutModel model;
    for (uint32_t layer = 0; layer < kGemmaLayers; ++layer) {
        for (uint32_t h = 0; h < kGemmaHeads; ++h) {
            model.caches.push_back(CacheConfig{
                .spec = spec,
                .topology = topology,
                .policy = policy,
                .base_addr = gemma_base(layer),
                .name = fmt::format("sliding_k_h{:02d}", h),
                .first_layer = layer,
                .num_layers = 1,
                .head = h});
        }
    }
    model.mesh_shape = MeshShape{kGemmaCp, kGemmaTp};
    model.geometry = MapGeometry{
        .num_layers = kGemmaLayers, .num_slots = kGemmaUsers, .max_seq_len = kGemmaSeqLen, .position_step = kTileLocal};
    return model;
}

TEST(ToChunkMap, CPU_GemmaSlidingHeadsMatchTheWritersShardRavel) {
    const KvChunkAddressTable table = to_chunk_map(gemma_sliding_model());
    ASSERT_EQ(table.num_configs(), kGemmaHeads);
    const uint32_t local_tokens = kGemmaChunk / kGemmaCp;
    const uint32_t blocks = kGemmaSeqLen / kGemmaCp / kTileLocal;
    uint32_t checked = 0;
    for (uint32_t h = 0; h < kGemmaHeads; ++h) {
        const uint32_t cfg = table.config_id_of(fmt::format("sliding_k_h{:02d}", h));
        EXPECT_EQ(table.config(cfg).num_slots, kGemmaUsers);
        for (uint32_t layer = 0; layer < kGemmaLayers; ++layer) {
            for (uint32_t slot = 0; slot < kGemmaUsers; ++slot) {
                for (uint32_t pos = 0; pos < kGemmaSeqLen; pos += kTileLocal) {
                    const uint32_t in_chunk = pos % kGemmaChunk;
                    const uint32_t cp_row = in_chunk / local_tokens;
                    const uint32_t local_row = (pos / kGemmaChunk) * local_tokens + in_chunk % local_tokens;
                    const uint32_t shard =
                        (slot * kGemmaHeadsPerChip + h % kGemmaHeadsPerChip) * blocks + local_row / kTileLocal;
                    const KvCacheLocation loc = table.lookup(layer, pos, slot, cfg);
                    EXPECT_EQ(loc.size_bytes, kGemmaChunkBytes);
                    EXPECT_EQ(addr_channel(loc.noc_addr), shard % kNumBanks) << "h=" << h << " pos=" << pos;
                    EXPECT_EQ(
                        addr_local(loc.noc_addr),
                        static_cast<uint32_t>(gemma_base(layer) + (shard / kNumBanks) * kGemmaChunkBytes))
                        << "h=" << h << " layer=" << layer << " slot=" << slot << " pos=" << pos;
                    const auto& group = table.get_device_group(loc.device_group_index);
                    ASSERT_EQ(group.fabric_node_ids.size(), 1u);
                    EXPECT_EQ(group.fabric_node_ids[0].chip_id, cp_row * kGemmaTp + h / kGemmaHeadsPerChip);
                    ++checked;
                }
            }
        }
    }
    EXPECT_EQ(checked, kGemmaHeads * kGemmaLayers * kGemmaUsers * (kGemmaSeqLen / kTileLocal));
}

TEST(ToChunkMap, CPU_AHeadSelectorNeedsAHeadShardedTopology) {
    CacheLayoutModel model = m3_prefill_model(kOneMesh);
    model.caches.front().head = 0;  // M3's K head config shards no head axis
    EXPECT_ANY_THROW(to_chunk_map(model));
    model = gemma_sliding_model();
    model.caches.front().head = kGemmaHeads;
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(CacheLayoutModel, CPU_AHeadSelectorRoundTrips) {
    const CacheLayoutModel model = gemma_sliding_model();
    const CacheLayoutModel imported = import_cache_layout_model_from_protobuf(export_to_protobuf(model));
    for (size_t i = 0; i < model.caches.size(); ++i) {
        EXPECT_EQ(imported.caches[i].head, model.caches[i].head);
    }
    EXPECT_EQ(first_table_difference(to_chunk_map(model), to_chunk_map(imported)), std::nullopt);
}

TEST(CacheLayoutModel, CPU_RoundTripsThroughProtobufToTheSameTable) {
    for (const PrefillLayout& layout : {kOneMesh, kTwoStages}) {
        const CacheLayoutModel model = m3_prefill_model(layout);
        const CacheLayoutModel imported = import_cache_layout_model_from_protobuf(export_to_protobuf(model));

        ASSERT_EQ(imported.caches.size(), model.caches.size());
        for (size_t i = 0; i < model.caches.size(); ++i) {
            const CacheConfig& want = model.caches[i];
            const CacheConfig& got = imported.caches[i];
            EXPECT_EQ(got.name, want.name);
            EXPECT_EQ(got.layers, want.layers);
            EXPECT_EQ(got.first_layer, want.first_layer);
            EXPECT_EQ(got.num_layers, want.num_layers);
            EXPECT_EQ(got.mesh_id, want.mesh_id);
            EXPECT_EQ(got.base_addr, want.base_addr);
            EXPECT_EQ(got.spec.tensor, want.spec.tensor);
            EXPECT_EQ(got.spec.arch, want.spec.arch);
            EXPECT_TRUE(std::holds_alternative<temporal::Dense>(got.spec.temporal));
            EXPECT_EQ(got.topology, want.topology);
            EXPECT_EQ(got.policy.bank_scheme, want.policy.bank_scheme);
            EXPECT_EQ(got.policy.bank_order, want.policy.bank_order);
            EXPECT_EQ(got.policy.k_chunk_size, want.policy.k_chunk_size);
            EXPECT_EQ(got.policy.device_chunk_size, want.policy.device_chunk_size);
        }
        EXPECT_EQ(*imported.mesh_id, *model.mesh_id);
        EXPECT_EQ(imported.mesh_shape, model.mesh_shape);
        EXPECT_EQ(imported.geometry.num_layers, model.geometry.num_layers);
        EXPECT_EQ(imported.geometry.num_slots, model.geometry.num_slots);
        EXPECT_EQ(imported.geometry.max_seq_len, model.geometry.max_seq_len);
        EXPECT_EQ(imported.geometry.position_step, model.geometry.position_step);
        expect_m3_prefill_table(layout, to_chunk_map(imported));
    }
}

TEST(CacheLayoutModel, CPU_EveryTemporalPolicyRoundTrips) {
    const std::vector<TemporalPolicy> policies = {
        temporal::Dense{},
        temporal::Window{WindowTokens{128}},
        temporal::BlockLocal{BlockLocalSpan{8192}},
        temporal::Rolling{},
        temporal::Static{},
        temporal::None{}};
    for (const TemporalPolicy& policy : policies) {
        CacheLayoutModel model = m3_prefill_model(kOneMesh);
        model.caches.front().spec.temporal = policy;
        const TemporalPolicy got =
            import_cache_layout_model_from_protobuf(export_to_protobuf(model)).caches.front().spec.temporal;
        ASSERT_EQ(got.index(), policy.index());
        if (const auto* window = std::get_if<temporal::Window>(&policy)) {
            EXPECT_EQ(std::get<temporal::Window>(got).width, window->width);
        }
        if (const auto* local = std::get_if<temporal::BlockLocal>(&policy)) {
            EXPECT_EQ(std::get<temporal::BlockLocal>(got).span, local->span);
        }
    }
}

TEST(CacheLayoutModel, CPU_RejectsBytesThatAreNotAModel) {
    EXPECT_ANY_THROW(import_cache_layout_model_from_protobuf(std::string("\xff\xff\xff", 3)));
}

// --- Kimi TP-deduped kvpe: the sequence sharded over both mesh axes, one chip per sub-slice ---
// Chip (r, c) owns [n*chunk + r*chunk/sp + c*chunk/(sp*tp), +chunk/(sp*tp)) of each prefill chunk n, and pages
// its own tokens from bank 0 (kv_cache_utils.create_kv_chunk_address_table with tp_axis set).
TEST(ToChunkMap, CPU_SequenceShardedOverBothMeshAxesMatchesTheDedupedWalk) {
    const uint32_t sp = 2;
    const uint32_t tp = 4;
    const uint64_t F = 576;
    const uint32_t chunk = 1024;
    const uint32_t tokens_per_device = chunk / (sp * tp);
    const uint32_t max_seq_len = 2048;
    const uint32_t per_dev_seq = max_seq_len / (sp * tp);
    const uint32_t num_slots = 2;
    const uint64_t base = 0x40000ull;
    const uint32_t csb = bfp8_chunk_bytes(kTileLocal, F);

    KvLayoutSpec spec{
        .tensor = make_bfp8_ndshard_spec(Shape{num_slots, max_seq_len, F}, Shape{1, kTileLocal, F}),
        .temporal = temporal::Dense{},
        .arch = tt::ARCH::BLACKHOLE};
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Shard{.dim = 1}, MeshMapperConfig::Shard{.dim = 1}};
    TensorTopology topology(MeshShape{sp, tp}, placements, row_major_coords(sp, tp));

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Natural;
    policy.bank_order = BankOrder::Identity;
    policy.k_chunk_size = KChunkSize{kTileLocal};
    policy.device_chunk_size = DeviceChunkSize{tokens_per_device};

    CacheConfig config{.spec = spec, .topology = topology, .policy = policy, .base_addr = base};
    MapGeometry geom{.num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = kTileLocal};

    auto table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, MeshShape{sp, tp}, geom);

    uint32_t checked = 0;
    for (uint32_t r = 0; r < sp; ++r) {
        for (uint32_t c = 0; c < tp; ++c) {
            uint32_t page = 0;
            for (uint32_t slot = 0; slot < num_slots; ++slot) {
                for (uint32_t n = 0; n < max_seq_len / chunk; ++n) {
                    const uint32_t start = n * chunk + r * (chunk / sp) + c * tokens_per_device;
                    for (uint32_t pos = start; pos < start + tokens_per_device; pos += kTileLocal, ++page) {
                        const KvCacheLocation loc = table.lookup(0, pos, slot, 0u);
                        EXPECT_EQ(addr_channel(loc.noc_addr), page % kNumBanks)
                            << "r=" << r << " c=" << c << " pos=" << pos;
                        EXPECT_EQ(addr_local(loc.noc_addr), base + (page / kNumBanks) * csb)
                            << "r=" << r << " c=" << c << " pos=" << pos;
                        const auto& grp = table.get_device_group(loc.device_group_index);
                        ASSERT_EQ(grp.fabric_node_ids.size(), 1u);
                        EXPECT_EQ(grp.fabric_node_ids[0].chip_id, r * tp + c);
                        ++checked;
                    }
                }
            }
            EXPECT_EQ(page, num_slots * per_dev_seq / kTileLocal);
        }
    }
    EXPECT_EQ(checked, num_slots * max_seq_len / kTileLocal);
}

// --- An attention cache on only some layers (Kimi's MLA layers between KDA ones): each stage's cache
// holds its published layers back to back, slot x held layer on the batch axis ---
const std::vector<uint32_t> kMlaLayers = {1, 2, 4};

CacheLayoutModel mla_only_model(const PrefillLayout& layout) {
    CacheLayoutModel model = m3_prefill_model(layout);
    std::vector<CacheConfig> caches;
    for (const CacheConfig& cache : model.caches) {
        if (cache.name != "00") {
            continue;
        }
        CacheConfig mla = cache;
        const auto first = std::lower_bound(kMlaLayers.begin(), kMlaLayers.end(), cache.first_layer);
        const auto end = std::lower_bound(kMlaLayers.begin(), kMlaLayers.end(), cache.first_layer + cache.num_layers);
        const uint32_t held = static_cast<uint32_t>(end - first);
        mla.spec.tensor = make_bfp8_ndshard_spec(
            Shape{kPrefillUsers * held, 1, kPrefillSeqLen, kPrefillHeadDim}, Shape{1, 1, kTileLocal, kPrefillHeadDim});
        mla.layers = kMlaLayers;
        mla.holds_only_published_layers = true;
        caches.push_back(mla);
    }
    model.caches = caches;
    return model;
}

TEST(ToChunkMap, CPU_ACacheHoldingOnlyItsPublishedLayersFoldsJustThose) {
    const KvChunkAddressTable table = to_chunk_map(mla_only_model(kTwoStages));
    uint32_t checked = 0;
    for (const PrefillStage& stage : kTwoStages.stages) {
        std::vector<uint32_t> held;
        for (uint32_t layer : kMlaLayers) {
            if (layer >= stage.first_layer && layer < stage.first_layer + stage.num_layers) {
                held.push_back(layer);
            }
        }
        for (uint32_t row = 0; row < kPrefillSp; ++row) {
            const uint32_t chunk_local = kPrefillChunk / kPrefillSp;
            uint32_t page = 0;
            for (uint32_t slot = 0; slot < kPrefillUsers; ++slot) {
                for (uint32_t layer : held) {
                    for (uint32_t n = 0; n < kPrefillSeqLen / kPrefillChunk; ++n) {
                        const uint32_t start = n * kPrefillChunk + row * chunk_local;
                        for (uint32_t pos = start; pos < start + chunk_local; pos += kTileLocal, ++page) {
                            const KvCacheLocation loc = table.lookup(layer, pos, slot, 0u);
                            EXPECT_EQ(addr_channel(loc.noc_addr), page % kNumBanks) << "layer=" << layer;
                            EXPECT_EQ(
                                addr_local(loc.noc_addr), stage.bases[0] + (page / kNumBanks) * kPrefillChunkBytes)
                                << "layer=" << layer << " slot=" << slot << " pos=" << pos;
                            ++checked;
                        }
                    }
                }
            }
        }
        for (uint32_t layer = stage.first_layer; layer < stage.first_layer + stage.num_layers; ++layer) {
            if (!std::binary_search(kMlaLayers.begin(), kMlaLayers.end(), layer)) {
                EXPECT_EQ(table.lookup(layer, 0, 0, 0u).size_bytes, 0u) << "non-MLA layer " << layer << " has a row";
            }
        }
    }
    EXPECT_EQ(checked, kMlaLayers.size() * kPrefillUsers * (kPrefillSeqLen / kTileLocal));
}

TEST(ToChunkMap, CPU_ACacheHoldingOnlyItsPublishedLayersMustBeSizedForThem) {
    CacheLayoutModel model = mla_only_model(kTwoStages);
    // Stage 1 holds layers 2 and 4; declare a batch axis of every stage layer instead.
    model.caches.back().spec.tensor = make_bfp8_ndshard_spec(
        Shape{kPrefillUsers * 3, 1, kPrefillSeqLen, kPrefillHeadDim}, Shape{1, 1, kTileLocal, kPrefillHeadDim});
    EXPECT_ANY_THROW(to_chunk_map(model));
    model = mla_only_model(kTwoStages);
    model.caches.front().layers.clear();
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(CacheLayoutModel, CPU_HoldingOnlyPublishedLayersRoundTrips) {
    const CacheLayoutModel model = mla_only_model(kTwoStages);
    const CacheLayoutModel imported = import_cache_layout_model_from_protobuf(export_to_protobuf(model));
    for (size_t i = 0; i < model.caches.size(); ++i) {
        EXPECT_TRUE(imported.caches[i].holds_only_published_layers);
    }
    EXPECT_EQ(first_table_difference(to_chunk_map(model), to_chunk_map(imported)), std::nullopt);
}

TEST(ToChunkMap, CPU_ATableOfSomeConfigsIsComparedByName) {
    const KvChunkAddressTable all = to_chunk_map(m3_prefill_model(kTwoStages));
    CacheLayoutModel some = m3_prefill_model(kTwoStages);
    std::erase_if(some.caches, [](const CacheConfig& cache) { return cache.name != "05" && cache.name != "08"; });
    const KvChunkAddressTable derived = to_chunk_map(some);

    EXPECT_NE(first_table_difference(all, derived), std::nullopt);
    EXPECT_EQ(first_table_difference(all, derived, /*only_actual_configs=*/true), std::nullopt);

    some.caches.back().base_addr += 0x1000;
    const auto difference = first_table_difference(all, to_chunk_map(some), /*only_actual_configs=*/true);
    ASSERT_TRUE(difference.has_value());
    EXPECT_NE(difference->find("config '08'"), std::string::npos) << *difference;

    some.caches.back().name = "09";
    EXPECT_NE(first_table_difference(all, to_chunk_map(some), /*only_actual_configs=*/true), std::nullopt);
}

// --- A ROW_MAJOR cache (GLM's scaled-FP8 KVPE): one page per token, padded to the DRAM alignment ---
TEST(ToChunkMap, CPU_ARowMajorCacheChunksTokensOfAlignedPages) {
    const uint32_t row_bytes = 656;  // packed FP8 latent + scales, not a multiple of the DRAM alignment
    const uint32_t aligned_row_bytes = 704;
    const uint32_t sp = 4;
    const uint32_t num_slots = 2;
    const uint32_t max_seq_len = 1024;
    const uint32_t per_dev_seq = max_seq_len / sp;
    const uint64_t base = 0x80000ull;

    CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(kNumBanks - 1, 0)));
    NdShardSpec nd{
        Shape{1, 1, kTileLocal, row_bytes},
        grid,
        ShardOrientation::ROW_MAJOR,
        ShardDistributionStrategy::ROUND_ROBIN_1D};
    const TensorSpec tensor(
        Shape{num_slots, 1, max_seq_len, row_bytes},
        TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), MemoryConfig(BufferType::DRAM, nd)));
    const KvLayoutSpec spec{.tensor = tensor, .temporal = temporal::Dense{}, .arch = tt::ARCH::BLACKHOLE};
    ASSERT_EQ(dram_alignment(tt::ARCH::BLACKHOLE), 64u);
    EXPECT_EQ(chunk_size_bytes(spec, kTileLocal), kTileLocal * aligned_row_bytes);

    GenerationPolicy policy;
    policy.bank_scheme = BankScheme::Natural;
    policy.bank_order = BankOrder::Identity;
    policy.k_chunk_size = KChunkSize{kTileLocal};
    policy.device_chunk_size = DeviceChunkSize{per_dev_seq};
    std::vector<MeshCoordinate> coords;
    for (uint32_t r = 0; r < sp; ++r) {
        coords.emplace_back(r, 0);
    }
    const CacheConfig config{
        .spec = spec,
        .topology = TensorTopology(MeshShape{sp}, {MeshMapperConfig::Shard{.dim = 2}}, coords),
        .policy = policy,
        .base_addr = base};
    const MapGeometry geom{
        .num_layers = 1, .num_slots = num_slots, .max_seq_len = max_seq_len, .position_step = kTileLocal};
    const KvChunkAddressTable table = to_chunk_map({config}, tt::tt_fabric::MeshId{0}, MeshShape{sp, 1}, geom);

    EXPECT_EQ(table.config(0).chunk_size_bytes, kTileLocal * aligned_row_bytes);
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (uint32_t pos = 0; pos < max_seq_len; pos += kTileLocal) {
            const uint32_t page = slot * (per_dev_seq / kTileLocal) + (pos % per_dev_seq) / kTileLocal;
            const KvCacheLocation loc = table.lookup(0, pos, slot, 0u);
            EXPECT_EQ(addr_channel(loc.noc_addr), page % kNumBanks) << "slot=" << slot << " pos=" << pos;
            EXPECT_EQ(addr_local(loc.noc_addr), base + (page / kNumBanks) * kTileLocal * aligned_row_bytes)
                << "slot=" << slot << " pos=" << pos;
            const auto& group = table.get_device_group(loc.device_group_index);
            ASSERT_EQ(group.fabric_node_ids.size(), 1u);
            EXPECT_EQ(group.fabric_node_ids[0].chip_id, pos / per_dev_seq);
        }
    }
}

// --- Layer aliases (GLM's MTP levels past the computed one read level 1's rows) ---
CacheLayoutModel aliased_model() {
    CacheLayoutModel model = m3_prefill_model(kTwoStages);
    model.geometry.num_layers = kTwoStages.num_layers() + 2;
    for (CacheConfig& cache : model.caches) {
        if (cache.name == "00" && cache.first_layer > 0) {
            cache.layer_aliases = {{5, 4}, {6, 4}};
        }
    }
    return model;
}

// The second stage's K head 0 cache, which holds layers [2, 5).
CacheConfig& second_stage_k_h0(CacheLayoutModel& model) {
    return *std::find_if(model.caches.begin(), model.caches.end(), [](const CacheConfig& cache) {
        return cache.name == "00" && cache.first_layer > 0;
    });
}

TEST(ToChunkMap, CPU_AnAliasLayerPublishesItsSourceLayersChunks) {
    const KvChunkAddressTable table = to_chunk_map(aliased_model());
    const uint32_t k_h0 = table.config_id_of("00");
    uint32_t checked = 0;
    for (uint32_t slot = 0; slot < kPrefillUsers; ++slot) {
        for (uint32_t pos = 0; pos < kPrefillSeqLen; pos += kTileLocal) {
            const KvCacheLocation source = table.lookup(4, pos, slot, k_h0);
            for (uint32_t alias : {5u, 6u}) {
                const KvCacheLocation loc = table.lookup(alias, pos, slot, k_h0);
                EXPECT_EQ(loc.size_bytes, source.size_bytes);
                EXPECT_EQ(loc.noc_addr, source.noc_addr) << "alias=" << alias << " pos=" << pos;
                EXPECT_EQ(loc.device_group_index, source.device_group_index);
            }
            EXPECT_EQ(table.lookup(5, pos, slot, table.config_id_of("01")).size_bytes, 0u);
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

TEST(ToChunkMap, CPU_AnAliasNeedsAPublishedSourceAndALayerOutsideTheCache) {
    CacheLayoutModel model = aliased_model();
    second_stage_k_h0(model).layer_aliases = {{5, 1}};  // layer 1 is the first stage's, not this cache's
    EXPECT_ANY_THROW(to_chunk_map(model));
    model = aliased_model();
    second_stage_k_h0(model).layer_aliases = {{3, 4}};  // layer 3 is a layer the cache holds
    EXPECT_ANY_THROW(to_chunk_map(model));
    model = aliased_model();
    second_stage_k_h0(model).layer_aliases = {{7, 4}};  // past the model
    EXPECT_ANY_THROW(to_chunk_map(model));
}

TEST(CacheLayoutModel, CPU_LayerAliasesRoundTrip) {
    const CacheLayoutModel model = aliased_model();
    const CacheLayoutModel imported = import_cache_layout_model_from_protobuf(export_to_protobuf(model));
    for (size_t i = 0; i < model.caches.size(); ++i) {
        EXPECT_EQ(imported.caches[i].layer_aliases, model.caches[i].layer_aliases);
    }
    EXPECT_EQ(first_table_difference(to_chunk_map(model), to_chunk_map(imported)), std::nullopt);
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
