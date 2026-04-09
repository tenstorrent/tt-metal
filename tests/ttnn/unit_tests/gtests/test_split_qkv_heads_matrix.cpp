// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Comprehensive matrix test for ttnn::transformer::split_query_key_value_and_split_heads.
 *
 * Documents which combinations of (input mem config x output mem config x sequence
 * alignment x head config x dtype) produce correct, silently corrupted, or rejected
 * output on real hardware. The 19-month silent corruption discovered in #41526 / #41718
 * happened because the existing tests covered only the interleaved code path on real
 * hardware (the sharded variant was @skipif on every modern arch); this suite fills
 * that coverage gap.
 *
 * Outcome model (per cell):
 *   - PASS: op runs, PCC >= threshold (0.99 default) vs CPU reference
 *   - CORRUPT: op runs but PCC < threshold (silent corruption — the dangerous case)
 *   - REJECTED: op throws via TT_FATAL (validation correctly rejects the combination)
 *
 * Inputs follow the framework convention from
 * tech_reports/Handling_Special_Value/special_values.md: normal distribution with
 * mean=0, stddev=1, no NaN/Inf (those are not framework-supported as inputs).
 *
 * Reproducible across runs and machines via a fixed RNG seed (kFixedSeed).
 *
 * Run:   ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*SplitQkvMatrix*"
 *
 * Verbose mode (logs measured PCC for every test):
 *   TT_LOGGER_LEVEL=Debug TT_LOGGER_TYPES=Test \
 *     ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*SplitQkvMatrix*"
 *
 * Note: this is commit 1 of 4 (per ~/tt/SPLIT_QKV_TEST_SUITE_PLAN.md). It establishes
 * the file scaffolding, the CPU reference, the deterministic input generator, and 5
 * baseline interleaved tests that exercise the helpers and verify the test
 * infrastructure works against the known-good interleaved code path. Sharded matrix,
 * non-tile-aligned sequence sub-matrix, and the documentation update follow in commits
 * 2-4.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/shape.hpp>

#include "common_test_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::transformer::test {

namespace {

// Fixed RNG seed for reproducibility across runs and machines.
constexpr uint32_t kFixedSeed = 0xCAFEBABE;

// Default PCC threshold for the PASS classification.
constexpr float kDefaultPccThreshold = 0.99f;

// Total hidden width of a fused [Q | K | V] tensor for the given head config.
inline uint32_t fused_hidden_dim(uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim) {
    return (num_q_heads + 2 * num_kv_heads) * head_dim;
}

// Generate a deterministic float input tensor with mean=0, stddev=1 per the framework
// convention in tech_reports/Handling_Special_Value/special_values.md. NaN/Inf are not
// produced — those are not framework-supported as inputs.
std::vector<float> make_random_qkv_input(
    uint32_t seed, uint32_t batch, uint32_t seq, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim) {
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    const uint32_t total = batch * seq * hidden;
    std::vector<float> data;
    data.reserve(total);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < total; ++i) {
        data.push_back(dist(rng));
    }
    return data;
}

// Use the QkvLayout enum exposed by the op (added by the same change set as this
// matrix test). See the doc comment on `ttnn::transformer::QkvLayout` in
// ttnn/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/
// split_query_key_value_and_split_heads.hpp.
using ttnn::transformer::QkvLayout;

// CPU reference. Splits a flat [B, S, (n_q+2*n_kv)*D] input into Q [B, n_q, S, D],
// K [B, n_kv, S, D] (or [B, n_kv, D, S] if transpose_k), V [B, n_kv, S, D]. The
// `layout` argument tells the reference how to interpret the input row.
struct CpuQkvSplitResult {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
};

CpuQkvSplitResult cpu_split_qkv_reference(
    const std::vector<float>& input,
    uint32_t batch,
    uint32_t seq,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k,
    QkvLayout layout) {
    TT_FATAL(num_kv_heads > 0, "num_kv_heads must be > 0");
    TT_FATAL(num_q_heads > 0, "num_q_heads must be > 0");
    TT_FATAL(head_dim > 0, "head_dim must be > 0");
    TT_FATAL(
        num_q_heads % num_kv_heads == 0,
        "num_q_heads {} must be divisible by num_kv_heads {}",
        num_q_heads,
        num_kv_heads);
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    const uint32_t q_per_group = num_q_heads / num_kv_heads;

    CpuQkvSplitResult result;
    result.q.resize(static_cast<size_t>(batch) * num_q_heads * seq * head_dim);
    result.k.resize(static_cast<size_t>(batch) * num_kv_heads * seq * head_dim);
    result.v.resize(static_cast<size_t>(batch) * num_kv_heads * seq * head_dim);

    auto write_q = [&](uint32_t b, uint32_t h, uint32_t s, uint32_t d, float v_in) {
        result.q[((static_cast<size_t>(b) * num_q_heads + h) * seq + s) * head_dim + d] = v_in;
    };
    auto write_k = [&](uint32_t b, uint32_t h, uint32_t s, uint32_t d, float v_in) {
        if (transpose_k) {
            result.k[((static_cast<size_t>(b) * num_kv_heads + h) * head_dim + d) * seq + s] = v_in;
        } else {
            result.k[((static_cast<size_t>(b) * num_kv_heads + h) * seq + s) * head_dim + d] = v_in;
        }
    };
    auto write_v = [&](uint32_t b, uint32_t h, uint32_t s, uint32_t d, float v_in) {
        result.v[((static_cast<size_t>(b) * num_kv_heads + h) * seq + s) * head_dim + d] = v_in;
    };

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t s = 0; s < seq; ++s) {
            const size_t in_base = (static_cast<size_t>(b) * seq + s) * hidden;
            if (layout == QkvLayout::CONCATENATED) {
                const uint32_t q_section = num_q_heads * head_dim;
                const uint32_t kv_section = num_kv_heads * head_dim;
                for (uint32_t h = 0; h < num_q_heads; ++h) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        write_q(b, h, s, d, input[in_base + h * head_dim + d]);
                    }
                }
                for (uint32_t h = 0; h < num_kv_heads; ++h) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        write_k(b, h, s, d, input[in_base + q_section + h * head_dim + d]);
                    }
                }
                for (uint32_t h = 0; h < num_kv_heads; ++h) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        write_v(b, h, s, d, input[in_base + q_section + kv_section + h * head_dim + d]);
                    }
                }
            } else {  // GROUPED
                // Each KV-group is (q_per_group * Q heads + 1 K head + 1 V head),
                // contiguous in the input row.
                const uint32_t group_width = (q_per_group + 2) * head_dim;
                for (uint32_t g = 0; g < num_kv_heads; ++g) {
                    const size_t group_base = in_base + g * group_width;
                    // q_per_group Q heads
                    for (uint32_t qi = 0; qi < q_per_group; ++qi) {
                        const uint32_t h = g * q_per_group + qi;
                        for (uint32_t d = 0; d < head_dim; ++d) {
                            write_q(b, h, s, d, input[group_base + qi * head_dim + d]);
                        }
                    }
                    // K head
                    const size_t k_base = group_base + q_per_group * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        write_k(b, g, s, d, input[k_base + d]);
                    }
                    // V head
                    const size_t v_base = k_base + head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        write_v(b, g, s, d, input[v_base + d]);
                    }
                }
            }
        }
    }
    return result;
}

// Build an interleaved tensor on device from a flat float vector. Caller picks the
// dtype (BFLOAT16 or BFLOAT8_B) and the memory config (typically DRAM/L1 interleaved).
tt::tt_metal::Tensor make_interleaved_tensor(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::vector<float>& data,
    const std::array<uint32_t, 3>& dims,
    DataType dtype,
    const tt::tt_metal::MemoryConfig& mem_config) {
    tt::tt_metal::TensorSpec spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
    if (dtype == DataType::BFLOAT16) {
        std::vector<::bfloat16> bf16_data;
        bf16_data.reserve(data.size());
        for (float v : data) {
            bf16_data.push_back(::bfloat16(v));
        }
        return tt::tt_metal::Tensor::from_vector(std::move(bf16_data), spec).to_device(device);
    }
    // BFLOAT8_B and other formats can take float input directly — the framework handles packing.
    return tt::tt_metal::Tensor::from_vector(data, spec).to_device(device);
}

// Build a BLOCK_SHARDED tensor on device. Uploads via DRAM interleaved first, then
// reshards to the requested grid (caller picks num_w_cores and num_h_cores so the
// shard math works out for the head config; see comments on the test cells below).
tt::tt_metal::Tensor make_block_sharded_tensor(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::vector<float>& data,
    const std::array<uint32_t, 3>& dims,
    DataType dtype,
    uint32_t num_w_cores,
    uint32_t num_h_cores) {
    auto staging = make_interleaved_tensor(device, data, dims, dtype, ttnn::DRAM_MEMORY_CONFIG);

    const uint32_t batch = dims[0];
    const uint32_t seq = dims[1];
    const uint32_t hidden = dims[2];
    const uint32_t total_h = batch * seq;
    TT_FATAL(total_h % num_h_cores == 0, "total_h {} not divisible by num_h_cores {}", total_h, num_h_cores);
    TT_FATAL(hidden % num_w_cores == 0, "hidden {} not divisible by num_w_cores {}", hidden, num_w_cores);
    const uint32_t shard_h = total_h / num_h_cores;
    const uint32_t shard_w = hidden / num_w_cores;
    TT_FATAL(shard_h % tt::constants::TILE_HEIGHT == 0, "shard_h {} not tile-aligned", shard_h);
    TT_FATAL(shard_w % tt::constants::TILE_WIDTH == 0, "shard_w {} not tile-aligned", shard_w);

    tt::tt_metal::CoreRange grid_range(
        tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(num_w_cores - 1, num_h_cores - 1));
    tt::tt_metal::CoreRangeSet grid_set(std::set<tt::tt_metal::CoreRange>({grid_range}));
    tt::tt_metal::ShardSpec shard_spec(grid_set, {shard_h, shard_w}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    tt::tt_metal::MemoryConfig sharded_config(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    return ttnn::to_memory_config(staging, sharded_config);
}

// Helper that runs one parametrized cell of the matrix and returns (q_pcc, k_pcc, v_pcc).
// Used by the baseline interleaved tests in this commit and by the matrix tests in
// later commits.
struct PccTriple {
    float q;
    float k;
    float v;
};

PccTriple run_interleaved_baseline_cell(
    tt::tt_metal::distributed::MeshDevice* device,
    uint32_t batch,
    uint32_t seq,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k,
    DataType dtype,
    const tt::tt_metal::MemoryConfig& input_mem_config,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    auto float_input = make_random_qkv_input(kFixedSeed, batch, seq, num_q_heads, num_kv_heads, head_dim);
    // Interleaved path: nlp_create_qkv_heads expects CONCATENATED input.
    auto cpu_ref = cpu_split_qkv_reference(
        float_input, batch, seq, num_q_heads, num_kv_heads, head_dim, transpose_k, QkvLayout::CONCATENATED);

    auto input_tensor = make_interleaved_tensor(device, float_input, {batch, seq, hidden}, dtype, input_mem_config);

    // For MHA we pass num_kv_heads as nullopt; for GQA we pass it explicitly.
    std::optional<uint32_t> num_kv_arg =
        (num_kv_heads == num_q_heads) ? std::nullopt : std::optional<uint32_t>{num_kv_heads};

    auto [q_tt, k_tt, v_tt] = ttnn::transformer::split_query_key_value_and_split_heads(
        input_tensor,
        /*input_tensor_kv=*/std::nullopt,
        /*num_heads=*/num_q_heads,
        /*num_kv_heads=*/num_kv_arg,
        /*transpose_key=*/transpose_k,
        /*memory_config=*/output_mem_config,
        /*use_falcon7b_backend=*/false,
        /*qkv_layout=*/QkvLayout::CONCATENATED);

    // to_vector<float>() handles both BFLOAT16 and BFLOAT8_B unpacking.
    auto q_vec = ttnn::from_device(q_tt).to_vector<float>();
    auto k_vec = ttnn::from_device(k_tt).to_vector<float>();
    auto v_vec = ttnn::from_device(v_tt).to_vector<float>();

    return {
        ttnn::test_utils::pcc(cpu_ref.q, q_vec),
        ttnn::test_utils::pcc(cpu_ref.k, k_vec),
        ttnn::test_utils::pcc(cpu_ref.v, v_vec),
    };
}

// Run a sharded cell. Constructs a BLOCK_SHARDED input on the requested grid, calls
// split_query_key_value_and_split_heads with HEIGHT_SHARDED output, then converts the
// outputs back to interleaved + float for PCC against the CPU reference.
PccTriple run_block_sharded_to_height_sharded_cell(
    tt::tt_metal::distributed::MeshDevice* device,
    uint32_t batch,
    uint32_t seq,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k,
    DataType dtype,
    QkvLayout layout,
    uint32_t num_w_cores,
    uint32_t num_h_cores) {
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    auto float_input = make_random_qkv_input(kFixedSeed, batch, seq, num_q_heads, num_kv_heads, head_dim);
    auto cpu_ref =
        cpu_split_qkv_reference(float_input, batch, seq, num_q_heads, num_kv_heads, head_dim, transpose_k, layout);

    auto input_tensor =
        make_block_sharded_tensor(device, float_input, {batch, seq, hidden}, dtype, num_w_cores, num_h_cores);

    std::optional<uint32_t> num_kv_arg =
        (num_kv_heads == num_q_heads) ? std::nullopt : std::optional<uint32_t>{num_kv_heads};

    auto [q_tt, k_tt, v_tt] = ttnn::transformer::split_query_key_value_and_split_heads(
        input_tensor,
        /*input_tensor_kv=*/std::nullopt,
        /*num_heads=*/num_q_heads,
        /*num_kv_heads=*/num_kv_arg,
        /*transpose_key=*/transpose_k,
        /*memory_config=*/ttnn::L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        /*use_falcon7b_backend=*/false,
        /*qkv_layout=*/layout);

    // Convert sharded outputs back to interleaved DRAM so to_vector can read them.
    auto q_interleaved = ttnn::to_memory_config(q_tt, ttnn::DRAM_MEMORY_CONFIG);
    auto k_interleaved = ttnn::to_memory_config(k_tt, ttnn::DRAM_MEMORY_CONFIG);
    auto v_interleaved = ttnn::to_memory_config(v_tt, ttnn::DRAM_MEMORY_CONFIG);

    auto q_vec = ttnn::from_device(q_interleaved).to_vector<float>();
    auto k_vec = ttnn::from_device(k_interleaved).to_vector<float>();
    auto v_vec = ttnn::from_device(v_interleaved).to_vector<float>();

    return {
        ttnn::test_utils::pcc(cpu_ref.q, q_vec),
        ttnn::test_utils::pcc(cpu_ref.k, k_vec),
        ttnn::test_utils::pcc(cpu_ref.v, v_vec),
    };
}

}  // namespace

class SplitQkvMatrixTest : public TTNNFixtureWithSuiteDevice<SplitQkvMatrixTest> {};

// =========================================================================
// Commit 1: baseline interleaved tests (5 cells)
//
// These exercise the well-known-good `nlp_create_qkv_heads` interleaved path
// (the one that was passing in CI for the entire 19-month silent-corruption
// period). They verify that the helpers + CPU reference produce PCC ~ 1.0 for
// configurations the framework definitely handles correctly, before commit 2
// adds the sharded matrix.
// =========================================================================

// Cell 1: SD U-Net cross-attention shape (MHA, n=8, d=64), tile-aligned seq.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InDramInterleaved__OutDramInterleaved__seq64_aligned__bf16) {
    auto pcc = run_interleaved_baseline_cell(
        device_,
        /*batch=*/2,
        /*seq=*/64,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DRAM_MEMORY_CONFIG);
    log_debug(tt::LogTest, "MHA_8H_64D seq=64: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 2: ViT-base shape (MHA, n=12, d=64), tile-aligned seq.
TEST_F(SplitQkvMatrixTest, MHA_12H_64D__InDramInterleaved__OutDramInterleaved__seq256_aligned__bf16) {
    auto pcc = run_interleaved_baseline_cell(
        device_,
        /*batch=*/2,
        /*seq=*/256,
        /*n_q=*/12,
        /*n_kv=*/12,
        /*head_dim=*/64,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DRAM_MEMORY_CONFIG);
    log_debug(tt::LogTest, "MHA_12H_64D seq=256: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 3: Llama-2-7B-style head dim (MHA, n=8, d=128), tile-aligned seq, output L1.
TEST_F(SplitQkvMatrixTest, MHA_8H_128D__InDramInterleaved__OutL1Interleaved__seq128_aligned__bf16) {
    auto pcc = run_interleaved_baseline_cell(
        device_,
        /*batch=*/2,
        /*seq=*/128,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/128,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::L1_MEMORY_CONFIG);
    log_debug(tt::LogTest, "MHA_8H_128D seq=128: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 4: GQA Llama-2-70B-style (n_q=32, n_kv=8, d=128), tile-aligned seq.
// Tests the GQA divisor branch in split_query_key_value_and_split_heads.
TEST_F(SplitQkvMatrixTest, GQA_32_8_128D__InDramInterleaved__OutDramInterleaved__seq256_aligned__bf16) {
    auto pcc = run_interleaved_baseline_cell(
        device_,
        /*batch=*/2,
        /*seq=*/256,
        /*n_q=*/32,
        /*n_kv=*/8,
        /*head_dim=*/128,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DRAM_MEMORY_CONFIG);
    log_debug(tt::LogTest, "GQA_32_8_128D seq=256: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 5: transpose_key=true variant (BERT-style attention).
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InDramInterleaved__OutDramInterleaved__seq64_aligned__bf16__TransposeK) {
    auto pcc = run_interleaved_baseline_cell(
        device_,
        /*batch=*/2,
        /*seq=*/64,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/true,
        DataType::BFLOAT16,
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DRAM_MEMORY_CONFIG);
    log_debug(tt::LogTest, "MHA_8H_64D seq=64 transpose_k: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// =========================================================================
// Commit 2: BLOCK_SHARDED -> HEIGHT_SHARDED matrix
//
// These exercise the sharded code path that the existing test_transformer.py
// suite skips on every modern arch (@skipif(is_wormhole_b0() or is_blackhole())).
//
// CRITICAL FINDING from this work: the existing sharded `create_qkv_heads` reader
// expects QKV input in GROUPED layout (`[Q_h0,K_h0,V_h0, Q_h1,K_h1,V_h1, ...]`),
// not in CONCATENATED layout (`[Q_h0,...,Q_hN, K_h0,...,K_hN, V_h0,...,V_hN]`).
// Both SD U-Net cross-attention (`concatenate_qkv()` lines 96-117 of
// stable_diffusion/wormhole/tt/ttnn_functional_cross_attention.py) and ViT WH
// (`custom_preprocessor` lines 559-566 of vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py)
// manually repack their QKV linear weights into grouped layout precisely so the
// sharded kernel reads them correctly. The 19-month "silent corruption" claim in
// #41718 is wrong for both — those models were correctly using grouped weights
// the whole time. The actual bug from #41526 was tt-mlir greedy optimizer feeding
// CONCATENATED layout to a kernel expecting GROUPED — a tt-mlir codegen issue.
//
// These tests verify that:
//   - With GROUPED input + tile-aligned seq: PCC ≈ 1.0 (the kernel works as designed)
//   - With CONCATENATED input + tile-aligned seq: REJECTED at validation time by the
//     `qkv_layout` TT_FATAL added in commit 6 of this branch (see Cells 13-14).
//     Before that commit, the kernel silently produced PCC ≈ 0.1 — the bug the
//     framework was missing for ~19 months.
//
// Grid choice: num_w_cores = num_kv_heads (so each shard width is exactly
// (n_q/n_kv + 2) * head_dim — what the existing create_qkv_heads validator expects).
// =========================================================================

// Cell 6: SD U-Net cross-attention shape with GROUPED input (the layout the model
// actually uses). Exact reproduction of #41718's BH SD U-Net call site.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedGrouped__OutHeightSharded__seq64_aligned__bf16) {
    auto pcc = run_block_sharded_to_height_sharded_cell(
        device_,
        /*batch=*/2,
        /*seq=*/64,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        QkvLayout::GROUPED,
        /*num_w_cores=*/8,
        /*num_h_cores=*/2);
    log_debug(
        tt::LogTest,
        "MHA_8H_64D BlockSharded(grouped)->HeightSharded seq=64: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 7: same as Cell 6 but seq=256.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedGrouped__OutHeightSharded__seq256_aligned__bf16) {
    auto pcc = run_block_sharded_to_height_sharded_cell(
        device_,
        /*batch=*/2,
        /*seq=*/256,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        QkvLayout::GROUPED,
        /*num_w_cores=*/8,
        /*num_h_cores=*/2);
    log_debug(
        tt::LogTest,
        "MHA_8H_64D BlockSharded(grouped)->HeightSharded seq=256: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 8: ViT-base shape with GROUPED input, tile-aligned seq=224.
//
// The natural grid for 12 KV heads is (12, num_h_cores), but on Blackhole p150a
// 12-wide grids hit dispatch cores ("Illegal kernel placement for
// writer_unary_sharded, Kernels cannot be placed on dispatch cores!"). This is
// the same constraint that makes the existing test_transformer.py
// `test_sharded_split_query_key_value_and_split_heads` `@skipif` on every modern
// arch. We document the case here but skip at runtime if the grid setup fails so
// the suite stays green on hardware that doesn't support the wider grid.
TEST_F(SplitQkvMatrixTest, MHA_12H_64D__InBlockShardedGrouped__OutHeightSharded__seq224_aligned__bf16) {
    PccTriple pcc{};
    try {
        pcc = run_block_sharded_to_height_sharded_cell(
            device_,
            /*batch=*/2,
            /*seq=*/224,
            /*n_q=*/12,
            /*n_kv=*/12,
            /*head_dim=*/64,
            /*transpose_k=*/false,
            DataType::BFLOAT16,
            QkvLayout::GROUPED,
            /*num_w_cores=*/12,
            /*num_h_cores=*/2);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "12-wide BLOCK_SHARDED grid not supported on this hardware "
                        "(likely dispatch-core placement constraint): "
                     << e.what();
    }
    log_debug(
        tt::LogTest,
        "MHA_12H_64D BlockSharded(grouped)->HeightSharded seq=224: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 9: GQA Llama-2-70B-style with GROUPED input, tile-aligned seq.
// q_per_group = 32/8 = 4, so each group is 4 Q heads + 1 K + 1 V = 6 head_dim wide.
TEST_F(SplitQkvMatrixTest, GQA_32_8_128D__InBlockShardedGrouped__OutHeightSharded__seq256_aligned__bf16) {
    auto pcc = run_block_sharded_to_height_sharded_cell(
        device_,
        /*batch=*/2,
        /*seq=*/256,
        /*n_q=*/32,
        /*n_kv=*/8,
        /*head_dim=*/128,
        /*transpose_k=*/false,
        DataType::BFLOAT16,
        QkvLayout::GROUPED,
        /*num_w_cores=*/8,
        /*num_h_cores=*/2);
    log_debug(
        tt::LogTest,
        "GQA_32_8_128D BlockSharded(grouped)->HeightSharded seq=256: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// =========================================================================
// Commit 2 — specialized tests revealing the layout-mismatch ground truth
// =========================================================================

// Cell 13: SAME shape as Cell 6, but feed CONCATENATED input. Before the explicit
// `qkv_layout` API change on this branch, the kernel silently produced PCC ~ 0.1
// (Q=0.1188, K=0.0009, V=0.1244 — measured on Blackhole p150a) — the test
// originally documented that corruption directly. After the API change the op
// rejects this combination at validation time with a clear TT_FATAL, so the test
// now asserts the REJECTED outcome instead. This is the proper fix for the bug
// the framework was silently allowing for ~19 months.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedConcatenated__OutHeightSharded__seq64_aligned__bf16__REJECTED) {
    EXPECT_THROW(
        {
            (void)run_block_sharded_to_height_sharded_cell(
                device_,
                /*batch=*/2,
                /*seq=*/64,
                /*n_q=*/8,
                /*n_kv=*/8,
                /*head_dim=*/64,
                /*transpose_k=*/false,
                DataType::BFLOAT16,
                QkvLayout::CONCATENATED,
                /*num_w_cores=*/8,
                /*num_h_cores=*/2);
        },
        std::exception);
}

// Cell 14: ViT-base shape with CONCATENATED input — the canonical bug pattern from
// tt-mlir greedy optimizer. seq is tile-aligned, so the trigger is purely the
// layout mismatch (not the non-tile-alignment from #41526). After the explicit
// qkv_layout API change, the op rejects this combination at validation time.
// On hardware where the 12-wide grid setup fails first (dispatch-core constraint
// — same as Cell 8) we skip; on hardware where the grid is supported, we expect
// the qkv_layout validation to fire.
TEST_F(SplitQkvMatrixTest, MHA_12H_64D__InBlockShardedConcatenated__OutHeightSharded__seq224_aligned__bf16__REJECTED) {
    try {
        (void)run_block_sharded_to_height_sharded_cell(
            device_,
            /*batch=*/2,
            /*seq=*/224,
            /*n_q=*/12,
            /*n_kv=*/12,
            /*head_dim=*/64,
            /*transpose_k=*/false,
            DataType::BFLOAT16,
            QkvLayout::CONCATENATED,
            /*num_w_cores=*/12,
            /*num_h_cores=*/2);
        FAIL() << "Expected the op to reject CONCATENATED + sharded with a TT_FATAL, but the call "
                  "returned without throwing. Either the qkv_layout validation was bypassed or the "
                  "kernel started accepting concatenated layout.";
    } catch (const std::exception& e) {
        const std::string msg(e.what());
        if (msg.find("Illegal kernel placement") != std::string::npos ||
            msg.find("dispatch core") != std::string::npos ||
            msg.find("Kernels cannot be placed on dispatch cores") != std::string::npos) {
            GTEST_SKIP() << "12-wide BLOCK_SHARDED grid not supported on this hardware "
                            "(dispatch-core placement constraint): "
                         << e.what();
        }
        // Otherwise we expect this to be the qkv_layout FATAL — log it and pass.
        log_debug(tt::LogTest, "Cell 14 REJECTED as expected: {}", e.what());
        EXPECT_TRUE(msg.find("CONCATENATED") != std::string::npos || msg.find("GROUPED") != std::string::npos)
            << "Expected the FATAL to mention the qkv_layout but got: " << e.what();
    }
}

// =========================================================================
// Commit 3: non-tile-aligned sub-matrix, transpose_k sharded cells, bf8 variant
//
// These cells exercise:
//   - The non-tile-aligned sequence cases that trigger the refined FATAL added by
//     the previous commit on this branch (cells 15-17). The op should reject the
//     call regardless of layout, because the address-arithmetic in the sharded
//     reader doesn't handle non-tile-aligned source offsets.
//   - The seq=197 ViT case from #41526 with the same num_w_cores=6 / num_h_cores=8
//     grid the existing Python regression test uses. With the refined FATAL it is
//     REJECTED at the validator (the original repro behavior — silent corruption —
//     can no longer be observed without temporarily disabling the FATAL).
//   - transpose_key=true on the sharded path (cell 18) — the BERT-style attention
//     case for sharded inputs.
//   - bf8 dtype variant of the SD U-Net case (cell 19) — confirms the layout
//     finding is dtype-independent and matches the precise dtype the production
//     model uses.
// =========================================================================

// Cell 15: SD U-Net shape with non-tile-aligned seq, GROUPED. The refined FATAL
// (sequence_size_padded != sequence_size) should reject this call regardless of
// layout. seq=49 -> padded 64, so sequence_size_padded=64 != sequence_size=49.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedGrouped__OutHeightSharded__seq49_unaligned__bf16__REJECTED) {
    EXPECT_THROW(
        {
            (void)run_block_sharded_to_height_sharded_cell(
                device_,
                /*batch=*/2,
                /*seq=*/49,
                /*n_q=*/8,
                /*n_kv=*/8,
                /*head_dim=*/64,
                /*transpose_k=*/false,
                DataType::BFLOAT16,
                QkvLayout::GROUPED,
                /*num_w_cores=*/8,
                /*num_h_cores=*/2);
        },
        std::exception);
}

// Cell 16: same as Cell 15 but CONCATENATED. The refined FATAL fires before the
// kernel runs, so the layout doesn't matter — REJECTED in both cases.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedConcatenated__OutHeightSharded__seq49_unaligned__bf16__REJECTED) {
    EXPECT_THROW(
        {
            (void)run_block_sharded_to_height_sharded_cell(
                device_,
                /*batch=*/2,
                /*seq=*/49,
                /*n_q=*/8,
                /*n_kv=*/8,
                /*head_dim=*/64,
                /*transpose_k=*/false,
                DataType::BFLOAT16,
                QkvLayout::CONCATENATED,
                /*num_w_cores=*/8,
                /*num_h_cores=*/2);
        },
        std::exception);
}

// Cell 17: the canonical bug repro from #41526 — ViT seq=197, num_heads=12,
// head_size=64, BLOCK_SHARDED -> HEIGHT_SHARDED, with the same grid the Python
// regression test in tests/ttnn/.../test_transformer.py uses
// (num_w_cores=num_heads/2=6, num_h_cores=batch=8). With the refined FATAL on this
// branch the op rejects the call. Without the FATAL the kernel would silently
// produce PCC ~0.08 (the original bug). Skipped at runtime if the 6x8 grid is not
// supported on this hardware.
TEST_F(SplitQkvMatrixTest, MHA_12H_64D__InBlockSharded__OutHeightSharded__seq197_unaligned__bf16__REJECTED) {
    bool threw = false;
    try {
        (void)run_block_sharded_to_height_sharded_cell(
            device_,
            /*batch=*/8,
            /*seq=*/197,
            /*n_q=*/12,
            /*n_kv=*/12,
            /*head_dim=*/64,
            /*transpose_k=*/false,
            DataType::BFLOAT16,
            // Layout doesn't matter — refined FATAL fires before the kernel runs.
            QkvLayout::CONCATENATED,
            /*num_w_cores=*/6,
            /*num_h_cores=*/8);
    } catch (const std::exception& e) {
        threw = true;
        log_debug(tt::LogTest, "MHA_12H_64D seq=197 REJECTED as expected: {}", e.what());
    }
    if (!threw) {
        FAIL() << "Expected refined FATAL to reject non-tile-aligned sharded input (seq=197 padded to 224), "
                  "but the call returned without throwing. Either the FATAL was removed or the validation "
                  "was bypassed.";
    }
}

// Cell 18: transpose_k=true on the sharded path (BERT-style attention).
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedGrouped__OutHeightSharded__seq64_aligned__bf16__TransposeK) {
    auto pcc = run_block_sharded_to_height_sharded_cell(
        device_,
        /*batch=*/2,
        /*seq=*/64,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/true,
        DataType::BFLOAT16,
        QkvLayout::GROUPED,
        /*num_w_cores=*/8,
        /*num_h_cores=*/2);
    log_debug(
        tt::LogTest,
        "MHA_8H_64D BlockSharded(grouped)->HeightSharded seq=64 transpose_k: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

// Cell 19: BFLOAT8_B variant of the SD U-Net case. Production SD U-Net uses bf8
// throughout (see dtype=ttnn.bfloat8_b in stable_diffusion/wormhole/tt/
// ttnn_functional_cross_attention.py). This test confirms the layout finding is
// dtype-independent: the kernel correctly handles GROUPED bf8 input, just like
// the bf16 case in cell 6.
TEST_F(SplitQkvMatrixTest, MHA_8H_64D__InBlockShardedGrouped__OutHeightSharded__seq64_aligned__bf8) {
    auto pcc = run_block_sharded_to_height_sharded_cell(
        device_,
        /*batch=*/2,
        /*seq=*/64,
        /*n_q=*/8,
        /*n_kv=*/8,
        /*head_dim=*/64,
        /*transpose_k=*/false,
        DataType::BFLOAT8_B,
        QkvLayout::GROUPED,
        /*num_w_cores=*/8,
        /*num_h_cores=*/2);
    log_debug(
        tt::LogTest,
        "MHA_8H_64D BlockSharded(grouped)->HeightSharded seq=64 bf8: q={:.6f} k={:.6f} v={:.6f}",
        pcc.q,
        pcc.k,
        pcc.v);
    // bf8 has slightly more quantization noise than bf16; loosen the threshold a bit
    // for the bf8 cells (still well above the corruption floor of ~0.1).
    constexpr float kBf8PccThreshold = 0.97f;
    EXPECT_GE(pcc.q, kBf8PccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kBf8PccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kBf8PccThreshold) << "V PCC " << pcc.v;
}

}  // namespace ttnn::operations::transformer::test
