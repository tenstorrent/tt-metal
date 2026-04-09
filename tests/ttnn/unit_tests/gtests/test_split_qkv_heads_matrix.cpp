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

// CPU reference: split a flat [B, S, (n_q+2*n_kv)*D] tensor in concatenated layout
// (`[Q_h0..Q_h_nq-1, K_h0..K_h_nkv-1, V_h0..V_h_nkv-1]` per (batch, seq) row) into:
//   Q [B, n_q,  S, D]
//   K [B, n_kv, S, D]   (or [B, n_kv, D, S] if transpose_k)
//   V [B, n_kv, S, D]
// All outputs as flat float vectors in row-major order.
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
    bool transpose_k) {
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    const uint32_t q_section = num_q_heads * head_dim;
    const uint32_t kv_section = num_kv_heads * head_dim;

    CpuQkvSplitResult result;
    result.q.resize(static_cast<size_t>(batch) * num_q_heads * seq * head_dim);
    result.k.resize(static_cast<size_t>(batch) * num_kv_heads * seq * head_dim);
    result.v.resize(static_cast<size_t>(batch) * num_kv_heads * seq * head_dim);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t s = 0; s < seq; ++s) {
            const size_t in_base = (static_cast<size_t>(b) * seq + s) * hidden;
            // Q
            for (uint32_t h = 0; h < num_q_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const float v_in = input[in_base + h * head_dim + d];
                    // out [B, n_q, S, D]: index = ((b*n_q + h)*S + s)*D + d
                    result.q[((static_cast<size_t>(b) * num_q_heads + h) * seq + s) * head_dim + d] = v_in;
                }
            }
            // K
            for (uint32_t h = 0; h < num_kv_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const float v_in = input[in_base + q_section + h * head_dim + d];
                    if (transpose_k) {
                        // out [B, n_kv, D, S]: index = ((b*n_kv + h)*D + d)*S + s
                        result.k[((static_cast<size_t>(b) * num_kv_heads + h) * head_dim + d) * seq + s] = v_in;
                    } else {
                        // out [B, n_kv, S, D]
                        result.k[((static_cast<size_t>(b) * num_kv_heads + h) * seq + s) * head_dim + d] = v_in;
                    }
                }
            }
            // V
            for (uint32_t h = 0; h < num_kv_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const float v_in = input[in_base + q_section + kv_section + h * head_dim + d];
                    result.v[((static_cast<size_t>(b) * num_kv_heads + h) * seq + s) * head_dim + d] = v_in;
                }
            }
        }
    }
    return result;
}

// Convert a vector of bfloat16 to float (lossless: bfloat16 fits exactly in float32).
std::vector<float> bf16_vec_to_float(const std::vector<::bfloat16>& src) {
    std::vector<float> out;
    out.reserve(src.size());
    for (const auto& v : src) {
        out.push_back(static_cast<float>(v));
    }
    return out;
}

// Build an interleaved BF16 tensor on device from a flat float vector. Caller picks
// the memory config (typically ttnn::DRAM_MEMORY_CONFIG or ttnn::L1_MEMORY_CONFIG).
tt::tt_metal::Tensor make_interleaved_bf16_tensor(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::vector<float>& data,
    const std::array<uint32_t, 3>& dims,
    const tt::tt_metal::MemoryConfig& mem_config) {
    std::vector<::bfloat16> bf16_data;
    bf16_data.reserve(data.size());
    for (float v : data) {
        bf16_data.push_back(::bfloat16(v));
    }

    tt::tt_metal::TensorSpec spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
    return tt::tt_metal::Tensor::from_vector(std::move(bf16_data), spec).to_device(device);
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
    const tt::tt_metal::MemoryConfig& input_mem_config,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    const uint32_t hidden = fused_hidden_dim(num_q_heads, num_kv_heads, head_dim);
    auto float_input = make_random_qkv_input(kFixedSeed, batch, seq, num_q_heads, num_kv_heads, head_dim);
    auto cpu_ref = cpu_split_qkv_reference(float_input, batch, seq, num_q_heads, num_kv_heads, head_dim, transpose_k);

    auto input_tensor = make_interleaved_bf16_tensor(device, float_input, {batch, seq, hidden}, input_mem_config);

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
        /*use_falcon7b_backend=*/false);

    auto q_vec = bf16_vec_to_float(ttnn::from_device(q_tt).to_vector<::bfloat16>());
    auto k_vec = bf16_vec_to_float(ttnn::from_device(k_tt).to_vector<::bfloat16>());
    auto v_vec = bf16_vec_to_float(ttnn::from_device(v_tt).to_vector<::bfloat16>());

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
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DRAM_MEMORY_CONFIG);
    log_debug(tt::LogTest, "MHA_8H_64D seq=64 transpose_k: q={:.6f} k={:.6f} v={:.6f}", pcc.q, pcc.k, pcc.v);
    EXPECT_GE(pcc.q, kDefaultPccThreshold) << "Q PCC " << pcc.q;
    EXPECT_GE(pcc.k, kDefaultPccThreshold) << "K PCC " << pcc.k;
    EXPECT_GE(pcc.v, kDefaultPccThreshold) << "V PCC " << pcc.v;
}

}  // namespace ttnn::operations::transformer::test
