// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

/**
 * Comprehensive test suite for nlp_create_qkv_heads operation
 *
 * This test was created to verify the fix for a bug where integer division
 * caused incorrect output shapes when head_dim < TILE_WIDTH (32).
 *
 * BUG: q_num_tiles = num_heads * (head_dim / TILE_WIDTH) returns 0 when head_dim < 32
 * FIX: q_num_tiles = (num_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH
 */

bool test_nlp_create_qkv_heads(
    distributed::MeshDevice* device,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    bool transpose_k_heads,
    std::optional<ttnn::MemoryConfig> memory_config,
    const std::string& test_name) {

    log_info(LogTest, "Running test: {}", test_name);
    log_info(LogTest, "  batch={}, seq_len={}, head_dim={}, num_q_heads={}, num_kv_heads={}, transpose_k={}",
             batch_size, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads);

    uint32_t q_embedding_dim = num_q_heads * head_dim;
    uint32_t kv_embedding_dim = num_kv_heads * head_dim;
    uint32_t qkv_width = q_embedding_dim + 2 * kv_embedding_dim;  // Q + K + V concatenated

    // Create fused QKV input tensor: [batch, 1, seq_len, qkv_width]
    ttnn::Shape input_shape({batch_size, 1, seq_len, qkv_width});
    Tensor input_tensor = ttnn::random::random(input_shape)
        .to_layout(Layout::TILE)
        .to_device(device);

    // Call nlp_create_qkv_heads
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        input_tensor,
        /*input_tensor_kv=*/std::nullopt,
        /*num_heads=*/num_q_heads,
        /*num_kv_heads=*/num_kv_heads,
        /*transpose_k_heads=*/transpose_k_heads,
        /*memory_config=*/memory_config);

    // Check output shapes
    auto q_shape = q.padded_shape();
    auto k_shape = k.padded_shape();
    auto v_shape = v.padded_shape();

    // Expected shapes
    // Q: [batch, num_q_heads, seq_len, head_dim]
    // K: [batch, num_kv_heads, seq_len, head_dim] or [batch, num_kv_heads, head_dim, seq_len] if transposed
    // V: [batch, num_kv_heads, seq_len, head_dim]

    log_info(LogTest, "  Q expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
             batch_size, num_q_heads, seq_len, head_dim,
             q_shape[0], q_shape[1], q_shape[2], q_shape[3]);

    if (transpose_k_heads) {
        log_info(LogTest, "  K expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}] (transposed)",
                 batch_size, num_kv_heads, head_dim, seq_len,
                 k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    } else {
        log_info(LogTest, "  K expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
                 batch_size, num_kv_heads, seq_len, head_dim,
                 k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    }

    log_info(LogTest, "  V expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
             batch_size, num_kv_heads, seq_len, head_dim,
             v_shape[0], v_shape[1], v_shape[2], v_shape[3]);

    // Verify Q shape
    bool q_shape_correct = (q_shape[0] == batch_size &&
                            q_shape[1] == num_q_heads &&
                            q_shape[2] == seq_len &&
                            q_shape[3] == head_dim);

    // Verify K shape (depends on transpose_k_heads)
    bool k_shape_correct;
    if (transpose_k_heads) {
        k_shape_correct = (k_shape[0] == batch_size &&
                          k_shape[1] == num_kv_heads &&
                          k_shape[2] == head_dim &&
                          k_shape[3] == seq_len);
    } else {
        k_shape_correct = (k_shape[0] == batch_size &&
                          k_shape[1] == num_kv_heads &&
                          k_shape[2] == seq_len &&
                          k_shape[3] == head_dim);
    }

    // Verify V shape
    bool v_shape_correct = (v_shape[0] == batch_size &&
                            v_shape[1] == num_kv_heads &&
                            v_shape[2] == seq_len &&
                            v_shape[3] == head_dim);

    if (!q_shape_correct || !k_shape_correct || !v_shape_correct) {
        if (!q_shape_correct) {
            log_error(LogTest, "  Q shape mismatch");
        }
        if (!k_shape_correct) {
            log_error(LogTest, "  K shape mismatch");
        }
        if (!v_shape_correct) {
            log_error(LogTest, "  V shape mismatch");
        }
        return false;
    }

    log_info(LogTest, "  Test passed");
    return true;
}

bool test_nlp_create_qkv_heads_with_separate_kv(
    distributed::MeshDevice* device,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    const std::string& test_name) {

    log_info(LogTest, "Running test: {}", test_name);
    log_info(LogTest, "  batch={}, seq_len={}, head_dim={}, num_q_heads={}, num_kv_heads={} (separate KV tensor)",
             batch_size, seq_len, head_dim, num_q_heads, num_kv_heads);

    uint32_t q_width = num_q_heads * head_dim;
    uint32_t kv_width = 2 * num_kv_heads * head_dim;  // K + V concatenated

    // Create separate Q and KV tensors
    ttnn::Shape q_input_shape({batch_size, 1, seq_len, q_width});
    ttnn::Shape kv_input_shape({batch_size, 1, seq_len, kv_width});

    Tensor q_tensor = ttnn::random::random(q_input_shape)
        .to_layout(Layout::TILE)
        .to_device(device);

    Tensor kv_tensor = ttnn::random::random(kv_input_shape)
        .to_layout(Layout::TILE)
        .to_device(device);

    // Call nlp_create_qkv_heads with separate KV tensor
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        q_tensor,
        /*input_tensor_kv=*/kv_tensor,
        /*num_heads=*/num_q_heads,
        /*num_kv_heads=*/num_kv_heads,
        /*transpose_k_heads=*/false,
        /*memory_config=*/std::nullopt);

    // Check output shapes
    auto q_out_shape = q.padded_shape();
    auto k_out_shape = k.padded_shape();
    auto v_out_shape = v.padded_shape();

    log_info(LogTest, "  Q expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
             batch_size, num_q_heads, seq_len, head_dim,
             q_out_shape[0], q_out_shape[1], q_out_shape[2], q_out_shape[3]);
    log_info(LogTest, "  K expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
             batch_size, num_kv_heads, seq_len, head_dim,
             k_out_shape[0], k_out_shape[1], k_out_shape[2], k_out_shape[3]);
    log_info(LogTest, "  V expected: [{}, {}, {}, {}], got: [{}, {}, {}, {}]",
             batch_size, num_kv_heads, seq_len, head_dim,
             v_out_shape[0], v_out_shape[1], v_out_shape[2], v_out_shape[3]);

    // Verify shapes
    bool shapes_correct =
        (q_out_shape[0] == batch_size && q_out_shape[1] == num_q_heads &&
         q_out_shape[2] == seq_len && q_out_shape[3] == head_dim) &&
        (k_out_shape[0] == batch_size && k_out_shape[1] == num_kv_heads &&
         k_out_shape[2] == seq_len && k_out_shape[3] == head_dim) &&
        (v_out_shape[0] == batch_size && v_out_shape[1] == num_kv_heads &&
         v_out_shape[2] == seq_len && v_out_shape[3] == head_dim);

    if (!shapes_correct) {
        log_error(LogTest, "  Shape mismatch with separate KV tensor");
        return false;
    }

    log_info(LogTest, "  Test passed");
    return true;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device_owner = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto device = device_owner.get();

        ////////////////////////////////////////////////////////////////////////////
        //                Bug Regression Tests (head_dim < TILE_WIDTH)
        ////////////////////////////////////////////////////////////////////////////

        // Critical: head_dim=16 triggers the original bug
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 16, 4, 4, false, std::nullopt,
            "bug_regression_head_dim_16");

        // Critical: head_dim=8 (even smaller)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 8, 8, 8, false, std::nullopt,
            "bug_regression_head_dim_8");

        // Different configurations with head_dim < 32
        pass &= test_nlp_create_qkv_heads(
            device, 2, 64, 16, 2, 2, false, std::nullopt,
            "bug_regression_batch2_head_dim_16");

        ////////////////////////////////////////////////////////////////////////////
        //                      K-Head Transpose Tests
        ////////////////////////////////////////////////////////////////////////////

        // Test transpose_k_heads=true with small head_dim
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 16, 4, 4, true, std::nullopt,
            "transpose_k_small_head_dim");

        // Test transpose_k_heads=true with normal head_dim
        pass &= test_nlp_create_qkv_heads(
            device, 1, 64, 64, 8, 8, true, std::nullopt,
            "transpose_k_normal_head_dim");

        // Test transpose_k_heads=true with large head_dim
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 128, 16, 16, true, std::nullopt,
            "transpose_k_large_head_dim");

        ////////////////////////////////////////////////////////////////////////////
        //            GQA (Grouped Query Attention) Tests
        ////////////////////////////////////////////////////////////////////////////

        // GQA with small head_dim (32 Q heads, 4 KV heads - 8x groups)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 16, 32, 4, false, std::nullopt,
            "gqa_head_dim_16_q32_kv4");

        // GQA typical case (16 Q heads, 2 KV heads - 8x groups)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 64, 16, 2, false, std::nullopt,
            "gqa_head_dim_64_q16_kv2");

        // GQA with single KV head (71 Q heads, 1 KV head - Falcon-like)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 64, 71, 1, false, std::nullopt,
            "gqa_falcon_q71_kv1");

        // GQA with small head_dim and transpose
        pass &= test_nlp_create_qkv_heads(
            device, 1, 64, 16, 8, 2, true, std::nullopt,
            "gqa_transpose_head_dim_16_q8_kv2");

        ////////////////////////////////////////////////////////////////////////////
        //                   Separate KV Tensor Tests
        ////////////////////////////////////////////////////////////////////////////

        // Separate KV with small head_dim
        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 32, 16, 8, 8,
            "separate_kv_head_dim_16");

        // Separate KV with normal head_dim
        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 64, 64, 16, 16,
            "separate_kv_head_dim_64");

        // Separate KV with GQA
        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 128, 64, 32, 4,
            "separate_kv_gqa_q32_kv4");

        ////////////////////////////////////////////////////////////////////////////
        //                  Memory Configuration Tests
        ////////////////////////////////////////////////////////////////////////////

        // Test with DRAM memory config
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 16, 4, 4, false, ttnn::DRAM_MEMORY_CONFIG,
            "memory_dram_head_dim_16");

        // Test with L1 memory config
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 64, 8, 8, false, ttnn::L1_MEMORY_CONFIG,
            "memory_l1_head_dim_64");

        ////////////////////////////////////////////////////////////////////////////
        //             Large head_dim Tests (Modern Transformers)
        ////////////////////////////////////////////////////////////////////////////

        // head_dim=128 (Llama-style)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 128, 32, 32, false, std::nullopt,
            "large_head_dim_128");

        // head_dim=128 with GQA (Llama 2/3 style: 32 Q, 4 KV)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 1024, 128, 32, 4, false, std::nullopt,
            "large_head_dim_128_gqa_llama");

        // head_dim=256 (very large)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 64, 256, 16, 16, false, std::nullopt,
            "large_head_dim_256");

        ////////////////////////////////////////////////////////////////////////////
        //                 Edge Cases and Stress Tests
        ////////////////////////////////////////////////////////////////////////////

        // Edge: head_dim=32 (exactly TILE_WIDTH)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 32, 4, 4, false, std::nullopt,
            "edge_head_dim_32");

        // Large batch size
        pass &= test_nlp_create_qkv_heads(
            device, 8, 128, 64, 16, 16, false, std::nullopt,
            "large_batch_8");

        // Long sequence
        pass &= test_nlp_create_qkv_heads(
            device, 1, 2048, 64, 16, 16, false, std::nullopt,
            "long_seq_2048");

        // Minimal configuration
        pass &= test_nlp_create_qkv_heads(
            device, 1, 32, 8, 1, 1, false, std::nullopt,
            "minimal_config");

        ////////////////////////////////////////////////////////////////////////////
        //          REGRESSION TESTS - Existing "Working" head_dims
        //   These MUST pass both BEFORE and AFTER the bug fix
        ////////////////////////////////////////////////////////////////////////////

        // head_dim=64 regression tests (most common, MUST NOT BREAK)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 256, 64, 16, 2, true, std::nullopt,
            "regression_head64_gqa_transpose");

        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 256, 64, 16, 2,
            "regression_head64_gqa_separate_kv");

        pass &= test_nlp_create_qkv_heads(
            device, 1, 256, 64, 16, 16, true, ttnn::L1_MEMORY_CONFIG,
            "regression_head64_transpose_l1");

        // head_dim=96 regression tests (from existing tests, MUST NOT BREAK)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 96, 8, 2, true, std::nullopt,
            "regression_head96_gqa_transpose");

        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 128, 96, 8, 2,
            "regression_head96_gqa_separate_kv");

        // head_dim=128 regression tests (Llama, MUST NOT BREAK)
        pass &= test_nlp_create_qkv_heads(
            device, 1, 512, 128, 32, 4, true, std::nullopt,
            "regression_head128_llama_transpose");

        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 512, 128, 32, 4,
            "regression_head128_llama_separate_kv");

        pass &= test_nlp_create_qkv_heads(
            device, 1, 512, 128, 32, 8, true, ttnn::DRAM_MEMORY_CONFIG,
            "regression_head128_gqa_transpose_dram");

        // Boundary case: head_dim=32 (CRITICAL - boundary between bug and no-bug)
        // This should work BOTH before and after the fix
        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 32, 8, 8, false, std::nullopt,
            "regression_boundary_head32_basic");

        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 32, 8, 8, true, std::nullopt,
            "regression_boundary_head32_transpose");

        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 1, 128, 32, 8, 2,
            "regression_boundary_head32_gqa_separate_kv");

        pass &= test_nlp_create_qkv_heads(
            device, 1, 128, 32, 16, 2, true, ttnn::L1_MEMORY_CONFIG,
            "regression_boundary_head32_all_features");

        // Stress test: All features combined with safe head_dims
        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 2, 256, 64, 16, 2,
            "regression_stress_head64_batch2_gqa");

        pass &= test_nlp_create_qkv_heads(
            device, 4, 128, 96, 8, 2, true, ttnn::DRAM_MEMORY_CONFIG,
            "regression_stress_head96_batch4_transpose");

        pass &= test_nlp_create_qkv_heads_with_separate_kv(
            device, 2, 512, 128, 32, 4,
            "regression_stress_head128_llama_batch2");

    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
