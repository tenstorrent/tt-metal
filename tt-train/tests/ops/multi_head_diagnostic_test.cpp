// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/multi_head_utils.hpp"

#include <gtest/gtest.h>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"

namespace ttml::ops::tests {

class MultiHeadDiagnosticTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// Test 1: Direct nlp_create_qkv_heads with concatenated input
TEST_F(MultiHeadDiagnosticTest, DirectNlpCreateQKVHeads_ConcatenatedInput) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 64;
    uint32_t num_heads = 4;
    uint32_t expected_head_dim = embedding_dim / num_heads;  // 16

    // Create concatenated QKV tensor [B, 1, S, 3*E]
    std::vector<float> qkv_data(batch * seq_len * embedding_dim * 3);
    for (size_t i = 0; i < qkv_data.size(); ++i) {
        qkv_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape{batch, 1, seq_len, embedding_dim * 3}, device);

    std::cout << "\nDIAGNOSTIC: Testing nlp_create_qkv_heads with concatenated input\n";
    std::cout << "  Input shape: " << qkv_tensor.logical_shape() << "\n";
    std::cout << "  Expected head_dim calculation: " << embedding_dim * 3 << " / (" << num_heads << " + 2*" << num_heads << ") = " << expected_head_dim << "\n";

    // Call the framework function directly
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qkv_tensor,
        std::nullopt,  // No separate KV tensor
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    auto q_shape = q.logical_shape();
    auto k_shape = k.logical_shape();
    auto v_shape = v.logical_shape();

    std::cout << "  Actual Q shape: " << q_shape << "\n";
    std::cout << "  Actual K shape: " << k_shape << "\n";
    std::cout << "  Actual V shape: " << v_shape << "\n";
    std::cout << "  Actual head_dim: " << q_shape[3] << "\n";

    if (q_shape[3] != expected_head_dim) {
        std::cout << "  ERROR: Head dimension mismatch! Expected " << expected_head_dim << ", got " << q_shape[3] << "\n";
        std::cout << "  FRAMEWORK BUG: nlp_create_qkv_heads incorrectly calculates head_dim with concatenated input\n";
    }

    EXPECT_EQ(q_shape[3], expected_head_dim) << "Framework bug in head dimension calculation";
}

// Test 2: Direct nlp_create_qkv_heads with separate Q and KV inputs
TEST_F(MultiHeadDiagnosticTest, DirectNlpCreateQKVHeads_SeparateInputs) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 64;
    uint32_t num_heads = 4;
    uint32_t expected_head_dim = embedding_dim / num_heads;  // 16

    // Create separate Q and KV tensors
    std::vector<float> q_data(batch * seq_len * embedding_dim);
    std::vector<float> kv_data(batch * seq_len * embedding_dim * 2);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    auto q_tensor = core::from_vector(q_data, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device);
    auto kv_tensor = core::from_vector(kv_data, ttnn::Shape{batch, 1, seq_len, embedding_dim * 2}, device);

    std::cout << "\nDIAGNOSTIC: Testing nlp_create_qkv_heads with separate Q and KV inputs\n";
    std::cout << "  Q shape: " << q_tensor.logical_shape() << "\n";
    std::cout << "  KV shape: " << kv_tensor.logical_shape() << "\n";
    std::cout << "  Expected head_dim: " << embedding_dim << " / " << num_heads << " = " << expected_head_dim << "\n";

    // Call the framework function directly
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        q_tensor,
        kv_tensor,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    auto q_shape = q.logical_shape();
    auto k_shape = k.logical_shape();
    auto v_shape = v.logical_shape();

    std::cout << "  Actual Q shape: " << q_shape << "\n";
    std::cout << "  Actual K shape: " << k_shape << "\n";
    std::cout << "  Actual V shape: " << v_shape << "\n";
    std::cout << "  Actual head_dim: " << q_shape[3] << "\n";

    if (q_shape[3] != expected_head_dim) {
        std::cout << "  ERROR: Head dimension mismatch! Expected " << expected_head_dim << ", got " << q_shape[3] << "\n";
        std::cout << "  FRAMEWORK BUG: nlp_create_qkv_heads incorrectly calculates head_dim even with separate inputs\n";
    }

    EXPECT_EQ(q_shape[3], expected_head_dim) << "Framework bug in head dimension calculation";
}

// Test 3: Grouped Query Attention dimension requirements
TEST_F(MultiHeadDiagnosticTest, GroupedQueryAttentionDimensions) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_q_heads = 8;
    uint32_t num_kv_heads = 2;  // Grouped: 8/2 = 4 queries per KV head

    uint32_t q_head_dim = embedding_dim / num_q_heads;    // 128/8 = 16
    uint32_t kv_head_dim = embedding_dim / num_kv_heads;  // 128/2 = 64

    std::cout << "\nDIAGNOSTIC: Testing Grouped Query Attention dimensions\n";
    std::cout << "  Embedding dim: " << embedding_dim << "\n";
    std::cout << "  Q heads: " << num_q_heads << ", KV heads: " << num_kv_heads << "\n";
    std::cout << "  Q head dim: " << q_head_dim << "\n";
    std::cout << "  KV head dim: " << kv_head_dim << "\n";
    std::cout << "  CORRECT BEHAVIOR: Q and KV should have different head dimensions\n";

    // Create Q and KV tensors
    std::vector<float> q_data(batch * seq_len * embedding_dim);
    std::vector<float> kv_data(batch * seq_len * embedding_dim * 2);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    auto q_tensor = core::from_vector(q_data, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device);
    auto kv_tensor = core::from_vector(kv_data, ttnn::Shape{batch, 1, seq_len, embedding_dim * 2}, device);

    std::cout << "  Q input shape: " << q_tensor.logical_shape() << "\n";
    std::cout << "  KV input shape: " << kv_tensor.logical_shape() << "\n";

    // This should work but the framework incorrectly enforces equal head dims
    bool caught_exception = false;
    try {
        auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
            q_tensor,
            kv_tensor,
            num_q_heads,
            num_kv_heads,
            /* transpose_k */ false,
            /* memory_config */ std::nullopt,
            /* optional_output_tensors */ std::nullopt);

        std::cout << "  SUCCESS: Framework accepted different head dimensions\n";
        std::cout << "  Output Q shape: " << q.logical_shape() << "\n";
        std::cout << "  Output K shape: " << k.logical_shape() << "\n";
        std::cout << "  Output V shape: " << v.logical_shape() << "\n";
    } catch (const std::exception& e) {
        caught_exception = true;
        std::string error_msg = e.what();
        if (error_msg.find("Head dims must be the same") != std::string::npos) {
            std::cout << "  FRAMEWORK BUG CONFIRMED: nlp_create_qkv_heads incorrectly enforces equal head dimensions\n";
            std::cout << "  Error message: " << error_msg << "\n";
            std::cout << "  This breaks Grouped Query Attention (GQA) implementation\n";
        }
    }

    EXPECT_FALSE(caught_exception) << "Framework incorrectly rejects valid GQA configuration";
}

// Test 4: Verify slice operation works correctly
TEST_F(MultiHeadDiagnosticTest, SliceOperationVerification) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 4;
    uint32_t total_dim = 12;

    // Create a tensor with known pattern
    std::vector<float> data(batch * seq_len * total_dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    auto tensor = core::from_vector(data, ttnn::Shape{batch, 1, seq_len, total_dim}, device);

    std::cout << "\nDIAGNOSTIC: Testing slice operation\n";
    std::cout << "  Input shape: " << tensor.logical_shape() << "\n";

    // Test slicing along last dimension
    std::array<uint32_t, 4> step = {1, 1, 1, 1};

    // Slice 1: [:, :, :, 0:4]
    std::array<uint32_t, 4> begin1 = {0, 0, 0, 0};
    std::array<uint32_t, 4> end1 = {batch, 1, seq_len, 4};
    auto slice1 = ttnn::slice(tensor, begin1, end1, step);

    // Slice 2: [:, :, :, 4:8]
    std::array<uint32_t, 4> begin2 = {0, 0, 0, 4};
    std::array<uint32_t, 4> end2 = {batch, 1, seq_len, 8};
    auto slice2 = ttnn::slice(tensor, begin2, end2, step);

    // Slice 3: [:, :, :, 8:12]
    std::array<uint32_t, 4> begin3 = {0, 0, 0, 8};
    std::array<uint32_t, 4> end3 = {batch, 1, seq_len, 12};
    auto slice3 = ttnn::slice(tensor, begin3, end3, step);

    std::cout << "  Slice1 shape: " << slice1.logical_shape() << "\n";
    std::cout << "  Slice2 shape: " << slice2.logical_shape() << "\n";
    std::cout << "  Slice3 shape: " << slice3.logical_shape() << "\n";

    // Verify the slices have correct shapes
    EXPECT_EQ(slice1.logical_shape()[3], 4);
    EXPECT_EQ(slice2.logical_shape()[3], 4);
    EXPECT_EQ(slice3.logical_shape()[3], 4);

    // Verify the values are correct
    auto slice1_data = core::to_vector(slice1);
    auto slice2_data = core::to_vector(slice2);
    auto slice3_data = core::to_vector(slice3);

    std::cout << "  First few values of slice1: ";
    for (int i = 0; i < std::min(8, (int)slice1_data.size()); ++i) {
        std::cout << slice1_data[i] << " ";
    }
    std::cout << "\n";

    std::cout << "  First few values of slice2: ";
    for (int i = 0; i < std::min(8, (int)slice2_data.size()); ++i) {
        std::cout << slice2_data[i] << " ";
    }
    std::cout << "\n";

    std::cout << "  First few values of slice3: ";
    for (int i = 0; i < std::min(8, (int)slice3_data.size()); ++i) {
        std::cout << slice3_data[i] << " ";
    }
    std::cout << "\n";
}

// Test 5: Test our workaround with slicing
TEST_F(MultiHeadDiagnosticTest, SlicingWorkaroundForQKVSplit) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 64;
    uint32_t num_heads = 4;

    // Create concatenated QKV
    std::vector<float> qkv_data(batch * seq_len * embedding_dim * 3);
    for (size_t i = 0; i < qkv_data.size(); ++i) {
        qkv_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape{batch, 1, seq_len, embedding_dim * 3}, device);

    std::cout << "\nDIAGNOSTIC: Testing slicing workaround for QKV split\n";
    std::cout << "  QKV shape: " << qkv_tensor.logical_shape() << "\n";

    // Split using slice
    std::array<uint32_t, 4> step = {1, 1, 1, 1};

    std::array<uint32_t, 4> q_begin = {0, 0, 0, 0};
    std::array<uint32_t, 4> q_end = {batch, 1, seq_len, embedding_dim};
    auto q_split = ttnn::slice(qkv_tensor, q_begin, q_end, step);

    std::array<uint32_t, 4> k_begin = {0, 0, 0, embedding_dim};
    std::array<uint32_t, 4> k_end = {batch, 1, seq_len, 2 * embedding_dim};
    auto k_split = ttnn::slice(qkv_tensor, k_begin, k_end, step);

    std::array<uint32_t, 4> v_begin = {0, 0, 0, 2 * embedding_dim};
    std::array<uint32_t, 4> v_end = {batch, 1, seq_len, 3 * embedding_dim};
    auto v_split = ttnn::slice(qkv_tensor, v_begin, v_end, step);

    std::cout << "  Q split shape: " << q_split.logical_shape() << "\n";
    std::cout << "  K split shape: " << k_split.logical_shape() << "\n";
    std::cout << "  V split shape: " << v_split.logical_shape() << "\n";

    // Now recombine K and V
    auto kv_concat = ttnn::concat(std::vector<ttnn::Tensor>({k_split, v_split}), /* dim */ 3);
    std::cout << "  KV concat shape: " << kv_concat.logical_shape() << "\n";

    // Try using nlp_create_qkv_heads with separate inputs
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        q_split,
        kv_concat,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    std::cout << "  Output Q shape: " << q.logical_shape() << "\n";
    std::cout << "  Output K shape: " << k.logical_shape() << "\n";
    std::cout << "  Output V shape: " << v.logical_shape() << "\n";

    uint32_t expected_head_dim = embedding_dim / num_heads;
    if (q.logical_shape()[3] != expected_head_dim) {
        std::cout << "  ERROR: Even with slicing workaround, head_dim is wrong!\n";
        std::cout << "  Expected: " << expected_head_dim << ", Got: " << q.logical_shape()[3] << "\n";
    }

    EXPECT_EQ(q.logical_shape()[3], expected_head_dim);
}

}  // namespace ttml::ops::tests

