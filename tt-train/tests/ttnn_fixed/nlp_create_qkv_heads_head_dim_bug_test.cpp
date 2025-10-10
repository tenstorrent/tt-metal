// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed::tests {

class NlpCreateQkvHeadsHeadDimBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Demonstrates bug in ttnn::experimental::nlp_create_qkv_heads when head_dim < 32
TEST_F(NlpCreateQkvHeadsHeadDimBugTest, HeadDimLessThan32ProducesWrongShapes) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Configuration that triggers the bug
    const uint32_t batch_size = 1;
    const uint32_t seq_len = 32;
    const uint32_t num_heads = 4;
    const uint32_t head_dim = 16;                         // < 32 (TILE_WIDTH) - triggers bug!
    const uint32_t embedding_dim = num_heads * head_dim;  // 64

    fmt::print("\n=== Test Configuration ===\n");
    fmt::print("batch_size: {}\n", batch_size);
    fmt::print("seq_len: {}\n", seq_len);
    fmt::print("num_heads: {}\n", num_heads);
    fmt::print("head_dim: {}\n", head_dim);
    fmt::print("embedding_dim: {}\n", embedding_dim);

    // Create fused QKV tensor: [batch, 1, seq_len, embedding_dim * 3]
    const uint32_t qkv_width = embedding_dim * 3;  // 192
    std::vector<float> qkv_data(batch_size * seq_len * qkv_width);
    for (size_t i = 0; i < qkv_data.size(); ++i) {
        qkv_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape{batch_size, 1, seq_len, qkv_width}, device);

    fmt::print("\n=== Input Shape ===\n");
    fmt::print("QKV input: {}\n", qkv_tensor.logical_shape());

    // Call nlp_create_qkv_heads - this is where the bug manifests
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qkv_tensor,
        /* input_tensor_kv */ std::nullopt,
        /* num_heads */ num_heads,
        /* num_kv_heads */ num_heads,
        /* transpose_k_heads */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    fmt::print("\n=== Output Shapes ===\n");
    fmt::print("Q output: {}\n", q.logical_shape());
    fmt::print("K output: {}\n", k.logical_shape());
    fmt::print("V output: {}\n", v.logical_shape());

    // Expected shapes: [batch_size, num_heads, seq_len, head_dim]
    const auto expected_shape = ttnn::Shape{batch_size, num_heads, seq_len, head_dim};

    fmt::print("\n=== Expected Shape ===\n");
    fmt::print("Expected: {}\n", expected_shape);

    fmt::print("\n=== Bug Analysis ===\n");
    fmt::print("When head_dim={} < TILE_WIDTH=32:\n", head_dim);
    fmt::print("  q_out_w_tiles = head_dim / TILE_WIDTH = {} / 32 = 0 (integer division!)\n", head_dim);
    fmt::print("  q_num_tiles = num_heads * q_out_w_tiles = {} * 0 = 0\n", num_heads);
    fmt::print("  Result: Kernel reads ZERO Q tiles, causing wrong offsets!\n");

    // Check if shapes match expected
    auto q_shape = q.logical_shape();
    bool shapes_correct =
        (q_shape[0] == batch_size && q_shape[1] == num_heads && q_shape[2] == seq_len && q_shape[3] == head_dim);

    if (!shapes_correct) {
        fmt::print("\n🔴 BUG CONFIRMED: Output shapes are WRONG!\n");
        fmt::print("Expected last dimension: {}, Got: {}\n", head_dim, q_shape[3]);
        fmt::print("The kernel produced dimension {} instead of {}\n", q_shape[3], head_dim);

        // This test documents the bug - it will FAIL until ttnn is fixed
        FAIL() << "nlp_create_qkv_heads produces wrong shapes when head_dim < TILE_WIDTH (32)";
    } else {
        fmt::print("\n✅ Shapes are correct - bug may be fixed!\n");
    }
}

// Demonstrates that head_dim >= 32 works correctly
TEST_F(NlpCreateQkvHeadsHeadDimBugTest, HeadDim32OrLargerWorksCorrectly) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Configuration that DOES NOT trigger the bug
    const uint32_t batch_size = 1;
    const uint32_t seq_len = 32;
    const uint32_t num_heads = 4;
    const uint32_t head_dim = 32;                         // >= 32 - should work!
    const uint32_t embedding_dim = num_heads * head_dim;  // 128

    fmt::print("\n=== Test Configuration (Control Case) ===\n");
    fmt::print("batch_size: {}\n", batch_size);
    fmt::print("seq_len: {}\n", seq_len);
    fmt::print("num_heads: {}\n", num_heads);
    fmt::print("head_dim: {}\n", head_dim);
    fmt::print("embedding_dim: {}\n", embedding_dim);

    // Create fused QKV tensor
    const uint32_t qkv_width = embedding_dim * 3;
    std::vector<float> qkv_data(batch_size * seq_len * qkv_width);
    for (size_t i = 0; i < qkv_data.size(); ++i) {
        qkv_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape{batch_size, 1, seq_len, qkv_width}, device);

    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qkv_tensor, std::nullopt, num_heads, num_heads, false, std::nullopt, std::nullopt);

    fmt::print("\n=== Output Shapes (Control Case) ===\n");
    fmt::print("Q output: {}\n", q.logical_shape());

    // Check shapes
    auto q_shape = q.logical_shape();
    EXPECT_EQ(q_shape[0], batch_size);
    EXPECT_EQ(q_shape[1], num_heads);
    EXPECT_EQ(q_shape[2], seq_len);
    EXPECT_EQ(q_shape[3], head_dim);

    fmt::print("✅ Control case works correctly with head_dim >= 32\n");
}

}  // namespace ttml::ttnn_fixed::tests
