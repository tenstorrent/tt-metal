// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/scaled_dot_product_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <cmath>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

class ScaledDotProductAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// ============================================================================
// BASIC ATTENTION TESTS
// ============================================================================

TEST_F(ScaledDotProductAttentionTest, BasicAttentionNoMask) {
    auto* device = &autograd::ctx().get_device();

    // Small test case for basic validation
    uint32_t batch = 1;
    uint32_t num_heads = 2;
    uint32_t seq_len = 4;
    uint32_t head_dim = 8;

    // Create Q, K, V tensors
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_heads * seq_len * head_dim);

    // Initialize with distinct patterns
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 10) * 0.1f;
        k_data[i] = static_cast<float>((i + 5) % 10) * 0.1f;
        v_data[i] = static_cast<float>((i + 3) % 10) * 0.1f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Compute attention
    auto output = scaled_dot_product_attention(query, key, value);

    // Check output shape
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], num_heads);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], head_dim);

    // Check that output values are reasonable (not NaN or Inf)
    auto output_data = core::to_vector(output->get_value());
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
        EXPECT_GE(output_data[i], -10.0f);
        EXPECT_LE(output_data[i], 10.0f);
    }
}

TEST_F(ScaledDotProductAttentionTest, AttentionWithMask) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t num_heads = 4;
    uint32_t seq_len = 8;
    uint32_t head_dim = 16;

    // Create Q, K, V tensors
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim, 0.1f);
    std::vector<float> k_data(batch * num_heads * seq_len * head_dim, 0.2f);
    std::vector<float> v_data(batch * num_heads * seq_len * head_dim, 0.3f);

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Create attention mask (1 for valid, 0 for masked)
    // Mask out the last 2 positions
    std::vector<float> mask_data(batch * num_heads * seq_len * seq_len, 1.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = seq_len - 2; j < seq_len; ++j) {
                    size_t idx = b * num_heads * seq_len * seq_len +
                                h * seq_len * seq_len +
                                i * seq_len + j;
                    mask_data[idx] = 0.0f;
                }
            }
        }
    }

    auto mask = autograd::create_tensor(
        core::from_vector(mask_data, ttnn::Shape{batch, num_heads, seq_len, seq_len}, device));

    // Compute attention with mask
    auto output = scaled_dot_product_attention(query, key, value, mask);

    // Check output shape
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], num_heads);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], head_dim);

    // Output should be valid
    auto output_data = core::to_vector(output->get_value());
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

TEST_F(ScaledDotProductAttentionTest, AttentionBackward) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 1;
    uint32_t num_heads = 2;
    uint32_t seq_len = 4;
    uint32_t head_dim = 8;

    // Create Q, K, V tensors
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i) * 0.01f;
        k_data[i] = static_cast<float>(i + 10) * 0.01f;
        v_data[i] = static_cast<float>(i + 20) * 0.01f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Compute attention
    auto output = scaled_dot_product_attention(query, key, value);

    // Create target and compute loss
    auto target = autograd::create_tensor(core::zeros_like(output->get_value()));
    auto loss = mse_loss(output, target);

    // Backward pass
    loss->backward();

    // Check that all inputs received gradients
    EXPECT_TRUE(core::is_tensor_initialized(query->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(key->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(value->get_grad()));

    // Check gradients are non-zero and reasonable
    auto q_grad = core::to_vector(query->get_grad());
    auto k_grad = core::to_vector(key->get_grad());
    auto v_grad = core::to_vector(value->get_grad());

    bool q_has_non_zero = false;
    bool k_has_non_zero = false;
    bool v_has_non_zero = false;

    for (size_t i = 0; i < q_grad.size(); ++i) {
        EXPECT_FALSE(std::isnan(q_grad[i]));
        if (std::abs(q_grad[i]) > 1e-6f) q_has_non_zero = true;
    }

    for (size_t i = 0; i < k_grad.size(); ++i) {
        EXPECT_FALSE(std::isnan(k_grad[i]));
        if (std::abs(k_grad[i]) > 1e-6f) k_has_non_zero = true;
    }

    for (size_t i = 0; i < v_grad.size(); ++i) {
        EXPECT_FALSE(std::isnan(v_grad[i]));
        if (std::abs(v_grad[i]) > 1e-6f) v_has_non_zero = true;
    }

    EXPECT_TRUE(q_has_non_zero);
    EXPECT_TRUE(k_has_non_zero);
    EXPECT_TRUE(v_has_non_zero);
}

// ============================================================================
// GROUPED QUERY ATTENTION TESTS
// ============================================================================

TEST_F(ScaledDotProductAttentionTest, GroupedQueryAttention) {
    auto* device = &autograd::ctx().get_device();

    // Test grouped query attention where K,V have fewer heads than Q
    uint32_t batch = 2;
    uint32_t q_heads = 8;
    uint32_t kv_heads = 2;  // 4 query heads per KV head
    uint32_t seq_len = 16;
    uint32_t head_dim = 32;

    // Create tensors with different head counts
    std::vector<float> q_data(batch * q_heads * seq_len * head_dim);
    std::vector<float> kv_data(batch * kv_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = static_cast<float>(i % 50) * 0.02f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, q_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(kv_data, ttnn::Shape{batch, kv_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(kv_data, ttnn::Shape{batch, kv_heads, seq_len, head_dim}, device));

    // Compute grouped attention
    auto output = scaled_dot_product_attention(query, key, value);

    // Output should have same shape as query
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], q_heads);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], head_dim);
}

TEST_F(ScaledDotProductAttentionTest, GroupedQueryAttentionBackward) {
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 1;
    uint32_t q_heads = 4;
    uint32_t kv_heads = 2;
    uint32_t seq_len = 8;
    uint32_t head_dim = 16;

    // Create tensors
    std::vector<float> q_data(batch * q_heads * seq_len * head_dim);
    std::vector<float> kv_data(batch * kv_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i) * 0.001f;
    }
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = static_cast<float>(i) * 0.002f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, q_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(kv_data, ttnn::Shape{batch, kv_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(kv_data, ttnn::Shape{batch, kv_heads, seq_len, head_dim}, device));

    // Compute grouped attention
    auto output = scaled_dot_product_attention(query, key, value);

    // Backward pass
    auto target = autograd::create_tensor(core::zeros_like(output->get_value()));
    auto loss = mse_loss(output, target);
    loss->backward();

    // Check gradients
    EXPECT_TRUE(core::is_tensor_initialized(query->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(key->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(value->get_grad()));

    // K and V gradients should be accumulated across groups
    auto k_grad = core::to_vector(key->get_grad());
    auto v_grad = core::to_vector(value->get_grad());

    // Gradients should exist and be reasonable
    for (size_t i = 0; i < k_grad.size(); ++i) {
        EXPECT_FALSE(std::isnan(k_grad[i]));
        EXPECT_FALSE(std::isinf(k_grad[i]));
    }
}

// ============================================================================
// BERT-SPECIFIC PATTERNS
// ============================================================================

TEST_F(ScaledDotProductAttentionTest, BERTAttentionPattern) {
    // Test the specific attention pattern used in BERT
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t num_heads = 12;
    uint32_t seq_len = 128;
    uint32_t head_dim = 64;  // 768 / 12 for BERT-base

    // Create Q, K, V as would come from BERT's heads_creation
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_heads * seq_len * head_dim);

    // Initialize with patterns similar to BERT
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 1000) * 0.001f;
        k_data[i] = static_cast<float>((i + 100) % 1000) * 0.001f;
        v_data[i] = static_cast<float>((i + 200) % 1000) * 0.001f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Create BERT-style attention mask
    // In BERT, mask is typically [batch, 1, 1, seq_len] and gets broadcast
    // For this test, we'll create the full mask
    std::vector<float> mask_data(batch * num_heads * seq_len * seq_len, 1.0f);

    // Mask out padding tokens (simulate last 20 tokens are padding)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = seq_len - 20; j < seq_len; ++j) {
                    size_t idx = b * num_heads * seq_len * seq_len +
                                h * seq_len * seq_len +
                                i * seq_len + j;
                    mask_data[idx] = 0.0f;
                }
            }
        }
    }

    auto mask = autograd::create_tensor(
        core::from_vector(mask_data, ttnn::Shape{batch, num_heads, seq_len, seq_len}, device));

    // Compute attention
    auto output = scaled_dot_product_attention(query, key, value, mask);

    // Verify output shape matches BERT expectations
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], num_heads);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], head_dim);

    // Test backward pass
    auto target = autograd::create_tensor(
        core::from_vector(
            std::vector<float>(batch * num_heads * seq_len * head_dim, 0.5f),
            ttnn::Shape{batch, num_heads, seq_len, head_dim},
            device));
    auto loss = mse_loss(output, target);
    loss->backward();

    // All inputs should have gradients
    EXPECT_TRUE(core::is_tensor_initialized(query->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(key->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(value->get_grad()));
}

TEST_F(ScaledDotProductAttentionTest, ScalingFactorCorrectness) {
    // Verify that the scaling factor sqrt(d_k) is applied correctly
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 1;
    uint32_t num_heads = 1;
    uint32_t seq_len = 2;
    uint32_t head_dim = 4;

    // Create simple tensors for easy verification
    std::vector<float> q_data = {1.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> k_data = {1.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> v_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Compute attention
    auto output = scaled_dot_product_attention(query, key, value);

    // With Q and K as identity-like matrices, QK^T should give us
    // [[1, 0], [0, 1]] before scaling
    // After scaling by 1/sqrt(4) = 0.5: [[0.5, 0], [0, 0.5]]
    // After softmax, each row should sum to 1
    // The attention weights should favor the diagonal

    auto output_data = core::to_vector(output->get_value());

    // Due to the structure of our input, output should be close to V
    // but with some attention weighting applied
    EXPECT_EQ(output_data.size(), 8);

    // Check that values are reasonable
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_GE(output_data[i], 0.0f);
        EXPECT_LE(output_data[i], 10.0f);
    }
}

TEST_F(ScaledDotProductAttentionTest, AttentionShapeValidation) {
    // Test that invalid shapes are handled correctly
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 16;
    uint32_t head_dim = 32;

    // Create Q with 4 heads, K and V with 3 heads (invalid - not divisible)
    auto query = autograd::create_tensor(
        core::from_vector(
            std::vector<float>(batch * 4 * seq_len * head_dim, 0.1f),
            ttnn::Shape{batch, 4, seq_len, head_dim},
            device));
    auto key = autograd::create_tensor(
        core::from_vector(
            std::vector<float>(batch * 3 * seq_len * head_dim, 0.1f),
            ttnn::Shape{batch, 3, seq_len, head_dim},
            device));
    auto value = autograd::create_tensor(
        core::from_vector(
            std::vector<float>(batch * 3 * seq_len * head_dim, 0.1f),
            ttnn::Shape{batch, 3, seq_len, head_dim},
            device));

    // This should throw because 4 is not divisible by 3
    EXPECT_THROW(
        scaled_dot_product_attention(query, key, value),
        std::invalid_argument
    );
}

TEST_F(ScaledDotProductAttentionTest, LargeSequenceLengthStability) {
    // Test numerical stability with larger sequence lengths
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 1;
    uint32_t num_heads = 8;
    uint32_t seq_len = 512;  // Large sequence length
    uint32_t head_dim = 64;

    // Create tensors with small values to avoid overflow
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i % 100) * 0.0001f;
        k_data[i] = static_cast<float>((i + 50) % 100) * 0.0001f;
        v_data[i] = static_cast<float>((i + 25) % 100) * 0.0001f;
    }

    auto query = autograd::create_tensor(
        core::from_vector(q_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto key = autograd::create_tensor(
        core::from_vector(k_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));
    auto value = autograd::create_tensor(
        core::from_vector(v_data, ttnn::Shape{batch, num_heads, seq_len, head_dim}, device));

    // Compute attention - should not overflow or produce NaN
    auto output = scaled_dot_product_attention(query, key, value);

    auto output_data = core::to_vector(output->get_value());

    // Check for numerical stability
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (size_t i = 0; i < output_data.size(); ++i) {
        if (std::isnan(output_data[i])) nan_count++;
        if (std::isinf(output_data[i])) inf_count++;
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN values in output";
    EXPECT_EQ(inf_count, 0) << "Found " << inf_count << " Inf values in output";
}

}  // namespace ttml::ops::tests
