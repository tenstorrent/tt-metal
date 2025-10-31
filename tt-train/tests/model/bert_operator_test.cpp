// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Comprehensive BERT Operator Unit Tests
 *
 * Tests each BERT operation in isolation using C++ directly.
 * This implements Option 2 from OPERATOR_TESTING_LIMITATIONS.md
 *
 * These tests verify:
 * 1. Heads creation (QKV splitting)
 * 2. Heads fusion (merging heads back)
 * 3. Scaled dot-product attention
 * 4. Complete MHA pipeline
 * 5. LayerNorm
 * 6. GELU activation
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/linear_op.hpp"
#include "ops/matmul_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ops/unary_ops.hpp"

using namespace ttml;

namespace {

/**
 * Helper function to compute Pearson Correlation Coefficient
 */
float compute_pcc(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0F;
    }

    float mean_a = 0.0F, mean_b = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= a.size();
    mean_b /= b.size();

    float numerator = 0.0F;
    float denom_a = 0.0F;
    float denom_b = 0.0F;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff_a = a[i] - mean_a;
        float diff_b = b[i] - mean_b;
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }

    float denominator = std::sqrt(denom_a * denom_b);
    return denominator > 0.0F ? numerator / denominator : 0.0F;
}

/**
 * Helper to create random tensor data
 */
std::vector<float> create_random_data(size_t size, float mean = 0.0F, float stddev = 1.0F, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

/**
 * Helper to print comparison stats
 */
void print_comparison(
    const std::string& name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float pcc_threshold = 0.999F) {
    float pcc = compute_pcc(expected, actual);

    float max_diff = 0.0F;
    float sum_diff = 0.0F;
    for (size_t i = 0; i < expected.size(); ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    float mean_diff = sum_diff / expected.size();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << name << ": PCC = " << pcc << (pcc >= pcc_threshold ? " ✅ PASS" : " ❌ FAIL") << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Size: " << expected.size() << "\n";
    std::cout << "Mean abs diff: " << mean_diff << ", Max abs diff: " << max_diff << "\n";

    // Print first few values
    std::cout << "First 5 expected: ";
    for (size_t i = 0; i < std::min(size_t(5), expected.size()); ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "\nFirst 5 actual: ";
    for (size_t i = 0; i < std::min(size_t(5), actual.size()); ++i) {
        std::cout << actual[i] << " ";
    }
    std::cout << "\n";

    EXPECT_GE(pcc, pcc_threshold) << name << " failed PCC threshold";
}

}  // namespace

class BERTOperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        batch_size = 1;
        seq_len = 32;
        hidden_dim = 128;
        num_heads = 2;
        head_dim = hidden_dim / num_heads;
    }

    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t head_dim;
};

/**
 * Test heads_creation operation
 *
 * This tests the QKV splitting into separate attention heads.
 * Input: [B, 1, S, E*3]
 * Output: Q, K, V each [B, H, S, E/H]
 */
TEST_F(BERTOperatorTest, HeadsCreation) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Heads Creation (QKV splitting)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random QKV input [B, 1, S, E*3]
    auto qkv_data = create_random_data(batch_size * 1 * seq_len * hidden_dim * 3);

    // Create TTML tensor
    auto qkv_tensor = core::from_vector(
        qkv_data, ttnn::Shape({batch_size, 1, seq_len, hidden_dim * 3}), &autograd::ctx().get_device());
    auto qkv = autograd::create_tensor(qkv_tensor);

    // Call heads_creation
    auto [q, k, v] = ops::heads_creation(qkv, num_heads);

    // Verify shapes
    auto q_shape = q->get_value().logical_shape();
    auto k_shape = k->get_value().logical_shape();
    auto v_shape = v->get_value().logical_shape();

    EXPECT_EQ(q_shape.rank(), 4);
    EXPECT_EQ(q_shape[0], batch_size);
    EXPECT_EQ(q_shape[1], num_heads);
    EXPECT_EQ(q_shape[2], seq_len);
    EXPECT_EQ(q_shape[3], head_dim);

    EXPECT_EQ(k_shape.rank(), 4);
    EXPECT_EQ(k_shape[0], batch_size);
    EXPECT_EQ(k_shape[1], num_heads);
    EXPECT_EQ(k_shape[2], seq_len);
    EXPECT_EQ(k_shape[3], head_dim);

    EXPECT_EQ(v_shape.rank(), 4);
    EXPECT_EQ(v_shape[0], batch_size);
    EXPECT_EQ(v_shape[1], num_heads);
    EXPECT_EQ(v_shape[2], seq_len);
    EXPECT_EQ(v_shape[3], head_dim);

    // Verify the splitting is correct by checking that each head gets contiguous dims
    // Extract Q, K, V from original data manually
    auto q_expected = std::vector<float>(batch_size * num_heads * seq_len * head_dim);
    auto k_expected = std::vector<float>(batch_size * num_heads * seq_len * head_dim);
    auto v_expected = std::vector<float>(batch_size * num_heads * seq_len * head_dim);

    // Manual head splitting for reference
    // Input layout: [B, 1, S, E*3] where E*3 = [Q_dims | K_dims | V_dims]
    // For each position (b, s):
    //   - Extract Q: dims [0:E]
    //   - Extract K: dims [E:2E]
    //   - Extract V: dims [2E:3E]
    // Then reshape each to [B, H, S, E/H]

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            for (uint32_t h = 0; h < num_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    // Input index: [b, 0, s, :]
                    // Q at dims [h*head_dim + d]
                    // K at dims [hidden_dim + h*head_dim + d]
                    // V at dims [2*hidden_dim + h*head_dim + d]

                    size_t input_base = b * seq_len * hidden_dim * 3 + s * hidden_dim * 3;

                    // Output index: [b, h, s, d]
                    size_t output_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;

                    q_expected[output_idx] = qkv_data[input_base + h * head_dim + d];
                    k_expected[output_idx] = qkv_data[input_base + hidden_dim + h * head_dim + d];
                    v_expected[output_idx] = qkv_data[input_base + 2 * hidden_dim + h * head_dim + d];
                }
            }
        }
    }

    // Get actual outputs
    auto q_actual = core::to_vector(q->get_value());
    auto k_actual = core::to_vector(k->get_value());
    auto v_actual = core::to_vector(v->get_value());

    // Compare
    print_comparison("Q Heads", q_expected, q_actual);
    print_comparison("K Heads", k_expected, k_actual);
    print_comparison("V Heads", v_expected, v_actual);
}

/**
 * Test heads_fusion operation
 *
 * This tests merging attention heads back into hidden dimension.
 * Input: [B, H, S, E/H]
 * Output: [B, 1, S, E]
 */
TEST_F(BERTOperatorTest, HeadsFusion) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Heads Fusion (merging heads)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random heads data [B, H, S, E/H]
    auto heads_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);

    // Create TTML tensor
    auto heads_tensor = core::from_vector(
        heads_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto heads = autograd::create_tensor(heads_tensor);

    // Call heads_fusion
    auto fused = ops::heads_fusion(heads);

    // Verify shape
    auto fused_shape = fused->get_value().logical_shape();
    EXPECT_EQ(fused_shape.rank(), 4);
    EXPECT_EQ(fused_shape[0], batch_size);
    EXPECT_EQ(fused_shape[1], 1);
    EXPECT_EQ(fused_shape[2], seq_len);
    EXPECT_EQ(fused_shape[3], hidden_dim);

    // Compute expected output manually
    // Goal: [B, H, S, E/H] -> [B, 1, S, E]
    // This is transpose(1,2) then reshape
    auto expected = std::vector<float>(batch_size * 1 * seq_len * hidden_dim);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            for (uint32_t h = 0; h < num_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    // Input index: [b, h, s, d]
                    size_t input_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;

                    // Output index: [b, 0, s, h*head_dim + d]
                    size_t output_idx = b * seq_len * hidden_dim + s * hidden_dim + h * head_dim + d;

                    expected[output_idx] = heads_data[input_idx];
                }
            }
        }
    }

    // Get actual output
    auto actual = core::to_vector(fused->get_value());

    // Compare
    print_comparison("Fused Heads", expected, actual);
}

/**
 * Test scaled_dot_product_attention operation
 *
 * Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttention) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Scaled Dot-Product Attention\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);

    // Call scaled_dot_product_attention
    auto output = ops::scaled_dot_product_attention(q, k, v);

    // Verify shape
    auto output_shape = output->get_value().logical_shape();
    EXPECT_EQ(output_shape.rank(), 4);
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], num_heads);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], head_dim);

    // NOTE: We can't easily compute the expected output here without reimplementing
    // the entire attention mechanism. This test primarily verifies:
    // 1. The operation runs without errors
    // 2. Output shape is correct
    // 3. Output values are reasonable (not NaN/Inf)

    auto actual = core::to_vector(output->get_value());

    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (const auto& val : actual) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Output contains NaN or Inf values";

    // Check output statistics are reasonable
    float sum = 0.0F;
    for (const auto& val : actual) {
        sum += val;
    }
    float mean = sum / actual.size();
    std::cout << "Output mean: " << mean << ", size: " << actual.size() << "\n";

    // Mean should be roughly 0 (since input is standard normal)
    EXPECT_LT(std::abs(mean), 1.0F) << "Output mean is unreasonable";
}

/**
 * Test LayerNorm operation
 */
TEST_F(BERTOperatorTest, LayerNorm) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: LayerNorm\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random input [B, 1, S, E]
    auto input_data = create_random_data(batch_size * 1 * seq_len * hidden_dim, 0.0F, 1.0F, 42);
    auto gamma_data = create_random_data(hidden_dim, 1.0F, 0.1F, 43);
    auto beta_data = create_random_data(hidden_dim, 0.0F, 0.1F, 44);

    float eps = 1e-12F;

    // Create TTML tensors
    auto input_tensor =
        core::from_vector(input_data, ttnn::Shape({batch_size, 1, seq_len, hidden_dim}), &autograd::ctx().get_device());
    auto gamma_tensor =
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, hidden_dim}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, hidden_dim}), &autograd::ctx().get_device());

    auto input = autograd::create_tensor(input_tensor);
    auto gamma = autograd::create_tensor(gamma_tensor);
    auto beta = autograd::create_tensor(beta_tensor);

    // Call layernorm with hardware clamp disabled
    auto output = ops::layernorm(input, gamma, beta, eps, false);

    // Verify shape
    auto output_shape = output->get_value().logical_shape();
    EXPECT_EQ(output_shape.rank(), 4);
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], hidden_dim);

    auto actual = core::to_vector(output->get_value());

    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (const auto& val : actual) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "LayerNorm output contains NaN or Inf values";

    std::cout << "LayerNorm test passed - output shape and values are valid\n";
}

/**
 * Test GELU activation
 */
TEST_F(BERTOperatorTest, GELU) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: GELU Activation\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random input [B, 1, S, E]
    auto input_data = create_random_data(batch_size * 1 * seq_len * hidden_dim, 0.0F, 1.0F, 42);

    // Create TTML tensor
    auto input_tensor =
        core::from_vector(input_data, ttnn::Shape({batch_size, 1, seq_len, hidden_dim}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tensor);

    // Call GELU
    auto output = ops::gelu(input);

    // Verify shape
    auto output_shape = output->get_value().logical_shape();
    EXPECT_EQ(output_shape.rank(), 4);
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], hidden_dim);

    auto actual = core::to_vector(output->get_value());

    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (const auto& val : actual) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "GELU output contains NaN or Inf values";

    // Verify GELU properties: output should be roughly in [-0.17, inf) range
    // and for positive inputs should be close to input
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (input_data[i] > 3.0F) {
            // For large positive x, GELU(x) ≈ x
            EXPECT_NEAR(actual[i], input_data[i], 0.1F);
        }
        if (input_data[i] < -3.0F) {
            // For large negative x, GELU(x) ≈ 0
            EXPECT_NEAR(actual[i], 0.0F, 0.1F);
        }
    }

    std::cout << "GELU test passed - output shape and values are valid\n";
}

/**
 * Test complete MHA pipeline
 *
 * Tests the entire multi-head attention flow:
 * Input -> QKV projection -> heads_creation -> attention -> heads_fusion -> output projection
 */
TEST_F(BERTOperatorTest, CompleteMHAPipeline) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Complete MHA Pipeline\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random input and weights
    auto input_data = create_random_data(batch_size * 1 * seq_len * hidden_dim, 0.0F, 1.0F, 42);
    auto qkv_weight_data = create_random_data(hidden_dim * 3 * hidden_dim, 0.0F, 0.02F, 43);
    auto qkv_bias_data = create_random_data(hidden_dim * 3, 0.0F, 0.01F, 44);
    auto out_weight_data = create_random_data(hidden_dim * hidden_dim, 0.0F, 0.02F, 45);
    auto out_bias_data = create_random_data(hidden_dim, 0.0F, 0.01F, 46);

    // Create TTML tensors
    auto input_tensor =
        core::from_vector(input_data, ttnn::Shape({batch_size, 1, seq_len, hidden_dim}), &autograd::ctx().get_device());
    auto qkv_weight_tensor =
        core::from_vector(qkv_weight_data, ttnn::Shape({hidden_dim * 3, hidden_dim}), &autograd::ctx().get_device());
    auto qkv_bias_tensor =
        core::from_vector(qkv_bias_data, ttnn::Shape({hidden_dim * 3}), &autograd::ctx().get_device());
    auto out_weight_tensor =
        core::from_vector(out_weight_data, ttnn::Shape({hidden_dim, hidden_dim}), &autograd::ctx().get_device());
    auto out_bias_tensor = core::from_vector(out_bias_data, ttnn::Shape({hidden_dim}), &autograd::ctx().get_device());

    auto input = autograd::create_tensor(input_tensor);
    auto qkv_weight = autograd::create_tensor(qkv_weight_tensor);
    auto qkv_bias = autograd::create_tensor(qkv_bias_tensor);
    auto out_weight = autograd::create_tensor(out_weight_tensor);
    auto out_bias = autograd::create_tensor(out_bias_tensor);

    // QKV projection
    auto qkv = ops::linear_op(input, qkv_weight, qkv_bias);

    // Split into heads
    auto [q, k, v] = ops::heads_creation(qkv, num_heads);

    // Attention
    auto attn_output = ops::scaled_dot_product_attention(q, k, v);

    // Merge heads
    auto merged = ops::heads_fusion(attn_output);

    // Output projection
    auto output = ops::linear_op(merged, out_weight, out_bias);

    // Verify final shape
    auto output_shape = output->get_value().logical_shape();
    EXPECT_EQ(output_shape.rank(), 4);
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], hidden_dim);

    auto actual = core::to_vector(output->get_value());

    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (const auto& val : actual) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "MHA pipeline output contains NaN or Inf values";

    std::cout << "Complete MHA pipeline test passed - all operations executed successfully\n";
}

/**
 * Reference implementation of softmax for validation
 */
std::vector<float> reference_softmax(const std::vector<float>& input, size_t last_dim_size) {
    std::vector<float> output(input.size());
    size_t num_rows = input.size() / last_dim_size;

    for (size_t row = 0; row < num_rows; ++row) {
        size_t offset = row * last_dim_size;

        // Find max for numerical stability
        float max_val = input[offset];
        for (size_t i = 1; i < last_dim_size; ++i) {
            max_val = std::max(max_val, input[offset + i]);
        }

        // Compute exp and sum
        float sum = 0.0F;
        for (size_t i = 0; i < last_dim_size; ++i) {
            output[offset + i] = std::exp(input[offset + i] - max_val);
            sum += output[offset + i];
        }

        // Normalize
        for (size_t i = 0; i < last_dim_size; ++i) {
            output[offset + i] /= sum;
        }
    }

    return output;
}

/**
 * Reference implementation of attention for validation
 * Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
 */
std::vector<float> reference_attention(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    const std::vector<float>* mask_data = nullptr) {
    float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    // Compute Q @ K^T for each batch and head
    std::vector<float> attn_scores(batch_size * num_heads * seq_len * seq_len, 0.0F);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t i = 0; i < seq_len; ++i) {
                for (uint32_t j = 0; j < seq_len; ++j) {
                    float sum = 0.0F;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        size_t q_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d;
                        size_t k_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                        sum += q_data[q_idx] * k_data[k_idx];
                    }
                    size_t score_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                    attn_scores[score_idx] = sum * scale;
                }
            }
        }
    }

    // Apply mask if provided (mask format: 1 = attend, 0 = mask out)
    if (mask_data != nullptr) {
        for (uint32_t b = 0; b < batch_size; ++b) {
            for (uint32_t h = 0; h < num_heads; ++h) {
                for (uint32_t i = 0; i < seq_len; ++i) {
                    for (uint32_t j = 0; j < seq_len; ++j) {
                        // Mask is [B, 1, 1, S] - broadcast across heads and query positions
                        size_t mask_idx = b * seq_len + j;
                        float mask_val = (*mask_data)[mask_idx];

                        if (mask_val == 0.0F) {
                            size_t score_idx =
                                b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            attn_scores[score_idx] = -1e9F;  // Large negative value
                        }
                    }
                }
            }
        }
    }

    // Apply softmax
    auto attn_weights = reference_softmax(attn_scores, seq_len);

    // Compute attn_weights @ V
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim, 0.0F);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t i = 0; i < seq_len; ++i) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0F;
                    for (uint32_t j = 0; j < seq_len; ++j) {
                        size_t weight_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        size_t v_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                        sum += attn_weights[weight_idx] * v_data[v_idx];
                    }
                    size_t out_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d;
                    output[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

/**
 * Test scaled_dot_product_attention WITH REFERENCE COMPARISON (no mask)
 *
 * This addresses deficiency #2: "They don't compare against a reference"
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttentionWithReference) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Scaled Dot-Product Attention (with reference comparison)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // Compute reference output
    auto expected = reference_attention(q_data, k_data, v_data, batch_size, num_heads, seq_len, head_dim);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);

    // Call TTML scaled_dot_product_attention (NO MASK)
    auto output = ops::scaled_dot_product_attention(q, k, v);

    // Get actual output
    auto actual = core::to_vector(output->get_value());

    // Compare with reference using PCC
    print_comparison("Attention (No Mask)", expected, actual, 0.99F);
}

/**
 * Test scaled_dot_product_attention WITH MASK AND REFERENCE
 *
 * This addresses deficiency #1: "They don't test masked attention at all"
 * This addresses deficiency #2: "They don't compare against a reference"
 * This addresses deficiency #3: "They only do sanity checks"
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttentionWithMaskAndReference) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Scaled Dot-Product Attention WITH MASK (reference comparison)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // Create attention mask [B, 1, 1, S] - mask out last 8 positions
    std::vector<float> mask_data(batch_size * 1 * 1 * seq_len, 1.0F);
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = seq_len - 8; s < seq_len; ++s) {
            mask_data[b * seq_len + s] = 0.0F;  // 0 = mask out
        }
    }

    // Compute reference output WITH MASK
    auto expected = reference_attention(q_data, k_data, v_data, batch_size, num_heads, seq_len, head_dim, &mask_data);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto mask_tensor =
        core::from_vector(mask_data, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);
    auto mask = autograd::create_tensor(mask_tensor);

    // Call TTML scaled_dot_product_attention WITH MASK
    auto output = ops::scaled_dot_product_attention(q, k, v, mask);

    // Get actual output
    auto actual = core::to_vector(output->get_value());

    // Compare with reference using PCC
    // THIS IS THE CRITICAL TEST - it will FAIL if masking is broken!
    print_comparison("Attention (WITH MASK)", expected, actual, 0.99F);
}

/**
 * Test that all-ones mask produces same result as no mask
 *
 * This validates mask=1 means "attend" (not "mask out")
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttentionAllOnesMask) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Attention with all-ones mask (should match no-mask)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // All-ones mask (attend to everything)
    std::vector<float> mask_data(batch_size * 1 * 1 * seq_len, 1.0F);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto mask_tensor =
        core::from_vector(mask_data, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);
    auto mask = autograd::create_tensor(mask_tensor);

    // Run with no mask
    auto output_no_mask = ops::scaled_dot_product_attention(q, k, v);
    auto no_mask_result = core::to_vector(output_no_mask->get_value());

    // Run with all-ones mask
    auto output_with_mask = ops::scaled_dot_product_attention(q, k, v, mask);
    auto with_mask_result = core::to_vector(output_with_mask->get_value());

    // They should be identical (or very close)
    print_comparison("All-ones mask vs no-mask", no_mask_result, with_mask_result, 0.999F);
}

/**
 * Test that all-zeros mask produces valid output
 *
 * When everything is masked, softmax should produce uniform weights
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttentionAllZerosMask) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Attention with all-zeros mask\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // All-zeros mask (mask out everything)
    std::vector<float> mask_data(batch_size * 1 * 1 * seq_len, 0.0F);

    // Compute reference
    auto expected = reference_attention(q_data, k_data, v_data, batch_size, num_heads, seq_len, head_dim, &mask_data);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto mask_tensor =
        core::from_vector(mask_data, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);
    auto mask = autograd::create_tensor(mask_tensor);

    // Run with all-zeros mask
    auto output = ops::scaled_dot_product_attention(q, k, v, mask);
    auto actual = core::to_vector(output->get_value());

    // Compare with reference
    print_comparison("All-zeros mask", expected, actual, 0.99F);
}

/**
 * Test partial mask (realistic BERT scenario)
 *
 * This simulates padding mask where some sequence positions are masked
 */
TEST_F(BERTOperatorTest, ScaledDotProductAttentionPartialMask) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: Attention with partial mask (realistic padding scenario)\n";
    std::cout << std::string(80, '=') << "\n";

    // Create random Q, K, V [B, H, S, E/H]
    auto q_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 42);
    auto k_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 43);
    auto v_data = create_random_data(batch_size * num_heads * seq_len * head_dim, 0.0F, 1.0F, 44);

    // Partial mask: first 24 positions = attend (1), last 8 positions = masked (0)
    std::vector<float> mask_data(batch_size * 1 * 1 * seq_len);
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            mask_data[b * seq_len + s] = (s < 24) ? 1.0F : 0.0F;
        }
    }

    // Compute reference
    auto expected = reference_attention(q_data, k_data, v_data, batch_size, num_heads, seq_len, head_dim, &mask_data);

    // Create TTML tensors
    auto q_tensor = core::from_vector(
        q_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto k_tensor = core::from_vector(
        k_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto v_tensor = core::from_vector(
        v_data, ttnn::Shape({batch_size, num_heads, seq_len, head_dim}), &autograd::ctx().get_device());
    auto mask_tensor =
        core::from_vector(mask_data, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());

    auto q = autograd::create_tensor(q_tensor);
    auto k = autograd::create_tensor(k_tensor);
    auto v = autograd::create_tensor(v_tensor);
    auto mask = autograd::create_tensor(mask_tensor);

    // Run with partial mask
    auto output = ops::scaled_dot_product_attention(q, k, v, mask);
    auto actual = core::to_vector(output->get_value());

    // Compare with reference - THIS IS THE MOST IMPORTANT TEST
    // It simulates real BERT usage with padding masks
    print_comparison("Partial mask (padding simulation)", expected, actual, 0.99F);
}
