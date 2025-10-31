// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/binary_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

class BinaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// ============================================================================
// ADD OPERATION TESTS
// ============================================================================

TEST_F(BinaryOpsTest, AddTwoTensors) {
    auto* device = &autograd::ctx().get_device();

    // Test basic addition of two tensors
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = add(a, b);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};

    ASSERT_EQ(result_data.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, AddBackward) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {0.5f, 1.5f, 2.5f, 3.5f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = add(a, b);

    // Create target and compute loss for backward
    auto target = autograd::create_tensor(
        core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    loss->backward();

    // Check gradients
    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // For MSE loss with target=0, gradient should be 2/N * result_value
    // Since addition has gradient 1 for both inputs, both should get same gradient
    for (size_t i = 0; i < grad_a.size(); ++i) {
        float expected_grad = (2.0f / 4.0f) * (data_a[i] + data_b[i]);
        EXPECT_NEAR(grad_a[i], expected_grad, 1e-2f);
        EXPECT_NEAR(grad_b[i], expected_grad, 1e-2f);
    }
}

TEST_F(BinaryOpsTest, AddWithBroadcast) {
    auto* device = &autograd::ctx().get_device();

    // Test broadcasting: [2, 1, 2, 2] + [1, 1, 1, 2]
    std::vector<float> data_a(8);
    std::iota(data_a.begin(), data_a.end(), 1.0f);
    std::vector<float> data_b = {10.0f, 20.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{2, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 1, 2}, device));

    auto result = add(a, b);

    auto result_shape = result->get_shape();
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 1);
    EXPECT_EQ(result_shape[2], 2);
    EXPECT_EQ(result_shape[3], 2);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {
        11.0f, 22.0f, 13.0f, 24.0f,  // batch 0
        15.0f, 26.0f, 17.0f, 28.0f   // batch 1
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, AddWithAutocastTensor) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 5.0f, 5.0f, 5.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b_tensor = core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device);
    autograd::AutocastTensor b(b_tensor);

    auto result = add(a, b);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

// ============================================================================
// SUBTRACT OPERATION TESTS
// ============================================================================

TEST_F(BinaryOpsTest, SubtractTwoTensors) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = sub(a, b);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {9.0f, 18.0f, 27.0f, 36.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, SubtractBackward) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = sub(a, b);
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    loss->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Gradient of subtraction: da = 1 * upstream, db = -1 * upstream
    for (size_t i = 0; i < grad_a.size(); ++i) {
        float diff = data_a[i] - data_b[i];
        float upstream_grad = (2.0f / 4.0f) * diff;
        EXPECT_NEAR(grad_a[i], upstream_grad, 1e-2f);
        EXPECT_NEAR(grad_b[i], -upstream_grad, 1e-2f);
    }
}

TEST_F(BinaryOpsTest, SubtractWithBroadcast) {
    // Broadcasting is now supported in subtraction
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> data_b = {5.0f};  // Broadcast scalar

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 1, 1}, device));

    auto result = sub(a, b);
    auto result_data = core::to_vector(result->get_value());

    std::vector<float> expected = {5.0f, 15.0f, 25.0f, 35.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, SubtractWithBroadcastBackward) {
    auto* device = &autograd::ctx().get_device();

    // Test gradient computation with broadcasting
    std::vector<float> data_a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> data_b = {5.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 1, 1}, device));

    auto result = sub(a, b);

    // Create a gradient for backward pass
    result->set_grad(core::ones_like(result->get_value()));
    result->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Gradient for a should be [1, 1, 1, 1]
    for (float val : grad_a) {
        EXPECT_NEAR(val, 1.0f, 1e-3f);
    }

    // Gradient for b should be sum of negative gradients = -4
    EXPECT_NEAR(grad_b[0], -4.0f, 1e-3f);
}

// ============================================================================
// MULTIPLY OPERATION TESTS
// ============================================================================

TEST_F(BinaryOpsTest, MultiplyTwoTensors) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data_b = {0.5f, 2.0f, 1.5f, 2.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = mul(a, b);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {1.0f, 6.0f, 6.0f, 10.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, MultiplyByScalar) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.5f;

    auto a = autograd::create_tensor(
        core::from_vector(data, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = mul(a, scalar);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {2.5f, 5.0f, 7.5f, 10.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, MultiplyBackward) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data_b = {1.0f, 2.0f, 0.5f, 2.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = mul(a, b);
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    loss->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Gradient of multiplication: da = b * upstream, db = a * upstream
    for (size_t i = 0; i < grad_a.size(); ++i) {
        float product = data_a[i] * data_b[i];
        float upstream_grad = (2.0f / 4.0f) * product;
        EXPECT_NEAR(grad_a[i], data_b[i] * upstream_grad, 1e-2f);
        EXPECT_NEAR(grad_b[i], data_a[i] * upstream_grad, 1e-2f);
    }
}

TEST_F(BinaryOpsTest, MultiplyWithBroadcast) {
    // Broadcasting is now supported in multiplication
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data_b = {2.0f};  // Broadcast scalar

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 1, 1}, device));

    auto result = mul(a, b);
    auto result_data = core::to_vector(result->get_value());

    std::vector<float> expected = {4.0f, 6.0f, 8.0f, 10.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, MultiplyWithBroadcastBackward) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data_b = {2.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 1, 1}, device));

    auto result = mul(a, b);

    // Create a gradient for backward pass
    result->set_grad(core::ones_like(result->get_value()));
    result->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Gradient for a should be [2, 2, 2, 2] (broadcasted b value)
    for (float val : grad_a) {
        EXPECT_NEAR(val, 2.0f, 1e-3f);
    }

    // Gradient for b should be sum of a values = 14
    EXPECT_NEAR(grad_b[0], 14.0f, 1e-3f);
}

// ============================================================================
// DIVIDE OPERATION TESTS
// ============================================================================

TEST_F(BinaryOpsTest, DivideTwoTensors) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> data_b = {2.0f, 4.0f, 5.0f, 8.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = div(a, b);

    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected = {5.0f, 5.0f, 6.0f, 5.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, DivideBackward) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {8.0f, 12.0f, 6.0f, 9.0f};
    std::vector<float> data_b = {2.0f, 3.0f, 2.0f, 3.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = div(a, b);
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    loss->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Check gradients exist and are reasonable
    for (size_t i = 0; i < grad_a.size(); ++i) {
        EXPECT_FALSE(std::isnan(grad_a[i]));
        EXPECT_FALSE(std::isnan(grad_b[i]));
    }
}

TEST_F(BinaryOpsTest, DivideBySmallNumber) {
    // Test division by very small numbers (near zero but not zero)
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {1e-5f, 1e-5f, 1e-5f, 1e-5f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto result = div(a, b);
    auto result_data = core::to_vector(result->get_value());

    // Results should be large but finite
    for (float val : result_data) {
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_GT(std::abs(val), 1e4f);  // Should be large
    }
}

// ============================================================================
// GRADIENT ACCUMULATION TEST
// ============================================================================

TEST_F(BinaryOpsTest, GradientAccumulation) {
    // Test that gradients accumulate correctly when a tensor appears multiple times
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));

    // Use 'a' multiple times in the computation: a + a * a
    auto a_squared = mul(a, a);
    auto result = add(a, a_squared);

    // Create target and compute loss
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    loss->backward();

    auto grad_a = core::to_vector(a->get_grad());

    // Gradient of a + a*a with respect to a is: 1 + 2*a
    // With MSE loss, the upstream gradient is 2/N * (a + a*a)
    for (size_t i = 0; i < grad_a.size(); ++i) {
        float expected_grad_component = 1.0f + 2.0f * data_a[i];
        float result_value = data_a[i] + data_a[i] * data_a[i];
        float expected_grad = (2.0f / 4.0f) * result_value * expected_grad_component;
        EXPECT_NEAR(grad_a[i], expected_grad, 1e-2f);
    }
}

TEST_F(BinaryOpsTest, ComplexExpressionGradient) {
    // Test a more complex expression: (a + b) * (a - b)
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    auto sum = add(a, b);
    auto diff = sub(a, b);
    auto result = mul(sum, diff);  // (a+b)*(a-b) = a^2 - b^2

    // Set gradient and backward
    result->set_grad(core::ones_like(result->get_value()));
    result->backward();

    auto grad_a = core::to_vector(a->get_grad());
    auto grad_b = core::to_vector(b->get_grad());

    // Gradient of a^2 - b^2 w.r.t. a is 2*a
    // Gradient of a^2 - b^2 w.r.t. b is -2*b
    for (size_t i = 0; i < grad_a.size(); ++i) {
        EXPECT_NEAR(grad_a[i], 2.0f * data_a[i], 1e-2f);
        EXPECT_NEAR(grad_b[i], -2.0f * data_b[i], 1e-2f);
    }
}

// ============================================================================
// BERT-SPECIFIC USE CASES
// ============================================================================

TEST_F(BinaryOpsTest, BERTResidualConnection) {
    // Test the residual connection pattern used in BERT
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 128;
    uint32_t embedding_dim = 768;

    // Create input and residual tensors
    std::vector<float> input_data(batch * seq_len * embedding_dim);
    std::vector<float> residual_data(batch * seq_len * embedding_dim);

    // Initialize with different patterns
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 100) * 0.01f;
        residual_data[i] = static_cast<float>((i + 50) % 100) * 0.01f;
    }

    auto input = autograd::create_tensor(
        core::from_vector(input_data, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device));
    auto residual = autograd::create_tensor(
        core::from_vector(residual_data, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device));

    // Perform residual addition
    auto output = add(input, residual);

    // Check shape preservation
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], embedding_dim);

    // Spot check some values
    auto output_data = core::to_vector(output->get_value());
    for (size_t i = 0; i < 10; ++i) {
        float expected = input_data[i] + residual_data[i];
        EXPECT_NEAR(output_data[i], expected, 1e-2f);
    }
}

TEST_F(BinaryOpsTest, BERTEmbeddingCombination) {
    // Test combining token, position, and token type embeddings as in BERT
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 64;

    // Create three types of embeddings
    std::vector<float> token_emb(batch * seq_len * embedding_dim);
    std::vector<float> pos_emb(batch * seq_len * embedding_dim);
    std::vector<float> type_emb(batch * seq_len * embedding_dim);

    // Initialize with distinct patterns
    for (size_t i = 0; i < token_emb.size(); ++i) {
        token_emb[i] = static_cast<float>(i % 10) * 0.1f;
        pos_emb[i] = static_cast<float>(i % 5) * 0.05f;
        type_emb[i] = static_cast<float>(i % 2) * 0.02f;
    }

    auto token_tensor = autograd::create_tensor(
        core::from_vector(token_emb, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device));
    auto pos_tensor = autograd::create_tensor(
        core::from_vector(pos_emb, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device));
    auto type_tensor = autograd::create_tensor(
        core::from_vector(type_emb, ttnn::Shape{batch, 1, seq_len, embedding_dim}, device));

    // Combine embeddings as in BERT
    auto combined = add(token_tensor, pos_tensor);
    combined = add(combined, type_tensor);

    // Test backward pass
    auto target = autograd::create_tensor(core::zeros_like(combined->get_value()));
    auto loss = mse_loss(combined, target);
    loss->backward();

    // Check all three tensors received gradients
    EXPECT_TRUE(core::is_tensor_initialized(token_tensor->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(pos_tensor->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(type_tensor->get_grad()));

    auto token_grad = core::to_vector(token_tensor->get_grad());
    auto pos_grad = core::to_vector(pos_tensor->get_grad());
    auto type_grad = core::to_vector(type_tensor->get_grad());

    // Gradients should be equal for addition
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(token_grad[i], pos_grad[i], 1e-3f);
        EXPECT_NEAR(token_grad[i], type_grad[i], 1e-3f);
    }
}

TEST_F(BinaryOpsTest, BERTAttentionMaskProcessing) {
    // Test the attention mask processing pattern in BERT
    auto* device = &autograd::ctx().get_device();

    uint32_t batch = 2;
    uint32_t seq_len = 32;

    // Create attention mask (1 for real tokens, 0 for padding)
    std::vector<float> mask_data(batch * seq_len);
    for (size_t i = 0; i < mask_data.size(); ++i) {
        // Make first 20 tokens real, rest padding
        mask_data[i] = (i % seq_len) < 20 ? 1.0f : 0.0f;
    }

    auto mask = autograd::create_tensor(
        core::from_vector(mask_data, ttnn::Shape{batch, 1, 1, seq_len}, device));

    // BERT mask processing: (1 - mask) * -10000
    auto ones = autograd::create_tensor(
        core::ones(mask->get_shape(), device));
    auto inverted_mask = sub(ones, mask);
    auto processed_mask = mul(inverted_mask, -10000.0f);

    auto result_data = core::to_vector(processed_mask->get_value());

    // Check that real tokens get 0 and padding gets -10000
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t idx = b * seq_len + s;
            if (s < 20) {
                EXPECT_NEAR(result_data[idx], 0.0f, 1e-3f);
            } else {
                // BFloat16 precision can cause -10000 to be represented as values like -9984
                // Allow 0.2% tolerance (20.0f) for this large negative value
                EXPECT_NEAR(result_data[idx], -10000.0f, 20.0f);
            }
        }
    }
}

TEST_F(BinaryOpsTest, OperatorOverloads) {
    // Test that operator overloads work correctly
    auto* device = &autograd::ctx().get_device();

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};

    auto a = autograd::create_tensor(
        core::from_vector(data_a, ttnn::Shape{1, 1, 2, 2}, device));
    auto b = autograd::create_tensor(
        core::from_vector(data_b, ttnn::Shape{1, 1, 2, 2}, device));

    // Test operator+
    auto sum = a + b;
    auto sum_data = core::to_vector(sum->get_value());
    EXPECT_NEAR(sum_data[0], 6.0f, 1e-3f);

    // Test operator-
    auto diff = b - a;
    auto diff_data = core::to_vector(diff->get_value());
    EXPECT_NEAR(diff_data[0], 4.0f, 1e-3f);

    // Test operator*
    auto prod = a * b;
    auto prod_data = core::to_vector(prod->get_value());
    EXPECT_NEAR(prod_data[0], 5.0f, 1e-3f);

    // Test operator/
    auto quot = b / a;
    auto quot_data = core::to_vector(quot->get_value());
    EXPECT_NEAR(quot_data[0], 5.0f, 1e-3f);

    // Test scalar multiplication
    auto scaled = a * 2.0f;
    auto scaled_data = core::to_vector(scaled->get_value());
    EXPECT_NEAR(scaled_data[0], 2.0f, 1e-3f);
}

}  // namespace ttml::ops::tests
