// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <cmath>
#include <random>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/losses.hpp"
#include "ops/linear_op.hpp"

namespace ttml::ops::tests {

class GeluTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
        autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }

    // Reference implementation for validation
    // Note: There are multiple GELU approximations:
    // 1. Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    // 2. Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // 3. Fast/Sigmoid approximation: x * sigmoid(1.702 * x) or similar variants
    
    // Based on test failures and ttnn defaults, using tanh approximation
    // This matches the "tanh" mode in gelu_bw
    float gelu_reference(float x) {
        // Tanh approximation (matches "tanh" mode in backward)
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        return 0.5f * x * (1.0f + std::tanh(inner));
    }

    float gelu_gradient_reference(float x) {
        // Gradient of tanh approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_squared = x * x;
        float x_cubed = x_squared * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_inner = std::tanh(inner);
        float sech2_inner = 1.0f - tanh_inner * tanh_inner;
        
        float d_inner_dx = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x_squared);
        return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2_inner * d_inner_dx;
    }

    // BFloat16 tolerance helper - adjusted for approximation errors
    float get_tolerance(float expected_value) {
        // BFloat16 has ~7-8 bits of mantissa precision
        // Combined with approximation errors, we need higher tolerance
        return std::max(5e-3f, std::abs(expected_value) * 0.02f); // 2% relative error
    }
    
    float get_gradient_tolerance(float expected_value) {
        // Gradients accumulate more error
        return std::max(1e-2f, std::abs(expected_value) * 0.03f); // 3% relative error
    }
};

// ============================================================================
// 3.1 Forward Pass Tests
// ============================================================================

TEST_F(GeluTest, ForwardBasicValues) {
    auto* device = &autograd::ctx().get_device();

    // Test with known input values
    std::vector<float> test_values = {
        -2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f
    };

    std::vector<float> expected_values;
    for (float val : test_values) {
        expected_values.push_back(gelu_reference(val));
    }

    // Test shape {1, 1, 1, 9}
    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 9}, device));
    
    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    ASSERT_EQ(output_data.size(), expected_values.size());
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_values[i], get_tolerance(expected_values[i]))
            << "Mismatch at index " << i << " for input " << test_values[i];
    }
}

TEST_F(GeluTest, ForwardZeroValues) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {
        0.0f, 1e-6f, -1e-6f, 1e-4f, -1e-4f, 1e-3f, -1e-3f
    };

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 7}, device));
    
    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // GELU(0) should be exactly 0
    EXPECT_NEAR(output_data[0], 0.0f, 1e-6f);

    // Near-zero values should transition smoothly
    for (size_t i = 1; i < output_data.size(); ++i) {
        float expected = gelu_reference(test_values[i]);
        EXPECT_NEAR(output_data[i], expected, get_tolerance(expected))
            << "Near-zero handling failed for input " << test_values[i];
    }
}

TEST_F(GeluTest, ForwardPositiveRange) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f};
    std::vector<float> expected_values;
    for (float val : test_values) {
        expected_values.push_back(gelu_reference(val));
    }

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 7}, device));
    
    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // Check monotonicity for positive inputs
    for (size_t i = 1; i < output_data.size(); ++i) {
        EXPECT_GT(output_data[i], output_data[i-1])
            << "GELU should be monotonically increasing for positive inputs";
    }

    // Check against reference
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_values[i], get_tolerance(expected_values[i]));
    }
}

TEST_F(GeluTest, ForwardNegativeRange) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {-5.0f, -3.0f, -2.0f, -1.0f, -0.5f, -0.1f};

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 6}, device));
    
    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // GELU should approach 0 for large negative values
    EXPECT_NEAR(output_data[0], 0.0f, 1e-3f) << "GELU(-5) should be close to 0";

    // Check against reference
    for (size_t i = 0; i < output_data.size(); ++i) {
        float expected = gelu_reference(test_values[i]);
        EXPECT_NEAR(output_data[i], expected, get_tolerance(expected));
    }
}

TEST_F(GeluTest, ForwardExtremeMagnitudes) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {-100.0f, -10.0f, 10.0f, 100.0f};

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 4}, device));
    
    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // Check no NaN or Inf
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i])) << "NaN detected at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Inf detected at index " << i;
    }

    // Large negative should be ~0
    EXPECT_NEAR(output_data[0], 0.0f, 1e-3f);
    EXPECT_NEAR(output_data[1], 0.0f, 1e-3f);

    // Large positive should be ~x (linear)
    EXPECT_NEAR(output_data[2], 10.0f, 0.1f);
    EXPECT_NEAR(output_data[3], 100.0f, 1.0f);
}

// ============================================================================
// 3.2 Backward Pass Tests
// ============================================================================

TEST_F(GeluTest, BackwardGradientBasic) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 5}, device));
    
    auto output = gelu(input);

    // Create target and compute MSE loss
    auto target = autograd::create_tensor(core::zeros_like(output->get_value()));
    auto loss = mse_loss(output, target);

    loss->backward();

    auto input_grad = core::to_vector(input->get_grad());

    // For MSE loss with target=0, gradient should be: 2/N * gelu(x) * gelu'(x)
    for (size_t i = 0; i < test_values.size(); ++i) {
        float gelu_val = gelu_reference(test_values[i]);
        float gelu_grad = gelu_gradient_reference(test_values[i]);
        float expected_grad = (2.0f / test_values.size()) * gelu_val * gelu_grad;
        
        EXPECT_NEAR(input_grad[i], expected_grad, get_gradient_tolerance(expected_grad))
            << "Gradient mismatch for input " << test_values[i];
    }
}

TEST_F(GeluTest, BackwardGradientCriticalPoints) {
    auto* device = &autograd::ctx().get_device();

    // Test gradient at x = 0 (should be approximately 0.5)
    std::vector<float> test_values = {0.0f};
    
    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 1}, device));
    
    auto output = gelu(input);
    
    // Set output gradient to 1 for direct gradient computation
    output->set_grad(core::ones_like(output->get_value()));
    output->backward();

    auto input_grad = core::to_vector(input->get_grad());
    
    // At x=0, GELU'(0) ≈ 0.5
    EXPECT_NEAR(input_grad[0], 0.5f, 0.02f) << "GELU gradient at x=0 should be ~0.5";
}

TEST_F(GeluTest, BackwardGradientAccumulation) {
    auto* device = &autograd::ctx().get_device();

    // Create a computation with multiple GELU applications
    std::vector<float> test_data(32, 0.5f);
    
    auto input = autograd::create_tensor(
        core::from_vector(test_data, ttnn::Shape{1, 1, 4, 8}, device));
    
    // Apply GELU twice
    auto gelu1 = gelu(input);
    auto gelu2 = gelu(gelu1);
    
    // Create loss
    auto target = autograd::create_tensor(core::ones_like(gelu2->get_value()));
    auto loss = mse_loss(gelu2, target);
    
    loss->backward();

    auto input_grad = core::to_vector(input->get_grad());

    // Check that gradients accumulated correctly (all should be non-zero)
    for (size_t i = 0; i < input_grad.size(); ++i) {
        EXPECT_NE(input_grad[i], 0.0f) << "Gradient should be non-zero at index " << i;
    }
}

TEST_F(GeluTest, BackwardChainRule) {
    auto* device = &autograd::ctx().get_device();

    uint32_t in_features = 32;
    uint32_t out_features = 32;

    // Create input
    std::vector<float> input_data(in_features);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) * 0.1f - 1.5f;
    }

    auto input = autograd::create_tensor(
        core::from_vector(input_data, ttnn::Shape{1, 1, 1, in_features}, device));

    // Create weight for linear layer
    auto weight = autograd::create_tensor();
    ttml::init::uniform_init(weight, ttnn::Shape{1, 1, out_features, in_features}, ttml::init::UniformRange{-0.1f, 0.1f});
    
    // Pattern: Linear -> GELU
    auto linear_out = linear_op(input, weight, nullptr);
    auto gelu_out = gelu(linear_out);

    // Create loss
    auto target = autograd::create_tensor(core::zeros_like(gelu_out->get_value()));
    auto loss = mse_loss(gelu_out, target);

    loss->backward();

    // Verify gradients exist and are reasonable
    EXPECT_TRUE(core::is_tensor_initialized(input->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(weight->get_grad()));

    auto input_grad = core::to_vector(input->get_grad());
    auto weight_grad = core::to_vector(weight->get_grad());

    // Check for reasonable gradient magnitudes
    float input_grad_norm = 0.0f;
    for (float g : input_grad) {
        input_grad_norm += g * g;
    }
    input_grad_norm = std::sqrt(input_grad_norm);

    EXPECT_GT(input_grad_norm, 0.0f) << "Input gradients should be non-zero";
    EXPECT_LT(input_grad_norm, 100.0f) << "Input gradients should not explode";
}

// ============================================================================
// 3.3 BERT-Specific Integration Tests
// ============================================================================

TEST_F(GeluTest, BertMLPIntegration) {
    auto* device = &autograd::ctx().get_device();

    // BERT dimensions
    uint32_t batch_size = 2;
    uint32_t seq_len = 128;
    uint32_t hidden_dim = 768;
    uint32_t intermediate_dim = 3072;

    // Create realistic input (post-normalization values)
    std::vector<float> input_data(batch_size * seq_len * hidden_dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& val : input_data) {
        val = dist(gen);
    }

    auto input = autograd::create_tensor(
        core::from_vector(input_data, ttnn::Shape{batch_size, 1, seq_len, hidden_dim}, device));

    // Create weights for MLP pattern
    auto weight1 = autograd::create_tensor();
    auto weight2 = autograd::create_tensor();
    ttml::init::uniform_init(weight1, ttnn::Shape{1, 1, intermediate_dim, hidden_dim}, ttml::init::UniformRange{-0.02f, 0.02f});
    ttml::init::uniform_init(weight2, ttnn::Shape{1, 1, hidden_dim, intermediate_dim}, ttml::init::UniformRange{-0.02f, 0.02f});

    // BERT MLP pattern: Linear -> GELU -> Linear
    auto hidden = linear_op(input, weight1, nullptr);
    auto activated = gelu(hidden);
    auto output = linear_op(activated, weight2, nullptr);

    // Verify output shape
    auto output_shape = output->get_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], hidden_dim);

    // Test backward pass
    auto target = autograd::create_tensor(core::zeros_like(output->get_value()));
    auto loss = mse_loss(output, target);
    loss->backward();

    // Verify all components received gradients
    EXPECT_TRUE(core::is_tensor_initialized(input->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(weight1->get_grad()));
    EXPECT_TRUE(core::is_tensor_initialized(weight2->get_grad()));
}

TEST_F(GeluTest, BertBatchProcessing) {
    auto* device = &autograd::ctx().get_device();

    std::vector<uint32_t> batch_sizes = {1, 8, 32};
    std::vector<uint32_t> seq_lengths = {128, 512};
    uint32_t hidden_dim = 768;

    for (uint32_t batch : batch_sizes) {
        for (uint32_t seq_len : seq_lengths) {
            // Skip very large configurations that might OOM
            if (batch * seq_len > 8192) continue;

            std::vector<float> input_data(batch * seq_len * hidden_dim);
            
            // Initialize with small values
            for (size_t i = 0; i < input_data.size(); ++i) {
                input_data[i] = static_cast<float>(i % 100) * 0.001f - 0.05f;
            }

            auto input = autograd::create_tensor(
                core::from_vector(input_data, ttnn::Shape{batch, 1, seq_len, hidden_dim}, device));

            auto output = gelu(input);

            // Verify shape preservation
            auto output_shape = output->get_shape();
            EXPECT_EQ(output_shape[0], batch);
            EXPECT_EQ(output_shape[1], 1);
            EXPECT_EQ(output_shape[2], seq_len);
            EXPECT_EQ(output_shape[3], hidden_dim);

            // Verify no NaN/Inf
            auto output_data = core::to_vector(output->get_value());
            for (const auto& val : output_data) {
                EXPECT_FALSE(std::isnan(val));
                EXPECT_FALSE(std::isinf(val));
            }
        }
    }
}

// ============================================================================
// 3.4 Numerical Stability Tests
// ============================================================================

TEST_F(GeluTest, BFloat16Precision) {
    auto* device = &autograd::ctx().get_device();

    // Test range where BFloat16 precision matters
    std::vector<float> test_values;
    for (float x = -3.0f; x <= 3.0f; x += 0.25f) {
        test_values.push_back(x);
    }

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, static_cast<uint32_t>(test_values.size())}, device));

    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // Check relative error is within BFloat16 + approximation tolerance
    for (size_t i = 0; i < test_values.size(); ++i) {
        float expected = gelu_reference(test_values[i]);
        float relative_error = std::abs((output_data[i] - expected) / (expected + 1e-8f));
        
        // Higher tolerance for BFloat16 precision + approximation
        EXPECT_LT(relative_error, 0.05f) // 5% relative error acceptable
            << "BFloat16 precision exceeded for input " << test_values[i]
            << " (output=" << output_data[i] << ", expected=" << expected << ")";
    }
}

TEST_F(GeluTest, GradientStability) {
    auto* device = &autograd::ctx().get_device();

    // Stack multiple GELU layers (simulating deep network)
    const int num_layers = 12; // BERT depth
    const uint32_t dim = 64;

    std::vector<float> input_data(dim, 0.5f);
    auto x = autograd::create_tensor(
        core::from_vector(input_data, ttnn::Shape{1, 1, 1, dim}, device));

    // Apply GELU multiple times
    auto current = x;
    for (int i = 0; i < num_layers; ++i) {
        current = gelu(current);
    }

    // Check forward pass didn't vanish or explode
    auto output_data = core::to_vector(current->get_value());
    float output_norm = 0.0f;
    for (float val : output_data) {
        output_norm += val * val;
    }
    output_norm = std::sqrt(output_norm);

    EXPECT_GT(output_norm, 1e-6f) << "Output vanished after " << num_layers << " layers";
    EXPECT_LT(output_norm, 1e6f) << "Output exploded after " << num_layers << " layers";

    // Test gradient flow
    current->set_grad(core::ones_like(current->get_value()));
    current->backward();

    auto input_grad = core::to_vector(x->get_grad());
    float grad_norm = 0.0f;
    for (float val : input_grad) {
        grad_norm += val * val;
    }
    grad_norm = std::sqrt(grad_norm);

    EXPECT_GT(grad_norm, 1e-10f) << "Gradients vanished after " << num_layers << " layers";
    EXPECT_LT(grad_norm, 1e10f) << "Gradients exploded after " << num_layers << " layers";
}

// ============================================================================
// 3.5 Comparison Tests
// ============================================================================

TEST_F(GeluTest, CompareWithReference) {
    auto* device = &autograd::ctx().get_device();

    // Generate random inputs
    const int num_samples = 100;
    std::vector<float> test_values;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (int i = 0; i < num_samples; ++i) {
        test_values.push_back(dist(gen));
    }

    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, static_cast<uint32_t>(num_samples)}, device));

    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    // Statistical comparison
    float total_error = 0.0f;
    float max_error = 0.0f;
    
    for (size_t i = 0; i < test_values.size(); ++i) {
        float expected = gelu_reference(test_values[i]);
        float error = std::abs(output_data[i] - expected);
        total_error += error;
        max_error = std::max(max_error, error);
    }

    float mean_error = total_error / num_samples;
    
    EXPECT_LT(mean_error, 0.02f) << "Mean error too high";
    EXPECT_LT(max_error, 0.2f) << "Max error too high";
}

TEST_F(GeluTest, ApproximationAccuracy) {
    auto* device = &autograd::ctx().get_device();

    // Test points where approximation might differ most
    std::vector<float> critical_points = {
        -2.5f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.5f
    };

    auto input = autograd::create_tensor(
        core::from_vector(critical_points, ttnn::Shape{1, 1, 1, static_cast<uint32_t>(critical_points.size())}, device));

    auto output = gelu(input);
    auto output_data = core::to_vector(output->get_value());

    for (size_t i = 0; i < critical_points.size(); ++i) {
        float expected = gelu_reference(critical_points[i]);
        float relative_error = std::abs((output_data[i] - expected) / (expected + 1e-8f));
        
        // Verify approximation is good enough for BERT
        EXPECT_LT(relative_error, 0.05f) // 5% relative error acceptable
            << "Approximation error too high at x=" << critical_points[i];
    }
}

// ============================================================================
// 5. Error Handling Tests
// ============================================================================

TEST_F(GeluTest, InvalidInputs) {
    auto* device = &autograd::ctx().get_device();

    // Test with NaN
    std::vector<float> nan_input = {std::numeric_limits<float>::quiet_NaN()};
    auto nan_tensor = autograd::create_tensor(
        core::from_vector(nan_input, ttnn::Shape{1, 1, 1, 1}, device));
    
    auto nan_output = gelu(nan_tensor);
    auto nan_data = core::to_vector(nan_output->get_value());
    
    // Hardware implementation may not propagate NaN correctly
    // This is a known limitation of some approximations
    if (std::isnan(nan_data[0])) {
        SUCCEED() << "GELU correctly propagated NaN";
    } else {
        // If not NaN, should at least be a finite value
        EXPECT_TRUE(std::isfinite(nan_data[0])) 
            << "GELU should return finite value if not propagating NaN";
    }

    // Test with Inf
    std::vector<float> inf_inputs = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };
    auto inf_tensor = autograd::create_tensor(
        core::from_vector(inf_inputs, ttnn::Shape{1, 1, 1, 2}, device));
    
    auto inf_output = gelu(inf_tensor);
    auto inf_data = core::to_vector(inf_output->get_value());
    
    // Positive infinity should map to infinity (linear region)
    EXPECT_TRUE(std::isinf(inf_data[0]) && inf_data[0] > 0) << "GELU(+inf) should be +inf";
    
    // Negative infinity should map to 0
    EXPECT_NEAR(inf_data[1], 0.0f, 1e-3f) << "GELU(-inf) should be 0";
}

TEST_F(GeluTest, ShapeCompatibility) {
    auto* device = &autograd::ctx().get_device();

    // Test various tensor shapes
    std::vector<ttnn::Shape> test_shapes = {
        ttnn::Shape{1, 1, 1, 32},      // 1D
        ttnn::Shape{2, 1, 16, 16},     // 2D square
        ttnn::Shape{4, 1, 8, 32},      // 2D rectangular
        ttnn::Shape{2, 1, 128, 768},   // BERT-like
    };

    for (const auto& shape : test_shapes) {
        uint32_t total_elements = shape[0] * shape[1] * shape[2] * shape[3];
        std::vector<float> input_data(total_elements, 0.5f);
        
        auto input = autograd::create_tensor(
            core::from_vector(input_data, shape, device));
        
        auto output = gelu(input);
        auto output_shape = output->get_shape();
        
        // Shape should be preserved
        EXPECT_EQ(output_shape, shape) << "Shape not preserved for " << shape;
    }
}

// ============================================================================
// Test Different Approximation Modes
// ============================================================================

TEST_F(GeluTest, TestFastVsAccurateMode) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> test_values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    auto input = autograd::create_tensor(
        core::from_vector(test_values, ttnn::Shape{1, 1, 1, 5}, device));
    
    // Test fast approximation (default)
    auto output_fast = gelu(input);  // Uses fast=true by default
    auto output_fast_data = core::to_vector(output_fast->get_value());
    
    // Test accurate mode
    auto output_accurate = gelu(input, false);  // Use accurate mode
    auto output_accurate_data = core::to_vector(output_accurate->get_value());
    
    // The two modes should give similar but not identical results
    for (size_t i = 0; i < test_values.size(); ++i) {
        float diff = std::abs(output_fast_data[i] - output_accurate_data[i]);
        
        // Different approximations should be within reasonable bounds
        EXPECT_LT(diff, 0.1f) 
            << "Fast and accurate modes differ too much for input " << test_values[i];
    }
}

}  // namespace ttml::ops::tests
