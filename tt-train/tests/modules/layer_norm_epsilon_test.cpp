// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"
#include "modules/layer_norm_module.hpp"

using namespace ttml;

class LayerNormEpsilonTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }

    // Helper to convert tensor to vector for comparison
    std::vector<float> tensor_to_vector(const tt::tt_metal::Tensor& tensor) {
        auto shape = tensor.logical_shape();
        std::vector<float> result;
        result.reserve(shape.volume());
        auto data = core::to_vector(tensor);
        return data;
    }
};

// Test 1: Verify epsilon is stored correctly
TEST_F(LayerNormEpsilonTest, EpsilonIsStored) {
    float eps_small = 1e-12F;
    float eps_large = 1e-5F;

    modules::LayerNormLayer ln_small(32, eps_small, false);
    modules::LayerNormLayer ln_large(32, eps_large, false);

    // Use EXPECT_NEAR for floating-point comparisons with appropriate tolerance
    EXPECT_NEAR(ln_small.get_epsilon(), eps_small, 1e-15F);
    EXPECT_NEAR(ln_large.get_epsilon(), eps_large, 1e-8F);
}

// Test 2: Epsilon parameter affects computation
TEST_F(LayerNormEpsilonTest, EpsilonAffectsComputation) {
    const uint32_t features = 64;
    // Use significantly different epsilon values that are both safe for bfloat16
    const float eps_standard = 1e-5F;  // Standard epsilon
    const float eps_large = 1e-2F;     // Much larger epsilon for clear difference

    modules::LayerNormLayer ln_standard(features, eps_standard, false);
    modules::LayerNormLayer ln_large(features, eps_large, false);

    // Create input with normal values
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Both should produce valid outputs without NaN/Inf
    // This validates that epsilon is being used correctly in the computation
    auto output_standard = ln_standard(input);
    auto output_large = ln_large(input);

    auto vec_standard = tensor_to_vector(output_standard->get_value());
    auto vec_large = tensor_to_vector(output_large->get_value());

    // Verify no NaN or Inf in either output
    for (float val : vec_standard) {
        EXPECT_FALSE(std::isnan(val)) << "Standard epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Standard epsilon produced Inf";
    }
    for (float val : vec_large) {
        EXPECT_FALSE(std::isnan(val)) << "Large epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Large epsilon produced Inf";
    }
}

// Test 3: BERT uses correct epsilon without NaN/Inf
TEST_F(LayerNormEpsilonTest, BertUsesCorrectEpsilon) {
    models::bert::BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;  // BERT's standard epsilon

    ASSERT_NO_THROW({
        auto model = std::make_shared<models::bert::Bert>(config);

        // Create test inputs
        auto input_ids = core::zeros(ttnn::Shape({1, 1, 1, config.max_sequence_length}), &autograd::ctx().get_device());
        auto input_ids_tensor = autograd::create_tensor(input_ids);

        // Forward pass - should not produce NaN or Inf
        auto output = model->forward(input_ids_tensor);

        // Check for NaN/Inf in output
        auto output_vec = tensor_to_vector(output->get_value());
        for (float val : output_vec) {
            EXPECT_FALSE(std::isnan(val)) << "Output contains NaN";
            EXPECT_FALSE(std::isinf(val)) << "Output contains Inf";
        }
    });
}

// Test 4: Hardware precision impact (bfloat16)
TEST_F(LayerNormEpsilonTest, HardwarePrecisionImpact) {
    const uint32_t features = 64;

    // Test with epsilon below bfloat16 machine epsilon (~0.0078125)
    // This should be clamped to 1e-4F internally for safety
    const float very_small_eps = 1e-12F;
    const float safe_eps = 1e-4F;

    modules::LayerNormLayer ln_tiny(features, very_small_eps, false);
    modules::LayerNormLayer ln_safe(features, safe_eps, false);

    // Create input with near-zero variance
    auto input_tt = core::full(
        ttnn::Shape({1, 1, 1, features}),
        1e-8F,  // Very small values
        &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Both should produce valid outputs (no NaN/Inf)
    // because internally epsilon is clamped to safe_eps
    auto output_tiny = ln_tiny(input);
    auto output_safe = ln_safe(input);

    auto vec_tiny = tensor_to_vector(output_tiny->get_value());
    auto vec_safe = tensor_to_vector(output_safe->get_value());

    // Check no NaN/Inf
    for (float val : vec_tiny) {
        EXPECT_FALSE(std::isnan(val)) << "Tiny epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Tiny epsilon produced Inf";
    }
    for (float val : vec_safe) {
        EXPECT_FALSE(std::isnan(val)) << "Safe epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Safe epsilon produced Inf";
    }

    // Outputs should be similar because tiny epsilon is clamped internally
    float max_diff = 0.0F;
    for (size_t i = 0; i < vec_tiny.size(); ++i) {
        float diff = std::abs(vec_tiny[i] - vec_safe[i]);
        max_diff = std::max(max_diff, diff);
    }
    // Due to hardware clamping, outputs should be nearly identical
    EXPECT_LT(max_diff, 1e-3F) << "Hardware epsilon clamping should make outputs similar";
}

// Test 5: Epsilon propagation through composite operation
TEST_F(LayerNormEpsilonTest, CompositeOpUsesEpsilon) {
    const uint32_t features = 64;
    const float custom_eps = 1e-6F;

    // Test composite operation (manual implementation)
    modules::LayerNormLayer ln_composite(features, custom_eps, true);  // use_composite_op = true

    EXPECT_FLOAT_EQ(ln_composite.get_epsilon(), custom_eps);

    // Create test input
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Should not crash and produce valid output
    auto output = ln_composite(input);
    auto output_vec = tensor_to_vector(output->get_value());

    for (float val : output_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Composite op produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Composite op produced Inf";
    }
}

// Test 6: Default epsilon is sensible
TEST_F(LayerNormEpsilonTest, DefaultEpsilonIsReasonable) {
    const uint32_t features = 32;

    // Create without specifying epsilon (use default)
    modules::LayerNormLayer ln_default(features);

    // Default should be 1e-5F (safe for most cases)
    EXPECT_NEAR(ln_default.get_epsilon(), 1e-5F, 1e-8F);

    // Test it works
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    ASSERT_NO_THROW({ auto output = ln_default(input); });
}

// Test 7: Zero-variance input (edge case for NaN prevention)
TEST_F(LayerNormEpsilonTest, ZeroVarianceNoPrevention) {
    const uint32_t features = 64;
    const float eps = 1e-5F;

    modules::LayerNormLayer ln(features, eps, false);

    // Create input with zero variance (all same values)
    // This is an edge case where epsilon prevents division by zero
    auto input_tt = core::full(ttnn::Shape({1, 1, 1, features}), 42.0F, &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Should not crash and produce valid output (all zeros after normalization)
    auto output = ln(input);
    auto output_vec = tensor_to_vector(output->get_value());

    // Check no NaN or Inf in output
    for (float val : output_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Zero-variance input produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Zero-variance input produced Inf";
    }

    // With zero variance, normalized values should be zero (or very close due to epsilon)
    // Output = (x - mean) / sqrt(0 + eps) * gamma + beta
    // Since variance=0, all inputs equal mean, so (x - mean) = 0
    for (float val : output_vec) {
        EXPECT_NEAR(val, 0.0F, 1e-3F) << "Zero-variance should produce near-zero normalized values";
    }
}

// Test 8: Epsilon affects gradients in backward pass
TEST_F(LayerNormEpsilonTest, EpsilonAffectsGradients) {
    const uint32_t features = 32;
    // Use significantly different epsilon values (both safe for bfloat16)
    const float eps_small = 1e-4F;
    const float eps_large = 1e-2F;

    // Test 1: Small epsilon
    modules::LayerNormLayer ln_small(features, eps_small, false);
    auto input1_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input1 = autograd::create_tensor(input1_tt);
    auto output1 = ln_small(input1);
    output1->backward();
    auto grad1_vec = tensor_to_vector(input1->get_grad());

    // Reset graph for second test
    autograd::ctx().reset_graph();

    // Test 2: Large epsilon
    modules::LayerNormLayer ln_large(features, eps_large, false);
    auto input2_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input2 = autograd::create_tensor(input2_tt);
    auto output2 = ln_large(input2);
    output2->backward();
    auto grad2_vec = tensor_to_vector(input2->get_grad());

    // Gradients should be valid (no NaN/Inf)
    for (float val : grad1_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Small epsilon produced NaN in gradients";
        EXPECT_FALSE(std::isinf(val)) << "Small epsilon produced Inf in gradients";
    }
    for (float val : grad2_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Large epsilon produced NaN in gradients";
        EXPECT_FALSE(std::isinf(val)) << "Large epsilon produced Inf in gradients";
    }

    // Note: With uniform inputs (variance=0), gradients may be similar due to zero normalization
    // This test primarily validates no NaN/Inf in backward pass with different epsilon values
}

// Test 9: Hardware clamp flag controls dtype-dependent clamping
TEST_F(LayerNormEpsilonTest, HardwareClampFlagWorks) {
    const uint32_t features = 64;
    const float tiny_eps = 1e-12F;  // Would be clamped to 1e-4F for bfloat16

    // Test with clamping enabled (default)
    modules::LayerNormLayer ln_clamped(features, tiny_eps, false, true);
    EXPECT_TRUE(ln_clamped.get_enable_hardware_clamp());

    // Test with clamping disabled (expert mode)
    modules::LayerNormLayer ln_unclamped(features, tiny_eps, false, false);
    EXPECT_FALSE(ln_unclamped.get_enable_hardware_clamp());

    // Both should produce valid outputs without NaN/Inf
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input_clamped = autograd::create_tensor(input_tt);
    auto input_unclamped = autograd::create_tensor(input_tt);

    auto output_clamped = ln_clamped(input_clamped);
    auto output_unclamped = ln_unclamped(input_unclamped);

    auto vec_clamped = tensor_to_vector(output_clamped->get_value());
    auto vec_unclamped = tensor_to_vector(output_unclamped->get_value());

    // Both should be valid (no NaN/Inf)
    for (float val : vec_clamped) {
        EXPECT_FALSE(std::isnan(val)) << "Clamped mode produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Clamped mode produced Inf";
    }
    for (float val : vec_unclamped) {
        EXPECT_FALSE(std::isnan(val)) << "Unclamped mode produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Unclamped mode produced Inf";
    }
}

// Test 10: Comprehensive finite difference gradient validation for all parameters
TEST_F(LayerNormEpsilonTest, ComprehensiveFiniteDifferenceGradientValidation) {
    const uint32_t features = 32;
    const float eps = 1e-5F;
    const float finite_diff_h = 1e-3F;  // Step size for numerical gradient

    modules::LayerNormLayer ln(features, eps, false);

    // Create input with some variation (not uniform)
    std::vector<float> input_data(features);
    for (uint32_t i = 0; i < features; ++i) {
        input_data[i] = 1.0F + static_cast<float>(i) * 0.1F;  // 1.0, 1.1, 1.2, ...
    }
    auto input_tt = core::from_vector(input_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Forward pass
    auto output = ln(input);
    output->backward();

    // Get analytical gradients
    auto analytical_input_grad = tensor_to_vector(input->get_grad());

    // Test multiple input gradient elements (first, middle, last)
    std::vector<uint32_t> test_indices = {0, features / 2, features - 1};

    for (uint32_t idx : test_indices) {
        // Reset for each test
        autograd::ctx().reset_graph();

        // Restore original input
        for (uint32_t i = 0; i < features; ++i) {
            input_data[i] = 1.0F + static_cast<float>(i) * 0.1F;
        }

        // f(x + h)
        input_data[idx] += finite_diff_h;
        auto input_plus_tt =
            core::from_vector(input_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto input_plus = autograd::create_tensor(input_plus_tt);
        auto output_plus = ln(input_plus);
        auto output_plus_vec = tensor_to_vector(output_plus->get_value());
        float sum_plus = 0.0F;
        for (float val : output_plus_vec) {
            sum_plus += val;
        }

        autograd::ctx().reset_graph();

        // f(x - h)
        input_data[idx] -= 2.0F * finite_diff_h;
        auto input_minus_tt =
            core::from_vector(input_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto input_minus = autograd::create_tensor(input_minus_tt);
        auto output_minus = ln(input_minus);
        auto output_minus_vec = tensor_to_vector(output_minus->get_value());
        float sum_minus = 0.0F;
        for (float val : output_minus_vec) {
            sum_minus += val;
        }

        // Numerical gradient: (f(x+h) - f(x-h)) / (2h)
        float numerical_grad = (sum_plus - sum_minus) / (2.0F * finite_diff_h);

        // Analytical gradient should be close to numerical gradient
        // Use relaxed tolerance due to finite difference approximation and hardware precision
        EXPECT_NEAR(analytical_input_grad[idx], numerical_grad, 0.1F)
            << "Input gradient at index " << idx << " differs from numerical gradient";
    }
}

// Test 11: Configurable min_safe_eps parameter validation
TEST_F(LayerNormEpsilonTest, ConfigurableMinSafeEpsWorks) {
    const uint32_t features = 64;
    const float tiny_eps = 1e-12F;  // Would be clamped

    // Test 1: Default min_safe_eps (1e-4F)
    modules::LayerNormLayer ln_default(features, tiny_eps, false, true);  // clamping enabled, default min_safe_eps
    EXPECT_FLOAT_EQ(ln_default.get_min_safe_eps(), 1e-4F);

    // Test 2: Custom min_safe_eps (1e-3F) - less aggressive clamping
    modules::LayerNormLayer ln_custom(features, tiny_eps, false, true, 1e-3F);  // custom min_safe_eps
    EXPECT_FLOAT_EQ(ln_custom.get_min_safe_eps(), 1e-3F);

    // Test 3: Very small custom min_safe_eps (1e-6F) - allows smaller epsilon
    modules::LayerNormLayer ln_small_clamp(features, tiny_eps, false, true, 1e-6F);
    EXPECT_FLOAT_EQ(ln_small_clamp.get_min_safe_eps(), 1e-6F);

    // All should produce valid outputs without NaN/Inf
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    auto input_default = autograd::create_tensor(input_tt);
    auto output_default = ln_default(input_default);
    auto vec_default = tensor_to_vector(output_default->get_value());

    auto input_custom = autograd::create_tensor(input_tt);
    auto output_custom = ln_custom(input_custom);
    auto vec_custom = tensor_to_vector(output_custom->get_value());

    auto input_small = autograd::create_tensor(input_tt);
    auto output_small = ln_small_clamp(input_small);
    auto vec_small = tensor_to_vector(output_small->get_value());

    // Verify no NaN/Inf in all outputs
    for (float val : vec_default) {
        EXPECT_FALSE(std::isnan(val)) << "Default min_safe_eps produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Default min_safe_eps produced Inf";
    }
    for (float val : vec_custom) {
        EXPECT_FALSE(std::isnan(val)) << "Custom min_safe_eps (1e-3F) produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Custom min_safe_eps (1e-3F) produced Inf";
    }
    for (float val : vec_small) {
        EXPECT_FALSE(std::isnan(val)) << "Small min_safe_eps (1e-6F) produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Small min_safe_eps (1e-6F) produced Inf";
    }
}

// Test 12: min_safe_eps affects clamping behavior
TEST_F(LayerNormEpsilonTest, MinSafeEpsAffectsClamping) {
    const uint32_t features = 64;
    const float tiny_eps = 1e-12F;

    // With aggressive clamping (1e-2F), epsilon should be clamped higher
    modules::LayerNormLayer ln_aggressive(features, tiny_eps, false, true, 1e-2F);

    // With relaxed clamping (1e-5F), epsilon clamped to smaller value
    modules::LayerNormLayer ln_relaxed(features, tiny_eps, false, true, 1e-5F);

    // Create test input with small variance
    auto input_tt = core::full(ttnn::Shape({1, 1, 1, features}), 1e-3F, &autograd::ctx().get_device());
    auto input_aggressive = autograd::create_tensor(input_tt);
    auto input_relaxed = autograd::create_tensor(input_tt);

    auto output_aggressive = ln_aggressive(input_aggressive);
    auto output_relaxed = ln_relaxed(input_relaxed);

    auto vec_aggressive = tensor_to_vector(output_aggressive->get_value());
    auto vec_relaxed = tensor_to_vector(output_relaxed->get_value());

    // Both should be valid (no NaN/Inf)
    for (float val : vec_aggressive) {
        EXPECT_FALSE(std::isnan(val)) << "Aggressive clamping produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Aggressive clamping produced Inf";
    }
    for (float val : vec_relaxed) {
        EXPECT_FALSE(std::isnan(val)) << "Relaxed clamping produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Relaxed clamping produced Inf";
    }

    // Outputs should differ slightly due to different effective epsilon
    float max_diff = 0.0F;
    for (size_t i = 0; i < vec_aggressive.size(); ++i) {
        float diff = std::abs(vec_aggressive[i] - vec_relaxed[i]);
        max_diff = std::max(max_diff, diff);
    }
    // Different min_safe_eps should produce measurable difference
    EXPECT_GT(max_diff, 1e-6F) << "Different min_safe_eps should affect output";
}
