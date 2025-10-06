// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"

// ============================================================================
// GELU Operation Test Suite
// ============================================================================
// Tests the GELU (Gaussian Error Linear Unit) activation function used in
// BERT and other transformer models.
//
// Implementation tested: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Test Coverage:
// 1. Forward and backward correctness vs reference implementation
// 2. BERT model shapes (base and large configurations)
// 3. Numerical stability (saturation, near-zero, gradient flow)
// 4. Memory configurations (L1/DRAM)
// 5. Block alignment patterns for tile-based hardware
// 6. Integration patterns used in BERT MLP
//
// Shape notation: [B, N, S, C] where:
//   B = batch size
//   N = number of heads (typically 1 for BERT)
//   S = sequence length
//   C = feature dimension (embedding/hidden dim)
// ============================================================================

class GELUOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Reference Implementations
// ============================================================================

namespace {

/**
 * Reference implementation of exact GELU forward pass
 * GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 */
xt::xarray<float> gelu_forward_reference(const xt::xarray<float>& x) {
    const float sqrt2 = std::sqrt(2.0f);
    return 0.5f * x * (1.0f + xt::erf(x / sqrt2));
}

/**
 * Reference implementation of exact GELU backward pass
 * GELU'(x) = Φ(x) + x * φ(x)
 * where Φ(x) = CDF of standard normal, φ(x) = PDF of standard normal
 */
xt::xarray<float> gelu_backward_reference(const xt::xarray<float>& x, const xt::xarray<float>& grad) {
    const float sqrt2 = std::sqrt(2.0f);
    const float sqrt_2pi = std::sqrt(2.0f * std::numbers::pi_v<float>);

    auto cdf = 0.5f * (1.0f + xt::erf(x / sqrt2));
    auto pdf = xt::exp(-0.5f * x * x) / sqrt_2pi;
    auto gelu_grad = cdf + x * pdf;

    return grad * gelu_grad;
}

/**
 * Compare TTML GELU implementation against reference
 * Tests both forward and backward passes with BFloat16-appropriate tolerances
 */
void CompareGELUVsReference(const xt::xarray<float>& input_data) {
    using namespace ttml;

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));

    // Forward pass
    auto result = ops::gelu(input);
    auto result_xtensor = core::to_xtensor(result->get_value());
    auto expected_result = gelu_forward_reference(input_data);

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-3F, 3e-2F))
        << "Forward pass failed tolerance check";

    // Backward pass
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    auto input_grad = core::to_xtensor(input->get_grad());

    // Compute reference gradient through MSE loss
    auto total_elements = static_cast<float>(input_data.size());
    auto mse_grad = (2.0f / total_elements) * expected_result;
    auto expected_grad = gelu_backward_reference(input_data, mse_grad);

    EXPECT_TRUE(xt::allclose(input_grad, expected_grad, 1e-3F, 3e-2F))
        << "Backward pass failed tolerance check";
}

/**
 * Helper to test GELU with a specific tensor shape
 */
void CompareGELUVsReferenceWithShape(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    xt::xarray<float> input_data = xt::empty<float>(shape);
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();
    // Use [-3, 3] range to cover GELU's interesting regions
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-3.0F, 3.0F); }, seed);

    CompareGELUVsReference(input_data);
}

}  // namespace

// ============================================================================
// Section 1: Basic Correctness Tests
// ============================================================================

TEST_F(GELUOpTest, GELU_Initial) {
    // Initial test: absorbs device initialization overhead
    // Uses minimal tile-aligned shape
    CompareGELUVsReferenceWithShape({1, 1, 1, 8});
}

TEST_F(GELUOpTest, GELU_Small) {
    // Small tensor with multiple elements - basic functionality
    CompareGELUVsReferenceWithShape({2, 1, 4, 32});
}

TEST_F(GELUOpTest, GELU_DeterministicValues) {
    using namespace ttml;

    // Test with known values to verify exact behavior
    std::vector<float> test_data = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

    auto input = autograd::create_tensor(
        core::from_vector(test_data, ttnn::Shape{2, 1, 1, 4}, &autograd::ctx().get_device()));

    auto result = ops::gelu(input);
    auto result_data = core::to_vector(result->get_value());

    // Expected GELU values (precomputed)
    std::vector<float> expected = {
        -0.04550f,  // GELU(-2.0)
        -0.15880f,  // GELU(-1.0)
        -0.15426f,  // GELU(-0.5)
         0.00000f,  // GELU(0.0)
         0.34574f,  // GELU(0.5)
         0.84134f,  // GELU(1.0)
         1.95450f,  // GELU(2.0)
         2.99595f   // GELU(3.0)
    };

    ASSERT_EQ(result_data.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], 3e-2f);
    }
}

// Block size alignment tests - testing Wt % 4 patterns
// where Wt = ceil(C / 32) is the number of tiles in width dimension

TEST_F(GELUOpTest, GELU_Aligned128) {
    // C=128, Wt=4, Wt%4=0 (perfectly aligned)
    CompareGELUVsReferenceWithShape({1, 1, 1, 128});
}

TEST_F(GELUOpTest, GELU_Aligned160) {
    // C=160, Wt=5, Wt%4=1
    CompareGELUVsReferenceWithShape({1, 1, 1, 160});
}

TEST_F(GELUOpTest, GELU_Aligned192) {
    // C=192, Wt=6, Wt%4=2
    CompareGELUVsReferenceWithShape({1, 1, 1, 192});
}

TEST_F(GELUOpTest, GELU_Aligned224) {
    // C=224, Wt=7, Wt%4=3
    CompareGELUVsReferenceWithShape({1, 1, 1, 224});
}

TEST_F(GELUOpTest, GELU_Large) {
    // Large tensor to test memory handling
    CompareGELUVsReferenceWithShape({1, 1, 1, 32768});
}

TEST_F(GELUOpTest, NIGHTLY_GELU_VeryLarge) {
    // Extreme size to stress-test memory subsystem
    CompareGELUVsReferenceWithShape({1, 1, 1, 1048576});
}

TEST_F(GELUOpTest, NIGHTLY_GELU_LargeRandom) {
    using namespace ttml;

    // Large random test: validates across full working range with 10K elements
    // Tests BFloat16 precision accumulation with deterministic seed
    uint32_t n = 10000;
    std::vector<uint32_t> shape = {1, 1, 1, n};

    xt::xarray<float> input_data = xt::empty<float>(shape);
    uint32_t fixed_seed = 424242;  // Deterministic
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); }, fixed_seed);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));

    // Forward pass - relaxed tolerances for 10K elements
    auto result = ops::gelu(input);
    auto result_xtensor = core::to_xtensor(result->get_value());
    auto expected_result = gelu_forward_reference(input_data);

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 2e-3F, 5e-2F));

    // Backward pass
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    auto input_grad = core::to_xtensor(input->get_grad());
    auto total_elements = static_cast<float>(input_data.size());
    auto mse_grad = (2.0f / total_elements) * expected_result;
    auto expected_grad = gelu_backward_reference(input_data, mse_grad);

    EXPECT_TRUE(xt::allclose(input_grad, expected_grad, 2e-3F, 5e-2F));
}

TEST_F(GELUOpTest, GELU_Unaligned) {
    // C dimension not multiple of 32 - validates padding/alignment
    CompareGELUVsReferenceWithShape({2, 1, 32, 100});
}

// ============================================================================
// Section 2: BERT-Specific Integration Tests
// ============================================================================

TEST_F(GELUOpTest, GELU_BERT_BaseHidden) {
    // BERT-base: batch=2, seq_len=64, hidden_dim=768
    CompareGELUVsReferenceWithShape({2, 1, 64, 768});
}

TEST_F(GELUOpTest, GELU_BERT_BaseIntermediate) {
    // BERT-base MLP: intermediate_dim=3072 (4x hidden_dim)
    // GELU is applied here: Linear(768→3072) → GELU → Linear(3072→768)
    CompareGELUVsReferenceWithShape({2, 1, 64, 3072});
}

TEST_F(GELUOpTest, GELU_BERT_LargeHidden) {
    // BERT-large: hidden_dim=1024
    CompareGELUVsReferenceWithShape({2, 1, 128, 1024});
}

TEST_F(GELUOpTest, GELU_BERT_LargeIntermediate) {
    // BERT-large MLP: intermediate_dim=4096
    CompareGELUVsReferenceWithShape({2, 1, 128, 4096});
}

TEST_F(GELUOpTest, GELU_BERT_MaxSeqLen) {
    // BERT maximum sequence length (512 tokens)
    CompareGELUVsReferenceWithShape({2, 1, 512, 768});
}

TEST_F(GELUOpTest, GELU_BERT_MultiBatch) {
    // Training scenario with larger batch
    CompareGELUVsReferenceWithShape({4, 1, 128, 768});
}

TEST_F(GELUOpTest, GELU_BERTMLPIntegration) {
    using namespace ttml;

    // Simulate BERT MLP pattern: Linear → GELU → Linear
    uint32_t batch = 2;
    uint32_t seq = 64;
    uint32_t intermediate = 3072;

    std::vector<uint32_t> shape = {batch, 1, seq, intermediate};
    xt::xarray<float> input_data = xt::empty<float>(shape);
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-2.0F, 2.0F); }, seed);

    auto intermediate_tensor = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()));

    // Apply GELU
    auto gelu_output = ops::gelu(intermediate_tensor);

    // Verify shape preservation
    EXPECT_EQ(gelu_output->get_shape()[0], batch);
    EXPECT_EQ(gelu_output->get_shape()[2], seq);
    EXPECT_EQ(gelu_output->get_shape()[3], intermediate);

    // Test backward through the activation
    auto target = autograd::create_tensor(core::zeros_like(gelu_output->get_value()));
    auto loss = ops::mse_loss(gelu_output, target);
    loss->backward();

    EXPECT_TRUE(core::is_tensor_initialized(intermediate_tensor->get_grad()));

    // Verify gradients are reasonable (non-NaN, non-Inf)
    auto grad_data = core::to_vector(intermediate_tensor->get_grad());
    for (float val : grad_data) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

// ============================================================================
// Section 3: Numerical Stability Tests
// ============================================================================

TEST_F(GELUOpTest, GELU_Saturation) {
    using namespace ttml;

    // Test extreme values where GELU saturates
    std::vector<float> test_data = {
        -10.0f, -5.0f, -3.0f, -1.0f,  // Negative saturation
        10.0f,  5.0f,  3.0f,  1.0f    // Positive saturation
    };

    auto input = autograd::create_tensor(
        core::from_vector(test_data, ttnn::Shape{2, 1, 1, 4}, &autograd::ctx().get_device()));

    auto result = ops::gelu(input);
    auto result_data = core::to_vector(result->get_value());

    // Validate saturation behavior
    EXPECT_NEAR(result_data[0], 0.0f, 1e-4f);       // GELU(-10) ≈ 0
    EXPECT_NEAR(result_data[1], 0.0f, 2e-4f);       // GELU(-5) ≈ 0 (very small negative)
    EXPECT_NEAR(result_data[5], 5.0f, 1e-3f);       // GELU(5) ≈ 5
    EXPECT_NEAR(result_data[4], 10.0f, 1e-2f);      // GELU(10) ≈ 10

    // Test gradients in saturation regions
    result->backward();
    auto grad = core::to_vector(input->get_grad());

    // Gradient should vanish for large negative x
    EXPECT_NEAR(grad[0], 0.0f, 1e-3f);
    // Gradient should approach 1 for large positive x
    EXPECT_NEAR(grad[4], 1.0f, 5e-2f);
}

TEST_F(GELUOpTest, GELU_NearZero) {
    using namespace ttml;

    // Test behavior near zero where GELU has interesting curvature
    std::vector<float> test_data = {-0.5f, -0.1f, -0.01f, 0.0f, 0.01f, 0.1f, 0.5f, 1.0f};

    auto input = autograd::create_tensor(
        core::from_vector(test_data, ttnn::Shape{2, 1, 1, 4}, &autograd::ctx().get_device()));

    auto result = ops::gelu(input);
    auto result_data = core::to_vector(result->get_value());

    // GELU(0) = 0 exactly
    EXPECT_NEAR(result_data[3], 0.0f, 1e-4f);

    // Near zero, GELU is approximately x/2
    EXPECT_NEAR(result_data[2], -0.005f, 1e-3f);   // GELU(-0.01) ≈ -0.005
    EXPECT_NEAR(result_data[4], 0.005f, 1e-3f);    // GELU(0.01) ≈ 0.005
    EXPECT_NEAR(result_data[5], 0.0540f, 1e-2f);   // GELU(0.1) ≈ 0.054
}

TEST_F(GELUOpTest, GELU_GradientFlow) {
    using namespace ttml;

    // Test gradient flow at extreme values
    std::vector<float> extreme_values = {
        -100.0f, -50.0f, -20.0f, -10.0f,  // Very negative
        100.0f,  50.0f,  20.0f,  10.0f    // Very positive
    };

    auto input = autograd::create_tensor(
        core::from_vector(extreme_values, ttnn::Shape{2, 1, 1, 4}, &autograd::ctx().get_device()));

    auto result = ops::gelu(input);

    // Set upstream gradient to 1 to isolate GELU gradient
    result->set_grad(core::ones_like(result->get_value()));
    result->backward();

    auto grad = core::to_vector(input->get_grad());

    // For very negative values, gradient should be effectively zero
    EXPECT_NEAR(grad[0], 0.0f, 1e-6f);  // x=-100
    EXPECT_NEAR(grad[1], 0.0f, 1e-6f);  // x=-50

    // For very positive values, gradient should be effectively one
    EXPECT_NEAR(grad[4], 1.0f, 1e-6f);  // x=100
    EXPECT_NEAR(grad[5], 1.0f, 1e-6f);  // x=50
}

TEST_F(GELUOpTest, GELU_GradientAccumulation) {
    using namespace ttml;

    // Test that gradients accumulate correctly when tensor appears multiple times
    std::vector<float> data = {1.0f, -1.0f, 2.0f, -2.0f};

    auto x = autograd::create_tensor(
        core::from_vector(data, ttnn::Shape{1, 1, 2, 2}, &autograd::ctx().get_device()));

    // Use GELU twice: gelu(x) + gelu(x) * 2
    auto gelu1 = ops::gelu(x);
    auto gelu2 = ops::gelu(x);
    auto gelu2_scaled = ops::mul(gelu2, 2.0f);
    auto result = ops::add(gelu1, gelu2_scaled);

    // Backward pass
    result->set_grad(core::ones_like(result->get_value()));
    result->backward();

    auto x_grad = core::to_vector(x->get_grad());

    // Gradient should be 3 * gelu'(x)
    xt::xarray<float> x_array = xt::adapt(data, std::vector<size_t>{1, 1, 2, 2});
    auto ones = xt::ones_like(x_array);
    auto expected_grad_single = gelu_backward_reference(x_array, ones);
    auto expected_grad = 3.0f * expected_grad_single;
    auto expected_grad_vec = std::vector<float>(expected_grad.begin(), expected_grad.end());

    for (size_t i = 0; i < x_grad.size(); ++i) {
        EXPECT_NEAR(x_grad[i], expected_grad_vec[i], 3e-2f);
    }
}

TEST_F(GELUOpTest, GELU_PrecisionCheck) {
    using namespace ttml;

    // Single precision check with typical BERT-base shape
    std::vector<uint32_t> shape = {2, 1, 64, 768};

    xt::xarray<float> input_data = xt::empty<float>(shape);
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-3.0F, 3.0F); }, seed);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto result = ops::gelu(input);

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    auto input_grad = core::to_xtensor(input->get_grad());

    // Compute reference gradient
    auto result_ref = gelu_forward_reference(input_data);
    auto total_elements = static_cast<float>(input_data.size());
    auto mse_grad = (2.0f / total_elements) * result_ref;
    auto grad_ref = gelu_backward_reference(input_data, mse_grad);

    // Calculate RMSE
    auto abs_diff = xt::abs(grad_ref - input_grad);
    float rmse = xt::sqrt(xt::mean(xt::square(abs_diff)))();

    // Validate precision
    EXPECT_LT(rmse, 1e-5f) << "RMSE exceeds threshold";
    EXPECT_TRUE(xt::allclose(input_grad, grad_ref, 1e-3f, 3e-2f));
}

// ============================================================================
// Section 4: Memory Configuration Tests
// ============================================================================

TEST_F(GELUOpTest, GELU_L1Memory) {
    using namespace ttml;

    std::vector<float> data(768, 0.5f);

    // Create tensor with L1 memory configuration
    auto tensor = core::from_vector(data, ttnn::Shape({1, 1, 1, 768}), &autograd::ctx().get_device());
    tensor = ttnn::to_memory_config(tensor, ttnn::L1_MEMORY_CONFIG);

    auto input = autograd::create_tensor(tensor);
    auto result = ops::gelu(input);

    // Verify shape preservation
    EXPECT_EQ(result->get_shape()[3], 768);

    // Verify correctness
    auto result_data = core::to_vector(result->get_value());
    xt::xarray<float> expected_input = xt::ones<float>({1, 1, 1, 768}) * 0.5f;
    auto expected = gelu_forward_reference(expected_input);
    auto expected_vec = std::vector<float>(expected.begin(), expected.end());

    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected_vec[i], 3e-2f);
    }
}

TEST_F(GELUOpTest, GELU_DRAMMemory) {
    using namespace ttml;

    std::vector<float> data(768, -0.5f);

    // Create tensor with DRAM memory configuration
    auto tensor = core::from_vector(data, ttnn::Shape({1, 1, 1, 768}), &autograd::ctx().get_device());
    tensor = ttnn::to_memory_config(tensor, ttnn::DRAM_MEMORY_CONFIG);

    auto input = autograd::create_tensor(tensor);
    auto result = ops::gelu(input);

    // Verify shape preservation
    EXPECT_EQ(result->get_shape()[3], 768);

    // Verify correctness
    auto result_data = core::to_vector(result->get_value());
    xt::xarray<float> expected_input = xt::ones<float>({1, 1, 1, 768}) * -0.5f;
    auto expected = gelu_forward_reference(expected_input);
    auto expected_vec = std::vector<float>(expected.begin(), expected.end());

    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected_vec[i], 3e-2f);
    }
}
