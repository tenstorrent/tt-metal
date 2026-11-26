// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/bf16_ulp.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/core/core.hpp"

class SiLUOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: ops::silu vs a float64 reference
// ============================================================================
// Like the other activation tests, the backward is seeded with ones (result->backward()), so the input
// gradient is silu'(x) itself and is compared element by element against float64 on the bf16 values the
// device read:
//     silu(x)  = x * s(x)
//     silu'(x) = s(x) * (1 + x * s(-x)),  s(x) = 1 / (1 + exp(-x))
// s(-x) is computed directly rather than as 1 - s(x), which loses precision for large x.
// ============================================================================

namespace {

xt::xarray<double> sigmoid_reference(const xt::xarray<double>& x) {
    return 1.0 / (1.0 + xt::exp(-x));
}

xt::xarray<double> silu_reference(const xt::xarray<double>& x) {
    return x * sigmoid_reference(x);
}

xt::xarray<double> silu_grad_reference(const xt::xarray<double>& x) {
    return sigmoid_reference(x) * (1.0 + x * sigmoid_reference(-x));
}

// ttnn::silu and ttnn::silu_bw (with a gradient of ones) are within 1 bf16 ULP of the exact result in every
// element (measured 0.89 and 0.5002).
constexpr double kSiluMaxUlp = 1.0;
constexpr double kSiluGradMaxUlp = 1.0;

void CompareKernelVsReference(const xt::xarray<float>& input_data) {
    using namespace ttml;

    auto input = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
    const xt::xarray<double> input_stored = xt::cast<double>(core::to_xtensor(input->get_value()));

    auto result = ops::silu(input);
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(result->get_value()), silu_reference(input_stored), "silu forward", kSiluMaxUlp);

    result->backward();
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(input->get_grad()), silu_grad_reference(input_stored), "silu backward", kSiluGradMaxUlp);
}

/**
 * Helper function to compare ops::silu against the float64 reference
 *
 * This function tests both forward and backward passes to ensure:
 * 1. Forward pass: within kSiluMaxUlp of silu(x) in every element
 * 2. Backward pass (gradient of ones): within kSiluGradMaxUlp of silu'(x) in every element
 * 3. All outputs and gradients are finite (no NaN/Inf values)
 *
 * @param shape Input tensor shape [B, N, S, C] where:
 *   - B: batch size
 *   - N: number of channels (usually 1 for transformers)
 *   - S: sequence length (height for transformers)
 *   - C: feature dimension (width/embedding dimension)
 *
 * Test cases cover various scenarios:
 * - Block size alignment: Wt % block_size patterns (0, 1, 2, 3)
 * - Memory patterns: small vs large tensors
 * - Realistic shapes: transformer model dimensions
 */
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    // Generate random input data with range [-2, 2] to test SiLU behavior across saturation regions
    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> input_data = xt::empty<float>(shape);
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-2.0F, 2.0F); }, /* seed */ rng());

    CompareKernelVsReference(input_data);
}

}  // namespace

// ============================================================================
// Section 2: ops::silu vs the float64 reference across shapes
// ============================================================================
// These tests systematically compare ops::silu against the float64
// reference across different scenarios:
//
// - Block size patterns: Wt % block_size = 0, 1, 2, 3 (where block_size = 4)
// - Memory patterns: small tensors vs large tensors
// - Training scenarios: realistic model shapes (NanoLlama, etc.)
// - Batch patterns: single vs multiple batches and sequences
//
// Note: Wt = ceil(C / 32), so we test different C values to cover all
// block_size remainder patterns.
// Shape notation: [B, N, S, C] where C is the feature dimension
// ============================================================================

// Test small tensor - basic functionality
TEST_F(SiLUOpTest, SiLU_Compare_Small) {
    // C=8, Wt=1, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 8});
}

// Test block_size alignment patterns
// Wt % block_size = 0 (perfectly aligned)
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder0) {
    // C=128, Wt=4, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 128});
}

// Wt % block_size = 1
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder1) {
    // C=160, Wt=5, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 160});
}

// Wt % block_size = 2
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder2) {
    // C=192, Wt=6, Wt%4=2
    CompareKernelVsReferenceWithShape({1, 1, 1, 192});
}

// Wt % block_size = 3
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder3) {
    // C=224, Wt=7, Wt%4=3
    CompareKernelVsReferenceWithShape({1, 1, 1, 224});
}

// Test large tensor - memory stress test
TEST_F(SiLUOpTest, SiLU_Compare_Large) {
    // Large C dimension to test memory handling
    // C=32768, Wt=1024, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 32768});
}

// Test very large tensor - extreme memory test
TEST_F(SiLUOpTest, SiLU_Compare_VeryLarge) {
    // Very large C dimension ~1M elements
    // C=1048576, Wt=32768, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 1048576});
}

// Test NanoLlama-like shape - realistic transformer model
TEST_F(SiLUOpTest, SiLU_Compare_NanoLlama_Shape) {
    // Typical NanoLlama dimensions: multiple batches and sequences
    // B=2, N=1, S=64, C=512 (hidden dimension)
    // C=512, Wt=16, Wt%4=0
    CompareKernelVsReferenceWithShape({2, 1, 64, 512});
}

// Test batch processing with different sequence lengths
TEST_F(SiLUOpTest, SiLU_Compare_MultiBatch_MultiSeq) {
    // Multiple batches with longer sequences
    // B=4, N=1, S=128, C=768
    // C=768, Wt=24, Wt%4=0
    CompareKernelVsReferenceWithShape({4, 1, 128, 768});
}

// Test smaller model dimensions with unaligned C
TEST_F(SiLUOpTest, SiLU_Compare_SmallModel_Unaligned) {
    // Smaller model with unaligned channel dimension
    // B=2, N=1, S=32, C=100
    // C=100, Wt=4, Wt%4=0 (but C is not multiple of 32)
    CompareKernelVsReferenceWithShape({2, 1, 32, 100});
}

// ============================================================================
// Section 3: ops::silu backward over the zero of silu' and both saturation tails
// ============================================================================

TEST_F(SiLUOpTest, SiLU_Backward_WideRange_vs_Reference) {
    using namespace ttml;

    // nanollama-like shape; [-8, 8] covers the zero of silu' (x ~ -1.28) and both saturation tails.
    const std::vector<uint32_t> shape = {2, 1, 64, 512};
    const auto input_data = test_utils::make_uniform_xarray<float>(shape, -8.0F, 8.0F, 7U);

    auto input = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
    const xt::xarray<double> expected = silu_grad_reference(xt::cast<double>(core::to_xtensor(input->get_value())));

    auto result = ops::silu(input);
    result->backward();
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(input->get_grad()), expected, "silu backward", kSiluGradMaxUlp);
}

// The upstream gradient in L1 and the input in DRAM: ttnn::silu_bw keeps its accurate kernel only when it is told
// where the result goes, and otherwise falls back to a composite that loses bits near the zero of silu'.
TEST_F(SiLUOpTest, SiLU_Backward_GradientInL1_InputInDram_vs_Reference) {
    using namespace ttml;

    const std::vector<uint32_t> shape = {2, 1, 64, 512};
    const auto input_data = test_utils::make_uniform_xarray<float>(shape, -8.0F, 8.0F, 11U);

    auto input = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
    ASSERT_EQ(input->get_value().memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    const xt::xarray<double> expected = silu_grad_reference(xt::cast<double>(core::to_xtensor(input->get_value())));

    auto result = ops::silu(input);
    result->set_grad(ttnn::to_memory_config(core::ones_like(result->get_value()), ttnn::L1_MEMORY_CONFIG));
    result->backward();
    EXPECT_EQ(input->get_grad().memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(input->get_grad()), expected, "silu backward, L1 gradient", kSiluGradMaxUlp);
}
