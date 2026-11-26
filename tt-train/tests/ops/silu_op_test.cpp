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
 * Compares ops::silu against the float64 reference on inputs drawn from [-2, 2]:
 * 1. Forward pass: within kSiluMaxUlp of silu(x) in every element
 * 2. Backward pass (gradient of ones): within kSiluGradMaxUlp of silu'(x) in every element
 * 3. All outputs and gradients are finite (no NaN/Inf values)
 *
 * @param shape Input tensor shape [B, N, S, C]
 */
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> input_data = xt::empty<float>(shape);
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-2.0F, 2.0F); }, /* seed */ rng());

    CompareKernelVsReference(input_data);
}

}  // namespace

// ============================================================================
// Section 2: ops::silu vs the float64 reference on a multi-core shape
// ============================================================================

// NanoLlama-like shape, [B, N, S, C] = [2, 1, 64, 512]: 64 tiles, spread over several cores.
TEST_F(SiLUOpTest, SiLU_Compare_NanoLlama_Shape) {
    CompareKernelVsReferenceWithShape({2, 1, 64, 512});
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
