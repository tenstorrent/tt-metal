// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_op.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tt-umd/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "ops/losses.hpp"
#include "test_utils/random_data.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class RMSNormOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

// ============================================================================
// Section 1: RMSNorm Kernel vs PyTorch Reference Implementation
// ============================================================================
// These tests validate the optimized RMSNorm kernel implementation against
// PyTorch's reference implementation to ensure numerical correctness.
//
// Test methodology:
// 1. Create test tensor `x` of shape [N,C,H,W] with x.requires_grad = True
// 2. Compute PyTorch RMSNorm: `x_norm_sum = torch.nn.functional.rms_norm(x).sum()`
// 3. Compute PyTorch gradient: `x_grad = torch.autograd.grad(x_norm_sum, x)[0]`
// 4. Compare TTML kernel results with PyTorch reference results
// ============================================================================
TEST_F(RMSNormOpTest, RMSNorm_Small_Forward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}};
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));
}

TEST_F(RMSNormOpTest, RMSNorm_Small_Backward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(
        core::from_xtensor(example_xtensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad = xt::xarray<float>(
        {{{{5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F,
            5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1.0e-3F, 1e-2F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    auto expected_gamma_grad =
        xt::xarray<float>({{{{0.0334F, 0.1338F, 0.2988F, 0.5352F, 0.0334F, 0.1338F, 0.2988F, 0.5352F}}}});
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 1.0e-3F, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Forward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor = autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {
        {{{0.00000F, 0.40820F, 0.81641F, 1.22656F, 1.63281F}, {0.69922F, 0.83984F, 0.98047F, 1.11719F, 1.25781F},
          {0.82812F, 0.91016F, 0.99219F, 1.07812F, 1.15625F}, {0.87891F, 0.93750F, 0.99609F, 1.05469F, 1.11719F},
          {0.90625F, 0.95312F, 0.99609F, 1.04688F, 1.08594F}, {0.92578F, 0.96094F, 1.00000F, 1.03906F, 1.07031F},
          {0.93750F, 0.96875F, 1.00000F, 1.03125F, 1.06250F}, {0.94531F, 0.97266F, 1.00000F, 1.02344F, 1.05469F},
          {0.95312F, 0.97656F, 1.00000F, 1.02344F, 1.04688F}, {0.95703F, 0.97656F, 1.00000F, 1.02344F, 1.03906F},
          {0.96094F, 0.98047F, 1.00000F, 1.01562F, 1.03906F}, {0.96484F, 0.98047F, 1.00000F, 1.01562F, 1.03125F},
          {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F},
          {0.97266F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.97266F, 0.98828F, 1.00000F, 1.01562F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.01562F, 1.02344F}, {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}, {0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}}},
        {{{0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.01562F}, {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}}}};
    assert((expected_result.shape() == result_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 6e-2F, 1e-8F));
}

TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Backward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor =
        autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();

    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    xt::xarray<float> expected_example_tensor_grad = xt::zeros_like(a_xarray);
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 5e-2F, 1e-3F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    xt::xarray<float> expected_gamma_grad = {{{{0.36111F, 0.37644F, 0.39589F, 0.41945F, 0.44712F}}}};
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 5e-2F));
}

// ============================================================================
// Section 2: RMSNorm Composite vs PyTorch Reference Implementation
// ============================================================================
// These tests validate the composite RMSNorm implementation (built from basic ops)
// against PyTorch's reference implementation to ensure numerical correctness.
//
// The composite implementation serves as a reference for the optimized kernel
// and uses standard operations like power, mean, sqrt, and multiply.
// Same test methodology as Section 1, but using rmsnorm_composite() instead.
// ============================================================================
TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Small_Forward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}};
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Small_Backward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(
        core::from_xtensor(example_xtensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad = xt::xarray<float>(
        {{{{5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F,
            5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1.0e-3F, 1e-2F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    auto expected_gamma_grad =
        xt::xarray<float>({{{{0.0334F, 0.1338F, 0.2988F, 0.5352F, 0.0334F, 0.1338F, 0.2988F, 0.5352F}}}});
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 1.0e-3F, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Forward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor = autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {
        {{{0.00000F, 0.40820F, 0.81641F, 1.22656F, 1.63281F}, {0.69922F, 0.83984F, 0.98047F, 1.11719F, 1.25781F},
          {0.82812F, 0.91016F, 0.99219F, 1.07812F, 1.15625F}, {0.87891F, 0.93750F, 0.99609F, 1.05469F, 1.11719F},
          {0.90625F, 0.95312F, 0.99609F, 1.04688F, 1.08594F}, {0.92578F, 0.96094F, 1.00000F, 1.03906F, 1.07031F},
          {0.93750F, 0.96875F, 1.00000F, 1.03125F, 1.06250F}, {0.94531F, 0.97266F, 1.00000F, 1.02344F, 1.05469F},
          {0.95312F, 0.97656F, 1.00000F, 1.02344F, 1.04688F}, {0.95703F, 0.97656F, 1.00000F, 1.02344F, 1.03906F},
          {0.96094F, 0.98047F, 1.00000F, 1.01562F, 1.03906F}, {0.96484F, 0.98047F, 1.00000F, 1.01562F, 1.03125F},
          {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F},
          {0.97266F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.97266F, 0.98828F, 1.00000F, 1.01562F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.01562F, 1.02344F}, {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}, {0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}}},
        {{{0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.01562F}, {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}}}};
    assert((expected_result.shape() == result_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 6e-2F, 1e-8F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Backward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor =
        autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();

    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    xt::xarray<float> expected_example_tensor_grad = xt::zeros_like(a_xarray);
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 5e-2F, 1e-3F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    xt::xarray<float> expected_gamma_grad = {{{{0.36111F, 0.37644F, 0.39589F, 0.41945F, 0.44712F}}}};
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 5e-2F));
}

// ============================================================================
// Section 3: RMSNorm Kernel vs Composite Implementation Comparison
// ============================================================================
// These tests compare the optimized RMSNorm kernel implementation against the
// composite implementation to ensure they produce identical results across
// different tensor shapes, sizes, and edge cases.
//
// This validation ensures that the kernel optimization maintains correctness
// while providing performance benefits. Both implementations should produce
// identical forward and backward pass results.
// ============================================================================

/**
 * Helper function to compare kernel vs composite implementations of RMSNorm
 *
 * This function tests both forward and backward passes to ensure:
 * 1. Forward pass: kernel and composite produce identical results
 * 2. Backward pass: gradients computed by both implementations match
 * 3. All outputs and gradients are finite (no NaN/Inf values)
 *
 * @param shape Input tensor shape [N, C, H, W] where:
 *   - N: batch size
 *   - C: number of channels (normalized dimension)
 *   - H: height
 *   - W: width
 *
 * Test cases cover various scenarios:
 * - Alignment: C % 32 == 0 (aligned) vs C % 32 != 0 (unaligned/masking)
 * - Memory: fits in L1 cache vs exceeds L1 cache capacity
 * - Block size: odd C (block_size=1) vs even C (block_size=2)
 * - Scale: small to very large C dimensions
 *
 * @param train_gamma With false, gamma is frozen: the kernel skips dL/dgamma and gamma gets no gradient.
 */
static void CompareKernelVsComposite(const std::vector<uint32_t>& shape, const bool train_gamma = true) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    float eps = 0.0078125F;

    // Generate random input data
    std::array<uint32_t, 4> gamma_shape = {1, 1, 1, shape[3]};
    auto rng = autograd::ctx().get_generator();
    uint32_t seed1 = rng();
    xt::xarray<float> x_data = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed1);
    uint32_t seed2 = rng();
    xt::xarray<float> gamma_data = ttml::test_utils::make_uniform_xarray<float>(gamma_shape, 0.0F, 1.0F, seed2);

    // Test forward pass - kernel vs composite
    auto x_kernel = autograd::create_tensor(core::from_xtensor(x_data, device), /* requires_grad */ true);
    auto gamma_kernel = autograd::create_tensor(core::from_xtensor(gamma_data, device), train_gamma);
    auto result_kernel = ops::rmsnorm(x_kernel, gamma_kernel, eps);
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    auto x_composite = autograd::create_tensor(core::from_xtensor(x_data, device), /* requires_grad */ true);
    auto gamma_composite = autograd::create_tensor(core::from_xtensor(gamma_data, device), train_gamma);
    auto result_composite = ops::rmsnorm_composite(x_composite, gamma_composite, eps);
    auto result_composite_xtensor = core::to_xtensor(result_composite->get_value());

    // Verify output shape matches input shape
    EXPECT_EQ(result_kernel_xtensor.shape(), x_data.shape());
    EXPECT_EQ(result_composite_xtensor.shape(), x_data.shape());

    // Compare forward results
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_composite_xtensor, 4e-2F, 3e-2F));
    EXPECT_TRUE(xt::all(xt::isfinite(result_kernel_xtensor)));
    EXPECT_TRUE(xt::all(xt::isfinite(result_composite_xtensor)));

    // Test backward pass - kernel vs composite
    auto target_composite = autograd::create_tensor(core::zeros_like(result_composite->get_value()));
    auto mse_composite = ops::mse_loss(result_composite, target_composite);
    mse_composite->backward();
    auto target_kernel = autograd::create_tensor(core::zeros_like(result_kernel->get_value()));
    auto mse_kernel = ops::mse_loss(result_kernel, target_kernel);
    mse_kernel->backward();

    // Since composite is finite, the kernel should also be finite
    auto x_grad_composite = core::to_xtensor(x_composite->get_grad());
    auto x_grad_kernel = core::to_xtensor(x_kernel->get_grad());

    // Verify gradients have correct shapes and are finite
    EXPECT_EQ(x_grad_kernel.shape(), x_data.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_kernel)));
    EXPECT_EQ(x_grad_composite.shape(), x_data.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_composite)));

    // Compare backward results
    EXPECT_TRUE(xt::allclose(x_grad_kernel, x_grad_composite, 1.0e-3F, 2e-3F));

    ASSERT_EQ(gamma_kernel->is_grad_initialized(), train_gamma);
    ASSERT_EQ(gamma_composite->is_grad_initialized(), train_gamma);
    if (train_gamma) {
        auto gamma_grad_composite = core::to_xtensor(gamma_composite->get_grad());
        auto gamma_grad_kernel = core::to_xtensor(gamma_kernel->get_grad());
        EXPECT_EQ(gamma_grad_kernel.shape()[3], shape[3]);
        EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_kernel)));
        EXPECT_EQ(gamma_grad_composite.shape()[3], shape[3]);
        EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_composite)));
        EXPECT_TRUE(xt::allclose(gamma_grad_kernel, gamma_grad_composite, 1.0e-3F, 2e-3F));
    }

    autograd::ctx().reset_graph();
}

// ============================================================================
// Section 3: Test Cases - RMSNorm Kernel vs Composite Comparison
// ============================================================================
// These tests systematically compare the optimized kernel implementation
// against the composite implementation across different scenarios:
//
// - Memory usage patterns: L1 cache fit vs overflow
// - Tensor alignment: 32-byte aligned vs unaligned (masking required)
// - Block sizes: odd vs even C dimensions
// - Scale testing: small to very large tensor dimensions
// - Training scenarios: realistic model shapes (NanoLlama, etc.)
// ============================================================================

TEST_F(RMSNormOpTest, RMSNorm_Compare_Basic_Small) {
    CompareKernelVsComposite({1U, 1U, 2U, 32U});
}

// Test aligned dimensions (C % 32 == 0) that fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_FitsInL1) {
    // C = 1024 (32 * 32), fits in L1 cache
    CompareKernelVsComposite({1U, 1U, 1U, 1024U});

    // C = 4096 (32 * 128), largest size that fits in L1 cache (1 << 12)
    CompareKernelVsComposite({1U, 1U, 1U, 4096U});
}

// Regression: Qwen3-32B hidden size. C = 5120 (Wt = 160) previously tripped the backward's
// under-counted L1 fit-check into the "everything fits in L1" path and then OOM'd on CB allocation.
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_Qwen3_32B_Hidden) {
    CompareKernelVsComposite({1U, 1U, 32U, 5120U});
}

// Test aligned dimensions (C % 32 == 0) that fit in L1 except for gamma
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_L1ExceptGamma) {
    // C = 8192 (1 << 13), fits in L1 except gamma parameter
    CompareKernelVsComposite({1U, 1U, 1U, 8192U});
}

// Test aligned dimensions (C % 32 == 0) that don't fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_DoesNotFitInL1) {
    // C = 16384 (1 << 14), does not fit in L1 cache
    CompareKernelVsComposite({1U, 1U, 1U, 16384U});
}

// Test aligned dimensions (C % 32 == 0) with very large C
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_VeryLargeC) {
    // C = 1048576 (1 << 20), very large C dimension (1M elements)
    CompareKernelVsComposite({1U, 1U, 1U, 1048576U});
}

// Test unaligned dimensions (C % 32 != 0) that fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_FitsInL1) {
    // C = 1023 (32 * 31 + 31), requires masking, fits in L1
    CompareKernelVsComposite({1U, 1U, 1U, 1023U});

    // C = 4095 (32 * 127 + 31), requires masking, fits in L1
    CompareKernelVsComposite({1U, 1U, 1U, 4095U});
}

// Test unaligned dimensions (C % 32 != 0) that don't fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_DoesNotFitInL1) {
    // C = 16383 (1 << 14 - 1), requires masking, does not fit in L1
    CompareKernelVsComposite({1U, 1U, 1U, 16383U});
}

// Test unaligned dimensions (C % 32 != 0) with very large C
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_VeryLargeC) {
    // C = 1048575 (1 << 20 - 1), very large C with masking
    CompareKernelVsComposite({1U, 1U, 1U, 1048575U});

    // C = 1048558 (1 << 20 - 18), very large C with different masking pattern
    CompareKernelVsComposite({1U, 1U, 1U, 1048558U});
}

// Test block_size = 1 (C is odd)
TEST_F(RMSNormOpTest, RMSNorm_Compare_BlockSize1_OddC) {
    CompareKernelVsComposite({1U, 1U, 1U, 33U});   // C = 33 (odd)
    CompareKernelVsComposite({1U, 1U, 1U, 127U});  // C = 127 (odd)
}

// Test block_size = 2 (C is even)
TEST_F(RMSNormOpTest, RMSNorm_Compare_BlockSize2_EvenC) {
    CompareKernelVsComposite({1U, 1U, 1U, 34U});   // C = 34 (even)
    CompareKernelVsComposite({1U, 1U, 1U, 126U});  // C = 126 (even)
}

// Frozen gamma (e.g. under LoRA): the backward skips dL/dgamma, on the resident and the streaming path. add_grad drops
// the gradient of a frozen gamma by itself, so the program cache is what shows the skip: after a trained run of the
// same shape, the frozen run compiles exactly one program, the backward without dL/dgamma.
TEST_F(RMSNormOpTest, RMSNorm_Compare_FrozenGamma) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    for (const auto& shape : {std::vector<uint32_t>{1U, 1U, 32U, 1024U}, std::vector<uint32_t>{1U, 1U, 1U, 16384U}}) {
        CompareKernelVsComposite(shape, /* train_gamma */ true);
        const auto num_programs = device->num_program_cache_entries();
        CompareKernelVsComposite(shape, /* train_gamma */ false);
        EXPECT_EQ(device->num_program_cache_entries(), num_programs + 1U) << "C = " << shape[3];
    }
}

// Test training-like shapes with NanoLlama dimensions
TEST_F(RMSNormOpTest, RMSNorm_Compare_TrainingShapes_NanoLlama) {
    // NanoLlama training shape: batch=64, seq_len=256, hidden_dim=384
    CompareKernelVsComposite({64U, 1U, 256U, 384U});
}

// Test training-like shapes with LLaMA 7B dimensions
TEST_F(RMSNormOpTest, RMSNorm_Compare_TrainingShapes_NanoGPT) {
    CompareKernelVsComposite({1U, 1U, 512U, 4096U});
}

// Test small batch and sequence dimensions (non-1 values)
TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Compare_SmallBatch_NonUnit) {
    CompareKernelVsComposite({2U, 1U, 4U, 64U});
    CompareKernelVsComposite({32U, 1U, 64U, 128U});
}

// Test different masking patterns with larger batches
TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Compare_Masking_Patterns) {
    CompareKernelVsComposite({32U, 1U, 1024U, 4091U});  // C % 32 = 11
    CompareKernelVsComposite({32U, 1U, 1024U, 4079U});  // C % 32 = 31
    CompareKernelVsComposite({32U, 1U, 1024U, 4097U});  // C % 32 = 1
}

// ============================================================================
// Section 4: RMSNorm Kernel vs Double-Precision Reference, Plain and Gated
// ============================================================================
// rmsnorm_fw/rmsnorm_bw against a host reference with bf16 and fp32 input and with dL/dgamma on and off. Gated mode
// computes rmsnorm(x) * gamma * silu(gate) per V-wide head: x and dx are head-major [B*H, 1, T, V], while gate,
// dL_dout, dgate and the output are head-merged [B, 1, T, H*V]. The plain mode is H = 1 without a gate.
// ============================================================================

namespace {

constexpr float kEps = 1e-6F;
// Max error over max |expected|, per index of dim 0. bf16 rounding alone is up to 2^-8 of the peak, so 1e-2 is ~2.5x
// that. dx also carries the bf16 1/rms error (up to 2^-7), so its budget is 2^-8 + 2^-7 ~ 1.2e-2.
constexpr double kTol = 1e-2;
constexpr double kDxTol = 1.5e-2;

struct Shape {
    uint32_t batch;
    uint32_t seq;
    uint32_t heads;
    uint32_t head_dim;

    [[nodiscard]] std::vector<uint32_t> head_major() const {
        return {batch * heads, 1U, seq, head_dim};
    }
    [[nodiscard]] std::vector<uint32_t> head_merged() const {
        return {batch, 1U, seq, heads * head_dim};
    }
    [[nodiscard]] std::vector<uint32_t> rms() const {
        return {batch * heads, 1U, seq, 1U};
    }
    [[nodiscard]] size_t head_major_index(
        const uint32_t b, const uint32_t h, const uint32_t t, const uint32_t c) const {
        return ((static_cast<size_t>(b) * heads + h) * seq + t) * head_dim + c;
    }
    [[nodiscard]] size_t head_merged_index(
        const uint32_t b, const uint32_t h, const uint32_t t, const uint32_t c) const {
        return ((static_cast<size_t>(b) * seq + t) * heads + h) * head_dim + c;
    }
};

struct Inputs {
    xt::xarray<float> x;
    std::optional<xt::xarray<float>> gate;  // nullopt in plain mode
    xt::xarray<float> gamma;
    xt::xarray<float> dy;
};

struct Reference {
    xt::xarray<double> out;
    xt::xarray<double> rms;
    xt::xarray<double> dx;
    xt::xarray<double> dgate;
    xt::xarray<double> dgamma;
};

ttnn::Tensor to_device(const xt::xarray<float>& a, const ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    auto* device = &ttml::autograd::ctx().get_device();
    return dtype == ttnn::DataType::FLOAT32 ? ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(a, device)
                                            : ttml::core::from_xtensor(a, device);
}

std::optional<ttnn::Tensor> to_device(const std::optional<xt::xarray<float>>& a) {
    return a.has_value() ? std::make_optional(to_device(*a)) : std::nullopt;
}

xt::xarray<float> as_stored(const xt::xarray<float>& a, const ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    return ttml::core::to_xtensor(to_device(a, dtype));
}

// Heads get RMS spread over 2^-2 .. 2^3, so normalizing over the wrong columns cannot pass. The gate reaches into
// both saturation tails of sigmoid.
Inputs make_inputs(const Shape& s, const uint32_t seed, const ttnn::DataType x_dtype, const bool gated) {
    auto x = ttml::test_utils::make_uniform_xarray<float>(s.head_major(), -1.0F, 1.0F, seed);
    for (uint32_t b = 0; b < s.batch; ++b) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            const float scale = std::exp2(static_cast<float>(static_cast<int>(h % 6U) - 2));
            for (uint32_t t = 0; t < s.seq; ++t) {
                for (uint32_t c = 0; c < s.head_dim; ++c) {
                    x.data()[s.head_major_index(b, h, t, c)] *= scale;
                }
            }
        }
    }
    std::optional<xt::xarray<float>> gate;
    if (gated) {
        gate = as_stored(ttml::test_utils::make_uniform_xarray<float>(s.head_merged(), -20.0F, 20.0F, seed + 1U));
    }
    return Inputs{
        .x = as_stored(x, x_dtype),
        .gate = std::move(gate),
        .gamma = as_stored(ttml::test_utils::make_uniform_xarray<float>(
            std::vector<uint32_t>{1U, 1U, 1U, s.head_dim}, 0.5F, 1.5F, seed + 2U)),
        .dy = as_stored(ttml::test_utils::make_uniform_xarray<float>(s.head_merged(), -1.0F, 1.0F, seed + 3U)),
    };
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

Reference reference(const Inputs& in, const xt::xarray<float>& dy, const Shape& s) {
    Reference ref{
        .out = xt::zeros<double>(s.head_merged()),
        .rms = xt::zeros<double>(s.rms()),
        .dx = xt::zeros<double>(s.head_major()),
        .dgate = xt::zeros<double>(s.head_merged()),
        .dgamma = xt::zeros<double>(in.gamma.shape()),
    };
    const uint32_t V = s.head_dim;
    std::vector<double> u(V);
    std::vector<double> du(V);
    for (uint32_t b = 0; b < s.batch; ++b) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            for (uint32_t t = 0; t < s.seq; ++t) {
                double sum_sq = 0.0;
                for (uint32_t c = 0; c < V; ++c) {
                    const double x = in.x.data()[s.head_major_index(b, h, t, c)];
                    sum_sq += x * x;
                }
                const double rms = std::sqrt(sum_sq / V + kEps);
                const double inv = 1.0 / rms;
                ref.rms.data()[(static_cast<size_t>(b) * s.heads + h) * s.seq + t] = rms;

                double dot = 0.0;
                for (uint32_t c = 0; c < V; ++c) {
                    const size_t merged = s.head_merged_index(b, h, t, c);
                    const double z = in.gate.has_value() ? in.gate->data()[merged] : 0.0;
                    const double silu = in.gate.has_value() ? z * sigmoid(z) : 1.0;
                    u[c] = in.x.data()[s.head_major_index(b, h, t, c)] * in.gamma.data()[c] * inv;
                    du[c] = dy.data()[merged] * silu;
                    dot += u[c] * du[c];
                    ref.out.data()[merged] = u[c] * silu;
                }
                const double scale = dot * inv / V;
                for (uint32_t c = 0; c < V; ++c) {
                    const size_t major = s.head_major_index(b, h, t, c);
                    const size_t merged = s.head_merged_index(b, h, t, c);
                    const double x = in.x.data()[major];
                    ref.dx.data()[major] = inv * (in.gamma.data()[c] * du[c] - scale * x);
                    ref.dgamma.data()[c] += x * inv * du[c];
                    if (in.gate.has_value()) {
                        const double z = in.gate->data()[merged];
                        const double sg = sigmoid(z);
                        ref.dgate.data()[merged] = dy.data()[merged] * u[c] * sg * (1.0 + z - z * sg);
                    }
                }
            }
        }
    }
    return ref;
}

// Bounds the error per index of dim 0: one head for the head-major rms and dx, one batch for the head-merged out and
// dgate, the whole tensor for dgamma.
template <typename Got>
void expect_close(
    const Got& got, const xt::xarray<double>& expected, const std::string& label, const double tol = kTol) {
    ASSERT_EQ(got.shape(), expected.shape()) << label;
    const size_t num_slices = expected.shape()[0];
    const size_t slice_size = expected.size() / num_slices;
    for (size_t slice = 0; slice < num_slices; ++slice) {
        double peak = 0.0;
        double max_err = 0.0;
        for (size_t i = slice * slice_size; i < (slice + 1U) * slice_size; ++i) {
            peak = std::max(peak, std::abs(expected.data()[i]));
            const double err = std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]);
            // Negated so that a NaN error is kept and fails the check below.
            if (!(err <= max_err)) {
                max_err = err;
            }
        }
        EXPECT_LE(max_err, tol * peak) << label << " [" << slice << "]: max error " << max_err << " vs peak " << peak;
    }
}

// In the negative tail of the gate dgate is ~1e-7 of the peak, far below what expect_close resolves, so dgate is also
// bounded per element relative to itself, by kDxTol: like dx, each element carries the bf16 1/rms error on top of its
// own bf16 rounding. The floor skips zeros and values below the bf16 normal range.
template <typename Got>
void expect_dgate_close(const Got& got, const xt::xarray<double>& expected, const std::string& label) {
    constexpr double kFloor = 1e-30;
    expect_close(got, expected, label);
    double max_rel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double magnitude = std::abs(expected.data()[i]);
        if (magnitude < kFloor) {
            continue;
        }
        const double rel = std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]) / magnitude;
        if (!(rel <= max_rel)) {
            max_rel = rel;
            worst = i;
        }
    }
    EXPECT_LE(max_rel, kDxTol) << label << ": max relative error " << max_rel << " at " << worst << ", expected "
                               << expected.data()[worst] << ", got " << got.data()[worst];
}

struct RMSNormCase {
    Shape shape;
    bool gated = false;
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16;
};

std::string rmsnorm_case_name(const ::testing::TestParamInfo<RMSNormCase>& info) {
    const auto& c = info.param;
    return std::string(c.gated ? "Gated_" : "Plain_") + std::to_string(c.shape.batch) + "x" +
           std::to_string(c.shape.seq) + "x" + std::to_string(c.shape.heads) + "x" + std::to_string(c.shape.head_dim) +
           (c.dtype == ttnn::DataType::FLOAT32 ? "_fp32" : "_bf16");
}

std::vector<RMSNormCase> rmsnorm_cases() {
    const std::vector<Shape> gated_shapes = {
        // V / 32 of 1, 2, 4 and 8 covers heads of 1 to 8 tiles.
        Shape{1, 32, 1, 32},
        Shape{1, 64, 3, 128},
        Shape{2, 96, 4, 64},
        Shape{1, 64, 2, 256},
        // A single head may be any width; V = 80 also covers the partial last tile.
        Shape{2, 64, 1, 80},
        // V = 96: block size 3 in the forward and 1 in the backward.
        Shape{1, 64, 4, 96},
        // T is not a multiple of 32.
        Shape{2, 50, 3, 128},
        // The Qwen3.8 TP=4 layout. Its 384 tile-rows don't divide evenly over the cores, so both core groups run and
        // every core processes more than one row.
        Shape{1, 1024, 12, 128},
    };
    // The plain mode as [B, 1, T, C], with C = head_dim.
    const std::vector<Shape> plain_shapes = {
        // C % 32 != 0 masks the last tile.
        Shape{1, 32, 1, 1000},
        // With fp32 input the forward keeps everything but gamma in L1 and the backward streams.
        Shape{1, 32, 1, 8192},
        // With fp32 input both stream.
        Shape{1, 32, 1, 16384},
    };
    std::vector<RMSNormCase> cases;
    for (const auto dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        for (const auto& s : gated_shapes) {
            cases.push_back({.shape = s, .gated = true, .dtype = dtype});
        }
        for (const auto& s : plain_shapes) {
            cases.push_back({.shape = s, .gated = false, .dtype = dtype});
        }
    }
    return cases;
}

}  // namespace

class RMSNormOpParamTest : public RMSNormOpTest, public ::testing::WithParamInterface<RMSNormCase> {};

TEST_P(RMSNormOpParamTest, MatchesReference) {
    const auto& c = GetParam();
    const auto in = make_inputs(c.shape, 11U, c.dtype, c.gated);
    const auto ref = reference(in, in.dy, c.shape);
    const auto x = to_device(in.x, c.dtype);
    const auto gate = to_device(in.gate);
    const auto gamma = to_device(in.gamma);
    const auto dy = to_device(in.dy);

    const auto fw = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, gate);
    expect_close(ttml::core::to_xtensor(fw[0].value()), ref.out, "out");
    expect_close(ttml::core::to_xtensor(fw[1].value()), ref.rms, "rms");

    for (const bool compute_dgamma : {true, false}) {
        const std::string suffix = compute_dgamma ? "" : " without dgamma";
        const auto bw = ttml::metal::rmsnorm_bw(x, gamma, fw[1].value(), dy, compute_dgamma, gate);
        ASSERT_EQ(bw.size(), 3U);
        expect_close(ttml::core::to_xtensor(bw[0].value()), ref.dx, "dx" + suffix, kDxTol);
        ASSERT_EQ(bw[1].has_value(), compute_dgamma);
        if (compute_dgamma) {
            expect_close(ttml::core::to_xtensor(bw[1].value()), ref.dgamma, "dgamma");
        }
        ASSERT_EQ(bw[2].has_value(), c.gated);
        if (c.gated) {
            expect_dgate_close(ttml::core::to_xtensor(bw[2].value()), ref.dgate, "dgate" + suffix);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(RMSNormModes, RMSNormOpParamTest, ::testing::ValuesIn(rmsnorm_cases()), rmsnorm_case_name);

TEST_F(RMSNormOpTest, GatedForwardWithoutIntermediates) {
    const Shape s{2, 50, 3, 128};
    const auto in = make_inputs(s, 17U, ttnn::DataType::BFLOAT16, /* gated */ true);
    const auto ref = reference(in, in.dy, s);

    const auto fw = ttml::metal::rmsnorm_fw(
        to_device(in.x), to_device(in.gamma), /* return_intermediates */ false, kEps, to_device(in.gate));
    ASSERT_EQ(fw.size(), 2U);
    EXPECT_FALSE(fw[1].has_value());
    expect_close(ttml::core::to_xtensor(fw[0].value()), ref.out, "out");
}

// Preallocated launches that hit cached programs must still read and write the new buffers, with and without
// dL_dgamma_components.
TEST_F(RMSNormOpTest, GatedPreallocatedOutputsOnCachedProgram) {
    const Shape s{1, 64, 3, 128};
    constexpr auto dtype = ttnn::DataType::FLOAT32;
    auto* device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    for (const bool compute_dgamma : {true, false}) {
        const std::string suffix = compute_dgamma ? "" : " without dgamma";
        // Caches the programs with launches on other inputs into fresh outputs. Everything stays alive, so no buffer
        // below can reuse their addresses.
        const auto num_programs_before = device->num_program_cache_entries();
        const auto first = make_inputs(s, 29U, dtype, /* gated */ true);
        const auto first_x = to_device(first.x, dtype);
        const auto first_gate = to_device(first.gate);
        const auto first_gamma = to_device(first.gamma);
        const auto first_dy = to_device(first.dy);
        const auto first_fw =
            ttml::metal::rmsnorm_fw(first_x, first_gamma, /* return_intermediates */ true, kEps, first_gate);
        const auto first_bw =
            ttml::metal::rmsnorm_bw(first_x, first_gamma, first_fw[1].value(), first_dy, compute_dgamma, first_gate);
        const auto num_programs = device->num_program_cache_entries();
        // Else the program cache is off and the unchanged count below proves nothing.
        ASSERT_GT(num_programs, num_programs_before) << "program cache not populated" << suffix;

        const auto in = make_inputs(s, 31U, dtype, /* gated */ true);
        const auto ref = reference(in, in.dy, s);
        const auto x = to_device(in.x, dtype);
        const auto gate = to_device(in.gate);
        const auto gamma = to_device(in.gamma);
        const auto dy = to_device(in.dy);
        const auto out = ttml::core::zeros(ttnn::Shape(s.head_merged()), device);
        const auto rms = ttml::core::zeros(ttnn::Shape(s.rms()), device);
        const auto da = ttml::core::zeros(ttnn::Shape(s.head_major()), device);
        const auto dgamma_components =
            compute_dgamma ? std::make_optional(ttml::core::zeros(ttnn::Shape(s.head_major()), device)) : std::nullopt;
        const auto dgate = ttml::core::zeros(ttnn::Shape(s.head_merged()), device);

        const auto fw = ttnn::prim::ttml_rmsnorm_fw(x, gamma, gate, /* return_intermediates */ true, kEps, rms, out);
        const auto bw =
            ttnn::prim::ttml_rmsnorm_bw(x, gamma, rms, dy, gate, compute_dgamma, da, dgamma_components, dgate);
        EXPECT_EQ(device->num_program_cache_entries(), num_programs) << suffix;

        EXPECT_EQ(fw[0].buffer()->address(), out.buffer()->address());
        EXPECT_EQ(fw[1].buffer()->address(), rms.buffer()->address());
        EXPECT_EQ(bw[0]->buffer()->address(), da.buffer()->address());
        EXPECT_EQ(bw[2]->buffer()->address(), dgate.buffer()->address());
        expect_close(ttml::core::to_xtensor(out), ref.out, "preallocated out" + suffix);
        expect_close(ttml::core::to_xtensor(rms), ref.rms, "preallocated rms" + suffix);
        expect_close(ttml::core::to_xtensor(da), ref.dx, "preallocated dx" + suffix, kDxTol);
        expect_close(ttml::core::to_xtensor(dgate), ref.dgate, "preallocated dgate" + suffix);
        ASSERT_EQ(bw[1].has_value(), compute_dgamma);
        if (compute_dgamma) {
            EXPECT_EQ(bw[1]->buffer()->address(), dgamma_components->buffer()->address());
            const xt::xarray<double> components = xt::cast<double>(ttml::core::to_xtensor(*dgamma_components));
            const xt::xarray<double> dgamma = xt::sum(components, {0, 1, 2}, xt::keep_dims);
            expect_close(dgamma, ref.dgamma, "preallocated dgamma");
        }
    }
}

// add_grad drops the gradient of a frozen gamma by itself, so the program cache is what shows that the frozen iteration
// ran the backward without dL/dgamma: it compiles exactly the one program the trained iteration did not.
TEST_F(RMSNormOpTest, GatedAutogradSkipsFrozenGamma) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    device->clear_program_cache();
    for (const bool train_gamma : {true, false}) {
        const Shape s{1, 64, 3, 128};
        const auto in = make_inputs(s, 37U, ttnn::DataType::FLOAT32, /* gated */ true);
        auto x = autograd::create_tensor(to_device(in.x, ttnn::DataType::FLOAT32), /* requires_grad */ true);
        auto gate = autograd::create_tensor(to_device(*in.gate), /* requires_grad */ true);
        auto gamma = autograd::create_tensor(to_device(in.gamma), /* requires_grad */ train_gamma);

        const auto num_programs = device->num_program_cache_entries();
        auto out = ops::gated_rmsnorm(x, gate, gamma, kEps);
        out->backward();
        if (!train_gamma) {
            EXPECT_EQ(device->num_program_cache_entries(), num_programs + 1U);
        }

        const xt::xarray<float> ones = xt::ones<float>(s.head_merged());  // backward() seeds dL/dout with ones
        const auto ref = reference(in, ones, s);
        expect_close(core::to_xtensor(out->get_value()), ref.out, "autograd out");
        ASSERT_TRUE(x->is_grad_initialized());
        ASSERT_TRUE(gate->is_grad_initialized());
        expect_close(core::to_xtensor(x->get_grad()), ref.dx, "autograd dx", kDxTol);
        expect_dgate_close(core::to_xtensor(gate->get_grad()), ref.dgate, "autograd dgate");
        ASSERT_EQ(gamma->is_grad_initialized(), train_gamma);
        if (train_gamma) {
            expect_close(core::to_xtensor(gamma->get_grad()), ref.dgamma, "autograd dgamma");
        }
        autograd::ctx().reset_graph();
    }
}

TEST_F(RMSNormOpTest, GatedRejectsMismatchedBackwardTensors) {
    const Shape s{1, 64, 3, 128};
    const auto in = make_inputs(s, 43U, ttnn::DataType::BFLOAT16, /* gated */ true);
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = to_device(in.x);
    const auto gate = to_device(in.gate);
    const auto gamma = to_device(in.gamma);
    const auto dy = to_device(in.dy);
    const auto rms = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, gate)[1].value();

    // In gated mode dL_dout follows the gate's layout, not the input's.
    EXPECT_THAT(
        ([&] { return ttml::metal::rmsnorm_bw(x, gamma, rms, x, /* compute_dgamma */ true, gate); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("dL_dout shape")));
    // dL_dgate only exists in gated mode.
    EXPECT_THAT(
        ([&] {
            return ttnn::prim::ttml_rmsnorm_bw(
                x,
                gamma,
                rms,
                x,
                /* gate */ std::nullopt,
                /* compute_dgamma */ true,
                std::nullopt,
                std::nullopt,
                ttml::core::zeros(ttnn::Shape(s.head_merged()), device));
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("given without a gate")));
    // No dL_dgamma_components are written without compute_dgamma.
    EXPECT_THAT(
        ([&] {
            return ttnn::prim::ttml_rmsnorm_bw(
                x,
                gamma,
                rms,
                dy,
                gate,
                /* compute_dgamma */ false,
                std::nullopt,
                ttml::core::zeros(ttnn::Shape(s.head_major()), device),
                std::nullopt);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("compute_dgamma is false")));
}
