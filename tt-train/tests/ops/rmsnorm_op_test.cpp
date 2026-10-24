// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <umd/device/cluster.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ops/losses.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class RMSNormOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
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
 */
static void CompareKernelVsComposite(const std::vector<uint32_t>& shape) {
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
    auto gamma_kernel = autograd::create_tensor(core::from_xtensor(gamma_data, device), /* requires_grad */ true);
    auto result_kernel = ops::rmsnorm(x_kernel, gamma_kernel, eps);
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    auto x_composite = autograd::create_tensor(core::from_xtensor(x_data, device), /* requires_grad */ true);
    auto gamma_composite = autograd::create_tensor(core::from_xtensor(gamma_data, device), /* requires_grad */ true);
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
    auto gamma_grad_composite = core::to_xtensor(gamma_composite->get_grad());
    auto x_grad_kernel = core::to_xtensor(x_kernel->get_grad());
    auto gamma_grad_kernel = core::to_xtensor(gamma_kernel->get_grad());
    // Should be composite here

    // Verify gradients have correct shapes and are finite
    EXPECT_EQ(x_grad_kernel.shape(), x_data.shape());
    EXPECT_EQ(gamma_grad_kernel.shape()[3], shape[3]);
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_kernel)));
    EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_kernel)));

    EXPECT_EQ(x_grad_composite.shape(), x_data.shape());
    EXPECT_EQ(gamma_grad_composite.shape()[3], shape[3]);
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_composite)));
    EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_composite)));

    // Compare backward results
    EXPECT_TRUE(xt::allclose(x_grad_kernel, x_grad_composite, 1.0e-3F, 2e-3F));
    EXPECT_TRUE(xt::allclose(gamma_grad_kernel, gamma_grad_composite, 1.0e-3F, 2e-3F));

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
// Section 4: rmsnorm_fw / rmsnorm_bw program-cache hits
// ============================================================================
// A launch whose specs and attributes match a cached program reuses it, and only the buffer addresses
// of the new tensors are patched in. Each test starts from an empty cache so that its first launch is the miss, and
// keeps every tensor of the earlier launches alive so that a later launch cannot land on a freed address: a
// hit must read and write the new buffers and leave the earlier ones untouched. Launch k draws its
// activations from [-(k + 1), k + 1] and flips the sign of gamma on odd k, so a launch that reads another
// launch's buffers lands far outside the tolerance.
// ============================================================================

class RMSNormCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

namespace {

using ttml::test_utils::CacheExpectation;
using ttml::test_utils::expect_cache;

// Both ops split the B * N * Ht tile rows over the cores, so two_core_group_rows_shape gives them two core groups.
constexpr uint32_t kCacheTestC = 64U;
// Pads to the same two tiles as kCacheTestC, but the kernels mask the last 24 columns and scale by 1/40.
constexpr uint32_t kMaskedC = 40U;
// 2^-7 is exact in bf16; the reader kernel gets epsilon truncated to bf16.
constexpr float kCacheTestEps = 0.0078125F;
// CompareKernelVsComposite's forward tolerances. dL_dout is O(1) here rather than an MSE gradient, so the
// backward outputs are O(1) as well and use the same tolerances.
constexpr float kCacheTestRtol = 4e-2F;
constexpr float kCacheTestAtol = 3e-2F;
// Initial value of every preallocated output: one the op does not write fails the reference check.
constexpr float kNotWritten = -1000.F;

// rms = sqrt(mean(x^2) + eps) over C; output = gamma * x / rms.
std::pair<xt::xarray<float>, xt::xarray<float>> rmsnorm_fw_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& gamma, float eps) {
    xt::xarray<float> rms = xt::sqrt(xt::mean(x * x, {3}, xt::keep_dims) + eps);
    xt::xarray<float> output = gamma * x / rms;
    return {std::move(output), std::move(rms)};
}

// For the rms the backward is given, with g = gamma * dL_dout / rms:
// dL_da = g - x * sum(x * g) / (C * rms^2) and dL_dgamma_components = dL_dout * x / rms.
std::pair<xt::xarray<float>, xt::xarray<float>> rmsnorm_bw_reference(
    const xt::xarray<float>& x,
    const xt::xarray<float>& gamma,
    const xt::xarray<float>& rms,
    const xt::xarray<float>& dL_dout) {
    const auto c = static_cast<float>(x.shape()[3]);
    xt::xarray<float> gained_dL_dout = gamma * dL_dout / rms;
    xt::xarray<float> scale = xt::sum(x * gained_dL_dout, {3}, xt::keep_dims);
    xt::xarray<float> dL_da = gained_dL_dout - x * scale / (c * rms * rms);
    xt::xarray<float> dL_dgamma_components = dL_dout * x / rms;
    return {std::move(dL_da), std::move(dL_dgamma_components)};
}

// A device tensor and the values the kernels read from it: the host values rounded to bf16.
struct StoredTensor {
    ttnn::Tensor tensor;
    xt::xarray<float> stored;
};

StoredTensor upload(const xt::xarray<float>& values) {
    auto tensor = ttml::core::from_xtensor(values, &ttml::autograd::ctx().get_device());
    auto stored = ttml::core::to_xtensor(tensor);
    return {std::move(tensor), std::move(stored)};
}

float x_range_of_launch(uint32_t launch) {
    return static_cast<float>(launch + 1U);
}

xt::xarray<float> make_gamma(uint32_t c, uint32_t launch) {
    const float sign = launch % 2U == 0U ? 1.F : -1.F;
    return sign * ttml::test_utils::make_uniform_xarray<float>(
                      std::array<uint32_t, 4>{1U, 1U, 1U, c}, 0.5F, 1.5F, 200U + launch);
}

struct ForwardInputs {
    StoredTensor x;
    StoredTensor gamma;
};

ForwardInputs make_forward_inputs(const std::vector<uint32_t>& shape, uint32_t launch, float x_range) {
    return {
        upload(ttml::test_utils::make_uniform_xarray<float>(shape, -x_range, x_range, 100U + launch)),
        upload(make_gamma(shape[3], launch))};
}

struct BackwardInputs {
    StoredTensor x;
    StoredTensor gamma;
    StoredTensor rms;
    StoredTensor dL_dout;
};

BackwardInputs make_backward_inputs(const std::vector<uint32_t>& shape, uint32_t launch) {
    const float range = x_range_of_launch(launch);
    const auto x = ttml::test_utils::make_uniform_xarray<float>(shape, -range, range, 300U + launch);
    const auto gamma = make_gamma(shape[3], launch);
    // What rmsnorm_fw returns as rms for x, as the backward gets it in training.
    const auto rms = rmsnorm_fw_reference(x, gamma, kCacheTestEps).second;
    const auto dL_dout = ttml::test_utils::make_uniform_xarray<float>(shape, -range, range, 400U + launch);
    return {upload(x), upload(gamma), upload(rms), upload(dL_dout)};
}

void expect_close(const ttnn::Tensor& result, const xt::xarray<float>& expected, const std::string& label) {
    const auto actual = ttml::core::to_xtensor(result);
    ASSERT_EQ(actual.shape(), expected.shape()) << label;
    EXPECT_TRUE(xt::allclose(actual, expected, kCacheTestRtol, kCacheTestAtol))
        << label << ": max abs diff " << xt::amax(xt::abs(actual - expected))();
}

void expect_rmsnorm_fw_matches_reference(
    const ttnn::Tensor& output,
    const std::optional<ttnn::Tensor>& rms,
    const ForwardInputs& in,
    float eps,
    const std::string& label) {
    const auto [expected_output, expected_rms] = rmsnorm_fw_reference(in.x.stored, in.gamma.stored, eps);
    expect_close(output, expected_output, label + " output");
    if (rms.has_value()) {
        expect_close(*rms, expected_rms, label + " rms");
    }
}

void expect_rmsnorm_bw_matches_reference(
    const ttnn::Tensor& dL_da,
    const ttnn::Tensor& dL_dgamma_components,
    const BackwardInputs& in,
    const std::string& label) {
    const auto [expected_dL_da, expected_dL_dgamma_components] =
        rmsnorm_bw_reference(in.x.stored, in.gamma.stored, in.rms.stored, in.dL_dout.stored);
    expect_close(dL_da, expected_dL_da, label + " dL_da");
    expect_close(dL_dgamma_components, expected_dL_dgamma_components, label + " dL_dgamma_components");
}

// The tensors of earlier launches and the values they must keep. Holding the tensors also keeps their
// buffers allocated, so that a later launch cannot land on their addresses.
class KeptTensors {
public:
    void add(const ttnn::Tensor& tensor, const std::string& label) {
        kept_.push_back({tensor, ttml::core::to_xtensor(tensor), label});
    }

    void add(const StoredTensor& input, const std::string& label) {
        kept_.push_back({input.tensor, input.stored, label});
    }

    void add_inputs(const ForwardInputs& in, const std::string& label) {
        add(in.x, label + " input");
        add(in.gamma, label + " gamma");
    }

    void add_inputs(const BackwardInputs& in, const std::string& label) {
        add(in.x, label + " input");
        add(in.gamma, label + " gamma");
        add(in.rms, label + " rms");
        add(in.dL_dout, label + " dL_dout");
    }

    void expect_unchanged() const {
        for (const auto& [tensor, values, label] : kept_) {
            ttml::test_utils::expect_unchanged(tensor, values, label);
        }
    }

private:
    struct Kept {
        ttnn::Tensor tensor;
        xt::xarray<float> values;
        std::string label;
    };
    std::vector<Kept> kept_;
};

// Launch A compiles the program; launch B, on new tensors of the same shape while A's stay alive, reuses it.
void run_rmsnorm_fw_cache_hit(bool return_intermediates) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    const auto a = make_forward_inputs(shape, 0U, x_range_of_launch(0U));
    const auto entries_before_miss = device->num_program_cache_entries();
    const auto a_result = metal::rmsnorm_fw(a.x.tensor, a.gamma.tensor, return_intermediates, kCacheTestEps);
    ASSERT_GT(device->num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    ASSERT_TRUE(a_result[0].has_value());
    ASSERT_EQ(a_result[1].has_value(), return_intermediates);
    expect_rmsnorm_fw_matches_reference(*a_result[0], a_result[1], a, kCacheTestEps, "launch A");
    kept.add_inputs(a, "launch A");
    kept.add(*a_result[0], "launch A output");
    if (return_intermediates) {
        kept.add(*a_result[1], "launch A rms");
    }

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_forward_inputs(shape, 1U, x_range_of_launch(1U));
    const auto entries_before_hit = device->num_program_cache_entries();
    const auto b_result = metal::rmsnorm_fw(b.x.tensor, b.gamma.tensor, return_intermediates, kCacheTestEps);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    ASSERT_TRUE(b_result[0].has_value());
    ASSERT_EQ(b_result[1].has_value(), return_intermediates);
    test_utils::expect_new_buffers(
        {a.x.tensor, a.gamma.tensor, *a_result[0]}, {b.x.tensor, b.gamma.tensor, *b_result[0]}, "launch B");
    if (return_intermediates) {
        EXPECT_NE(b_result[1]->buffer()->address(), a_result[1]->buffer()->address());
    }
    expect_rmsnorm_fw_matches_reference(*b_result[0], b_result[1], b, kCacheTestEps, "launch B (cache hit)");

    kept.expect_unchanged();
}

// Launches without and with the preallocated rms / output on one shape. Whether or not the program key
// records which optionals are present, the repeated patterns must hit, and every launch must write exactly
// the tensors it returns and leave the tensors of all other launches untouched. Without return_intermediates
// the op returns the output alone and ignores a preallocated rms, which must keep its fill.
void run_rmsnorm_fw_preallocated(bool return_intermediates) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = test_utils::two_core_group_rows_shape(kCacheTestC);
    auto rms_shape = shape;
    rms_shape[3] = 1U;
    KeptTensors kept;

    struct Launch {
        const char* label;
        bool preallocated_rms;
        bool preallocated_output;
        bool repeats_earlier_pattern;
    };
    constexpr std::array<Launch, 6> kLaunches = {{
        {"no preallocation", false, false, false},
        {"preallocated rms and output", true, true, false},
        {"preallocated rms and output again", true, true, true},
        {"preallocated output only", false, true, false},
        {"preallocated rms only", true, false, false},
        {"no preallocation again", false, false, true},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& pattern = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + pattern.label + ")";
        const auto in = make_forward_inputs(shape, launch, x_range_of_launch(launch));
        std::optional<ttnn::Tensor> preallocated_rms;
        std::optional<ttnn::Tensor> preallocated_output;
        if (pattern.preallocated_rms) {
            preallocated_rms = test_utils::filled_on_device(rms_shape, kNotWritten);
        }
        if (pattern.preallocated_output) {
            preallocated_output = test_utils::filled_on_device(shape, kNotWritten);
        }
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_fw(
            in.x.tensor, in.gamma.tensor, return_intermediates, kCacheTestEps, preallocated_rms, preallocated_output);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (pattern.repeats_earlier_pattern) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of its pattern";
        }
        ASSERT_EQ(result.size(), return_intermediates ? 2U : 1U) << label;
        if (preallocated_output.has_value()) {
            EXPECT_EQ(result[0].buffer()->address(), preallocated_output->buffer()->address())
                << label << " must return the preallocated output";
        }
        std::optional<ttnn::Tensor> rms;
        if (return_intermediates) {
            rms = result[1];
            if (preallocated_rms.has_value()) {
                EXPECT_EQ(rms->buffer()->address(), preallocated_rms->buffer()->address())
                    << label << " must return the preallocated rms";
            }
        }
        expect_rmsnorm_fw_matches_reference(result[0], rms, in, kCacheTestEps, label);
        kept.add_inputs(in, label);
        kept.add(result[0], label + " output");
        if (rms.has_value()) {
            kept.add(*rms, label + " rms");
        } else if (preallocated_rms.has_value()) {
            EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(*preallocated_rms), kNotWritten)))
                << label << " wrote the preallocated rms, which it does not return";
            kept.add(*preallocated_rms, label + " ignored rms");
        }
    }

    kept.expect_unchanged();
}

}  // namespace

TEST_F(RMSNormCacheHitTest, Forward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    run_rmsnorm_fw_cache_hit(/*return_intermediates=*/true);
}

// Without the rms output the op returns a single tensor, and the writer's rms address slot has no buffer of
// its own.
TEST_F(RMSNormCacheHitTest, Forward_NoIntermediates_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    run_rmsnorm_fw_cache_hit(/*return_intermediates=*/false);
}

// Alternates epsilon over fresh tensors. Whether epsilon is part of the program key or re-applied on a hit,
// each launch must normalize with its own epsilon, and launches 2 and 3 repeat the programs of launches 0 and 1.
TEST_F(RMSNormCacheHitTest, Forward_EpsilonChangeBetweenCachedPrograms_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    // With x in [-1, 1], mean(x^2) is about 1/3, so epsilon = 1 moves the rms by about 2x.
    constexpr std::array<float, 4> kEpsilons = {kCacheTestEps, 1.F, kCacheTestEps, 1.F};
    for (uint32_t launch = 0; launch < kEpsilons.size(); ++launch) {
        const std::string label =
            "launch " + std::to_string(launch) + " (epsilon " + std::to_string(kEpsilons[launch]) + ")";
        const auto in = make_forward_inputs(shape, launch, /*x_range=*/1.F);
        const auto entries_before = device->num_program_cache_entries();
        const auto result =
            metal::rmsnorm_fw(in.x.tensor, in.gamma.tensor, /*return_intermediates=*/true, kEpsilons[launch]);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (launch >= 2U) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of launch " << launch - 2U;
        }
        ASSERT_TRUE(result[0].has_value() && result[1].has_value()) << label;
        expect_rmsnorm_fw_matches_reference(*result[0], result[1], in, kEpsilons[launch], label);
        kept.add_inputs(in, label);
        kept.add(*result[0], label + " output");
        kept.add(*result[1], label + " rms");
    }

    kept.expect_unchanged();
}

TEST_F(RMSNormCacheHitTest, Forward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    run_rmsnorm_fw_preallocated(/*return_intermediates=*/true);
}

// The preallocated rms is still passed, but the op only returns the output. If its presence is not part of the
// program key, a program compiled without it is reused with it, and a writer that finds the output by position
// among the passed tensors writes into the preallocated rms instead.
TEST_F(RMSNormCacheHitTest, Forward_NoIntermediates_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    run_rmsnorm_fw_preallocated(/*return_intermediates=*/false);
}

// Launches whose program keys differ, one after another into one cache. C = 40 pads to the same two tiles as
// C = 64, but it needs the width mask and a 1/40 scaler. The number of tile rows sets the work split, and
// return_intermediates decides whether the writer also writes the rms. Each such launch must compile its own
// program: with a key that ignores one of these, it reuses a program built for the other value and its
// outputs are off. The one-tile-row shapes go first, so that a program reused for the two-core-group shape
// leaves most of its rows unwritten. The last launches repeat earlier keys and must hit their programs, the
// single-core one included.
TEST_F(RMSNormCacheHitTest, Forward_DistinctKeysCompileDistinctPrograms_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const std::vector<uint32_t> one_tile_row = {1U, 1U, 32U, kCacheTestC};
    const std::vector<uint32_t> one_tile_row_masked = {1U, 1U, 32U, kMaskedC};
    const auto two_core_groups = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    struct Launch {
        const char* label;
        std::vector<uint32_t> shape;
        bool return_intermediates;
        bool new_key;
    };
    const std::vector<Launch> launches = {
        {"one tile row", one_tile_row, true, true},
        {"one tile row, C = 40", one_tile_row_masked, true, true},
        {"two core groups", two_core_groups, true, true},
        {"two core groups, no rms", two_core_groups, false, true},
        {"one tile row, C = 40 again", one_tile_row_masked, true, false},
        {"two core groups again", two_core_groups, true, false},
        {"two core groups, no rms again", two_core_groups, false, false},
    };
    for (uint32_t launch = 0; launch < launches.size(); ++launch) {
        const auto& spec = launches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + spec.label + ")";
        const auto in = make_forward_inputs(spec.shape, launch, x_range_of_launch(launch));
        const auto entries_before = device->num_program_cache_entries();
        const auto result = metal::rmsnorm_fw(in.x.tensor, in.gamma.tensor, spec.return_intermediates, kCacheTestEps);
        expect_cache(entries_before, spec.new_key ? CacheExpectation::Miss : CacheExpectation::Hit, label);
        ASSERT_TRUE(result[0].has_value()) << label;
        ASSERT_EQ(result[1].has_value(), spec.return_intermediates) << label;
        expect_rmsnorm_fw_matches_reference(*result[0], result[1], in, kCacheTestEps, label);
        kept.add_inputs(in, label);
        kept.add(*result[0], label + " output");
        if (spec.return_intermediates) {
            kept.add(*result[1], label + " rms");
        }
    }

    kept.expect_unchanged();
}

TEST_F(RMSNormCacheHitTest, Backward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    const auto a = make_backward_inputs(shape, 0U);
    const auto entries_before_miss = device->num_program_cache_entries();
    const auto a_result = ttnn::prim::ttml_rmsnorm_bw(a.x.tensor, a.gamma.tensor, a.rms.tensor, a.dL_dout.tensor);
    ASSERT_GT(device->num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    ASSERT_EQ(a_result.size(), 2U);
    expect_rmsnorm_bw_matches_reference(a_result[0], a_result[1], a, "launch A");
    kept.add_inputs(a, "launch A");
    kept.add(a_result[0], "launch A dL_da");
    kept.add(a_result[1], "launch A dL_dgamma_components");

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_backward_inputs(shape, 1U);
    const auto entries_before_hit = device->num_program_cache_entries();
    const auto b_result = ttnn::prim::ttml_rmsnorm_bw(b.x.tensor, b.gamma.tensor, b.rms.tensor, b.dL_dout.tensor);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    ASSERT_EQ(b_result.size(), 2U);
    test_utils::expect_new_buffers(
        {a.x.tensor, a.gamma.tensor, a.rms.tensor, a.dL_dout.tensor, a_result[0], a_result[1]},
        {b.x.tensor, b.gamma.tensor, b.rms.tensor, b.dL_dout.tensor, b_result[0], b_result[1]},
        "launch B");
    expect_rmsnorm_bw_matches_reference(b_result[0], b_result[1], b, "launch B (cache hit)");

    kept.expect_unchanged();
}

// The backward counterpart of Forward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups, for the
// preallocated dL_da / dL_dgamma_components.
TEST_F(RMSNormCacheHitTest, Backward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    struct Launch {
        const char* label;
        bool preallocated_dL_da;
        bool preallocated_dL_dgamma_components;
        bool repeats_earlier_pattern;
    };
    constexpr std::array<Launch, 6> kLaunches = {{
        {"no preallocation", false, false, false},
        {"preallocated dL_da and dL_dgamma_components", true, true, false},
        {"preallocated dL_da and dL_dgamma_components again", true, true, true},
        {"preallocated dL_da only", true, false, false},
        {"preallocated dL_dgamma_components only", false, true, false},
        {"no preallocation again", false, false, true},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& pattern = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + pattern.label + ")";
        const auto in = make_backward_inputs(shape, launch);
        std::optional<ttnn::Tensor> preallocated_dL_da;
        std::optional<ttnn::Tensor> preallocated_dL_dgamma_components;
        if (pattern.preallocated_dL_da) {
            preallocated_dL_da = test_utils::filled_on_device(shape, kNotWritten);
        }
        if (pattern.preallocated_dL_dgamma_components) {
            preallocated_dL_dgamma_components = test_utils::filled_on_device(shape, kNotWritten);
        }
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_bw(
            in.x.tensor,
            in.gamma.tensor,
            in.rms.tensor,
            in.dL_dout.tensor,
            /*epsilon=*/1e-6F,
            preallocated_dL_da,
            preallocated_dL_dgamma_components);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (pattern.repeats_earlier_pattern) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of its pattern";
        }
        ASSERT_EQ(result.size(), 2U) << label;
        if (preallocated_dL_da.has_value()) {
            EXPECT_EQ(result[0].buffer()->address(), preallocated_dL_da->buffer()->address())
                << label << " must return the preallocated dL_da";
        }
        if (preallocated_dL_dgamma_components.has_value()) {
            EXPECT_EQ(result[1].buffer()->address(), preallocated_dL_dgamma_components->buffer()->address())
                << label << " must return the preallocated dL_dgamma_components";
        }
        expect_rmsnorm_bw_matches_reference(result[0], result[1], in, label);
        kept.add_inputs(in, label);
        kept.add(result[0], label + " dL_da");
        kept.add(result[1], label + " dL_dgamma_components");
    }

    kept.expect_unchanged();
}

// The backward counterpart of Forward_DistinctKeysCompileDistinctPrograms_TwoCoreGroups, over the input shape:
// the logical width at an equal padded width, then the number of tile rows, then both again.
TEST_F(RMSNormCacheHitTest, Backward_DistinctKeysCompileDistinctPrograms_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const std::vector<uint32_t> one_tile_row = {1U, 1U, 32U, kCacheTestC};
    const std::vector<uint32_t> one_tile_row_masked = {1U, 1U, 32U, kMaskedC};
    const auto two_core_groups = test_utils::two_core_group_rows_shape(kCacheTestC);
    KeptTensors kept;

    struct Launch {
        const char* label;
        std::vector<uint32_t> shape;
        bool new_key;
    };
    const std::vector<Launch> launches = {
        {"one tile row", one_tile_row, true},
        {"one tile row, C = 40", one_tile_row_masked, true},
        {"two core groups", two_core_groups, true},
        {"one tile row, C = 40 again", one_tile_row_masked, false},
        {"two core groups again", two_core_groups, false},
    };
    for (uint32_t launch = 0; launch < launches.size(); ++launch) {
        const auto& spec = launches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + spec.label + ")";
        const auto in = make_backward_inputs(spec.shape, launch);
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_bw(in.x.tensor, in.gamma.tensor, in.rms.tensor, in.dL_dout.tensor);
        expect_cache(entries_before, spec.new_key ? CacheExpectation::Miss : CacheExpectation::Hit, label);
        ASSERT_EQ(result.size(), 2U) << label;
        expect_rmsnorm_bw_matches_reference(result[0], result[1], in, label);
        kept.add_inputs(in, label);
        kept.add(result[0], label + " dL_da");
        kept.add(result[1], label + " dL_dgamma_components");
    }

    kept.expect_unchanged();
}
