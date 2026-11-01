// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_op.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tt-umd-workload/cluster.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ops/losses.hpp"
#include "test_utils/program_cache.hpp"
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

// ============================================================================
// Section 5: rmsnorm_fw / rmsnorm_bw program-cache hits
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

void expect_allclose(const ttnn::Tensor& result, const xt::xarray<float>& expected, const std::string& label) {
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
    expect_allclose(output, expected_output, label + " output");
    if (rms.has_value()) {
        expect_allclose(*rms, expected_rms, label + " rms");
    }
}

void expect_rmsnorm_bw_matches_reference(
    const ttnn::Tensor& dL_da,
    const ttnn::Tensor& dL_dgamma_components,
    const BackwardInputs& in,
    const std::string& label) {
    const auto [expected_dL_da, expected_dL_dgamma_components] =
        rmsnorm_bw_reference(in.x.stored, in.gamma.stored, in.rms.stored, in.dL_dout.stored);
    expect_allclose(dL_da, expected_dL_da, label + " dL_da");
    expect_allclose(dL_dgamma_components, expected_dL_dgamma_components, label + " dL_dgamma_components");
}

// The backward without a gate and with compute_dgamma returns {dL_da, dL_dgamma_components, nullopt}; the
// first two of them.
std::vector<ttnn::Tensor> plain_backward_outputs(
    const std::vector<std::optional<ttnn::Tensor>>& result, const std::string& label) {
    EXPECT_EQ(result.size(), 3U) << label;
    EXPECT_FALSE(result.size() > 2U && result[2].has_value()) << label << " returned a dL_dgate without a gate";
    return {result.at(0).value(), result.at(1).value()};
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
// the op returns the output alone and rejects a preallocated rms, so only the output is preallocated.
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
    const std::vector<Launch> launches = return_intermediates ? std::vector<Launch>{
        {"no preallocation", false, false, false},
        {"preallocated rms and output", true, true, false},
        {"preallocated rms and output again", true, true, true},
        {"preallocated output only", false, true, false},
        {"preallocated rms only", true, false, false},
        {"no preallocation again", false, false, true},
    } : std::vector<Launch>{
        {"no preallocation", false, false, false},
        {"preallocated output", false, true, false},
        {"preallocated output again", false, true, true},
        {"no preallocation again", false, false, true},
    };
    for (uint32_t launch = 0; launch < launches.size(); ++launch) {
        const auto& pattern = launches[launch];
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
            in.x.tensor,
            in.gamma.tensor,
            /*gate=*/std::nullopt,
            return_intermediates,
            kCacheTestEps,
            preallocated_rms,
            preallocated_output);
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
        }
    }

    if (!return_intermediates) {
        // With no rms to write, a preallocated one is rejected rather than ignored, also on a hit of the cached
        // program: the op has no hit-only validation, so ttnn validates hits as misses.
        const auto launch = static_cast<uint32_t>(launches.size());
        const auto in = make_forward_inputs(shape, launch, x_range_of_launch(launch));
        const auto preallocated_rms = test_utils::filled_on_device(rms_shape, kNotWritten);
        EXPECT_THAT(
            ([&] {
                return ttnn::prim::ttml_rmsnorm_fw(
                    in.x.tensor,
                    in.gamma.tensor,
                    /*gate=*/std::nullopt,
                    /*return_intermediates=*/false,
                    kCacheTestEps,
                    preallocated_rms,
                    /*preallocated_output=*/std::nullopt);
            }),
            ThrowsMessage<std::runtime_error>(HasSubstr("return_intermediates is false")));
        EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(preallocated_rms), kNotWritten)))
            << "the rejected launch wrote the preallocated rms";
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

// The op only returns the output, so only the output's presence flips, and a preallocated rms is rejected once the
// program is cached as well as before.
TEST_F(RMSNormCacheHitTest, Forward_NoIntermediates_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    run_rmsnorm_fw_preallocated(/*return_intermediates=*/false);
}

// Launches whose program keys differ, one after another into one cache. C = 40 pads to the same two tiles as
// C = 64, but it needs the width mask and a 1/40 scaler. The number of tile rows sets the work split, and
// return_intermediates decides whether the writer also writes the rms. Each such launch must compile its own
// program: with a key that ignores one of these, it reuses a program built for the other value and its
// outputs are off. The one-tile-row shapes go first, so that a program reused for the two-core-group shape
// leaves most of its rows unwritten, and the launch without rms goes before the one with it, so that a launch
// reusing the other's program leaves its rms unwritten rather than writing one through the rms slot's address 0.
// The last launches repeat earlier keys and must hit their programs, the single-core one included.
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
        {"two core groups, no rms", two_core_groups, false, true},
        {"two core groups", two_core_groups, true, true},
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
    const auto a_result = plain_backward_outputs(
        ttnn::prim::ttml_rmsnorm_bw(a.x.tensor, a.gamma.tensor, a.rms.tensor, a.dL_dout.tensor, /*gate=*/std::nullopt),
        "launch A");
    ASSERT_GT(device->num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    expect_rmsnorm_bw_matches_reference(a_result[0], a_result[1], a, "launch A");
    kept.add_inputs(a, "launch A");
    kept.add(a_result[0], "launch A dL_da");
    kept.add(a_result[1], "launch A dL_dgamma_components");

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_backward_inputs(shape, 1U);
    const auto entries_before_hit = device->num_program_cache_entries();
    const auto b_result = plain_backward_outputs(
        ttnn::prim::ttml_rmsnorm_bw(b.x.tensor, b.gamma.tensor, b.rms.tensor, b.dL_dout.tensor, /*gate=*/std::nullopt),
        "launch B");
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
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
        const auto result = plain_backward_outputs(
            ttnn::prim::ttml_rmsnorm_bw(
                in.x.tensor,
                in.gamma.tensor,
                in.rms.tensor,
                in.dL_dout.tensor,
                /*gate=*/std::nullopt,
                /*compute_dgamma=*/true,
                preallocated_dL_da,
                preallocated_dL_dgamma_components),
            label);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (pattern.repeats_earlier_pattern) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of its pattern";
        }
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
        const auto result = plain_backward_outputs(
            ttnn::prim::ttml_rmsnorm_bw(
                in.x.tensor, in.gamma.tensor, in.rms.tensor, in.dL_dout.tensor, /*gate=*/std::nullopt),
            label);
        expect_cache(entries_before, spec.new_key ? CacheExpectation::Miss : CacheExpectation::Hit, label);
        expect_rmsnorm_bw_matches_reference(result[0], result[1], in, label);
        kept.add_inputs(in, label);
        kept.add(result[0], label + " dL_da");
        kept.add(result[1], label + " dL_dgamma_components");
    }

    kept.expect_unchanged();
}

// ----------------------------------------------------------------------------
// Gated mode (see Section 4): the gate is one more input and dL_dgate one more output, and the program key holds the
// head count, which is 0 without a gate. The inputs and references are Section 4's, at its kEps. Launch k scales x by
// 2^k, which is exact, and negates gamma on odd k, so that a launch that reads another launch's x, rms or gamma fails
// the reference check; each launch draws its own gate and dL_dout.
// ----------------------------------------------------------------------------

namespace {

// The head-major input [2, 1, S, V] with S = two_core_group_seq_len(0): 2 * cores + 4 tile rows, so both core groups
// run on any grid. One head takes the two rows as two batch entries, and two heads take them as the heads of one.
Shape two_core_group_gated_shape(const uint32_t heads) {
    return Shape{2U / heads, ttml::test_utils::two_core_group_seq_len(0U), heads, kCacheTestC};
}

// The inputs of launch `launch` on the device, the values the kernels read from them and the reference results. rms
// is what the forward returns for x, as the backward gets it in training.
struct ModeInputs {
    bool gated;
    Shape shape;
    Reference expected;
    ttnn::Tensor x;
    ttnn::Tensor gamma;
    std::optional<ttnn::Tensor> gate;
    ttnn::Tensor rms;
    ttnn::Tensor dL_dout;
};

ModeInputs upload_mode_inputs(const Shape& s, const uint32_t launch, const ttnn::DataType x_dtype, const bool gated) {
    auto in = make_inputs(s, 500U + 10U * launch, x_dtype, gated);
    in.x *= std::exp2(static_cast<float>(launch));
    if (launch % 2U == 1U) {
        in.gamma = -in.gamma;
    }
    auto expected = reference(in, in.dy, s);
    auto rms = to_device(xt::xarray<float>(xt::cast<float>(expected.rms)));
    return ModeInputs{
        .gated = gated,
        .shape = s,
        .expected = std::move(expected),
        .x = to_device(in.x, x_dtype),
        .gamma = to_device(in.gamma),
        .gate = to_device(in.gate),
        .rms = std::move(rms),
        .dL_dout = to_device(in.dy),
    };
}

void keep_forward_inputs(KeptTensors& kept, const ModeInputs& in, const std::string& label) {
    kept.add(in.x, label + " input");
    kept.add(in.gamma, label + " gamma");
    if (in.gate.has_value()) {
        kept.add(*in.gate, label + " gate");
    }
}

void keep_backward_inputs(KeptTensors& kept, const ModeInputs& in, const std::string& label) {
    keep_forward_inputs(kept, in, label);
    kept.add(in.rms, label + " rms");
    kept.add(in.dL_dout, label + " dL_dout");
}

void keep_forward_outputs(KeptTensors& kept, const std::vector<ttnn::Tensor>& result, const std::string& label) {
    const std::array<const char*, 2> names = {" output", " rms"};
    for (size_t i = 0; i < std::min(names.size(), result.size()); ++i) {
        kept.add(result[i], label + names[i]);
    }
}

// {output, rms}, or {output} without return_intermediates.
void expect_mode_fw_matches_reference(
    const std::vector<ttnn::Tensor>& result, const ModeInputs& in, const std::string& label) {
    ASSERT_FALSE(result.empty()) << label;
    expect_close(ttml::core::to_xtensor(result[0]), in.expected.out, label + " output");
    if (result.size() > 1U) {
        expect_close(ttml::core::to_xtensor(result[1]), in.expected.rms, label + " rms");
    }
}

// {dL_da, dL_dgamma_components, dL_dgate}; dL_dgamma_components only with compute_dgamma and dL_dgate only with a gate.
void expect_mode_bw_matches_reference(
    const std::vector<std::optional<ttnn::Tensor>>& result,
    const ModeInputs& in,
    const bool compute_dgamma,
    const std::string& label) {
    ASSERT_EQ(result.size(), 3U) << label;
    ASSERT_TRUE(result[0].has_value()) << label;
    expect_close(ttml::core::to_xtensor(*result[0]), in.expected.dx, label + " dL_da", kDxTol);
    ASSERT_EQ(result[1].has_value(), compute_dgamma) << label;
    if (compute_dgamma) {
        const xt::xarray<double> components = xt::cast<double>(ttml::core::to_xtensor(*result[1]));
        const xt::xarray<double> dgamma = xt::sum(components, {0, 1, 2}, xt::keep_dims);
        expect_close(dgamma, in.expected.dgamma, label + " dL_dgamma");
    }
    ASSERT_EQ(result[2].has_value(), in.gated) << label;
    if (in.gated) {
        expect_dgate_close(ttml::core::to_xtensor(*result[2]), in.expected.dgate, label + " dL_dgate");
    }
}

void keep_backward_outputs(
    KeptTensors& kept, const std::vector<std::optional<ttnn::Tensor>>& result, const std::string& label) {
    const std::array<const char*, 3> names = {" dL_da", " dL_dgamma_components", " dL_dgate"};
    for (size_t i = 0; i < std::min(names.size(), result.size()); ++i) {
        if (result[i].has_value()) {
            kept.add(*result[i], label + names[i]);
        }
    }
}

std::string dtype_label(const ttnn::DataType x_dtype) {
    return x_dtype == ttnn::DataType::FLOAT32 ? "fp32 input" : "bf16 input";
}

}  // namespace

// The gated counterpart of Forward_CacheHitReadsFreshBuffers_TwoCoreGroups and its NoIntermediates variant, with bf16
// and with fp32 input: launch B, on a new input, gamma and gate while launch A's tensors stay alive, reuses A's
// program. The mode without rms goes first, so that a launch with rms that reused its program would leave the rms
// unwritten rather than a launch without it writing one through the rms slot's address 0.
TEST_F(RMSNormCacheHitTest, GatedForward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = two_core_group_gated_shape(/*heads=*/2U);
    KeptTensors kept;

    struct Mode {
        ttnn::DataType x_dtype;
        bool return_intermediates;
    };
    constexpr std::array<Mode, 3> kModes = {{
        {ttnn::DataType::BFLOAT16, false},
        {ttnn::DataType::BFLOAT16, true},
        {ttnn::DataType::FLOAT32, true},
    }};
    uint32_t launch = 0U;
    for (const auto& m : kModes) {
        const std::string mode = dtype_label(m.x_dtype) + (m.return_intermediates ? "" : ", no rms");
        const size_t num_outputs = m.return_intermediates ? 2U : 1U;
        const auto a = upload_mode_inputs(shape, launch++, m.x_dtype, /*gated=*/true);
        const auto entries_before_miss = device->num_program_cache_entries();
        const auto a_result = ttnn::prim::ttml_rmsnorm_fw(a.x, a.gamma, a.gate, m.return_intermediates, kEps);
        ASSERT_GT(device->num_program_cache_entries(), entries_before_miss)
            << mode << ": the first launch must compile a program";
        ASSERT_EQ(a_result.size(), num_outputs) << mode;
        expect_mode_fw_matches_reference(a_result, a, mode + " launch A");
        keep_forward_inputs(kept, a, mode + " launch A");
        keep_forward_outputs(kept, a_result, mode + " launch A");

        // Same specs, new values, launch A's tensors still alive.
        const auto b = upload_mode_inputs(shape, launch++, m.x_dtype, /*gated=*/true);
        const auto entries_before_hit = device->num_program_cache_entries();
        const auto b_result = ttnn::prim::ttml_rmsnorm_fw(b.x, b.gamma, b.gate, m.return_intermediates, kEps);
        EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit)
            << mode << ": launch B must reuse launch A's program";
        ASSERT_EQ(b_result.size(), num_outputs) << mode;
        std::vector<ttnn::Tensor> a_tensors = {a.x, a.gamma, a.gate.value()};
        std::vector<ttnn::Tensor> b_tensors = {b.x, b.gamma, b.gate.value()};
        a_tensors.insert(a_tensors.end(), a_result.begin(), a_result.end());
        b_tensors.insert(b_tensors.end(), b_result.begin(), b_result.end());
        test_utils::expect_new_buffers(a_tensors, b_tensors, mode + " launch B");
        expect_mode_fw_matches_reference(b_result, b, mode + " launch B (cache hit)");
    }

    kept.expect_unchanged();
}

// The gated counterpart of Backward_CacheHitReadsFreshBuffers_TwoCoreGroups, with bf16 and with fp32 input.
TEST_F(RMSNormCacheHitTest, GatedBackward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = two_core_group_gated_shape(/*heads=*/2U);
    KeptTensors kept;
    uint32_t launch = 0U;
    for (const auto x_dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        const std::string mode = dtype_label(x_dtype);
        const auto a = upload_mode_inputs(shape, launch++, x_dtype, /*gated=*/true);
        const auto entries_before_miss = device->num_program_cache_entries();
        const auto a_result = ttnn::prim::ttml_rmsnorm_bw(a.x, a.gamma, a.rms, a.dL_dout, a.gate);
        ASSERT_GT(device->num_program_cache_entries(), entries_before_miss)
            << mode << ": the first launch must compile a program";
        expect_mode_bw_matches_reference(a_result, a, /*compute_dgamma=*/true, mode + " launch A");
        keep_backward_inputs(kept, a, mode + " launch A");
        keep_backward_outputs(kept, a_result, mode + " launch A");

        // Same specs, new values, launch A's tensors still alive.
        const auto b = upload_mode_inputs(shape, launch++, x_dtype, /*gated=*/true);
        const auto entries_before_hit = device->num_program_cache_entries();
        const auto b_result = ttnn::prim::ttml_rmsnorm_bw(b.x, b.gamma, b.rms, b.dL_dout, b.gate);
        EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit)
            << mode << ": launch B must reuse launch A's program";
        expect_mode_bw_matches_reference(b_result, b, /*compute_dgamma=*/true, mode + " launch B (cache hit)");
        test_utils::expect_new_buffers(
            {a.x,
             a.gamma,
             a.gate.value(),
             a.rms,
             a.dL_dout,
             a_result.at(0).value(),
             a_result.at(1).value(),
             a_result.at(2).value()},
            {b.x,
             b.gamma,
             b.gate.value(),
             b.rms,
             b.dL_dout,
             b_result.at(0).value(),
             b_result.at(1).value(),
             b_result.at(2).value()},
            mode + " launch B");
    }

    kept.expect_unchanged();
}

// The gated counterpart of Forward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups: the head-merged output
// and the head-major rms given or not, in every combination, on one gated key.
TEST_F(RMSNormCacheHitTest, GatedForward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = two_core_group_gated_shape(/*heads=*/2U);
    KeptTensors kept;

    struct Launch {
        const char* label;
        bool preallocated_rms;
        bool preallocated_output;
        bool repeats_earlier_pattern;
    };
    constexpr std::array<Launch, 6> kLaunches = {{
        {"no preallocation", false, false, false},
        {"preallocated output and rms", true, true, false},
        {"preallocated output and rms again", true, true, true},
        {"preallocated output only", false, true, false},
        {"preallocated rms only", true, false, false},
        {"no preallocation again", false, false, true},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& pattern = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + pattern.label + ")";
        const auto in = upload_mode_inputs(shape, launch, ttnn::DataType::BFLOAT16, /*gated=*/true);
        std::optional<ttnn::Tensor> preallocated_rms;
        std::optional<ttnn::Tensor> preallocated_output;
        if (pattern.preallocated_rms) {
            preallocated_rms = test_utils::filled_on_device(shape.rms(), kNotWritten);
        }
        if (pattern.preallocated_output) {
            preallocated_output = test_utils::filled_on_device(shape.head_merged(), kNotWritten);
        }
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_fw(
            in.x,
            in.gamma,
            in.gate,
            /*return_intermediates=*/true,
            kEps,
            preallocated_rms,
            preallocated_output);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (pattern.repeats_earlier_pattern) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of its pattern";
        }
        ASSERT_EQ(result.size(), 2U) << label;
        if (preallocated_output.has_value()) {
            EXPECT_EQ(result[0].buffer()->address(), preallocated_output->buffer()->address())
                << label << " must return the preallocated output";
        }
        if (preallocated_rms.has_value()) {
            EXPECT_EQ(result[1].buffer()->address(), preallocated_rms->buffer()->address())
                << label << " must return the preallocated rms";
        }
        expect_mode_fw_matches_reference(result, in, label);
        keep_forward_inputs(kept, in, label);
        kept.add(result[0], label + " output");
        kept.add(result[1], label + " rms");
    }

    kept.expect_unchanged();
}

// The gated counterpart of Backward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups, with the preallocated
// dL_dgate as the third optional output.
TEST_F(RMSNormCacheHitTest, GatedBackward_PreallocatedOutputsAcrossOptionalPresence_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto shape = two_core_group_gated_shape(/*heads=*/2U);
    KeptTensors kept;

    struct Launch {
        const char* label;
        bool preallocated_dL_da;
        bool preallocated_dL_dgamma_components;
        bool preallocated_dL_dgate;
        bool repeats_earlier_pattern;
    };
    constexpr std::array<Launch, 7> kLaunches = {{
        {"no preallocation", false, false, false, false},
        {"preallocated dL_da, dL_dgamma_components and dL_dgate", true, true, true, false},
        {"preallocated dL_da, dL_dgamma_components and dL_dgate again", true, true, true, true},
        {"preallocated dL_da only", true, false, false, false},
        {"preallocated dL_dgamma_components only", false, true, false, false},
        {"preallocated dL_dgate only", false, false, true, false},
        {"no preallocation again", false, false, false, true},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& pattern = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + pattern.label + ")";
        const auto in = upload_mode_inputs(shape, launch, ttnn::DataType::BFLOAT16, /*gated=*/true);
        std::optional<ttnn::Tensor> preallocated_dL_da;
        std::optional<ttnn::Tensor> preallocated_dL_dgamma_components;
        std::optional<ttnn::Tensor> preallocated_dL_dgate;
        if (pattern.preallocated_dL_da) {
            preallocated_dL_da = test_utils::filled_on_device(shape.head_major(), kNotWritten);
        }
        if (pattern.preallocated_dL_dgamma_components) {
            preallocated_dL_dgamma_components = test_utils::filled_on_device(shape.head_major(), kNotWritten);
        }
        if (pattern.preallocated_dL_dgate) {
            preallocated_dL_dgate = test_utils::filled_on_device(shape.head_merged(), kNotWritten);
        }
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_bw(
            in.x,
            in.gamma,
            in.rms,
            in.dL_dout,
            in.gate,
            /*compute_dgamma=*/true,
            preallocated_dL_da,
            preallocated_dL_dgamma_components,
            preallocated_dL_dgate);
        const auto entries_after = device->num_program_cache_entries();
        if (launch == 0U) {
            ASSERT_GT(entries_after, entries_before) << "the first launch must compile a program";
        }
        if (pattern.repeats_earlier_pattern) {
            EXPECT_EQ(entries_after, entries_before) << label << " must reuse the program of its pattern";
        }
        ASSERT_EQ(result.size(), 3U) << label;
        const std::array<const std::optional<ttnn::Tensor>*, 3> preallocated = {
            &preallocated_dL_da, &preallocated_dL_dgamma_components, &preallocated_dL_dgate};
        for (size_t i = 0; i < preallocated.size(); ++i) {
            if (preallocated[i]->has_value() && result[i].has_value()) {
                EXPECT_EQ(result[i]->buffer()->address(), (*preallocated[i])->buffer()->address())
                    << label << " must return preallocated output " << i;
            }
        }
        expect_mode_bw_matches_reference(result, in, /*compute_dgamma=*/true, label);
        keep_backward_inputs(kept, in, label);
        keep_backward_outputs(kept, result, label);
    }

    kept.expect_unchanged();
}

// The plain mode, a gate with one head and a gate with two heads, all on the same [2, 1, S, V] input: the key must tell
// all three apart, or a gated launch computes without the gate or splits its gate into the wrong heads.
TEST_F(RMSNormCacheHitTest, Forward_GateAndHeadCountCompileDistinctPrograms_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    KeptTensors kept;

    struct Launch {
        const char* label;
        uint32_t heads;  // 0: plain
        bool new_key;
    };
    constexpr std::array<Launch, 6> kLaunches = {{
        {"plain", 0U, true},
        {"gated, one head", 1U, true},
        {"gated, two heads", 2U, true},
        {"plain again", 0U, false},
        {"gated, two heads again", 2U, false},
        {"gated, one head again", 1U, false},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& spec = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + spec.label + ")";
        const bool gated = spec.heads > 0U;
        const auto in = upload_mode_inputs(
            two_core_group_gated_shape(gated ? spec.heads : 1U), launch, ttnn::DataType::BFLOAT16, gated);
        const auto entries_before = device->num_program_cache_entries();
        const auto result = ttnn::prim::ttml_rmsnorm_fw(in.x, in.gamma, in.gate, /*return_intermediates=*/true, kEps);
        expect_cache(entries_before, spec.new_key ? CacheExpectation::Miss : CacheExpectation::Hit, label);
        ASSERT_EQ(result.size(), 2U) << label;
        expect_mode_fw_matches_reference(result, in, label);
        keep_forward_inputs(kept, in, label);
        kept.add(result[0], label + " output");
        kept.add(result[1], label + " rms");
    }

    kept.expect_unchanged();
}

// The backward counterpart of Forward_GateAndHeadCountCompileDistinctPrograms_TwoCoreGroups, each mode without and with
// compute_dgamma, which a frozen gamma turns off. In each mode the launch without dL_dgamma goes first: a launch with
// it that reused that program would leave its dL_dgamma_components unwritten, while in the other order a launch
// without it would write them through the absent output's address 0. Once the programs are cached, hits must still
// reject a preallocated dL_dgamma_components without compute_dgamma and a preallocated dL_dgate without a gate: the op
// has no hit-only validation, so ttnn validates hits as misses.
TEST_F(RMSNormCacheHitTest, Backward_GateHeadCountAndDgammaCompileDistinctPrograms_TwoCoreGroups) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    KeptTensors kept;

    struct Launch {
        const char* label;
        uint32_t heads;  // 0: plain
        bool compute_dgamma;
        bool new_key;
    };
    constexpr std::array<Launch, 10> kLaunches = {{
        {"plain, no dL_dgamma", 0U, false, true},
        {"plain", 0U, true, true},
        {"gated, one head, no dL_dgamma", 1U, false, true},
        {"gated, one head", 1U, true, true},
        {"gated, two heads, no dL_dgamma", 2U, false, true},
        {"gated, two heads", 2U, true, true},
        {"plain, no dL_dgamma again", 0U, false, false},
        {"gated, two heads, no dL_dgamma again", 2U, false, false},
        {"plain again", 0U, true, false},
        {"gated, one head again", 1U, true, false},
    }};
    for (uint32_t launch = 0; launch < kLaunches.size(); ++launch) {
        const auto& spec = kLaunches[launch];
        const std::string label = "launch " + std::to_string(launch) + " (" + spec.label + ")";
        const bool gated = spec.heads > 0U;
        const auto in = upload_mode_inputs(
            two_core_group_gated_shape(gated ? spec.heads : 1U), launch, ttnn::DataType::BFLOAT16, gated);
        const auto entries_before = device->num_program_cache_entries();
        const auto result =
            ttnn::prim::ttml_rmsnorm_bw(in.x, in.gamma, in.rms, in.dL_dout, in.gate, spec.compute_dgamma);
        expect_cache(entries_before, spec.new_key ? CacheExpectation::Miss : CacheExpectation::Hit, label);
        expect_mode_bw_matches_reference(result, in, spec.compute_dgamma, label);
        keep_backward_inputs(kept, in, label);
        keep_backward_outputs(kept, result, label);
    }

    const auto launch = static_cast<uint32_t>(kLaunches.size());
    const auto plain = upload_mode_inputs(two_core_group_gated_shape(1U), launch, ttnn::DataType::BFLOAT16, false);
    const auto preallocated_dL_dgate = test_utils::filled_on_device(plain.shape.head_merged(), kNotWritten);
    EXPECT_THAT(
        ([&] {
            return ttnn::prim::ttml_rmsnorm_bw(
                plain.x,
                plain.gamma,
                plain.rms,
                plain.dL_dout,
                /*gate=*/std::nullopt,
                /*compute_dgamma=*/true,
                std::nullopt,
                std::nullopt,
                preallocated_dL_dgate);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("given without a gate")));
    const auto gated = upload_mode_inputs(two_core_group_gated_shape(2U), launch + 1U, ttnn::DataType::BFLOAT16, true);
    const auto preallocated_dL_dgamma_components = test_utils::filled_on_device(gated.shape.head_major(), kNotWritten);
    EXPECT_THAT(
        ([&] {
            return ttnn::prim::ttml_rmsnorm_bw(
                gated.x,
                gated.gamma,
                gated.rms,
                gated.dL_dout,
                gated.gate,
                /*compute_dgamma=*/false,
                std::nullopt,
                preallocated_dL_dgamma_components,
                std::nullopt);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("compute_dgamma is false")));
    EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(preallocated_dL_dgate), kNotWritten)))
        << "the rejected launch wrote the preallocated dL_dgate";
    EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(preallocated_dL_dgamma_components), kNotWritten)))
        << "the rejected launch wrote the preallocated dL_dgamma_components";

    kept.expect_unchanged();
}
