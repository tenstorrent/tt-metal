// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <random>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"

class RMSNormOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Forward and backward tests are given by comparing with results from PyTorch:
// For test tensor `x` of shape [N,C,H,W] we set x.requires_grad = True
// and compute the RMSNorm as `x_norm_sum = torch.nn.functional.rms_norm(x).sum()`
// and compute its gradient with respect to `x` as `x_grad = torch.autograd.grad(x_norm_sum, x)[0]`
// We then compare the results of the RMSNorm and its gradient with the results of the RMSNorm and its gradient
// computed by the RMSNorm op in TTML.
TEST_F(RMSNormOpTest, RMSNorm_Small_Forward_Backward) {
    using namespace ttml;
    float eps = 0.0078125F;  // default in PyTorch for bf16

    uint32_t N = 1, C = 1, H = 1, W = 8;
    uint32_t size = N * C * H * W;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}};
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));

    auto sum_result = ttml::ops::sum(result);
    sum_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad =
        xt::xarray<float>({{{{0.2432, 0.1211, -0.0020, -0.1230, 0.2432, 0.1211, -0.0020, -0.1230}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1.0e-3F, 1e-2F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    auto expected_gamma_grad =
        xt::xarray<float>({{{{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}}}});
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 1.0e-3F, 1e-2F));
}
