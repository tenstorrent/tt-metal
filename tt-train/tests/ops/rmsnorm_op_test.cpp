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

TEST_F(RMSNormOpTest, RMSNormOp_Forward) {
    using namespace ttml;

    xt::xarray<float> example_xtensor = {
        {{{0.0037F, 1.0103F, -0.0769F, -1.0242F, 0.7413F}, {1.5342F, -1.4141F, 0.9436F, -0.3354F, 0.5814F}},
         {{2.1460F, 0.7238F, -0.2614F, -0.0608F, 1.3787F}, {0.2094F, -1.3087F, -1.8958F, 0.6596F, -1.3471F}}},
        {{{1.2607F, 1.7451F, -1.6049F, -0.0411F, -0.9609F}, {0.1918F, -1.2580F, -0.5534F, -0.3971F, -0.6368F}},
         {{0.2271F, 0.0791F, 0.8026F, 0.4299F, 0.8505F}, {1.5362F, 0.9735F, 0.4186F, -1.4561F, 1.3001F}}}};

    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));

    uint32_t N = 2, C = 2, H = 2, W = 5;

    uint32_t size = N * C * H * W;

    auto gamma = autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0F);

    // Compare result with torch
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto expected_result = xt::xarray<float>(
        {{{{{0.0051F, 1.3943F, -0.1061F, -1.4135F, 1.0230F}, {1.4376F, -1.3251F, 0.8842F, -0.3143F, 0.5448F}},

           {{1.8006F, 0.6073F, -0.2194F, -0.0510F, 1.1568F}, {0.1698F, -1.0614F, -1.5377F, 0.5350F, -1.0926F}}},

          {{{0.9884F, 1.3681F, -1.2582F, -0.0322F, -0.7533F}, {0.2719F, -1.7830F, -0.7845F, -0.5629F, -0.9025F}},

           {{0.4003F, 0.1393F, 1.4143F, 0.7576F, 1.4987F}, {1.2719F, 0.8061F, 0.3466F, -1.2056F, 1.0765F}}}}});

    std::cout << "result_xtensor: " << result_xtensor << "\n";
    std::cout << "expected_result: " << expected_result << "\n";

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-4F));

    // Take grad of sum of result with respect to example_tensor
    auto sum_result = ttml::ops::sum(result);
    sum_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());

    auto expected_example_tensor_grad = xt::xarray<float>(
        {{{{{1.3788, 1.0326, 1.4065, 1.7323, 1.1251}, {0.6064, 1.2418, 0.7337, 1.0093, 0.8117}},

           {{-0.1564, 0.5033, 0.9603, 0.8673, 0.1995}, {0.8934, 0.2968, 0.0660, 1.0703, 0.2817}}}},

         {{{{0.7355, 0.7169, 0.8457, 0.7855, 0.8209}, {1.7072, -0.4837, 0.5811, 0.8173, 0.4551}},

           {{1.1684, 1.5554, -0.3364, 0.6381, -0.4617}, {0.3445, 0.5216, 0.6962, 1.2863, 0.4188}}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1e-4F));
}

TEST_F(RMSNormOpTest, RMSNormOp_Forward_Small) {
    using namespace ttml;

    xt::xarray<float> example_xtensor = {{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));

    uint32_t H = 1, W = 8;

    uint32_t size = H * W;

    auto gamma = autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, W}), &autograd::ctx().get_device()));
    auto result = ops::rmsnorm(example_tensor, gamma, 0.0F);

    // Compare result with torch
    auto result_xtensor = core::to_xtensor(result->get_value());

    xt::xarray<float> expected_result = {{0.3651F, 0.7303F, 1.0954F, 1.4606F, 0.3651F, 0.7303F, 1.0954F, 1.4606F}};
    std::cout << "result_xtensor: " << result_xtensor << "\n";
    std::cout << "expected_result: " << expected_result << "\n";

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));

    auto sum_result = ttml::ops::sum(result);
    sum_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad = xt::xarray<float>(
        {{2.4343e-01F, 1.2172e-01F, 2.9802e-08F, -1.2172e-01F, 2.4343e-01F, 1.2172e-01F, 2.9802e-08F, -1.2172e-01F}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 3e-2F));
}
