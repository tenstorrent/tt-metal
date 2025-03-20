// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "modules/dytanh_module.hpp"
#include "ops/losses.hpp"

namespace ttml::modules::tests {

class DyTanhTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(DyTanhTest, Forward) {
    // Input query tensor
    xt::xarray<float> xq = xt::ones<float>({4, 2, 5, 32});

    auto* device = &ttml::autograd::ctx().get_device();

    auto dytanh_mod = DyTanhLayer(32, 0.5F);

    auto xq_autograd_tensor = autograd::create_tensor(core::from_xtensor(xq, device));

    auto actual_xq_out = dytanh_mod(xq_autograd_tensor);

    auto actual_xq_out_xt = core::to_xtensor(actual_xq_out->get_value());

    auto gain = xt::ones_like(xq);
    auto bias = xt::zeros_like(xq);
    // of course the gain and bias do nothing at init; we use them here for clarity.
    auto expected_xq_out = xt::tanh(xq * 0.5F) * gain + bias;

    // Check that outputs match the expected values
    EXPECT_TRUE(xt::allclose(actual_xq_out_xt, expected_xq_out, 2e-1, 2e-1));
}

TEST_F(DyTanhTest, Backward) {
    // Input query tensor
    xt::xarray<float> x = xt::ones<float>({4, 2, 256, 32});
    auto* device = &ttml::autograd::ctx().get_device();

    auto dytanh_mod = DyTanhLayer(32, 0.5F);

    auto x_autograd_tensor = autograd::create_tensor(core::from_xtensor(x, device));

    auto x_out = dytanh_mod(x_autograd_tensor);
    auto target = autograd::create_tensor(core::from_xtensor(xt::xarray<float>{xt::ones_like(x) * 500.0F}, device));

    auto loss = ttml::ops::mse_loss(x_out, target);
    fmt::println("loss: {}", core::to_xtensor(loss->get_value()));
    loss->backward();

    xt::xarray<float> expected_x_grad = xt::full_like(x, -0.06000F);
    xt::xarray<float> expected_scale_grad = xt::full_like(xt::ones<float>({1, 1, 1, 1}), -7863.75049F);
    xt::xarray<float> expected_gain_grad = xt::full_like(xt::ones<float>({1, 1, 1, 32}), -144.39830F);
    xt::xarray<float> expected_bias_grad = xt::full_like(xt::ones<float>({1, 1, 1, 32}), -312.47116F);

    xt::xarray<float> actual_x_grad = core::to_xtensor(x_autograd_tensor->get_grad());
    xt::xarray<float> actual_scale_grad = core::to_xtensor(dytanh_mod.m_scale->get_grad());
    xt::xarray<float> actual_gain_grad = core::to_xtensor(dytanh_mod.m_gain->get_grad());
    xt::xarray<float> actual_bias_grad = core::to_xtensor(dytanh_mod.m_bias->get_grad());

    fmt::println("actual_x_grad: {}", actual_x_grad);
    fmt::println("actual_scale_grad: {}", actual_scale_grad);
    fmt::println("actual_gain_grad: {}", actual_gain_grad);
    fmt::println("actual_bias_grad: {}", actual_bias_grad);

    /* EXPECT_TRUE(xt::allclose(actual_x_grad, expected_x_grad, 1e-1, 2e-4));
    EXPECT_TRUE(xt::allclose(actual_scale_grad, expected_scale_grad, 1e-1, 2e-4));
    EXPECT_TRUE(xt::allclose(actual_gain_grad, expected_gain_grad, 1e-1, 2e-4));
    EXPECT_TRUE(xt::allclose(actual_bias_grad, expected_bias_grad, 1e-1, 2e-4)); */
}

TEST_F(DyTanhTest, Big) {
    xt::random::seed(42);
    // Input query tensor
    xt::xarray<float> x = xt::random::randn<float>({64, 1, 256, 384});
    auto* device = &ttml::autograd::ctx().get_device();

    auto dytanh_mod = DyTanhLayer(384, 0.5F);

    auto x_autograd_tensor = autograd::create_tensor(core::from_xtensor(x, device));

    auto actual_out = dytanh_mod(x_autograd_tensor);

    auto actual_out_xt = core::to_xtensor(actual_out->get_value());

    auto expected_out = xt::tanh(x * 0.5F);
    EXPECT_TRUE(xt::allclose(actual_out_xt, expected_out, 1e-1, 2e-4));

    auto expected_out_grad_finite_diff = (xt::tanh((x + 1e-6) * 0.5F) - xt::tanh(x * 0.5F)) / 1e-6;
    actual_out->backward();
    auto actual_out_grad = core::to_xtensor(x_autograd_tensor->get_grad());

    auto average_expected_grad = xt::mean(xt::abs(expected_out_grad_finite_diff))();
    fmt::println("average_expected_grad: {}", average_expected_grad);
    auto average_grad = xt::mean(xt::abs(actual_out_grad))();
    fmt::println("average_grad: {}", average_grad);
    auto average_diff = xt::mean(xt::abs(actual_out_grad - expected_out_grad_finite_diff))();
    fmt::println("average_diff: {}", average_diff);
    auto max_abs_diff = xt::amax(xt::abs(actual_out_grad - expected_out_grad_finite_diff))();
    fmt::println("max_abs_diff: {}", max_abs_diff);
    EXPECT_TRUE(xt::allclose(actual_out_grad, expected_out_grad_finite_diff, 1e-1, 2e-1));
}

}  // namespace ttml::modules::tests
