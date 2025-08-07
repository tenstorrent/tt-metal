// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class ReduceOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ReduceOpTest, TestMeanDim0) {
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> xtensor_a = xt::random::rand({128 * 64}, -0.5, 0.5).reshape({2, 1, 64, 64});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim0 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 0, true);
    auto moreh_mean_dim0 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 0, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {0}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({1, 1, 64, 64});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim0);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim0);

    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestSumDim0) {
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> xtensor_a = xt::random::rand({128 * 64}, -0.1, 0.1).reshape({2, 1, 64, 64});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_sum_dim0 = ttml::ttnn_fixed::sum_ttnn(xtensor_a_tensor, 0, true);
    auto moreh_sum_dim0 = ttml::ttnn_fixed::sum_moreh(xtensor_a_tensor, 0, true);

    xt::xarray<float> sum_xtensor = xt::sum(xtensor_a, {0}, xt::evaluation_strategy::immediate);
    sum_xtensor.reshape({1, 1, 64, 64});

    auto sum_ttnn = ttml::core::to_xtensor(ttnn_sum_dim0);
    auto sum_moreh = ttml::core::to_xtensor(moreh_sum_dim0);

    EXPECT_TRUE(xt::allclose(sum_ttnn, sum_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_ttnn, /*rtol=*/1e-2, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_moreh, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestMeanDim3) {
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> xtensor_a = xt::random::rand({128 * 64}, -0.5, 0.5).reshape({2, 1, 64, 64});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim3 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_mean_dim3 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({2, 1, 64, 1});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim3);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim3);

    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestSumDim3) {
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> xtensor_a = xt::random::rand({128 * 64}, -0.1, 0.1).reshape({2, 1, 64, 64});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_sum_dim3 = ttml::ttnn_fixed::sum_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_sum_dim3 = ttml::ttnn_fixed::sum_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> sum_xtensor = xt::sum(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    sum_xtensor.reshape({2, 1, 64, 1});

    auto sum_ttnn = ttml::core::to_xtensor(ttnn_sum_dim3);
    auto sum_moreh = ttml::core::to_xtensor(moreh_sum_dim3);

    EXPECT_TRUE(xt::allclose(sum_ttnn, sum_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_ttnn, /*rtol=*/1e-2, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_moreh, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestMeanLargeDim3) {
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> xtensor_a = xt::random::rand({1024 * 1024}, -0.5, 0.5).reshape({2, 1, 512, 1024});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim3 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_mean_dim3 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({2, 1, 512, 1});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim3);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim3);

    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}
