// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class DropoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(DropoutTest, TestSeed) {
    uint32_t dropout_seed1 = 42;
    uint32_t dropout_seed2 = 32;
    float scale = 2.0F;
    float prob = 0.5F;
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    auto shapes = {std::vector<int>{64, 1, 256, 384}, std::vector<int>{1, 1, 32, 32}};
    for (auto& shape : shapes) {
        fmt::println("Testing shape: {}", shape);
        xt::xarray<float> xtensor_a = xt::random::rand(shape, -0.5, 0.5);

        auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);
        auto num_cache_before = device->num_program_cache_entries();
        auto result01 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
        auto result02 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
        auto result11 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
        auto result12 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
        auto num_cache_after = device->num_program_cache_entries();

        auto result01_vec = ttml::core::to_xtensor(result01);
        auto result02_vec = ttml::core::to_xtensor(result02);
        auto result11_vec = ttml::core::to_xtensor(result11);
        auto result12_vec = ttml::core::to_xtensor(result12);

        EXPECT_TRUE(xt::allclose(result01_vec, result11_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_TRUE(xt::allclose(result02_vec, result12_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_FALSE(xt::allclose(result01_vec, result02_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_EQ(num_cache_before, num_cache_after - 1);
    }
}

TEST_F(DropoutTest, TestProb) {
    uint32_t dropout_seed = 42;
    float scale = 1.0F;
    float prob = 0.2F;
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    xt::xarray<float> xtensor_a = xt::ones<float>({64, 1, 256, 38});
    std::vector<float> ratios;
    ratios.reserve(100);
    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);
    for (int i = 0; i < 100; i++) {
        auto result01 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed);
        auto result01_vec = ttml::core::to_xtensor(result01);
        float ratio = xt::sum(result01_vec)() / xt::sum(xtensor_a)();
        ratios.push_back(ratio);
    }
    auto xt_ratios = xt::adapt(ratios);
    auto mean_ratio = xt::mean(xt_ratios)();
    auto std_ratio = xt::stddev(xt_ratios)();
    EXPECT_NEAR(mean_ratio, 1.0F - prob, 0.05);
    EXPECT_NEAR(std_ratio, 0.05, 0.05);
}
