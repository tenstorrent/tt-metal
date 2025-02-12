// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"

class UnaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = ttml::core::create_shape({2, 1, 1, 4});
    auto tensor = ttml::core::from_vector(test_data, shape, &ttml::autograd::ctx().get_device());

    auto tensor_ptr = ttml::autograd::create_tensor(tensor);

    auto result = ttml::ops::mean(tensor_ptr);
    auto result_data = ttml::core::to_vector(result->get_value());

    ASSERT_EQ(result_data.size(), 1);
    EXPECT_FLOAT_EQ(result_data[0], 2.5F);

    result->backward();
    auto tensor_grad = ttml::core::to_vector(tensor_ptr->get_grad());
    ASSERT_EQ(tensor_grad.size(), test_data.size());
    for (float it : tensor_grad) {
        EXPECT_FLOAT_EQ(it, 0.125F);
    }
}

TEST_F(UnaryOpsTest, Sum) {
    xt::xarray<float> test_vector = {{1.F, 2.F, 3.F, 4.F}, {1.F, 2.F, 3.F, 4.F}};
    auto test_tensor_ptr =
        ttml::autograd::create_tensor(ttml::core::from_xtensor(test_vector, &ttml::autograd::ctx().get_device()));

    auto result = ttml::ops::sum(test_tensor_ptr);
    auto result_vector = ttml::core::to_xtensor(result->get_value());

    ASSERT_TRUE(xt::allclose(result_vector, xt::sum(test_vector), 1e-5F));

    result->backward();
    auto test_tensor_grad = ttml::core::to_xtensor(test_tensor_ptr->get_grad());

    ASSERT_TRUE(xt::allclose(xt::ones_like(test_vector), test_tensor_grad, 1e-5F));
}

TEST_F(UnaryOpsTest, SumMultiBatch_BROKEN) {
    xt::xarray<float> a_xarray = {
        {{{0.00000F, 1.00000F, 2.00000F, 3.00000F, 4.00000F},
          {5.00000F, 6.00000F, 7.00000F, 8.00000F, 9.00000F},
          {10.00000F, 11.00000F, 12.00000F, 13.00000F, 14.00000F},
          {15.00000F, 16.00000F, 17.00000F, 18.00000F, 19.00000F}},
         {{20.00000F, 21.00000F, 22.00000F, 23.00000F, 24.00000F},
          {25.00000F, 26.00000F, 27.00000F, 28.00000F, 29.00000F},
          {30.00000F, 31.00000F, 32.00000F, 33.00000F, 34.00000F},
          {35.00000F, 36.00000F, 37.00000F, 38.00000F, 39.00000F}},
         {{40.00000F, 41.00000F, 42.00000F, 43.00000F, 44.00000F},
          {45.00000F, 46.00000F, 47.00000F, 48.00000F, 49.00000F},
          {50.00000F, 51.00000F, 52.00000F, 53.00000F, 54.00000F},
          {55.00000F, 56.00000F, 57.00000F, 58.00000F, 59.00000F}}},
        {{{60.00000F, 61.00000F, 62.00000F, 63.00000F, 64.00000F},
          {65.00000F, 66.00000F, 67.00000F, 68.00000F, 69.00000F},
          {70.00000F, 71.00000F, 72.00000F, 73.00000F, 74.00000F},
          {75.00000F, 76.00000F, 77.00000F, 78.00000F, 79.00000F}},
         {{80.00000F, 81.00000F, 82.00000F, 83.00000F, 84.00000F},
          {85.00000F, 86.00000F, 87.00000F, 88.00000F, 89.00000F},
          {90.00000F, 91.00000F, 92.00000F, 93.00000F, 94.00000F},
          {95.00000F, 96.00000F, 97.00000F, 98.00000F, 99.00000F}},
         {{100.00000F, 101.00000F, 102.00000F, 103.00000F, 104.00000F},
          {105.00000F, 106.00000F, 107.00000F, 108.00000F, 109.00000F},
          {110.00000F, 111.00000F, 112.00000F, 113.00000F, 114.00000F},
          {115.00000F, 116.00000F, 117.00000F, 118.00000F, 119.00000F}}}};
    auto a_shape = {2, 3, 4, 5};
    auto a = ttml::autograd::create_tensor(ttml::core::from_xtensor(a_xarray, &ttml::autograd::ctx().get_device()));

    auto result = ttml::ops::sum(a);
    auto sum_a = ttml::core::to_xtensor(result->get_value());
    std::cout << "sum(a_xarray): " << xt::sum(a_xarray) << std::endl;
    std::cout << "sum_a: " << sum_a << std::endl;
    ASSERT_TRUE(xt::allclose(sum_a, xt::sum(a_xarray), 1e-5F));

    result->backward();
    auto a_grad = ttml::core::to_xtensor(a->get_grad());

    std::cout << "a_grad" << a_grad << "\n";

    ASSERT_TRUE(xt::allclose(xt::ones_like(a_xarray), a_grad, 1e-5F));
}

TEST_F(UnaryOpsTest, LogSoftmax) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data = {-0.1F, -0.2F, -0.3F, -0.4F, 0.F, -0.2F, -0.3F, -0.4F};
    auto tensor = ttml::core::from_vector(test_data, ttml::core::create_shape({2, 1, 1, 4}), device);
    auto tensor_ptr = ttml::autograd::create_tensor(tensor);
    auto result = ttml::ops::log_softmax_moreh(tensor_ptr, 3);
    auto result_data = ttml::core::to_vector(result->get_value());
    std::vector<float> expected_data = {
        -1.24253553F, -1.34253553F, -1.44253553F, -1.54253553F, -1.17244159F, -1.37244159F, -1.47244159F, -1.57244159F};
    EXPECT_EQ(result_data.size(), expected_data.size());
    for (uint32_t idx = 0; idx < result_data.size(); ++idx) {
        EXPECT_NEAR(result_data[idx], expected_data[idx], 2e-2F);
    }

    result->backward();
    auto tensor_grad = ttml::core::to_vector(tensor_ptr->get_grad());
    std::vector<float> expected_grad = {-0.156F, -0.03906F, 0.05078F, 0.1406F, -0.25F, -0.0156F, 0.07421F, 0.16406F};
    EXPECT_EQ(tensor_grad.size(), expected_grad.size());
    for (uint32_t idx = 0; idx < tensor_grad.size(); ++idx) {
        EXPECT_NEAR(tensor_grad[idx], expected_grad[idx], 2e-2F);
    }
}
