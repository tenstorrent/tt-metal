// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/xtensor_all_includes.hpp>

#include "core/xtensor_utils.hpp"

TEST(XTensorTest, BasicOperations) {
    // Create an xtensor array
    xt::xarray<double> arr = {1.0, 2.0, 3.0, 4.0};

    // Compute the sum
    double sum = xt::sum(arr)();

    // Check if the sum is correct
    EXPECT_DOUBLE_EQ(sum, 10.0);

    // Perform element-wise addition
    xt::xarray<double> arr2 = arr + 2.0;

    // Expected result
    xt::xarray<double> expected = {3.0, 4.0, 5.0, 6.0};

    // Verify the result
    EXPECT_TRUE(xt::allclose(arr2, expected));
}

TEST(XTensorTest, SpanToXtensor) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    std::span<int> data_span(data.data(), data.size());
    ttnn::SimpleShape shape({2, 3});

    auto result = ttml::core::span_to_xtensor_view(data_span, shape);

    // Check shape
    EXPECT_EQ(result.shape().size(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);

    // Check data
    int expected_val = 1;
    for (size_t i = 0; i < result.shape()[0]; ++i) {
        for (size_t j = 0; j < result.shape()[1]; ++j) {
            EXPECT_EQ(result(i, j), expected_val++);
        }
    }
}

// Test xtensor_to_span
TEST(XTensorTest, XtensorToSpan) {
    xt::xarray<float> arr = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto span_result = ttml::core::xtensor_to_span(arr);

    EXPECT_EQ(span_result.size(), arr.size());

    // Check data
    size_t index = 0;
    for (float val : arr) {
        EXPECT_FLOAT_EQ(span_result[index++], val);
    }
}

// Test get_shape_from_xarray
TEST(XTensorTest, GetShape4D) {
    xt::xarray<int> arr_5d = xt::xarray<int>::from_shape({2, 2, 2, 2, 2});
    auto shape5d = ttnn::experimental::xtensor::get_shape_from_xarray(arr_5d);
    ASSERT_EQ(shape5d.size(), 5);
    EXPECT_EQ(shape5d[0], 2);
    EXPECT_EQ(shape5d[1], 2);
    EXPECT_EQ(shape5d[2], 2);
    EXPECT_EQ(shape5d[3], 2);
    EXPECT_EQ(shape5d[4], 2);

    // Test a 1D shape
    xt::xarray<int> arr_1d = xt::xarray<int>::from_shape({7});
    auto shape1d = ttnn::experimental::xtensor::get_shape_from_xarray(arr_1d);
    ASSERT_EQ(shape1d.size(), 1);
    EXPECT_EQ(shape1d[0], 7);
}
