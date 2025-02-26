// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

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
