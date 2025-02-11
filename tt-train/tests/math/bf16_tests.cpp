// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/xtensor_utils.hpp>

#include "math/bf16.hpp"
#include "xtensor/xmath.hpp"

// Test construction from float and reconversion
TEST(BFloat16Test, BasicConstruction) {
    // 1) Zero
    ttml::math::bfloat16 z(0.0f);
    EXPECT_EQ(static_cast<float>(z), 0.0f);

    // 2) Positive value
    ttml::math::bfloat16 p(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(p), 1.0f);

    // 3) Negative value
    ttml::math::bfloat16 n(-2.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(n), -2.5f);
}

TEST(BFloat16Test, ValueRounding) {
    // This test checks that rounding to nearest-even is happening.
    // Example: 1.00007f is slightly more than 1, might round up or remain 1
    float val = 1.00007f;
    ttml::math::bfloat16 a(val);
    float reconstructed = static_cast<float>(a);

    // We can't say EXACT, because we expect it to be 1.0 or slightly more
    // Check closeness with an appropriate epsilon
    EXPECT_NEAR(reconstructed, val, 1e-3f);
}

TEST(BFloat16Test, ConversionDouble) {
    // Construct from double
    double d = 3.141592653589793;
    ttml::math::bfloat16 bf(d);

    // Check float equivalence
    float f = static_cast<float>(bf);
    EXPECT_NEAR(f, static_cast<float>(d), 1e-3f);
}
/*
import torch

# Create bfloat16 tensors for a and b
a = torch.tensor(1.5, dtype=torch.bfloat16)
b = torch.tensor(2.5, dtype=torch.bfloat16)

# Perform arithmetic operations
sum_val = a + b
diff_val = a - b
prod_val = a * b
quot_val = a / b

# Print results. Note that arithmetic with bfloat16 might internally use float32 for computation.
print("a      =", a.item())
print("b      =", b.item())
print("sum    =", sum_val.item())
print("diff   =", diff_val.item())
print("prod   =", prod_val.item())
print("quot   =", quot_val.item())

# Output:
a      = 1.5
b      = 2.5
sum    = 4.0
diff   = -1.0
prod   = 3.75
quot   = 0.6015625
*/
TEST(BFloat16Test, ArithmeticOperations) {
    ttml::math::bfloat16 a(1.5f);
    ttml::math::bfloat16 b(2.5f);

    ttml::math::bfloat16 sum = a + b;
    ttml::math::bfloat16 diff = a - b;
    ttml::math::bfloat16 prod = a * b;
    ttml::math::bfloat16 quot = a / b;

    EXPECT_NEAR(static_cast<float>(sum), 4.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(diff), -1.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(prod), 3.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(quot), 0.6f, 1e-2f);

    // Compound assignments
    ttml::math::bfloat16 c(2.0f);
    c += ttml::math::bfloat16(3.0f);
    EXPECT_NEAR(static_cast<float>(c), 5.0f, 1e-3f);

    c -= ttml::math::bfloat16(1.0f);
    EXPECT_NEAR(static_cast<float>(c), 4.0f, 1e-3f);

    c *= ttml::math::bfloat16(2.0f);
    EXPECT_NEAR(static_cast<float>(c), 8.0f, 1e-3f);

    c /= ttml::math::bfloat16(4.0f);
    EXPECT_NEAR(static_cast<float>(c), 2.0f, 1e-3f);
}

TEST(BFloat16Test, ComparisonOperators) {
    ttml::math::bfloat16 a(1.0f), b(2.0f), c(1.0f);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(a != b);
}
/*
import torch
import math

# Create a list with the desired values
values = [65504.0, -65504.0, float('inf'), float('-inf'), float('nan')]

# Create a tensor with dtype torch.bfloat16
bf16_tensor = torch.tensor(values, dtype=torch.bfloat16)

# Print the bfloat16 tensor
print("bfloat16 tensor:", bf16_tensor)

# Optionally, convert it back to float for a clearer view
print("Converted to float:", bf16_tensor.to(torch.float32))

# Output:

bfloat16 tensor: tensor([ 65536., -65536.,     inf,    -inf,     nan], dtype=torch.bfloat16)
Converted to float: tensor([ 65536., -65536.,     inf,    -inf,     nan])
*/
TEST(BFloat16Test, CornerCases) {
    // Very large float
    float large_f = 65504.0f;  // near max for float16, but let's see for ttml::math::bfloat16
    ttml::math::bfloat16 large_bf(large_f);
    float large_f_back = static_cast<float>(large_bf);
    std::cout << "large_f_back: " << large_f_back << std::endl;
    float expected_value = 65536;  // 65504 + 32
    EXPECT_NEAR(large_f_back, expected_value, 1e-1f);

    // Negative large
    float neg_large_f = -65504.0f;
    ttml::math::bfloat16 neg_large_bf(neg_large_f);
    float neg_large_f_back = static_cast<float>(neg_large_bf);
    std::cout << "neg_large_f_back: " << neg_large_f_back << std::endl;
    float expected_neg_value = -65536;  // 65504 + 32
    EXPECT_NEAR(neg_large_f_back, expected_neg_value, 1e-1f);

    // Infinity
    float inf = std::numeric_limits<float>::infinity();
    ttml::math::bfloat16 bf_inf(inf);
    float reconstructed_inf = static_cast<float>(bf_inf);
    EXPECT_TRUE(std::isinf(reconstructed_inf));

    // Negative Infinity
    float neg_inf = -std::numeric_limits<float>::infinity();
    ttml::math::bfloat16 bf_neg_inf(neg_inf);
    float reconstructed_neg_inf = static_cast<float>(bf_neg_inf);
    EXPECT_TRUE(std::isinf(reconstructed_neg_inf));
    EXPECT_LT(reconstructed_neg_inf, 0.0f);

    // NaN
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    ttml::math::bfloat16 bf_nan(nan_val);
    float reconstructed_nan = static_cast<float>(bf_nan);
    EXPECT_TRUE(std::isnan(reconstructed_nan));
}
TEST(BFloat16Test, Xtensor) {
    // Create an xtensor array of floats
    xt::xarray<float> float_array = {1.5f, 2.5f, 3.5f};

    xt::xarray<ttml::math::bfloat16> bf16_array = xt::cast<ttml::math::bfloat16>(float_array);
    xt::xarray<float> sum_orig = float_array + float_array;
    xt::xarray<ttml::math::bfloat16> sum_bf16 = bf16_array + bf16_array;
    xt::xarray<float> bf16_sum_back = xt::cast<float>(sum_bf16);
    std::cout << "sum_orig: " << sum_orig << std::endl;
    std::cout << "sum_bf16: " << bf16_sum_back << std::endl;
    EXPECT_TRUE(xt::allclose(bf16_sum_back, sum_orig, /*rtol=*/1e-3, /*atol=*/1e-2));
}
