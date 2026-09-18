// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <exception>
#include <optional>

#include "ttnn/device.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

// The unary POWER and POWER_ITERATIVE kernels compute in float. Fed an integer tile they read the
// integer bits as float32 and store the float result through the integer output tensor (#56853).
// ttnn.pow routes integer tensors around these kernels, so this validation is the only guard for
// direct ttnn::power / ttnn::power_iterative callers and for the Quasar composite; it must reject
// both an integer input and an integer preallocated output.
class UnaryPowerDtypeValidationFixture : public TTNNFixtureWithSuiteDevice<UnaryPowerDtypeValidationFixture> {};

TEST_F(UnaryPowerDtypeValidationFixture, PowerRejectsIntegerInput) {
    auto& device = *device_;
    const auto input = ttnn::full(ttnn::Shape({1, 1, 32, 32}), 7, DataType::INT32, ttnn::TILE_LAYOUT, device);
    EXPECT_THROW(ttnn::power(input, ScalarVariant(4), std::nullopt, std::nullopt, std::nullopt), std::exception);
    EXPECT_THROW(ttnn::power(input, ScalarVariant(2.5f), std::nullopt, std::nullopt, std::nullopt), std::exception);
}

TEST_F(UnaryPowerDtypeValidationFixture, PowerIterativeRejectsIntegerInput) {
    auto& device = *device_;
    const auto input = ttnn::full(ttnn::Shape({1, 1, 32, 32}), 7, DataType::INT32, ttnn::TILE_LAYOUT, device);
    for (uint32_t exponent : {0u, 1u, 2u, 3u}) {
        EXPECT_THROW(ttnn::power_iterative(input, exponent, std::nullopt, std::nullopt, std::nullopt), std::exception);
    }
}

TEST_F(UnaryPowerDtypeValidationFixture, RejectsIntegerPreallocatedOutput) {
    auto& device = *device_;
    const auto input = ttnn::full(ttnn::Shape({1, 1, 32, 32}), 7.0f, DataType::FLOAT32, ttnn::TILE_LAYOUT, device);
    auto int_output = ttnn::zeros(ttnn::Shape({1, 1, 32, 32}), DataType::INT32, ttnn::TILE_LAYOUT, device);
    EXPECT_THROW(ttnn::power(input, ScalarVariant(4), std::nullopt, int_output, std::nullopt), std::exception);
    EXPECT_THROW(ttnn::power_iterative(input, 2, std::nullopt, int_output, std::nullopt), std::exception);
}

// Float control: the tightened validation must keep accepting what the float path dispatches.
TEST_F(UnaryPowerDtypeValidationFixture, AcceptsFloatInputAndOutput) {
    auto& device = *device_;
    const auto input = ttnn::full(ttnn::Shape({1, 1, 32, 32}), 7.0f, DataType::FLOAT32, ttnn::TILE_LAYOUT, device);
    auto f32_output = ttnn::zeros(ttnn::Shape({1, 1, 32, 32}), DataType::FLOAT32, ttnn::TILE_LAYOUT, device);
    EXPECT_NO_THROW(ttnn::power(input, ScalarVariant(4), std::nullopt, std::nullopt, std::nullopt));
    EXPECT_NO_THROW(ttnn::power_iterative(input, 2, std::nullopt, f32_output, std::nullopt));
}

}  // namespace ttnn::operations::unary::test
