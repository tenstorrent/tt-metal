// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <exception>

#include "ttnn/dynamic_length.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::dynamic_length::test {

TEST(DynamicLength, SplitScalarPassesThrough) {
    const auto split_length = split(ValidLength{std::size_t{100}}, 128);
    EXPECT_EQ(split_length.value, 100);
    EXPECT_FALSE(split_length.tensor.has_value());
}

TEST(DynamicLength, SplitTensorYieldsCapacity) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::UINT32, ttnn::ROW_MAJOR_LAYOUT);
    const auto split_length = split(ValidLength{length}, 128);
    EXPECT_EQ(split_length.value, 128);
    EXPECT_TRUE(split_length.tensor.has_value());
}

TEST(DynamicLength, AcceptsLengthUpToCapacity) {
    const std::array axes{
        DynamicAxis{.name = "n", .capacity = 128, .value = 128},
        DynamicAxis{.name = "m", .capacity = 128, .value = 0},
        DynamicAxis{.name = "l", .value = 1'000'000},
    };
    EXPECT_NO_THROW(validate_axes(axes, "op"));
}

TEST(DynamicLength, RejectsLengthAboveCapacity) {
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 129}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

TEST(DynamicLength, RejectsMisalignedLength) {
    const std::array aligned{DynamicAxis{.name = "n", .capacity = 128, .value = 64, .alignment = 32}};
    EXPECT_NO_THROW(validate_axes(aligned, "op"));
    const std::array misaligned{DynamicAxis{.name = "n", .capacity = 128, .value = 65, .alignment = 32}};
    EXPECT_THROW(validate_axes(misaligned, "op"), std::exception);
}

TEST(DynamicLength, RejectsHostLengthTensor) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::UINT32, ttnn::ROW_MAJOR_LAYOUT);
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 128, .tensor = &length}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

TEST(DynamicLength, RejectsWrongDtypeLengthTensor) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT);
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 128, .tensor = &length}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

}  // namespace ttnn::dynamic_length::test
