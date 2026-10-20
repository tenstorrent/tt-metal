// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace {

using ttnn::DataType;
using ttnn::operations::data_movement::pack_scalar_into_uint32;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

TEST(PackScalarIntoUint32, Float32KeepsBitPattern) {
    EXPECT_EQ(pack_scalar_into_uint32(DataType::FLOAT32, 1.0f), 0x3F800000u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::FLOAT32, -kInf), 0xFF800000u);
}

TEST(PackScalarIntoUint32, SixteenBitTypesFillBothHalves) {
    EXPECT_EQ(pack_scalar_into_uint32(DataType::BFLOAT16, 1.0f), 0x3F803F80u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT16, 7.0f), 0x00070007u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT16, -1.0f), 0x00000000u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT16, kInf), 0xFFFFFFFFu);
}

TEST(PackScalarIntoUint32, IntegerTypesAreNumericAndSaturate) {
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, 5.0f), 5u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, -1.0f), 0xFFFFFFFFu);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, -kInf), 0x80000000u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, kInf), 0x7FFFFFFFu);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, kNaN), 0u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT32, 5.0f), 5u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT32, -1.0f), 0u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT32, kInf), 0xFFFFFFFFu);
}

TEST(PackScalarIntoUint32, ThirtyTwoBitIntegersTruncate) {
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, 1.75f), 1u);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::INT32, -1.75f), 0xFFFFFFFFu);
    EXPECT_EQ(pack_scalar_into_uint32(DataType::UINT32, 1.75f), 1u);
}

TEST(PackScalarIntoUint32, UnsupportedTypesThrow) {
    for (DataType dtype :
         {DataType::BFLOAT8_B,
          DataType::BFLOAT4_B,
          DataType::FP8_E4M3,
          DataType::UINT8,
          DataType::INT8,
          DataType::INVALID}) {
        EXPECT_ANY_THROW(pack_scalar_into_uint32(dtype, 0.0f));
    }
}

}  // namespace
