// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.hpp"

namespace ttnn::prim::test {

namespace fill_pad = ttnn::prim::detail;

// A 2-byte fill is duplicated across both halves of the word by pack_fill_value.
constexpr std::uint32_t duplicated(std::uint16_t value) {
    return (static_cast<std::uint32_t>(value) << 16) | static_cast<std::uint32_t>(value);
}

// Floats that no integral destination can hold. A plain static_cast of any of them is undefined
// behavior and the host architectures disagree on the result -- -1.0f to uint32_t is 0xFFFFFFFF on
// x86_64 and 0 on aarch64 -- so the word packed into the kernel depended on the build host.
constexpr float k_two_pow_31 = 2147483648.0f;
constexpr float k_two_pow_32 = 4294967296.0f;
constexpr float k_infinity = std::numeric_limits<float>::infinity();

// The pad value is read back through a volatile so it reaches the conversion at run time. A
// compiler that constant-folds an out-of-range float-to-integer conversion is free to pick a
// different answer than the instruction it emits for the same expression -- GCC 13 folds
// (uint16_t)65536.0f to 65535 on aarch64 while the fcvtzu it emits produces 0 -- so a test built
// from literals alone would pass against the unfixed code.
std::uint32_t packed(ttnn::DataType dtype, float pad_value) {
    volatile float runtime_pad_value = pad_value;
    return fill_pad::pack_fill_value_for_dtype(dtype, ttnn::PadValue(static_cast<float>(runtime_pad_value)));
}

TEST(FillPadValuePacking, Uint32SaturatesOutOfRange) {
    EXPECT_EQ(packed(ttnn::DataType::UINT32, 7.0f), 7u);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, 4294967040.0f), 4294967040u);  // largest float below 2^32
    EXPECT_EQ(packed(ttnn::DataType::UINT32, k_two_pow_31), 2147483648u);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, k_two_pow_32), 0xFFFFFFFFu);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, k_infinity), 0xFFFFFFFFu);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, -1.0f), 0u);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, -k_infinity), 0u);
    EXPECT_EQ(packed(ttnn::DataType::UINT32, std::numeric_limits<float>::quiet_NaN()), 0u);
}

TEST(FillPadValuePacking, Int32SaturatesOutOfRange) {
    EXPECT_EQ(packed(ttnn::DataType::INT32, 7.0f), 7u);
    EXPECT_EQ(packed(ttnn::DataType::INT32, -1.0f), 0xFFFFFFFFu);          // two's complement -1
    EXPECT_EQ(packed(ttnn::DataType::INT32, 2147483520.0f), 2147483520u);  // largest float below 2^31
    EXPECT_EQ(packed(ttnn::DataType::INT32, k_two_pow_31), 0x7FFFFFFFu);
    EXPECT_EQ(packed(ttnn::DataType::INT32, k_two_pow_32), 0x7FFFFFFFu);
    EXPECT_EQ(packed(ttnn::DataType::INT32, k_infinity), 0x7FFFFFFFu);
    EXPECT_EQ(packed(ttnn::DataType::INT32, -k_infinity), 0x80000000u);
    EXPECT_EQ(packed(ttnn::DataType::INT32, std::numeric_limits<float>::quiet_NaN()), 0u);
}

TEST(FillPadValuePacking, Uint16SaturatesOutOfRange) {
    EXPECT_EQ(packed(ttnn::DataType::UINT16, 7.0f), duplicated(7));
    EXPECT_EQ(packed(ttnn::DataType::UINT16, 65535.0f), duplicated(65535));
    EXPECT_EQ(packed(ttnn::DataType::UINT16, 65536.0f), duplicated(65535));
    EXPECT_EQ(packed(ttnn::DataType::UINT16, k_two_pow_32), duplicated(65535));
    EXPECT_EQ(packed(ttnn::DataType::UINT16, k_infinity), duplicated(65535));
    EXPECT_EQ(packed(ttnn::DataType::UINT16, -1.0f), 0u);
    EXPECT_EQ(packed(ttnn::DataType::UINT16, -k_infinity), 0u);
    EXPECT_EQ(packed(ttnn::DataType::UINT16, std::numeric_limits<float>::quiet_NaN()), 0u);
}

// The integer arm carries a raw bit pattern and must stay untouched by the saturation above.
TEST(FillPadValuePacking, IntegerArmIsUnchanged) {
    EXPECT_EQ(fill_pad::pack_fill_value_for_dtype(ttnn::DataType::UINT32, ttnn::PadValue(0xFFFFFFFFu)), 0xFFFFFFFFu);
    EXPECT_EQ(fill_pad::pack_fill_value_for_dtype(ttnn::DataType::INT32, ttnn::PadValue(0x80000000u)), 0x80000000u);
}

}  // namespace ttnn::prim::test
