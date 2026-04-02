// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

namespace test::deep {
enum Enum1 {
    Value1 = 0,
    Value2 = 1,
    Value3 = 2,
};
}

namespace test_shallow {
enum Enum2 {
    ValueA = 0,
    ValueB = 1,
    ValueC = 2,
};
}

enum class EnumClass : uint8_t {
    ValueX = 0,
    ValueY = 1,
    ValueZ = 2,
};

namespace flags {
enum class BitEnum : uint32_t {
    Flag1 = 1 << 0,
    Flag2 = 1 << 1,
    Flag3 = 1 << 2,
};

constexpr BitEnum operator|(BitEnum a, BitEnum b) {
    return static_cast<BitEnum>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
}  // namespace flags

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    DEVICE_PRINT("Enum1 value: {}\n", test::deep::Value2);
    DEVICE_PRINT("Enum1 full name value: {:#}\n", test::deep::Value3);
    DEVICE_PRINT("Enum1 unrecognized value: {}\n", (test::deep::Enum1)100);
    DEVICE_PRINT("Enum1 full name unrecognized value: {:#}\n", (test::deep::Enum1)100);
    DEVICE_PRINT("Enum2 value: {}\n", test_shallow::ValueB);
    DEVICE_PRINT("Enum2 full name value: {:#}\n", test_shallow::ValueC);
    DEVICE_PRINT("EnumClass value: {}\n", EnumClass::ValueY);
    DEVICE_PRINT("EnumClass full name value: {:#}\n", EnumClass::ValueZ);
    DEVICE_PRINT("BitEnum value: {}\n", flags::BitEnum::Flag1 | flags::BitEnum::Flag3);
    DEVICE_PRINT("BitEnum full name value: {:#}\n", flags::BitEnum::Flag2 | flags::BitEnum::Flag3);
}
