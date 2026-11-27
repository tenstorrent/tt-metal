// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_format.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

enum class IndexedName {
    First,
    Second,
};

constexpr std::array<const char*, 2> kIndexedNames = {
    "first",
    "second",
};

TEST(X280Format, EnumNamesMapKnownValuesAndRejectUnknownValues) {
    EXPECT_STREQ(x280::enum_name(IndexedName::First, kIndexedNames), "first");
    EXPECT_STREQ(x280::enum_name(IndexedName::Second, kIndexedNames), "second");
    EXPECT_STREQ(x280::enum_name(static_cast<IndexedName>(-1), kIndexedNames), "unknown");
    EXPECT_STREQ(x280::enum_name(static_cast<IndexedName>(2), kIndexedNames), "unknown");
}

TEST(X280Format, FormatsHexValuesExactly) {
    EXPECT_EQ(x280::hex(0), "0x0");
    EXPECT_EQ(x280::hex(0xABCDEF), "0xabcdef");
    EXPECT_EQ(x280::hex(std::numeric_limits<uint64_t>::max()), "0xffffffffffffffff");
}

TEST(X280Format, FormatsTopologyLabelsExactly) {
    EXPECT_EQ(x280::label_chip_l2cpu(3, 2), "chip 3 L2CPU 2");
    EXPECT_EQ(x280::label_noc0_xy(8, 5), "noc0 (8,5)");
    EXPECT_EQ(x280::label_noc0_xy(std::numeric_limits<uint64_t>::max(), 0), "noc0 (18446744073709551615,0)");
}

TEST(X280Format, FormatsCompactFragmentsExactly) {
    EXPECT_EQ(x280::format_trap_compact(0x7, 0x08001234, 0), "trap mcause 0x7 mepc 0x8001234 mtval 0x0");
    EXPECT_EQ(x280::format_readback_values(0xF, 0xE), "wrote 0xf, read 0xe");
}

TEST(X280Format, FailThrowsRuntimeErrorWithExactPrefix) {
    try {
        x280::fail("format failure");
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "x280: format failure");
        return;
    } catch (...) {
        FAIL() << "fail() threw a type other than std::runtime_error";
    }
    FAIL() << "fail() did not throw";
}

}  // namespace
