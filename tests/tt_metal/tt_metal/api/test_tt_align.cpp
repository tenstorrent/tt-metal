// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tt_align.hpp>

#include <csignal>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

TEST(Align, RoundsUpToPowerOfTwoBoundary) {
    struct TestCase {
        uint32_t addr;
        uint32_t alignment;
        uint32_t expected;
    };
    constexpr TestCase cases[] = {
        {0, 1, 0},    {17, 1, 17},     {0, 4, 0},          {1, 4, 4},          {4, 4, 4},    {5, 4, 8},
        {0, 16, 0},   {1, 16, 16},     {16, 16, 16},       {17, 16, 32},       {0, 32, 0},   {1, 32, 32},
        {32, 32, 32}, {33, 32, 64},    {0, 64, 0},         {1, 64, 64},        {64, 64, 64}, {65, 64, 128},
        {0, 4096, 0}, {1, 4096, 4096}, {4096, 4096, 4096}, {4097, 4096, 8192},
    };
    for (const auto& test : cases) {
        EXPECT_EQ(tt::align(test.addr, test.alignment), test.expected)
            << "addr=" << test.addr << ", alignment=" << test.alignment;
    }
}

TEST(Align, SupportsWideAndMixedIntegerTypes) {
    EXPECT_EQ(tt::align(uint64_t{0x100000001}, uint32_t{4096}), uint64_t{0x100001000});
    EXPECT_EQ(tt::align(uint32_t{1}, uint64_t{0x100000000}), uint64_t{0x100000000});
    EXPECT_EQ(tt::align(uint32_t{17}, int{16}), uint32_t{32});
    EXPECT_EQ(tt::align(uint8_t{17}, uint8_t{16}), uint8_t{32});
    EXPECT_EQ(tt::align(17, 16), 32);
}

TEST(Align, SupportsLargestRepresentableResults) {
    EXPECT_EQ(tt::align(uint32_t{0xffffffef}, uint32_t{16}), uint32_t{0xfffffff0});
    EXPECT_EQ(tt::align(uint64_t{0xffffffffffffffef}, uint64_t{16}), uint64_t{0xfffffffffffffff0});
    EXPECT_EQ(tt::align(std::numeric_limits<int32_t>::max(), 1), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(tt::align(std::numeric_limits<uint32_t>::max(), 1u), std::numeric_limits<uint32_t>::max());
}

TEST(Align, SupportsConstantEvaluationAndPreservesReturnType) {
    static_assert(tt::align(0, 16) == 0);
    static_assert(tt::align(0u, 16u) == 0u);
    static_assert(tt::align(17, 16) == 32);
    static_assert(tt::align(32u, 16u) == 32u);
    static_assert(tt::align(uint64_t{0x100000001}, uint32_t{4096}) == uint64_t{0x100001000});
    static_assert(std::is_same_v<decltype(tt::align(uint32_t{1}, uint64_t{16})), uint64_t>);
    static_assert(std::is_same_v<decltype(tt::align(uint8_t{1}, uint8_t{16})), uint8_t>);
    static_assert(std::is_same_v<decltype(tt::align(1, 16)), int>);
}

#ifndef NDEBUG
TEST(AlignDeathTest, RejectsNonPowerOfTwoAlignment) {
    EXPECT_EXIT((void)tt::align(17u, 6u), ::testing::KilledBySignal(SIGABRT), "positive power of two alignment");
}

TEST(AlignDeathTest, RejectsZeroAlignment) {
    EXPECT_EXIT((void)tt::align(17u, 0u), ::testing::KilledBySignal(SIGABRT), "positive power of two alignment");
    EXPECT_EXIT((void)tt::align(0, 0), ::testing::KilledBySignal(SIGABRT), "positive power of two alignment");
}

TEST(AlignDeathTest, RejectsNegativeAlignmentBeforeUnsignedConversion) {
    EXPECT_EXIT((void)tt::align(17u, -8), ::testing::KilledBySignal(SIGABRT), "positive power of two alignment");
    EXPECT_EXIT(
        (void)tt::align(uint32_t{17}, std::numeric_limits<int32_t>::min()),
        ::testing::KilledBySignal(SIGABRT),
        "positive power of two alignment");
}
#endif

}  // namespace
