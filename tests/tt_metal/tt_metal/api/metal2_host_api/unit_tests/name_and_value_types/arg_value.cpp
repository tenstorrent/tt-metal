// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for ArgValue (arg_value.hpp).
//
// Pure host-side type tests — no device required.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>
#include <tt_stl/reflection.hpp>

#include "impl/metal2_host_api/arg_value_access.hpp"

namespace {

namespace m2 = tt::tt_metal::experimental;
using m2::ArgValue;
using m2::ArgValueAccess;

enum UnscopedEnum { kUnscoped = 7 };
enum class ScopedEnum { kScoped = 7 };

// ---- conversions (compile-time) ----------------------------------------------

// ArgValue accepts exactly what converts implicitly to uint32_t...
static_assert(std::is_convertible_v<int, ArgValue>);
static_assert(std::is_convertible_v<std::size_t, ArgValue>);
static_assert(std::is_convertible_v<std::int64_t, ArgValue>);
static_assert(std::is_convertible_v<bool, ArgValue>);
static_assert(std::is_convertible_v<double, ArgValue>);
static_assert(std::is_convertible_v<float, ArgValue>);
static_assert(std::is_convertible_v<UnscopedEnum, ArgValue>);

// ...and nothing else.
static_assert(!std::is_constructible_v<ArgValue, ScopedEnum>);
static_assert(!std::is_constructible_v<ArgValue, const char*>);
static_assert(!std::is_constructible_v<ArgValue, std::string>);
static_assert(!std::is_constructible_v<ArgValue, std::nullptr_t>);

// Write-only: no conversion back to an integer, and no ordering.
static_assert(!std::is_convertible_v<ArgValue, std::uint32_t>);
static_assert(!std::is_constructible_v<std::uint32_t, ArgValue>);
static_assert(std::equality_comparable<ArgValue>);
static_assert(!std::totally_ordered<ArgValue>);

// ---- stored value ------------------------------------------------------------

template <typename T>
std::uint32_t stored(T source) {
    return ArgValueAccess::u32(ArgValue(source));
}

TEST(ArgValueTest, CPU_DefaultConstructedIsZero) { EXPECT_EQ(ArgValueAccess::u32(ArgValue()), 0u); }

TEST(ArgValueTest, CPU_StoresTheImplicitConversionToUint32) {
    // Each result must be exactly what assigning the source to a uint32_t produces.
    EXPECT_EQ(stored(5), static_cast<std::uint32_t>(5));
    EXPECT_EQ(stored(-1), static_cast<std::uint32_t>(-1));  // wraps to 0xFFFFFFFF
    EXPECT_EQ(stored(std::size_t{12}), static_cast<std::uint32_t>(std::size_t{12}));
    EXPECT_EQ(stored(std::uint64_t{0x1'0000'0005}), static_cast<std::uint32_t>(std::uint64_t{0x1'0000'0005}));
    EXPECT_EQ(stored(std::int64_t{-2}), static_cast<std::uint32_t>(std::int64_t{-2}));
    EXPECT_EQ(stored(std::int16_t{-3}), static_cast<std::uint32_t>(std::int16_t{-3}));
    EXPECT_EQ(stored(std::uint8_t{200}), static_cast<std::uint32_t>(std::uint8_t{200}));
    EXPECT_EQ(stored('A'), static_cast<std::uint32_t>('A'));
    EXPECT_EQ(stored(true), static_cast<std::uint32_t>(true));
    EXPECT_EQ(stored(false), static_cast<std::uint32_t>(false));
    EXPECT_EQ(stored(2.9), static_cast<std::uint32_t>(2.9));  // truncates
    EXPECT_EQ(stored(2.9f), static_cast<std::uint32_t>(2.9f));
    EXPECT_EQ(stored(kUnscoped), static_cast<std::uint32_t>(kUnscoped));
}

// ---- comparison --------------------------------------------------------------

TEST(ArgValueTest, CPU_ComparesEqualToTheSameValue) {
    EXPECT_TRUE(ArgValue(1) == ArgValue(1));
    EXPECT_TRUE(ArgValue(1) == 1);
    EXPECT_TRUE(1u == ArgValue(1));
    EXPECT_TRUE(ArgValue(-1) == 0xFFFFFFFFu);
    EXPECT_TRUE(ArgValue(1) != 2);
    EXPECT_EQ(ArgValue(7), 7u);
}

// ---- hashing -----------------------------------------------------------------

TEST(ArgValueTest, CPU_EqualValuesHashEqual) {
    EXPECT_EQ(std::hash<ArgValue>{}(ArgValue(5)), std::hash<ArgValue>{}(ArgValue(5u)));
    EXPECT_EQ(
        ttsl::hash::hash_objects_with_default_seed(ArgValue(5)),
        ttsl::hash::hash_objects_with_default_seed(ArgValue(5u)));
    EXPECT_NE(
        ttsl::hash::hash_objects_with_default_seed(ArgValue(5)),
        ttsl::hash::hash_objects_with_default_seed(ArgValue(6)));
}

// ---- formatting --------------------------------------------------------------

TEST(ArgValueTest, CPU_FormatsAsTheStoredValue) {
    EXPECT_EQ(fmt::format("{}", ArgValue(42)), "42");
    EXPECT_EQ(fmt::format("{}", ArgValue(-1)), "4294967295");
    EXPECT_EQ(fmt::format("[{}, {}]", ArgValue(1), ArgValue(2)), "[1, 2]");

    std::ostringstream os;
    os << ArgValue(42);
    EXPECT_EQ(os.str(), "42");
}

}  // namespace
