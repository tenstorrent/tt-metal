// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for ArgValue (arg_value.hpp).
//
// Pure host-side type tests — no device required.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>
#include <tt_stl/reflection.hpp>

namespace {

namespace m2 = tt::tt_metal::experimental;
using m2::ArgValue;

enum UnscopedEnum { kUnscoped = 7 };
enum class ScopedEnum { kScoped = 7 };

struct PackedConfig {
    std::uint32_t low : 8;
    std::uint32_t high : 24;
};

// Has both an explicit conversion to uint32_t and an implicit one to int. Assigning it to a uint32_t
// ignores the explicit operator, and so must ArgValue.
struct ExplicitAndImplicit {
    explicit operator std::uint32_t() const { return 1; }
    operator int() const { return 2; }
};

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

// No conversion to an integer (to_uint32() reads the value), and no ordering.
static_assert(std::is_same_v<decltype(std::declval<const ArgValue&>().to_uint32()), std::uint32_t>);
static_assert(!std::is_convertible_v<ArgValue, std::uint32_t>);
static_assert(!std::is_constructible_v<std::uint32_t, ArgValue>);
static_assert(std::equality_comparable<ArgValue>);
static_assert(!std::totally_ordered<ArgValue>);

// ---- to_uint32 ---------------------------------------------------------------

template <typename T>
std::uint32_t stored(T source) {
    return ArgValue(source).to_uint32();
}

TEST(ArgValueTest, CPU_DefaultConstructedIsZero) { EXPECT_EQ(ArgValue().to_uint32(), 0u); }

TEST(ArgValueTest, CPU_AcceptsSourcesThatAssignToUint32) {
    const std::atomic<std::uint32_t> atomic{5};  // not copyable
    EXPECT_EQ(ArgValue(atomic).to_uint32(), 5u);

    const PackedConfig config{.low = 3, .high = 4};
    EXPECT_EQ(ArgValue(config.low).to_uint32(), 3u);  // a bit-field

    const ExplicitAndImplicit both;
    const std::uint32_t assigned = both;
    EXPECT_EQ(assigned, 2u);
    EXPECT_EQ(ArgValue(both).to_uint32(), assigned);  // the implicit conversion, as for assignment
}

TEST(ArgValueTest, CPU_ToUint32IsTheImplicitConversionToUint32) {
    // Each result is what assigning the source to a uint32_t produces.
    EXPECT_EQ(stored(5), 5u);
    EXPECT_EQ(stored(-1), 0xFFFFFFFFu);  // wraps
    EXPECT_EQ(stored(std::size_t{12}), 12u);
    EXPECT_EQ(stored(std::uint64_t{0x1'0000'0005}), 5u);  // keeps the low 32 bits
    EXPECT_EQ(stored(std::int64_t{-2}), 0xFFFFFFFEu);
    EXPECT_EQ(stored(std::int16_t{-3}), 0xFFFFFFFDu);
    EXPECT_EQ(stored(std::uint8_t{200}), 200u);
    EXPECT_EQ(stored('A'), 65u);
    EXPECT_EQ(stored(true), 1u);
    EXPECT_EQ(stored(false), 0u);
    EXPECT_EQ(stored(2.9), 2u);  // truncates
    EXPECT_EQ(stored(2.9f), 2u);
    EXPECT_EQ(stored(kUnscoped), 7u);
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
