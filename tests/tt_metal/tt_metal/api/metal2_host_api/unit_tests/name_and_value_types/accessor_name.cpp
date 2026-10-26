// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for AccessorName / ArgName (accessor_name.hpp).
//
// Pure host-side type tests — no device required.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <concepts>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include <fmt/format.h>

#include <tt-metalium/experimental/metal2_host_api/accessor_name.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt_stl/reflection.hpp>

namespace {

namespace m2 = tt::tt_metal::experimental;
using m2::AccessorName;

// ---- conversions (compile-time) ----------------------------------------------

static_assert(std::is_same_v<m2::ArgName, AccessorName>, "ArgName must be the same type as AccessorName");

// Implicit construction from strings, so string literals work wherever a name is expected.
static_assert(std::is_convertible_v<const char*, AccessorName>);
static_assert(std::is_convertible_v<std::string, AccessorName>);
static_assert(std::is_convertible_v<std::string_view, AccessorName>);

// No implicit conversion out: it would make `name == x` ambiguous, and the representation may change.
static_assert(!std::is_convertible_v<AccessorName, std::string>);
static_assert(!std::is_convertible_v<AccessorName, std::string_view>);
static_assert(!std::is_convertible_v<AccessorName, const char*>);

static_assert(std::equality_comparable_with<AccessorName, const char*>);
static_assert(std::equality_comparable_with<AccessorName, std::string>);
static_assert(std::equality_comparable_with<AccessorName, std::string_view>);
static_assert(std::totally_ordered<AccessorName>);

// ---- construction and access -------------------------------------------------

TEST(AccessorNameTest, CPU_DefaultConstructedIsEmpty) {
    const AccessorName name;
    EXPECT_TRUE(name.view().empty());
    EXPECT_TRUE(name.str().empty());
}

TEST(AccessorNameTest, CPU_ConstructsFromEveryStringKind) {
    const std::string str = "num_tiles";
    const std::string_view view = "num_tiles";
    const AccessorName from_literal = "num_tiles";
    const AccessorName from_string = str;
    const AccessorName from_view = view;

    for (const AccessorName& name : {from_literal, from_string, from_view}) {
        EXPECT_EQ(name.view(), "num_tiles");
        EXPECT_EQ(name.str(), "num_tiles");
    }
}

// ---- comparison --------------------------------------------------------------

TEST(AccessorNameTest, CPU_ComparesEqualToStringsWithTheSameCharacters) {
    const AccessorName name = "in";
    const std::string str = "in";
    const std::string_view view = "in";

    EXPECT_TRUE(name == "in");
    EXPECT_TRUE("in" == name);
    EXPECT_TRUE(name == str);
    EXPECT_TRUE(str == name);
    EXPECT_TRUE(name == view);
    EXPECT_TRUE(view == name);
    EXPECT_TRUE(name == AccessorName("in"));

    EXPECT_TRUE(name != "out");
    EXPECT_FALSE(name == "in_");
}

TEST(AccessorNameTest, CPU_OrdersLikeTheUnderlyingStrings) {
    EXPECT_LT(AccessorName("a"), AccessorName("b"));
    EXPECT_LT(AccessorName("a"), AccessorName("aa"));
    EXPECT_GT(AccessorName("b"), AccessorName("a"));

    m2::Group<m2::ArgName> names = {"c", "a", "b"};
    std::ranges::sort(names);
    EXPECT_EQ(names, (m2::Group<m2::ArgName>{"a", "b", "c"}));
}

TEST(AccessorNameTest, CPU_FindsInGroupsByString) {
    const m2::Group<m2::ArgName> names = {"a", "b"};
    EXPECT_NE(std::ranges::find(names, "b"), names.end());
    EXPECT_NE(std::ranges::find(names, std::string("b")), names.end());
    EXPECT_EQ(std::ranges::find(names, "c"), names.end());
}

// ---- hashing -----------------------------------------------------------------

TEST(AccessorNameTest, CPU_HashesLikeTheName) {
    EXPECT_EQ(std::hash<AccessorName>{}(AccessorName("x")), std::hash<std::string_view>{}("x"));

    const std::unordered_set<AccessorName> names = {"x", "y"};
    EXPECT_TRUE(names.contains("x"));
    EXPECT_FALSE(names.contains("z"));
}

TEST(AccessorNameTest, CPU_IsTtslHashable) {
    EXPECT_EQ(
        ttsl::hash::hash_objects_with_default_seed(AccessorName("x")),
        ttsl::hash::hash_objects_with_default_seed(AccessorName("x")));
    EXPECT_NE(
        ttsl::hash::hash_objects_with_default_seed(AccessorName("x")),
        ttsl::hash::hash_objects_with_default_seed(AccessorName("y")));
}

// ---- formatting --------------------------------------------------------------

TEST(AccessorNameTest, CPU_FormatsAsTheName) {
    EXPECT_EQ(fmt::format("'{}'", AccessorName("num_tiles")), "'num_tiles'");

    std::ostringstream os;
    os << AccessorName("num_tiles");
    EXPECT_EQ(os.str(), "num_tiles");
}

TEST(AccessorNameTest, CPU_GroupFormatsLikeAGroupOfStrings) {
    // Error messages format whole name groups; the output must not change with the element type.
    const m2::Group<m2::ArgName> names = {"a", "b"};
    const m2::Group<std::string> strings = {"a", "b"};
    EXPECT_EQ(fmt::format("{}", names), fmt::format("{}", strings));
}

}  // namespace
