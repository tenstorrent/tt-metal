// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for SemiStrongType (metal2_host_api/util/semi_strong_type.hpp).
//
// SemiStrongType is a temporary local mirror of ttsl::StrongType that permits
// IMPLICIT construction from the wrapped value. These tests are deliberately
// self-contained and free of any Metal-specific dependencies so that they can be
// lifted into tt_stl's test suite when SemiStrongType is promoted into
// ttsl::StrongType (as an opt-in implicit mode).
//
// The heart of the file is the compile-time block: SemiStrongType's one piece of
// novel machinery is its constrained forwarding constructor, and its contract
// ("lenient inbound from the wrapped type, strict across wrapper tags") is a
// statement about the type system. So we assert it with static_assert; the
// runtime TESTs then cover the ordinary value-type behaviors.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>

#include <tt-metalium/experimental/metal2_host_api/util/semi_strong_type.hpp>

namespace tt::tt_metal::experimental {
namespace {

// Two distinct string-valued wrappers (the ProgramSpec identifier use case) and
// one numeric wrapper (to confirm the generic template behaves for non-strings).
using NameA = SemiStrongType<std::string, struct NameATag>;
using NameB = SemiStrongType<std::string, struct NameBTag>;
using Count = SemiStrongType<int, struct CountTag>;

// ---- The contract, proven at compile time -------------------------------------

// Lenient inbound: implicitly constructible from the wrapped type and from
// anything convertible to it in a single hop (notably const char* -> std::string,
// the literal-authoring case that motivated this type).
static_assert(std::is_convertible_v<std::string, NameA>);
static_assert(std::is_convertible_v<const char*, NameA>);
static_assert(std::is_convertible_v<int, Count>);

// Strict across wrapper tags: distinct tags do not interconvert, cross-assign, or
// cross-construct. This is the property that makes the type worth having.
static_assert(!std::is_convertible_v<NameA, NameB>);
static_assert(!std::is_assignable_v<NameA&, NameB>);
static_assert(!std::is_constructible_v<NameA, NameB>);

// Strict outbound: no implicit conversion back to the wrapped type (you must
// dereference / .get()), so a wrapper can't silently decay into a bare string.
static_assert(!std::is_convertible_v<NameA, std::string>);

// The forwarding ctor's constraint really constrains: a string wrapper is not
// constructible from something std::string itself can't be built from.
static_assert(!std::is_constructible_v<NameA, int>);

// The forwarding ctor must not shadow the copy/move special members.
static_assert(std::is_copy_constructible_v<NameA>);
static_assert(std::is_move_constructible_v<NameA>);
static_assert(std::is_copy_assignable_v<NameA>);
static_assert(std::is_move_assignable_v<NameA>);

// Trait detection.
static_assert(is_semi_strong_type_v<NameA>);
static_assert(is_semi_strong_type_v<Count>);
static_assert(!is_semi_strong_type_v<std::string>);

// ---- Runtime behaviors ---------------------------------------------------------

TEST(SemiStrongTypeTest, ImplicitConstructionAndAccess) {
    NameA from_literal = "in0";
    EXPECT_EQ(*from_literal, "in0");
    EXPECT_EQ(from_literal.get(), "in0");

    std::string owned = "out0";
    NameA from_string = owned;
    EXPECT_EQ(*from_string, "out0");
}

TEST(SemiStrongTypeTest, DefaultConstructsToWrappedDefault) {
    NameA a;
    EXPECT_EQ(*a, std::string{});
    Count c;
    EXPECT_EQ(*c, 0);
}

TEST(SemiStrongTypeTest, EqualityAndOrdering) {
    NameA alpha = "alpha";
    NameA alpha_again = "alpha";
    NameA beta = "beta";

    EXPECT_EQ(alpha, alpha_again);
    EXPECT_NE(alpha, beta);
    EXPECT_LT(alpha, beta);
    EXPECT_GT(beta, alpha);
}

TEST(SemiStrongTypeTest, UsableInUnorderedSet) {
    std::unordered_set<NameA> names;
    names.insert(NameA{"x"});
    names.insert(NameA{"y"});
    names.insert(NameA{"x"});  // duplicate

    EXPECT_EQ(names.size(), 2u);
    EXPECT_TRUE(names.contains(NameA{"x"}));
    EXPECT_FALSE(names.contains(NameA{"z"}));
}

TEST(SemiStrongTypeTest, UsableAsOrderedMapKey) {
    std::map<NameA, int> by_name;
    by_name[NameA{"a"}] = 1;
    by_name[NameA{"b"}] = 2;

    EXPECT_EQ(by_name.at(NameA{"a"}), 1);
    EXPECT_EQ(by_name.at(NameA{"b"}), 2);
}

TEST(SemiStrongTypeTest, Streams) {
    NameA a = "hello";
    std::ostringstream oss;
    oss << a;
    EXPECT_EQ(oss.str(), "hello");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
