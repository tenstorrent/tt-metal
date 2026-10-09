// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Debug event metadata, layer 1a: the rules a metadata type must follow and the host's unmarshalling
// (hostdev/debug_event_meta.h), checked on the host with no device and no device compiler. The rules are
// static_asserts on the header's predicates, so breaking one fails this file's build; the round trips are gtests.
// Not covered here: the device compiler's view (its spelling of type names, the asm it emits, the error text a
// user sees) -- that is the offline-compile layer -- and base classes, which a structured binding rejects with a
// hard error before any predicate can answer.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "hostdev/debug_event_meta.h"

// Named, not anonymous: a metadata type in an anonymous namespace is itself a rule violation.
namespace debug_event_meta_test {

using namespace tt::debug_event;
using namespace tt::debug_event::detail;

// ---- Types that follow every rule ------------------------------------------------------------------------------

enum class Level : int8_t { Low = -1, High = 7 };
enum Wide : uint64_t { kWideMax = std::numeric_limits<uint64_t>::max() };

struct Plain {
    const char* name;
    uint32_t color;
};
struct Style {
    uint32_t color;
    uint8_t verbosity;
};
struct Nested {
    const char* name;
    Style style;
    uint16_t flags;
};
struct AllWidths {
    const char* name;
    bool on;
    int8_t i8;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    Wide wide;
    Level level;
};
struct Defaulted {
    const char* name;
    uint32_t color = 0x123456;
};
template <class U>
struct Templated {
    const char* name;
    U value;
};

// ---- Types that break exactly one rule each --------------------------------------------------------------------

struct WithArray {
    const char* name;
    uint8_t tags[3];
};
struct InnerArray {
    uint8_t bytes[2];
};
struct NestedArray {
    const char* name;
    InnerArray inner;
};
struct WithConst {
    const char* name;
    const uint32_t color;
};
struct WithFloat {
    const char* name;
    float scale;
};
struct WithBytePointer {
    const char* name;
    const uint8_t* bytes;
};
struct WithStringView {
    const char* name;
    std::string_view text;
};
struct TooMany {
    const char* name;
    uint8_t a, b, c, d, e, f, g, h;
};
struct Empty {};
struct WithConstructor {
    WithConstructor() : name("x") {}
    const char* name;
};
struct ColorFirst {
    uint32_t color;
    const char* name;
};
struct StyleFirst {
    Style style;
    const char* name;
};

namespace {
struct InAnonymousNamespace {
    const char* name;
};
}  // namespace

// How many rules T breaks, counted the way check_type / check_site_type report them: a type with the wrong shape
// is not also told its first field is wrong.
template <class T>
constexpr int rules_broken() {
    return !is_simple_aggregate_v<T> + !has_no_arrays<T>() + !field_count_ok<T>() + !has_no_const_members<T>() +
           !fields_supported<T>() + !has_portable_name<T>() + (shape_ok<T>() && !first_field_is_name<T>());
}

// ---- The rules ---------------------------------------------------------------------------------------------------

static_assert(is_valid_site_type<ZoneMeta>() && is_valid_site_type<ZoneColorMeta>());
static_assert(is_valid_site_type<Plain>() && is_valid_site_type<Nested>() && is_valid_site_type<AllWidths>());
static_assert(is_valid_site_type<Defaulted>() && is_valid_site_type<Templated<uint32_t>>());

static_assert(!has_no_arrays<WithArray>() && rules_broken<WithArray>() == 1);
static_assert(!has_no_arrays<NestedArray>() && rules_broken<NestedArray>() == 1);
static_assert(!has_no_const_members<WithConst>() && rules_broken<WithConst>() == 1);
static_assert(!fields_supported<WithFloat>() && rules_broken<WithFloat>() == 1);
static_assert(!fields_supported<WithBytePointer>() && rules_broken<WithBytePointer>() == 1);
static_assert(!fields_supported<WithStringView>() && rules_broken<WithStringView>() == 1);
static_assert(!field_count_ok<TooMany>() && rules_broken<TooMany>() == 1);
static_assert(!field_count_ok<Empty>() && rules_broken<Empty>() == 1);
static_assert(!is_simple_aggregate_v<WithConstructor> && rules_broken<WithConstructor>() == 1);
static_assert(!first_field_is_name<ColorFirst>() && rules_broken<ColorFirst>() == 1);
static_assert(!first_field_is_name<StyleFirst>() && rules_broken<StyleFirst>() == 1);
static_assert(!has_portable_name<InAnonymousNamespace>() && rules_broken<InAnonymousNamespace>() == 1);
// "<unsigned int>" starts with '<' like "<unnamed struct>"; only the latter is unportable.
static_assert(has_portable_name<Templated<unsigned int>>());

static_assert(site_name_ok(ZoneMeta{"x"}));
static_assert(!site_name_ok(ZoneMeta{""}) && !site_name_ok(ZoneMeta{nullptr}));

// The handle label's hash: equal values share a handle, any difference -- a field, or the type -- gets its own.
static_assert(meta_hash(ZoneColorMeta{"a", 1}) == meta_hash(ZoneColorMeta{"a", 1}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(ZoneColorMeta{"a", 2}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(ZoneColorMeta{"b", 1}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(Plain{"a", 1}));
static_assert(meta_hash(Nested{"a", {1, 2}, 3}) != meta_hash(Nested{"a", {1, 3}, 3}));

// A SiteMeta as the host loader builds it from the device's record: one FieldValue per leaf field, integers as the
// unsigned value of their own width (what the device emits and the loader reads back).
template <class T>
SiteMeta site_meta_of(const T& value) {
    SiteMeta meta;
    meta.signature = signature_of<T>();
    for_each_leaf(value, [&](const auto& field) {
        using F = field_t<decltype(field)>;
        FieldValue v{.code = field_code<F>()};
        if constexpr (std::is_same_v<F, const char*>) {
            v.s = field;
        } else {
            v.bits = field_bits(field);
        }
        meta.fields.push_back(v);
    });
    return meta;
}

// ---- Signatures --------------------------------------------------------------------------------------------------

TEST(DebugEventMeta, CPU_SignatureSpellsTypeNameAndFieldCodes) {
    EXPECT_EQ(signature_of<ZoneMeta>(), "tt::debug_event::ZoneMeta:s");
    EXPECT_EQ(signature_of<ZoneColorMeta>(), "tt::debug_event::ZoneColorMeta:sw");
    EXPECT_EQ(signature_of<Nested>(), "debug_event_meta_test::Nested:s{wb}h");
    EXPECT_EQ(signature_of<AllWidths>(), "debug_event_meta_test::AllWidths:sbbhwqqb");
    EXPECT_EQ(site_meta_of(ZoneColorMeta{"x", 0}).type_name(), "tt::debug_event::ZoneColorMeta");
}

// ---- Round trips through SiteMeta::as<T>() -----------------------------------------------------------------------

TEST(DebugEventMeta, CPU_RoundTripsEveryFieldWidthAndSign) {
    const AllWidths in{
        "all",
        true,
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<uint16_t>::max(),
        -123456,
        std::numeric_limits<int64_t>::min(),
        kWideMax,
        Level::Low};
    const auto out = site_meta_of(in).as<AllWidths>();
    ASSERT_TRUE(out.has_value());
    EXPECT_STREQ(out->name, "all");
    EXPECT_EQ(out->on, in.on);
    EXPECT_EQ(out->i8, in.i8);
    EXPECT_EQ(out->u16, in.u16);
    EXPECT_EQ(out->i32, in.i32);
    EXPECT_EQ(out->i64, in.i64);
    EXPECT_EQ(out->wide, in.wide);
    EXPECT_EQ(out->level, in.level);
}

TEST(DebugEventMeta, CPU_RoundTripsNestedStructs) {
    const auto out = site_meta_of(Nested{"nested", {0xA9A9A9, 2}, 0xBEEF}).as<Nested>();
    ASSERT_TRUE(out.has_value());
    EXPECT_STREQ(out->name, "nested");
    EXPECT_EQ(out->style.color, 0xA9A9A9u);
    EXPECT_EQ(out->style.verbosity, 2u);
    EXPECT_EQ(out->flags, 0xBEEFu);
}

TEST(DebugEventMeta, CPU_RoundTripsDefaultedFields) {
    const auto out = site_meta_of(Defaulted{"defaulted"}).as<Defaulted>();
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->color, 0x123456u);
}

TEST(DebugEventMeta, CPU_AsReturnsNothingForAnotherType) {
    const SiteMeta color = site_meta_of(ZoneColorMeta{"c", 0xFF0000});
    EXPECT_FALSE(color.as<ZoneMeta>().has_value());
    // Same field codes ("sw"), different type: the name in the signature tells them apart.
    EXPECT_FALSE(color.as<Plain>().has_value());
    EXPECT_TRUE(color.as<ZoneColorMeta>().has_value());
}

}  // namespace debug_event_meta_test
