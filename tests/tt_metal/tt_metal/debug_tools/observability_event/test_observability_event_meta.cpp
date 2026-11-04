// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Observability event metadata, layer 1a: the rules a metadata type must follow
// (observability/api/observability_event_meta.h), the device's marshalling helpers
// (hostdev/observability_event_descriptor.h) and the host's unmarshalling
// (tt-metalium/experimental/observability_event_meta.hpp), checked on the host with no device and no device compiler.
// The rules are static_asserts on the header's predicates, so breaking one fails this file's build. Round trips of
// every shape through the loader are in test_observability_event_meta_loader.cpp. Not covered here: the device
// compiler's view (its spelling of type names, the asm it emits, the error text a user sees) -- that is the
// offline-compile layer, test_observability_event_meta_compile.cpp.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/observability_event_meta.hpp>

#include "hostdev/observability_event_descriptor.h"

// Named, not anonymous: a metadata type in an anonymous namespace is itself a rule violation.
namespace observability_event_meta_test {

using namespace tt::observability_event;
using namespace tt::observability_event::detail;

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
struct NameBase {
    const char* name;
};
struct WithBase : NameBase {
    uint32_t color;
};
struct EmptyBase {};
struct WithEmptyBase : EmptyBase {
    const char* name;
};
struct NestedBase {
    const char* name;
    WithBase inner;
};

namespace {
struct InAnonymousNamespace {
    const char* name;
};
}  // namespace

// How many rules T breaks, counted the way check_type / check_event_type report them: a type with the wrong shape
// is not also told its first field is wrong.
template <class T>
constexpr int rules_broken() {
    return !is_simple_aggregate_v<T> + !has_no_bases<T>() + !has_no_arrays<T>() + !field_count_ok<T>() +
           !has_no_const_members<T>() + !fields_supported<T>() + !has_portable_name<T>() +
           (shape_ok<T>() && !first_field_is_name<T>());
}

// ---- The rules ---------------------------------------------------------------------------------------------------

static_assert(is_valid_event_type<EventMeta>() && is_valid_event_type<ZoneColorMeta>());
static_assert(is_valid_event_type<Plain>() && is_valid_event_type<Nested>() && is_valid_event_type<AllWidths>());
static_assert(is_valid_event_type<Defaulted>() && is_valid_event_type<Templated<uint32_t>>());

static_assert(!has_no_bases<WithBase>() && rules_broken<WithBase>() == 1);
static_assert(!has_no_bases<WithEmptyBase>() && rules_broken<WithEmptyBase>() == 1);
static_assert(!has_no_bases<NestedBase>() && rules_broken<NestedBase>() == 1);
static_assert(has_no_bases<Nested>() && has_no_bases<AllWidths>());
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

// Strings reach the assembler on one line: quotes and backslashes escaped, every other byte outside printable ASCII
// as a three-digit octal escape (which the assembler never extends into a following digit).
constexpr bool quotes_as(const char* in, std::string_view want) {
    AsmText a;
    a.put_quoted(in);
    return std::string_view(a.data(), a.size()) == want;
}
static_assert(quotes_as("plain", R"("plain")"));
static_assert(quotes_as(R"(a"b\c)", R"("a\"b\\c")"));
static_assert(quotes_as("line\nbreak\ttab", R"("line\012break\011tab")"));
static_assert(quotes_as("\n1", R"("\0121")"));
static_assert(quotes_as("\xC3\xA9\x7F\x01", R"("\303\251\177\001")"));

static_assert(event_name_ok(EventMeta{"x"}));
static_assert(!event_name_ok(EventMeta{""}) && !event_name_ok(EventMeta{nullptr}));

// The handle label's hash: equal values share a handle, any difference -- a field, or the type -- gets its own.
static_assert(meta_hash(ZoneColorMeta{"a", 1}) == meta_hash(ZoneColorMeta{"a", 1}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(ZoneColorMeta{"a", 2}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(ZoneColorMeta{"b", 1}));
static_assert(meta_hash(ZoneColorMeta{"a", 1}) != meta_hash(Plain{"a", 1}));
static_assert(meta_hash(Nested{"a", {1, 2}, 3}) != meta_hash(Nested{"a", {1, 3}, 3}));

// A Metadata as the host loader builds it from the device's record: one FieldValue per leaf field, integers as 64
// bits, sign-extended from a signed type (what the device emits and the loader reads back).
template <class T>
Metadata metadata_of(const T& value) {
    Metadata meta;
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

// ---- Integer fields --------------------------------------------------------------------------------------------
// One code and 8 bytes whatever the width, so the device's data model (long and pointers 4 bytes) and the host's
// (8 bytes) name a type alike.
static_assert(field_code<bool>() == 'i' && field_code<int8_t>() == 'i' && field_code<uint16_t>() == 'i');
static_assert(field_code<long>() == 'i' && field_code<int32_t>() == 'i' && field_code<uint64_t>() == 'i');
static_assert(field_code<Level>() == 'i' && field_code<Wide>() == 'i');
static_assert(code_bytes('i') == 8 && code_bytes('s') == 4 && code_bytes('w') == 0);

// Sign-extended from a signed type, so the value survives any width on either side.
static_assert(field_bits(int8_t{-1}) == ~uint64_t{0} && field_bits(int32_t{-3}) == ~uint64_t{2});
static_assert(field_bits(uint8_t{0xFF}) == 0xFF && field_bits(Level::Low) == ~uint64_t{0});

template <class F>
constexpr bool reads_back(uint64_t bits, F expected) {
    F out{};
    return field_from_bits(bits, out) && out == expected;
}
template <class F>
constexpr bool does_not_fit(uint64_t bits) {
    F out{};
    return !field_from_bits(bits, out);
}
// A narrower or wider host field reads any value that fits it.
static_assert(reads_back<int64_t>(field_bits(int8_t{-1}), -1) && reads_back<int8_t>(field_bits(int64_t{-1}), -1));
static_assert(reads_back<uint8_t>(field_bits(uint32_t{200}), 200) && reads_back<long>(field_bits(int32_t{-5}), -5L));
static_assert(reads_back<Level>(field_bits(Level::Low), Level::Low) && reads_back<bool>(1, true));
// A value the host's field cannot hold is not read.
static_assert(does_not_fit<uint8_t>(256) && does_not_fit<uint8_t>(field_bits(int8_t{-1})));
static_assert(does_not_fit<int8_t>(field_bits(uint8_t{200})) && does_not_fit<bool>(2));
static_assert(does_not_fit<Level>(field_bits(int16_t{300})));

// ---- Signatures --------------------------------------------------------------------------------------------------

TEST(ObservabilityEventMeta, CPU_SignatureSpellsTypeNameAndFieldCodes) {
    EXPECT_EQ(signature_of<EventMeta>(), "tt::observability_event::EventMeta:s");
    EXPECT_EQ(signature_of<ZoneColorMeta>(), "tt::observability_event::ZoneColorMeta:si");
    EXPECT_EQ(signature_of<Nested>(), "observability_event_meta_test::Nested:s{ii}i");
    EXPECT_EQ(signature_of<AllWidths>(), "observability_event_meta_test::AllWidths:siiiiiii");
    EXPECT_EQ(metadata_of(ZoneColorMeta{"x", 0}).type_name(), "tt::observability_event::ZoneColorMeta");
}

// ---- Decoding through Metadata::as<T>() -----------------------------------------------------------------------------

TEST(ObservabilityEventMeta, CPU_AsReturnsNothingForAnotherType) {
    const Metadata color = metadata_of(ZoneColorMeta{"c", 0xFF0000});
    EXPECT_FALSE(color.as<EventMeta>().has_value());
    // Same field codes ("si"), different type: the name in the signature tells them apart.
    EXPECT_FALSE(color.as<Plain>().has_value());
    EXPECT_TRUE(color.as<ZoneColorMeta>().has_value());
}

TEST(ObservabilityEventMeta, CPU_AsReadsEveryWidthBack) {
    const AllWidths value{"all", true, -1, 0xFFFF, -3, std::numeric_limits<int64_t>::min(), kWideMax, Level::Low};
    const auto back = metadata_of(value).as<AllWidths>();
    ASSERT_TRUE(back.has_value());
    EXPECT_STREQ(back->name, "all");
    EXPECT_TRUE(back->on);
    EXPECT_EQ(back->i8, -1);
    EXPECT_EQ(back->u16, 0xFFFF);
    EXPECT_EQ(back->i32, -3);
    EXPECT_EQ(back->i64, std::numeric_limits<int64_t>::min());
    EXPECT_EQ(back->wide, kWideMax);
    EXPECT_EQ(back->level, Level::Low);
}

// The device recorded a type by T's name whose fields or values differ from the host's T: a type that differs between
// the two sides. as<T>() returns nothing and logs a warning naming both signatures.
TEST(ObservabilityEventMeta, CPU_AsReturnsNothingWhenTheSameNameDiffers) {
    Metadata extra_field = metadata_of(ZoneColorMeta{"c", 1});
    extra_field.signature = "tt::observability_event::ZoneColorMeta:sii";
    extra_field.fields.push_back(FieldValue{.code = 'i', .bits = 2});
    EXPECT_FALSE(extra_field.as<ZoneColorMeta>().has_value());

    Metadata too_big = metadata_of(Plain{"p", 1});
    too_big.fields[1].bits = uint64_t{1} << 32;  // the device's color was wider than the host's uint32_t
    EXPECT_FALSE(too_big.as<Plain>().has_value());
    too_big.fields[1].bits = 0xFFFFFFFF;
    EXPECT_TRUE(too_big.as<Plain>().has_value());
}

}  // namespace observability_event_meta_test
