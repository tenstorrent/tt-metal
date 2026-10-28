// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Validates the behavior of the fabric manifest's struct layout helpers (struct_layout.hpp), mostly
// for error cases that we aren't representing in the real layouts.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout_helpers.hpp"

namespace tt::tt_fabric::layout_test {

struct ThreeWords {
    uint32_t a;
    uint32_t b;
    uint32_t c;
};

struct HalvedWord {
    union {
        uint32_t all;
        struct {
            uint16_t lo;
            uint16_t hi;
        };
    };
};

enum class Color : uint8_t { RED, GREEN };

// An enum with a value far outside the range enchantum scans, which needs an EnumLayout.
enum class ContainsOutsideRangeEnumerator : uint32_t { NEAR = 1, AWAY = 0xA0B0C0D0 };

struct Inner {
    uint16_t x;
    uint16_t y;
};

struct Outer {
    Inner inner;
    Color color;
    int8_t delta;
    uint16_t counts[2];
};

struct WithArrays {
    uint8_t id;
    Color colors[3];
    std::array<Inner, 2> points;
};

// A class template, whose instances enchantum::type_name can't name.
template <uint32_t N>
struct ByteRun {
    uint8_t bytes[N];
};

struct HoldsByteRun {
    uint32_t word;
    ByteRun<4> run;
};

struct WordAndByte {
    uint32_t word;
    uint8_t byte;
};

// WordAndByte with a field added inside its tail padding: sizeof and every other offset stay the same.
struct WordAndByteThenExtra {
    uint32_t word;
    uint8_t byte;
    uint8_t extra;
};

// 3 bytes of compiler padding between a and b.
struct ByteThenWord {
    uint8_t a;
    uint32_t b;
};

// ByteThenWord with a field added inside its padding: sizeof and b's offset stay the same.
struct ByteThenFlagsThenWord {
    uint8_t a;
    uint8_t flags;
    uint32_t b;
};

// A non-aggregate, since it declares a constructor, with compiler padding between a and b.
struct ConstructedByteThenWord {
    ConstructedByteThenWord() = default;
    uint8_t a = 0;
    uint32_t b = 0;
};

// A non-aggregate with no compiler padding.
struct ConstructedWords {
    ConstructedWords() = default;
    uint32_t a = 0;
    uint32_t b = 0;
};

enum class Shape : uint8_t { DOT, LINE, BOX };

union ShapeFields {
    Inner dot;
    uint32_t line[2];
    uint16_t box;
};

struct Tagged {
    Shape shape;
    ShapeFields fields;
};

enum class Width : uint8_t { HALF, WORD };

// A union class template, whose instances enchantum::type_name can't name.
template <uint32_t N>
union Words {
    uint16_t halves[2 * N];
    uint32_t words[N];
};

struct HoldsWords {
    Width width;
    Words<1> value;
};

// A setting fixed for the whole run, and a struct whose anonymous union it picks the member of.
enum class Mode : uint8_t { NARROW, WIDE };

struct Table {
    uint32_t header;
    union {
        uint32_t narrow;
        uint32_t wide[2];
    };
    uint32_t trailer;
};

}  // namespace tt::tt_fabric::layout_test

namespace tt::tt_fabric::layout {

template <>
struct EnumLayout<layout_test::ContainsOutsideRangeEnumerator> {
    static constexpr std::array enumerators = {
        Enumerator{"NEAR", static_cast<uint32_t>(layout_test::ContainsOutsideRangeEnumerator::NEAR)},
        Enumerator{"AWAY", static_cast<uint32_t>(layout_test::ContainsOutsideRangeEnumerator::AWAY)},
    };
};

template <>
struct StructLayout<layout_test::Inner> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_test::Inner, x),
        LAYOUT_MEMBER(layout_test::Inner, y),
    };
};

template <>
struct StructLayout<layout_test::ByteRun<4>> {
    static constexpr std::string_view name = template_name<"ByteRun", 4>;
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_test::ByteRun<4>, bytes),
    };
};

template <>
struct TaggedUnionLayout<layout_test::ShapeFields> {
    using Tag = layout_test::Shape;
    static constexpr std::array cases = {
        LAYOUT_TAGGED_UNION_CASE(layout_test::ShapeFields, dot, Tag::DOT),
        LAYOUT_TAGGED_UNION_CASE(layout_test::ShapeFields, line, Tag::LINE),
        LAYOUT_TAGGED_UNION_CASE(layout_test::ShapeFields, box, Tag::BOX),
    };
};

template <>
struct TaggedUnionLayout<layout_test::Words<1>> {
    using Tag = layout_test::Width;
    static constexpr std::string_view name = "Words<1>";
    static constexpr std::array cases = {
        LAYOUT_TAGGED_UNION_CASE(layout_test::Words<1>, halves, Tag::HALF),
        LAYOUT_TAGGED_UNION_CASE(layout_test::Words<1>, words, Tag::WORD),
    };
};

template <>
struct StructLayout<layout_test::Table> {
    static constexpr std::array value_cases = {
        LAYOUT_RUN_SETTING_UNION_CASE(layout_test::Table, narrow, layout_test::Mode::NARROW),
        LAYOUT_RUN_SETTING_UNION_CASE(layout_test::Table, wide, layout_test::Mode::WIDE),
    };
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_test::Table, header),
        LAYOUT_RUN_SETTING_UNION(layout_test::Mode, value_cases),
        LAYOUT_MEMBER(layout_test::Table, trailer),
    };
};
static_assert(check_struct_layout<layout_test::Table>());

namespace {

using layout_test::ByteRun;
using layout_test::ByteThenFlagsThenWord;
using layout_test::ByteThenWord;
using layout_test::Color;
using layout_test::ConstructedByteThenWord;
using layout_test::ConstructedWords;
using layout_test::ContainsOutsideRangeEnumerator;
using layout_test::HalvedWord;
using layout_test::HoldsByteRun;
using layout_test::HoldsWords;
using layout_test::Inner;
using layout_test::Mode;
using layout_test::Outer;
using layout_test::Shape;
using layout_test::ShapeFields;
using layout_test::Table;
using layout_test::Tagged;
using layout_test::ThreeWords;
using layout_test::WithArrays;
using layout_test::WordAndByte;
using layout_test::WordAndByteThenExtra;
using layout_test::Words;

// A member missing from the middle of the list, such as a field added to a device struct but not to its layout, leaves
// a gap which should cause an error since the next listed member doesn't start where the one before it ended.
TEST(StructLayout, CoverageRejectsAGap) {
    EXPECT_EQ(
        find_layout_error<ThreeWords>(std::array{
            LAYOUT_MEMBER(ThreeWords, a),
            LAYOUT_MEMBER(ThreeWords, c),
        }),
        LayoutError::COVERAGE);
}

// A member missing from the end of the list should cause an error since the last listed member doesn't end at the end
// of the actual struct.
TEST(StructLayout, CoverageRejectsAMissingTail) {
    EXPECT_EQ(
        find_layout_error<ThreeWords>(std::array{
            LAYOUT_MEMBER(ThreeWords, a),
            LAYOUT_MEMBER(ThreeWords, b),
        }),
        LayoutError::COVERAGE);
}

// Listing more than one of a union's members should cause an error since their bytes overlap, so the same bytes are
// counted twice. Listing just one of them should pass since that one member covers the union's bytes.
TEST(StructLayout, CoverageRejectsOverlappingUnionMembers) {
    EXPECT_EQ(
        find_layout_error<HalvedWord>(std::array{
            LAYOUT_MEMBER(HalvedWord, all),
            LAYOUT_MEMBER(HalvedWord, lo),
            LAYOUT_MEMBER(HalvedWord, hi),
        }),
        LayoutError::COVERAGE);
    EXPECT_EQ(find_layout_error<HalvedWord>(std::array{LAYOUT_MEMBER(HalvedWord, all)}), LayoutError::NONE);
}

// Each kind of member should get its own element since decode reads each kind differently. A C array member should
// have a size that covers all of its elements and a count of its length.
TEST(StructLayout, MemberTypesAndNames) {
    constexpr std::array members = {
        LAYOUT_MEMBER(Outer, inner),
        LAYOUT_MEMBER(Outer, color),
        LAYOUT_MEMBER(Outer, delta),
        LAYOUT_MEMBER(Outer, counts),
    };
    static_assert(find_layout_error<Outer>(members) == LayoutError::NONE);

    EXPECT_TRUE(members[0].type.element == Element{element::Struct{&struct_description<Inner>}});
    EXPECT_TRUE(members[1].type.element == Element{element::Enum{&enum_description<Color>}});
    EXPECT_TRUE(members[2].type.element == Element{element::Int{}});
    EXPECT_TRUE(members[3].type.element == Element{element::Uint{}});
    EXPECT_EQ(members[3].offset, 6u);
    EXPECT_EQ(members[3].type.size, 4u);
    EXPECT_EQ(members[3].type.count, 2u);
}

// A struct whose StructLayout sets a name, such as a class template instance, should be known by that name wherever
// it's a member, since its type name isn't available.
TEST(StructLayout, StructLayoutNameNamesTheStruct) {
    EXPECT_EQ(struct_description<ByteRun<4>>.name, "ByteRun<4>");
    EXPECT_TRUE(type_of<ByteRun<4>>() == (Type{element::Struct{&struct_description<ByteRun<4>>}, 4, 0}));

    constexpr std::array members = {
        LAYOUT_MEMBER(HoldsByteRun, word),
        LAYOUT_MEMBER(HoldsByteRun, run),
    };
    static_assert(find_layout_error<HoldsByteRun>(members) == LayoutError::NONE);
    EXPECT_TRUE(members[1].type.element == Element{element::Struct{&struct_description<ByteRun<4>>}});
}

// A template instance's name should be spelled as it is written in code.
TEST(StructLayout, TemplateNameSpellsTheInstance) {
    EXPECT_EQ((template_name<"ByteRun", 4>), "ByteRun<4>");
    EXPECT_EQ((template_name<"ByteRun", 0>), "ByteRun<0>");
    EXPECT_EQ((template_name<"T", 1, false>), "T<1,false>");
    EXPECT_EQ((template_name<"T", true, uint64_t{18446744073709551615u}>), "T<true,18446744073709551615>");
}

// One initializer fills an anonymous union, so listing the union by two of its members counts one more initializer
// than the struct takes. Marking them as filled by one initializer should make the count match, without changing
// their offsets or types.
TEST(StructLayout, FilledByOneInitializerCountsTheGroupOnce) {
    constexpr std::array halves = {LAYOUT_MEMBER(HalvedWord, lo), LAYOUT_MEMBER(HalvedWord, hi)};
    static_assert(initializer_count<HalvedWord> == 1);
    EXPECT_EQ(find_layout_error<HalvedWord>(halves), LayoutError::UNLISTED_FIELD);

    constexpr auto filled = filled_by_one_initializer(halves);
    EXPECT_EQ(find_layout_error<HalvedWord>(filled), LayoutError::NONE);
    EXPECT_EQ(filled[0].initializers, 1u);
    EXPECT_EQ(filled[1].initializers, 0u);
    EXPECT_TRUE(filled[1].type == halves[1].type);
    EXPECT_EQ(filled[1].offset, halves[1].offset);
}

constexpr auto three_words_joined_members = concat_members(
    std::array{LAYOUT_MEMBER(ThreeWords, a)}, std::array{LAYOUT_MEMBER(ThreeWords, b), LAYOUT_MEMBER(ThreeWords, c)});

// Joined member lists should keep every member in order, so a layout built from shared groups of members passes the
// same checks as one listed in full.
TEST(StructLayout, ConcatMembersJoinsListsInOrder) {
    static_assert(check_struct_layout<ThreeWords, three_words_joined_members>());
    ASSERT_EQ(three_words_joined_members.size(), 3u);
    EXPECT_EQ(three_words_joined_members[0].name, "a");
    EXPECT_EQ(three_words_joined_members[1].name, "b");
    EXPECT_EQ(three_words_joined_members[2].name, "c");
}

// A C array of enums should be an Enum element with the array's length as its count, rather than a scalar, since
// decode reads that many elements. It should match both on its own and as a struct member.
TEST(StructLayout, TypeOfCArray) {
    EXPECT_TRUE(type_of<Color[3]>() == (Type{element::Enum{&enum_description<Color>}, 3, 3}));

    constexpr std::array members = {
        LAYOUT_MEMBER(WithArrays, id),
        LAYOUT_MEMBER(WithArrays, colors),
        LAYOUT_MEMBER(WithArrays, points),
    };
    static_assert(find_layout_error<WithArrays>(members) == LayoutError::NONE);
    EXPECT_EQ(members[1].offset, 1u);
    EXPECT_TRUE(members[1].type == (Type{element::Enum{&enum_description<Color>}, 3, 3}));
}

// A std::array should get the same Type as a C array of the same element and length since both lay out their elements
// the same way, so decode reads them the same way. It should match both on its own and as a struct member.
TEST(StructLayout, TypeOfStdArray) {
    constexpr Type type = type_of<std::array<Inner, 2>>();
    EXPECT_TRUE(type == (Type{element::Struct{&struct_description<Inner>}, 8, 2}));
    EXPECT_TRUE(type == type_of<Inner[2]>());

    constexpr Member points = LAYOUT_MEMBER(WithArrays, points);
    EXPECT_EQ(points.offset, 4u);
    EXPECT_TRUE(points.type == (Type{element::Struct{&struct_description<Inner>}, 8, 2}));
}

// An array whose length is known only at run time is described with array_of, which should give the same Type as a C
// array of that length since its elements are laid out the same way.
TEST(StructLayout, ArrayOfMatchesAFixedLengthArray) {
    EXPECT_TRUE(array_of<Inner>(2) == type_of<Inner[2]>());
    EXPECT_TRUE(array_of<uint32_t>(5) == (Type{element::Uint{}, 20, 5}));
}

// The compiler's padding after a struct's last member has no name to list, so LAYOUT_TAIL_PAD should cover it from
// where the last member ends to the end of the struct, since the coverage check would otherwise fail on the missing
// tail.
TEST(StructLayout, TailPadCoversTheBytesAfterTheLastMember) {
    constexpr std::array members = {
        LAYOUT_MEMBER(WordAndByte, word),
        LAYOUT_MEMBER(WordAndByte, byte),
        LAYOUT_TAIL_PAD(WordAndByte, byte),
    };
    static_assert(find_layout_error<WordAndByte>(members) == LayoutError::NONE);
    EXPECT_EQ(members[2].offset, 5u);
    EXPECT_TRUE(members[2].type == (Type{element::Pad{}, 3, 0}));
    EXPECT_EQ(members[2].initializers, 0u);
}

// The compiler's padding between two members has no name to list either, so LAYOUT_PAD should cover it from where the
// first member ends to where the second starts. It should take no initializers since no field backs it.
TEST(StructLayout, PadCoversTheBytesBetweenMembers) {
    constexpr std::array members = {
        LAYOUT_MEMBER(ByteThenWord, a),
        LAYOUT_PAD(ByteThenWord, a, b),
        LAYOUT_MEMBER(ByteThenWord, b),
    };
    static_assert(find_layout_error<ByteThenWord>(members) == LayoutError::NONE);
    EXPECT_EQ(members[1].name, "padding_after_a");
    EXPECT_EQ(members[1].offset, 1u);
    EXPECT_TRUE(members[1].type == (Type{element::Pad{}, 3, 0}));
    EXPECT_EQ(members[1].initializers, 0u);
}

// Each field should take as many initializers as brace elision gives it in its struct's T{...}, since
// initializer_count counts a struct's fields that way: one per element of a C array, including an array of arrays, and
// one for anything else, a struct, a union or a std::array included.
TEST(StructLayout, InitializersOfFollowsBraceElision) {
    static_assert(initializers_of<uint32_t>() == 1);
    static_assert(initializers_of<Color>() == 1);
    static_assert(initializers_of<uint8_t[12]>() == 12);
    static_assert(initializers_of<uint8_t[2][3]>() == 6);
    static_assert(initializers_of<Inner[2]>() == 2);
    static_assert(initializers_of<Inner>() == 1);
    static_assert(initializers_of<ShapeFields>() == 1);
    static_assert(initializers_of<std::array<Inner, 2>>() == 1);
}

// initializer_count should count a struct's fields the way initializers_of does, since the layout check compares the
// two: Outer's counts[2] takes 2, WithArrays' colors[3] takes 3 and its std::array takes 1, a named union field takes
// 1, and an anonymous union takes 1 too, since the dummy value converts to the union's unnamed type.
TEST(StructLayout, InitializerCountMatchesInitializersOf) {
    static_assert(initializer_count<Outer> == 5);
    static_assert(initializer_count<WithArrays> == 5);
    static_assert(initializer_count<Tagged> == 2);
    static_assert(initializer_count<HalvedWord> == 1);
}

// A field added inside compiler padding between members should cause an error. The padding entry still covers the same
// bytes, so the coverage check passes, but the struct now takes one more initializer than the layout accounts for.
TEST(StructLayout, CountRejectsAFieldAddedInsidePadding) {
    constexpr std::array members = {
        LAYOUT_MEMBER(ByteThenFlagsThenWord, a),
        LAYOUT_PAD(ByteThenFlagsThenWord, a, b),  // now also covers flags
        LAYOUT_MEMBER(ByteThenFlagsThenWord, b),
    };
    EXPECT_EQ(find_layout_error<ByteThenFlagsThenWord>(members), LayoutError::UNLISTED_FIELD);
}

// A field added inside tail padding should cause an error for the same reason: the tail padding entry still covers the
// same bytes, but the struct now takes one more initializer than the layout accounts for.
TEST(StructLayout, CountRejectsAFieldAddedInsideTailPadding) {
    constexpr std::array members = {
        LAYOUT_MEMBER(WordAndByteThenExtra, word),
        LAYOUT_MEMBER(WordAndByteThenExtra, byte),
        LAYOUT_TAIL_PAD(WordAndByteThenExtra, byte),  // now also covers extra
    };
    EXPECT_EQ(find_layout_error<WordAndByteThenExtra>(members), LayoutError::UNLISTED_FIELD);
}

constexpr std::array constructed_byte_then_word_members = {
    LAYOUT_MEMBER(ConstructedByteThenWord, a),
    LAYOUT_PAD(ConstructedByteThenWord, a, b),
    LAYOUT_MEMBER(ConstructedByteThenWord, b),
};

// A non-aggregate's initializers can't be counted, so where its layout lists compiler padding, its fields should be
// counted with a structured binding instead, since a field added inside that padding would otherwise go unnoticed. This
// layout lists 2 fields (the padding isn't one) and the struct has 2, so it binds and builds. A layout listing fewer
// fields than the struct has fails to compile ("decomposes into 3 elements, but only 2 names were provided"), so that
// case can't be a test here. Without compiler padding, no binding should be needed, since a new field would then have
// to shift later fields or grow the struct, which coverage catches.
TEST(StructLayout, NonAggregateWithPaddingIsCountedByBinding) {
    static_assert(!std::is_aggregate_v<ConstructedByteThenWord>);
    static_assert(listed_fields(constructed_byte_then_word_members) == 2);
    static_assert(check_struct_layout<ConstructedByteThenWord, constructed_byte_then_word_members>());
    EXPECT_EQ(find_layout_error<ConstructedByteThenWord>(constructed_byte_then_word_members), LayoutError::NONE);

    EXPECT_EQ(
        find_layout_error<ConstructedWords>(std::array{
            LAYOUT_MEMBER(ConstructedWords, a),
            LAYOUT_MEMBER(ConstructedWords, b),
        }),
        LayoutError::NONE);
}

// A union whose every tag value picks exactly one member, and whose widest member fills it, should pass since decode
// can then read it whatever the tag holds.
TEST(StructLayout, TaggedUnionAcceptsEveryTagPickingOneMember) {
    EXPECT_TRUE(validate_tagged_union_cases<ShapeFields>(TaggedUnionLayout<ShapeFields>::cases));
}

// One member picked by several tag values should pass since each tag value still picks exactly one member, so decode
// knows what to read whatever the tag holds. Here line is picked by both LINE and BOX.
TEST(StructLayout, TaggedUnionAcceptsAMemberPickedBySeveralTags) {
    EXPECT_TRUE(validate_tagged_union_cases<ShapeFields>(std::array{
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, dot, Shape::DOT),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::BOX),
    }));
}

// Two members listing the same tag value should cause an error since decode can't tell which representation to use.
TEST(StructLayout, TaggedUnionRejectsATagPickedTwice) {
    EXPECT_FALSE(validate_tagged_union_cases<ShapeFields>(std::array{
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, dot, Shape::DOT),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, box, Shape::BOX),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::DOT),
    }));
}

// A tag value that no member lists should cause an error since decode would have no member to read when the tag holds
// it. This is also what fails when an enumerator is added to the tag enum, so whoever adds it decides what it picks.
TEST(StructLayout, TaggedUnionRejectsATagNothingPicks) {
    EXPECT_FALSE(validate_tagged_union_cases<ShapeFields>(std::array{
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, dot, Shape::DOT),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        // not listing BOX
    }));
}

// A member listed at a nonzero offset should cause an error since every member of a union starts at its first byte.
TEST(StructLayout, TaggedUnionRejectsAMemberNotAtOffsetZero) {
    EXPECT_FALSE(validate_tagged_union_cases<ShapeFields>(std::array{
        TaggedUnionCase{Member{"dot", 2, type_of<Inner>(), initializers_of<Inner>()}, "DOT"},
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, box, Shape::BOX),
    }));
}

// A tag value that names no enumerator of the tag enum should cause an error since the tag can never hold it, so the
// member it should pick would never be read. That covers both a value outside the enum and a misspelled name.
TEST(StructLayout, TaggedUnionRejectsATagValueThatIsNotAnEnumerator) {
    EXPECT_FALSE(validate_tagged_union_cases<ShapeFields>(std::array{
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, dot, Shape::DOT),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, box, Shape::BOX),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, box, static_cast<Shape>(7)),
    }));
    EXPECT_FALSE(validate_tagged_union_cases<ShapeFields>(std::array{
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, dot, Shape::DOT),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, line, Shape::LINE),
        LAYOUT_TAGGED_UNION_CASE(ShapeFields, box, Shape::BOX),
        TaggedUnionCase{LAYOUT_MEMBER(ShapeFields, box), "SQUARE"},
    }));
}

// A tagged union member should be a TaggedUnion element that names both the union and the enclosing struct's member
// holding its tag, since decode reads the tag first to pick which of the union's members to read.
TEST(StructLayout, TaggedUnionNamesItsTag) {
    constexpr Member fields = LAYOUT_TAGGED_UNION(Tagged, fields, shape);
    EXPECT_EQ(fields.name, "fields");
    EXPECT_EQ(fields.offset, offsetof(Tagged, fields));
    EXPECT_TRUE(
        fields.type ==
        (Type{element::TaggedUnion{&tagged_union_description<ShapeFields>, "shape"}, sizeof(ShapeFields), 0}));
}

// A tagged union whose TaggedUnionLayout sets a name, such as a union class template instance, should be known by
// that name wherever it's a member, since its type name isn't available.
TEST(StructLayout, TaggedUnionLayoutNameNamesTheUnion) {
    static_assert(validate_tagged_union_cases<Words<1>>(TaggedUnionLayout<Words<1>>::cases));
    constexpr Member value = LAYOUT_TAGGED_UNION(HoldsWords, value, width);
    EXPECT_EQ(tagged_union_description<Words<1>>.name, "Words<1>");
    EXPECT_TRUE(
        value.type == (Type{element::TaggedUnion{&tagged_union_description<Words<1>>, "width"}, sizeof(Words<1>), 0}));
}

// A struct's description should view its StructLayout's members, so following a Struct element reaches everything
// decode needs to read the struct: its name, size and members.
TEST(StructLayout, StructDescriptionViewsTheLayout) {
    constexpr const StructDescription& inner = struct_description<Inner>;
    EXPECT_EQ(inner.name, "Inner");
    EXPECT_EQ(inner.size, sizeof(Inner));
    EXPECT_EQ(inner.members.data(), StructLayout<Inner>::members.data());
    EXPECT_EQ(inner.members.size(), StructLayout<Inner>::members.size());
}

// An enum's description should list every enumerator enchantum finds, with its value, so decode can name each value
// the enum holds.
TEST(StructLayout, EnumDescriptionListsEveryEnumerator) {
    constexpr const EnumDescription& color = enum_description<Color>;
    EXPECT_EQ(color.name, "Color");
    ASSERT_EQ(color.enumerators.size(), 2u);
    EXPECT_EQ(color.enumerators[0].name, "RED");
    EXPECT_EQ(color.enumerators[0].value, 0u);
    EXPECT_EQ(color.enumerators[1].name, "GREEN");
    EXPECT_EQ(color.enumerators[1].value, 1u);
}

// An enum with an EnumLayout should be described by its listed enumerators, since enchantum can't find values outside
// the range it scans.
TEST(StructLayout, EnumLayoutListsTheEnumerators) {
    constexpr const EnumDescription& description = enum_description<ContainsOutsideRangeEnumerator>;
    EXPECT_EQ(description.name, "ContainsOutsideRangeEnumerator");
    ASSERT_EQ(description.enumerators.size(), 2u);
    EXPECT_EQ(description.enumerators[0].name, "NEAR");
    EXPECT_EQ(description.enumerators[0].value, 1u);
    EXPECT_EQ(description.enumerators[1].name, "AWAY");
    EXPECT_EQ(description.enumerators[1].value, 0xA0B0C0D0u);
}

// A tagged union's description should point to its tag enum's description and view its cases, so decode can read the
// tag and pick the case it names.
TEST(StructLayout, TaggedUnionDescriptionNamesItsTagEnum) {
    constexpr const TaggedUnionDescription& shape_fields = tagged_union_description<ShapeFields>;
    EXPECT_EQ(shape_fields.name, "ShapeFields");
    EXPECT_EQ(shape_fields.size, sizeof(ShapeFields));
    EXPECT_EQ(shape_fields.tag, &enum_description<Shape>);
    EXPECT_EQ(shape_fields.cases.data(), TaggedUnionLayout<ShapeFields>::cases.data());
}

// A union whose every setting value picks exactly one member should pass, since the manifest can then write the member
// the run picks whatever the setting holds.
TEST(StructLayout, RunSettingUnionAcceptsEveryValuePickingOneMember) {
    EXPECT_TRUE(validate_run_setting_union_cases<Mode>(StructLayout<Table>::value_cases));
}

// A setting value that no member lists should cause an error, since a run with that value would have no member to
// write. This is also what fails when an enumerator is added to the setting, so whoever adds it decides what it picks.
TEST(StructLayout, RunSettingUnionRejectsAValueNothingPicks) {
    EXPECT_FALSE(validate_run_setting_union_cases<Mode>(std::array{
        LAYOUT_RUN_SETTING_UNION_CASE(Table, narrow, Mode::NARROW),
        // not listing WIDE
    }));
}

// Two members listing the same setting value should cause an error, since the manifest couldn't tell which to write.
TEST(StructLayout, RunSettingUnionRejectsAValuePickedTwice) {
    EXPECT_FALSE(validate_run_setting_union_cases<Mode>(std::array{
        LAYOUT_RUN_SETTING_UNION_CASE(Table, narrow, Mode::NARROW),
        LAYOUT_RUN_SETTING_UNION_CASE(Table, wide, Mode::WIDE),
        LAYOUT_RUN_SETTING_UNION_CASE(Table, wide, Mode::NARROW),
    }));
}

// A case that doesn't start where the others do should cause an error, since every member of a union starts at its
// first byte, so the case can't be the union's.
TEST(StructLayout, RunSettingUnionRejectsACaseAtAnotherOffset) {
    EXPECT_FALSE(validate_run_setting_union_cases<Mode>(std::array{
        LAYOUT_RUN_SETTING_UNION_CASE(Table, narrow, Mode::NARROW),
        RunSettingUnionCase{LAYOUT_MEMBER(Table, trailer), "WIDE"},  // trailer starts at a different offset
    }));
}

// A setting value that names no enumerator of the setting should cause an error, since the setting can never hold it.
TEST(StructLayout, RunSettingUnionRejectsAValueThatIsNotAnEnumerator) {
    EXPECT_FALSE(validate_run_setting_union_cases<Mode>(std::array{
        LAYOUT_RUN_SETTING_UNION_CASE(Table, narrow, Mode::NARROW),
        LAYOUT_RUN_SETTING_UNION_CASE(Table, wide, Mode::WIDE),
        RunSettingUnionCase{LAYOUT_MEMBER(Table, wide), "WIDEST"}, // widest is not an enumerator of Mode
    }));
}

// A run-setting union's description should be as big as its widest case, point to its setting enum's description and
// view its cases, and the enclosing struct's member for it should have no name, start where its cases do and point to
// the description, so the manifest can write the case the run picks.
TEST(StructLayout, RunSettingUnionMemberViewsItsDescription) {
    constexpr const RunSettingUnionDescription& value =
        get_run_setting_union_description<Mode, StructLayout<Table>::value_cases>;
    EXPECT_EQ(value.size, sizeof(Table::wide));
    EXPECT_EQ(value.setting, &enum_description<Mode>);
    EXPECT_EQ(value.cases.data(), StructLayout<Table>::value_cases.data());

    const Member& member = StructLayout<Table>::members[1];
    EXPECT_EQ(member.name, "");
    EXPECT_EQ(member.offset, offsetof(Table, narrow));
    EXPECT_TRUE(member.type == (Type{element::RunSettingUnion{&value}, sizeof(Table::wide), 0}));
}

}  // namespace
}  // namespace tt::tt_fabric::layout
