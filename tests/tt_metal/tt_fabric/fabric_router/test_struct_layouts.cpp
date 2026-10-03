// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Verifies the struct layout helper: the coverage check, how member types are classified and named, and the
// field lists of the described fabric structs.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>

#include "tt_metal/fabric/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/fabric/manifest/struct_layout.hpp"

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

struct ThreeBitTable {
    static constexpr uint32_t BITS_PER_COMPRESSED_ENTRY = 3;
    uint8_t packed[3];
};

struct WithTable {
    uint32_t id;
    ThreeBitTable table;
    uint8_t tail;
};

}  // namespace tt::tt_fabric::layout_test

namespace tt::tt_fabric {

template <>
struct StructLayout<layout_test::Inner> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(layout_test::Inner, x),
        LAYOUT_FIELD(layout_test::Inner, y),
    };
};

namespace {

using layout_test::HalvedWord;
using layout_test::Outer;
using layout_test::ThreeWords;
using layout_test::WithTable;

TEST(StructLayout, CoverageAcceptsEveryByteListedOnce) {
    EXPECT_TRUE(validate_struct_fields<ThreeWords>(std::array{
        LAYOUT_FIELD(ThreeWords, a),
        LAYOUT_FIELD(ThreeWords, b),
        LAYOUT_FIELD(ThreeWords, c),
    }));
}

TEST(StructLayout, CoverageRejectsFieldsOutOfOrder) {
    EXPECT_FALSE(validate_struct_fields<ThreeWords>(std::array{
        LAYOUT_FIELD(ThreeWords, a),
        LAYOUT_FIELD(ThreeWords, c),
        LAYOUT_FIELD(ThreeWords, b),
    }));
}

TEST(StructLayout, CoverageRejectsAGap) {
    EXPECT_FALSE(validate_struct_fields<ThreeWords>(std::array{
        LAYOUT_FIELD(ThreeWords, a),
        LAYOUT_FIELD(ThreeWords, c),
    }));
}

TEST(StructLayout, CoverageRejectsAMissingTail) {
    EXPECT_FALSE(validate_struct_fields<ThreeWords>(std::array{
        LAYOUT_FIELD(ThreeWords, a),
        LAYOUT_FIELD(ThreeWords, b),
    }));
}

TEST(StructLayout, CoverageRejectsOverlappingUnionMembers) {
    EXPECT_FALSE(validate_struct_fields<HalvedWord>(std::array{
        LAYOUT_FIELD(HalvedWord, all),
        LAYOUT_FIELD(HalvedWord, lo),
        LAYOUT_FIELD(HalvedWord, hi),
    }));
    EXPECT_TRUE(validate_struct_fields<HalvedWord>(std::array{LAYOUT_FIELD(HalvedWord, all)}));
}

TEST(StructLayout, FieldTypesAndNames) {
    constexpr std::array fields = {
        LAYOUT_FIELD(Outer, inner),
        LAYOUT_FIELD(Outer, color),
        LAYOUT_FIELD(Outer, delta),
        LAYOUT_FIELD(Outer, counts),
    };
    static_assert(validate_struct_fields<Outer>(fields));

    EXPECT_TRUE(fields[0].type == FieldType{field::Struct{"Inner"}});
    EXPECT_TRUE(fields[1].type == FieldType{field::Enum{"Color"}});
    EXPECT_TRUE(fields[2].type == FieldType{field::Int{}});
    EXPECT_TRUE(fields[3].type == FieldType{field::Uint{}});
    EXPECT_EQ(fields[3].offset, 6u);
    EXPECT_EQ(fields[3].size, 4u);
    EXPECT_EQ(fields[3].intra_field_element_count, 2u);
}

TEST(StructLayout, PackedTableCountsEntries) {
    constexpr std::array fields = {
        LAYOUT_FIELD(WithTable, id),
        LAYOUT_PACKED(WithTable, table, "three_bit"),
        LAYOUT_FIELD(WithTable, tail),
    };
    static_assert(validate_struct_fields<WithTable>(fields));

    const FieldType expected = field::Packed{"three_bit", 3};
    EXPECT_TRUE(fields[1].type == expected);
    EXPECT_EQ(fields[1].offset, 4u);
    EXPECT_EQ(fields[1].size, 3u);
    EXPECT_EQ(fields[1].intra_field_element_count, 8u);
}

}  // namespace
}  // namespace tt::tt_fabric
