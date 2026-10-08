// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Verifies the struct layout helper (the coverage check, and how member types are classified and named) and the
// go_msg_t layout built from the HAL.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string_view>

#include <llrt/hal.hpp>
#include <umd/device/types/arch.hpp>

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

struct WithArrays {
    uint8_t id;
    Color colors[3];
    std::array<Inner, 2> points;
};

}  // namespace tt::tt_fabric::layout_test

namespace tt::tt_fabric::layout {

template <>
struct StructLayout<layout_test::Inner> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_test::Inner, x),
        LAYOUT_MEMBER(layout_test::Inner, y),
    };
};

namespace {

using layout_test::Color;
using layout_test::HalvedWord;
using layout_test::Inner;
using layout_test::Outer;
using layout_test::ThreeWords;
using layout_test::WithArrays;
using layout_test::WithTable;

TEST(StructLayout, CoverageAcceptsEveryByteListedOnce) {
    EXPECT_TRUE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, b),
        LAYOUT_MEMBER(ThreeWords, c),
    }));
}

TEST(StructLayout, CoverageRejectsMembersOutOfOrder) {
    EXPECT_FALSE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, c),
        LAYOUT_MEMBER(ThreeWords, b),
    }));
}

TEST(StructLayout, CoverageRejectsAGap) {
    EXPECT_FALSE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, c),
    }));
}

TEST(StructLayout, CoverageRejectsAMissingTail) {
    EXPECT_FALSE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, b),
    }));
}

TEST(StructLayout, CoverageRejectsOverlappingUnionMembers) {
    EXPECT_FALSE(validate_struct_members<HalvedWord>(std::array{
        LAYOUT_MEMBER(HalvedWord, all),
        LAYOUT_MEMBER(HalvedWord, lo),
        LAYOUT_MEMBER(HalvedWord, hi),
    }));
    EXPECT_TRUE(validate_struct_members<HalvedWord>(std::array{LAYOUT_MEMBER(HalvedWord, all)}));
}

TEST(StructLayout, MemberTypesAndNames) {
    constexpr std::array members = {
        LAYOUT_MEMBER(Outer, inner),
        LAYOUT_MEMBER(Outer, color),
        LAYOUT_MEMBER(Outer, delta),
        LAYOUT_MEMBER(Outer, counts),
    };
    static_assert(validate_struct_members<Outer>(members));

    EXPECT_TRUE(members[0].type.element == Element{element::Struct{"Inner"}});
    EXPECT_TRUE(members[1].type.element == Element{element::Enum{"Color"}});
    EXPECT_TRUE(members[2].type.element == Element{element::Int{}});
    EXPECT_TRUE(members[3].type.element == Element{element::Uint{}});
    EXPECT_EQ(members[3].offset, 6u);
    EXPECT_EQ(members[3].type.size, 4u);
    EXPECT_EQ(members[3].type.count, 2u);
}

TEST(StructLayout, PackedTableCountsEntries) {
    constexpr std::array members = {
        LAYOUT_MEMBER(WithTable, id),
        LAYOUT_PACKED(WithTable, table, "three_bit"),
        LAYOUT_MEMBER(WithTable, tail),
    };
    static_assert(validate_struct_members<WithTable>(members));

    const Element expected = element::Packed{"three_bit", 3};
    EXPECT_TRUE(members[1].type.element == expected);
    EXPECT_EQ(members[1].offset, 4u);
    EXPECT_EQ(members[1].type.size, 3u);
    EXPECT_EQ(members[1].type.count, 8u);
}

TEST(StructLayout, TypeOfCArray) {
    EXPECT_TRUE(type_of<Color[3]>() == (Type{element::Enum{"Color"}, 3, 3}));

    constexpr std::array members = {
        LAYOUT_MEMBER(WithArrays, id),
        LAYOUT_MEMBER(WithArrays, colors),
        LAYOUT_MEMBER(WithArrays, points),
    };
    static_assert(validate_struct_members<WithArrays>(members));
    EXPECT_EQ(members[1].offset, 1u);
    EXPECT_TRUE(members[1].type == (Type{element::Enum{"Color"}, 3, 3}));
}

TEST(StructLayout, TypeOfStdArray) {
    constexpr Type type = type_of<std::array<Inner, 2>>();
    EXPECT_TRUE(type == (Type{element::Struct{"Inner"}, 8, 2}));
    EXPECT_TRUE(type == type_of<Inner[2]>());

    constexpr Member points = LAYOUT_MEMBER(WithArrays, points);
    EXPECT_EQ(points.offset, 4u);
    EXPECT_TRUE(points.type == (Type{element::Struct{"Inner"}, 8, 2}));
}

TEST(StructLayout, ArrayOfMatchesAFixedLengthArray) {
    EXPECT_TRUE(array_of<Inner>(2) == type_of<Inner[2]>());
    EXPECT_TRUE(array_of<uint32_t>(5) == (Type{element::Uint{}, 20, 5}));
}

// go_msg_layout is built at run time, so this is where a change to go_msg_t fails, for each arch, with no device.
TEST(StructLayout, GoMsgLayoutFromHal) {
    for (const auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE}) {
        SCOPED_TRACE(tt::arch_to_str(arch));
        const tt::tt_metal::Hal hal(arch, false, false, 0, false);
        const auto members = go_msg_layout(hal);
        const auto size = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH)
                              .size_of<tt::tt_metal::dev_msgs::go_msg_t>();
        EXPECT_TRUE(validate_members(members, size));

        // Each member's offset is where make_go_msg_u32, which packs the message into one word, puts its value.
        ASSERT_EQ(size, sizeof(uint32_t));
        const auto bytes = std::bit_cast<std::array<uint8_t, 4>>(hal.make_go_msg_u32(0xAB, 0x01, 0x02, 0x03));
        const auto byte_at = [&](std::string_view name) {
            const auto it = std::find_if(members.begin(), members.end(), [&](const auto& m) { return m.name == name; });
            EXPECT_NE(it, members.end()) << name;
            return it == members.end() ? 0 : bytes.at(it->offset);
        };
        EXPECT_EQ(byte_at("signal"), 0xAB);
        EXPECT_EQ(byte_at("master_x"), 0x01);
        EXPECT_EQ(byte_at("master_y"), 0x02);
        EXPECT_EQ(byte_at("dispatch_message_offset"), 0x03);
    }
}

}  // namespace
}  // namespace tt::tt_fabric::layout
