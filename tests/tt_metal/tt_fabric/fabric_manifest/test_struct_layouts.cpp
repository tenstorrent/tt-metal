// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Validates the behavior of the fabric manifest's struct layout helpers (struct_layout.hpp) where the real layouts'
// static_asserts can't:
// - The coverage check fails when it should. Each real layout's static_assert only shows that a correct member list
//   passes it.
// - Each member gets the right element kind, type name and count. The coverage check only compares offsets and sizes,
//   so a member described as the wrong kind still passes it, and decode would read it wrong.
//
// Layouts built at run time (i.e., go_msg_layout, which depends on the HAL) are also validated here, since no
// static_assert can check them.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <llrt/hal.hpp>
#include <umd/device/types/arch.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

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

// A member missing from the middle of the list, such as a field added to a device struct but not to its layout, leaves
// a gap which should cause an error since the next listed member doesn't start where the one before it ended.
TEST(StructLayout, CoverageRejectsAGap) {
    EXPECT_FALSE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, c),
    }));
}

// A member missing from the end of the list should cause an error since the last listed member doesn't end at the end
// of the actual struct.
TEST(StructLayout, CoverageRejectsAMissingTail) {
    EXPECT_FALSE(validate_struct_members<ThreeWords>(std::array{
        LAYOUT_MEMBER(ThreeWords, a),
        LAYOUT_MEMBER(ThreeWords, b),
    }));
}

// Listing more than one of a union's members should cause an error since their bytes overlap, so the same bytes are
// counted twice. Listing just one of them should pass since that one member covers the union's bytes.
TEST(StructLayout, CoverageRejectsOverlappingUnionMembers) {
    EXPECT_FALSE(validate_struct_members<HalvedWord>(std::array{
        LAYOUT_MEMBER(HalvedWord, all),
        LAYOUT_MEMBER(HalvedWord, lo),
        LAYOUT_MEMBER(HalvedWord, hi),
    }));
    EXPECT_TRUE(validate_struct_members<HalvedWord>(std::array{LAYOUT_MEMBER(HalvedWord, all)}));
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
    static_assert(validate_struct_members<Outer>(members));

    EXPECT_TRUE(members[0].type.element == Element{element::Struct{"Inner"}});
    EXPECT_TRUE(members[1].type.element == Element{element::Enum{"Color"}});
    EXPECT_TRUE(members[2].type.element == Element{element::Int{}});
    EXPECT_TRUE(members[3].type.element == Element{element::Uint{}});
    EXPECT_EQ(members[3].offset, 6u);
    EXPECT_EQ(members[3].type.size, 4u);
    EXPECT_EQ(members[3].type.count, 2u);
}

// A packed table holds entries smaller than a byte, so its count should be its number of entries rather than bytes
// since decode reads it entry by entry: 3 bytes of 3-bit entries should count as 8. Its element should carry the table
// name decode looks up and the entry width from the member type's BITS_PER_COMPRESSED_ENTRY.
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

// A C array of enums should be an Enum element with the array's length as its count, rather than a scalar, since
// decode reads that many elements. It should match both on its own and as a struct member.
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

// A std::array should get the same Type as a C array of the same element and length since both lay out their elements
// the same way, so decode reads them the same way. It should match both on its own and as a struct member.
TEST(StructLayout, TypeOfStdArray) {
    constexpr Type type = type_of<std::array<Inner, 2>>();
    EXPECT_TRUE(type == (Type{element::Struct{"Inner"}, 8, 2}));
    EXPECT_TRUE(type == type_of<Inner[2]>());

    constexpr Member points = LAYOUT_MEMBER(WithArrays, points);
    EXPECT_EQ(points.offset, 4u);
    EXPECT_TRUE(points.type == (Type{element::Struct{"Inner"}, 8, 2}));
}

// An array whose length is known only at run time is described with array_of, which should give the same Type as a C
// array of that length since its elements are laid out the same way.
TEST(StructLayout, ArrayOfMatchesAFixedLengthArray) {
    EXPECT_TRUE(array_of<Inner>(2) == type_of<Inner[2]>());
    EXPECT_TRUE(array_of<uint32_t>(5) == (Type{element::Uint{}, 20, 5}));
}

// go_msg_layout is built at run time from the HAL, so no static_assert can check it. For each arch, with no device, its
// members should cover go_msg_t, and a value written through the HAL's accessors should land at the offset the layout
// gives its member since that is where decode reads it.
TEST(StructLayout, GoMsgLayoutFromHal) {
    for (const auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE}) {
        SCOPED_TRACE(tt::arch_to_str(arch));
        const tt::tt_metal::Hal hal(arch, false, false, 0, false);
        const auto members = go_msg_layout(hal);
        const auto& factory = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        const auto size = factory.size_of<tt::tt_metal::dev_msgs::go_msg_t>();
        EXPECT_TRUE(validate_members(members, size));

        // Each member's offset is where the HAL's go_msg_t accessors put its value in the message's one word.
        ASSERT_EQ(size, sizeof(uint32_t));
        uint32_t word = 0;
        auto go_msg = factory.create_view<tt::tt_metal::dev_msgs::go_msg_t>(reinterpret_cast<std::byte*>(&word));
        go_msg.signal() = 0xAB;
        go_msg.master_x() = 0x01;
        go_msg.master_y() = 0x02;
        go_msg.dispatch_message_offset() = 0x03;
        const auto bytes = std::bit_cast<std::array<uint8_t, 4>>(word);
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
