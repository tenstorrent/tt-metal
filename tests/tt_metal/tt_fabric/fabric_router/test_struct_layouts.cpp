// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Verifies the struct layout helper (the coverage check, the tagged union check, how member types are classified and
// named, and the examples in its comments), the go_msg_t and launch_msg_t layouts built from the HAL, the packet header
// layouts and their send-type unions, the routing table's layout in each routing mode, and the hand-listed EDMStatus
// enumerators.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <llrt/hal.hpp>
#include <umd/device/types/arch.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

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

struct WordAndByte {
    uint32_t word;
    uint8_t byte;
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

}  // namespace tt::tt_fabric::layout_test

// The types the examples in struct_layout.hpp's comments describe.
namespace tt::tt_fabric::layout_example {

enum class Color : uint8_t { RED, GREEN, BLUE };

struct Point {
    uint16_t x;
    uint16_t y;
};

struct Pixel {
    Point at;
    Color color;
    int8_t brightness;
    uint16_t counts[2];
    uint8_t history;
    uint8_t reserved;
};

enum class Kind : uint32_t { DOT, LINE };

union Geometry {
    Point dot;
    uint32_t line[2];
};

struct Shape {
    Kind kind;
    Geometry geometry;
};

}  // namespace tt::tt_fabric::layout_example

namespace tt::tt_fabric::layout {

template <>
struct StructLayout<layout_test::Inner> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_test::Inner, x),
        LAYOUT_MEMBER(layout_test::Inner, y),
    };
};

template <>
struct StructLayout<layout_example::Point> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_example::Point, x),
        LAYOUT_MEMBER(layout_example::Point, y),
    };
};

template <>
struct StructLayout<layout_example::Pixel> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_example::Pixel, at),
        LAYOUT_MEMBER(layout_example::Pixel, color),
        LAYOUT_MEMBER(layout_example::Pixel, brightness),
        LAYOUT_MEMBER(layout_example::Pixel, counts),
        LAYOUT_PACKED_BITS(layout_example::Pixel, history, layout_example::Color, 2),
        LAYOUT_PAD(layout_example::Pixel, reserved),
    };
};

template <>
struct UnionLayout<layout_example::Geometry> {
    using Tag = layout_example::Kind;
    static constexpr std::array members = {
        LAYOUT_UNION_MEMBER(layout_example::Geometry, dot, Tag::DOT),
        LAYOUT_UNION_MEMBER(layout_example::Geometry, line, Tag::LINE),
    };
};

template <>
struct StructLayout<layout_example::Shape> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(layout_example::Shape, kind),
        LAYOUT_TAGGED_UNION(layout_example::Shape, geometry, kind),
    };
};

template <>
struct UnionLayout<layout_test::ShapeFields> {
    using Tag = layout_test::Shape;
    static constexpr std::array members = {
        LAYOUT_UNION_MEMBER(layout_test::ShapeFields, dot, Tag::DOT),
        LAYOUT_UNION_MEMBER(layout_test::ShapeFields, line, Tag::LINE),
        LAYOUT_UNION_MEMBER(layout_test::ShapeFields, box, Tag::BOX),
    };
};

namespace {

using layout_test::Color;
using layout_test::HalvedWord;
using layout_test::Inner;
using layout_test::Outer;
using layout_test::Shape;
using layout_test::ShapeFields;
using layout_test::Tagged;
using layout_test::ThreeWords;
using layout_test::WithArrays;
using layout_test::WithTable;
using layout_test::WordAndByte;

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
        LAYOUT_PACKED(WithTable, table, Shape),
        LAYOUT_MEMBER(WithTable, tail),
    };
    static_assert(validate_struct_members<WithTable>(members));

    const Element expected = element::Packed{"Shape", 3};
    EXPECT_TRUE(members[1].type.element == expected);
    EXPECT_EQ(members[1].offset, 4u);
    EXPECT_EQ(members[1].type.size, 3u);
    EXPECT_EQ(members[1].type.count, 8u);
}

// An integer member packs at the width it is given, which every value of the entry enum must fit.
TEST(StructLayout, PackedBitsTakesItsWidth) {
    constexpr Member packed = LAYOUT_PACKED_BITS(WordAndByte, byte, Shape, 2);
    EXPECT_EQ(packed.offset, 4u);
    EXPECT_TRUE(packed.type == (Type{element::Packed{"Shape", 2}, 1, 4}));
    static_assert(detail::enumerators_fit<Shape>(2));
    static_assert(!detail::enumerators_fit<Shape>(1));
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

TEST(StructLayout, TailPadCoversTheBytesAfterTheLastMember) {
    constexpr std::array members = {
        LAYOUT_MEMBER(WordAndByte, word),
        LAYOUT_MEMBER(WordAndByte, byte),
        LAYOUT_TAIL_PAD(WordAndByte, byte),
    };
    static_assert(validate_struct_members<WordAndByte>(members));
    EXPECT_EQ(members[2].offset, 5u);
    EXPECT_TRUE(members[2].type == (Type{element::Pad{}, 3, 0}));
}

TEST(StructLayout, UnionAcceptsEveryTagPickingOneMember) {
    EXPECT_TRUE(validate_union_members<ShapeFields>(UnionLayout<ShapeFields>::members));
}

TEST(StructLayout, UnionDefaultPicksTheTagsNoMemberLists) {
    EXPECT_TRUE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, line),
        LAYOUT_UNION_MEMBER(ShapeFields, dot, Shape::DOT),
    }));
}

TEST(StructLayout, UnionRejectsATagPickedTwice) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, line),
        LAYOUT_UNION_MEMBER(ShapeFields, dot, Shape::DOT),
        LAYOUT_UNION_MEMBER(ShapeFields, box, Shape::DOT),
    }));
}

TEST(StructLayout, UnionRejectsATagNothingPicks) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_MEMBER(ShapeFields, dot, Shape::DOT),
        LAYOUT_UNION_MEMBER(ShapeFields, line, Shape::LINE),
    }));
}

TEST(StructLayout, UnionRejectsTwoDefaults) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, dot),
        LAYOUT_UNION_DEFAULT(ShapeFields, line),
        LAYOUT_UNION_MEMBER(ShapeFields, box, Shape::BOX),
    }));
}

TEST(StructLayout, UnionRejectsWhenNoMemberFillsIt) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, dot),
        LAYOUT_UNION_MEMBER(ShapeFields, box, Shape::BOX),
    }));
}

TEST(StructLayout, UnionRejectsAMemberNotAtOffsetZero) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        UnionMember{Member{"dot", 2, type_of<Inner>()}, "DOT"},
        LAYOUT_UNION_MEMBER(ShapeFields, line, Shape::LINE),
        LAYOUT_UNION_MEMBER(ShapeFields, box, Shape::BOX),
    }));
}

TEST(StructLayout, UnionRejectsATagValueThatIsNotAnEnumerator) {
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, line),
        LAYOUT_UNION_MEMBER(ShapeFields, dot, Shape::DOT),
        LAYOUT_UNION_MEMBER(ShapeFields, box, static_cast<Shape>(7)),
    }));
    EXPECT_FALSE(validate_union_members<ShapeFields>(std::array{
        LAYOUT_UNION_DEFAULT(ShapeFields, line),
        LAYOUT_UNION_MEMBER(ShapeFields, dot, Shape::DOT),
        UnionMember{LAYOUT_MEMBER(ShapeFields, box), "SQUARE"},
    }));
}

TEST(StructLayout, TaggedUnionNamesItsTag) {
    constexpr Member fields = LAYOUT_TAGGED_UNION(Tagged, fields, shape);
    EXPECT_EQ(fields.name, "fields");
    EXPECT_EQ(fields.offset, offsetof(Tagged, fields));
    EXPECT_TRUE(fields.type == (Type{element::Union{"ShapeFields", "shape"}, sizeof(ShapeFields), 0}));
}

TEST(StructLayout, UnionTypeNamesTagValues) {
    const UnionType type = union_type<ShapeFields>();
    EXPECT_EQ(type.name, "ShapeFields");
    EXPECT_EQ(type.size, sizeof(ShapeFields));
    EXPECT_EQ(type.tag_enum, "Shape");
    ASSERT_EQ(type.members.size(), 3u);
    EXPECT_EQ(type.members[0].member.name, "dot");
    EXPECT_EQ(type.members[0].tag_value.value_or(""), "DOT");
    EXPECT_EQ(type.members[2].member.name, "box");
    EXPECT_EQ(type.members[2].tag_value.value_or(""), "BOX");
}

bool is_member(const Member& member, std::string_view name, uint32_t offset, const Type& type) {
    return member.name == name && member.offset == offset && member.type == type;
}

// Every value the examples in struct_layout.hpp's comments state.
TEST(StructLayout, HeaderExamples) {
    namespace ex = layout_example;
    static_assert(sizeof(ex::Pixel) == 12);
    static_assert(sizeof(ex::Geometry) == 8);

    constexpr auto& pixel = StructLayout<ex::Pixel>::members;
    static_assert(validate_struct_members<ex::Pixel>(pixel));
    EXPECT_TRUE((is_member(pixel[0], "at", 0, {element::Struct{"Point"}, 4, 0})));
    EXPECT_TRUE((is_member(pixel[1], "color", 4, {element::Enum{"Color"}, 1, 0})));
    EXPECT_TRUE((is_member(pixel[2], "brightness", 5, {element::Int{}, 1, 0})));
    EXPECT_TRUE((is_member(pixel[3], "counts", 6, {element::Uint{}, 4, 2})));
    EXPECT_TRUE((is_member(pixel[4], "history", 10, {element::Packed{"Color", 2}, 1, 4})));
    EXPECT_TRUE((is_member(pixel[5], "reserved", 11, {element::Pad{}, 1, 0})));
    EXPECT_TRUE((is_member(LAYOUT_BYTES(ex::Pixel, history), "history", 10, {element::Bytes{}, 1, 0})));
    EXPECT_TRUE((is_member(LAYOUT_TAIL_PAD(WordAndByte, byte), "tail_padding", 5, {element::Pad{}, 3, 0})));

    EXPECT_FALSE(
        validate_struct_members<ex::Pixel>(std::array{pixel[1], pixel[0], pixel[2], pixel[3], pixel[4], pixel[5]}));
    EXPECT_FALSE(validate_struct_members<ex::Pixel>(std::array{pixel[0], pixel[1], pixel[2], pixel[3], pixel[4]}));
    const std::array head = {pixel[0], pixel[1], pixel[2], pixel[3], pixel[4]};
    const auto joined = concat_members(head, std::array{LAYOUT_PAD(ex::Pixel, reserved)});
    EXPECT_TRUE(validate_struct_members<ex::Pixel>(joined));

    EXPECT_TRUE(type_of<ex::Color>() == (Type{element::Enum{"Color"}, 1, 0}));
    EXPECT_TRUE(array_of<ex::Point>(3) == (Type{element::Struct{"Point"}, 12, 3}));

    const StructType point = struct_type<ex::Point>();
    EXPECT_EQ(point.name, "Point");
    EXPECT_EQ(point.size, 4u);
    ASSERT_EQ(point.members.size(), 2u);
    EXPECT_TRUE((is_member(point.members[0], "x", 0, {element::Uint{}, 2, 0})));
    EXPECT_TRUE((is_member(point.members[1], "y", 2, {element::Uint{}, 2, 0})));

    constexpr auto& geometry_members = UnionLayout<ex::Geometry>::members;
    static_assert(validate_union_members<ex::Geometry>(geometry_members));
    const UnionType geometry = union_type<ex::Geometry>();
    EXPECT_EQ(geometry.name, "Geometry");
    EXPECT_EQ(geometry.size, 8u);
    EXPECT_EQ(geometry.tag_enum, "Kind");
    ASSERT_EQ(geometry.members.size(), 2u);
    EXPECT_TRUE((is_member(geometry.members[0].member, "dot", 0, {element::Struct{"Point"}, 4, 0})));
    EXPECT_EQ(geometry.members[0].tag_value.value_or(""), "DOT");
    EXPECT_EQ(geometry.members[1].member.name, "line");
    EXPECT_EQ(geometry.members[1].tag_value.value_or(""), "LINE");
    EXPECT_FALSE(validate_union_members<ex::Geometry>(std::array{geometry_members[0]}));
    EXPECT_TRUE(validate_union_members<ex::Geometry>(
        std::array{geometry_members[0], LAYOUT_UNION_DEFAULT(ex::Geometry, line)}));

    constexpr auto& shape = StructLayout<ex::Shape>::members;
    static_assert(validate_struct_members<ex::Shape>(shape));
    EXPECT_TRUE((is_member(shape[1], "geometry", 4, {element::Union{"Geometry", "kind"}, 8, 0})));
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

const StructType* find_type(const std::vector<StructType>& types, std::string_view name) {
    const auto it = std::ranges::find(types, name, &StructType::name);
    return it == types.end() ? nullptr : &*it;
}

const Member* find_member(const StructType& type, std::string_view name) {
    const auto it = std::ranges::find(type.members, name, &Member::name);
    return it == type.members.end() ? nullptr : &*it;
}

// launch_msg_t's layout is built at run time by walking the HAL's fields, so this is where a change to it fails, for
// each arch, with no device. Values written through the HAL's accessors land where the layout puts their members.
TEST(StructLayout, LaunchMsgLayoutFromHal) {
    namespace dev_msgs = tt::tt_metal::dev_msgs;
    for (const auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE}) {
        SCOPED_TRACE(tt::arch_to_str(arch));
        const tt::tt_metal::Hal hal(arch, false, false, 0, false);
        const auto& factory = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        const auto types = hal_struct_types(hal);
        const StructType* launch = find_type(types, "launch_msg_t");
        const StructType* config = find_type(types, "kernel_config_msg_t");
        const StructType* rta = find_type(types, "rta_offset_t");
        ASSERT_NE(launch, nullptr);
        ASSERT_NE(config, nullptr);
        ASSERT_NE(rta, nullptr);
        EXPECT_EQ(launch->size, factory.size_of<dev_msgs::launch_msg_t>());
        for (const StructType* type : {launch, config, rta}) {
            EXPECT_TRUE(validate_members(type->members, type->size)) << type->name;
        }

        const Member* kernel_config = find_member(*launch, "kernel_config");
        const Member* exit_erisc_kernel = find_member(*config, "exit_erisc_kernel");
        const Member* host_assigned_id = find_member(*config, "host_assigned_id");
        const Member* rta_offsets = find_member(*config, "rta_offset");
        const Member* crta_offset = find_member(*rta, "crta_offset");
        for (const Member* member : {kernel_config, exit_erisc_kernel, host_assigned_id, rta_offsets, crta_offset}) {
            ASSERT_NE(member, nullptr);
        }
        EXPECT_TRUE(kernel_config->type.element == Element{element::Struct{"kernel_config_msg_t"}});
        EXPECT_TRUE(rta_offsets->type.element == Element{element::Struct{"rta_offset_t"}});
        ASSERT_GT(rta_offsets->type.count, 0u);
        const uint32_t last_rta = rta_offsets->type.count - 1;

        auto message = factory.create<dev_msgs::launch_msg_t>();
        auto config_view = message.view().kernel_config();
        config_view.exit_erisc_kernel() = 0xAB;
        config_view.host_assigned_id() = 0x12345678;
        config_view.rta_offset()[last_rta].crta_offset() = 0x5A6B;

        const auto read = [&](uint32_t offset, size_t bytes) {
            uint32_t value = 0;
            std::memcpy(&value, message.data() + offset, bytes);
            return value;
        };
        const uint32_t base = kernel_config->offset;
        EXPECT_EQ(read(base + exit_erisc_kernel->offset, 1), 0xABu);
        EXPECT_EQ(read(base + host_assigned_id->offset, 4), 0x12345678u);
        const uint32_t rta_size = rta_offsets->type.size / rta_offsets->type.count;
        EXPECT_EQ(read(base + rta_offsets->offset + (last_rta * rta_size) + crta_offset->offset, 2), 0x5A6Bu);
    }
}

const UnionType* find_union(const std::vector<UnionType>& unions, std::string_view name) {
    const auto it = std::ranges::find(unions, name, &UnionType::name);
    return it == unions.end() ? nullptr : &*it;
}

// For every header FabricContext can choose, the manifest names the header the visitor picks, and every struct, union
// and enum its layouts name is described: by those layouts, or by DescribedStructs, DescribedUnions and DescribedEnums.
TEST(PacketHeaderLayouts, EveryConfig) {
    const auto described = described_struct_types();
    const auto unions = described_union_types();
    std::set<std::string_view> enums;
    [&]<typename... Es>(std::type_identity<std::tuple<Es...>>) {
        (enums.insert(enum_type<Es>().name), ...);
    }(std::type_identity<DescribedEnums>{});

    for (const auto& config : PACKET_HEADER_CONFIGS) {
        const Type header = packet_header_type(config);
        const auto* header_struct = std::get_if<element::Struct>(&header.element);
        ASSERT_NE(header_struct, nullptr);
        SCOPED_TRACE(header_struct->name);
        const size_t size =
            visit_packet_header_type(config, [](auto type) { return sizeof(typename decltype(type)::type); });
        EXPECT_EQ(header.size, size);

        const auto types = packet_header_struct_types(config);
        const StructType* header_type = find_type(types, header_struct->name);
        ASSERT_NE(header_type, nullptr);
        EXPECT_EQ(header_type->size, size);
        for (const auto& type : types) {
            EXPECT_EQ(find_type(described, type.name), nullptr) << type.name << " is also in DescribedStructs";
            for (const auto& member : type.members) {
                if (const auto* s = std::get_if<element::Struct>(&member.type.element)) {
                    EXPECT_TRUE(find_type(types, s->name) != nullptr || find_type(described, s->name) != nullptr)
                        << type.name << "." << member.name << " names " << s->name;
                } else if (const auto* e = std::get_if<element::Enum>(&member.type.element)) {
                    EXPECT_TRUE(enums.contains(e->name)) << type.name << "." << member.name << " names " << e->name;
                } else if (const auto* u = std::get_if<element::Union>(&member.type.element)) {
                    EXPECT_NE(find_union(unions, u->name), nullptr)
                        << type.name << "." << member.name << " names " << u->name;
                    EXPECT_EQ(u->tag, "noc_send_type") << type.name << "." << member.name;
                }
            }
        }
    }
}

// The send-type unions as the manifest writes them: each member's tag value is the enumerator's own name (enchantum
// names NOC_SEND_TYPE_LAST's value NOC_SPARSE_MCAST_WRITE), and each member's struct is in DescribedStructs.
TEST(PacketHeaderLayouts, SendTypeUnions) {
    const auto described = described_struct_types();
    const auto unions = described_union_types();
    const auto tag_of = [](const UnionType& type, std::string_view member) -> std::optional<std::string_view> {
        const auto it = std::ranges::find(type.members, member, [](const auto& m) { return m.member.name; });
        EXPECT_NE(it, type.members.end()) << type.name << "." << member;
        return it == type.members.end() ? std::nullopt : it->tag_value;
    };

    const UnionType* command = find_union(unions, "NocCommandFields");
    ASSERT_NE(command, nullptr);
    EXPECT_EQ(command->size, sizeof(NocCommandFields));
    EXPECT_EQ(command->tag_enum, "NocSendType");
    EXPECT_EQ(command->members.size(), enchantum::count<NocSendType>);
    EXPECT_EQ(tag_of(*command, "unicast_write").value_or("none"), "NOC_UNICAST_WRITE");
    EXPECT_EQ(tag_of(*command, "unicast_seminc_fused").value_or("none"), "NOC_FUSED_UNICAST_ATOMIC_INC");
    EXPECT_EQ(tag_of(*command, "sparse_mcast_write").value_or("none"), "NOC_SPARSE_MCAST_WRITE");

    const UnionType* udm = find_union(unions, "UDMControlFields");
    ASSERT_NE(udm, nullptr);
    EXPECT_EQ(udm->size, sizeof(UDMControlFields));
    EXPECT_EQ(udm->tag_enum, "NocSendType");
    EXPECT_FALSE(tag_of(*udm, "write").has_value());
    EXPECT_EQ(tag_of(*udm, "read").value_or("none"), "NOC_UNICAST_READ");

    for (const UnionType* type : {command, udm}) {
        for (const auto& entry : type->members) {
            const Member& member = entry.member;
            const auto* s = std::get_if<element::Struct>(&member.type.element);
            ASSERT_NE(s, nullptr) << type->name << "." << member.name;
            EXPECT_NE(find_type(described, s->name), nullptr)
                << type->name << "." << member.name << " names " << s->name;
        }
    }
}

// A scatter write's chunk_encoding holds one NocScatterWriteChunkEncoding per chunk.
TEST(PacketHeaderLayouts, ScatterChunkEncodingIsPacked) {
    const Member& chunk_encoding = StructLayout<NocUnicastScatterCommandHeader>::members.back();
    const Type expected{
        element::Packed{"NocScatterWriteChunkEncoding", NOC_SCATTER_CHUNK_ENCODING_BITS},
        1,
        NOC_SCATTER_WRITE_MAX_CHUNKS};
    EXPECT_EQ(chunk_encoding.name, "chunk_encoding");
    EXPECT_TRUE(chunk_encoding.type == expected);
}

// In each routing mode, routing_l1_info_t's members cover it, its union holds that mode's table, only 2D reads
// exit_node_table, and every struct and enum the layouts name is described: by those layouts, or by DescribedStructs
// and DescribedEnums.
TEST(StructLayout, RoutingTableInEachMode) {
    const auto described = described_struct_types();
    std::set<std::string_view> enums;
    [&]<typename... Es>(std::type_identity<std::tuple<Es...>>) {
        (enums.insert(enum_type<Es>().name), ...);
    }(std::type_identity<DescribedEnums>{});

    for (const bool is_2d : {false, true}) {
        SCOPED_TRACE(is_2d ? "2D" : "1D");
        const auto types = routing_table_struct_types(is_2d);
        const StructType* routing = find_type(types, "routing_l1_info_t");
        ASSERT_NE(routing, nullptr);
        EXPECT_EQ(routing->size, sizeof(routing_l1_info_t));
        EXPECT_TRUE(validate_members(routing->members, routing->size));

        const Member* table_1d = find_member(*routing, "routing_path_table_1d");
        const Member* table_2d = find_member(*routing, "route_table_2d");
        const Member* exit_nodes = find_member(*routing, "exit_node_table");
        const Member* direction_table = find_member(*routing, "intra_mesh_direction_table");
        ASSERT_NE(exit_nodes, nullptr);
        ASSERT_NE(direction_table, nullptr);
        EXPECT_EQ(table_1d != nullptr, !is_2d);
        EXPECT_EQ(table_2d != nullptr, is_2d);
        EXPECT_EQ(std::holds_alternative<element::Pad>(exit_nodes->type.element), !is_2d);
        EXPECT_TRUE(direction_table->type.element == (Element{element::Packed{"compressed_routing_values", 3}}));

        for (const auto& type : types) {
            EXPECT_EQ(find_type(described, type.name), nullptr) << type.name << " is also in DescribedStructs";
            EXPECT_TRUE(validate_members(type.members, type.size)) << type.name;
            for (const auto& member : type.members) {
                if (const auto* s = std::get_if<element::Struct>(&member.type.element)) {
                    EXPECT_TRUE(find_type(types, s->name) != nullptr || find_type(described, s->name) != nullptr)
                        << type.name << "." << member.name << " names " << s->name;
                } else if (const auto* e = std::get_if<element::Enum>(&member.type.element)) {
                    EXPECT_TRUE(enums.contains(e->name)) << type.name << "." << member.name << " names " << e->name;
                } else if (const auto* p = std::get_if<element::Packed>(&member.type.element)) {
                    EXPECT_TRUE(enums.contains(p->entry_enum))
                        << type.name << "." << member.name << " names " << p->entry_enum;
                }
            }
        }
    }
}

// EDMStatus is listed by hand, since enchantum cannot reflect it. Each listed name has its enumerator's value.
TEST(StructLayout, EdmStatusEnumerators) {
    const auto type = enum_type<EDMStatus>();
    EXPECT_EQ(type.name, "EDMStatus");
    const auto value_of = [&](std::string_view name) {
        const auto it = std::ranges::find(type.enumerators, name, &Enumerator::name);
        EXPECT_NE(it, type.enumerators.end()) << name;
        return it == type.enumerators.end() ? 0u : it->value;
    };
    EXPECT_EQ(value_of("STARTED"), static_cast<uint32_t>(EDMStatus::STARTED));
    EXPECT_EQ(value_of("READY_FOR_TRAFFIC"), static_cast<uint32_t>(EDMStatus::READY_FOR_TRAFFIC));
    EXPECT_EQ(value_of("INITIALIZATION_COMPLETE"), static_cast<uint32_t>(EDMStatus::INITIALIZATION_COMPLETE));
}

}  // namespace
}  // namespace tt::tt_fabric::layout
