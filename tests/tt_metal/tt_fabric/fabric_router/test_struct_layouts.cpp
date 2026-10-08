// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Verifies the struct layout helper (the coverage check, and how member types are classified and named), the
// go_msg_t and launch_msg_t layouts built from the HAL, and the hand-listed EDMStatus enumerators.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

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
