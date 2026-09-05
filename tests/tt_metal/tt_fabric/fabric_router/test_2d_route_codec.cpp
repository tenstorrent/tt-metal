// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Machine-free action-map packing and decode coverage.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/routing_2d_table_builder.hpp"

namespace tt::tt_fabric::routing_2d_codec_tests {
namespace {

using Codec = Routing2DCodec;

struct Shape {
    const char* name;
    uint32_t y;
    uint32_t x;
};

// Representative geometries, including both maximum-layout orientations.
constexpr std::array<Shape, 4> kRepresentativeShapes = {{
    {"[8,4]", 8, 4},
    {"[1,16]", 1, 16},
    {"[64,4]", 64, 4},
    {"[4,64]", 4, 64},
}};

// Dimension-order oracle for a plain (chordless) mesh: rows increase southward, columns eastward.
eth_chan_directions dor_y(uint32_t cur, uint32_t dst) {
    return cur < dst ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
}
eth_chan_directions dor_x(uint32_t cur, uint32_t dst) {
    return cur < dst ? eth_chan_directions::EAST : eth_chan_directions::WEST;
}

constexpr uint32_t action_vector_bytes(uint32_t y_size, uint32_t x_size) {
    return Codec::table_bytes(y_size) + Codec::table_bytes(x_size);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Shape admissibility
// ---------------------------------------------------------------------------------------------

TEST(Routing2DCodec, ActionVectorFootprintMatchesThePackedLayout) {
    // Y table is y_size rows of ceil(y_size/4) bytes; X table likewise. Checked against hand
    // arithmetic so a change to the packing density cannot pass unnoticed.
    EXPECT_EQ(action_vector_bytes(8, 8), 8u * 2u + 8u * 2u);     // square
    EXPECT_EQ(action_vector_bytes(1, 16), 1u * 1u + 16u * 4u);   // narrow rectangle
    EXPECT_EQ(action_vector_bytes(64, 4), 64u * 16u + 4u * 1u);  // 1028
}

TEST(Routing2DCodec, ShapesBeyondTheAddressableRangeAreRejected) {
    EXPECT_FALSE(is_valid_2d_route_table_shape(0, 4));
    EXPECT_FALSE(is_valid_2d_route_table_shape(4, 0));
    EXPECT_FALSE(is_valid_2d_route_table_shape(Codec::MAX_AXIS_SIZE + 1, 4));
    EXPECT_FALSE(is_valid_2d_route_table_shape(4, Codec::MAX_AXIS_SIZE + 1));
    // 64x8 is within the per-axis range, but has 512 chips and needs 1040 vector bytes.
    EXPECT_FALSE(is_valid_2d_route_table_shape(64, 8));
    EXPECT_FALSE(is_valid_2d_route_table_shape(64, 64));
}

// ---------------------------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------------------------

TEST(Routing2DCodec, PackDecodeRoundTripsOnAPlainMesh) {
    for (const auto& s : kRepresentativeShapes) {
        std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0xAB);
        ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), s.y, s.x, dor_y, dor_x)) << s.name;

        for (uint32_t dst = 0; dst < s.y; ++dst) {
            const std::uint8_t* row = Codec::y_row(table.data(), s.y, dst);
            for (uint32_t cur = 0; cur < s.y; ++cur) {
                const uint8_t got = Codec::get_action_2bit(row, cur);
                if (cur == dst) {
                    EXPECT_EQ(got, Codec::Y2_STOP) << s.name << " y[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? Codec::Y2_SOUTH : Codec::Y2_NORTH)
                        << s.name << " y[" << dst << "][" << cur << "]";
                }
            }
        }
        for (uint32_t dst = 0; dst < s.x; ++dst) {
            const std::uint8_t* row = Codec::x_row(table.data(), s.y, s.x, dst);
            for (uint32_t cur = 0; cur < s.x; ++cur) {
                const uint8_t got = Codec::get_action_2bit(row, cur);
                if (cur == dst) {
                    EXPECT_EQ(got, Codec::X2_STOP) << s.name << " x[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? Codec::X2_EAST : Codec::X2_WEST)
                        << s.name << " x[" << dst << "][" << cur << "]";
                }
            }
        }
    }
}

TEST(Routing2DCodec, PackWritesOnlyItsOwnRegion) {
    constexpr uint32_t kY = 8, kX = 4;
    constexpr std::uint8_t kSentinel = 0xAB;
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, kSentinel);
    ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), kY, kX, dor_y, dor_x));

    for (uint32_t i = action_vector_bytes(kY, kX); i < Codec::ACTION_VECTOR_CAPACITY_BYTES; ++i) {
        EXPECT_EQ(table[i], kSentinel) << "pack scribbled past its region at byte " << i;
    }
}

TEST(Routing2DCodec, PackRejectsAShortOutputSpanWithoutWriting) {
    constexpr uint32_t kY = 8, kX = 4;
    constexpr std::uint8_t kSentinel = 0xAB;
    std::vector<std::uint8_t> table(action_vector_bytes(kY, kX) - 1, kSentinel);

    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), kY, kX, dor_y, dor_x));
    EXPECT_EQ(table, std::vector<std::uint8_t>(table.size(), kSentinel));
}

TEST(Routing2DCodec, PackRejectsShapesItCannotRepresent) {
    constexpr std::uint8_t kSentinel = 0xAB;
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, kSentinel);
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), 0, 4, dor_y, dor_x));
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), 64, 8, dor_y, dor_x));
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), Codec::MAX_AXIS_SIZE + 1, 4, dor_y, dor_x));
    EXPECT_EQ(table, std::vector<std::uint8_t>(Codec::ACTION_VECTOR_CAPACITY_BYTES, kSentinel));
}

// An axis action that does not belong to that axis is a caller bug, not something to encode as a
// zero and forward blindly.
TEST(Routing2DCodec, PackRejectsOffAxisActions) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    auto east_on_y = [](uint32_t, uint32_t) { return eth_chan_directions::EAST; };
    auto north_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::NORTH; };
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, east_on_y, dor_x));
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, dor_y, north_on_x));
}

// Z is legal on the Y axis (an express chord jumps along rows) and never on X.
TEST(Routing2DCodec, ZIsAYAxisActionOnly) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    auto z_on_y = [](uint32_t cur, uint32_t dst) {
        return cur == dst ? eth_chan_directions::NORTH : eth_chan_directions::Z;
    };
    auto z_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::Z; };
    EXPECT_TRUE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, z_on_y, dor_x));
    EXPECT_FALSE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, dor_y, z_on_x));
}

TEST(Routing2DCodec, WidenMapsEveryTwoBitCode) {
    EXPECT_EQ(Codec::widen_y(Codec::Y2_NORTH), Codec::ACTION_NORTH);
    EXPECT_EQ(Codec::widen_y(Codec::Y2_SOUTH), Codec::ACTION_SOUTH);
    EXPECT_EQ(Codec::widen_y(Codec::Y2_Z), Codec::ACTION_Z);
    EXPECT_EQ(Codec::widen_y(Codec::Y2_STOP), 0);
    EXPECT_EQ(Codec::widen_x(Codec::X2_EAST), Codec::ACTION_EAST);
    EXPECT_EQ(Codec::widen_x(Codec::X2_WEST), Codec::ACTION_WEST);
    EXPECT_EQ(Codec::widen_x(Codec::X2_STOP), 0);
    EXPECT_EQ(Codec::widen_x(Codec::X2_INVALID), 0);
}

// ---------------------------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------------------------

// THE decode invariant. A router facing N/S/Z consumes its Y action whenever the Y byte is nonzero,
// and only otherwise falls through to X. The tempting-but-wrong test is `action_y & (N|S|Z)`: a Y
// byte holding LOCAL_DELIVER alone is nonzero but has no eth bit set, so the masked test would fall
// through to X and forward a packet that should have terminated here.
TEST(Routing2DCodec, LocalDeliverOnlyYByteDoesNotFallThroughToX) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    route_buffer[kLocalY] = Codec::ACTION_LOCAL_DELIVER;  // terminate here
    route_buffer[kY + kLocalX] = Codec::ACTION_EAST;      // a stale X action that must NOT be taken

    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), kLocalY, kLocalX, kY),
        Codec::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::SOUTH>(route_buffer.data(), kLocalY, kLocalX, kY),
        Codec::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::Z>(route_buffer.data(), kLocalY, kLocalX, kY),
        Codec::ACTION_LOCAL_DELIVER);
}

TEST(Routing2DCodec, EastWestFacingRoutersReadTheXMapOnly) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    route_buffer[kLocalY] = Codec::ACTION_SOUTH;  // must be ignored by an E/W-facing router
    route_buffer[kY + kLocalX] = Codec::ACTION_EAST;

    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::EAST>(route_buffer.data(), kLocalY, kLocalX, kY), Codec::ACTION_EAST);
    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::WEST>(route_buffer.data(), kLocalY, kLocalX, kY), Codec::ACTION_EAST);
}

TEST(Routing2DCodec, NorthSouthFacingRoutersPreferYThenFallThroughToX) {
    constexpr uint32_t kY = 4, kX = 4;
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    {  // rows still differ -> take the Y action (dimension order)
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kLocalY] = Codec::ACTION_SOUTH;
        rb[kY + kLocalX] = Codec::ACTION_EAST;
        EXPECT_EQ(
            Codec::decode_action<eth_chan_directions::NORTH>(rb.data(), kLocalY, kLocalX, kY), Codec::ACTION_SOUTH);
    }
    {  // row reached (Y byte zero) -> fall through to X
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kY + kLocalX] = Codec::ACTION_EAST;
        EXPECT_EQ(
            Codec::decode_action<eth_chan_directions::NORTH>(rb.data(), kLocalY, kLocalX, kY), Codec::ACTION_EAST);
    }
}

// A zeroed route buffer means "no action anywhere" and decodes to 0 for every router facing.
TEST(Routing2DCodec, ZeroedRouteBufferDecodesToNothing) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};

    EXPECT_EQ(Codec::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), 2, 1, kY), 0);
    EXPECT_EQ(Codec::decode_action<eth_chan_directions::EAST>(route_buffer.data(), 2, 1, kY), 0);
}

}  // namespace tt::tt_fabric::routing_2d_codec_tests
