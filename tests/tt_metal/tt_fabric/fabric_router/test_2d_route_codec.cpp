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
                    EXPECT_EQ(got, Codec::CODE_STOP) << s.name << " y[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? Codec::CODE_INCREASING : Codec::CODE_DECREASING)
                        << s.name << " y[" << dst << "][" << cur << "]";
                }
            }
        }
        for (uint32_t dst = 0; dst < s.x; ++dst) {
            const std::uint8_t* row = Codec::x_row(table.data(), s.y, s.x, dst);
            for (uint32_t cur = 0; cur < s.x; ++cur) {
                const uint8_t got = Codec::get_action_2bit(row, cur);
                if (cur == dst) {
                    EXPECT_EQ(got, Codec::CODE_STOP) << s.name << " x[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? Codec::CODE_INCREASING : Codec::CODE_DECREASING)
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

// Z is legal on whichever axis carries the express chords; the same CODE_Z packs on either leg.
TEST(Routing2DCodec, ZPacksOnEitherAxis) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    auto z_on_any = [](uint32_t, uint32_t) { return eth_chan_directions::Z; };

    ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, z_on_any, dor_x));
    EXPECT_EQ(Codec::get_action_2bit(Codec::y_row(table.data(), 8, /*dst_y=*/5), /*index=*/0), Codec::CODE_Z);

    ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), 8, 4, dor_y, z_on_any));
    EXPECT_EQ(Codec::get_action_2bit(Codec::x_row(table.data(), 8, 4, /*dst_x=*/3), /*index=*/0), Codec::CODE_Z);
}

TEST(Routing2DCodec, WidenMapsEveryTwoBitCode) {
    EXPECT_EQ(Codec::widen(Fabric2DAxis::Y, Codec::CODE_DECREASING), Codec::ACTION_NORTH);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::Y, Codec::CODE_INCREASING), Codec::ACTION_SOUTH);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::Y, Codec::CODE_Z), Codec::ACTION_Z);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::Y, Codec::CODE_STOP), 0);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::X, Codec::CODE_DECREASING), Codec::ACTION_WEST);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::X, Codec::CODE_INCREASING), Codec::ACTION_EAST);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::X, Codec::CODE_Z), Codec::ACTION_Z);
    EXPECT_EQ(Codec::widen(Fabric2DAxis::X, Codec::CODE_STOP), 0);
}

// encode is the inverse of widen on each axis and rejects the other axis's cardinals.
TEST(Routing2DCodec, EncodeInvertsWidenPerAxis) {
    for (const Fabric2DAxis axis : {Fabric2DAxis::Y, Fabric2DAxis::X}) {
        for (const std::uint8_t code : {Codec::CODE_DECREASING, Codec::CODE_INCREASING, Codec::CODE_Z}) {
            const std::uint8_t action = Codec::widen(axis, code);
            eth_chan_directions dir = eth_chan_directions::COUNT;
            for (int d = 0; d < static_cast<int>(eth_chan_directions::COUNT); ++d) {
                if (Codec::action_bit(static_cast<eth_chan_directions>(d)) == action) {
                    dir = static_cast<eth_chan_directions>(d);
                }
            }
            ASSERT_NE(dir, eth_chan_directions::COUNT);
            EXPECT_EQ(Codec::encode(axis, dir), code);
        }
    }
    EXPECT_EQ(Codec::encode(Fabric2DAxis::Y, eth_chan_directions::EAST), Codec::INVALID_CODE);
    EXPECT_EQ(Codec::encode(Fabric2DAxis::Y, eth_chan_directions::WEST), Codec::INVALID_CODE);
    EXPECT_EQ(Codec::encode(Fabric2DAxis::X, eth_chan_directions::NORTH), Codec::INVALID_CODE);
    EXPECT_EQ(Codec::encode(Fabric2DAxis::X, eth_chan_directions::SOUTH), Codec::INVALID_CODE);
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
        (Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::SOUTH, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::Z, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_LOCAL_DELIVER);
}

TEST(Routing2DCodec, EastWestFacingRoutersReadTheXMapOnly) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    route_buffer[kLocalY] = Codec::ACTION_SOUTH;  // must be ignored by an E/W-facing router
    route_buffer[kY + kLocalX] = Codec::ACTION_EAST;

    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::EAST, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_EAST);
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::WEST, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_EAST);
}

// An intermesh landing rebuilds the map and restarts the Y leg, so the E/W shortcut above would
// drop the owed N/S hop and strand every destination off the landing row.
TEST(Routing2DCodec, IntermeshLandingDecodesYFirstOnEveryFacing) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    // A freshly landed map owing a N/S hop, X row already holding the post-turn action.
    route_buffer[kLocalY] = Codec::ACTION_NORTH;
    route_buffer[kY + kLocalX] = Codec::ACTION_EAST;

    EXPECT_EQ(Codec::decode_action_y_first(route_buffer.data(), kLocalY, kLocalX, kY), Codec::ACTION_NORTH);

    // The facing-keyed decode disagrees precisely on E/W, which is what the landing must avoid.
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_NORTH);
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::EAST, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        Codec::ACTION_EAST);

    // Y leg spent: Y-first agrees with every facing again.
    route_buffer[kLocalY] = 0;
    EXPECT_EQ(Codec::decode_action_y_first(route_buffer.data(), kLocalY, kLocalX, kY), Codec::ACTION_EAST);
}

TEST(Routing2DCodec, NorthSouthFacingRoutersPreferYThenFallThroughToX) {
    constexpr uint32_t kY = 4, kX = 4;
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    {  // rows still differ -> take the Y action (dimension order)
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kLocalY] = Codec::ACTION_SOUTH;
        rb[kY + kLocalX] = Codec::ACTION_EAST;
        EXPECT_EQ(
            (Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::Y>(rb.data(), kLocalY, kLocalX, kY)),
            Codec::ACTION_SOUTH);
    }
    {  // row reached (Y byte zero) -> fall through to X
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kY + kLocalX] = Codec::ACTION_EAST;
        EXPECT_EQ(
            (Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::Y>(rb.data(), kLocalY, kLocalX, kY)),
            Codec::ACTION_EAST);
    }
}

// On an X-express mesh the chords run along E/W, so a Z-facing router must read its own X byte the
// way E/W-facing routers do. Y-first is wrong there: for multicast the Y byte of every target row is
// nonzero by design (LOCAL_DELIVER plus the root column's teeth plus the spine continuation), so the
// landing would act on the row's spine byte instead of its own column's. The bytes below are the ones
// McastReverseTreeTest.XAxisChordAtRootIsCopiedToTargetRows encodes for root (0,2), 2 hops S, 8 hops
// E, as seen by the chord landing at (1,5).
TEST(Routing2DCodec, ZFacingRouterOnXExpressReadsTheXMapOnly) {
    constexpr uint32_t kY = 4, kX = 32;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 1, kLocalX = 5;

    // Row 1: deliver, continue the spine S, and launch the root column's teeth E and Z.
    const std::uint8_t row_byte =
        Codec::ACTION_LOCAL_DELIVER | Codec::ACTION_SOUTH | Codec::ACTION_EAST | Codec::ACTION_Z;
    // Column 5: deliver, W to the leaf column 4, E onward to column 6.
    const std::uint8_t column_byte = Codec::ACTION_LOCAL_DELIVER | Codec::ACTION_WEST | Codec::ACTION_EAST;
    route_buffer[kLocalY] = row_byte;
    route_buffer[kY + kLocalX] = column_byte;

    // X-express: the landing acts on its column.
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::Z, Fabric2DAxis::X>(route_buffer.data(), kLocalY, kLocalX, kY)),
        column_byte);
    // Decoding the same bytes as a Y-express mesh is the misroute this axis exists to prevent: W is
    // lost, so column 4 is never delivered, and S starts a second spine down column 5.
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::Z, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        row_byte);

    // The axis changes only Z-facing decode: N/S-facing routers keep Y-first, E/W-facing keep X-only.
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::X>(route_buffer.data(), kLocalY, kLocalX, kY)),
        row_byte);
    EXPECT_EQ(
        (Codec::decode_action<eth_chan_directions::EAST, Fabric2DAxis::Y>(route_buffer.data(), kLocalY, kLocalX, kY)),
        column_byte);
}

// A zeroed route buffer means "no action anywhere" and decodes to 0 for every router facing.
TEST(Routing2DCodec, ZeroedRouteBufferDecodesToNothing) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};

    EXPECT_EQ((Codec::decode_action<eth_chan_directions::NORTH, Fabric2DAxis::Y>(route_buffer.data(), 2, 1, kY)), 0);
    EXPECT_EQ((Codec::decode_action<eth_chan_directions::EAST, Fabric2DAxis::Y>(route_buffer.data(), 2, 1, kY)), 0);
}

}  // namespace tt::tt_fabric::routing_2d_codec_tests
