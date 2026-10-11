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

// The representative shapes plus axes that aren't a multiple of four, so the widen's tail bytes are covered.
constexpr std::array<Shape, 6> kWidenShapes = {{
    {"[8,4]", 8, 4},
    {"[1,16]", 1, 16},
    {"[64,4]", 64, 4},
    {"[4,64]", 4, 64},
    {"[3,5]", 3, 5},
    {"[6,7]", 6, 7},
}};

// Route2DWriter that records the byte count of every append and when flush is called.
struct RecordingWriter {
    std::vector<uint32_t> append_sizes;
    uint32_t flushes = 0;
    bool appended_after_flush = false;
    void append(uint32_t /*widened*/, uint32_t num_bytes) {
        appended_after_flush = appended_after_flush || flushes != 0;
        append_sizes.push_back(num_bytes);
    }
    void flush() { ++flushes; }
};

// The widen works in constant expressions. A 2x2 mesh, widened toward chip 3 at (1, 1).
static_assert([] {
    const std::array<std::uint8_t, 4> table = {
        Codec::Y2_NORTH << 2, Codec::Y2_SOUTH, Codec::X2_WEST << 2, Codec::X2_EAST};
    std::array<std::uint8_t, 4> map = {};
    HostRoute2DWriter writer{map.data()};
    widen_2d_route(writer, table.data(), 3, 2, 2);
    return map;
}() == std::array<std::uint8_t, 4>{Codec::ACTION_SOUTH, 0, Codec::ACTION_EAST, Codec::ACTION_LOCAL_DELIVER});

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

TEST(Routing2DCodec, PackedByteWidenMatchesPerFieldWiden) {
    for (uint32_t b = 0; b < 256; ++b) {
        const uint32_t widened_y = Codec::widen_y_packed_byte(b);
        const uint32_t widened_x = Codec::widen_x_packed_byte(b);
        for (uint32_t k = 0; k < Codec::ACTIONS_PER_BYTE; ++k) {
            const uint8_t code = (b >> (Codec::BITS_PER_ACTION * k)) & 0b11;
            EXPECT_EQ((widened_y >> (8 * k)) & 0xFF, Codec::widen_y(code)) << "byte " << b << " field " << k;
            EXPECT_EQ((widened_x >> (8 * k)) & 0xFF, Codec::widen_x(code)) << "byte " << b << " field " << k;
        }
    }
}

// Every destination's widened map routes by dimension order: Y toward dst_y, then X toward dst_x, delivering at
// dst_x. Nothing past the Y + X bytes is written.
TEST(Routing2DCodec, WidenFollowsDimensionOrder) {
    constexpr std::uint8_t kSentinel = 0xAB;
    constexpr uint32_t kSlackBytes = 4;
    for (const auto& s : kWidenShapes) {
        std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
        ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), s.y, s.x, dor_y, dor_x)) << s.name;

        for (uint32_t dst = 0; dst < s.y * s.x; ++dst) {
            const uint32_t dst_y = dst / s.x;
            const uint32_t dst_x = dst % s.x;
            std::vector<std::uint8_t> map(s.y + s.x + kSlackBytes, kSentinel);
            HostRoute2DWriter writer{map.data()};
            widen_2d_route(writer, table.data(), dst, s.y, s.x);

            for (uint32_t y = 0; y < s.y; ++y) {
                const uint8_t want = y < dst_y ? Codec::ACTION_SOUTH : (y > dst_y ? Codec::ACTION_NORTH : 0);
                EXPECT_EQ(map[y], want) << s.name << " dst " << dst << " y " << y;
            }
            for (uint32_t x = 0; x < s.x; ++x) {
                const uint8_t want =
                    x < dst_x ? Codec::ACTION_EAST : (x > dst_x ? Codec::ACTION_WEST : Codec::ACTION_LOCAL_DELIVER);
                EXPECT_EQ(map[s.y + x], want) << s.name << " dst " << dst << " x " << x;
            }
            for (uint32_t i = s.y + s.x; i < map.size(); ++i) {
                EXPECT_EQ(map[i], kSentinel) << s.name << " dst " << dst << " wrote past the map at byte " << i;
            }
        }
    }
}

// Route2DWordWriter relies on this contract: one to four bytes per append, Y + X bytes in total, then one flush.
TEST(Routing2DCodec, WidenAppendsAtMostOneWordAtATime) {
    for (const auto& s : kWidenShapes) {
        std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
        ASSERT_TRUE(pack_2d_route_vectors(table.data(), table.size(), s.y, s.x, dor_y, dor_x)) << s.name;

        for (uint32_t dst = 0; dst < s.y * s.x; ++dst) {
            RecordingWriter writer;
            widen_2d_route(writer, table.data(), dst, s.y, s.x);

            uint32_t total = 0;
            for (uint32_t n : writer.append_sizes) {
                EXPECT_GE(n, 1u) << s.name << " dst " << dst;
                EXPECT_LE(n, 4u) << s.name << " dst " << dst;
                total += n;
            }
            EXPECT_EQ(total, s.y + s.x) << s.name << " dst " << dst;
            EXPECT_EQ(writer.flushes, 1u) << s.name << " dst " << dst;
            EXPECT_FALSE(writer.appended_after_flush) << s.name << " dst " << dst;
        }
    }
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
        Codec::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), kLocalY, kLocalX, kY),
        Codec::ACTION_NORTH);
    EXPECT_EQ(
        Codec::decode_action<eth_chan_directions::EAST>(route_buffer.data(), kLocalY, kLocalX, kY), Codec::ACTION_EAST);

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
