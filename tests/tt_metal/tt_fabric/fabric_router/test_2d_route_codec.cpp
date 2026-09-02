// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side unit tests for the destination-major 2D action-map codec -- the single encoding all 2D
// fabric traffic uses after the codec unification.
//
// Machine-free by construction: every function under test is constexpr/inline arithmetic over plain
// buffers, so these need no cluster, no control plane and no device. They cover the parts of the
// codec that a device test would only exercise indirectly, and that a silent mis-encode would make
// look like a hang rather than a wrong answer.
//
// What is deliberately NOT here: anything that needs L1 (the device-side route_buffer *builder* in
// tt_fabric_api.h reads the packed vectors out of the routing table), and anything that needs a
// real mesh graph (see test_express_ring_topology.cpp for the topology-derivation goldens).

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::routing_2d_codec_tests {
namespace {

using Codec = Routing2DCodec;
constexpr uint32_t kMaximumActionMapBytes = 64 + 4;

struct Shape {
    const char* name;
    uint32_t y;
    uint32_t x;
};

// Representative packing geometries. The descriptor sweep owns exhaustive in-tree shape coverage.
constexpr std::array<Shape, 4> kRepresentativeShapes = {{
    {"[8,4]", 8, 4},
    {"[8,8]", 8, 8},
    {"[8,16]", 8, 16},
    {"[1,16]", 1, 16},
}};

// Dimension-order oracle for a plain (chordless) mesh: rows increase southward, columns eastward.
eth_chan_directions dor_y(uint32_t cur, uint32_t dst) {
    return cur < dst ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
}
eth_chan_directions dor_x(uint32_t cur, uint32_t dst) {
    return cur < dst ? eth_chan_directions::EAST : eth_chan_directions::WEST;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Shape admissibility
// ---------------------------------------------------------------------------------------------

TEST(Routing2DCodec, VectorsRegionBytesMatchesThePackedLayout) {
    // Y table is y_size rows of ceil(y_size/4) bytes; X table likewise. Checked against hand
    // arithmetic so a change to the packing density cannot pass unnoticed.
    EXPECT_EQ(Codec::vectors_region_bytes(8, 8), 8u * 2u + 8u * 2u);     // square
    EXPECT_EQ(Codec::vectors_region_bytes(1, 16), 1u * 1u + 16u * 4u);   // narrow rectangle
    EXPECT_EQ(Codec::vectors_region_bytes(64, 4), 64u * 16u + 4u * 1u);  // 1028
}

TEST(Routing2DCodec, MaximumShapesFillTheHybridSlotExactly) {
    for (const auto& shape : std::array<Shape, 2>{{{"[64,4]", 64, 4}, {"[4,64]", 4, 64}}}) {
        EXPECT_TRUE(Codec::shape_fits_route_table(shape.y, shape.x));
        EXPECT_EQ(Codec::vectors_region_bytes(shape.y, shape.x), Codec::ACTION_VECTOR_CAPACITY_BYTES) << shape.name;
        EXPECT_EQ(Codec::mcast_tree_region_bytes(shape.y, shape.x), Codec::MCAST_TREE_CAPACITY_BYTES) << shape.name;
        EXPECT_TRUE(Codec::route_table_regions_fit(shape.y, shape.x)) << shape.name;
    }
}

TEST(Routing2DCodec, ShapesBeyondTheAddressableRangeAreRejected) {
    EXPECT_FALSE(Codec::shape_fits_route_table(Codec::MAX_AXIS_SIZE + 1, 4));
    EXPECT_FALSE(Codec::shape_fits_route_table(4, Codec::MAX_AXIS_SIZE + 1));
    // Within the per-axis range but too large to pack.
    EXPECT_FALSE(Codec::shape_fits_route_table(64, 64));
}

// ---------------------------------------------------------------------------------------------
// Packet header sizing
// ---------------------------------------------------------------------------------------------

TEST(Routing2DCodec, HeaderTiersAreSizeClassAligned) {
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<20>), 80u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<36>), 96u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<52>), 112u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<kMaximumActionMapBytes>), 128u);
    EXPECT_EQ(64u + 4u, kMaximumActionMapBytes);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<kMaximumActionMapBytes>) + sizeof(UDMControlFields), 144u);
}

// ---------------------------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------------------------

TEST(Routing2DCodec, PackDecodeRoundTripsOnAPlainMesh) {
    for (const auto& s : kRepresentativeShapes) {
        std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0xAB);
        ASSERT_TRUE(Codec::pack_route_vectors(table.data(), s.y, s.x, dor_y, dor_x)) << s.name;

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

TEST(Routing2DCodec, PackDecodeRoundTripsAtMaximumShapes) {
    for (const auto& shape : std::array<Shape, 2>{{{"[64,4]", 64, 4}, {"[4,64]", 4, 64}}}) {
        std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
        ASSERT_TRUE(Codec::pack_route_vectors(table.data(), shape.y, shape.x, dor_y, dor_x)) << shape.name;

        for (uint32_t dst = 0; dst < shape.y; ++dst) {
            const std::uint8_t* row = Codec::y_row(table.data(), shape.y, dst);
            for (uint32_t cur = 0; cur < shape.y; ++cur) {
                const uint8_t expected = cur == dst ? Codec::Y2_STOP : (cur < dst ? Codec::Y2_SOUTH : Codec::Y2_NORTH);
                EXPECT_EQ(Codec::get_action_2bit(row, cur), expected)
                    << shape.name << " y[" << dst << "][" << cur << "]";
            }
        }
        for (uint32_t dst = 0; dst < shape.x; ++dst) {
            const std::uint8_t* row = Codec::x_row(table.data(), shape.y, shape.x, dst);
            for (uint32_t cur = 0; cur < shape.x; ++cur) {
                const uint8_t expected = cur == dst ? Codec::X2_STOP : (cur < dst ? Codec::X2_EAST : Codec::X2_WEST);
                EXPECT_EQ(Codec::get_action_2bit(row, cur), expected)
                    << shape.name << " x[" << dst << "][" << cur << "]";
            }
        }
    }
}

TEST(Routing2DCodec, PackWritesOnlyItsOwnRegion) {
    constexpr uint32_t kY = 8, kX = 4;
    constexpr std::uint8_t kSentinel = 0xAB;
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, kSentinel);
    ASSERT_TRUE(Codec::pack_route_vectors(table.data(), kY, kX, dor_y, dor_x));

    for (uint32_t i = Codec::vectors_region_bytes(kY, kX); i < Codec::ACTION_VECTOR_CAPACITY_BYTES; ++i) {
        EXPECT_EQ(table[i], kSentinel) << "pack scribbled past its region at byte " << i;
    }
}

TEST(Routing2DCodec, PackRejectsShapesItCannotRepresent) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    EXPECT_FALSE(Codec::pack_route_vectors(table.data(), 64, 64, dor_y, dor_x));
    EXPECT_FALSE(Codec::pack_route_vectors(table.data(), Codec::MAX_AXIS_SIZE + 1, 4, dor_y, dor_x));
}

// An axis action that does not belong to that axis is a caller bug, not something to encode as a
// zero and forward blindly.
TEST(Routing2DCodec, PackRejectsOffAxisActions) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    auto east_on_y = [](uint32_t, uint32_t) { return eth_chan_directions::EAST; };
    auto north_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::NORTH; };
    EXPECT_FALSE(Codec::pack_route_vectors(table.data(), 8, 4, east_on_y, dor_x));
    EXPECT_FALSE(Codec::pack_route_vectors(table.data(), 8, 4, dor_y, north_on_x));
}

// Z is legal on the Y axis (an express chord jumps along rows) and never on X.
TEST(Routing2DCodec, ZIsAYAxisActionOnly) {
    std::vector<std::uint8_t> table(Codec::ACTION_VECTOR_CAPACITY_BYTES, 0);
    auto z_on_y = [](uint32_t cur, uint32_t dst) {
        return cur == dst ? eth_chan_directions::NORTH : eth_chan_directions::Z;
    };
    auto z_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::Z; };
    EXPECT_TRUE(Codec::pack_route_vectors(table.data(), 8, 4, z_on_y, dor_x));
    EXPECT_FALSE(Codec::pack_route_vectors(table.data(), 8, 4, dor_y, z_on_x));
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

// A zeroed route buffer means "no action anywhere". It must decode to 0 and be rejected by the
// validity check, not silently forwarded -- a zeroed region is how a mis-encode presents itself.
TEST(Routing2DCodec, ZeroedRouteBufferDecodesToNothingAndIsInvalid) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};

    EXPECT_EQ(Codec::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), 2, 1, kY), 0);
    EXPECT_EQ(Codec::decode_action<eth_chan_directions::EAST>(route_buffer.data(), 2, 1, kY), 0);
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::NORTH>(0));
}

// ---------------------------------------------------------------------------------------------
// Action validity and dispatch keys
// ---------------------------------------------------------------------------------------------

// A packet must never be sent back the way it came: the action's self-facing bit is a hard error,
// because acting on it would be an immediate two-node loop.
TEST(Routing2DCodec, SelfFacingActionIsRejected) {
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_NORTH));
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::EAST>(Codec::ACTION_EAST));
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::Z>(Codec::ACTION_Z));
    // The same bit set alongside a legal one is still a rejection.
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_NORTH | Codec::ACTION_EAST));
}

TEST(Routing2DCodec, ReservedBitsAreRejected) {
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_SOUTH | 0b01000000));
    EXPECT_FALSE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_SOUTH | 0b10000000));
}

TEST(Routing2DCodec, LegalActionsAreAccepted) {
    EXPECT_TRUE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_SOUTH));
    EXPECT_TRUE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_LOCAL_DELIVER));
    EXPECT_TRUE(Codec::action_is_valid<eth_chan_directions::NORTH>(Codec::ACTION_SOUTH | Codec::ACTION_LOCAL_DELIVER));
    EXPECT_TRUE(Codec::action_is_valid<eth_chan_directions::EAST>(Codec::ACTION_WEST | Codec::ACTION_NORTH));
}

// fwd_dirs is the slot order the dispatch key is packed against; every facing must exclude itself
// and list exactly the other four, or a key bit would select the wrong outgoing sender.
TEST(Routing2DCodec, ForwardDirectionsExcludeSelfAndCoverTheRest) {
    const auto check = [](auto dirs, eth_chan_directions self) {
        std::array<bool, 5> seen = {};
        for (auto d : dirs) {
            EXPECT_NE(d, self) << "a router must not list its own facing as an output";
            seen[static_cast<size_t>(d)] = true;
        }
        for (size_t i = 0; i < seen.size(); ++i) {
            if (static_cast<eth_chan_directions>(i) != self) {
                EXPECT_TRUE(seen[i]) << "missing output direction " << i;
            }
        }
    };
    check(Codec::fwd_dirs<eth_chan_directions::EAST>(), eth_chan_directions::EAST);
    check(Codec::fwd_dirs<eth_chan_directions::WEST>(), eth_chan_directions::WEST);
    check(Codec::fwd_dirs<eth_chan_directions::NORTH>(), eth_chan_directions::NORTH);
    check(Codec::fwd_dirs<eth_chan_directions::SOUTH>(), eth_chan_directions::SOUTH);
    check(Codec::fwd_dirs<eth_chan_directions::Z>(), eth_chan_directions::Z);
}

TEST(Routing2DCodec, ForwardKeyPacksEachSelectedDirection) {
    constexpr auto kDirs = Codec::fwd_dirs<eth_chan_directions::NORTH>();
    for (size_t slot = 0; slot < kDirs.size(); ++slot) {
        const std::uint8_t action = Codec::action_bit(kDirs[slot]);
        EXPECT_EQ(Codec::pack_fwd_key<eth_chan_directions::NORTH>(action), 1u << slot)
            << "slot " << slot << " did not round-trip";
    }
    // LOCAL_DELIVER is handled after the eth fanout and must stay out of the key.
    EXPECT_EQ(Codec::pack_fwd_key<eth_chan_directions::NORTH>(Codec::ACTION_LOCAL_DELIVER), 0u);
}

}  // namespace tt::tt_fabric::routing_2d_codec_tests
