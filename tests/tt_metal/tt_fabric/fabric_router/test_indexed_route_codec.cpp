// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side unit tests for the indexed 2D route codec -- the single encoding all 2D fabric traffic
// uses after the codec unification.
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

namespace tt::tt_fabric::indexed_route_codec_tests {
namespace {

using IRF = IndexedMeshRoutingFields;

// Every 2D mesh shape that exists in-tree, plus the two boundary shapes the bounds are written
// against. {name, Y, X}.
struct Shape {
    const char* name;
    uint32_t y;
    uint32_t x;
};

constexpr std::array<Shape, 10> kInTreeShapes = {{
    {"[32,4] Galaxy", 32, 4},
    {"[8,4]", 8, 4},
    {"[8,8]", 8, 8},
    {"[8,16]", 8, 16},
    {"[16,8]", 16, 8},
    {"[16,4]", 16, 4},
    {"[4,4]", 4, 4},
    {"[2,4]", 2, 4},
    {"[2,2]", 2, 2},
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

// Guards B1: MAX_INDEXED_MESH_X used to be 4, which silently excluded [8,8], [8,16], [16,8] and
// [1,16]. The bound is now the addressable coordinate range, not the physical slot shape.
TEST(IndexedRouteCodec, EveryInTreeShapeIsIndexable) {
    for (const auto& s : kInTreeShapes) {
        EXPECT_TRUE(IRF::shape_is_indexable(s.y, s.x)) << s.name << " must be indexable";
        EXPECT_LE(IRF::vectors_region_bytes(s.y, s.x), IRF::INDEXED_VECTOR_TABLE_BYTES) << s.name;
    }
}

TEST(IndexedRouteCodec, VectorsRegionBytesMatchesThePackedLayout) {
    // Y table is y_size rows of ceil(y_size/4) bytes; X table likewise. Checked against hand
    // arithmetic so a change to the packing density cannot pass unnoticed.
    EXPECT_EQ(IRF::vectors_region_bytes(32, 4), 32u * 8u + 4u * 1u);   // 260
    EXPECT_EQ(IRF::vectors_region_bytes(8, 8), 8u * 2u + 8u * 2u);     // 32
    EXPECT_EQ(IRF::vectors_region_bytes(1, 16), 1u * 1u + 16u * 4u);   // 65
    EXPECT_EQ(IRF::vectors_region_bytes(64, 4), 64u * 16u + 4u * 1u);  // 1028
}

// [64,4] is the shape the L1 slot was sized for; it fits the vectors *exactly* and nothing else.
TEST(IndexedRouteCodec, SixtyFourByFourFillsTheSlotExactly) {
    EXPECT_TRUE(IRF::shape_is_indexable(64, 4));
    EXPECT_EQ(IRF::vectors_region_bytes(64, 4), IRF::INDEXED_VECTOR_TABLE_BYTES);
    // ...which leaves no room for the multicast trees in the same slot.
    EXPECT_FALSE(IRF::hybrid_region_fits(64, 4));
}

TEST(IndexedRouteCodec, ShapesBeyondTheAddressableRangeAreRejected) {
    EXPECT_FALSE(IRF::shape_is_indexable(IRF::MAX_INDEXED_MESH_AXIS + 1, 4));
    EXPECT_FALSE(IRF::shape_is_indexable(4, IRF::MAX_INDEXED_MESH_AXIS + 1));
    // Within the per-axis range but too large to pack.
    EXPECT_FALSE(IRF::shape_is_indexable(64, 64));
}

TEST(IndexedRouteCodec, GalaxyLeavesRoomForItsMulticastTrees) {
    EXPECT_TRUE(IRF::hybrid_region_fits(32, 4));
    EXPECT_TRUE(IRF::hybrid_region_fits(8, 4));
    EXPECT_TRUE(IRF::hybrid_region_fits(8, 8));
}

// ---------------------------------------------------------------------------------------------
// Packet header sizing
// ---------------------------------------------------------------------------------------------

// The indexed maps occupy Y + X bytes -- two more than the (Y-1) + (X-1) hop count the tiers were
// originally sized from. Guards B6: [32,4] needs 36, which is one past the old 35 B tier.
TEST(IndexedRouteCodec, EveryInTreeShapeFitsTheNinetySixByteHeader) {
    constexpr uint32_t kTier96 = 36;
    for (const auto& s : kInTreeShapes) {
        EXPECT_LE(s.y + s.x, kTier96) << s.name << " needs " << (s.y + s.x) << " route bytes";
    }
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<36>), 96u);
}

TEST(IndexedRouteCodec, HeaderTiersAreSizeClassAligned) {
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<20>), 80u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<36>), 96u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<52>), 112u);
    EXPECT_EQ(sizeof(HybridMeshPacketHeaderT<67>), 128u);
}

// [64,4] is blocked from the indexed codec by the packet header alone, by exactly one byte. If this
// test ever fails because the shortfall closed, [64,4] becomes a live option -- see issue #32237.
TEST(IndexedRouteCodec, SixtyFourByFourMissesTheHeaderBoundByOneByte) {
    constexpr uint32_t kMaxRouteBytes = 67;
    EXPECT_EQ(64u + 4u, kMaxRouteBytes + 1);
}

// ---------------------------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------------------------

TEST(IndexedRouteCodec, PackDecodeRoundTripsOnAPlainMesh) {
    for (const auto& s : kInTreeShapes) {
        std::vector<std::uint8_t> table(IRF::INDEXED_VECTOR_TABLE_BYTES, 0xAB);
        ASSERT_TRUE(IRF::pack_indexed_route_vectors(table.data(), s.y, s.x, dor_y, dor_x)) << s.name;

        for (uint32_t dst = 0; dst < s.y; ++dst) {
            const std::uint8_t* row = IRF::y_row(table.data(), s.y, dst);
            for (uint32_t cur = 0; cur < s.y; ++cur) {
                const uint8_t got = IRF::get_action_2bit(row, cur);
                if (cur == dst) {
                    EXPECT_EQ(got, IRF::Y2_STOP) << s.name << " y[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? IRF::Y2_SOUTH : IRF::Y2_NORTH)
                        << s.name << " y[" << dst << "][" << cur << "]";
                }
            }
        }
        for (uint32_t dst = 0; dst < s.x; ++dst) {
            const std::uint8_t* row = IRF::x_row(table.data(), s.y, s.x, dst);
            for (uint32_t cur = 0; cur < s.x; ++cur) {
                const uint8_t got = IRF::get_action_2bit(row, cur);
                if (cur == dst) {
                    EXPECT_EQ(got, IRF::X2_STOP) << s.name << " x[" << dst << "][" << cur << "]";
                } else {
                    EXPECT_EQ(got, cur < dst ? IRF::X2_EAST : IRF::X2_WEST)
                        << s.name << " x[" << dst << "][" << cur << "]";
                }
            }
        }
    }
}

TEST(IndexedRouteCodec, PackWritesOnlyItsOwnRegion) {
    constexpr uint32_t kY = 8, kX = 4;
    constexpr std::uint8_t kSentinel = 0xAB;
    std::vector<std::uint8_t> table(IRF::INDEXED_VECTOR_TABLE_BYTES, kSentinel);
    ASSERT_TRUE(IRF::pack_indexed_route_vectors(table.data(), kY, kX, dor_y, dor_x));

    for (uint32_t i = IRF::vectors_region_bytes(kY, kX); i < IRF::INDEXED_VECTOR_TABLE_BYTES; ++i) {
        EXPECT_EQ(table[i], kSentinel) << "pack scribbled past its region at byte " << i;
    }
}

TEST(IndexedRouteCodec, PackRejectsShapesItCannotRepresent) {
    std::vector<std::uint8_t> table(IRF::INDEXED_VECTOR_TABLE_BYTES, 0);
    EXPECT_FALSE(IRF::pack_indexed_route_vectors(table.data(), 64, 64, dor_y, dor_x));
    EXPECT_FALSE(IRF::pack_indexed_route_vectors(table.data(), IRF::MAX_INDEXED_MESH_AXIS + 1, 4, dor_y, dor_x));
}

// An axis action that does not belong to that axis is a caller bug, not something to encode as a
// zero and forward blindly.
TEST(IndexedRouteCodec, PackRejectsOffAxisActions) {
    std::vector<std::uint8_t> table(IRF::INDEXED_VECTOR_TABLE_BYTES, 0);
    auto east_on_y = [](uint32_t, uint32_t) { return eth_chan_directions::EAST; };
    auto north_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::NORTH; };
    EXPECT_FALSE(IRF::pack_indexed_route_vectors(table.data(), 8, 4, east_on_y, dor_x));
    EXPECT_FALSE(IRF::pack_indexed_route_vectors(table.data(), 8, 4, dor_y, north_on_x));
}

// Z is legal on the Y axis (an express chord jumps along rows) and never on X.
TEST(IndexedRouteCodec, ZIsAYAxisActionOnly) {
    std::vector<std::uint8_t> table(IRF::INDEXED_VECTOR_TABLE_BYTES, 0);
    auto z_on_y = [](uint32_t cur, uint32_t dst) {
        return cur == dst ? eth_chan_directions::NORTH : eth_chan_directions::Z;
    };
    auto z_on_x = [](uint32_t, uint32_t) { return eth_chan_directions::Z; };
    EXPECT_TRUE(IRF::pack_indexed_route_vectors(table.data(), 8, 4, z_on_y, dor_x));
    EXPECT_FALSE(IRF::pack_indexed_route_vectors(table.data(), 8, 4, dor_y, z_on_x));
}

TEST(IndexedRouteCodec, WidenMapsEveryTwoBitCode) {
    EXPECT_EQ(IRF::widen_y(IRF::Y2_NORTH), IRF::ACTION_NORTH);
    EXPECT_EQ(IRF::widen_y(IRF::Y2_SOUTH), IRF::ACTION_SOUTH);
    EXPECT_EQ(IRF::widen_y(IRF::Y2_Z), IRF::ACTION_Z);
    EXPECT_EQ(IRF::widen_y(IRF::Y2_STOP), 0);
    EXPECT_EQ(IRF::widen_x(IRF::X2_EAST), IRF::ACTION_EAST);
    EXPECT_EQ(IRF::widen_x(IRF::X2_WEST), IRF::ACTION_WEST);
    EXPECT_EQ(IRF::widen_x(IRF::X2_STOP), 0);
    EXPECT_EQ(IRF::widen_x(IRF::X2_INVALID), 0);
}

// ---------------------------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------------------------

// THE decode invariant. A router facing N/S/Z consumes its Y action whenever the Y byte is nonzero,
// and only otherwise falls through to X. The tempting-but-wrong test is `action_y & (N|S|Z)`: a Y
// byte holding LOCAL_DELIVER alone is nonzero but has no eth bit set, so the masked test would fall
// through to X and forward a packet that should have terminated here.
TEST(IndexedRouteCodec, LocalDeliverOnlyYByteDoesNotFallThroughToX) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    route_buffer[kLocalY] = IRF::ACTION_LOCAL_DELIVER;  // terminate here
    route_buffer[kY + kLocalX] = IRF::ACTION_EAST;      // a stale X action that must NOT be taken

    EXPECT_EQ(
        IRF::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), kLocalY, kLocalX, kY),
        IRF::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        IRF::decode_action<eth_chan_directions::SOUTH>(route_buffer.data(), kLocalY, kLocalX, kY),
        IRF::ACTION_LOCAL_DELIVER);
    EXPECT_EQ(
        IRF::decode_action<eth_chan_directions::Z>(route_buffer.data(), kLocalY, kLocalX, kY),
        IRF::ACTION_LOCAL_DELIVER);
}

TEST(IndexedRouteCodec, EastWestFacingRoutersReadTheXMapOnly) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    route_buffer[kLocalY] = IRF::ACTION_SOUTH;  // must be ignored by an E/W-facing router
    route_buffer[kY + kLocalX] = IRF::ACTION_EAST;

    EXPECT_EQ(
        IRF::decode_action<eth_chan_directions::EAST>(route_buffer.data(), kLocalY, kLocalX, kY), IRF::ACTION_EAST);
    EXPECT_EQ(
        IRF::decode_action<eth_chan_directions::WEST>(route_buffer.data(), kLocalY, kLocalX, kY), IRF::ACTION_EAST);
}

TEST(IndexedRouteCodec, NorthSouthFacingRoutersPreferYThenFallThroughToX) {
    constexpr uint32_t kY = 4, kX = 4;
    constexpr uint32_t kLocalY = 2, kLocalX = 1;

    {  // rows still differ -> take the Y action (dimension order)
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kLocalY] = IRF::ACTION_SOUTH;
        rb[kY + kLocalX] = IRF::ACTION_EAST;
        EXPECT_EQ(IRF::decode_action<eth_chan_directions::NORTH>(rb.data(), kLocalY, kLocalX, kY), IRF::ACTION_SOUTH);
    }
    {  // row reached (Y byte zero) -> fall through to X
        std::array<std::uint8_t, kY + kX> rb = {};
        rb[kY + kLocalX] = IRF::ACTION_EAST;
        EXPECT_EQ(IRF::decode_action<eth_chan_directions::NORTH>(rb.data(), kLocalY, kLocalX, kY), IRF::ACTION_EAST);
    }
}

// A zeroed route buffer means "no action anywhere". It must decode to 0 and be rejected by the
// validity check, not silently forwarded -- a zeroed region is how a mis-encode presents itself.
TEST(IndexedRouteCodec, ZeroedRouteBufferDecodesToNothingAndIsInvalid) {
    constexpr uint32_t kY = 4, kX = 4;
    std::array<std::uint8_t, kY + kX> route_buffer = {};

    EXPECT_EQ(IRF::decode_action<eth_chan_directions::NORTH>(route_buffer.data(), 2, 1, kY), 0);
    EXPECT_EQ(IRF::decode_action<eth_chan_directions::EAST>(route_buffer.data(), 2, 1, kY), 0);
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::NORTH>(0));
}

// ---------------------------------------------------------------------------------------------
// Action validity and dispatch keys
// ---------------------------------------------------------------------------------------------

// A packet must never be sent back the way it came: the action's self-facing bit is a hard error,
// because acting on it would be an immediate two-node loop.
TEST(IndexedRouteCodec, SelfFacingActionIsRejected) {
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_NORTH));
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::EAST>(IRF::ACTION_EAST));
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::Z>(IRF::ACTION_Z));
    // The same bit set alongside a legal one is still a rejection.
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_NORTH | IRF::ACTION_EAST));
}

TEST(IndexedRouteCodec, ReservedBitsAreRejected) {
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_SOUTH | 0b01000000));
    EXPECT_FALSE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_SOUTH | 0b10000000));
}

TEST(IndexedRouteCodec, LegalActionsAreAccepted) {
    EXPECT_TRUE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_SOUTH));
    EXPECT_TRUE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_LOCAL_DELIVER));
    EXPECT_TRUE(IRF::action_is_valid<eth_chan_directions::NORTH>(IRF::ACTION_SOUTH | IRF::ACTION_LOCAL_DELIVER));
    EXPECT_TRUE(IRF::action_is_valid<eth_chan_directions::EAST>(IRF::ACTION_WEST | IRF::ACTION_NORTH));
}

// fwd_dirs is the slot order the dispatch key is packed against; every facing must exclude itself
// and list exactly the other four, or a key bit would select the wrong outgoing sender.
TEST(IndexedRouteCodec, ForwardDirectionsExcludeSelfAndCoverTheRest) {
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
    check(IRF::fwd_dirs<eth_chan_directions::EAST>(), eth_chan_directions::EAST);
    check(IRF::fwd_dirs<eth_chan_directions::WEST>(), eth_chan_directions::WEST);
    check(IRF::fwd_dirs<eth_chan_directions::NORTH>(), eth_chan_directions::NORTH);
    check(IRF::fwd_dirs<eth_chan_directions::SOUTH>(), eth_chan_directions::SOUTH);
    check(IRF::fwd_dirs<eth_chan_directions::Z>(), eth_chan_directions::Z);
}

TEST(IndexedRouteCodec, ForwardKeyPacksEachSelectedDirection) {
    constexpr auto kDirs = IRF::fwd_dirs<eth_chan_directions::NORTH>();
    for (size_t slot = 0; slot < kDirs.size(); ++slot) {
        const std::uint8_t action = IRF::action_bit(kDirs[slot]);
        EXPECT_EQ(IRF::pack_fwd_key<eth_chan_directions::NORTH>(action), 1u << slot)
            << "slot " << slot << " did not round-trip";
    }
    // LOCAL_DELIVER is handled after the eth fanout and must stay out of the key.
    EXPECT_EQ(IRF::pack_fwd_key<eth_chan_directions::NORTH>(IRF::ACTION_LOCAL_DELIVER), 0u);
}

}  // namespace tt::tt_fabric::indexed_route_codec_tests
