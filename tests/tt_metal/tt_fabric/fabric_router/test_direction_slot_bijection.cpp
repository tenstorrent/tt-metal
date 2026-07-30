// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the canonical direction <-> slot bijection in fabric_builder_helpers.
//
// Connection placement (the get_downstream_sender_channel arithmetic) and producer naming
// (get_sender_channel_direction) used to be two independent implementations of this relation,
// agreeing only because the enum ordering happened to line up with five hand-written tables.
// They now share one derivation; these tests pin the relation itself so the two sides cannot
// drift -- a divergence here would silently stamp the wrong flow-control guard on a live sender.

#include <gtest/gtest.h>

#include <array>
#include <map>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"

namespace tt::tt_fabric {
namespace {

using eth_chan_directions::EAST;
using eth_chan_directions::NORTH;
using eth_chan_directions::SOUTH;
using eth_chan_directions::WEST;
using eth_chan_directions::Z;

constexpr std::array<eth_chan_directions, 5> k_directions = {EAST, WEST, NORTH, SOUTH, Z};

TEST(DirectionSlotBijectionTest, PlacementAndNamingRoundTrip) {
    // The slot a producer is placed into and the producer direction read back from that slot are
    // the same fact. VC0 layout: sender channel 0 is the local worker, so producers live at
    // 1 + compact (and the VC1 view reads slot compact + 1 for the same producer).
    for (const auto facing : k_directions) {
        for (const auto producer : k_directions) {
            if (producer == facing) {
                continue;
            }
            const size_t compact = builder::direction_compact_index(producer, facing);
            EXPECT_EQ(builder::get_sender_channel_direction(facing, 1 + compact), producer)
                << "facing " << static_cast<int>(facing) << ", producer " << static_cast<int>(producer);
        }
    }
}

TEST(DirectionSlotBijectionTest, CompactIndexInvertsBothWays) {
    for (const auto facing : k_directions) {
        for (const auto producer : k_directions) {
            if (producer == facing) {
                continue;
            }
            EXPECT_EQ(
                builder::direction_from_compact_index(facing, builder::direction_compact_index(producer, facing)),
                producer)
                << "facing " << static_cast<int>(facing) << ", producer " << static_cast<int>(producer);
        }
        for (size_t compact = 0; compact < 4; ++compact) {
            EXPECT_EQ(
                builder::direction_compact_index(builder::direction_from_compact_index(facing, compact), facing),
                compact)
                << "facing " << static_cast<int>(facing) << ", compact " << compact;
            EXPECT_NE(builder::direction_from_compact_index(facing, compact), facing)
                << "compact " << compact << " must never name the facing direction itself";
        }
    }
}

TEST(DirectionSlotBijectionTest, MatchesTheRetiredHandWrittenTables) {
    // The five tables this derivation replaced, pinned as the regression oracle.
    using D = eth_chan_directions;
    const std::map<eth_chan_directions, std::array<eth_chan_directions, 5>> legacy_tables = {
        {EAST, {D::COUNT, WEST, NORTH, SOUTH, Z}},
        {WEST, {D::COUNT, EAST, NORTH, SOUTH, Z}},
        {NORTH, {D::COUNT, EAST, WEST, SOUTH, Z}},
        {SOUTH, {D::COUNT, EAST, WEST, NORTH, Z}},
        {Z, {D::COUNT, EAST, WEST, NORTH, SOUTH}},
    };
    for (const auto& [facing, table] : legacy_tables) {
        for (size_t slot = 1; slot < table.size(); ++slot) {
            EXPECT_EQ(builder::get_sender_channel_direction(facing, slot), table[slot])
                << "facing " << static_cast<int>(facing) << ", slot " << slot;
        }
    }
}

TEST(DirectionSlotBijectionTest, ReceiverDownstreamIndexIsTheSameRelation) {
    // The receiver-side downstream view (writer adapter indexing) reads the same bijection from
    // the other side: the downstream direction's rank among this router's non-self directions.
    for (const auto facing : k_directions) {
        for (const auto downstream : k_directions) {
            if (downstream == facing) {
                continue;
            }
            EXPECT_EQ(
                get_receiver_channel_compact_index(facing, downstream),
                builder::direction_compact_index(downstream, facing))
                << "facing " << static_cast<int>(facing) << ", downstream " << static_cast<int>(downstream);
            if (downstream == Z) {
                EXPECT_EQ(get_receiver_channel_compact_index(facing, downstream), 3u);
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric
