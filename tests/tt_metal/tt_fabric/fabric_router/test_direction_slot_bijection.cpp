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
#include <enchantum/enchantum.hpp>
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
                << "facing " << enchantum::to_string(facing) << ", producer " << enchantum::to_string(producer);
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
                << "facing " << enchantum::to_string(facing) << ", producer " << enchantum::to_string(producer);
        }
        for (size_t compact = 0; compact < 4; ++compact) {
            EXPECT_EQ(
                builder::direction_compact_index(builder::direction_from_compact_index(facing, compact), facing),
                compact)
                << "facing " << enchantum::to_string(facing) << ", compact " << compact;
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
                << "facing " << enchantum::to_string(facing) << ", slot " << slot;
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
                << "facing " << enchantum::to_string(facing) << ", downstream " << enchantum::to_string(downstream);
            if (downstream == Z) {
                EXPECT_EQ(get_receiver_channel_compact_index(facing, downstream), 3u);
            }
        }
    }
}

TEST(DirectionSlotBijectionTest, DownstreamSenderChannelVcOffsetsArePinned) {
    // get_downstream_sender_channel_for_vc is the placement production uses, so nothing else pins
    // its VC0 +1 (worker) / VC1 +0 offset. A hardcoded table pins it directly.
    using D = eth_chan_directions;
    struct Row {
        eth_chan_directions producer;
        eth_chan_directions facing;
        uint32_t vc0_slot;
        uint32_t vc1_slot;
    };
    constexpr Row rows[] = {
        {D::NORTH, D::Z, 3, 2},      // boundary feed: compact of N among non-Z = 2
        {D::Z, D::NORTH, 4, 3},      // from-boundary slot: compact of Z among non-N = 3
        {D::SOUTH, D::NORTH, 3, 2},  // opposite-Y transit
        {D::EAST, D::NORTH, 1, 0},   // cross turn, first compact slot
        {D::NORTH, D::SOUTH, 3, 2},  // opposite of the above
        {D::WEST, D::EAST, 1, 0},    // X transit
    };
    for (const auto& row : rows) {
        EXPECT_EQ(
            builder::get_downstream_sender_channel_for_vc(/*is_2d_routing=*/true, 0, row.producer, row.facing),
            row.vc0_slot)
            << "producer " << enchantum::to_string(row.producer) << " -> facing " << enchantum::to_string(row.facing)
            << " (VC0)";
        EXPECT_EQ(
            builder::get_downstream_sender_channel_for_vc(/*is_2d_routing=*/true, 1, row.producer, row.facing),
            row.vc1_slot)
            << "producer " << enchantum::to_string(row.producer) << " -> facing " << enchantum::to_string(row.facing)
            << " (VC1)";
    }

    // 1D is always the single forwarding channel, regardless of the pair.
    EXPECT_EQ(builder::get_downstream_sender_channel_for_vc(/*is_2d_routing=*/false, 0, D::NORTH, D::Z), 1u);
}

TEST(DirectionSlotBijectionTest, ProducerSlotsRoundTripsAndNeverNamesFacing) {
    // One level up from the bijection: the type that carries it. Every slot a router actually has
    // round-trips through both directions of the mapping, no producer slot names the facing
    // direction (the U-turn), and the worker offset is the only difference between carrier VCs.
    for (const auto facing : k_directions) {
        const builder::RouterProducerSlots slots(facing, {5, 4, 1});
        for (const uint32_t vc : {0u, 1u}) {
            const uint32_t offset = (vc == 0) ? 1u : 0u;
            // The worker slot exists on VC0 only, and has no producer.
            EXPECT_EQ(slots.worker_channel(vc).has_value(), vc == 0);
            if (vc == 0) {
                EXPECT_EQ(*slots.worker_channel(0), 0u);
                EXPECT_FALSE(slots.producer_at(0, 0).has_value());
            }
            // The producer slots are exactly channels [offset, count), in eth order.
            const auto producers = slots.producer_slots(vc);
            ASSERT_EQ(producers.size(), 4u);
            for (size_t i = 0; i < producers.size(); ++i) {
                EXPECT_EQ(producers[i].channel, offset + i);
                EXPECT_NE(producers[i].producer, facing) << "facing " << enchantum::to_string(facing);
                // Round trip: channel -> producer -> channel.
                const auto read_back = slots.producer_at(vc, producers[i].channel);
                ASSERT_TRUE(read_back.has_value());
                EXPECT_EQ(*read_back, producers[i].producer);
                const auto back = slots.channel_for(vc, producers[i].producer);
                ASSERT_TRUE(back.has_value());
                EXPECT_EQ(*back, producers[i].channel);
            }
            // The facing direction itself has no slot, and channels outside the count map nothing.
            EXPECT_FALSE(slots.channel_for(vc, facing).has_value());
            EXPECT_FALSE(slots.producer_at(vc, slots.sender_count(vc)).has_value());
        }
        // VC2 has no producer mapping: its single sender is worker-type (channel 0).
        EXPECT_TRUE(slots.producer_slots(2).empty());
        EXPECT_FALSE(slots.channel_for(2, EAST).has_value());
        EXPECT_EQ(*slots.worker_channel(2), 0u);
    }

    // A router narrower than the family max exposes only its own slots.
    const builder::RouterProducerSlots narrow(EAST, {3, 2, 0});
    EXPECT_EQ(narrow.producer_slots(0).size(), 2u);  // channels 1, 2
    EXPECT_EQ(narrow.producer_slots(1).size(), 2u);  // channels 0, 1
    // The producer ranked at compact 2 exists in the family layout but not on this router.
    EXPECT_FALSE(narrow.channel_for(0, builder::direction_from_compact_index(EAST, 2)).has_value());
}

}  // namespace
}  // namespace tt::tt_fabric
