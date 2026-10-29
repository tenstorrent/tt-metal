// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pins the direction-to-slot relation shared by connection placement and producer naming. A
// divergence would stamp a flow-control guard onto the wrong sender.

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

TEST(DirectionSlotBijectionTest, DownstreamSenderChannelVcOffsetsArePinned) {
    using D = eth_chan_directions;
    EXPECT_EQ(builder::get_downstream_sender_channel_for_vc(true, 0, D::Z, D::NORTH), 4u);
    EXPECT_EQ(builder::get_downstream_sender_channel_for_vc(true, 1, D::Z, D::NORTH), 3u);
    EXPECT_EQ(builder::get_downstream_sender_channel_for_vc(false, 0, D::NORTH, D::Z), 1u);
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
