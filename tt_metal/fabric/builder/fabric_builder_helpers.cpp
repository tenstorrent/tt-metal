// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder_helpers.hpp"

namespace tt::tt_fabric::builder {

bool is_east_or_west(eth_chan_directions direction) {
    return direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST;
}
bool is_north_or_south(eth_chan_directions direction) {
    return direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;
}

eth_chan_directions get_sender_channel_direction(eth_chan_directions my_direction, size_t sender_channel_index) {
    using eth_chan_directions::COUNT;
    TT_FATAL(
        my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST ||
            my_direction == eth_chan_directions::NORTH || my_direction == eth_chan_directions::SOUTH ||
            my_direction == eth_chan_directions::Z,
        "Internal error: In get_sender_channel_direction, invalid direction");
    TT_FATAL(
        sender_channel_index < COUNT,
        "Internal error: In get_sender_channel_direction, sender channel index out of bounds. Got index {}",
        sender_channel_index);
    TT_FATAL(
        sender_channel_index > 0,
        "Internal error: In get_sender_channel_direction, sender channel index must be greater than 0. Got index {}",
        sender_channel_index);
    // Sender channel 0 is the local worker (no producer direction); channels 1-4 name the four
    // non-self producers, derived from the canonical bijection rather than per-facing tables.
    return direction_from_compact_index(my_direction, sender_channel_index - 1);
}

RouterProducerSlots::RouterProducerSlots(
    eth_chan_directions facing, const std::array<uint32_t, builder_config::MAX_NUM_VCS>& sender_counts) :
    facing_(facing), sender_counts_(sender_counts) {}

uint32_t RouterProducerSlots::sender_count(uint32_t vc) const {
    return vc < builder_config::MAX_NUM_VCS ? sender_counts_[vc] : 0;
}

std::optional<uint32_t> RouterProducerSlots::worker_channel(uint32_t vc) const {
    // VC0's channel 0 is the local worker; VC2's single sender is worker-type by definition. VC1
    // has no worker channel.
    if (vc == 1 || sender_count(vc) == 0) {
        return std::nullopt;
    }
    return 0;
}

std::optional<eth_chan_directions> RouterProducerSlots::producer_at(uint32_t vc, uint32_t channel) const {
    if (vc >= 2 || channel < worker_offset(vc) || channel >= sender_count(vc)) {
        return std::nullopt;
    }
    return direction_from_compact_index(facing_, channel - worker_offset(vc));
}

std::optional<uint32_t> RouterProducerSlots::channel_for(uint32_t vc, eth_chan_directions producer) const {
    if (vc >= 2 || producer == facing_) {
        return std::nullopt;
    }
    const uint32_t channel = worker_offset(vc) + direction_compact_index(producer, facing_);
    return channel < sender_count(vc) ? std::optional{channel} : std::nullopt;
}

ttsl::SmallVector<RouterProducerSlots::Slot, 4> RouterProducerSlots::producer_slots(uint32_t vc) const {
    ttsl::SmallVector<Slot, 4> slots;
    if (vc >= 2) {
        return slots;
    }
    for (uint32_t compact = 0; compact < 4; ++compact) {
        const uint32_t channel = worker_offset(vc) + compact;
        if (channel >= sender_count(vc)) {
            break;
        }
        slots.push_back(Slot{channel, direction_from_compact_index(facing_, compact)});
    }
    return slots;
}

std::pair<eth_chan_directions, eth_chan_directions> get_perpendicular_directions(eth_chan_directions direction) {
    if (direction == eth_chan_directions::Z) {
        TT_FATAL(false, "Internal error: In get_perpendicular_directions, Z direction is not supported");
    }
    if (direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST) {
        // E/W -> perpendicular are N/S
        return {eth_chan_directions::NORTH, eth_chan_directions::SOUTH};
    }  // N/S -> perpendicular are E/W
    return {eth_chan_directions::EAST, eth_chan_directions::WEST};
}

std::vector<eth_chan_directions> get_all_other_directions(eth_chan_directions direction, bool exclude_z) {
    std::vector<eth_chan_directions> all_directions = {
        eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH};

    // Only include Z direction if not excluded (Z not supported in some modes like MUX)
    if (!exclude_z) {
        all_directions.push_back(eth_chan_directions::Z);
    }

    std::vector<eth_chan_directions> dirs;
    dirs.reserve(all_directions.size());
    for (auto dir : all_directions) {
        if (dir != direction) {
            dirs.push_back(dir);
        }
    }

    return dirs;
}

}  // namespace tt::tt_fabric::builder
