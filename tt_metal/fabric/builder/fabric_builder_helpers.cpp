// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    using eth_chan_directions::NORTH, eth_chan_directions::SOUTH, eth_chan_directions::EAST, eth_chan_directions::WEST,
        eth_chan_directions::COUNT;
    static constexpr std::array<eth_chan_directions, COUNT> east_channels = {COUNT, WEST, NORTH, SOUTH};
    static constexpr std::array<eth_chan_directions, COUNT> west_channels = {COUNT, EAST, NORTH, SOUTH};
    static constexpr std::array<eth_chan_directions, COUNT> north_channels = {COUNT, EAST, WEST, SOUTH};
    static constexpr std::array<eth_chan_directions, COUNT> south_channels = {COUNT, EAST, WEST, NORTH};

    TT_FATAL(
        sender_channel_index < COUNT,
        "Internal error: In get_sender_channel_direction, sender channel index out of bounds. Got index {}",
        sender_channel_index);
    TT_FATAL(
        sender_channel_index > 0,
        "Internal error: In get_sender_channel_direction, sender channel index must be greater than 0. Got index {}",
        sender_channel_index);
    switch (my_direction) {
        case EAST: return east_channels[static_cast<size_t>(sender_channel_index)];
        case WEST: return west_channels[static_cast<size_t>(sender_channel_index)];
        case NORTH: return north_channels[static_cast<size_t>(sender_channel_index)];
        case SOUTH: return south_channels[static_cast<size_t>(sender_channel_index)];
        default: TT_FATAL(false, "Internal error: In get_sender_channel_direction, invalid direction");
    }
}

std::pair<eth_chan_directions, eth_chan_directions> get_perpendicular_directions(eth_chan_directions direction) {
    if (direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST) {
        // E/W -> perpendicular are N/S
        return {eth_chan_directions::NORTH, eth_chan_directions::SOUTH};
    }  // N/S -> perpendicular are E/W
    return {eth_chan_directions::EAST, eth_chan_directions::WEST};
}

std::vector<eth_chan_directions> get_all_other_directions(eth_chan_directions direction) {
    std::vector<eth_chan_directions> all_directions = {
        eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH};

    std::vector<eth_chan_directions> dirs;
    for (auto dir : all_directions) {
        if (dir != direction) {
            dirs.push_back(dir);
        }
    }

    return dirs;
}

}  // namespace tt::tt_fabric::builder
