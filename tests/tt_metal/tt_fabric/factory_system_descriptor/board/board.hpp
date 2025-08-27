// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <umd/device/types/cluster_descriptor_types.h>

// Forward declaration and hash specialization for AsicChannel
namespace tt::tt_fabric {
struct AsicChannel;
}

namespace std {
template <>
struct hash<tt::tt_fabric::AsicChannel> {
    std::size_t operator()(const tt::tt_fabric::AsicChannel& asic_channel) const;
};
}  // namespace std

namespace tt::tt_fabric {

enum class PortType {
    TRACE,
    QSFP,  // TODO: Should distinguish between QSFP types?
    WARP100,
    WARP400,
    LINKING_BOARD_1,
    LINKING_BOARD_2,
    LINKING_BOARD_3,
};

struct AsicChannel {
    // TODO: This might become asic_location, or get mapped from it
    uint32_t asic_index;
    uint32_t channel_id;
};

inline bool operator==(const AsicChannel& lhs, const AsicChannel& rhs) {
    return lhs.asic_index == rhs.asic_index && lhs.channel_id == rhs.channel_id;
}

struct Port {
    PortType port_type;
    uint32_t port_id;
};

class Board {
public:
    // Constructor that takes maps directly
    Board(
        const std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>>& ports,
        const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& internal_connections,
        const tt::umd::BoardType& board_type);

    const tt::umd::BoardType& get_board_type() const;

    // Get available port IDs for a specific port type
    const std::vector<uint32_t>& get_available_port_ids(PortType port_type) const;

    // Get channels for a specific port
    const std::vector<AsicChannel>& get_port_channels(PortType port_type, uint32_t port_id) const;

    // Get internal connections for this board
    const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& get_internal_connections() const;

    // Get port for a specific asic_channel
    const Port& get_port_for_asic_channel(const AsicChannel& asic_channel) const;

    // Mark a port as used (remove from available list)
    void mark_port_used(PortType port_type, uint32_t port_id);

protected:
    // Unconnected ports
    std::unordered_map<PortType, std::vector<uint32_t>> available_port_ids_;
    // All port to channel mappings
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports_;
    // Internal connections between ports
    // Note: Internal connections currently just use dummy trace ports
    // Could switch to just direct channel mapping in the future
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections_;
    // Map from asic_channel to port
    std::unordered_map<AsicChannel, Port> asic_to_port_map_;
    // Board type
    tt::umd::BoardType board_type_;

    std::unordered_set<uint32_t> asic_indices_;
};

Board create_board(const std::string& board_name);

}  // namespace tt::tt_fabric
