// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt_stl/strong_type.hpp>

// Forward declaration and hash specialization for AsicChannel
namespace tt::scaleout_tools {
struct AsicChannel;
}

namespace std {
template <>
struct hash<tt::scaleout_tools::AsicChannel> {
    std::size_t operator()(const tt::scaleout_tools::AsicChannel& asic_channel) const;
};
}  // namespace std

namespace tt::scaleout_tools {

// Strong types for port and channel identification
using PortId = ttsl::StrongType<uint32_t, struct PortIdTag>;
using ChanId = ttsl::StrongType<uint32_t, struct ChanIdTag>;

enum class PortType {
    TRACE,
    QSFP_DD,
    WARP100,
    WARP400,
    LINKING_BOARD_1,
    LINKING_BOARD_2,
    LINKING_BOARD_3,
};

struct AsicChannel {
    uint32_t asic_location = 0;
    ChanId channel_id{0};

    auto operator<=>(const AsicChannel& other) const = default;
};

std::ostream& operator<<(std::ostream& os, const AsicChannel& asic_channel);

struct Port {
    PortType port_type = PortType::TRACE;
    PortId port_id{0};
};

class Board {
public:
    // Constructor that takes maps directly
    Board(
        const std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
        const std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& internal_connections,
        const tt::umd::BoardType& board_type);

    // Constructor that takes a pair of ports and connections
    Board(
        const std::pair<
            std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
            std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>& ports_and_connections,
        const tt::umd::BoardType& board_type);

    tt::ARCH get_arch() const;

    tt::umd::BoardType get_board_type() const;

    // Get available port IDs for a specific port type
    const std::vector<PortId>& get_available_port_ids(PortType port_type) const;

    // Get channels for a specific port
    const std::vector<AsicChannel>& get_port_channels(PortType port_type, PortId port_id) const;

    // Get internal connections for this board
    const std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& get_internal_connections() const;

    // Get port for a specific asic_channel
    const Port& get_port_for_asic_channel(const AsicChannel& asic_channel) const;

    // Mark a port as used (remove from available list)
    void mark_port_used(PortType port_type, PortId port_id);

protected:
    // Unconnected ports
    std::unordered_map<PortType, std::vector<PortId>> available_port_ids_;
    // All port to channel mappings
    std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports_;
    // Internal connections between ports
    // Note: Internal connections currently just use dummy trace ports
    // Could switch to just direct channel mapping in the future
    std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections_;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    std::unordered_map<AsicChannel, Port> asic_to_port_map_;
    tt::umd::BoardType board_type_ = tt::umd::BoardType::UNKNOWN;
    std::unordered_set<uint32_t> asic_locations_;
};

Board create_board(tt::umd::BoardType board_type);

tt::umd::BoardType get_board_type_from_string(const std::string& board_name);

}  // namespace tt::scaleout_tools
