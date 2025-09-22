// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "board.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::scaleout_tools {

namespace {
// Helper function to add a port with ASIC channels
void add_port(
    std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
    PortType port_type,
    PortId port_id,
    const std::vector<AsicChannel>& channels) {
    ports[port_type][port_id] = channels;
}

// Helper function to add ports with sequential channels
void add_sequential_port(
    std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
    PortType port_type,
    PortId port_id,
    uint32_t asic_id,
    uint32_t start_channel,
    uint32_t end_channel) {
    uint32_t num_channels = end_channel - start_channel + 1;
    std::vector<AsicChannel> channels;
    channels.reserve(num_channels);
    for (uint32_t i = start_channel; i <= end_channel; ++i) {
        channels.push_back({asic_id, ChanId(i)});
    }
    add_port(ports, port_type, port_id, channels);
}

// Helper function to add sequential port pair connections (1-2, 3-4, 5-6, etc.)
// Automatically determines number of pairs from the ports map
void add_sequential_port_pairs(
    std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& internal_connections,
    PortType port_type,
    const std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports) {
    auto it = ports.find(port_type);
    if (it == ports.end()) {
        return;
    }

    uint32_t total_ports = it->second.size();
    if (total_ports % 2 != 0) {
        throw std::runtime_error("Number of ports must be even");
    }
    uint32_t num_pairs = total_ports / 2;
    for (uint32_t i = 0; i < num_pairs; ++i) {
        uint32_t port_a = i * 2 + 1;
        uint32_t port_b = i * 2 + 2;
        internal_connections[port_type].push_back({PortId(port_a), PortId(port_b)});
    }
}

// Helper function to create internal connections automatically from ports
std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_internal_connections_from_ports(
    const std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
    PortType port_type) {
    std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections;
    add_sequential_port_pairs(internal_connections, port_type, ports);
    return internal_connections;
}

// Helper function to create both ports and internal connections efficiently
std::pair<
    std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
    std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>
create_ports_and_connections(
    const std::function<std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>()>&
        ports_func,
    PortType connection_port_type) {
    auto ports = ports_func();
    auto internal_connections = create_internal_connections_from_ports(ports, connection_port_type);
    return {ports, internal_connections};
}

}  // anonymous namespace

Board::Board(
    const std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
    const std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& internal_connections,
    const tt::umd::BoardType& board_type) :
    ports_(ports), internal_connections_(internal_connections), board_type_(board_type), asic_locations_() {
    switch (board_type_) {
        case tt::umd::BoardType::N150:
        case tt::umd::BoardType::N300:
        case tt::umd::BoardType::UBB_WORMHOLE: arch_ = tt::ARCH::WORMHOLE_B0; break;
        case tt::umd::BoardType::P100:
        case tt::umd::BoardType::P150:
        case tt::umd::BoardType::P300:
        case tt::umd::BoardType::UBB_BLACKHOLE: arch_ = tt::ARCH::BLACKHOLE; break;
        default: throw std::runtime_error("Invalid board type");
    }
    // Initialize available_ports from ports
    for (const auto& [port_type, port_mapping] : ports) {
        auto& available_ports = available_port_ids_[port_type];
        for (const auto& [port_id, asic_channels] : port_mapping) {
            for (const auto& asic_channel : asic_channels) {
                asic_locations_.insert(asic_channel.asic_location);
                asic_to_port_map_[asic_channel] = Port{port_type, port_id};
            }
            available_ports.push_back(port_id);
        }
    }

    // Mark ports as used for internal connections
    for (const auto& [port_type, connections] : internal_connections) {
        for (const auto& connection : connections) {
            mark_port_used(port_type, connection.first);
            mark_port_used(port_type, connection.second);
        }
    }
    // Currently UBB has a different definition of board in this representation compared to UMD.
    // TODO: This exception shouldn't live here.
    uint32_t expected_asic_indices = 0;
    if (board_type_ == tt::umd::BoardType::UBB_WORMHOLE || board_type_ == tt::umd::BoardType::UBB_BLACKHOLE) {
        expected_asic_indices = 8;
    } else {
        expected_asic_indices = tt::umd::get_number_of_chips_from_board_type(board_type_);
    }
    if (asic_locations_.size() != expected_asic_indices) {
        throw std::runtime_error("Number of ASICs in board configuration does not match number of chips in board type");
    }
}

Board::Board(
    const std::pair<
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>& ports_and_connections,
    const tt::umd::BoardType& board_type) :
    Board(ports_and_connections.first, ports_and_connections.second, board_type) {}

tt::ARCH Board::get_arch() const { return arch_; }

tt::umd::BoardType Board::get_board_type() const { return board_type_; }

const std::vector<PortId>& Board::get_available_port_ids(PortType port_type) const {
    auto it = available_port_ids_.find(port_type);
    if (it == available_port_ids_.end()) {
        throw std::runtime_error("Port type not found in board configuration");
    }
    return it->second;
}

const std::vector<AsicChannel>& Board::get_port_channels(PortType port_type, PortId port_id) const {
    auto port_type_it = ports_.find(port_type);
    if (port_type_it == ports_.end()) {
        throw std::runtime_error("Port type not found");
    }

    auto port_it = port_type_it->second.find(port_id);
    if (port_it == port_type_it->second.end()) {
        throw std::runtime_error("Port ID not found");
    }

    return port_it->second;
}

const Port& Board::get_port_for_asic_channel(const AsicChannel& asic_channel) const {
    auto it = asic_to_port_map_.find(asic_channel);
    if (it == asic_to_port_map_.end()) {
        throw std::runtime_error("Asic channel not found");
    }
    return it->second;
}

void Board::mark_port_used(PortType port_type, PortId port_id) {
    auto& available_ports = available_port_ids_[port_type];
    auto it = std::find(available_ports.begin(), available_ports.end(), port_id);
    if (it != available_ports.end()) {
        available_ports.erase(it);
    }
}

// Get internal connections for this board
const std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& Board::get_internal_connections() const {
    return internal_connections_;
}

// N150 board class
class N150 : public Board {
public:
    N150() : Board(create_n150_ports(), create_n150_internal_connections(), tt::umd::BoardType::N150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_n150_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP_DD ports (400G)
        add_sequential_port(ports, PortType::QSFP_DD, PortId(1), 0, 6, 7);  // ASIC 0, channels 6,7
        add_sequential_port(ports, PortType::QSFP_DD, PortId(2), 0, 0, 1);  // ASIC 0, channels 0,1

        // WARP100 ports
        add_sequential_port(ports, PortType::WARP100, PortId(1), 0, 14, 15);  // ASIC 0, channels 14,15

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_n150_internal_connections() {
        // No internal connections for N150
        return std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>();
    }
};

// N300 board class
class N300 : public Board {
public:
    N300() : Board(create_n300_ports_and_connections(), tt::umd::BoardType::N300) {}

private:
    static std::pair<
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>
    create_n300_ports_and_connections() {
        auto ports = create_n300_ports();
        auto internal_connections = create_internal_connections_from_ports(ports, PortType::TRACE);
        return {ports, internal_connections};
    }
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_n300_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP_DD ports (400G)
        add_sequential_port(ports, PortType::QSFP_DD, PortId(1), 0, 6, 7);  // ASIC 0, channels 6,7
        add_sequential_port(ports, PortType::QSFP_DD, PortId(2), 0, 0, 1);  // ASIC 0, channels 0,1

        // WARP100 ports
        add_sequential_port(ports, PortType::WARP100, PortId(1), 0, 14, 15);  // ASIC 0, channels 14,15
        add_sequential_port(ports, PortType::WARP100, PortId(2), 1, 6, 7);    // ASIC 1, channels 6,7

        // TRACE ports
        add_sequential_port(ports, PortType::TRACE, PortId(1), 0, 8, 9);  // ASIC 0, channels 8,9
        add_sequential_port(ports, PortType::TRACE, PortId(2), 1, 0, 1);  // ASIC 1, channels 0,1

        return ports;
    }
};

// UBB_WORMHOLE board class
class UBB_WORMHOLE : public Board {
public:
    UBB_WORMHOLE() : Board(create_ubb_wormhole_ports_and_connections(), tt::umd::BoardType::UBB_WORMHOLE) {}

private:
    static std::pair<
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>
    create_ubb_wormhole_ports_and_connections() {
        auto ports = create_ubb_wormhole_ports();
        auto internal_connections = create_internal_connections_from_ports(ports, PortType::TRACE);
        return {ports, internal_connections};
    }
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>
    create_ubb_wormhole_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP_DD ports (400G)
        add_sequential_port(ports, PortType::QSFP_DD, PortId(1), 5, 4, 7);  // ASIC 5, channels 4,5,6,7
        add_sequential_port(ports, PortType::QSFP_DD, PortId(2), 1, 4, 7);  // ASIC 1, channels 4,5,6,7
        add_sequential_port(ports, PortType::QSFP_DD, PortId(3), 1, 0, 3);  // ASIC 1, channels 0,1,2,3
        add_sequential_port(ports, PortType::QSFP_DD, PortId(4), 2, 0, 3);  // ASIC 2, channels 0,1,2,3
        add_sequential_port(ports, PortType::QSFP_DD, PortId(5), 3, 0, 3);  // ASIC 3, channels 0,1,2,3
        add_sequential_port(ports, PortType::QSFP_DD, PortId(6), 4, 0, 3);  // ASIC 4, channels 0,1,2,3

        // LINKING_BOARD_1 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_1, PortId(1), 5, 12, 15);  // ASIC 5, channels 12,13,14,15
        add_sequential_port(ports, PortType::LINKING_BOARD_1, PortId(2), 6, 12, 15);  // ASIC 6, channels 12,13,14,15

        // LINKING_BOARD_2 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_2, PortId(1), 7, 12, 15);  // ASIC 7, channels 12,13,14,15
        add_sequential_port(ports, PortType::LINKING_BOARD_2, PortId(2), 8, 12, 15);  // ASIC 8, channels 12,13,14,15

        // LINKING_BOARD_3 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_3, PortId(1), 8, 8, 11);  // ASIC 8, channels 8,9,10,11
        add_sequential_port(ports, PortType::LINKING_BOARD_3, PortId(2), 4, 8, 11);  // ASIC 4, channels 8,9,10,11

        // TRACE ports
        add_sequential_port(ports, PortType::TRACE, PortId(1), 5, 0, 3);     // ASIC 5, channels 0,1,2,3
        add_sequential_port(ports, PortType::TRACE, PortId(2), 1, 12, 15);   // ASIC 1, channels 12,13,14,15
        add_sequential_port(ports, PortType::TRACE, PortId(3), 5, 8, 11);    // ASIC 5, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(4), 6, 4, 7);     // ASIC 6, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(5), 1, 8, 11);    // ASIC 1, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(6), 2, 4, 7);     // ASIC 2, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(7), 6, 0, 3);     // ASIC 6, channels 0,1,2,3
        add_sequential_port(ports, PortType::TRACE, PortId(8), 2, 12, 15);   // ASIC 2, channels 12,13,14,15
        add_sequential_port(ports, PortType::TRACE, PortId(9), 6, 8, 11);    // ASIC 6, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(10), 7, 4, 7);    // ASIC 7, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(11), 2, 8, 11);   // ASIC 2, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(12), 3, 4, 7);    // ASIC 3, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(13), 7, 0, 3);    // ASIC 7, channels 0,1,2,3
        add_sequential_port(ports, PortType::TRACE, PortId(14), 3, 12, 15);  // ASIC 3, channels 12,13,14,15
        add_sequential_port(ports, PortType::TRACE, PortId(15), 7, 8, 11);   // ASIC 7, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(16), 8, 4, 7);    // ASIC 8, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(17), 3, 8, 11);   // ASIC 3, channels 8,9,10,11
        add_sequential_port(ports, PortType::TRACE, PortId(18), 4, 4, 7);    // ASIC 4, channels 4,5,6,7
        add_sequential_port(ports, PortType::TRACE, PortId(19), 8, 0, 3);    // ASIC 8, channels 0,1,2,3
        add_sequential_port(ports, PortType::TRACE, PortId(20), 4, 12, 15);  // ASIC 4, channels 12,13,14,15

        return ports;
    }
};

// P150 board class
class P150 : public Board {
public:
    P150() : Board(create_p150_ports(), create_p150_internal_connections(), tt::umd::BoardType::P150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_p150_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP_DD ports (800G)
        add_port(ports, PortType::QSFP_DD, PortId(1), {{0, ChanId(9)}, {0, ChanId(11)}});  // ASIC 0, channels 9,11
        add_port(ports, PortType::QSFP_DD, PortId(2), {{0, ChanId(8)}, {0, ChanId(10)}});  // ASIC 0, channels 8,10
        add_port(ports, PortType::QSFP_DD, PortId(3), {{0, ChanId(5)}, {0, ChanId(7)}});   // ASIC 0, channels 5,7
        add_port(ports, PortType::QSFP_DD, PortId(4), {{0, ChanId(4)}, {0, ChanId(6)}});   // ASIC 0, channels 4,6

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_p150_internal_connections() {
        // No internal connections for P150
        return std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>();
    }
};

// P300 board class
class P300 : public Board {
public:
    P300() : Board(create_p300_ports_and_connections(), tt::umd::BoardType::P300) {}

private:
    static std::pair<
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>
    create_p300_ports_and_connections() {
        auto ports = create_p300_ports();
        auto internal_connections = create_internal_connections_from_ports(ports, PortType::TRACE);
        return {ports, internal_connections};
    }
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_p300_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // WARP400 ports
        add_port(
            ports,
            PortType::WARP400,
            PortId(1),
            {{1, ChanId(3)},
             {1, ChanId(6)},
             {0, ChanId(7)},
             {0, ChanId(9)}});  // ASIC 1: channels 3,6; ASIC 0: channels 7,9
        add_port(
            ports,
            PortType::WARP400,
            PortId(2),
            {{1, ChanId(2)},
             {1, ChanId(4)},
             {0, ChanId(5)},
             {0, ChanId(4)}});  // ASIC 1: channels 2,4; ASIC 0: channels 5,4

        // TRACE ports
        add_sequential_port(ports, PortType::TRACE, PortId(1), 1, 8, 9);  // ASIC 1, channels 8,9
        add_sequential_port(ports, PortType::TRACE, PortId(2), 0, 3, 4);  // ASIC 0, channels 3,4

        return ports;
    }
};

// UBB_BLACKHOLE board class
class UBB_BLACKHOLE : public Board {
public:
    UBB_BLACKHOLE() : Board(create_ubb_blackhole_ports_and_connections(), tt::umd::BoardType::UBB_BLACKHOLE) {}

private:
    static std::pair<
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>,
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>>
    create_ubb_blackhole_ports_and_connections() {
        auto ports = create_ubb_blackhole_ports();
        auto internal_connections = create_internal_connections_from_ports(ports, PortType::TRACE);
        return {ports, internal_connections};
    }
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>
    create_ubb_blackhole_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP_DD ports (800G)
        add_sequential_port(ports, PortType::QSFP_DD, PortId(1), 5, 2, 3);                 // ASIC 5, channels 2,3
        add_sequential_port(ports, PortType::QSFP_DD, PortId(2), 1, 2, 3);                 // ASIC 1, channels 2,3
        add_sequential_port(ports, PortType::QSFP_DD, PortId(3), 1, 0, 1);                 // ASIC 1, channels 0,1
        add_sequential_port(ports, PortType::QSFP_DD, PortId(4), 2, 0, 1);                 // ASIC 2, channels 0,1
        add_sequential_port(ports, PortType::QSFP_DD, PortId(5), 3, 0, 1);                 // ASIC 3, channels 0,1
        add_sequential_port(ports, PortType::QSFP_DD, PortId(6), 4, 0, 1);                 // ASIC 4, channels 0,1
        add_port(ports, PortType::QSFP_DD, PortId(7), {{1, ChanId(8)}, {2, ChanId(8)}});   // ASIC 1,2: channel 8
        add_port(ports, PortType::QSFP_DD, PortId(8), {{5, ChanId(8)}, {6, ChanId(8)}});   // ASIC 5,6: channel 8
        add_port(ports, PortType::QSFP_DD, PortId(9), {{3, ChanId(8)}, {4, ChanId(8)}});   // ASIC 3,4: channel 8
        add_port(ports, PortType::QSFP_DD, PortId(10), {{7, ChanId(8)}, {8, ChanId(8)}});  // ASIC 7,8: channel 8
        add_port(ports, PortType::QSFP_DD, PortId(11), {{1, ChanId(9)}, {2, ChanId(9)}});  // ASIC 1,2: channel 9
        add_port(ports, PortType::QSFP_DD, PortId(12), {{5, ChanId(9)}, {6, ChanId(9)}});  // ASIC 5,6: channel 9
        add_port(ports, PortType::QSFP_DD, PortId(13), {{3, ChanId(9)}, {4, ChanId(9)}});  // ASIC 3,4: channel 9
        add_port(ports, PortType::QSFP_DD, PortId(14), {{7, ChanId(9)}, {8, ChanId(9)}});  // ASIC 7,8: channel 9

        // NOTE: These Linking Board connections are not finalized yet.
        log_warning(tt::LogDistributed, "UBB_BLACKHOLE: Linking Board connections are not finalized yet.");

        // LINKING_BOARD_1 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_1, PortId(1), 5, 6, 7);  // ASIC 5, channels 6,7
        add_sequential_port(ports, PortType::LINKING_BOARD_1, PortId(2), 6, 6, 7);  // ASIC 6, channels 6,7

        // LINKING_BOARD_2 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_2, PortId(1), 7, 6, 7);  // ASIC 7, channels 6,7
        add_sequential_port(ports, PortType::LINKING_BOARD_2, PortId(2), 8, 6, 7);  // ASIC 8, channels 6,7

        // LINKING_BOARD_3 ports
        add_sequential_port(ports, PortType::LINKING_BOARD_3, PortId(1), 8, 4, 5);  // ASIC 8, channels 4,5
        add_sequential_port(ports, PortType::LINKING_BOARD_3, PortId(2), 4, 4, 5);  // ASIC 4, channels 4,5

        // TRACE ports
        add_sequential_port(ports, PortType::TRACE, PortId(1), 5, 0, 1);   // ASIC 5, channels 0,1
        add_sequential_port(ports, PortType::TRACE, PortId(2), 1, 6, 7);   // ASIC 1, channels 6,7
        add_sequential_port(ports, PortType::TRACE, PortId(3), 5, 4, 5);   // ASIC 5, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(4), 6, 2, 3);   // ASIC 6, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(5), 1, 4, 5);   // ASIC 1, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(6), 2, 2, 3);   // ASIC 2, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(7), 6, 0, 1);   // ASIC 6, channels 0,1
        add_sequential_port(ports, PortType::TRACE, PortId(8), 2, 6, 7);   // ASIC 2, channels 6,7
        add_sequential_port(ports, PortType::TRACE, PortId(9), 6, 4, 5);   // ASIC 6, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(10), 7, 2, 3);  // ASIC 7, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(11), 2, 4, 5);  // ASIC 2, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(12), 3, 2, 3);  // ASIC 3, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(13), 7, 0, 1);  // ASIC 7, channels 0,1
        add_sequential_port(ports, PortType::TRACE, PortId(14), 3, 6, 7);  // ASIC 3, channels 6,7
        add_sequential_port(ports, PortType::TRACE, PortId(15), 7, 4, 5);  // ASIC 7, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(16), 8, 2, 3);  // ASIC 8, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(17), 3, 4, 5);  // ASIC 3, channels 4,5
        add_sequential_port(ports, PortType::TRACE, PortId(18), 4, 2, 3);  // ASIC 4, channels 2,3
        add_sequential_port(ports, PortType::TRACE, PortId(19), 8, 0, 1);  // ASIC 8, channels 0,1
        add_sequential_port(ports, PortType::TRACE, PortId(20), 4, 6, 7);  // ASIC 4, channels 6,7

        return ports;
    }
};

// Factory function to create boards by type (for backward compatibility)
Board create_board(tt::umd::BoardType board_type) {
    switch (board_type) {
        case BoardType::N150: return N150();
        case BoardType::N300: return N300();
        case BoardType::UBB_WORMHOLE: return UBB_WORMHOLE();
        case BoardType::P150: return P150();
        case BoardType::P300: return P300();
        case BoardType::UBB_BLACKHOLE: return UBB_BLACKHOLE();
        default: throw std::runtime_error("Unknown board type: " + std::string(enchantum::to_string(board_type)));
    }
}

tt::umd::BoardType get_board_type_from_string(const std::string& board_name) {
    auto board_type = enchantum::cast<tt::umd::BoardType>(board_name, ttsl::ascii_caseless_comp);
    if (!board_type.has_value()) {
        throw std::runtime_error("Invalid board type: " + std::string(board_name));
    }
    return *board_type;
}

std::ostream& operator<<(std::ostream& os, const AsicChannel& asic_channel) {
    os << "AsicChannel{asic_location=" << asic_channel.asic_location << ", channel_id=" << *asic_channel.channel_id
       << "}";
    return os;
}

}  // namespace tt::scaleout_tools

// Hash function implementation for AsicChannel
namespace std {
std::size_t hash<tt::scaleout_tools::AsicChannel>::operator()(
    const tt::scaleout_tools::AsicChannel& asic_channel) const {
    return tt::stl::hash::hash_objects_with_default_seed(asic_channel.asic_location, asic_channel.channel_id);
}
}  // namespace std
