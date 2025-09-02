// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "board.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <tt_stl/caseless_comparison.hpp>
#include <umd/device/types/cluster_descriptor_types.h>
#include <tt_stl/reflection.hpp>

namespace tt::scaleout_tools {

Board::Board(
    const std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>>& ports,
    const std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>& internal_connections,
    const tt::umd::BoardType& board_type) :
    ports_(ports), internal_connections_(internal_connections), board_type_(board_type), asic_locations_() {
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
    if (board_type_ == tt::umd::BoardType::UBB) {
        expected_asic_indices = 8;
    } else {
        expected_asic_indices = tt::umd::get_number_of_chips_from_board_type(board_type_);
    }
    if (asic_locations_.size() != expected_asic_indices) {
        throw std::runtime_error("Number of ASICs in board configuration does not match number of chips in board type");
    }
}

const tt::umd::BoardType& Board::get_board_type() const { return board_type_; }

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

// N300 board class
class N300 : public Board {
public:
    N300() : Board(create_n300_ports(), create_n300_internal_connections(), tt::umd::BoardType::N300) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_n300_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {PortId(1), {{0, ChanId(6)}, {0, ChanId(7)}}},  // ASIC 0, channels 6,7
            {PortId(2), {{0, ChanId(0)}, {0, ChanId(1)}}},  // ASIC 0, channels 0,1
        };

        // WARP100 ports
        auto& warp100_ports = ports[PortType::WARP100];
        warp100_ports = {
            {PortId(1), {{0, ChanId(14)}, {0, ChanId(15)}}},  // ASIC 0, channels 14,15
            {PortId(2), {{1, ChanId(6)}, {1, ChanId(7)}}},    // ASIC 1, channels 6,7
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {PortId(1), {{0, ChanId(8)}, {0, ChanId(9)}}},  // ASIC 0, channels 8,9
            {PortId(2), {{1, ChanId(0)}, {1, ChanId(1)}}},  // ASIC 1, channels 0,1
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_n300_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections;
        internal_connections[PortType::TRACE] = {{PortId(1), PortId(2)}};  // TRACE connection between ports 1 and 2
        return internal_connections;
    }
};

// WH_UBB board class
class WH_UBB : public Board {
public:
    WH_UBB() : Board(create_wh_ubb_ports(), create_wh_ubb_internal_connections(), tt::umd::BoardType::UBB) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_wh_ubb_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {PortId(1), {{5, ChanId(4)}, {5, ChanId(5)}, {5, ChanId(6)}, {5, ChanId(7)}}},  // ASIC 5, channels 4,5,6,7
            {PortId(2), {{1, ChanId(4)}, {1, ChanId(5)}, {1, ChanId(6)}, {1, ChanId(7)}}},  // ASIC 1, channels 4,5,6,7
            {PortId(3), {{1, ChanId(0)}, {1, ChanId(1)}, {1, ChanId(2)}, {1, ChanId(3)}}},  // ASIC 1, channels 0,1,2,3
            {PortId(4), {{2, ChanId(0)}, {2, ChanId(1)}, {2, ChanId(2)}, {2, ChanId(3)}}},  // ASIC 2, channels 0,1,2,3
            {PortId(5), {{3, ChanId(0)}, {3, ChanId(1)}, {3, ChanId(2)}, {3, ChanId(3)}}},  // ASIC 3, channels 0,1,2,3
            {PortId(6), {{4, ChanId(0)}, {4, ChanId(1)}, {4, ChanId(2)}, {4, ChanId(3)}}},  // ASIC 4, channels 0,1,2,3
        };

        // LINKING_BOARD_1 ports
        auto& linking_board_1_ports = ports[PortType::LINKING_BOARD_1];
        linking_board_1_ports = {
            {PortId(1),
             {{5, ChanId(12)}, {5, ChanId(13)}, {5, ChanId(14)}, {5, ChanId(15)}}},  // ASIC 5, channels 12,13,14,15
            {PortId(2),
             {{6, ChanId(12)}, {6, ChanId(13)}, {6, ChanId(14)}, {6, ChanId(15)}}},  // ASIC 6, channels 12,13,14,15
        };

        // LINKING_BOARD_2 ports
        auto& linking_board_2_ports = ports[PortType::LINKING_BOARD_2];
        linking_board_2_ports = {
            {PortId(1),
             {{7, ChanId(12)}, {7, ChanId(13)}, {7, ChanId(14)}, {7, ChanId(15)}}},  // ASIC 7, channels 12,13,14,15
            {PortId(2),
             {{8, ChanId(12)}, {8, ChanId(13)}, {8, ChanId(14)}, {8, ChanId(15)}}},  // ASIC 8, channels 12,13,14,15
        };

        // LINKING_BOARD_3 ports
        auto& linking_board_3_ports = ports[PortType::LINKING_BOARD_3];
        linking_board_3_ports = {
            {PortId(1),
             {{8, ChanId(8)}, {8, ChanId(9)}, {8, ChanId(10)}, {8, ChanId(11)}}},  // ASIC 8, channels 8,9,10,11
            {PortId(2),
             {{4, ChanId(8)}, {4, ChanId(9)}, {4, ChanId(10)}, {4, ChanId(11)}}},  // ASIC 4, channels 8,9,10,11
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {PortId(1), {{5, ChanId(0)}, {5, ChanId(1)}, {5, ChanId(2)}, {5, ChanId(3)}}},  // ASIC 5, channels 0,1,2,3
            {PortId(2),
             {{1, ChanId(12)}, {1, ChanId(13)}, {1, ChanId(14)}, {1, ChanId(15)}}},  // ASIC 1, channels 12,13,14,15
            {PortId(3),
             {{5, ChanId(8)}, {5, ChanId(9)}, {5, ChanId(10)}, {5, ChanId(11)}}},  // ASIC 5, channels 8,9,10,11
            {PortId(4), {{6, ChanId(4)}, {6, ChanId(5)}, {6, ChanId(6)}, {6, ChanId(7)}}},  // ASIC 6, channels 4,5,6,7
            {PortId(5),
             {{1, ChanId(8)}, {1, ChanId(9)}, {1, ChanId(10)}, {1, ChanId(11)}}},  // ASIC 1, channels 8,9,10,11
            {PortId(6), {{2, ChanId(4)}, {2, ChanId(5)}, {2, ChanId(6)}, {2, ChanId(7)}}},  // ASIC 2, channels 4,5,6,7
            {PortId(7), {{6, ChanId(0)}, {6, ChanId(1)}, {6, ChanId(2)}, {6, ChanId(3)}}},  // ASIC 6, channels 0,1,2,3
            {PortId(8),
             {{2, ChanId(12)}, {2, ChanId(13)}, {2, ChanId(14)}, {2, ChanId(15)}}},  // ASIC 2, channels 12,13,14,15
            {PortId(9),
             {{6, ChanId(8)}, {6, ChanId(9)}, {6, ChanId(10)}, {6, ChanId(11)}}},  // ASIC 6, channels 8,9,10,11
            {PortId(10), {{7, ChanId(4)}, {7, ChanId(5)}, {7, ChanId(6)}, {7, ChanId(7)}}},  // ASIC 7, channels 4,5,6,7
            {PortId(11),
             {{2, ChanId(8)}, {2, ChanId(9)}, {2, ChanId(10)}, {2, ChanId(11)}}},  // ASIC 2, channels 8,9,10,11
            {PortId(12), {{3, ChanId(4)}, {3, ChanId(5)}, {3, ChanId(6)}, {3, ChanId(7)}}},  // ASIC 3, channels 4,5,6,7
            {PortId(13), {{7, ChanId(0)}, {7, ChanId(1)}, {7, ChanId(2)}, {7, ChanId(3)}}},  // ASIC 7, channels 0,1,2,3
            {PortId(14),
             {{3, ChanId(12)}, {3, ChanId(13)}, {3, ChanId(14)}, {3, ChanId(15)}}},  // ASIC 3, channels 12,13,14,15
            {PortId(15),
             {{7, ChanId(8)}, {7, ChanId(9)}, {7, ChanId(10)}, {7, ChanId(11)}}},  // ASIC 7, channels 8,9,10,11
            {PortId(16), {{8, ChanId(4)}, {8, ChanId(5)}, {8, ChanId(6)}, {8, ChanId(7)}}},  // ASIC 8, channels 4,5,6,7
            {PortId(17),
             {{3, ChanId(8)}, {3, ChanId(9)}, {3, ChanId(10)}, {3, ChanId(11)}}},  // ASIC 3, channels 8,9,10,11
            {PortId(18), {{4, ChanId(4)}, {4, ChanId(5)}, {4, ChanId(6)}, {4, ChanId(7)}}},  // ASIC 4, channels 4,5,6,7
            {PortId(19), {{8, ChanId(0)}, {8, ChanId(1)}, {8, ChanId(2)}, {8, ChanId(3)}}},  // ASIC 8, channels 0,1,2,3
            {PortId(20),
             {{4, ChanId(12)}, {4, ChanId(13)}, {4, ChanId(14)}, {4, ChanId(15)}}},  // ASIC 4, channels 12,13,14,15
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_wh_ubb_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections;
        internal_connections[PortType::TRACE] = {
            {PortId(1), PortId(2)},    // TRACE connection between ports 1 and 2
            {PortId(3), PortId(4)},    // TRACE connection between ports 3 and 4
            {PortId(5), PortId(6)},    // TRACE connection between ports 5 and 6
            {PortId(7), PortId(8)},    // TRACE connection between ports 7 and 8
            {PortId(9), PortId(10)},   // TRACE connection between ports 9 and 10
            {PortId(11), PortId(12)},  // TRACE connection between ports 11 and 12
            {PortId(13), PortId(14)},  // TRACE connection between ports 13 and 14
            {PortId(15), PortId(16)},  // TRACE connection between ports 15 and 16
            {PortId(17), PortId(18)},  // TRACE connection between ports 17 and 18
            {PortId(19), PortId(20)},  // TRACE connection between ports 19 and 20
        };
        return internal_connections;
    }
};

// P300 board class
class P300 : public Board {
public:
    P300() : Board(create_p300_ports(), create_p300_internal_connections(), tt::umd::BoardType::P300) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_p300_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // WARP400 ports
        auto& warp400_ports = ports[PortType::WARP400];
        warp400_ports = {
            {PortId(1), {{1, ChanId(3)}, {1, ChanId(6)}, {0, ChanId(7)}, {0, ChanId(9)}}},  // ASIC 1: channels 3,6;
                                                                                            // ASIC 0: channels 7,9
            {PortId(2), {{1, ChanId(2)}, {1, ChanId(4)}, {0, ChanId(5)}, {0, ChanId(4)}}},  // ASIC 1: channels 2,4;
                                                                                            // ASIC 0: channels 5,4
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {PortId(1), {{1, ChanId(8)}, {1, ChanId(9)}}},  // ASIC 1, channels 8,9
            {PortId(2), {{0, ChanId(3)}, {0, ChanId(2)}}},  // ASIC 0, channels 3,2
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_p300_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections;
        internal_connections[PortType::TRACE] = {{PortId(1), PortId(2)}};  // TRACE connection between ports 1 and 2
        return internal_connections;
    }
};

// P150 board class
class P150 : public Board {
public:
    P150() : Board(create_p150_ports(), create_p150_internal_connections(), tt::umd::BoardType::P150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_p150_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {PortId(1), {{0, ChanId(9)}, {0, ChanId(11)}}},  // ASIC 0, channels 9,11
            {PortId(2), {{0, ChanId(8)}, {0, ChanId(10)}}},  // ASIC 0, channels 8,10
            {PortId(3), {{0, ChanId(5)}, {0, ChanId(7)}}},   // ASIC 0, channels 5,7
            {PortId(4), {{0, ChanId(4)}, {0, ChanId(6)}}},   // ASIC 0, channels 4,6
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_p150_internal_connections() {
        // No internal connections for P150
        return std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>();
    }
};

// N150 board class
class N150 : public Board {
public:
    N150() : Board(create_n150_ports(), create_n150_internal_connections(), tt::umd::BoardType::N150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_n150_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {PortId(1), {{0, ChanId(6)}, {0, ChanId(7)}}},  // ASIC 0, channels 6,7
            {PortId(2), {{0, ChanId(0)}, {0, ChanId(1)}}},  // ASIC 0, channels 0,1
        };

        // WARP100 ports
        auto& warp100_ports = ports[PortType::WARP100];
        warp100_ports = {
            {PortId(1), {{0, ChanId(14)}, {0, ChanId(15)}}},  // ASIC 0, channels 14,15
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_n150_internal_connections() {
        // No internal connections for N150
        return std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>>();
    }
};

// BH_UBB board class
class BH_UBB : public Board {
public:
    BH_UBB() : Board(create_bh_ubb_ports(), create_bh_ubb_internal_connections(), tt::umd::BoardType::UBB) {}

private:
    static std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> create_bh_ubb_ports() {
        std::unordered_map<PortType, std::unordered_map<PortId, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {PortId(1), {{5, ChanId(2)}, {5, ChanId(3)}}},   // ASIC 5, channels 2,3
            {PortId(2), {{1, ChanId(2)}, {1, ChanId(3)}}},   // ASIC 1, channels 2,3
            {PortId(3), {{1, ChanId(0)}, {1, ChanId(1)}}},   // ASIC 1, channels 0,1
            {PortId(4), {{2, ChanId(0)}, {2, ChanId(1)}}},   // ASIC 2, channels 0,1
            {PortId(5), {{3, ChanId(0)}, {3, ChanId(1)}}},   // ASIC 3, channels 0,1
            {PortId(6), {{4, ChanId(0)}, {4, ChanId(1)}}},   // ASIC 4, channels 0,1
            {PortId(7), {{1, ChanId(8)}, {2, ChanId(8)}}},   // ASIC 1,2: channel 8
            {PortId(8), {{5, ChanId(8)}, {6, ChanId(8)}}},   // ASIC 5,6: channel 8
            {PortId(9), {{3, ChanId(8)}, {4, ChanId(8)}}},   // ASIC 3,4: channel 8
            {PortId(10), {{7, ChanId(8)}, {8, ChanId(8)}}},  // ASIC 7,8: channel 8
            {PortId(11), {{1, ChanId(9)}, {2, ChanId(9)}}},  // ASIC 1,2: channel 9
            {PortId(12), {{5, ChanId(9)}, {6, ChanId(9)}}},  // ASIC 5,6: channel 9
            {PortId(13), {{3, ChanId(9)}, {4, ChanId(9)}}},  // ASIC 3,4: channel 9
            {PortId(14), {{7, ChanId(9)}, {8, ChanId(9)}}},  // ASIC 7,8: channel 9
        };

        // NOTE: These Linking Board connections are not finalized yet.

        // LINKING_BOARD_1 ports
        auto& linking_board_1_ports = ports[PortType::LINKING_BOARD_1];
        linking_board_1_ports = {
            {PortId(1), {{5, ChanId(6)}, {5, ChanId(7)}}},  // ASIC 5, channels 6,7
            {PortId(2), {{6, ChanId(6)}, {6, ChanId(7)}}},  // ASIC 6, channels 6,7
        };

        // LINKING_BOARD_2 ports
        auto& linking_board_2_ports = ports[PortType::LINKING_BOARD_2];
        linking_board_2_ports = {
            {PortId(1), {{7, ChanId(6)}, {7, ChanId(7)}}},  // ASIC 7, channels 6,7
            {PortId(2), {{8, ChanId(6)}, {8, ChanId(7)}}},  // ASIC 8, channels 6,7
        };

        // LINKING_BOARD_3 ports
        auto& linking_board_3_ports = ports[PortType::LINKING_BOARD_3];
        linking_board_3_ports = {
            {PortId(1), {{8, ChanId(4)}, {8, ChanId(5)}}},  // ASIC 8, channels 4,5
            {PortId(2), {{4, ChanId(4)}, {4, ChanId(5)}}},  // ASIC 4, channels 4,5
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {PortId(1), {{5, ChanId(0)}, {5, ChanId(1)}}},   // ASIC 5, channels 0,1
            {PortId(2), {{1, ChanId(6)}, {1, ChanId(7)}}},   // ASIC 1, channels 6,7
            {PortId(3), {{5, ChanId(4)}, {5, ChanId(5)}}},   // ASIC 5, channels 4,5
            {PortId(4), {{6, ChanId(2)}, {6, ChanId(3)}}},   // ASIC 6, channels 2,3
            {PortId(5), {{1, ChanId(4)}, {1, ChanId(5)}}},   // ASIC 1, channels 4,5
            {PortId(6), {{2, ChanId(2)}, {2, ChanId(3)}}},   // ASIC 2, channels 2,3
            {PortId(7), {{6, ChanId(0)}, {6, ChanId(1)}}},   // ASIC 6, channels 0,1
            {PortId(8), {{2, ChanId(6)}, {2, ChanId(7)}}},   // ASIC 2, channels 6,7
            {PortId(9), {{6, ChanId(4)}, {6, ChanId(5)}}},   // ASIC 6, channels 4,5
            {PortId(10), {{7, ChanId(2)}, {7, ChanId(3)}}},  // ASIC 7, channels 2,3
            {PortId(11), {{2, ChanId(4)}, {2, ChanId(5)}}},  // ASIC 2, channels 4,5
            {PortId(12), {{3, ChanId(2)}, {3, ChanId(3)}}},  // ASIC 3, channels 2,3
            {PortId(13), {{7, ChanId(0)}, {7, ChanId(1)}}},  // ASIC 7, channels 0,1
            {PortId(14), {{3, ChanId(6)}, {3, ChanId(7)}}},  // ASIC 3, channels 6,7
            {PortId(15), {{7, ChanId(4)}, {7, ChanId(5)}}},  // ASIC 7, channels 4,5
            {PortId(16), {{8, ChanId(2)}, {8, ChanId(3)}}},  // ASIC 8, channels 2,3
            {PortId(17), {{3, ChanId(4)}, {3, ChanId(5)}}},  // ASIC 3, channels 4,5
            {PortId(18), {{4, ChanId(2)}, {4, ChanId(3)}}},  // ASIC 4, channels 2,3
            {PortId(19), {{8, ChanId(0)}, {8, ChanId(1)}}},  // ASIC 8, channels 0,1
            {PortId(20), {{4, ChanId(6)}, {4, ChanId(7)}}},  // ASIC 4, channels 6,7
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> create_bh_ubb_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<PortId, PortId>>> internal_connections;
        internal_connections[PortType::TRACE] = {
            {PortId(1), PortId(2)},    // TRACE connection between ports 1 and 2
            {PortId(3), PortId(4)},    // TRACE connection between ports 3 and 4
            {PortId(5), PortId(6)},    // TRACE connection between ports 5 and 6
            {PortId(7), PortId(8)},    // TRACE connection between ports 7 and 8
            {PortId(9), PortId(10)},   // TRACE connection between ports 9 and 10
            {PortId(11), PortId(12)},  // TRACE connection between ports 11 and 12
            {PortId(13), PortId(14)},  // TRACE connection between ports 13 and 14
            {PortId(15), PortId(16)},  // TRACE connection between ports 15 and 16
            {PortId(17), PortId(18)},  // TRACE connection between ports 17 and 18
            {PortId(19), PortId(20)},  // TRACE connection between ports 19 and 20
        };
        return internal_connections;
    }
};

// Factory function to create boards by type (for backward compatibility)
Board create_board(const std::string& board_name) {
    auto board_type = enchantum::cast<tt::umd::BoardType>(board_name, ttsl::ascii_caseless_comp);
    if (!board_type.has_value()) {
        throw std::runtime_error("Invalid board type: " + board_name);
    }
    switch (*board_type) {
        case BoardType::N300: return N300();
        case BoardType::UBB: return WH_UBB();
        case BoardType::P300: return P300();
        case BoardType::P150: return P150();
        case BoardType::N150: return N150();
        // case BoardType::BH_UBB:
        //     return BH_UBB();
        default: throw std::runtime_error("Unknown board type: " + board_name);
    }

    throw std::runtime_error("Unknown board type: " + board_name);
}

}  // namespace tt::scaleout_tools

// Hash function implementation for AsicChannel
namespace std {
std::size_t hash<tt::scaleout_tools::AsicChannel>::operator()(
    const tt::scaleout_tools::AsicChannel& asic_channel) const {
    return tt::stl::hash::hash_objects_with_default_seed(asic_channel.asic_location, asic_channel.channel_id);
}
}  // namespace std
