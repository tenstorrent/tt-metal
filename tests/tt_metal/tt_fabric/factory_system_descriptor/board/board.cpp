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

namespace tt::tt_fabric {

Board::Board(
    const std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>>& ports,
    const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& internal_connections,
    const tt::umd::BoardType& board_type) :
    ports_(ports), internal_connections_(internal_connections), board_type_(board_type), asic_indices_() {
    // Initialize available_ports from ports
    for (const auto& [port_type, port_mapping] : ports) {
        auto& available_ports = available_port_ids_[port_type];
        for (const auto& [port_id, asic_channels] : port_mapping) {
            // TODO: Could probably optimize this
            for (const auto& asic_channel : asic_channels) {
                asic_indices_.insert(asic_channel.asic_location);
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
    if (asic_indices_.size() != expected_asic_indices) {
        throw std::runtime_error("Number of ASICs in board configuration does not match number of chips in board type");
    }
}

const tt::umd::BoardType& Board::get_board_type() const { return board_type_; }

const std::vector<uint32_t>& Board::get_available_port_ids(PortType port_type) const {
    auto it = available_port_ids_.find(port_type);
    if (it == available_port_ids_.end()) {
        throw std::runtime_error("Port type not found in board configuration");
    }
    return it->second;
}

const std::vector<AsicChannel>& Board::get_port_channels(PortType port_type, uint32_t port_id) const {
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

void Board::mark_port_used(PortType port_type, uint32_t port_id) {
    auto& available_ports = available_port_ids_[port_type];
    auto it = std::find(available_ports.begin(), available_ports.end(), port_id);
    if (it != available_ports.end()) {
        available_ports.erase(it);
    }
}

// Get internal connections for this board
const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& Board::get_internal_connections()
    const {
    return internal_connections_;
}

// N300 board class
class N300 : public Board {
public:
    N300() : Board(create_n300_ports(), create_n300_internal_connections(), tt::umd::BoardType::N300) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_n300_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {1, {{0, 6}, {0, 7}}},  // ASIC 0, channels 6,7
            {2, {{0, 0}, {0, 1}}},  // ASIC 0, channels 0,1
        };

        // WARP100 ports
        auto& warp100_ports = ports[PortType::WARP100];
        warp100_ports = {
            {1, {{0, 14}, {0, 15}}},  // ASIC 0, channels 14,15
            {2, {{1, 6}, {1, 7}}},    // ASIC 1, channels 6,7
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {1, {{0, 8}, {0, 9}}},  // ASIC 0, channels 8,9
            {2, {{1, 0}, {1, 1}}},  // ASIC 1, channels 0,1
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> create_n300_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
        internal_connections[PortType::TRACE] = {{1, 2}};  // TRACE connection between ports 1 and 2
        return internal_connections;
    }
};

// WH_UBB board class
class WH_UBB : public Board {
public:
    WH_UBB() : Board(create_wh_ubb_ports(), create_wh_ubb_internal_connections(), tt::umd::BoardType::UBB) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_wh_ubb_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {1, {{5, 4}, {5, 5}, {5, 6}, {5, 7}}},  // ASIC 5, channels 4,5,6,7
            {2, {{1, 4}, {1, 5}, {1, 6}, {1, 7}}},  // ASIC 1, channels 4,5,6,7
            {3, {{1, 0}, {1, 1}, {1, 2}, {1, 3}}},  // ASIC 1, channels 0,1,2,3
            {4, {{2, 0}, {2, 1}, {2, 2}, {2, 3}}},  // ASIC 2, channels 0,1,2,3
            {5, {{3, 0}, {3, 1}, {3, 2}, {3, 3}}},  // ASIC 3, channels 0,1,2,3
            {6, {{4, 0}, {4, 1}, {4, 2}, {4, 3}}},  // ASIC 4, channels 0,1,2,3
        };

        // LINKING_BOARD_1 ports
        auto& linking_board_1_ports = ports[PortType::LINKING_BOARD_1];
        linking_board_1_ports = {
            {1, {{5, 12}, {5, 13}, {5, 14}, {5, 15}}},  // ASIC 5, channels 12,13,14,15
            {2, {{6, 12}, {6, 13}, {6, 14}, {6, 15}}},  // ASIC 6, channels 12,13,14,15
        };

        // LINKING_BOARD_2 ports
        auto& linking_board_2_ports = ports[PortType::LINKING_BOARD_2];
        linking_board_2_ports = {
            {1, {{7, 12}, {7, 13}, {7, 14}, {7, 15}}},  // ASIC 7, channels 12,13,14,15
            {2, {{8, 12}, {8, 13}, {8, 14}, {8, 15}}},  // ASIC 8, channels 12,13,14,15
        };

        // LINKING_BOARD_3 ports
        auto& linking_board_3_ports = ports[PortType::LINKING_BOARD_3];
        linking_board_3_ports = {
            {1, {{8, 8}, {8, 9}, {8, 10}, {8, 11}}},  // ASIC 8, channels 8,9,10,11
            {2, {{4, 8}, {4, 9}, {4, 10}, {4, 11}}},  // ASIC 4, channels 8,9,10,11
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {1, {{5, 0}, {5, 1}, {5, 2}, {5, 3}}},       // ASIC 5, channels 0,1,2,3
            {2, {{1, 12}, {1, 13}, {1, 14}, {1, 15}}},   // ASIC 1, channels 12,13,14,15
            {3, {{5, 8}, {5, 9}, {5, 10}, {5, 11}}},     // ASIC 5, channels 8,9,10,11
            {4, {{6, 4}, {6, 5}, {6, 6}, {6, 7}}},       // ASIC 6, channels 4,5,6,7
            {5, {{1, 8}, {1, 9}, {1, 10}, {1, 11}}},     // ASIC 1, channels 8,9,10,11
            {6, {{2, 4}, {2, 5}, {2, 6}, {2, 7}}},       // ASIC 2, channels 4,5,6,7
            {7, {{6, 0}, {6, 1}, {6, 2}, {6, 3}}},       // ASIC 6, channels 0,1,2,3
            {8, {{2, 12}, {2, 13}, {2, 14}, {2, 15}}},   // ASIC 2, channels 12,13,14,15
            {9, {{6, 8}, {6, 9}, {6, 10}, {6, 11}}},     // ASIC 6, channels 8,9,10,11
            {10, {{7, 4}, {7, 5}, {7, 6}, {7, 7}}},      // ASIC 7, channels 4,5,6,7
            {11, {{2, 8}, {2, 9}, {2, 10}, {2, 11}}},    // ASIC 2, channels 8,9,10,11
            {12, {{3, 4}, {3, 5}, {3, 6}, {3, 7}}},      // ASIC 3, channels 4,5,6,7
            {13, {{7, 0}, {7, 1}, {7, 2}, {7, 3}}},      // ASIC 7, channels 0,1,2,3
            {14, {{3, 12}, {3, 13}, {3, 14}, {3, 15}}},  // ASIC 3, channels 12,13,14,15
            {15, {{7, 8}, {7, 9}, {7, 10}, {7, 11}}},    // ASIC 7, channels 8,9,10,11
            {16, {{8, 4}, {8, 5}, {8, 6}, {8, 7}}},      // ASIC 8, channels 4,5,6,7
            {17, {{3, 8}, {3, 9}, {3, 10}, {3, 11}}},    // ASIC 3, channels 8,9,10,11
            {18, {{4, 4}, {4, 5}, {4, 6}, {4, 7}}},      // ASIC 4, channels 4,5,6,7
            {19, {{8, 0}, {8, 1}, {8, 2}, {8, 3}}},      // ASIC 8, channels 0,1,2,3
            {20, {{4, 12}, {4, 13}, {4, 14}, {4, 15}}},  // ASIC 4, channels 12,13,14,15
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>
    create_wh_ubb_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
        internal_connections[PortType::TRACE] = {
            {1, 2},    // TRACE connection between ports 1 and 2
            {3, 4},    // TRACE connection between ports 3 and 4
            {5, 6},    // TRACE connection between ports 5 and 6
            {7, 8},    // TRACE connection between ports 7 and 8
            {9, 10},   // TRACE connection between ports 9 and 10
            {11, 12},  // TRACE connection between ports 11 and 12
            {13, 14},  // TRACE connection between ports 13 and 14
            {15, 16},  // TRACE connection between ports 15 and 16
            {17, 18},  // TRACE connection between ports 17 and 18
            {19, 20},  // TRACE connection between ports 19 and 20
        };
        return internal_connections;
    }
};

// P300 board class
class P300 : public Board {
public:
    P300() : Board(create_p300_ports(), create_p300_internal_connections(), tt::umd::BoardType::P300) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_p300_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // WARP400 ports
        auto& warp400_ports = ports[PortType::WARP400];
        warp400_ports = {
            {1, {{1, 3}, {1, 6}, {0, 7}, {0, 9}}},  // ASIC 1: channels 3,6; ASIC 0: channels 7,9
            {2, {{1, 2}, {1, 4}, {0, 5}, {0, 4}}},  // ASIC 1: channels 2,4; ASIC 0: channels 5,4
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {1, {{1, 8}, {1, 9}}},  // ASIC 1, channels 8,9
            {2, {{0, 3}, {0, 2}}},  // ASIC 0, channels 3,2
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> create_p300_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
        internal_connections[PortType::TRACE] = {{1, 2}};  // TRACE connection between ports 1 and 2
        return internal_connections;
    }
};

// P150 board class
class P150 : public Board {
public:
    P150() : Board(create_p150_ports(), create_p150_internal_connections(), tt::umd::BoardType::P150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_p150_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {1, {{0, 9}, {0, 11}}},  // ASIC 0, channels 9,11
            {2, {{0, 8}, {0, 10}}},  // ASIC 0, channels 8,10
            {3, {{0, 5}, {0, 7}}},   // ASIC 0, channels 5,7
            {4, {{0, 4}, {0, 6}}},   // ASIC 0, channels 4,6
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> create_p150_internal_connections() {
        // No internal connections for P150
        return std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>();
    }
};

// N150 board class
class N150 : public Board {
public:
    N150() : Board(create_n150_ports(), create_n150_internal_connections(), tt::umd::BoardType::N150) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_n150_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {1, {{0, 6}, {0, 7}}},  // ASIC 0, channels 6,7
            {2, {{0, 0}, {0, 1}}},  // ASIC 0, channels 0,1
        };

        // WARP100 ports
        auto& warp100_ports = ports[PortType::WARP100];
        warp100_ports = {
            {1, {{0, 14}, {0, 15}}},  // ASIC 0, channels 14,15
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> create_n150_internal_connections() {
        // No internal connections for N150
        return std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>();
    }
};

// BH_UBB board class
class BH_UBB : public Board {
public:
    BH_UBB() : Board(create_bh_ubb_ports(), create_bh_ubb_internal_connections(), tt::umd::BoardType::UBB) {}

private:
    static std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> create_bh_ubb_ports() {
        std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

        // QSFP ports
        auto& qsfp_ports = ports[PortType::QSFP];
        qsfp_ports = {
            {1, {{5, 2}, {5, 3}}},   // ASIC 5, channels 2,3
            {2, {{1, 2}, {1, 3}}},   // ASIC 1, channels 2,3
            {3, {{1, 0}, {1, 1}}},   // ASIC 1, channels 0,1
            {4, {{2, 0}, {2, 1}}},   // ASIC 2, channels 0,1
            {5, {{3, 0}, {3, 1}}},   // ASIC 3, channels 0,1
            {6, {{4, 0}, {4, 1}}},   // ASIC 4, channels 0,1
            {7, {{1, 8}, {2, 8}}},   // ASIC 1,2: channel 8
            {8, {{5, 8}, {6, 8}}},   // ASIC 5,6: channel 8
            {9, {{3, 8}, {4, 8}}},   // ASIC 3,4: channel 8
            {10, {{7, 8}, {8, 8}}},  // ASIC 7,8: channel 8
            {11, {{1, 9}, {2, 9}}},  // ASIC 1,2: channel 9
            {12, {{5, 9}, {6, 9}}},  // ASIC 5,6: channel 9
            {13, {{3, 9}, {4, 9}}},  // ASIC 3,4: channel 9
            {14, {{7, 9}, {8, 9}}},  // ASIC 7,8: channel 9
        };

        // NOTE: These Linking Board connections are not finalized yet.

        // LINKING_BOARD_1 ports
        auto& linking_board_1_ports = ports[PortType::LINKING_BOARD_1];
        linking_board_1_ports = {
            {1, {{5, 6}, {5, 7}}},  // ASIC 5, channels 6,7
            {2, {{6, 6}, {6, 7}}},  // ASIC 6, channels 6,7
        };

        // LINKING_BOARD_2 ports
        auto& linking_board_2_ports = ports[PortType::LINKING_BOARD_2];
        linking_board_2_ports = {
            {1, {{7, 6}, {7, 7}}},  // ASIC 7, channels 6,7
            {2, {{8, 6}, {8, 7}}},  // ASIC 8, channels 6,7
        };

        // LINKING_BOARD_3 ports
        auto& linking_board_3_ports = ports[PortType::LINKING_BOARD_3];
        linking_board_3_ports = {
            {1, {{8, 4}, {8, 5}}},  // ASIC 8, channels 4,5
            {2, {{4, 4}, {4, 5}}},  // ASIC 4, channels 4,5
        };

        // TRACE ports
        auto& trace_ports = ports[PortType::TRACE];
        trace_ports = {
            {1, {{5, 0}, {5, 1}}},   // ASIC 5, channels 0,1
            {2, {{1, 6}, {1, 7}}},   // ASIC 1, channels 6,7
            {3, {{5, 4}, {5, 5}}},   // ASIC 5, channels 4,5
            {4, {{6, 2}, {6, 3}}},   // ASIC 6, channels 2,3
            {5, {{1, 4}, {1, 5}}},   // ASIC 1, channels 4,5
            {6, {{2, 2}, {2, 3}}},   // ASIC 2, channels 2,3
            {7, {{6, 0}, {6, 1}}},   // ASIC 6, channels 0,1
            {8, {{2, 6}, {2, 7}}},   // ASIC 2, channels 6,7
            {9, {{6, 4}, {6, 5}}},   // ASIC 6, channels 4,5
            {10, {{7, 2}, {7, 3}}},  // ASIC 7, channels 2,3
            {11, {{2, 4}, {2, 5}}},  // ASIC 2, channels 4,5
            {12, {{3, 2}, {3, 3}}},  // ASIC 3, channels 2,3
            {13, {{7, 0}, {7, 1}}},  // ASIC 7, channels 0,1
            {14, {{3, 6}, {3, 7}}},  // ASIC 3, channels 6,7
            {15, {{7, 4}, {7, 5}}},  // ASIC 7, channels 4,5
            {16, {{8, 2}, {8, 3}}},  // ASIC 8, channels 2,3
            {17, {{3, 4}, {3, 5}}},  // ASIC 3, channels 4,5
            {18, {{4, 2}, {4, 3}}},  // ASIC 4, channels 2,3
            {19, {{8, 0}, {8, 1}}},  // ASIC 8, channels 0,1
            {20, {{4, 6}, {4, 7}}},  // ASIC 4, channels 6,7
        };

        return ports;
    }

    static std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>
    create_bh_ubb_internal_connections() {
        std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
        internal_connections[PortType::TRACE] = {
            {1, 2},    // TRACE connection between ports 1 and 2
            {3, 4},    // TRACE connection between ports 3 and 4
            {5, 6},    // TRACE connection between ports 5 and 6
            {7, 8},    // TRACE connection between ports 7 and 8
            {9, 10},   // TRACE connection between ports 9 and 10
            {11, 12},  // TRACE connection between ports 11 and 12
            {13, 14},  // TRACE connection between ports 13 and 14
            {15, 16},  // TRACE connection between ports 15 and 16
            {17, 18},  // TRACE connection between ports 17 and 18
            {19, 20},  // TRACE connection between ports 19 and 20
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

}  // namespace tt::tt_fabric

// Hash function implementation for AsicChannel
namespace std {
std::size_t hash<tt::tt_fabric::AsicChannel>::operator()(const tt::tt_fabric::AsicChannel& asic_channel) const {
    return tt::stl::hash::hash_objects_with_default_seed(asic_channel.asic_location, asic_channel.channel_id);
}
}  // namespace std
