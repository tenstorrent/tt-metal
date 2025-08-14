// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <set>
#include <cstdio>
#include <umd/device/types/cluster_descriptor_types.h>
#include <yaml-cpp/yaml.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_fabric {
namespace system_health_tests {

struct AsicChannel {
    uint32_t asic_index;
    uint32_t channel_id;

    bool operator==(const AsicChannel& other) const {
        return asic_index == other.asic_index && channel_id == other.channel_id;
    }
};

enum class PortType {
    TRACE,
    QSFP,
    WARP100,
    WARP400,
    LINKING_BOARD_1,
    LINKING_BOARD_2,
    LINKING_BOARD_3,
};

struct LogicalChipConnection {
    uint32_t host_id;
    uint32_t tray_id;
    AsicChannel asic_channel;

    bool operator==(const LogicalChipConnection& other) const {
        return host_id == other.host_id && tray_id == other.tray_id &&
               asic_channel.asic_index == other.asic_channel.asic_index &&
               asic_channel.channel_id == other.asic_channel.channel_id;
    }
};

struct PhysicalChipConnection {
    std::string hostname;
    uint32_t tray_id;
    uint32_t asic_index;
    uint32_t channel_id;

    bool operator<(const PhysicalChipConnection& other) const {
        if (hostname != other.hostname) {
            return hostname < other.hostname;
        }
        if (tray_id != other.tray_id) {
            return tray_id < other.tray_id;
        }
        if (asic_index != other.asic_index) {
            return asic_index < other.asic_index;
        }
        return channel_id < other.channel_id;
    }

    bool operator==(const PhysicalChipConnection& other) const {
        return hostname == other.hostname && tray_id == other.tray_id && asic_index == other.asic_index &&
               channel_id == other.channel_id;
    }
};

}  // namespace system_health_tests
}  // namespace tt::tt_fabric

// Overload operator<< for PhysicalChipConnection to enable readable test output
namespace tt::tt_fabric::system_health_tests {
inline std::ostream& operator<<(std::ostream& os, const PhysicalChipConnection& conn) {
    os << "PhysicalChipConnection{hostname='" << conn.hostname << "', tray_id=" << conn.tray_id
       << ", asic_index=" << conn.asic_index << ", channel_id=" << conn.channel_id << "}";
    return os;
}
}  // namespace tt::tt_fabric::system_health_tests

// Overload std::hash for LogicalChipConnection
namespace std {
template <>
struct hash<tt::tt_fabric::system_health_tests::LogicalChipConnection> {
    std::size_t operator()(const tt::tt_fabric::system_health_tests::LogicalChipConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.host_id, conn.tray_id, conn.asic_channel.asic_index, conn.asic_channel.channel_id);
    }
};
template <>
struct hash<tt::tt_fabric::system_health_tests::PhysicalChipConnection> {
    std::size_t operator()(const tt::tt_fabric::system_health_tests::PhysicalChipConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.tray_id, conn.asic_index, conn.channel_id);
    }
};
}  // namespace std

namespace tt::tt_fabric {
namespace system_health_tests {

using LogicalChipConnectionPair = std::pair<LogicalChipConnection, LogicalChipConnection>;
using AsicChannelPair = std::pair<AsicChannel, AsicChannel>;

// TODO: Should look into cleaning this up
template<typename T>
struct Connector {
    static std::vector<AsicChannelPair> get_port_mapping(std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
        return T::get_port_mapping(start_channels, end_channels);
    }
};

struct LinearConnector : public Connector<LinearConnector> {
    static std::vector<AsicChannelPair> get_port_mapping(std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
        assert(start_channels.size() == end_channels.size());
        std::vector<AsicChannelPair> port_mapping;
        port_mapping.reserve(start_channels.size());
        for (size_t i = 0; i < start_channels.size(); i++) {
            port_mapping.emplace_back(start_channels[i], end_channels[i]);
        }
        return port_mapping;
    }
};

struct TraceConnector : public LinearConnector {
};

struct QSFPConnector : public LinearConnector {
};

struct LinkingBoard1Connector : public LinearConnector {};

struct LinkingBoard2Connector : public LinearConnector {};

struct LinkingBoard3Connector : public LinearConnector {};

struct Warp100Connector : public LinearConnector {
};

struct Warp400Connector : public Connector<Warp400Connector> {
    static std::vector<AsicChannelPair> get_port_mapping(std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
        assert(start_channels.size() == end_channels.size());
        assert(start_channels.size() % 2 == 0);
        std::vector<AsicChannelPair> port_mapping;
        port_mapping.reserve(start_channels.size());

        const size_t half_size = start_channels.size() / 2;
        for (size_t i = 0; i < half_size; i++) {
            port_mapping.emplace_back(start_channels[i], end_channels[half_size + i]);
        }
        for (size_t i = 0; i < half_size; i++) {
            port_mapping.emplace_back(start_channels[half_size + i], end_channels[i]);
        }
        return port_mapping;
    }
};

std::vector<AsicChannelPair> get_asic_channel_connections(PortType port_type, std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
    switch (port_type) {
        case PortType::QSFP:
            return QSFPConnector::get_port_mapping(start_channels, end_channels);
        case PortType::WARP100:
            return Warp100Connector::get_port_mapping(start_channels, end_channels);
        case PortType::WARP400:
            return Warp400Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_1: return LinkingBoard1Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_2: return LinkingBoard2Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_3: return LinkingBoard3Connector::get_port_mapping(start_channels, end_channels);
        case PortType::TRACE:
            return TraceConnector::get_port_mapping(start_channels, end_channels);
    }
}

class Board {
public:
    explicit Board(const YAML::Node& yaml_config) {
        parse_ports(yaml_config);
        parse_internal_connections(yaml_config);
    }

    // Constructor that takes maps directly instead of YAML
    Board(
        const std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>>& ports,
        const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& internal_connections) :
        ports_(ports), internal_connections_(internal_connections) {
        // Initialize available_port_ids_ from ports
        for (const auto& [port_type, port_mapping] : ports) {
            auto& available_ports = available_port_ids_[port_type];
            for (const auto& [port_id, _] : port_mapping) {
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
    }

    // Get available port IDs for a specific port type
    const std::vector<uint32_t>& get_available_port_ids(PortType port_type) const {
        auto it = available_port_ids_.find(port_type);
        if (it == available_port_ids_.end()) {
            throw std::runtime_error("Port type not found in board configuration");
        }
        return it->second;
    }

    // Get channels for a specific port
    const std::vector<AsicChannel>& get_port_channels(PortType port_type, uint32_t port_id) const {
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

    // Mark a port as used (remove from available list)
    void mark_port_used(PortType port_type, uint32_t port_id) {
        auto& available_ports = available_port_ids_[port_type];
        auto it = std::find(available_ports.begin(), available_ports.end(), port_id);
        if (it != available_ports.end()) {
            available_ports.erase(it);
        }
    }

    // Get internal connections for this board
    const std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>>& get_internal_connections() const {
        return internal_connections_;
    }

private:
    void parse_ports(const YAML::Node& yaml_config) {
        for (const auto& port_config : yaml_config["PORTS"]) {
            auto port_type = enchantum::cast<PortType>(port_config.first.as<std::string>(), ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_config.first.as<std::string>());
            }

            auto& port_mapping = ports_[*port_type];
            auto& available_ports = available_port_ids_[*port_type];

            for (const auto& channel_config : port_config.second) {
                uint32_t port_id = channel_config.first.as<uint32_t>();
                available_ports.push_back(port_id);

                for (const auto& asic_channel_config : channel_config.second) {
                    uint32_t asic_index = asic_channel_config["ASIC"].as<uint32_t>();
                    auto channel_ids = asic_channel_config["CHAN"].as<std::vector<uint32_t>>();

                    for (uint32_t channel_id : channel_ids) {
                        port_mapping[port_id].push_back(AsicChannel{
                            .asic_index = asic_index,
                            .channel_id = channel_id,
                        });
                    }
                }
            }
        }
    }

    void parse_internal_connections(const YAML::Node& yaml_config) {
        for (const auto& connection_config : yaml_config["CONNECTIONS"]) {
            auto port_type = enchantum::cast<PortType>(connection_config.first.as<std::string>(), ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + connection_config.first.as<std::string>());
            }

            auto& connections = internal_connections_[*port_type];
            for (const auto& connection : connection_config.second) {
                assert(connection.size() == 2);
                uint32_t port_0_id = connection[0].as<uint32_t>();
                uint32_t port_1_id = connection[1].as<uint32_t>();

                // Mark ports as used
                mark_port_used(*port_type, port_0_id);
                mark_port_used(*port_type, port_1_id);

                connections.push_back(std::make_pair(port_0_id, port_1_id));
            }
        }
    }

    std::unordered_map<PortType, std::vector<uint32_t>> available_port_ids_;
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports_;
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections_;
};

// Factory function to create an N300 board programmatically
Board create_n300_board() {
    // Define ports for N300 board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // QSFP ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> qsfp_ports = {
        {1, {{1, 6}, {1, 7}}},  // ASIC 1, channels 6,7
        {2, {{1, 0}, {1, 1}}},  // ASIC 1, channels 0,1
    };
    ports[PortType::QSFP] = qsfp_ports;

    // WARP100 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> warp100_ports = {
        {1, {{1, 14}, {1, 15}}},  // ASIC 1, channels 14,15
        {2, {{2, 6}, {2, 7}}},    // ASIC 2, channels 6,7
    };
    ports[PortType::WARP100] = warp100_ports;

    // TRACE ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> trace_ports = {
        {1, {{1, 8}, {1, 9}}},  // ASIC 1, channels 8,9
        {2, {{2, 0}, {2, 1}}},  // ASIC 2, channels 0,1
    };
    ports[PortType::TRACE] = trace_ports;

    // Define internal connections
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
    internal_connections[PortType::TRACE] = {{1, 2}};  // TRACE connection between ports 1 and 2

    return Board(ports, internal_connections);
}

// Factory function to create a WH_UBB board programmatically
Board create_wh_ubb_board() {
    // Define ports for WH_UBB board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // QSFP ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> qsfp_ports = {
        {1, {{5, 4}, {5, 5}, {5, 6}, {5, 7}}},  // ASIC 5, channels 4,5,6,7
        {2, {{1, 4}, {1, 5}, {1, 6}, {1, 7}}},  // ASIC 1, channels 4,5,6,7
        {3, {{1, 0}, {1, 1}, {1, 2}, {1, 3}}},  // ASIC 1, channels 0,1,2,3
        {4, {{2, 0}, {2, 1}, {2, 2}, {2, 3}}},  // ASIC 2, channels 0,1,2,3
        {5, {{3, 0}, {3, 1}, {3, 2}, {3, 3}}},  // ASIC 3, channels 0,1,2,3
        {6, {{4, 0}, {4, 1}, {4, 2}, {4, 3}}},  // ASIC 4, channels 0,1,2,3
    };
    ports[PortType::QSFP] = qsfp_ports;

    // LINKING_BOARD_1 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_1_ports = {
        {1, {{5, 12}, {5, 13}, {5, 14}, {5, 15}}},  // ASIC 5, channels 12,13,14,15
        {2, {{6, 12}, {6, 13}, {6, 14}, {6, 15}}},  // ASIC 6, channels 12,13,14,15
    };
    ports[PortType::LINKING_BOARD_1] = linking_board_1_ports;

    // LINKING_BOARD_2 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_2_ports = {
        {1, {{7, 12}, {7, 13}, {7, 14}, {7, 15}}},  // ASIC 7, channels 12,13,14,15
        {2, {{8, 12}, {8, 13}, {8, 14}, {8, 15}}},  // ASIC 8, channels 12,13,14,15
    };
    ports[PortType::LINKING_BOARD_2] = linking_board_2_ports;

    // LINKING_BOARD_3 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_3_ports = {
        {1, {{8, 8}, {8, 9}, {8, 10}, {8, 11}}},  // ASIC 8, channels 8,9,10,11
        {2, {{4, 8}, {4, 9}, {4, 10}, {4, 11}}},  // ASIC 4, channels 8,9,10,11
    };
    ports[PortType::LINKING_BOARD_3] = linking_board_3_ports;

    // TRACE ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> trace_ports = {
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
    ports[PortType::TRACE] = trace_ports;

    // Define internal connections
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

    return Board(ports, internal_connections);
}

// Factory function to create a P300 board programmatically
Board create_p300_board() {
    // Define ports for P300 board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // WARP400 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> warp400_ports = {
        {1, {{1, 3}, {1, 6}, {2, 7}, {2, 9}}},  // ASIC 1: channels 3,6; ASIC 2: channels 7,9
        {2, {{1, 2}, {1, 4}, {2, 5}, {2, 4}}},  // ASIC 1: channels 2,4; ASIC 2: channels 5,4
    };
    ports[PortType::WARP400] = warp400_ports;

    // TRACE ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> trace_ports = {
        {1, {{1, 8}, {1, 9}}},  // ASIC 1, channels 8,9
        {2, {{2, 3}, {2, 2}}},  // ASIC 2, channels 3,2
    };
    ports[PortType::TRACE] = trace_ports;

    // Define internal connections
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;
    internal_connections[PortType::TRACE] = {{1, 2}};  // TRACE connection between ports 1 and 2

    return Board(ports, internal_connections);
}

// Factory function to create a P150 board programmatically
Board create_p150_board() {
    // Define ports for P150 board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // QSFP ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> qsfp_ports = {
        {1, {{1, 9}, {1, 11}}},  // ASIC 1, channels 9,11
        {2, {{1, 8}, {1, 10}}},  // ASIC 1, channels 8,10
        {3, {{1, 5}, {1, 7}}},   // ASIC 1, channels 5,7
        {4, {{1, 4}, {1, 6}}},   // ASIC 1, channels 4,6
    };
    ports[PortType::QSFP] = qsfp_ports;

    // No internal connections for P150
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;

    return Board(ports, internal_connections);
}

// Factory function to create an N150 board programmatically
Board create_n150_board() {
    // Define ports for N150 board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // QSFP ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> qsfp_ports = {
        {1, {{1, 6}, {1, 7}}},  // ASIC 1, channels 6,7
        {2, {{1, 0}, {1, 1}}},  // ASIC 1, channels 0,1
    };
    ports[PortType::QSFP] = qsfp_ports;

    // WARP100 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> warp100_ports = {
        {1, {{1, 14}, {1, 15}}},  // ASIC 1, channels 14,15
        {2, {{2, 6}, {2, 7}}},    // ASIC 2, channels 6,7
    };
    ports[PortType::WARP100] = warp100_ports;

    // No internal connections for N150
    std::unordered_map<PortType, std::vector<std::pair<uint32_t, uint32_t>>> internal_connections;

    return Board(ports, internal_connections);
}

// Factory function to create a BH_UBB board programmatically
Board create_bh_ubb_board() {
    // Define ports for BH_UBB board
    std::unordered_map<PortType, std::unordered_map<uint32_t, std::vector<AsicChannel>>> ports;

    // QSFP ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> qsfp_ports = {
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
    ports[PortType::QSFP] = qsfp_ports;

    // LINKING_BOARD_1 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_1_ports = {
        {1, {{5, 6}, {5, 7}}},  // ASIC 5, channels 6,7
        {2, {{6, 6}, {6, 7}}},  // ASIC 6, channels 6,7
    };
    ports[PortType::LINKING_BOARD_1] = linking_board_1_ports;

    // LINKING_BOARD_2 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_2_ports = {
        {1, {{7, 6}, {7, 7}}},  // ASIC 7, channels 6,7
        {2, {{8, 6}, {8, 7}}},  // ASIC 8, channels 6,7
    };
    ports[PortType::LINKING_BOARD_2] = linking_board_2_ports;

    // LINKING_BOARD_3 ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> linking_board_3_ports = {
        {1, {{8, 4}, {8, 5}}},  // ASIC 8, channels 4,5
        {2, {{4, 4}, {4, 5}}},  // ASIC 4, channels 4,5
    };
    ports[PortType::LINKING_BOARD_3] = linking_board_3_ports;

    // TRACE ports
    std::unordered_map<uint32_t, std::vector<AsicChannel>> trace_ports = {
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
    ports[PortType::TRACE] = trace_ports;

    // Define internal connections
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

    return Board(ports, internal_connections);
}

// Hierarchical system components
struct Pod {
    std::unordered_map<uint32_t, Board> boards;
    std::vector<LogicalChipConnectionPair> connections;
};

struct Superpod {
    std::unordered_map<uint32_t, Pod> pods;
    std::vector<LogicalChipConnectionPair> connections;
};

struct Cluster {
    std::unordered_map<uint32_t, Superpod> superpods;
    std::vector<LogicalChipConnectionPair> connections;
};

class HierarchicalSystemBuilder {
public:
    static std::vector<LogicalChipConnectionPair> build_board_connections(
        const Board& board, uint32_t host_id, uint32_t tray_id) {
        std::vector<LogicalChipConnectionPair> chip_connections;
        // Add internal connections within each board
        add_internal_board_connections(board, chip_connections, host_id, tray_id);
        return chip_connections;
    }

    static std::vector<LogicalChipConnectionPair> build_pod_connections(
        std::unordered_map<uint32_t, Board>& boards, const YAML::Node& pod_config, uint32_t host_id) {
        std::vector<LogicalChipConnectionPair> chip_connections;

        // Add internal connections within each board
        for (const auto& [tray_id, board] : boards) {
            auto board_connections = build_board_connections(board, host_id, tray_id);
            chip_connections.insert(chip_connections.end(), board_connections.begin(), board_connections.end());
        }

        // Add inter-board connections
        add_inter_board_connections(boards, pod_config, chip_connections, host_id);

        return chip_connections;
    }

    static std::vector<LogicalChipConnectionPair> build_superpod_connections(
        std::unordered_map<uint32_t, Pod>& pods,
        const YAML::Node& superpod_config,
        const YAML::Node& pod_config,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        std::vector<LogicalChipConnectionPair> chip_connections;

        // Add connections from each pod (these already include internal connections)
        for (auto& [pod_id, pod] : pods) {
            auto pod_connections = build_pod_connections(pod.boards, pod_config, host_ids_map.at(pod_id));
            chip_connections.insert(chip_connections.end(), pod_connections.begin(), pod_connections.end());
        }

        // Add inter-pod connections
        add_inter_pod_connections(pods, superpod_config, chip_connections, host_ids_map);

        return chip_connections;
    }

    static std::vector<LogicalChipConnectionPair> build_cluster_connections(
        std::unordered_map<uint32_t, Superpod>& superpods,
        const YAML::Node& cluster_config,
        const YAML::Node& superpod_config,
        const YAML::Node& pod_config,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        std::vector<LogicalChipConnectionPair> chip_connections;

        // Add connections from each superpod (these already include internal connections)
        for (auto& [superpod_id, superpod] : superpods) {
            auto superpod_connections =
                build_superpod_connections(superpod.pods, superpod_config, pod_config, host_ids_map.at(superpod_id));
            chip_connections.insert(chip_connections.end(), superpod_connections.begin(), superpod_connections.end());
        }

        // Add inter-superpod connections
        add_inter_superpod_connections(superpods, cluster_config, chip_connections, host_ids_map);

        return chip_connections;
    }

    static void add_internal_board_connections(
        const Board& board,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_id,
        uint32_t tray_id) {
        for (const auto& [port_type, connections] : board.get_internal_connections()) {
            for (const auto& connection : connections) {
                const auto& start_channels = board.get_port_channels(port_type, connection.first);
                const auto& end_channels = board.get_port_channels(port_type, connection.second);

                auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);

                for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
                    chip_connections.emplace_back(
                        LogicalChipConnection{.host_id = host_id, .tray_id = tray_id, .asic_channel = start_channel},
                        LogicalChipConnection{.host_id = host_id, .tray_id = tray_id, .asic_channel = end_channel});
                }
            }
        }
    }

private:
    static void add_inter_board_connections(
        std::unordered_map<uint32_t, Board>& boards,
        const YAML::Node& pod_config,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_id) {
        for (const auto& connection_config : pod_config["CONNECTIONS"]) {
            auto port_type = enchantum::cast<PortType>(connection_config.first.as<std::string>(), ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + connection_config.first.as<std::string>());
            }

            for (const auto& connection : connection_config.second) {
                connect_boards(*port_type, connection, boards, chip_connections, host_id);
            }
        }
    }

    static void add_inter_pod_connections(
        std::unordered_map<uint32_t, Pod>& pods,
        const YAML::Node& superpod_config,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        for (const auto& connection_config : superpod_config["CONNECTIONS"]) {
            auto port_type = enchantum::cast<PortType>(connection_config.first.as<std::string>(), ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + connection_config.first.as<std::string>());
            }

            for (const auto& connection : connection_config.second) {
                connect_pods(*port_type, connection, pods, chip_connections, host_ids_map);
            }
        }
    }

    static void add_inter_superpod_connections(
        std::unordered_map<uint32_t, Superpod>& superpods,
        const YAML::Node& cluster_config,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        for (const auto& connection_config : cluster_config["CONNECTIONS"]) {
            auto port_type = enchantum::cast<PortType>(connection_config.first.as<std::string>(), ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + connection_config.first.as<std::string>());
            }

            for (const auto& connection : connection_config.second) {
                connect_superpods(*port_type, connection, superpods, chip_connections, host_ids_map);
            }
        }
    }

    static void connect_boards(
        PortType port_type,
        const YAML::Node& connection,
        std::unordered_map<uint32_t, Board>& boards,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_id) {
        uint32_t board_0_id = connection[0]["BOARD"].as<uint32_t>();
        uint32_t port_0_id = connection[0]["PORT"].as<uint32_t>();
        uint32_t board_1_id = connection[1]["BOARD"].as<uint32_t>();
        uint32_t port_1_id = connection[1]["PORT"].as<uint32_t>();

        Board& board_0 = boards.at(board_0_id);
        Board& board_1 = boards.at(board_1_id);

        // Verify ports are available
        const auto& board_0_available = board_0.get_available_port_ids(port_type);
        const auto& board_1_available = board_1.get_available_port_ids(port_type);

        if (std::find(board_0_available.begin(), board_0_available.end(), port_0_id) == board_0_available.end()) {
            throw std::runtime_error("Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id));
        }

        if (std::find(board_1_available.begin(), board_1_available.end(), port_1_id) == board_1_available.end()) {
            throw std::runtime_error("Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id));
        }

        // Mark ports as used
        board_0.mark_port_used(port_type, port_0_id);
        board_1.mark_port_used(port_type, port_1_id);

        // Get channels and create connections
        const auto& start_channels = board_0.get_port_channels(port_type, port_0_id);
        const auto& end_channels = board_1.get_port_channels(port_type, port_1_id);

        auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);

        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
            chip_connections.emplace_back(
                LogicalChipConnection{.host_id = host_id, .tray_id = board_0_id, .asic_channel = start_channel},
                LogicalChipConnection{.host_id = host_id, .tray_id = board_1_id, .asic_channel = end_channel});
        }
    }

    static void connect_pods(
        PortType port_type,
        const YAML::Node& connection,
        std::unordered_map<uint32_t, Pod>& pods,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        uint32_t pod_0_id = connection[0]["POD"].as<uint32_t>();
        uint32_t board_0_id = connection[0]["BOARD"].as<uint32_t>();
        uint32_t port_0_id = connection[0]["PORT"].as<uint32_t>();
        uint32_t pod_1_id = connection[1]["POD"].as<uint32_t>();
        uint32_t board_1_id = connection[1]["BOARD"].as<uint32_t>();
        uint32_t port_1_id = connection[1]["PORT"].as<uint32_t>();

        Pod& pod_0 = pods.at(pod_0_id);
        Pod& pod_1 = pods.at(pod_1_id);

        Board& board_0 = pod_0.boards.at(board_0_id);
        Board& board_1 = pod_1.boards.at(board_1_id);

        // Verify ports are available
        const auto& board_0_available = board_0.get_available_port_ids(port_type);
        const auto& board_1_available = board_1.get_available_port_ids(port_type);

        if (std::find(board_0_available.begin(), board_0_available.end(), port_0_id) == board_0_available.end()) {
            throw std::runtime_error(
                "Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id) +
                " in pod " + std::to_string(pod_0_id));
        }

        if (std::find(board_1_available.begin(), board_1_available.end(), port_1_id) == board_1_available.end()) {
            throw std::runtime_error(
                "Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id) +
                " in pod " + std::to_string(pod_1_id));
        }

        // Mark ports as used
        board_0.mark_port_used(port_type, port_0_id);
        board_1.mark_port_used(port_type, port_1_id);

        // Get channels and create connections
        const auto& start_channels = board_0.get_port_channels(port_type, port_0_id);
        const auto& end_channels = board_1.get_port_channels(port_type, port_1_id);

        auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);

        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
            chip_connections.emplace_back(
                LogicalChipConnection{
                    .host_id = host_ids_map.at(pod_0_id), .tray_id = board_0_id, .asic_channel = start_channel},
                LogicalChipConnection{
                    .host_id = host_ids_map.at(pod_1_id), .tray_id = board_1_id, .asic_channel = end_channel});
        }
    }

    static void connect_superpods(
        PortType port_type,
        const YAML::Node& connection,
        std::unordered_map<uint32_t, Superpod>& superpods,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        uint32_t superpod_0_id = connection[0]["SUPERPOD"].as<uint32_t>();
        uint32_t pod_0_id = connection[0]["POD"].as<uint32_t>();
        uint32_t board_0_id = connection[0]["BOARD"].as<uint32_t>();
        uint32_t port_0_id = connection[0]["PORT"].as<uint32_t>();
        uint32_t superpod_1_id = connection[1]["SUPERPOD"].as<uint32_t>();
        uint32_t pod_1_id = connection[1]["POD"].as<uint32_t>();
        uint32_t board_1_id = connection[1]["BOARD"].as<uint32_t>();
        uint32_t port_1_id = connection[1]["PORT"].as<uint32_t>();

        Superpod& superpod_0 = superpods.at(superpod_0_id);
        Superpod& superpod_1 = superpods.at(superpod_1_id);

        Pod& pod_0 = superpod_0.pods.at(pod_0_id);
        Pod& pod_1 = superpod_1.pods.at(pod_1_id);

        Board& board_0 = pod_0.boards.at(board_0_id);
        Board& board_1 = pod_1.boards.at(board_1_id);

        // Verify ports are available
        const auto& board_0_available = board_0.get_available_port_ids(port_type);
        const auto& board_1_available = board_1.get_available_port_ids(port_type);

        if (std::find(board_0_available.begin(), board_0_available.end(), port_0_id) == board_0_available.end()) {
            throw std::runtime_error(
                "Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id) +
                " in pod " + std::to_string(pod_0_id) + " in superpod " + std::to_string(superpod_0_id));
        }

        if (std::find(board_1_available.begin(), board_1_available.end(), port_1_id) == board_1_available.end()) {
            throw std::runtime_error(
                "Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id) +
                " in pod " + std::to_string(pod_1_id) + " in superpod " + std::to_string(superpod_1_id));
        }

        // Mark ports as used
        board_0.mark_port_used(port_type, port_0_id);
        board_1.mark_port_used(port_type, port_1_id);

        // Get channels and create connections
        const auto& start_channels = board_0.get_port_channels(port_type, port_0_id);
        const auto& end_channels = board_1.get_port_channels(port_type, port_1_id);

        auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);

        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
            chip_connections.emplace_back(
                LogicalChipConnection{
                    .host_id = host_ids_map.at(superpod_0_id).at(pod_0_id),
                    .tray_id = board_0_id,
                    .asic_channel = start_channel},
                LogicalChipConnection{
                    .host_id = host_ids_map.at(superpod_1_id).at(pod_1_id),
                    .tray_id = board_1_id,
                    .asic_channel = end_channel});
        }
    }
};

class SystemDescriptorEmitter {
public:
    static void emit_factory_system_descriptor(
        const std::string_view yaml_output_path, const std::vector<LogicalChipConnectionPair>& chip_connections) {
        emit_factory_system_descriptor(
            yaml_output_path, chip_connections, std::vector<std::tuple<uint32_t, uint32_t, std::string>>{});
    }

    static void emit_factory_system_descriptor(
        const std::string_view yaml_output_path,
        const std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::vector<std::tuple<uint32_t, uint32_t, std::string>>& host_tray_to_board_type,
        const std::vector<std::string>& hostnames = {}) {
        // Validate that there are no duplicate connections
        validate_no_duplicate_connections(chip_connections);

        YAML::Node factory_system_descriptor;

        // Add hostnames list if provided
        for (const auto& hostname : hostnames) {
            factory_system_descriptor["HOSTNAMES"].push_back(hostname);
        }

        // Add board types
        for (const auto& [host_id, tray_id, board_type] : host_tray_to_board_type) {
            YAML::Node board_type_entry;
            board_type_entry["host_id"] = host_id;
            board_type_entry["tray_id"] = tray_id;
            board_type_entry["board_type"] = board_type;
            board_type_entry.SetStyle(YAML::EmitterStyle::Flow);
            factory_system_descriptor["BOARD_TYPES"].push_back(board_type_entry);
        }

        for (const auto& chip_connection : chip_connections) {
            YAML::Node connection_pair = create_connection_pair_yaml(chip_connection);
            factory_system_descriptor["CHIP_CONNECTIONS"].push_back(connection_pair);
        }

        std::ofstream out(yaml_output_path.data());
        out << factory_system_descriptor;
    }

private:
    static void validate_no_duplicate_connections(const std::vector<LogicalChipConnectionPair>& chip_connections) {
        std::unordered_set<LogicalChipConnection> unique_connections;
        std::unordered_set<LogicalChipConnection> duplicate_connections;

        for (const auto& connection_pair : chip_connections) {
            // Check first connection
            if (unique_connections.find(connection_pair.first) != unique_connections.end()) {
                duplicate_connections.insert(connection_pair.first);
            } else {
                unique_connections.insert(connection_pair.first);
            }

            // Check second connection
            if (unique_connections.find(connection_pair.second) != unique_connections.end()) {
                duplicate_connections.insert(connection_pair.second);
            } else {
                unique_connections.insert(connection_pair.second);
            }
        }

        if (!duplicate_connections.empty()) {
            std::string error_msg = "Duplicate ChipConnection found:\n";
            for (const auto& dup : duplicate_connections) {
                error_msg += "  - {host_id: " + std::to_string(dup.host_id) +
                             ", tray_id: " + std::to_string(dup.tray_id) +
                             ", asic_index: " + std::to_string(dup.asic_channel.asic_index) +
                             ", chan_id: " + std::to_string(dup.asic_channel.channel_id) + "}\n";
            }
            throw std::runtime_error(error_msg);
        }
    }

    static YAML::Node create_connection_pair_yaml(const LogicalChipConnectionPair& connection) {
        YAML::Node connection_pair;

        // First connection in the pair
        YAML::Node first_connection;
        first_connection["host_id"] = connection.first.host_id;
        first_connection["tray_id"] = connection.first.tray_id;
        first_connection["asic_index"] = connection.first.asic_channel.asic_index;
        first_connection["chan_id"] = connection.first.asic_channel.channel_id;

        // Second connection in the pair
        YAML::Node second_connection;
        second_connection["host_id"] = connection.second.host_id;
        second_connection["tray_id"] = connection.second.tray_id;
        second_connection["asic_index"] = connection.second.asic_channel.asic_index;
        second_connection["chan_id"] = connection.second.asic_channel.channel_id;

        // Add both connections as a pair to the list
        connection_pair.push_back(first_connection);
        connection_pair.push_back(second_connection);

        // Set flow style for this specific connection pair to make it inline
        connection_pair.SetStyle(YAML::EmitterStyle::Flow);

        return connection_pair;
    }
};

// Common utility function for validating FSD against discovered GSD
void validate_fsd_against_gsd(
    const std::string& fsd_filename,
    const std::string& gsd_filename,
    const std::vector<std::string>& hostnames,
    bool strict_validation = true) {
    // Read the generated FSD
    YAML::Node generated_fsd = YAML::LoadFile(fsd_filename);

    // Read the discovered GSD (Global System Descriptor)
    YAML::Node discovered_gsd = YAML::LoadFile(gsd_filename);

    // Compare the FSD with the discovered GSD
    // First, compare hostnames
    ASSERT_TRUE(generated_fsd["HOSTNAMES"]);

    // Handle the new GSD structure with compute_node_specs
    ASSERT_TRUE(discovered_gsd["compute_node_specs"]);
    YAML::Node asic_info_node = discovered_gsd["compute_node_specs"];

    // Check that all discovered hostnames are present in the generated FSD
    std::set<std::string> generated_hostnames;
    for (const auto& hostname : generated_fsd["HOSTNAMES"]) {
        generated_hostnames.insert(hostname.as<std::string>());
    }

    std::set<std::string> discovered_hostnames;
    for (const auto& hostname_entry : asic_info_node) {
        discovered_hostnames.insert(hostname_entry.first.as<std::string>());
    }

    if (strict_validation) {
        EXPECT_EQ(generated_hostnames, discovered_hostnames) << "Hostnames mismatch";
    } else {
        for (const auto& hostname : discovered_hostnames) {
            EXPECT_TRUE(generated_hostnames.find(hostname) != generated_hostnames.end())
                << "Hostname not found in FSD: " << hostname;
        }
    }

    // Compare board types
    ASSERT_TRUE(generated_fsd["BOARD_TYPES"]);
    std::set<std::tuple<uint32_t, uint32_t, std::string>> generated_board_types;
    for (const auto& board_type : generated_fsd["BOARD_TYPES"]) {
        uint32_t host_id = board_type["host_id"].as<uint32_t>();
        uint32_t tray_id = board_type["tray_id"].as<uint32_t>();
        std::string board_type_name = board_type["board_type"].as<std::string>();
        generated_board_types.insert(std::make_tuple(host_id, tray_id, board_type_name));
    }

    // Strict validation: Each host, tray combination should have the same board type between FSD and GSD
    std::map<std::pair<std::string, uint32_t>, std::string> fsd_board_types;

    // Extract board types from FSD
    for (const auto& board_type : generated_fsd["BOARD_TYPES"]) {
        uint32_t host_id = board_type["host_id"].as<uint32_t>();
        uint32_t tray_id = board_type["tray_id"].as<uint32_t>();
        std::string board_type_name = board_type["board_type"].as<std::string>();
        fsd_board_types[std::make_pair(hostnames[host_id], tray_id)] = board_type_name;
    }

    // Compare board types between GSD and FSD
    for (const auto& host_entry : asic_info_node) {
        std::string hostname = host_entry.first.as<std::string>();
        YAML::Node host_node = host_entry.second;
        ASSERT_TRUE(host_node["asic_info"]) << "Host " << hostname << " missing asic_info";

        for (const auto& asic_info : host_node["asic_info"]) {
            uint32_t tray_id = asic_info["tray_id"].as<uint32_t>();
            std::string gsd_board_type = asic_info["board_type"].as<std::string>();

            auto fsd_key = std::make_pair(hostname, tray_id);
            if (strict_validation) {
                auto fsd_board_type = fsd_board_types.extract(fsd_key);

                ASSERT_TRUE(!fsd_board_type.empty())
                    << "Board type not found in FSD for host " << hostname << ", tray " << tray_id;

                EXPECT_EQ(fsd_board_type.mapped(), gsd_board_type)
                    << "Board type mismatch for host " << hostname << ", tray " << tray_id
                    << ": FSD=" << fsd_board_type.mapped() << ", GSD=" << gsd_board_type;
            } else {
                auto fsd_board_type = fsd_board_types.find(fsd_key);
                if (fsd_board_type != fsd_board_types.end()) {
                    EXPECT_EQ(fsd_board_type->second, gsd_board_type)
                        << "Board type mismatch for host " << hostname << ", tray " << tray_id
                        << ": FSD=" << fsd_board_type->second << ", GSD=" << gsd_board_type;
                } else {
                    FAIL() << "Board type not found in FSD for host " << hostname << ", tray " << tray_id;
                }
            }
        }
    }
    if (strict_validation) {
        EXPECT_EQ(fsd_board_types.size(), 0) << "Expected all board types to be found in FSD";
    }

    // Compare chip connections
    ASSERT_TRUE(generated_fsd["CHIP_CONNECTIONS"]);

    // Determine which connection types exist in the discovered GSD
    bool has_local_eth_connections =
        discovered_gsd["local_eth_connections"] && !discovered_gsd["local_connections"].IsNull();
    bool has_global_eth_connections =
        discovered_gsd["global_eth_connections"] && !discovered_gsd["global_eth_connections"].IsNull();

    // At least one connection type should exist
    ASSERT_TRUE(has_local_eth_connections || has_global_eth_connections)
        << "No connection types found in discovered GSD";

    // Convert generated connections to a comparable format
    std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> generated_connections;
    std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> duplicate_generated_connections;

    for (const auto& connection_pair : generated_fsd["CHIP_CONNECTIONS"]) {
        ASSERT_EQ(connection_pair.size(), 2) << "Each connection should have exactly 2 endpoints";

        auto first_conn = connection_pair[0];
        auto second_conn = connection_pair[1];

        uint32_t host_id_1 = first_conn["host_id"].as<uint32_t>();
        uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
        uint32_t asic_index_1 = first_conn["asic_index"].as<uint32_t>();
        uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();

        uint32_t host_id_2 = second_conn["host_id"].as<uint32_t>();
        uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
        uint32_t asic_index_2 = second_conn["asic_index"].as<uint32_t>();
        uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();

        std::string hostname_1 = hostnames[host_id_1];
        std::string hostname_2 = hostnames[host_id_2];

        PhysicalChipConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
        PhysicalChipConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

        // Sort to ensure consistent ordering
        std::pair<PhysicalChipConnection, PhysicalChipConnection> connection_pair_sorted;
        if (conn_1 < conn_2) {
            connection_pair_sorted = std::make_pair(conn_1, conn_2);
        } else {
            connection_pair_sorted = std::make_pair(conn_2, conn_1);
        }

        // Check for duplicates before inserting
        if (generated_connections.find(connection_pair_sorted) != generated_connections.end()) {
            duplicate_generated_connections.insert(connection_pair_sorted);
        } else {
            generated_connections.insert(connection_pair_sorted);
        }
    }

    // Report any duplicates found in generated connections
    if (!duplicate_generated_connections.empty()) {
        std::string error_msg = "Duplicate connections found in generated FSD:\n";
        for (const auto& dup : duplicate_generated_connections) {
            std::ostringstream oss;
            oss << "  - " << dup.first << " <-> " << dup.second;
            error_msg += oss.str() + "\n";
        }
        FAIL() << error_msg;
    }

    // Convert discovered GSD connections to the same format
    std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> discovered_connections;
    std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> duplicate_discovered_connections;

    // Process local connections if they exist
    if (has_local_eth_connections) {
        for (const auto& connection_pair : discovered_gsd["local_eth_connections"]) {
            ASSERT_EQ(connection_pair.size(), 2) << "Each connection should have exactly 2 endpoints";

            const auto& first_conn = connection_pair[0];
            const auto& second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_index_1 = first_conn["asic_index"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_index_2 = second_conn["asic_index"].as<uint32_t>();

            PhysicalChipConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
            PhysicalChipConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

            // Sort to ensure consistent ordering
            std::pair<PhysicalChipConnection, PhysicalChipConnection> connection_pair_sorted;
            if (conn_1 < conn_2) {
                connection_pair_sorted = std::make_pair(conn_1, conn_2);
            } else {
                connection_pair_sorted = std::make_pair(conn_2, conn_1);
            }

            // Check for duplicates before inserting
            if (discovered_connections.find(connection_pair_sorted) != discovered_connections.end()) {
                duplicate_discovered_connections.insert(connection_pair_sorted);
            } else {
                discovered_connections.insert(connection_pair_sorted);
            }
        }
    }

    // Process global_eth_connections if they exist (for 5WHGalaxyYTorusSuperpod)
    if (has_global_eth_connections) {
        for (const auto& connection_pair : discovered_gsd["global_eth_connections"]) {
            ASSERT_EQ(connection_pair.size(), 2) << "Each connection should have exactly 2 endpoints";

            auto first_conn = connection_pair[0];
            auto second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_index_1 = first_conn["asic_index"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_index_2 = second_conn["asic_index"].as<uint32_t>();

            PhysicalChipConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
            PhysicalChipConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

            // Sort to ensure consistent ordering
            std::pair<PhysicalChipConnection, PhysicalChipConnection> connection_pair_sorted;
            if (conn_1 < conn_2) {
                connection_pair_sorted = std::make_pair(conn_1, conn_2);
            } else {
                connection_pair_sorted = std::make_pair(conn_2, conn_1);
            }

            // Check for duplicates before inserting
            if (discovered_connections.find(connection_pair_sorted) != discovered_connections.end()) {
                duplicate_discovered_connections.insert(connection_pair_sorted);
            } else {
                discovered_connections.insert(connection_pair_sorted);
            }
        }
    }

    // Report any duplicates found in discovered GSD connections
    if (!duplicate_discovered_connections.empty()) {
        std::string error_msg = "Duplicate connections found in discovered GSD:\n";
        for (const auto& dup : duplicate_discovered_connections) {
            std::ostringstream oss;
            oss << "  - " << dup.first << " <-> " << dup.second;
            error_msg += oss.str() + "\n";
        }
        FAIL() << error_msg;
    }

    if (strict_validation) {
        EXPECT_EQ(generated_connections.size(), discovered_connections.size()) << "Number of connections mismatch";

        // Find missing and extra connections for detailed reporting
        std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> missing_in_gsd;
        std::set<std::pair<PhysicalChipConnection, PhysicalChipConnection>> extra_in_gsd;

        // Find connections in FSD but not in GSD
        for (const auto& conn : generated_connections) {
            if (discovered_connections.find(conn) == discovered_connections.end()) {
                missing_in_gsd.insert(conn);
            }
        }

        // Find connections in GSD but not in FSD
        for (const auto& conn : discovered_connections) {
            if (generated_connections.find(conn) == generated_connections.end()) {
                extra_in_gsd.insert(conn);
            }
        }

        // Report missing connections (in FSD but not in GSD)
        if (!missing_in_gsd.empty()) {
            std::string error_msg = "Connections found in FSD but missing in GSD (" +
                                    std::to_string(missing_in_gsd.size()) + " connections):\n";
            for (const auto& conn : missing_in_gsd) {
                std::ostringstream oss;
                oss << "  - " << conn.first << " <-> " << conn.second;
                error_msg += oss.str() + "\n";
            }
            std::cout << error_msg << std::endl;
        }

        // Report extra connections (in GSD but not in FSD)
        if (!extra_in_gsd.empty()) {
            std::string error_msg = "Extra connections found in GSD but not in FSD (" +
                                    std::to_string(extra_in_gsd.size()) + " connections):\n";
            for (const auto& conn : extra_in_gsd) {
                std::ostringstream oss;
                oss << "  - " << conn.first << " <-> " << conn.second;
                error_msg += oss.str() + "\n";
            }
            std::cout << error_msg << std::endl;
        }

        // Fail the test if there are any mismatches
        if (!missing_in_gsd.empty() || !extra_in_gsd.empty()) {
            FAIL() << "Connection mismatch detected. Check console output for details.";
        }

        // If we get here, all connections match
        std::cout << "All connections match between FSD and GSD (" << generated_connections.size() << " connections)"
                  << std::endl;
    } else {
        for (const auto& conn : discovered_connections) {
            EXPECT_TRUE(generated_connections.find(conn) != generated_connections.end())
                << "Connection not found in FSD: " << conn.first << " <-> " << conn.second;
        }
    }
}

TEST(Cluster, TestFactorySystemDescriptor) {
    std::vector<std::string> hostnames = {
        "metal-wh-09",
        "metal-wh-10",
        "metal-wh-11",
        "metal-wh-12",
        "metal-wh-18",
        "metal-wh-08",
        "metal-wh-01",
        "metal-wh-02",
        "metal-wh-13",
        "metal-wh-14",
        "metal-wh-15",
        "metal-wh-16",
        "metal-wh-03",
        "metal-wh-04",
        "metal-wh-05",
        "metal-wh-06",
    };
    // Indexed by superpod, the pod
    std::map<uint32_t, std::map<uint32_t, uint32_t>> host_ids_map = {
        {1,
         {
             {1, 0},
             {2, 1},
             {3, 2},
             {4, 3},
         }},
        {2,
         {
             {1, 4},
             {2, 5},
             {3, 6},
             {4, 7},
         }},
        {3,
         {
             {1, 8},
             {2, 9},
             {3, 10},
             {4, 11},
         }},
        {4,
         {
             {1, 12},
             {2, 13},
             {3, 14},
             {4, 15},
         }},
    };

    // Create board templates using factory function
    std::unordered_map<std::string, Board> board_templates;
    board_templates.emplace("N300", create_n300_board());

    // Test 0: Single board level (n300_board.yaml)
    {
        // For single board, we only need internal connections
        uint32_t host_id = host_ids_map.begin()->second.begin()->second;
        uint32_t tray_id = 1;

        // Create board type mapping (host_id is index into hostnames list)
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;
        board_types.emplace_back(host_id, tray_id, "N300");

        auto board_connections =
            HierarchicalSystemBuilder::build_board_connections(board_templates.at("N300"), host_id, tray_id);

        SystemDescriptorEmitter::emit_factory_system_descriptor(
            "factory_system_descriptor_board.yaml", board_connections, board_types, hostnames);
    }

    // Test 1: Pod level (n300_t3k_pod.yaml)
    YAML::Node pod_config = YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/n300_t3k_pod.yaml");
    {
        std::unordered_map<uint32_t, Board> pod_boards;
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        // For pod level, use hostnames from first superpod (index 0-3)
        uint32_t host_id = host_ids_map.begin()->second.begin()->second;  // First superpod, first pod
        for (const auto& board_config : pod_config["PODS"]["N300_T3K_POD"]["BOARDS"]) {
            uint32_t board_id = board_config.first.as<uint32_t>();
            std::string board_type = board_config.second.as<std::string>();
            pod_boards.emplace(board_id, board_templates.at(board_type));
            board_types.emplace_back(host_id, board_id, board_type);
        }

        auto pod_connections = HierarchicalSystemBuilder::build_pod_connections(
            pod_boards,
            pod_config["PODS"]["N300_T3K_POD"],
            host_id  // Use host_id from host_ids_map
        );

        SystemDescriptorEmitter::emit_factory_system_descriptor(
            "factory_system_descriptor_pod.yaml", pod_connections, board_types, hostnames);
    }

    // Test 2: Superpod level (n300_t3k_superpod.yaml)
    YAML::Node superpod_config =
        YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/n300_t3k_superpod.yaml");
    {
        std::unordered_map<uint32_t, Pod> superpod_pods;
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        for (const auto& pod_config_item : superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"]["PODS"]) {
            uint32_t pod_id = pod_config_item.first.as<uint32_t>();
            std::string pod_type = pod_config_item.second.as<std::string>();

            Pod& pod = superpod_pods[pod_id];

            // Create boards for this pod
            // For superpod level, use hostnames from first superpod (index 0-3)
            uint32_t host_id = host_ids_map.begin()->second.at(pod_id);  // First superpod
            for (const auto& board_config : pod_config["PODS"]["N300_T3K_POD"]["BOARDS"]) {
                uint32_t board_id = board_config.first.as<uint32_t>();
                std::string board_type = board_config.second.as<std::string>();
                pod.boards.emplace(board_id, board_templates.at(board_type));
                board_types.emplace_back(host_id, board_id, board_type);
            }
        }

        auto superpod_connections = HierarchicalSystemBuilder::build_superpod_connections(
            superpod_pods,
            superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"],
            pod_config["PODS"]["N300_T3K_POD"],
            host_ids_map.at(1));

        SystemDescriptorEmitter::emit_factory_system_descriptor(
            "factory_system_descriptor_superpod.yaml", superpod_connections, board_types, hostnames);
    }

    // Test 3: Cluster level (n300_t3k_cluster.yaml)
    YAML::Node cluster_config =
        YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/n300_t3k_cluster.yaml");
    {
        std::unordered_map<uint32_t, Superpod> cluster_superpods;
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        for (const auto& superpod_config_item : cluster_config["CLUSTERS"]["N300_T3K_CLUSTER"]["SUPERPODS"]) {
            uint32_t superpod_id = superpod_config_item.first.as<uint32_t>();
            std::string superpod_type = superpod_config_item.second.as<std::string>();

            Superpod& superpod = cluster_superpods[superpod_id];

            // Create pods for this superpod
            for (const auto& pod_config_item : superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"]["PODS"]) {
                uint32_t pod_id = pod_config_item.first.as<uint32_t>();
                std::string pod_type = pod_config_item.second.as<std::string>();

                Pod& pod = superpod.pods[pod_id];

                // Create boards for this pod
                // Use host_ids_map to get the correct host_id for this superpod/pod combination
                uint32_t host_id = host_ids_map.at(superpod_id).at(pod_id);
                for (const auto& board_config : pod_config["PODS"]["N300_T3K_POD"]["BOARDS"]) {
                    uint32_t board_id = board_config.first.as<uint32_t>();
                    std::string board_type = board_config.second.as<std::string>();
                    pod.boards.emplace(board_id, board_templates.at(board_type));
                    board_types.emplace_back(host_id, board_id, board_type);
                }
            }
        }

        auto cluster_connections = HierarchicalSystemBuilder::build_cluster_connections(
            cluster_superpods,
            cluster_config["CLUSTERS"]["N300_T3K_CLUSTER"],
            superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"],
            pod_config["PODS"]["N300_T3K_POD"],
            host_ids_map);

        SystemDescriptorEmitter::emit_factory_system_descriptor(
            "factory_system_descriptor_cluster.yaml", cluster_connections, board_types, hostnames);
    }
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    std::vector<std::string> hostnames = {
        "metal-wh-03_1",
        "metal-wh-08_0",
        "metal-wh-02_3",
        "metal-wh-01_4",
        "metal-wh-05_2",
    };
    // Indexed by superpod, the pod
    std::map<uint32_t, std::map<uint32_t, uint32_t>> host_ids_map = {
        {1,
         {
             {1, 0},
             {2, 1},
             {3, 2},
             {4, 3},
             {5, 4},
         }},
    };

    // Create board templates using factory function
    std::unordered_map<std::string, Board> board_templates;
    board_templates.emplace("N300", create_n300_board());

    // Load the 5LB configuration
    YAML::Node superpod_config = YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/n300-5lb.yaml");
    YAML::Node pod_config = YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/n300_t3k_pod.yaml");

    // Build the superpod connections
    std::unordered_map<uint32_t, Pod> superpod_pods;
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

    for (const auto& pod_config_item : superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"]["PODS"]) {
        uint32_t pod_id = pod_config_item.first.as<uint32_t>();
        std::string pod_type = pod_config_item.second.as<std::string>();

        Pod& pod = superpod_pods[pod_id];

        // Create boards for this pod
        uint32_t host_id = host_ids_map.at(1).at(pod_id);
        for (const auto& board_config : pod_config["PODS"]["N300_T3K_POD"]["BOARDS"]) {
            uint32_t board_id = board_config.first.as<uint32_t>();
            std::string board_type = board_config.second.as<std::string>();
            pod.boards.emplace(board_id, board_templates.at(board_type));
            board_types.emplace_back(host_id, board_id, board_type);
        }
    }

    auto superpod_connections = HierarchicalSystemBuilder::build_superpod_connections(
        superpod_pods,
        superpod_config["SUPERPODS"]["N300_T3K_SUPERPOD"],
        pod_config["PODS"]["N300_T3K_POD"],
        host_ids_map.at(1));

    // Generate the FSD
    SystemDescriptorEmitter::emit_factory_system_descriptor(
        "factory_system_descriptor_5lb.yaml", superpod_connections, board_types, hostnames);

    // Read the generated FSD
    YAML::Node generated_fsd = YAML::LoadFile("factory_system_descriptor_5lb.yaml");

    // Read the discovered GSD (Global System Descriptor)
    YAML::Node discovered_gsd = YAML::LoadFile("5_t3k_physical_desc.yaml");

    // Compare the FSD with the discovered GSD using the common utility function
    validate_fsd_against_gsd("factory_system_descriptor_5lb.yaml", "5_t3k_physical_desc.yaml", hostnames, true);
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorusSuperpod) {
    std::vector<std::string> hostnames = {
        "wh-glx-a03u02_0",
        "wh-glx-a03u08_1",
        "wh-glx-a03u14_2",
        "wh-glx-a04u02_3",
        "wh-glx-a04u08_4",
    };

    // Indexed by superpod, the pod
    std::map<uint32_t, std::map<uint32_t, uint32_t>> host_ids_map = {
        {1,
         {
             {1, 0},
             {2, 1},
             {3, 2},
             {4, 3},
             {5, 4},
         }},
    };

    // Create board templates using factory function
    std::unordered_map<std::string, Board> board_templates;
    board_templates.emplace("UBB", create_wh_ubb_board());

    // Load the 5WH_GALAXY_Y_TORUS_SUPERPOD configuration
    YAML::Node superpod_config =
        YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/5_wh_galaxy_y_torus_superpod.yaml");
    YAML::Node pod_config = YAML::LoadFile("tests/tt_metal/tt_fabric/system_health/descriptors/wh_galaxy_y_torus.yaml");

    // Build the superpod connections
    std::unordered_map<uint32_t, Pod> superpod_pods;
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

    for (const auto& pod_config_item : superpod_config["SUPERPODS"]["WH_GALAXY_Y_TORUS_SUPERPOD"]["PODS"]) {
        uint32_t pod_id = pod_config_item.first.as<uint32_t>();
        std::string pod_type = pod_config_item.second.as<std::string>();

        Pod& pod = superpod_pods[pod_id];

        // Create boards for this pod
        uint32_t host_id = host_ids_map.at(1).at(pod_id);
        for (const auto& board_config : pod_config["PODS"]["WH_GALAXY_POD"]["BOARDS"]) {
            uint32_t board_id = board_config.first.as<uint32_t>();
            std::string board_type = board_config.second.as<std::string>();
            pod.boards.emplace(board_id, board_templates.at(board_type));
            board_types.emplace_back(host_id, board_id, board_type);
        }
    }

    auto superpod_connections = HierarchicalSystemBuilder::build_superpod_connections(
        superpod_pods,
        superpod_config["SUPERPODS"]["WH_GALAXY_Y_TORUS_SUPERPOD"],
        pod_config["PODS"]["WH_GALAXY_POD"],
        host_ids_map.at(1));

    // Generate the FSD
    SystemDescriptorEmitter::emit_factory_system_descriptor(
        "factory_system_descriptor_5wh_galaxy_y_torus_superpod.yaml", superpod_connections, board_types, hostnames);

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "factory_system_descriptor_5wh_galaxy_y_torus_superpod.yaml",
        "5_6u_physical_system_descriptor.yaml",
        hostnames);
}

}  // namespace system_health_tests
}  // namespace tt::tt_fabric

//        const uint32_t number_chips_from_board = get_number_of_chips_from_board_type(board_type);
