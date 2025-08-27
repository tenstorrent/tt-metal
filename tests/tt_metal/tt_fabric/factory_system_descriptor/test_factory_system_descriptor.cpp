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

// Add protobuf includes
#include "factory_system_descriptor.pb.h"
#include "pod_config.pb.h"
#include "superpod_config.pb.h"
#include "cluster_config.pb.h"
#include "deployment.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>
#include "board/board.hpp"
#include <yaml-cpp/yaml.h>

namespace tt::tt_fabric {
namespace fsd_tests {

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

struct PhysicalChannelConnection {
    std::string hostname = "";
    uint32_t tray_id = 0;
    uint32_t asic_index = 0;
    uint32_t channel_id = 0;

    auto operator<=>(const PhysicalChannelConnection& other) const = default;
};

struct PhysicalPortConnection {
    std::string hostname = "";
    std::string aisle = "";
    uint32_t rack = 0;
    uint32_t shelf_u = 0;
    tt::tt_fabric::PortType port_type = tt::tt_fabric::PortType::TRACE;
    uint32_t port_id = 0;

    auto operator<=>(const PhysicalPortConnection& other) const = default;
};

}  // namespace fsd_tests
}  // namespace tt::tt_fabric

// Overload operator<< for PhysicalChannelConnection to enable readable test output
namespace tt::tt_fabric::fsd_tests {
inline std::ostream& operator<<(std::ostream& os, const PhysicalChannelConnection& conn) {
    os << "PhysicalChannelConnection{hostname='" << conn.hostname << "', tray_id=" << conn.tray_id
       << ", asic_index=" << conn.asic_index << ", channel_id=" << conn.channel_id << "}";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const PhysicalPortConnection& conn) {
    os << "PhysicalPortConnection{hostname='" << conn.hostname << "', aisle='" << conn.aisle << "', rack=" << conn.rack
       << ", shelf_u=" << conn.shelf_u << ", port_type=" << enchantum::to_string(conn.port_type)
       << ", port_id=" << conn.port_id << "}";
    return os;
}
}  // namespace tt::tt_fabric::fsd_tests

// Overload std::hash for LogicalChipConnection
namespace std {
template <>
struct hash<tt::tt_fabric::fsd_tests::LogicalChipConnection> {
    std::size_t operator()(const tt::tt_fabric::fsd_tests::LogicalChipConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.host_id, conn.tray_id, conn.asic_channel.asic_index, conn.asic_channel.channel_id);
    }
};
template <>
struct hash<tt::tt_fabric::fsd_tests::PhysicalChannelConnection> {
    std::size_t operator()(const tt::tt_fabric::fsd_tests::PhysicalChannelConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.tray_id, conn.asic_index, conn.channel_id);
    }
};

template <>
struct hash<tt::tt_fabric::fsd_tests::PhysicalPortConnection> {
    std::size_t operator()(const tt::tt_fabric::fsd_tests::PhysicalPortConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.aisle, conn.rack, conn.shelf_u, conn.port_type, conn.port_id);
    }
};
}  // namespace std

namespace tt::tt_fabric {
namespace fsd_tests {

using LogicalChipConnectionPair = std::pair<LogicalChipConnection, LogicalChipConnection>;
using AsicChannelPair = std::pair<AsicChannel, AsicChannel>;

// TODO: Should look into cleaning this up
template <typename T>
struct Connector {
    static std::vector<AsicChannelPair> get_port_mapping(
        std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
        return T::get_port_mapping(start_channels, end_channels);
    }
};

struct LinearConnector : public Connector<LinearConnector> {
    static std::vector<AsicChannelPair> get_port_mapping(
        std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
        assert(start_channels.size() == end_channels.size());
        std::vector<AsicChannelPair> port_mapping;
        port_mapping.reserve(start_channels.size());
        for (size_t i = 0; i < start_channels.size(); i++) {
            port_mapping.emplace_back(start_channels[i], end_channels[i]);
        }
        return port_mapping;
    }
};

struct TraceConnector : public LinearConnector {};

struct QSFPConnector : public LinearConnector {};

struct LinkingBoard1Connector : public LinearConnector {};

struct LinkingBoard2Connector : public LinearConnector {};

struct LinkingBoard3Connector : public LinearConnector {};

struct Warp100Connector : public LinearConnector {};

struct Warp400Connector : public Connector<Warp400Connector> {
    static std::vector<AsicChannelPair> get_port_mapping(
        std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
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

std::vector<AsicChannelPair> get_asic_channel_connections(
    PortType port_type, std::vector<AsicChannel> start_channels, std::vector<AsicChannel> end_channels) {
    switch (port_type) {
        case PortType::QSFP: return QSFPConnector::get_port_mapping(start_channels, end_channels);
        case PortType::WARP100: return Warp100Connector::get_port_mapping(start_channels, end_channels);
        case PortType::WARP400: return Warp400Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_1: return LinkingBoard1Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_2: return LinkingBoard2Connector::get_port_mapping(start_channels, end_channels);
        case PortType::LINKING_BOARD_3: return LinkingBoard3Connector::get_port_mapping(start_channels, end_channels);
        case PortType::TRACE: return TraceConnector::get_port_mapping(start_channels, end_channels);
    }
}

struct Pod {
    std::unordered_map<uint32_t, Board> boards;
};

struct Superpod {
    std::unordered_map<uint32_t, Pod> pods;
};

struct Cluster {
    std::unordered_map<uint32_t, Superpod> superpods;
};

class HierarchicalSystemBuilder {
    enum class ConnectionResult { SUCCESS, BOARD_0_PORT_NOT_AVAILABLE, BOARD_1_PORT_NOT_AVAILABLE };

public:
    static std::pair<Board, std::vector<LogicalChipConnectionPair>> build_board_connections(
        const std::string& board_type, uint32_t host_id, uint32_t tray_id) {
        Board board = create_board(board_type);
        std::vector<LogicalChipConnectionPair> chip_connections;

        // Add internal connections within the board
        add_internal_board_connections(board, chip_connections, host_id, tray_id);

        return std::make_pair(board, chip_connections);
    }

    static std::pair<Pod, std::vector<LogicalChipConnectionPair>> build_pod_connections(
        const tt::fsd::proto::PodDescriptor& pod_descriptor, uint32_t host_id) {
        std::vector<LogicalChipConnectionPair> chip_connections;
        Pod pod = Pod();

        // Add internal connections within each board
        for (const auto& board_item : pod_descriptor.boards().board()) {
            int32_t board_id = board_item.tray_id();
            const std::string& board_type = board_item.board_type();
            auto [board, board_connections] = build_board_connections(board_type, host_id, board_id);
            pod.boards.emplace(board_id, std::move(board));
            chip_connections.insert(chip_connections.end(), board_connections.begin(), board_connections.end());
        }

        // Add inter-board connections
        add_inter_board_connections(pod, pod_descriptor, chip_connections, host_id);

        return std::make_pair(pod, chip_connections);
    }

    static std::pair<Superpod, std::vector<LogicalChipConnectionPair>> build_superpod_connections(
        const tt::fsd::proto::SuperPodDescriptor& superpod_descriptor,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        std::vector<LogicalChipConnectionPair> chip_connections;
        Superpod superpod;

        // Add connections from each pod (these already include internal connections)
        for (const auto& pod_item : superpod_descriptor.pods().pod()) {
            uint32_t pod_id = pod_item.pod_id();
            const std::string& pod_type = pod_item.pod_type();

            // Load the pod descriptor based on pod_type
            auto pod_descriptor = load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
                "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + pod_type + ".textproto");

            auto [pod, pod_connections] = build_pod_connections(pod_descriptor, host_ids_map.at(pod_id));
            superpod.pods.emplace(pod_id, std::move(pod));
            chip_connections.insert(chip_connections.end(), pod_connections.begin(), pod_connections.end());
        }

        // Add inter-pod connections
        add_inter_pod_connections(superpod, superpod_descriptor, chip_connections, host_ids_map);

        return std::make_pair(superpod, chip_connections);
    }

    static std::pair<Cluster, std::vector<LogicalChipConnectionPair>> build_cluster_connections(
        const tt::fsd::proto::ClusterDescriptor& cluster_descriptor,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        std::vector<LogicalChipConnectionPair> chip_connections;
        Cluster cluster;

        // Add connections from each superpod (these already include internal connections)
        for (const auto& superpod_item : cluster_descriptor.superpods().superpod()) {
            uint32_t superpod_id = superpod_item.superpod_id();
            const std::string& superpod_type = superpod_item.superpod_type();

            // Load the superpod descriptor based on superpod_type
            auto superpod_descriptor = load_descriptor_from_textproto<tt::fsd::proto::SuperPodDescriptor>(
                "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + superpod_type +
                ".textproto");

            auto [superpod, superpod_connections] =
                build_superpod_connections(superpod_descriptor, host_ids_map.at(superpod_id));
            cluster.superpods.emplace(superpod_id, std::move(superpod));
            chip_connections.insert(chip_connections.end(), superpod_connections.begin(), superpod_connections.end());
        }

        // Add inter-superpod connections
        add_inter_superpod_connections(cluster, cluster_descriptor, chip_connections, host_ids_map);

        return std::make_pair(cluster, chip_connections);
    }

private:
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

    static void add_inter_board_connections(
        Pod& pod,
        const tt::fsd::proto::PodDescriptor& pod_descriptor,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_id) {
        for (const auto& port_type_connection : pod_descriptor.port_type_connections()) {
            const std::string& port_type_str = port_type_connection.first;
            auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_type_str);
            }

            for (const auto& connection : port_type_connection.second.connections()) {
                connect_boards_from_protobuf(*port_type, connection, pod.boards, chip_connections, host_id);
            }
        }
    }

    static void add_inter_pod_connections(
        Superpod& superpod,
        const tt::fsd::proto::SuperPodDescriptor& superpod_descriptor,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        for (const auto& port_type_connection : superpod_descriptor.port_type_connections()) {
            const std::string& port_type_str = port_type_connection.first;
            auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_type_str);
            }

            for (const auto& connection : port_type_connection.second.connections()) {
                connect_pods_from_protobuf(*port_type, connection, superpod.pods, chip_connections, host_ids_map);
            }
        }
    }

    static void add_inter_superpod_connections(
        Cluster& cluster,
        const tt::fsd::proto::ClusterDescriptor& cluster_descriptor,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        for (const auto& port_type_connection : cluster_descriptor.port_type_connections()) {
            const std::string& port_type_str = port_type_connection.first;
            auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_type_str);
            }

            for (const auto& connection : port_type_connection.second.connections()) {
                connect_superpods_from_protobuf(
                    *port_type, connection, cluster.superpods, chip_connections, host_ids_map);
            }
        }
    }

    static ConnectionResult connect_boards(
        Board& board_0,
        Board& board_1,
        PortType port_type,
        uint32_t port_0_id,
        uint32_t port_1_id,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_0_id,
        uint32_t host_1_id,
        uint32_t board_0_id,
        uint32_t board_1_id) {
        // Verify ports are available
        const auto& board_0_available = board_0.get_available_port_ids(port_type);
        const auto& board_1_available = board_1.get_available_port_ids(port_type);

        if (std::find(board_0_available.begin(), board_0_available.end(), port_0_id) == board_0_available.end()) {
            return ConnectionResult::BOARD_0_PORT_NOT_AVAILABLE;
        }

        if (std::find(board_1_available.begin(), board_1_available.end(), port_1_id) == board_1_available.end()) {
            return ConnectionResult::BOARD_1_PORT_NOT_AVAILABLE;
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
                LogicalChipConnection{.host_id = host_0_id, .tray_id = board_0_id, .asic_channel = start_channel},
                LogicalChipConnection{.host_id = host_1_id, .tray_id = board_1_id, .asic_channel = end_channel});
        }

        return ConnectionResult::SUCCESS;
    }

    static void connect_boards_from_protobuf(
        PortType port_type,
        const tt::fsd::proto::PodDescriptor::Connection& connection,
        std::unordered_map<uint32_t, Board>& boards,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        uint32_t host_id) {
        uint32_t board_0_id = connection.port_a().tray_id();
        uint32_t port_0_id = connection.port_a().port_id();
        uint32_t board_1_id = connection.port_b().tray_id();
        uint32_t port_1_id = connection.port_b().port_id();

        Board& board_0 = boards.at(board_0_id);
        Board& board_1 = boards.at(board_1_id);

        auto result = connect_boards(
            board_0,
            board_1,
            port_type,
            port_0_id,
            port_1_id,
            chip_connections,
            host_id,
            host_id,
            board_0_id,
            board_1_id);
        if (result != ConnectionResult::SUCCESS) {
            if (result == ConnectionResult::BOARD_0_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id));
            } else if (result == ConnectionResult::BOARD_1_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id));
            } else {
                throw std::runtime_error("Unknown connection result");
            }
        }
    }

    static void connect_pods_from_protobuf(
        PortType port_type,
        const tt::fsd::proto::SuperPodDescriptor::Connection& connection,
        std::unordered_map<uint32_t, Pod>& pods,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, uint32_t>& host_ids_map) {
        uint32_t pod_0_id = connection.port_a().pod_id();
        uint32_t board_0_id = connection.port_a().tray_id();
        uint32_t port_0_id = connection.port_a().port_id();
        uint32_t pod_1_id = connection.port_b().pod_id();
        uint32_t board_1_id = connection.port_b().tray_id();
        uint32_t port_1_id = connection.port_b().port_id();

        // Get host IDs for both pods
        uint32_t host_0_id = host_ids_map.at(pod_0_id);
        uint32_t host_1_id = host_ids_map.at(pod_1_id);

        Board& board_0 = pods.at(pod_0_id).boards.at(board_0_id);
        Board& board_1 = pods.at(pod_1_id).boards.at(board_1_id);

        auto result = connect_boards(
            board_0,
            board_1,
            port_type,
            port_0_id,
            port_1_id,
            chip_connections,
            host_0_id,
            host_1_id,
            board_0_id,
            board_1_id);
        if (result != ConnectionResult::SUCCESS) {
            if (result == ConnectionResult::BOARD_0_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id) +
                    " in pod " + std::to_string(pod_0_id));
            } else if (result == ConnectionResult::BOARD_1_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id) +
                    " in pod " + std::to_string(pod_1_id));
            } else {
                throw std::runtime_error("Unknown connection result");
            }
        }
    }

    static void connect_superpods_from_protobuf(
        PortType port_type,
        const tt::fsd::proto::ClusterDescriptor::Connection& connection,
        std::unordered_map<uint32_t, Superpod>& superpods,
        std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::map<uint32_t, std::map<uint32_t, uint32_t>>& host_ids_map) {
        uint32_t superpod_0_id = connection.port_a().superpod_id();
        uint32_t pod_0_id = connection.port_a().pod_id();
        uint32_t board_0_id = connection.port_a().tray_id();
        uint32_t port_0_id = connection.port_a().port_id();
        uint32_t superpod_1_id = connection.port_b().superpod_id();
        uint32_t pod_1_id = connection.port_b().pod_id();
        uint32_t board_1_id = connection.port_b().tray_id();
        uint32_t port_1_id = connection.port_b().port_id();

        // Get host IDs for both superpod/pod combinations
        uint32_t host_0_id = host_ids_map.at(superpod_0_id).at(pod_0_id);
        uint32_t host_1_id = host_ids_map.at(superpod_1_id).at(pod_1_id);

        Board& board_0 = superpods.at(superpod_0_id).pods.at(pod_0_id).boards.at(board_0_id);
        Board& board_1 = superpods.at(superpod_1_id).pods.at(pod_1_id).boards.at(board_1_id);

        auto result = connect_boards(
            board_0,
            board_1,
            port_type,
            port_0_id,
            port_1_id,
            chip_connections,
            host_0_id,
            host_1_id,
            board_0_id,
            board_1_id);
        if (result != ConnectionResult::SUCCESS) {
            if (result == ConnectionResult::BOARD_0_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_0_id) + " not available on board " + std::to_string(board_0_id) +
                    " in pod " + std::to_string(pod_0_id) + " in superpod " + std::to_string(superpod_0_id));
            } else if (result == ConnectionResult::BOARD_1_PORT_NOT_AVAILABLE) {
                throw std::runtime_error(
                    "Port " + std::to_string(port_1_id) + " not available on board " + std::to_string(board_1_id) +
                    " in pod " + std::to_string(pod_1_id) + " in superpod " + std::to_string(superpod_1_id));
            } else {
                throw std::runtime_error("Unknown connection result");
            }
        }
    }

public:
    // Helper functions to load protobuf descriptors from textproto files
    template <typename Descriptor>
    static Descriptor load_descriptor_from_textproto(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        Descriptor descriptor;

        if (!google::protobuf::TextFormat::ParseFromString(content, &descriptor)) {
            throw std::runtime_error("Failed to parse textproto file: " + file_path);
        }

        return descriptor;
    }
};

// SystemDescriptorEmitter class for generating factory system descriptors
class SystemDescriptorEmitter {
public:
    static void emit_textproto_factory_system_descriptor(
        const std::string& output_path,
        const std::vector<LogicalChipConnectionPair>& chip_connections,
        const std::vector<std::tuple<uint32_t, uint32_t, std::string>>& host_tray_to_board_type,
        const tt::deployment::DeploymentDescriptor& deployment_descriptor) {
        tt::fsd::proto::FactorySystemDescriptor fsd;

        // Add hostnames
        for (const auto& host : deployment_descriptor.hosts()) {
            fsd.mutable_hostnames()->add_hostname(host.host());
        }

        // Add board types
        for (const auto& [host_id, tray_id, board_type] : host_tray_to_board_type) {
            auto* board_location = fsd.mutable_board_types()->add_board_locations();
            board_location->set_host_id(host_id);
            board_location->set_tray_id(tray_id);
            board_location->set_board_type(board_type);
        }

        // Add ASIC connections
        for (const auto& [start, end] : chip_connections) {
            auto* connection = fsd.mutable_eth_connections()->add_connection();

            auto* endpoint_a = connection->mutable_endpoint_a();
            endpoint_a->set_host_id(start.host_id);
            endpoint_a->set_tray_id(start.tray_id);
            endpoint_a->set_asic_index(start.asic_channel.asic_index);
            endpoint_a->set_chan_id(start.asic_channel.channel_id);

            auto* endpoint_b = connection->mutable_endpoint_b();
            endpoint_b->set_host_id(end.host_id);
            endpoint_b->set_tray_id(end.tray_id);
            endpoint_b->set_asic_index(end.asic_channel.asic_index);
            endpoint_b->set_chan_id(end.asic_channel.channel_id);
        }

        // Write the protobuf message to textproto format
        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }

        std::string textproto_string;
        google::protobuf::TextFormat::Printer printer;
        printer.SetUseShortRepeatedPrimitives(true);
        printer.SetUseUtf8StringEscaping(true);
        printer.SetSingleLineMode(false);  // Enable multiline output
        printer.SetInitialIndentLevel(0);  // Start with no indentation

        if (!printer.PrintToString(fsd, &textproto_string)) {
            throw std::runtime_error("Failed to convert protobuf to textproto string");
        }

        // Write the string to file
        output_file << textproto_string;
        if (output_file.fail()) {
            throw std::runtime_error("Failed to write textproto to file: " + output_path);
        }
        output_file.close();
    }
};

// Common utility function for validating FSD against discovered GSD
void validate_fsd_against_gsd(
    const std::string& fsd_filename,
    const std::string& gsd_filename,
    const tt::deployment::DeploymentDescriptor& deployment_descriptor,
    bool strict_validation = true) {
    // Read the generated FSD using protobuf
    tt::fsd::proto::FactorySystemDescriptor generated_fsd;
    std::ifstream fsd_file(fsd_filename);
    if (!fsd_file.is_open()) {
        FAIL() << "Failed to open FSD file: " << fsd_filename;
    }

    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();

    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &generated_fsd)) {
        FAIL() << "Failed to parse FSD protobuf from file: " << fsd_filename;
    }

    const auto& hosts = deployment_descriptor.hosts();

    // Read the discovered GSD (Global System Descriptor) - still using YAML
    YAML::Node discovered_gsd = YAML::LoadFile(gsd_filename);

    // Compare the FSD with the discovered GSD
    // First, compare hostnames
    ASSERT_TRUE(generated_fsd.has_hostnames()) << "FSD missing hostnames";

    // Handle the new GSD structure with compute_node_specs
    ASSERT_TRUE(discovered_gsd["compute_node_specs"]) << "GSD missing compute_node_specs";
    YAML::Node asic_info_node = discovered_gsd["compute_node_specs"];

    // Check that all discovered hostnames are present in the generated FSD
    std::set<std::string> generated_hostnames;
    for (const auto& hostname : generated_fsd.hostnames().hostname()) {
        generated_hostnames.insert(hostname);
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
    ASSERT_TRUE(generated_fsd.has_board_types()) << "FSD missing board_types";
    std::set<std::tuple<uint32_t, uint32_t, std::string>> generated_board_types;
    for (const auto& board_location : generated_fsd.board_types().board_locations()) {
        uint32_t host_id = board_location.host_id();
        uint32_t tray_id = board_location.tray_id();
        const std::string& board_type_name = board_location.board_type();
        generated_board_types.insert(std::make_tuple(host_id, tray_id, board_type_name));
    }

    // Strict validation: Each host, tray combination should have the same board type between FSD and GSD
    std::map<std::pair<std::string, uint32_t>, std::string> fsd_board_types;

    // Extract board types from FSD
    for (const auto& board_location : generated_fsd.board_types().board_locations()) {
        uint32_t host_id = board_location.host_id();
        uint32_t tray_id = board_location.tray_id();
        const std::string& board_type_name = board_location.board_type();
        fsd_board_types[std::make_pair(hosts[host_id].host(), tray_id)] = board_type_name;
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
    ASSERT_TRUE(generated_fsd.has_eth_connections()) << "FSD missing eth_connections";

    // Determine which connection types exist in the discovered GSD
    bool has_local_eth_connections =
        discovered_gsd["local_eth_connections"] && !discovered_gsd["local_eth_connections"].IsNull();
    bool has_global_eth_connections =
        discovered_gsd["global_eth_connections"] && !discovered_gsd["global_eth_connections"].IsNull();

    // At least one connection type should exist
    ASSERT_TRUE(has_local_eth_connections || has_global_eth_connections)
        << "No connection types found in discovered GSD";

    // Convert generated connections to a comparable format
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> generated_connections;
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> duplicate_generated_connections;

    for (const auto& connection : generated_fsd.eth_connections().connection()) {
        const auto& endpoint_a = connection.endpoint_a();
        const auto& endpoint_b = connection.endpoint_b();

        uint32_t host_id_1 = endpoint_a.host_id();
        uint32_t tray_id_1 = endpoint_a.tray_id();
        uint32_t asic_index_1 = endpoint_a.asic_index();
        uint32_t chan_id_1 = endpoint_a.chan_id();

        uint32_t host_id_2 = endpoint_b.host_id();
        uint32_t tray_id_2 = endpoint_b.tray_id();
        uint32_t asic_index_2 = endpoint_b.asic_index();
        uint32_t chan_id_2 = endpoint_b.chan_id();

        const std::string& hostname_1 = hosts[host_id_1].host();
        const std::string& hostname_2 = hosts[host_id_2].host();

        PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
        PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

        // Sort to ensure consistent ordering
        std::pair<PhysicalChannelConnection, PhysicalChannelConnection> connection_pair_sorted;
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
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> discovered_connections;
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> duplicate_discovered_connections;

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

            PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
            PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

            // Sort to ensure consistent ordering
            std::pair<PhysicalChannelConnection, PhysicalChannelConnection> connection_pair_sorted;
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

            PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_index_1, chan_id_1};
            PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_index_2, chan_id_2};

            // Sort to ensure consistent ordering
            std::pair<PhysicalChannelConnection, PhysicalChannelConnection> connection_pair_sorted;
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
        std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> missing_in_gsd;
        std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> extra_in_gsd;

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
            // Collect and display port information for all missing/extra connections
            std::set<std::pair<PhysicalPortConnection, PhysicalPortConnection>> missing_port_info;

            // Process missing connections
            for (const auto& conn : missing_in_gsd) {
                // Get board types from FSD for both connections independently
                std::string board_type_a = "";
                std::string board_type_b = "";

                // Find host_id for each connection by matching hostname
                uint32_t host_id_a = 0;
                uint32_t host_id_b = 0;
                for (uint32_t i = 0; i < hosts.size(); ++i) {
                    if (hosts[i].host() == conn.first.hostname) {
                        host_id_a = i;
                    }
                    if (hosts[i].host() == conn.second.hostname) {
                        host_id_b = i;
                    }
                }

                // Look up board types for each connection
                for (const auto& board_location : generated_fsd.board_types().board_locations()) {
                    if (board_location.host_id() == host_id_a && board_location.tray_id() == conn.first.tray_id) {
                        board_type_a = board_location.board_type();
                    }
                    if (board_location.host_id() == host_id_b && board_location.tray_id() == conn.second.tray_id) {
                        board_type_b = board_location.board_type();
                    }
                }
                if (board_type_a.empty() || board_type_b.empty()) {
                    throw std::runtime_error(
                        "Board type not found for connection: " + conn.first.hostname + " <-> " + conn.second.hostname);
                }

                Board board_a = create_board(board_type_a);
                Board board_b = create_board(board_type_b);
                auto port_a = board_a.get_port_for_asic_channel({conn.first.asic_index, conn.first.channel_id});
                auto port_b = board_b.get_port_for_asic_channel({conn.second.asic_index, conn.second.channel_id});

                PhysicalPortConnection port_a_conn;
                PhysicalPortConnection port_b_conn;

                // Add deployment info for first connection if available
                for (const auto& host : hosts) {
                    if (host.host() == conn.first.hostname) {
                        port_a_conn = PhysicalPortConnection{
                            conn.first.hostname,
                            host.aisle(),
                            host.rack(),
                            host.shelf_u(),
                            port_a.port_type,
                            port_a.port_id};
                        break;
                    }
                }

                // Add deployment info for second connection if available
                for (const auto& host : hosts) {
                    if (host.host() == conn.second.hostname) {
                        port_b_conn = PhysicalPortConnection{
                            conn.second.hostname,
                            host.aisle(),
                            host.rack(),
                            host.shelf_u(),
                            port_b.port_type,
                            port_b.port_id};
                        break;
                    }
                }

                missing_port_info.insert(std::make_pair(port_a_conn, port_b_conn));
            }
            std::ostringstream oss;
            oss << "Channel Connections found in FSD but missing in GSD (" << std::to_string(missing_in_gsd.size())
                << " connections):\n";
            for (const auto& conn : missing_in_gsd) {
                oss << "  - " << conn.first << " <-> " << conn.second << "\n";
            }
            oss << "\n";

            oss << "Port Connections found in FSD but missing in GSD ("
                << std::to_string(missing_port_info.size()) + " connections):\n";
            for (const auto& conn : missing_port_info) {
                oss << "  - " << conn.first << " <-> " << conn.second << "\n";
            }
            std::cout << oss.str() << std::endl;
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
    auto deployment_descriptor = tt::tt_fabric::fsd_tests::HierarchicalSystemBuilder::load_descriptor_from_textproto<
        tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/16_lb_deployment.textproto");
    std::unordered_map<uint64_t, Board> board_map;

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

    // Test 0: Single board level
    {
        // For single board, we only need internal connections
        uint32_t host_id = host_ids_map.begin()->second.begin()->second;
        uint32_t tray_id = 1;

        // Create board type mapping (host_id is index into hostnames list)
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;
        board_types.emplace_back(host_id, tray_id, "N300");

        auto [board, board_connections] = HierarchicalSystemBuilder::build_board_connections("N300", host_id, tray_id);

        SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
            "fsd/factory_system_descriptor_board.textproto", board_connections, board_types, deployment_descriptor);
    }

    // Test 1: Pod level (n300_t3k_pod.textproto)
    auto pod_descriptor = HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/n300_t3k_pod.textproto");
    {
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        // For pod level, use hostnames from first superpod (index 0-3)
        uint32_t host_id = host_ids_map.begin()->second.begin()->second;  // First superpod, first pod

        // Create board types from protobuf data
        for (const auto& board : pod_descriptor.boards().board()) {
            uint32_t board_id = board.tray_id();
            const std::string& board_type = board.board_type();
            board_types.emplace_back(host_id, board_id, board_type);
        }

        auto [pod, pod_connections] = HierarchicalSystemBuilder::build_pod_connections(
            pod_descriptor,
            host_id  // Use host_id from host_ids_map
        );

        SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
            "fsd/factory_system_descriptor_pod.textproto", pod_connections, board_types, deployment_descriptor);
    }

    // Test 2: Superpod level (n300_t3k_superpod.textproto)
    auto superpod_descriptor =
        HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::SuperPodDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/n300_t3k_superpod.textproto");
    {
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        // Create board types from all pods in the superpod
        for (const auto& pod_item : superpod_descriptor.pods().pod()) {
            uint32_t pod_id = pod_item.pod_id();
            const std::string& pod_type = pod_item.pod_type();

            // Load the pod descriptor to get board information
            auto pod_descriptor =
                HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
                    "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + pod_type +
                    ".textproto");

            // For superpod level, use hostnames from first superpod (index 0-3)
            uint32_t host_id = host_ids_map.begin()->second.at(pod_id);  // First superpod
            for (const auto& board : pod_descriptor.boards().board()) {
                uint32_t board_id = board.tray_id();
                const std::string& board_type = board.board_type();
                board_types.emplace_back(host_id, board_id, board_type);
            }
        }

        auto [superpod, superpod_connections] =
            HierarchicalSystemBuilder::build_superpod_connections(superpod_descriptor, host_ids_map.at(1));

        SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
            "fsd/factory_system_descriptor_superpod.textproto",
            superpod_connections,
            board_types,
            deployment_descriptor);
    }

    // Test 3: Cluster level (n300_t3k_cluster.textproto)
    auto cluster_descriptor =
        HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::ClusterDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/n300_t3k_cluster.textproto");
    {
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

        // Create board types from all superpods in the cluster
        for (const auto& superpod_item : cluster_descriptor.superpods().superpod()) {
            uint32_t superpod_id = superpod_item.superpod_id();
            const std::string& superpod_type = superpod_item.superpod_type();

            // Load the superpod descriptor to get pod information
            auto superpod_descriptor =
                HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::SuperPodDescriptor>(
                    "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + superpod_type +
                    ".textproto");

            // Create pods for this superpod
            for (const auto& pod_item : superpod_descriptor.pods().pod()) {
                uint32_t pod_id = pod_item.pod_id();
                const std::string& pod_type = pod_item.pod_type();

                // Load the pod descriptor to get board information
                auto pod_descriptor =
                    HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
                        "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + pod_type +
                        ".textproto");

                // Create boards for this pod
                // Use host_ids_map to get the correct host_id for this superpod/pod combination
                uint32_t host_id = host_ids_map.at(superpod_id).at(pod_id);
                for (const auto& board : pod_descriptor.boards().board()) {
                    uint32_t board_id = board.tray_id();
                    const std::string& board_type = board.board_type();
                    board_types.emplace_back(host_id, board_id, board_type);
                }
            }
        }

        auto [cluster, cluster_connections] =
            HierarchicalSystemBuilder::build_cluster_connections(cluster_descriptor, host_ids_map);

        SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
            "fsd/factory_system_descriptor_cluster.textproto", cluster_connections, board_types, deployment_descriptor);
    }
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    // Load deployment descriptor for enhanced port information
    auto deployment_descriptor = tt::tt_fabric::fsd_tests::HierarchicalSystemBuilder::load_descriptor_from_textproto<
        tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/5_lb_deployment.textproto");

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

    // Load the 5LB configuration
    auto superpod_descriptor =
        HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::SuperPodDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/n300-5lb.textproto");

    // Build the superpod connections
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

    // Create board types from all pods in the superpod
    for (const auto& pod_item : superpod_descriptor.pods().pod()) {
        uint32_t pod_id = pod_item.pod_id();
        const std::string& pod_type = pod_item.pod_type();

        // Load the pod descriptor to get board information
        auto pod_descriptor = HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + pod_type + ".textproto");

        // For superpod level, use hostnames from first superpod (index 0-3)
        uint32_t host_id = host_ids_map.begin()->second.at(pod_id);  // First superpod
        for (const auto& board : pod_descriptor.boards().board()) {
            uint32_t board_id = board.tray_id();
            const std::string& board_type = board.board_type();
            board_types.emplace_back(host_id, board_id, board_type);
        }
    }

    auto [superpod, superpod_connections] =
        HierarchicalSystemBuilder::build_superpod_connections(superpod_descriptor, host_ids_map.at(1));

    // Generate the FSD (textproto format)
    SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
        "fsd/factory_system_descriptor_5lb.textproto", superpod_connections, board_types, deployment_descriptor);

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_5lb.textproto",
        "tests/tt_metal/tt_fabric/factory_system_descriptor/global_system_descriptors/5_lb_physical_desc.yaml",
        deployment_descriptor);
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorusSuperpod) {
    // Load deployment descriptor
    auto deployment_descriptor = tt::tt_fabric::fsd_tests::HierarchicalSystemBuilder::load_descriptor_from_textproto<
        tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/5_wh_galaxy_y_torus_deployment.textproto");

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

    // Load the 5LB configuration
    auto superpod_descriptor =
        HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::SuperPodDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/"
            "5_wh_galaxy_y_torus_superpod.textproto");

    // Build the superpod connections
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> board_types;

    // Create board types from all pods in the superpod
    for (const auto& pod_item : superpod_descriptor.pods().pod()) {
        uint32_t pod_id = pod_item.pod_id();
        const std::string& pod_type = pod_item.pod_type();

        // Load the pod descriptor to get board information
        auto pod_descriptor = HierarchicalSystemBuilder::load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
            "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/" + pod_type + ".textproto");

        // For superpod level, use hostnames from first superpod (index 0-3)
        uint32_t host_id = host_ids_map.begin()->second.at(pod_id);  // First superpod
        for (const auto& board : pod_descriptor.boards().board()) {
            uint32_t board_id = board.tray_id();
            const std::string& board_type = board.board_type();
            board_types.emplace_back(host_id, board_id, board_type);
        }
    }

    auto [superpod, superpod_connections] =
        HierarchicalSystemBuilder::build_superpod_connections(superpod_descriptor, host_ids_map.at(1));

    // Generate the FSD (textproto format)
    SystemDescriptorEmitter::emit_textproto_factory_system_descriptor(
        "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto",
        superpod_connections,
        board_types,
        deployment_descriptor);

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto",
        "tests/tt_metal/tt_fabric/factory_system_descriptor/global_system_descriptors/"
        "5_wh_galaxy_y_torus_physical_desc.yaml",
        deployment_descriptor);
}

}  // namespace fsd_tests
}  // namespace tt::tt_fabric
