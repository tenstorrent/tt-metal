// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <filesystem>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include <utility>
#include <set>
#include <cstdio>
#include <umd/device/types/cluster_descriptor_types.h>

// Add protobuf includes
#include "factory_system_descriptor.pb.h"
#include "pod_config.pb.h"
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
               asic_channel.asic_location == other.asic_channel.asic_location &&
               asic_channel.channel_id == other.asic_channel.channel_id;
    }
};

struct PhysicalChannelConnection {
    std::string hostname = "";
    uint32_t tray_id = 0;
    uint32_t asic_location = 0;
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
       << ", asic_location=" << conn.asic_location << ", channel_id=" << conn.channel_id << "}";
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
            conn.host_id, conn.tray_id, conn.asic_channel.asic_location, conn.asic_channel.channel_id);
    }
};
template <>
struct hash<tt::tt_fabric::fsd_tests::PhysicalChannelConnection> {
    std::size_t operator()(const tt::tt_fabric::fsd_tests::PhysicalChannelConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.tray_id, conn.asic_location, conn.channel_id);
    }
};
template <>
struct hash<std::pair<uint32_t, uint32_t>> {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
        return tt::stl::hash::hash_objects_with_default_seed(p.first, p.second);
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
        const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels) {
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
    PortType port_type, const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels) {
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
    uint32_t host_id = 0;
    // Board-to-board connections within this pod: PortType -> [(board_id, port_id) <-> (board_id, port_id)]
    std::unordered_map<PortType, std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>>>>
        inter_board_connections;
};

// Resolved graph instance with concrete pods
struct ResolvedGraphInstance {
    std::string template_name;
    std::string instance_name;
    std::unordered_map<std::string, Pod> pods;                                          // Direct pod children
    std::unordered_map<std::string, std::shared_ptr<ResolvedGraphInstance>> subgraphs;  // Nested graph children

    // All connections within this graph instance
    std::unordered_map<
        PortType,
        std::vector<std::pair<
            std::tuple<std::vector<std::string>, uint32_t, uint32_t>,  // Path, tray_id, port_id
            std::tuple<std::vector<std::string>, uint32_t, uint32_t>>>>
        internal_connections;
};

struct Cluster {
    std::unordered_map<std::string, tt::fsd::proto::GraphTemplate> graph_templates;
    std::shared_ptr<ResolvedGraphInstance> root_instance;
};

class CablingGenerator {
public:
    // Helper to load protobuf descriptors
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

    // Constructor
    CablingGenerator(
        const tt::fsd::proto::ClusterDescriptor& cluster_descriptor,
        const tt::deployment::DeploymentDescriptor& deployment_descriptor) {
        // Store deployment hosts
        deployment_hosts.assign(deployment_descriptor.hosts().begin(), deployment_descriptor.hosts().end());

        // Build cluster with all connections and port validation
        build_cluster_from_descriptor(cluster_descriptor);

        // Populate the boards_by_host_tray map
        populate_boards_by_host_tray();

        // Generate all logical chip connections
        generate_logical_chip_connections();
    }

    // Getters for all data
    const std::vector<tt::deployment::Host>& get_deployment_hosts() const { return deployment_hosts; }
    const Cluster& get_cluster() const { return cluster; }
    const std::unordered_map<std::pair<uint32_t, uint32_t>, Board*, std::hash<std::pair<uint32_t, uint32_t>>>&
    get_boards_by_host_tray() const {
        return boards_by_host_tray;
    }
    const std::vector<LogicalChipConnectionPair>& get_chip_connections() const { return chip_connections; }

    // Method to emit textproto factory system descriptor
    void emit_textproto_factory_system_descriptor(const std::string& output_path) const {
        tt::fsd::proto::FactorySystemDescriptor fsd;

        // Add host information with deployment details
        for (const auto& deployment_host : deployment_hosts) {
            auto* host = fsd.add_hosts();
            host->set_hostname(deployment_host.host());
            host->set_hall(deployment_host.hall());
            host->set_aisle(deployment_host.aisle());
            host->set_rack(deployment_host.rack());
            host->set_shelf_u(deployment_host.shelf_u());
        }

        // Add board types from boards_by_host_tray
        for (const auto& [host_tray_pair, board_ptr] : boards_by_host_tray) {
            uint32_t host_id = host_tray_pair.first;
            uint32_t tray_id = host_tray_pair.second;
            std::string board_type = std::string(enchantum::to_string(board_ptr->get_board_type()));

            auto* board_location = fsd.mutable_board_types()->add_board_locations();
            board_location->set_host_id(host_id);
            board_location->set_tray_id(tray_id);
            board_location->set_board_type(board_type);
        }

        // Add ASIC connections from chip_connections
        for (const auto& [start, end] : chip_connections) {
            auto* connection = fsd.mutable_eth_connections()->add_connection();

            auto* endpoint_a = connection->mutable_endpoint_a();
            endpoint_a->set_host_id(start.host_id);
            endpoint_a->set_tray_id(start.tray_id);
            endpoint_a->set_asic_location(start.asic_channel.asic_location);
            endpoint_a->set_chan_id(start.asic_channel.channel_id);

            auto* endpoint_b = connection->mutable_endpoint_b();
            endpoint_b->set_host_id(end.host_id);
            endpoint_b->set_tray_id(end.tray_id);
            endpoint_b->set_asic_location(end.asic_channel.asic_location);
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

private:
    // Build pod from descriptor with port connections and validation
    Pod build_pod(const tt::fsd::proto::PodDescriptor& pod_descriptor, uint32_t host_id) {
        Pod pod;
        pod.host_id = host_id;

        // Create boards with internal connections marked
        for (const auto& board_item : pod_descriptor.boards().board()) {
            uint32_t tray_id = board_item.tray_id();
            const std::string& board_type = board_item.board_type();
            pod.boards.emplace(tray_id, create_board(board_type));
        }

        // Add inter-board connections and validate/mark ports
        for (const auto& [port_type_str, port_connections] : pod_descriptor.port_type_connections()) {
            auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_type_str);
            }

            for (const auto& conn : port_connections.connections()) {
                uint32_t board_a_id = conn.port_a().tray_id();
                uint32_t port_a_id = conn.port_a().port_id();
                uint32_t board_b_id = conn.port_b().tray_id();
                uint32_t port_b_id = conn.port_b().port_id();

                // Validate and mark ports as used
                auto& board_a = pod.boards.at(board_a_id);
                auto& board_b = pod.boards.at(board_b_id);

                const auto& available_a = board_a.get_available_port_ids(*port_type);
                const auto& available_b = board_b.get_available_port_ids(*port_type);

                if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                    throw std::runtime_error(
                        "Port " + std::to_string(port_a_id) + " not available on board " + std::to_string(board_a_id));
                }
                if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                    throw std::runtime_error(
                        "Port " + std::to_string(port_b_id) + " not available on board " + std::to_string(board_b_id));
                }

                board_a.mark_port_used(*port_type, port_a_id);
                board_b.mark_port_used(*port_type, port_b_id);

                // Store connection
                pod.inter_board_connections[*port_type].emplace_back(
                    std::make_pair(board_a_id, port_a_id), std::make_pair(board_b_id, port_b_id));
            }
        }
        return pod;
    }

    // Build resolved graph instance from template and concrete host mappings
    std::shared_ptr<ResolvedGraphInstance> build_graph_instance(
        const tt::fsd::proto::GraphInstance& graph_instance, const std::string& instance_name = "root") {
        auto resolved = std::make_shared<ResolvedGraphInstance>();
        resolved->template_name = graph_instance.template_name();
        resolved->instance_name = instance_name;

        // Get the template definition
        const auto& template_def = cluster.graph_templates.at(graph_instance.template_name());

        // Build children based on template + instance mapping
        for (const auto& child_def : template_def.children()) {
            const std::string& child_name = child_def.name();
            const auto& child_mapping = graph_instance.child_mappings().at(child_name);

            if (child_def.has_pod_ref()) {
                // Leaf node - create pod
                if (child_mapping.mapping_case() != tt::fsd::proto::ChildMapping::kHostId) {
                    throw std::runtime_error("Pod child must have host_id mapping: " + child_name);
                }

                uint32_t host_id = child_mapping.host_id();
                const std::string& pod_descriptor_name = child_def.pod_ref().pod_descriptor();

                // Validate deployment pod type if specified
                if (host_id < deployment_hosts.size()) {
                    const auto& deployment_host = deployment_hosts[host_id];
                    if (!deployment_host.pod_type().empty() && deployment_host.pod_type() != pod_descriptor_name) {
                        throw std::runtime_error(
                            "Pod type mismatch for host " + deployment_host.host() + " (host_id " +
                            std::to_string(host_id) + "): " + "deployment specifies '" + deployment_host.pod_type() +
                            "' " + "but cluster configuration expects '" + pod_descriptor_name + "'");
                    }
                } else {
                    throw std::runtime_error("Host ID " + std::to_string(host_id) + " not found in deployment");
                }

                // Load pod descriptor and build pod
                auto pod_descriptor = load_descriptor_from_textproto<tt::fsd::proto::PodDescriptor>(
                    "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/instances/" +
                    pod_descriptor_name + ".textproto");
                resolved->pods[child_name] = build_pod(pod_descriptor, host_id);

            } else if (child_def.has_graph_ref()) {
                // Non-leaf node - recursively build subgraph
                if (child_mapping.mapping_case() != tt::fsd::proto::ChildMapping::kSubInstance) {
                    throw std::runtime_error("Graph child must have sub_instance mapping: " + child_name);
                }

                resolved->subgraphs[child_name] = build_graph_instance(child_mapping.sub_instance(), child_name);
            }
        }

        // Process internal connections within this graph instance
        for (const auto& [port_type_str, port_connections] : template_def.internal_connections()) {
            auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
            if (!port_type.has_value()) {
                throw std::runtime_error("Invalid port type: " + port_type_str);
            }

            for (const auto& conn : port_connections.connections()) {
                std::vector<std::string> path_a(conn.port_a().path().begin(), conn.port_a().path().end());
                uint32_t board_a_id = conn.port_a().tray_id();
                uint32_t port_a_id = conn.port_a().port_id();

                std::vector<std::string> path_b(conn.port_b().path().begin(), conn.port_b().path().end());
                uint32_t board_b_id = conn.port_b().tray_id();
                uint32_t port_b_id = conn.port_b().port_id();

                // Validate and mark ports as used for direct pod connections
                if (path_a.size() == 1 && path_b.size() == 1 && resolved->pods.count(path_a[0]) &&
                    resolved->pods.count(path_b[0])) {
                    auto& board_a = resolved->pods.at(path_a[0]).boards.at(board_a_id);
                    auto& board_b = resolved->pods.at(path_b[0]).boards.at(board_b_id);

                    const auto& available_a = board_a.get_available_port_ids(*port_type);
                    const auto& available_b = board_b.get_available_port_ids(*port_type);

                    if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                        throw std::runtime_error(
                            "Port " + std::to_string(port_a_id) + " not available on board " +
                            std::to_string(board_a_id) + " in pod " + path_a[0]);
                    }
                    if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                        throw std::runtime_error(
                            "Port " + std::to_string(port_b_id) + " not available on board " +
                            std::to_string(board_b_id) + " in pod " + path_b[0]);
                    }

                    board_a.mark_port_used(*port_type, port_a_id);
                    board_b.mark_port_used(*port_type, port_b_id);
                }

                // Store connection
                resolved->internal_connections[*port_type].emplace_back(
                    std::make_tuple(path_a, board_a_id, port_a_id), std::make_tuple(path_b, board_b_id, port_b_id));
            }
        }

        return resolved;
    }

    // Validate that each host_id is assigned to exactly one pod
    void validate_host_id_uniqueness() {
        std::unordered_map<uint32_t, std::string> host_to_pod_path;
        collect_host_assignments(cluster.root_instance, "", host_to_pod_path);
    }

    // Recursively collect all host_id assignments with their pod paths
    void collect_host_assignments(
        std::shared_ptr<ResolvedGraphInstance> graph,
        const std::string& path_prefix,
        std::unordered_map<uint32_t, std::string>& host_to_pod_path) {
        // Check direct pods in this graph
        for (const auto& [pod_name, pod] : graph->pods) {
            uint32_t host_id = pod.host_id;
            std::string full_pod_path = path_prefix.empty() ? pod_name : path_prefix + "/" + pod_name;

            if (host_to_pod_path.count(host_id)) {
                throw std::runtime_error(
                    "Host ID " + std::to_string(host_id) + " is assigned to multiple pods: '" +
                    host_to_pod_path[host_id] + "' and '" + full_pod_path + "'");
            }

            host_to_pod_path[host_id] = full_pod_path;
        }

        // Recursively check subgraphs
        for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
            std::string subgraph_path = path_prefix.empty() ? subgraph_name : path_prefix + "/" + subgraph_name;
            collect_host_assignments(subgraph, subgraph_path, host_to_pod_path);
        }
    }

    // Build cluster from descriptor with port connections and validation
    void build_cluster_from_descriptor(const tt::fsd::proto::ClusterDescriptor& cluster_descriptor) {
        // Load graph templates
        for (const auto& template_def : cluster_descriptor.graph_templates()) {
            cluster.graph_templates[template_def.name()] = template_def;
        }

        // Build the root instance
        cluster.root_instance = build_graph_instance(cluster_descriptor.root_instance());

        // Validate host_id uniqueness across all pods
        validate_host_id_uniqueness();
    }

    // Populate boards_by_host_tray map with pointers to boards in the cluster
    void populate_boards_by_host_tray() {
        if (cluster.root_instance) {
            populate_boards_from_resolved_graph(cluster.root_instance);
        }
    }

    void populate_boards_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph) {
        // Add boards from direct pods in this graph
        for (const auto& [pod_name, pod] : graph->pods) {
            // Get host ID from the pod structure
            uint32_t host_id = pod.host_id;
            for (const auto& [tray_id, board] : pod.boards) {
                boards_by_host_tray[{host_id, tray_id}] = const_cast<Board*>(&board);
            }
        }

        // Recursively add boards from subgraphs
        for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
            populate_boards_from_resolved_graph(subgraph);
        }
    }

    // Simple path resolution for connection processing
    std::pair<Pod&, uint32_t> resolve_pod_from_path(
        const std::vector<std::string>& path, std::shared_ptr<ResolvedGraphInstance> graph = nullptr) {
        if (!graph) {
            graph = cluster.root_instance;
        }

        if (path.size() == 1) {
            // Direct pod reference
            if (graph->pods.count(path[0])) {
                auto& pod = graph->pods.at(path[0]);
                return {pod, pod.host_id};
            }
            throw std::runtime_error("Pod not found: " + path[0]);
        } else {
            // Multi-level path - descend into subgraph
            const std::string& next_level = path[0];
            if (!graph->subgraphs.count(next_level)) {
                throw std::runtime_error("Subgraph not found: " + next_level);
            }

            std::vector<std::string> remaining_path(path.begin() + 1, path.end());
            return resolve_pod_from_path(remaining_path, graph->subgraphs.at(next_level));
        }
    }

    // Utility function to generate logical chip connections from cluster hierarchy
    void generate_logical_chip_connections() {
        chip_connections.clear();

        if (cluster.root_instance) {
            generate_connections_from_resolved_graph(cluster.root_instance);
        }
    }

    void generate_connections_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph) {
        // Process pods in this graph
        for (const auto& [pod_name, pod] : graph->pods) {
            uint32_t host_id = pod.host_id;

            // Add internal board connections
            for (const auto& [tray_id, board] : pod.boards) {
                for (const auto& [port_type, connections] : board.get_internal_connections()) {
                    for (const auto& connection : connections) {
                        const auto& start_channels = board.get_port_channels(port_type, connection.first);
                        const auto& end_channels = board.get_port_channels(port_type, connection.second);
                        auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);
                        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
                            chip_connections.emplace_back(
                                LogicalChipConnection{
                                    .host_id = host_id, .tray_id = tray_id, .asic_channel = start_channel},
                                LogicalChipConnection{
                                    .host_id = host_id, .tray_id = tray_id, .asic_channel = end_channel});
                        }
                    }
                }
            }

            // Add inter-board connections within pod
            for (const auto& [port_type, connections] : pod.inter_board_connections) {
                for (const auto& [board_a, board_b] : connections) {
                    uint32_t board_a_id = board_a.first;
                    uint32_t port_a_id = board_a.second;
                    uint32_t board_b_id = board_b.first;
                    uint32_t port_b_id = board_b.second;

                    const auto& board_a_ref = pod.boards.at(board_a_id);
                    const auto& board_b_ref = pod.boards.at(board_b_id);
                    const auto& start_channels = board_a_ref.get_port_channels(port_type, port_a_id);
                    const auto& end_channels = board_b_ref.get_port_channels(port_type, port_b_id);
                    auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);
                    for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
                        chip_connections.emplace_back(
                            LogicalChipConnection{
                                .host_id = host_id, .tray_id = board_a_id, .asic_channel = start_channel},
                            LogicalChipConnection{
                                .host_id = host_id, .tray_id = board_b_id, .asic_channel = end_channel});
                    }
                }
            }
        }

        // Process internal connections within this graph
        for (const auto& [port_type, connections] : graph->internal_connections) {
            for (const auto& [conn_a, conn_b] : connections) {
                const auto& path_a = std::get<0>(conn_a);
                uint32_t board_a_id = std::get<1>(conn_a);
                uint32_t port_a_id = std::get<2>(conn_a);

                const auto& path_b = std::get<0>(conn_b);
                uint32_t board_b_id = std::get<1>(conn_b);
                uint32_t port_b_id = std::get<2>(conn_b);

                // Resolve pods using path-based addressing
                try {
                    auto [pod_a, host_a_id] = resolve_pod_from_path(path_a, graph);
                    auto [pod_b, host_b_id] = resolve_pod_from_path(path_b, graph);

                    const auto& board_a_ref = pod_a.boards.at(board_a_id);
                    const auto& board_b_ref = pod_b.boards.at(board_b_id);
                    const auto& start_channels = board_a_ref.get_port_channels(port_type, port_a_id);
                    const auto& end_channels = board_b_ref.get_port_channels(port_type, port_b_id);
                    auto asic_channel_pairs = get_asic_channel_connections(port_type, start_channels, end_channels);
                    for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
                        chip_connections.emplace_back(
                            LogicalChipConnection{
                                .host_id = host_a_id, .tray_id = board_a_id, .asic_channel = start_channel},
                            LogicalChipConnection{
                                .host_id = host_b_id, .tray_id = board_b_id, .asic_channel = end_channel});
                    }
                } catch (const std::exception& e) {
                    // Connection may span multiple graph levels - skip for now
                    // In full implementation, would need more sophisticated path resolution
                }
            }
        }

        // Recursively process subgraphs
        for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
            generate_connections_from_resolved_graph(subgraph);
        }
    }

private:
    // Additional state
    std::vector<tt::deployment::Host> deployment_hosts;

    // Core data members
    Cluster cluster;
    std::unordered_map<std::pair<uint32_t, uint32_t>, Board*, std::hash<std::pair<uint32_t, uint32_t>>>
        boards_by_host_tray;
    std::vector<LogicalChipConnectionPair> chip_connections;
};

// Common utility function for validating FSD against discovered GSD
void validate_fsd_against_gsd(
    const std::string& fsd_filename, const std::string& gsd_filename, bool strict_validation = true) {
    // Read the generated FSD using protobuf
    tt::fsd::proto::FactorySystemDescriptor generated_fsd;
    std::ifstream fsd_file(fsd_filename);
    if (!fsd_file.is_open()) {
        throw std::runtime_error("Failed to open FSD file: " + fsd_filename);
    }

    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();

    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &generated_fsd)) {
        throw std::runtime_error("Failed to parse FSD protobuf from file: " + fsd_filename);
    }

    const auto& hosts = generated_fsd.hosts();

    // Read the discovered GSD (Global System Descriptor) - still using YAML
    YAML::Node discovered_gsd = YAML::LoadFile(gsd_filename);

    // Compare the FSD with the discovered GSD
    // First, compare hostnames from the hosts field
    if (generated_fsd.hosts().empty()) {
        throw std::runtime_error("FSD missing hosts");
    }

    // Handle the new GSD structure with compute_node_specs
    if (!discovered_gsd["compute_node_specs"]) {
        throw std::runtime_error("GSD missing compute_node_specs");
    }
    YAML::Node asic_info_node = discovered_gsd["compute_node_specs"];

    // Check that all discovered hostnames are present in the generated FSD hosts
    std::set<std::string> generated_hostnames;
    for (const auto& host : generated_fsd.hosts()) {
        generated_hostnames.insert(host.hostname());
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
    if (!generated_fsd.has_board_types()) {
        throw std::runtime_error("FSD missing board_types");
    }
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
        fsd_board_types[std::make_pair(hosts[host_id].hostname(), tray_id)] = board_type_name;
    }

    // Compare board types between GSD and FSD
    for (const auto& host_entry : asic_info_node) {
        std::string hostname = host_entry.first.as<std::string>();
        YAML::Node host_node = host_entry.second;
        if (!host_node["asic_info"]) {
            throw std::runtime_error("Host " + hostname + " missing asic_info");
        }

        for (const auto& asic_info : host_node["asic_info"]) {
            uint32_t tray_id = asic_info["tray_id"].as<uint32_t>();
            std::string gsd_board_type = asic_info["board_type"].as<std::string>();

            auto fsd_key = std::make_pair(hostname, tray_id);
            if (strict_validation) {
                auto fsd_board_type = fsd_board_types.extract(fsd_key);

                if (fsd_board_type.empty()) {
                    throw std::runtime_error(
                        "Board type not found in FSD for host " + hostname + ", tray " + std::to_string(tray_id));
                }

                if (fsd_board_type.mapped() != gsd_board_type) {
                    throw std::runtime_error(
                        "Board type mismatch for host " + hostname + ", tray " + std::to_string(tray_id) +
                        ": FSD=" + fsd_board_type.mapped() + ", GSD=" + gsd_board_type);
                }
            } else {
                auto fsd_board_type = fsd_board_types.find(fsd_key);
                if (fsd_board_type != fsd_board_types.end()) {
                    if (fsd_board_type->second != gsd_board_type) {
                        throw std::runtime_error(
                            "Board type mismatch for host " + hostname + ", tray " + std::to_string(tray_id) +
                            ": FSD=" + fsd_board_type->second + ", GSD=" + gsd_board_type);
                    }
                } else {
                    throw std::runtime_error(
                        "Board type not found in FSD for host " + hostname + ", tray " + std::to_string(tray_id));
                }
            }
        }
    }
    if (strict_validation) {
        EXPECT_EQ(fsd_board_types.size(), 0) << "Expected all board types to be found in FSD";
    }

    // Compare chip connections
    if (!generated_fsd.has_eth_connections()) {
        throw std::runtime_error("FSD missing eth_connections");
    }

    // Determine which connection types exist in the discovered GSD
    bool has_local_eth_connections =
        discovered_gsd["local_eth_connections"] && !discovered_gsd["local_eth_connections"].IsNull();
    bool has_global_eth_connections =
        discovered_gsd["global_eth_connections"] && !discovered_gsd["global_eth_connections"].IsNull();

    // At least one connection type should exist
    if (!has_local_eth_connections && !has_global_eth_connections) {
        throw std::runtime_error("No connection types found in discovered GSD");
    }

    // Convert generated connections to a comparable format
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> generated_connections;
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> duplicate_generated_connections;

    for (const auto& connection : generated_fsd.eth_connections().connection()) {
        const auto& endpoint_a = connection.endpoint_a();
        const auto& endpoint_b = connection.endpoint_b();

        uint32_t host_id_1 = endpoint_a.host_id();
        uint32_t tray_id_1 = endpoint_a.tray_id();
        uint32_t asic_location_1 = endpoint_a.asic_location();
        uint32_t chan_id_1 = endpoint_a.chan_id();

        uint32_t host_id_2 = endpoint_b.host_id();
        uint32_t tray_id_2 = endpoint_b.tray_id();
        uint32_t asic_location_2 = endpoint_b.asic_location();
        uint32_t chan_id_2 = endpoint_b.chan_id();

        const std::string& hostname_1 = hosts[host_id_1].hostname();
        const std::string& hostname_2 = hosts[host_id_2].hostname();

        PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_location_1, chan_id_1};
        PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_location_2, chan_id_2};

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
        throw std::runtime_error(error_msg);
    }

    // Convert discovered GSD connections to the same format
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> discovered_connections;
    std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>> duplicate_discovered_connections;

    // Process local connections if they exist
    if (has_local_eth_connections) {
        for (const auto& connection_pair : discovered_gsd["local_eth_connections"]) {
            if (connection_pair.size() != 2) {
                throw std::runtime_error("Each connection should have exactly 2 endpoints");
            }

            const auto& first_conn = connection_pair[0];
            const auto& second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_1 = first_conn["asic_location"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_2 = second_conn["asic_location"].as<uint32_t>();

            PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_location_1, chan_id_1};
            PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_location_2, chan_id_2};

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
            if (connection_pair.size() != 2) {
                throw std::runtime_error("Each connection should have exactly 2 endpoints");
            }

            auto first_conn = connection_pair[0];
            auto second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_1 = first_conn["asic_location"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_2 = second_conn["asic_location"].as<uint32_t>();

            PhysicalChannelConnection conn_1{hostname_1, tray_id_1, asic_location_1, chan_id_1};
            PhysicalChannelConnection conn_2{hostname_2, tray_id_2, asic_location_2, chan_id_2};

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
        throw std::runtime_error(error_msg);
    }

    if (strict_validation) {
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

        // Lambda to extract port information from channel connections
        auto extract_port_info =
            [&](const std::set<std::pair<PhysicalChannelConnection, PhysicalChannelConnection>>& connections) {
                std::set<std::pair<PhysicalPortConnection, PhysicalPortConnection>> port_info;

                for (const auto& conn : connections) {
                    // Get board types from FSD for both connections independently
                    std::string board_type_a = "";
                    std::string board_type_b = "";

                    // Find host_id for each connection by matching hostname
                    uint32_t host_id_a = 0;
                    uint32_t host_id_b = 0;
                    for (uint32_t i = 0; i < hosts.size(); ++i) {
                        if (hosts[i].hostname() == conn.first.hostname) {
                            host_id_a = i;
                        }
                        if (hosts[i].hostname() == conn.second.hostname) {
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
                            "Board type not found for connection: " + conn.first.hostname + " <-> " +
                            conn.second.hostname);
                    }

                    Board board_a = create_board(board_type_a);
                    Board board_b = create_board(board_type_b);
                    auto port_a = board_a.get_port_for_asic_channel({conn.first.asic_location, conn.first.channel_id});
                    auto port_b =
                        board_b.get_port_for_asic_channel({conn.second.asic_location, conn.second.channel_id});

                    PhysicalPortConnection port_a_conn;
                    PhysicalPortConnection port_b_conn;

                    // Add deployment info for first connection if available
                    for (const auto& host : hosts) {
                        if (host.hostname() == conn.first.hostname) {
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
                        if (host.hostname() == conn.second.hostname) {
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

                    port_info.insert(std::make_pair(port_a_conn, port_b_conn));
                }

                return port_info;
            };

        // Report missing connections (in FSD but not in GSD)
        if (!missing_in_gsd.empty()) {
            // Collect and display port information for all missing connections
            auto missing_port_info = extract_port_info(missing_in_gsd);
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
            // Collect and display port information for all extra connections
            auto extra_port_info = extract_port_info(extra_in_gsd);

            std::ostringstream oss;
            oss << "Channel Connections found in GSD but missing in FSD (" << std::to_string(extra_in_gsd.size())
                << " connections):\n";
            for (const auto& conn : extra_in_gsd) {
                oss << "  - " << conn.first << " <-> " << conn.second << "\n";
            }
            oss << "\n";

            oss << "Port Connections found in GSD but missing in FSD ("
                << std::to_string(extra_port_info.size()) + " connections):\n";
            for (const auto& conn : extra_port_info) {
                oss << "  - " << conn.first << " <-> " << conn.second << "\n";
            }
            std::cout << oss.str() << std::endl;
        }

        // Fail the test if there are any mismatches
        if (!missing_in_gsd.empty() || !extra_in_gsd.empty()) {
            throw std::runtime_error("Connection mismatch detected. Check console output for details.");
        }

        // If we get here, all connections match
        std::cout << "All connections match between FSD and GSD (" << generated_connections.size() << " connections)"
                  << std::endl;
    } else {
        for (const auto& conn : discovered_connections) {
            if (generated_connections.find(conn) == generated_connections.end()) {
                throw std::runtime_error(
                    "Connection not found in FSD: " + conn.first.hostname + " <-> " + conn.second.hostname);
            }
        }
    }
}

TEST(Cluster, TestFactorySystemDescriptor16LB) {
    // Load deployment descriptor for validation only
    auto deployment_descriptor = CablingGenerator::load_descriptor_from_textproto<tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/16_lb_deployment.textproto");

    // Load descriptors for testing
    auto cluster_descriptor = CablingGenerator::load_descriptor_from_textproto<tt::fsd::proto::ClusterDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/instances/"
        "16_n300_lb_cluster.textproto");

    // Create the cabling generator
    CablingGenerator cabling_generator(cluster_descriptor, deployment_descriptor);

    cabling_generator.emit_textproto_factory_system_descriptor("fsd/factory_system_descriptor_16_n300_lb.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_16_n300_lb.textproto",
        "tests/tt_metal/tt_fabric/factory_system_descriptor/global_system_descriptors/16_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    // Load deployment descriptor for validation only
    auto deployment_descriptor = CablingGenerator::load_descriptor_from_textproto<tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/5_lb_deployment.textproto");

    // Load the 5LB configuration
    auto cluster_descriptor = CablingGenerator::load_descriptor_from_textproto<tt::fsd::proto::ClusterDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/instances/"
        "5_n300_lb_superpod.textproto");

    // Create the cabling generator
    CablingGenerator cabling_generator(cluster_descriptor, deployment_descriptor);

    // Generate the FSD (textproto format)
    cabling_generator.emit_textproto_factory_system_descriptor("fsd/factory_system_descriptor_5_n300_lb.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_5_n300_lb.textproto",
        "tests/tt_metal/tt_fabric/factory_system_descriptor/global_system_descriptors/5_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorus) {
    // Load deployment descriptor
    auto deployment_descriptor = tt::tt_fabric::fsd_tests::CablingGenerator::load_descriptor_from_textproto<
        tt::deployment::DeploymentDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/deployment/5_wh_galaxy_y_torus_deployment.textproto");

    // Load the WH Galaxy Y Torus configuration
    auto cluster_descriptor = CablingGenerator::load_descriptor_from_textproto<tt::fsd::proto::ClusterDescriptor>(
        "tests/tt_metal/tt_fabric/factory_system_descriptor/cabling_descriptors/instances/"
        "5_wh_galaxy_y_torus_superpod.textproto");

    // Create the cabling generator
    CablingGenerator cabling_generator(cluster_descriptor, deployment_descriptor);

    // Generate the FSD (textproto format)
    cabling_generator.emit_textproto_factory_system_descriptor(
        "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    EXPECT_THROW(
        {
            try {
                validate_fsd_against_gsd(
                    "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto",
                    "tests/tt_metal/tt_fabric/factory_system_descriptor/global_system_descriptors/"
                    "5_wh_galaxy_y_torus_physical_desc.yaml");
            } catch (const std::runtime_error& e) {
                std::cout << e.what() << std::endl;
                throw;
            }
        },
        std::runtime_error);
}

}  // namespace fsd_tests
}  // namespace tt::tt_fabric
