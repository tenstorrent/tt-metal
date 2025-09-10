// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cabling_generator.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <scaleout_tools/connector/connector.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

// Add protobuf includes
#include "protobuf/cluster_config.pb.h"
#include "protobuf/deployment.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

// Helper to load protobuf descriptors
template <typename Descriptor>
Descriptor load_descriptor_from_textproto(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Descriptor descriptor;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &descriptor)) {
        throw std::runtime_error("Failed to parse textproto file: " + file_path);
    }
    return descriptor;
}

// Find node descriptor by name - search inline first, then fallback to file
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor find_node_descriptor(
    const std::string& node_descriptor_name,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor) {
    // First, search in inline node descriptors
    auto it = cluster_descriptor.node_descriptors().find(node_descriptor_name);
    if (it != cluster_descriptor.node_descriptors().end()) {
        return it->second;
    }

    // Fallback: load from file
    // TODO: This should be converted to factory functions
    return load_descriptor_from_textproto<tt::scaleout_tools::cabling_generator::proto::NodeDescriptor>(
        "scaleout_tools/scaleout_tools/cabling_descriptor/instances/" + node_descriptor_name + ".textproto");
}

// Build node from descriptor with port connections and validation
Node build_node(
    const std::string& node_descriptor_name,
    HostId host_id,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    std::unordered_map<std::string, Node>& node_templates,
    std::unordered_map<std::string, Board>& board_templates) {
    const std::string& node_type = node_descriptor_name;
    auto it = node_templates.find(node_type);
    if (it != node_templates.end()) {
        Node node = it->second;  // Copy template
        node.host_id = host_id;
        return node;
    }

    // Build new node template (with host_id=0)
    Node template_node;

    auto node_descriptor = find_node_descriptor(node_descriptor_name, cluster_descriptor);

    // Create boards with internal connections marked (using cached boards)
    for (const auto& board_item : node_descriptor.boards().board()) {
        TrayId tray_id = TrayId(board_item.tray_id());
        const std::string& board_type = board_item.board_type();

        // Check cache first
        auto board_it = board_templates.find(board_type);
        if (board_it != board_templates.end()) {
            template_node.boards.emplace(tray_id, board_it->second);
        } else {
            // Create new board and cache it
            Board board = create_board(board_type);
            board_templates.emplace(board_type, board);
            template_node.boards.emplace(tray_id, board);
        }
    }

    // Add inter-board connections and validate/mark ports
    for (const auto& [port_type_str, port_connections] : node_descriptor.port_type_connections()) {
        auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
        if (!port_type.has_value()) {
            throw std::runtime_error("Invalid port type: " + port_type_str);
        }

        for (const auto& conn : port_connections.connections()) {
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());
            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            // Validate and mark ports as used
            auto& board_a = template_node.boards.at(board_a_id);
            auto& board_b = template_node.boards.at(board_b_id);

            const auto& available_a = board_a.get_available_port_ids(*port_type);
            const auto& available_b = board_b.get_available_port_ids(*port_type);

            if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                throw std::runtime_error(
                    port_type_str + " Port " + std::to_string(*port_a_id) + " not available on board " +
                    std::to_string(*board_a_id) + " in node " + node_descriptor_name);
            }
            if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                throw std::runtime_error(
                    port_type_str + " Port " + std::to_string(*port_b_id) + " not available on board " +
                    std::to_string(*board_b_id) + " in node " + node_descriptor_name);
            }

            board_a.mark_port_used(*port_type, port_a_id);
            board_b.mark_port_used(*port_type, port_b_id);

            // Store connection
            template_node.inter_board_connections[*port_type].emplace_back(
                std::make_pair(board_a_id, port_a_id), std::make_pair(board_b_id, port_b_id));
        }
    }

    // Cache the template
    node_templates[node_type] = template_node;

    // Create instance with actual host_id
    Node node = template_node;
    node.host_id = host_id;
    return node;
}

// Build resolved graph instance from template and concrete host mappings
std::unique_ptr<ResolvedGraphInstance> build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates,
    std::unordered_map<std::string, Board>& board_templates) {
    auto resolved = std::make_unique<ResolvedGraphInstance>();
    resolved->template_name = graph_instance.template_name();
    resolved->instance_name = instance_name;

    // Get the template definition
    const auto& template_def = cluster_descriptor.graph_templates().at(graph_instance.template_name());

    // Build children based on template + instance mapping
    for (const auto& child_def : template_def.children()) {
        const std::string& child_name = child_def.name();
        const auto& child_mapping = graph_instance.child_mappings().at(child_name);

        if (child_def.has_node_ref()) {
            // Leaf node - create node
            if (child_mapping.mapping_case() != tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
                throw std::runtime_error("Node child must have host_id mapping: " + child_name);
            }

            HostId host_id = HostId(child_mapping.host_id());
            const std::string& node_descriptor_name = child_def.node_ref().node_descriptor();

            // Validate deployment node type if specified
            if (*host_id < deployment_descriptor.hosts().size()) {
                const auto& deployment_host = deployment_descriptor.hosts()[*host_id];
                if (!deployment_host.node_type().empty() && deployment_host.node_type() != node_descriptor_name) {
                    throw std::runtime_error(
                        "Node type mismatch for host " + deployment_host.host() + " (host_id " +
                        std::to_string(*host_id) + "): " + "deployment specifies '" + deployment_host.node_type() +
                        "' " + "but cluster configuration expects '" + node_descriptor_name + "'");
                }
            } else {
                throw std::runtime_error("Host ID " + std::to_string(*host_id) + " not found in deployment");
            }

            // Find node descriptor and build node inside build_node
            resolved->nodes[child_name] =
                build_node(node_descriptor_name, host_id, cluster_descriptor, node_templates, board_templates);

        } else if (child_def.has_graph_ref()) {
            // Non-leaf node - recursively build subgraph
            if (child_mapping.mapping_case() !=
                tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
                throw std::runtime_error("Graph child must have sub_instance mapping: " + child_name);
            }

            resolved->subgraphs[child_name] = build_graph_instance(
                child_mapping.sub_instance(),
                cluster_descriptor,
                deployment_descriptor,
                child_name,
                node_templates,
                board_templates);
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
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());

            std::vector<std::string> path_b(conn.port_b().path().begin(), conn.port_b().path().end());
            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            // Validate and mark ports as used for direct node connections
            if (path_a.size() == 1 && path_b.size() == 1 && resolved->nodes.count(path_a[0]) &&
                resolved->nodes.count(path_b[0])) {
                auto& board_a = resolved->nodes.at(path_a[0]).boards.at(board_a_id);
                auto& board_b = resolved->nodes.at(path_b[0]).boards.at(board_b_id);

                const auto& available_a = board_a.get_available_port_ids(*port_type);
                const auto& available_b = board_b.get_available_port_ids(*port_type);

                if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                    throw std::runtime_error(
                        port_type_str + " Port " + std::to_string(*port_a_id) + " not available on board " +
                        std::to_string(*board_a_id) + " in node " + path_a[0]);
                }
                if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                    throw std::runtime_error(
                        port_type_str + " Port " + std::to_string(*port_b_id) + " not available on board " +
                        std::to_string(*board_b_id) + " in node " + path_b[0]);
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

// Simple path resolution for connection processing
std::pair<Node&, HostId> resolve_node_from_path(
    ttsl::Span<const std::string> path, const std::unique_ptr<ResolvedGraphInstance>& graph) {
    if (!graph) {
        throw std::runtime_error("Graph not set");
    }

    if (path.size() == 1) {
        // Direct node reference
        if (graph->nodes.count(path[0])) {
            auto& node = graph->nodes.at(path[0]);
            return {node, node.host_id};
        }
        throw std::runtime_error("Node not found: " + path[0]);
    } else {
        // Multi-level path - descend into subgraph
        const std::string& next_level = path[0];
        if (!graph->subgraphs.count(next_level)) {
            throw std::runtime_error("Subgraph not found: " + next_level);
        }

        return resolve_node_from_path(path.subspan(1), graph->subgraphs.at(next_level));
    }
}

// Constructor
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path) {
    // Load descriptors from file paths
    auto cluster_descriptor =
        load_descriptor_from_textproto<tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor>(
            cluster_descriptor_path);
    auto deployment_descriptor =
        load_descriptor_from_textproto<tt::scaleout_tools::deployment::proto::DeploymentDescriptor>(
            deployment_descriptor_path);

    // Store deployment hosts
    deployment_hosts_.reserve(deployment_descriptor.hosts().size());
    for (const auto& proto_host : deployment_descriptor.hosts()) {
        deployment_hosts_.emplace_back(Host{
            .hostname = proto_host.host(),
            .hall = proto_host.hall(),
            .aisle = proto_host.aisle(),
            .rack = proto_host.rack(),
            .shelf_u = proto_host.shelf_u()});
    }

    // Build cluster with all connections and port validation
    root_instance_ = build_graph_instance(
        cluster_descriptor.root_instance(),
        cluster_descriptor,
        deployment_descriptor,
        "",
        node_templates_,
        board_templates_);

    // Validate host_id uniqueness across all nodes
    validate_host_id_uniqueness();

    // Populate the boards_by_host_tray_ map
    populate_boards_by_host_tray();

    // Generate all logical chip connections
    generate_logical_chip_connections();
}

// Getters for all data
const std::vector<Host>& CablingGenerator::get_deployment_hosts() const { return deployment_hosts_; }

const std::unordered_map<std::pair<HostId, TrayId>, const Board*, HostTrayHasher>&
CablingGenerator::get_boards_by_host_tray() const {
    return boards_by_host_tray_;
}

const std::vector<LogicalChannelConnection>& CablingGenerator::get_chip_connections() const {
    return chip_connections_;
}

// Method to emit textproto factory system descriptor
void CablingGenerator::emit_factory_system_descriptor(const std::string& output_path) const {
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;

    // Add host information from deployment hosts (indexed by host_id)
    for (size_t i = 0; i < deployment_hosts_.size(); ++i) {
        const auto& deployment_host = deployment_hosts_[i];
        auto* host = fsd.add_hosts();
        host->set_hostname(deployment_host.hostname);
        host->set_hall(deployment_host.hall);
        host->set_aisle(deployment_host.aisle);
        host->set_rack(deployment_host.rack);
        host->set_shelf_u(deployment_host.shelf_u);
    }

    // Add board types
    for (const auto& [host_tray_pair, board] : boards_by_host_tray_) {
        auto* board_location = fsd.mutable_board_types()->add_board_locations();
        board_location->set_host_id(*host_tray_pair.first);  // Extract HostId value
        board_location->set_tray_id(*host_tray_pair.second);
        board_location->set_board_type(enchantum::to_string(board->get_board_type()).data());
    }

    // Add ASIC connections from chip_connections_
    for (const auto& [start, end] : chip_connections_) {
        auto* connection = fsd.mutable_eth_connections()->add_connection();

        auto* endpoint_a = connection->mutable_endpoint_a();
        endpoint_a->set_host_id(*start.host_id);
        endpoint_a->set_tray_id(*start.tray_id);
        endpoint_a->set_asic_location(start.asic_channel.asic_location);
        endpoint_a->set_chan_id(*start.asic_channel.channel_id);

        auto* endpoint_b = connection->mutable_endpoint_b();
        endpoint_b->set_host_id(*end.host_id);
        endpoint_b->set_tray_id(*end.tray_id);
        endpoint_b->set_asic_location(end.asic_channel.asic_location);
        endpoint_b->set_chan_id(*end.asic_channel.channel_id);
    }

    // Create parent directory if it doesn't exist
    std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    std::string output_string;
    google::protobuf::TextFormat::Printer printer;
    printer.SetUseShortRepeatedPrimitives(true);
    printer.SetUseUtf8StringEscaping(true);
    printer.SetSingleLineMode(false);

    if (!printer.PrintToString(fsd, &output_string)) {
        throw std::runtime_error("Failed to write textproto to file: " + output_path);
    }

    output_file << output_string;
    output_file.close();
}

// Validate that each host_id is assigned to exactly one node
void CablingGenerator::validate_host_id_uniqueness() {
    std::unordered_map<HostId, std::string> host_to_node_path;
    collect_host_assignments(root_instance_, "", host_to_node_path);
}

// Recursively collect all host_id assignments with their node paths
void CablingGenerator::collect_host_assignments(
    const std::unique_ptr<ResolvedGraphInstance>& graph,
    const std::string& path_prefix,
    std::unordered_map<HostId, std::string>& host_to_node_path) {
    // Check direct nodes in this graph
    for (const auto& [node_name, node] : graph->nodes) {
        HostId host_id = node.host_id;
        std::string full_node_path = path_prefix.empty() ? node_name : path_prefix + "/" + node_name;

        if (host_to_node_path.count(host_id)) {
            throw std::runtime_error(
                "Host ID " + std::to_string(*host_id) + " is assigned to multiple nodes: '" +
                host_to_node_path[host_id] + "' and '" + full_node_path + "'");
        }
        host_to_node_path[host_id] = full_node_path;
    }

    // Recursively check subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        std::string sub_path = path_prefix.empty() ? subgraph_name : path_prefix + "/" + subgraph_name;
        collect_host_assignments(subgraph, sub_path, host_to_node_path);
    }
}

// Utility function to generate logical chip connections from cluster hierarchy
void CablingGenerator::generate_logical_chip_connections() {
    chip_connections_.clear();

    if (root_instance_) {
        generate_connections_from_resolved_graph(root_instance_);
    }
}

void CablingGenerator::generate_connections_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph) {
    // Lambda to create connections between two ports
    auto create_port_connection = [&](PortType port_type,
                                      const Board& start_board,
                                      PortId start_port_id,
                                      const Board& end_board,
                                      PortId end_port_id,
                                      HostId start_host_id,
                                      TrayId start_tray_id,
                                      HostId end_host_id,
                                      TrayId end_tray_id) {
        const auto& start_channels = start_board.get_port_channels(port_type, start_port_id);
        const auto& end_channels = end_board.get_port_channels(port_type, end_port_id);
        auto asic_channel_pairs =
            tt::scaleout_tools::get_asic_channel_connections(port_type, start_channels, end_channels);
        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
            chip_connections_.emplace_back(
                LogicalChannelEndpoint{
                    .host_id = start_host_id, .tray_id = start_tray_id, .asic_channel = start_channel},
                LogicalChannelEndpoint{.host_id = end_host_id, .tray_id = end_tray_id, .asic_channel = end_channel});
        }
    };

    // Process nodes in this graph
    for (const auto& [node_name, node] : graph->nodes) {
        HostId host_id = node.host_id;

        // Add internal board connections
        for (const auto& [tray_id, board] : node.boards) {
            for (const auto& [port_type, connections] : board.get_internal_connections()) {
                for (const auto& connection : connections) {
                    create_port_connection(
                        port_type,
                        board,
                        PortId(connection.first),
                        board,
                        PortId(connection.second),
                        host_id,
                        tray_id,
                        host_id,
                        tray_id);
                }
            }
        }

        // Add inter-board connections within node
        for (const auto& [port_type, connections] : node.inter_board_connections) {
            for (const auto& [board_a, board_b] : connections) {
                TrayId board_a_id = board_a.first;
                PortId port_a_id = board_a.second;
                TrayId board_b_id = board_b.first;
                PortId port_b_id = board_b.second;

                const auto& board_a_ref = node.boards.at(board_a_id);
                const auto& board_b_ref = node.boards.at(board_b_id);
                create_port_connection(
                    port_type,
                    board_a_ref,
                    port_a_id,
                    board_b_ref,
                    port_b_id,
                    host_id,
                    board_a_id,
                    host_id,
                    board_b_id);
            }
        }
    }

    // Process internal connections within this graph
    for (const auto& [port_type, connections] : graph->internal_connections) {
        for (const auto& [conn_a, conn_b] : connections) {
            const auto& path_a = std::get<0>(conn_a);
            TrayId board_a_id = std::get<1>(conn_a);
            PortId port_a_id = std::get<2>(conn_a);

            const auto& path_b = std::get<0>(conn_b);
            TrayId board_b_id = std::get<1>(conn_b);
            PortId port_b_id = std::get<2>(conn_b);

            // Resolve nodes using path-based addressing
            auto [node_a, host_a_id] = resolve_node_from_path(path_a, graph);
            auto [node_b, host_b_id] = resolve_node_from_path(path_b, graph);

            const auto& board_a_ref = node_a.boards.at(board_a_id);
            const auto& board_b_ref = node_b.boards.at(board_b_id);
            create_port_connection(
                port_type,
                board_a_ref,
                port_a_id,
                board_b_ref,
                port_b_id,
                host_a_id,
                board_a_id,
                host_b_id,
                board_b_id);
        }
    }

    // Recursively process subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        generate_connections_from_resolved_graph(subgraph);
    }
}

void CablingGenerator::populate_boards_by_host_tray() {
    boards_by_host_tray_.clear();

    if (root_instance_) {
        populate_boards_from_resolved_graph(root_instance_);
    }
}

void CablingGenerator::populate_boards_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph) {
    // Add boards from direct nodes
    for (auto& [node_name, node] : graph->nodes) {
        for (auto& [tray_id, board] : node.boards) {
            std::pair<HostId, TrayId> key = std::make_pair(node.host_id, tray_id);
            boards_by_host_tray_.emplace(key, &board);
        }
    }

    // Recursively add boards from subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        populate_boards_from_resolved_graph(subgraph);
    }
}

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelEndpoint& conn) {
    os << "PhysicalChannelEndpoint{hostname='" << conn.hostname << "', tray_id=" << *conn.tray_id
       << ", asic_location=" << conn.asic_location << ", channel_id=" << *conn.channel_id << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const PhysicalPortEndpoint& conn) {
    os << "PhysicalPortEndpoint{hostname='" << conn.hostname << "', aisle='" << conn.aisle << "', rack=" << conn.rack
       << ", shelf_u=" << conn.shelf_u << ", port_type=" << enchantum::to_string(conn.port_type)
       << ", port_id=" << *conn.port_id << "}";
    return os;
}

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::LogicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            *conn.host_id, conn.tray_id, conn.asic_channel.asic_location, conn.asic_channel.channel_id);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, *conn.tray_id, conn.asic_location, conn.channel_id);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalPortEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalPortEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.aisle, conn.rack, conn.shelf_u, conn.port_type, *conn.port_id);
    }
};

}  // namespace std
