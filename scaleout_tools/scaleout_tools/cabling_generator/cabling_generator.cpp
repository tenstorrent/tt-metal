// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cabling_generator.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <fstream>

#include <scaleout_tools/connector/connector.hpp>
#include "factory_system_descriptor.pb.h"
#include "deployment.pb.h"

namespace tt::scaleout_tools {

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
    build_cluster_from_descriptor(cluster_descriptor, deployment_descriptor);

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

const std::vector<LogicalChipConnectionPair>& CablingGenerator::get_chip_connections() const {
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

// Find pod descriptor by name - search inline first, then fallback to file
tt::scaleout_tools::cabling_generator::proto::PodDescriptor CablingGenerator::find_pod_descriptor(
    const std::string& pod_descriptor_name,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor) {
    // First, search in inline pod descriptors
    auto it = cluster_descriptor.pod_descriptors().find(pod_descriptor_name);
    if (it != cluster_descriptor.pod_descriptors().end()) {
        return it->second;
    }

    // Fallback: load from file
    // TODO: This should be converted to factory functions
    return load_descriptor_from_textproto<tt::scaleout_tools::cabling_generator::proto::PodDescriptor>(
        "scaleout_tools/scaleout_tools/cabling_descriptor/instances/" + pod_descriptor_name + ".textproto");
}

// Build pod from descriptor with port connections and validation
Pod CablingGenerator::build_pod(
    const std::string& pod_descriptor_name,
    HostId host_id,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor) {
    const std::string& pod_type = pod_descriptor_name;
    auto it = pod_templates_.find(pod_type);
    if (it != pod_templates_.end()) {
        Pod pod = it->second;  // Copy template
        pod.host_id = host_id;
        return pod;
    }

    // Build new pod template (with host_id=0)
    Pod template_pod;

    auto pod_descriptor = find_pod_descriptor(pod_descriptor_name, cluster_descriptor);

    // Create boards with internal connections marked (using cached boards)
    for (const auto& board_item : pod_descriptor.boards().board()) {
        TrayId tray_id = TrayId(board_item.tray_id());
        const std::string& board_type = board_item.board_type();

        // Check cache first
        auto board_it = board_templates_.find(board_type);
        if (board_it != board_templates_.end()) {
            template_pod.boards.emplace(tray_id, board_it->second);
        } else {
            // Create new board and cache it
            Board board = create_board(board_type);
            board_templates_.emplace(board_type, board);
            template_pod.boards.emplace(tray_id, board);
        }
    }

    // Add inter-board connections and validate/mark ports
    for (const auto& [port_type_str, port_connections] : pod_descriptor.port_type_connections()) {
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
            auto& board_a = template_pod.boards.at(board_a_id);
            auto& board_b = template_pod.boards.at(board_b_id);

            const auto& available_a = board_a.get_available_port_ids(*port_type);
            const auto& available_b = board_b.get_available_port_ids(*port_type);

            if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                throw std::runtime_error(
                    "Port " + std::to_string(*port_a_id) + " not available on board " + std::to_string(*board_a_id) +
                    " in pod " + pod_descriptor_name);
            }
            if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                throw std::runtime_error(
                    "Port " + std::to_string(*port_b_id) + " not available on board " + std::to_string(*board_b_id) +
                    " in pod " + pod_descriptor_name);
            }

            board_a.mark_port_used(*port_type, port_a_id);
            board_b.mark_port_used(*port_type, port_b_id);

            // Store connection
            template_pod.inter_board_connections[*port_type].emplace_back(
                std::make_pair(board_a_id, port_a_id), std::make_pair(board_b_id, port_b_id));
        }
    }

    // Cache the template
    pod_templates_[pod_type] = template_pod;

    // Create instance with actual host_id
    Pod pod = template_pod;
    pod.host_id = host_id;
    return pod;
}

// Build resolved graph instance from template and concrete host mappings
std::shared_ptr<ResolvedGraphInstance> CablingGenerator::build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::string& instance_name) {
    auto resolved = std::make_shared<ResolvedGraphInstance>();
    resolved->template_name = graph_instance.template_name();
    resolved->instance_name = instance_name;

    // Get the template definition
    const auto& template_def = cluster_descriptor.graph_templates().at(graph_instance.template_name());

    // Build children based on template + instance mapping
    for (const auto& child_def : template_def.children()) {
        const std::string& child_name = child_def.name();
        const auto& child_mapping = graph_instance.child_mappings().at(child_name);

        if (child_def.has_pod_ref()) {
            // Leaf node - create pod
            if (child_mapping.mapping_case() != tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
                throw std::runtime_error("Pod child must have host_id mapping: " + child_name);
            }

            HostId host_id = HostId(child_mapping.host_id());
            const std::string& pod_descriptor_name = child_def.pod_ref().pod_descriptor();

            // Validate deployment pod type if specified
            if (*host_id < deployment_descriptor.hosts().size()) {
                const auto& deployment_host = deployment_descriptor.hosts()[*host_id];
                if (!deployment_host.pod_type().empty() && deployment_host.pod_type() != pod_descriptor_name) {
                    throw std::runtime_error(
                        "Pod type mismatch for host " + deployment_host.host() + " (host_id " +
                        std::to_string(*host_id) + "): " + "deployment specifies '" + deployment_host.pod_type() +
                        "' " + "but cluster configuration expects '" + pod_descriptor_name + "'");
                }
            } else {
                throw std::runtime_error("Host ID " + std::to_string(*host_id) + " not found in deployment");
            }

            // Find pod descriptor and build pod inside build_pod
            resolved->pods[child_name] = build_pod(pod_descriptor_name, host_id, cluster_descriptor);

        } else if (child_def.has_graph_ref()) {
            // Non-leaf node - recursively build subgraph
            if (child_mapping.mapping_case() !=
                tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
                throw std::runtime_error("Graph child must have sub_instance mapping: " + child_name);
            }

            resolved->subgraphs[child_name] = build_graph_instance(
                child_mapping.sub_instance(), cluster_descriptor, deployment_descriptor, child_name);
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

            // Validate and mark ports as used for direct pod connections
            if (path_a.size() == 1 && path_b.size() == 1 && resolved->pods.count(path_a[0]) &&
                resolved->pods.count(path_b[0])) {
                auto& board_a = resolved->pods.at(path_a[0]).boards.at(board_a_id);
                auto& board_b = resolved->pods.at(path_b[0]).boards.at(board_b_id);

                const auto& available_a = board_a.get_available_port_ids(*port_type);
                const auto& available_b = board_b.get_available_port_ids(*port_type);

                if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
                    throw std::runtime_error(
                        "Port " + std::to_string(*port_a_id) + " not available on board " +
                        std::to_string(*board_a_id) + " in pod " + path_a[0]);
                }
                if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
                    throw std::runtime_error(
                        "Port " + std::to_string(*port_b_id) + " not available on board " +
                        std::to_string(*board_b_id) + " in pod " + path_b[0]);
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

// Build cluster from descriptor with port connections and validation
void CablingGenerator::build_cluster_from_descriptor(
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor) {
    // Build the root instance
    root_instance_ =
        build_graph_instance(cluster_descriptor.root_instance(), cluster_descriptor, deployment_descriptor);

    // Validate host_id uniqueness across all pods
    validate_host_id_uniqueness();
}

// Validate that each host_id is assigned to exactly one pod
void CablingGenerator::validate_host_id_uniqueness() {
    std::unordered_map<HostId, std::string> host_to_pod_path;
    collect_host_assignments(root_instance_, "", host_to_pod_path);
}

// Recursively collect all host_id assignments with their pod paths
void CablingGenerator::collect_host_assignments(
    std::shared_ptr<ResolvedGraphInstance> graph,
    const std::string& path_prefix,
    std::unordered_map<HostId, std::string>& host_to_pod_path) {
    // Check direct pods in this graph
    for (const auto& [pod_name, pod] : graph->pods) {
        HostId host_id = pod.host_id;
        std::string full_pod_path = path_prefix.empty() ? pod_name : path_prefix + "/" + pod_name;

        if (host_to_pod_path.count(host_id)) {
            throw std::runtime_error(
                "Host ID " + std::to_string(*host_id) + " is assigned to multiple pods: '" + host_to_pod_path[host_id] +
                "' and '" + full_pod_path + "'");
        }
        host_to_pod_path[host_id] = full_pod_path;
    }

    // Recursively check subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        std::string sub_path = path_prefix.empty() ? subgraph_name : path_prefix + "/" + subgraph_name;
        collect_host_assignments(subgraph, sub_path, host_to_pod_path);
    }
}

// Simple path resolution for connection processing
std::pair<Pod&, HostId> CablingGenerator::resolve_pod_from_path(
    ttsl::Span<const std::string> path, std::shared_ptr<ResolvedGraphInstance> graph) {
    if (!graph) {
        graph = root_instance_;
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

        return resolve_pod_from_path(path.subspan(1), graph->subgraphs.at(next_level));
    }
}

// Utility function to generate logical chip connections from cluster hierarchy
void CablingGenerator::generate_logical_chip_connections() {
    chip_connections_.clear();

    if (root_instance_) {
        generate_connections_from_resolved_graph(root_instance_);
    }
}

void CablingGenerator::generate_connections_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph) {
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
                LogicalChipConnection{
                    .host_id = start_host_id, .tray_id = start_tray_id, .asic_channel = start_channel},
                LogicalChipConnection{.host_id = end_host_id, .tray_id = end_tray_id, .asic_channel = end_channel});
        }
    };

    // Process pods in this graph
    for (const auto& [pod_name, pod] : graph->pods) {
        HostId host_id = pod.host_id;

        // Add internal board connections
        for (const auto& [tray_id, board] : pod.boards) {
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

        // Add inter-board connections within pod
        for (const auto& [port_type, connections] : pod.inter_board_connections) {
            for (const auto& [board_a, board_b] : connections) {
                TrayId board_a_id = board_a.first;
                PortId port_a_id = board_a.second;
                TrayId board_b_id = board_b.first;
                PortId port_b_id = board_b.second;

                const auto& board_a_ref = pod.boards.at(board_a_id);
                const auto& board_b_ref = pod.boards.at(board_b_id);
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

            // Resolve pods using path-based addressing
            auto [pod_a, host_a_id] = resolve_pod_from_path(path_a, graph);
            auto [pod_b, host_b_id] = resolve_pod_from_path(path_b, graph);

            const auto& board_a_ref = pod_a.boards.at(board_a_id);
            const auto& board_b_ref = pod_b.boards.at(board_b_id);
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

void CablingGenerator::populate_boards_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph) {
    // Add boards from direct pods
    for (auto& [pod_name, pod] : graph->pods) {
        for (auto& [tray_id, board] : pod.boards) {
            std::pair<HostId, TrayId> key = std::make_pair(pod.host_id, tray_id);
            boards_by_host_tray_.emplace(key, &board);
        }
    }

    // Recursively add boards from subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        populate_boards_from_resolved_graph(subgraph);
    }
}

}  // namespace tt::scaleout_tools
