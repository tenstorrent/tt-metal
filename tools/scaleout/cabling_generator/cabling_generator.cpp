// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cabling_generator.hpp"

#include <board/board.hpp>
#include <connector/connector.hpp>
#include <node/node_types.hpp>
#include <node/node.hpp>

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <filesystem>
#include <fstream>
#include <fmt/base.h>
#include <google/protobuf/text_format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>

// Add protobuf includes
#include "protobuf/cluster_config.pb.h"
#include "protobuf/deployment.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

namespace {

// Helper to get source file description for error messages
inline std::string get_source_description(const std::string& source_file) {
    return source_file.empty() ? "merged descriptor" : source_file;
}

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

// Helper function to mark ports as used for inter-board connections
// Only marks ports that are currently available (skips already-marked ports)
void mark_ports_used_for_connections(Node& node) {
    for (const auto& [port_type, connections] : node.inter_board_connections) {
        for (const auto& [board_a, board_b] : connections) {
            // Check if port is still available before marking (may already be marked from template)
            const auto& available_ports_a = node.boards.at(board_a.first).get_available_port_ids(port_type);
            if (std::find(available_ports_a.begin(), available_ports_a.end(), board_a.second) !=
                available_ports_a.end()) {
                node.boards.at(board_a.first).mark_port_used(port_type, board_a.second);
            }
            const auto& available_ports_b = node.boards.at(board_b.first).get_available_port_ids(port_type);
            if (std::find(available_ports_b.begin(), available_ports_b.end(), board_b.second) !=
                available_ports_b.end()) {
                node.boards.at(board_b.first).mark_port_used(port_type, board_b.second);
            }
        }
    }
}

// Helper functions for loading cluster and deployment descriptors
cabling_generator::proto::ClusterDescriptor load_cluster_descriptor(const std::string& file_path) {
    return load_descriptor_from_textproto<cabling_generator::proto::ClusterDescriptor>(file_path);
}

deployment::proto::DeploymentDescriptor load_deployment_descriptor(const std::string& file_path) {
    return load_descriptor_from_textproto<deployment::proto::DeploymentDescriptor>(file_path);
}

// Build endpoint map for node template inter-board connections (validates conflicts per port type)
std::map<Node::BoardEndpoint, Node::BoardEndpoint> build_endpoint_map_for_port_type(
    const std::vector<Node::BoardConnection>& connections,
    const std::string& node_template_name,
    PortType port_type,
    const std::string& template_source) {
    std::map<Node::BoardEndpoint, Node::BoardEndpoint> endpoint_to_dest;
    std::set<Node::BoardConnection> seen_connections;

    // Lambda to check for duplicate endpoint in connection map
    auto check_duplicate_endpoint = [&](const Node::BoardEndpoint& endpoint) {
        if (endpoint_to_dest.contains(endpoint)) {
            throw std::runtime_error(fmt::format(
                "Duplicate connection definition in node template '{}' for port type {} in {}: port (tray_id: {}, "
                "port_id: {}) appears multiple times",
                node_template_name,
                enchantum::to_string(port_type),
                template_source,
                endpoint.first.get(),
                endpoint.second.get()));
        }
    };

    for (const auto& [endpoint_a, endpoint_b] : connections) {
        auto normalized = normalize_node_connection(Node::BoardConnection(endpoint_a, endpoint_b));
        if (seen_connections.contains(normalized)) {
            continue;  // Skip duplicate
        }
        seen_connections.insert(normalized);

        check_duplicate_endpoint(endpoint_a);
        check_duplicate_endpoint(endpoint_b);

        endpoint_to_dest[endpoint_a] = endpoint_b;
        endpoint_to_dest[endpoint_b] = endpoint_a;
    }
    return endpoint_to_dest;
}

// Validate and merge node templates from another CablingGenerator
// Helper: Validate basic node structure (motherboard, boards)
void validate_node_structure(
    const Node& this_template,
    const Node& other_template,
    const std::string& node_desc_name,
    const std::string& new_source_file) {
    if (this_template.motherboard != other_template.motherboard) {
        throw std::runtime_error(fmt::format(
            "Node template '{}' has conflicting motherboard: '{}' vs '{}' from {}",
            node_desc_name,
            this_template.motherboard,
            other_template.motherboard,
            get_source_description(new_source_file)));
    }
    if (this_template.boards.size() != other_template.boards.size()) {
        throw std::runtime_error(fmt::format(
            "Node template '{}' has conflicting board count: {} vs {} from {}",
            node_desc_name,
            this_template.boards.size(),
            other_template.boards.size(),
            get_source_description(new_source_file)));
    }
    for (const auto& [tray_id, this_board] : this_template.boards) {
        if (!other_template.boards.contains(tray_id)) {
            throw std::runtime_error(fmt::format(
                "Node template '{}' missing board at tray_id {} from {}",
                node_desc_name,
                *tray_id,
                get_source_description(new_source_file)));
        }
        const auto& other_board = other_template.boards.at(tray_id);
        if (this_board.get_arch() != other_board.get_arch()) {
            throw std::runtime_error(fmt::format(
                "Node template '{}' board at tray_id {} has conflicting architecture from {}",
                node_desc_name,
                *tray_id,
                get_source_description(new_source_file)));
        }
    }
}

// Helper: Validate inter_board_connections for conflicts
void validate_inter_board_connections(
    const Node& this_template,
    const Node& other_template,
    const std::string& node_desc_name,
    const std::string& existing_source_file,
    const std::string& new_source_file) {
    for (const auto& [port_type, this_conns] : this_template.inter_board_connections) {
        if (!other_template.inter_board_connections.contains(port_type)) {
            continue;  // Port type not in other template, skip
        }
        const auto& other_conns = other_template.inter_board_connections.at(port_type);

        // Build endpoint maps for this port type only
        auto this_endpoint_to_dest = build_endpoint_map_for_port_type(
            this_conns,
            node_desc_name,
            port_type,
            existing_source_file.empty() ? "merged descriptor" : existing_source_file);
        auto other_endpoint_to_dest = build_endpoint_map_for_port_type(
            other_conns, node_desc_name, port_type, new_source_file.empty() ? "merged descriptor" : new_source_file);

        // Check for conflicts between templates for this port type
        for (const auto& [endpoint, this_dest] : this_endpoint_to_dest) {
            if (other_endpoint_to_dest.contains(endpoint)) {
                const auto& other_dest = other_endpoint_to_dest.at(endpoint);
                if (this_dest != other_dest) {
                    throw std::runtime_error(fmt::format(
                        "Connection conflict in node template '{}' for port type {} between templates: port "
                        "(tray_id: {}, port_id: {}) "
                        "connected to (tray_id: {}, port_id: {}) in {} vs (tray_id: {}, port_id: {}) in {}",
                        node_desc_name,
                        enchantum::to_string(port_type),
                        endpoint.first.get(),
                        endpoint.second.get(),
                        this_dest.first.get(),
                        this_dest.second.get(),
                        get_source_description(existing_source_file),
                        other_dest.first.get(),
                        other_dest.second.get(),
                        get_source_description(new_source_file)));
                }
            }
        }
    }
}

// Helper: Merge inter_board_connections with deduplication
void merge_inter_board_connections(Node& target_template, const Node& source_template) {
    for (const auto& [port_type, source_conns] : source_template.inter_board_connections) {
        auto& target_conns = target_template.inter_board_connections[port_type];

        // Use set to deduplicate connections
        std::set<Node::BoardConnection> connection_set;
        for (const auto& conn : target_conns) {
            connection_set.insert(normalize_node_connection(conn));
        }
        for (const auto& conn : source_conns) {
            connection_set.insert(normalize_node_connection(conn));
        }

        // Write back deduplicated connections
        target_conns.clear();
        target_conns.assign(connection_set.begin(), connection_set.end());
    }
}

// Helper: Build normalized set of node-level board connections
std::set<Node::BoardConnection> build_normalized_board_connection_set(
    const std::vector<Node::BoardConnection>& connections) {
    std::set<Node::BoardConnection> normalized_set;
    for (const auto& conn : connections) {
        normalized_set.insert(normalize_node_connection(conn));
    }
    return normalized_set;
}

// Helper: Build normalized set of graph-level port connections
std::set<PortConnection> build_normalized_graph_connection_set(const std::vector<PortConnection>& connections) {
    std::set<PortConnection> normalized_set;
    for (const auto& conn : connections) {
        normalized_set.insert(normalize_graph_connection(conn));
    }
    return normalized_set;
}

// Helper: Check if two nodes (by template key) are torus-compatible for merging
bool are_torus_compatible_for_merge(const std::string& desc_name_a, const std::string& desc_name_b) {
    auto node_type_a = get_node_type_from_string(desc_name_a);
    auto node_type_b = get_node_type_from_string(desc_name_b);

    if (!is_torus(node_type_a) || !is_torus(node_type_b)) {
        return false;
    }

    auto node_a = create_node_instance(node_type_a);
    auto node_b = create_node_instance(node_type_b);

    return node_a->get_architecture() == node_b->get_architecture();
}

// Helper to find a torus-compatible template descriptor name (returns empty optional if not found)
std::optional<std::string> find_torus_compatible_template(
    const std::unordered_map<std::string, Node>& templates_to_search, const std::string& missing_node_desc_name) {
    // Look for torus-compatible template (same architecture and both torus)
    for (const auto& [desc_name, template_node] : templates_to_search) {
        if (are_torus_compatible_for_merge(missing_node_desc_name, desc_name)) {
            return desc_name;  // Found compatible template
        }
    }

    return std::nullopt;
}

// Helper: Try to find and merge torus-compatible template
bool try_merge_torus_compatible_template(
    std::unordered_map<std::string, Node>& this_node_desc_name_to_node,
    const std::string& missing_node_desc_name,
    const Node& missing_template,
    const std::string& existing_source_file,
    const std::string& new_source_file) {
    // Find a compatible template
    auto compatible_desc_name = find_torus_compatible_template(this_node_desc_name_to_node, missing_node_desc_name);

    if (!compatible_desc_name) {
        return false;  // No compatible template found
    }

    // Found compatible template - merge connections
    log_info(
        tt::LogDistributed,
        "Merging torus-compatible node templates '{}' (from {}) with '{}' (from {})",
        missing_node_desc_name,
        get_source_description(new_source_file),
        *compatible_desc_name,
        get_source_description(existing_source_file));

    merge_inter_board_connections(this_node_desc_name_to_node[*compatible_desc_name], missing_template);

    // Add the missing template as an alias
    this_node_desc_name_to_node[missing_node_desc_name] = this_node_desc_name_to_node[*compatible_desc_name];

    return true;
}

void validate_and_merge_node_templates(
    std::unordered_map<std::string, Node>& this_node_desc_name_to_node,
    const std::unordered_map<std::string, Node>& other_node_desc_name_to_node,
    const std::string& existing_source_file,
    const std::string& new_source_file) {
    // Forward pass: validate/merge templates that exist in both
    for (const auto& [node_desc_name, other_template] : other_node_desc_name_to_node) {
        if (this_node_desc_name_to_node.contains(node_desc_name)) {
            // Template exists in both - validate it matches
            const auto& this_template = this_node_desc_name_to_node.at(node_desc_name);
            validate_node_structure(this_template, other_template, node_desc_name, new_source_file);
            validate_inter_board_connections(
                this_template, other_template, node_desc_name, existing_source_file, new_source_file);
        } else {
            // Template missing in 'this' - try torus-compatible merge or throw error
            bool merged = try_merge_torus_compatible_template(
                this_node_desc_name_to_node, node_desc_name, other_template, existing_source_file, new_source_file);

            if (!merged) {
                throw std::runtime_error(fmt::format(
                    "Node template '{}' exists in {} but not in {} - structural mismatch",
                    node_desc_name,
                    get_source_description(new_source_file),
                    get_source_description(existing_source_file)));
            }
        }
    }

    // Backward pass: check for templates in 'this' that don't exist in 'other'
    for (const auto& [node_desc_name, this_template] : this_node_desc_name_to_node) {
        if (!other_node_desc_name_to_node.contains(node_desc_name)) {
            // Check if this is a torus type that can be merged with another torus variant
            bool found_compatible =
                find_torus_compatible_template(other_node_desc_name_to_node, node_desc_name).has_value();

            if (!found_compatible) {
                throw std::runtime_error(fmt::format(
                    "Node template '{}' exists in {} but not in {} - structural mismatch",
                    node_desc_name,
                    get_source_description(existing_source_file),
                    get_source_description(new_source_file)));
            }
        }
    }
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

    auto node_type = get_node_type_from_string(node_descriptor_name);
    return create_node_descriptor(node_type);
}

void create_port_connection(
    Board& board_a,
    Board& board_b,
    PortType port_type,
    HostId host_a_id,
    HostId host_b_id,
    TrayId board_a_id,
    TrayId board_b_id,
    PortId port_a_id,
    PortId port_b_id) {
    const auto& available_a = board_a.get_available_port_ids(port_type);
    const auto& available_b = board_b.get_available_port_ids(port_type);

    if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
        throw std::runtime_error(fmt::format(
            "Port not available: {} Port {} on board {} in host {}",
            enchantum::to_string(port_type),
            *port_a_id,
            *board_a_id,
            *host_a_id));
    }
    if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
        throw std::runtime_error(fmt::format(
            "Port not available: {} Port {} on board {} in host {}",
            enchantum::to_string(port_type),
            *port_b_id,
            *board_b_id,
            *host_b_id));
    }

    if (board_a.get_arch() != board_b.get_arch()) {
        throw std::runtime_error("Trying to connect boards with different architectures");
    }

    board_a.mark_port_used(port_type, port_a_id);
    board_b.mark_port_used(port_type, port_b_id);
}
// Build node from descriptor with port connections and validation
Node build_node(
    const std::string& node_descriptor_name,
    HostId host_id,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    std::unordered_map<std::string, Node>& node_templates) {
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

    // Lambda to validate endpoint conflicts in connection map
    auto validate_endpoint_conflict = [](const Node::BoardEndpoint& endpoint,
                                         const Node::BoardEndpoint& expected_dest,
                                         const std::map<Node::BoardEndpoint, Node::BoardEndpoint>& endpoint_to_dest,
                                         const std::string& context_name) {
        auto it = endpoint_to_dest.find(endpoint);
        if (it != endpoint_to_dest.end() && it->second != expected_dest) {
            throw std::runtime_error(fmt::format(
                "Connection conflict in {}: port (tray_id: {}, port_id: {}) "
                "connected to both (tray_id: {}, port_id: {}) and (tray_id: {}, port_id: {})",
                context_name,
                endpoint.first.get(),
                endpoint.second.get(),
                it->second.first.get(),
                it->second.second.get(),
                expected_dest.first.get(),
                expected_dest.second.get()));
        }
    };

    // Validate connection conflicts FIRST - this only needs the connection pairs, not boards/motherboard
    // Add inter-board connections and validate/mark ports
    // First, validate no conflicts (same port connected to different destinations) - per port type
    for (const auto& [port_type_str, port_connections] : node_descriptor.port_type_connections()) {
        auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
        if (!port_type.has_value()) {
            throw std::runtime_error("Invalid port type: " + port_type_str);
        }

        // Validate conflicts per port type (same port can connect to different destinations on different port types)
        std::map<Node::BoardEndpoint, Node::BoardEndpoint> endpoint_to_dest;
        for (const auto& conn : port_connections.connections()) {
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());
            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            Node::BoardEndpoint endpoint_a = std::make_pair(board_a_id, port_a_id);
            Node::BoardEndpoint endpoint_b = std::make_pair(board_b_id, port_b_id);

            // Check for conflicts: same endpoint connected to different destinations (within this port type)
            validate_endpoint_conflict(endpoint_a, endpoint_b, endpoint_to_dest, "node descriptor '" + node_descriptor_name + "'");
            validate_endpoint_conflict(endpoint_b, endpoint_a, endpoint_to_dest, "node descriptor '" + node_descriptor_name + "'");

            endpoint_to_dest[endpoint_a] = endpoint_b;
            endpoint_to_dest[endpoint_b] = endpoint_a;
        }
    }

    if (node_descriptor.motherboard().empty()) {
        throw std::runtime_error("Node descriptor " + node_descriptor_name + " missing motherboard");
    }
    template_node.motherboard = node_descriptor.motherboard();

    // Create boards with internal connections marked (using cached boards)
    for (const auto& board_item : node_descriptor.boards().board()) {
        TrayId tray_id = TrayId(board_item.tray_id());
        auto board_type = get_board_type_from_string(board_item.board_type());
        template_node.boards.emplace(tray_id, create_board(board_type));
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

            create_port_connection(
                template_node.boards.at(board_a_id),
                template_node.boards.at(board_b_id),
                *port_type,
                host_id,
                host_id,
                board_a_id,
                board_b_id,
                port_a_id,
                port_b_id);

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

// Resolve path from proto data to HostId
HostId resolve_path_from_proto(
    const google::protobuf::RepeatedPtrField<std::string>& path,
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    uint32_t index = 0) {
    if (path.empty()) {
        throw std::runtime_error("Empty path in connection - invalid descriptor");
    }
    if (index >= path.size()) {
        throw std::runtime_error("Path index out of bounds - invalid descriptor");
    }
    if (index == path.size() - 1) {
        // Direct node reference - look up in child_mappings
        const std::string& node_name = path[index];
        if (!graph_instance.child_mappings().contains(node_name)) {
            throw std::runtime_error(
                "Node '" + node_name + "' not found in child_mappings of instance '" + graph_instance.template_name() +
                "'");
        }
        const auto& child_mapping = graph_instance.child_mappings().at(node_name);

        if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
            return HostId(child_mapping.host_id());
        }
        throw std::runtime_error("Node " + node_name + " is not a leaf node");
    }
    // Multi-level path - descend into subgraph
    const std::string& subgraph_name = path[index];
    if (!graph_instance.child_mappings().contains(subgraph_name)) {
        throw std::runtime_error(fmt::format(
            "Child mapping not found: '{}' in instance '{}'", subgraph_name, graph_instance.template_name()));
    }
    const auto& child_mapping = graph_instance.child_mappings().at(subgraph_name);

    if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
        return resolve_path_from_proto(path, child_mapping.sub_instance(), cluster_descriptor, index + 1);
    }
    throw std::runtime_error("Subgraph " + subgraph_name + " is not a graph instance");
}

// Builds a resolved graph instance from a graph instance and deployment descriptor.
// Recursively build tree structure from protobuf graph instance
// deployment_descriptor is optional - if nullptr, no validation is performed.
// Internal implementation of build_graph_instance that handles optional deployment_descriptor
std::unique_ptr<ResolvedGraphInstance> build_graph_instance_impl(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor* deployment_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates) {
    auto resolved = std::make_unique<ResolvedGraphInstance>();
    resolved->template_name = graph_instance.template_name();
    resolved->instance_name = instance_name;

    // Get the template definition
    if (!cluster_descriptor.graph_templates().contains(graph_instance.template_name())) {
        throw std::runtime_error(fmt::format("Graph template not found: '{}'", graph_instance.template_name()));
    }
    const auto& template_def = cluster_descriptor.graph_templates().at(graph_instance.template_name());

    // Build children based on template + instance mapping
    for (const auto& child_def : template_def.children()) {
        const std::string& child_name = child_def.name();
        if (!graph_instance.child_mappings().contains(child_name)) {
            throw std::runtime_error(fmt::format(
                "Child mapping not found: '{}' in instance '{}'", child_name, graph_instance.template_name()));
        }
        const auto& child_mapping = graph_instance.child_mappings().at(child_name);

        if (child_def.has_node_ref()) {
            // Leaf node - create node
            if (child_mapping.mapping_case() != tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
                throw std::runtime_error("Node child must have host_id mapping: " + child_name);
            }

            HostId host_id = HostId(child_mapping.host_id());
            const std::string& node_descriptor_name = child_def.node_ref().node_descriptor();

            // Validate deployment node type if deployment descriptor is provided
            if (deployment_descriptor != nullptr) {
                if (*host_id < deployment_descriptor->hosts().size()) {
                    const auto& deployment_host = deployment_descriptor->hosts()[*host_id];
                    if (!deployment_host.node_type().empty() && deployment_host.node_type() != node_descriptor_name) {
                        throw std::runtime_error(
                            "Node type mismatch for host " + deployment_host.host() + " (host_id " +
                            std::to_string(*host_id) + "): deployment specifies '" + deployment_host.node_type() +
                            "' but cluster configuration expects '" + node_descriptor_name + "'");
                    }
                } else {
                    throw std::runtime_error("Host ID " + std::to_string(*host_id) + " not found in deployment");
                }
            }

            // Find node descriptor and build node
            resolved->nodes[child_name] = build_node(node_descriptor_name, host_id, cluster_descriptor, node_templates);

        } else if (child_def.has_graph_ref()) {
            // Non-leaf node - recursively build subgraph
            if (child_mapping.mapping_case() !=
                tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
                throw std::runtime_error("Graph child must have sub_instance mapping: " + child_name);
            }

            resolved->subgraphs[child_name] = build_graph_instance_impl(
                child_mapping.sub_instance(), cluster_descriptor, deployment_descriptor, child_name, node_templates);
        }
    }

    // Process internal connections within this graph instance
    for (const auto& [port_type_str, port_connections] : template_def.internal_connections()) {
        auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
        if (!port_type.has_value()) {
            throw std::runtime_error("Invalid port type: " + port_type_str);
        }

        for (const auto& conn : port_connections.connections()) {
            const auto& path_a = conn.port_a().path();
            const auto& path_b = conn.port_b().path();
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());

            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            // Resolve paths to HostId using proto data
            HostId host_a_id = resolve_path_from_proto(path_a, graph_instance, cluster_descriptor);
            HostId host_b_id = resolve_path_from_proto(path_b, graph_instance, cluster_descriptor);

            // Store connection with resolved HostId
            PortConnection port_conn(
                std::make_tuple(host_a_id, board_a_id, port_a_id), std::make_tuple(host_b_id, board_b_id, port_b_id));

            // Add to this graph instance's connections
            resolved->add_connection(*port_type, port_conn);
        }
    }

    return resolved;
}

// Wrapper: build graph instance without deployment descriptor
std::unique_ptr<ResolvedGraphInstance> build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates) {
    return build_graph_instance_impl(graph_instance, cluster_descriptor, nullptr, instance_name, node_templates);
}

void populate_deployment_hosts(
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::unordered_map<std::string, Node>& node_templates,
    std::vector<Host>& deployment_hosts) {
    // Store deployment hosts
    deployment_hosts.reserve(deployment_descriptor.hosts().size());
    for (const auto& proto_host : deployment_descriptor.hosts()) {
        if (!node_templates.contains(proto_host.node_type())) {
            throw std::runtime_error(
                "Node type '" + proto_host.node_type() + "' from deployment descriptor host '" + proto_host.host() +
                "' not found in cluster descriptor templates");
        }
        deployment_hosts.emplace_back(Host{
            .hostname = proto_host.host(),
            .hall = proto_host.hall(),
            .aisle = proto_host.aisle(),
            .rack = proto_host.rack(),
            .shelf_u = proto_host.shelf_u(),
            .motherboard = node_templates.at(proto_host.node_type()).motherboard,
            .node_type = proto_host.node_type()});
    }
}

void populate_deployment_hosts_from_hostnames(
    const std::vector<std::string>& hostnames,
    const std::map<HostId, Node*>& host_id_to_node,
    std::vector<Host>& deployment_hosts) {
    // Store deployment hosts with just hostname and motherboard (no physical location info)
    deployment_hosts.reserve(hostnames.size());
    for (size_t i = 0; i < hostnames.size(); ++i) {
        HostId host_id = HostId(i);
        auto it = host_id_to_node.find(host_id);
        if (it == host_id_to_node.end()) {
            throw std::runtime_error("Host ID " + std::to_string(i) + " not found in cluster configuration");
        }
        deployment_hosts.emplace_back(Host{
            .hostname = hostnames[i],
            .hall = "",
            .aisle = "",
            .rack = 0,
            .shelf_u = 0,
            .motherboard = it->second->motherboard});
    }
}

// Wrapper: build graph instance with deployment descriptor
std::unique_ptr<ResolvedGraphInstance> build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates) {
    return build_graph_instance_impl(
        graph_instance, cluster_descriptor, &deployment_descriptor, instance_name, node_templates);
}

}  // anonymous namespace

// Common initialization logic shared by all constructors
void CablingGenerator::initialize_cluster(
    const cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    std::optional<std::reference_wrapper<const deployment::proto::DeploymentDescriptor>> deployment_descriptor) {
    // Build cluster with all connections and port validation
    if (deployment_descriptor.has_value()) {
        root_instance_ = build_graph_instance(
            cluster_descriptor.root_instance(), cluster_descriptor, deployment_descriptor->get(), "", node_templates_);
    } else {
        root_instance_ =
            build_graph_instance(cluster_descriptor.root_instance(), cluster_descriptor, "", node_templates_);
    }

    // Validate host_id uniqueness across all nodes
    validate_host_id_uniqueness();

    // Populate the host_id_to_node_ map
    populate_host_id_to_node();

    // Generate all logical chip connections
    generate_logical_chip_connections();
}

std::vector<std::string> CablingGenerator::find_descriptor_files(const std::string& directory_path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".textproto") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Error reading directory " + directory_path + ": " + e.what());
    }

    // Sorting files so we don't have non-deterministic order of files in the directory, making errors easier to debug.
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        throw std::runtime_error("No .textproto files found in directory: " + directory_path);
    }
    return files;
}

// Helper to build from directory by merging multiple files (friend of CablingGenerator)
template <typename DeploymentArg>
CablingGenerator build_from_directory(const std::string& dir_path, const DeploymentArg& deployment_arg) {
    auto descriptor_files = CablingGenerator::find_descriptor_files(dir_path);

    log_info(
        tt::LogDistributed, "Found {} cabling descriptor files in directory: {}", descriptor_files.size(), dir_path);
    for (const auto& file : descriptor_files) {
        log_info(tt::LogDistributed, "  - {}", file);
    }

    // Create the first CablingGenerator from the first file
    CablingGenerator merged(descriptor_files[0], deployment_arg);
    std::string merged_source_description = descriptor_files[0];

    // Merge all remaining files into it
    for (size_t i = 1; i < descriptor_files.size(); ++i) {
        merged.merge(descriptor_files[i], deployment_arg, merged_source_description);
        merged_source_description += ", " + descriptor_files[i];
    }
    return merged;
}

// Helper to update lookup structures when adding a connection to ResolvedGraphInstance
void ResolvedGraphInstance::add_connection(PortType port_type, const PortConnection& conn) {
    internal_connections[port_type].push_back(conn);
    endpoint_to_dest[conn.first] = conn.second;
    endpoint_to_dest[conn.second] = conn.first;
    // Normalize for duplicate detection (smaller endpoint first)
    auto normalized = normalize_graph_connection(conn);
    connection_pairs.insert(normalized);
}

// Constructor with full deployment descriptor
// cluster_descriptor_path can be a single file or a directory
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path) {
    if (!std::filesystem::exists(cluster_descriptor_path)) {
        throw std::runtime_error("Path does not exist: " + cluster_descriptor_path);
    }
    if (std::filesystem::is_directory(cluster_descriptor_path)) {
        auto merged = build_from_directory(cluster_descriptor_path, deployment_descriptor_path);
        node_templates_ = std::move(merged.node_templates_);
        root_instance_ = std::move(merged.root_instance_);
        host_id_to_node_ = std::move(merged.host_id_to_node_);
        chip_connections_ = std::move(merged.chip_connections_);
        deployment_hosts_ = std::move(merged.deployment_hosts_);
    } else {
        auto cluster_descriptor = load_cluster_descriptor(cluster_descriptor_path);
        auto deployment_descriptor = load_deployment_descriptor(deployment_descriptor_path);
        initialize_cluster(cluster_descriptor, deployment_descriptor);
        populate_deployment_hosts(deployment_descriptor, node_templates_, deployment_hosts_);
    }
}

// Constructor with just hostnames (no physical location info)
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::vector<std::string>& hostnames) {
    if (!std::filesystem::exists(cluster_descriptor_path)) {
        throw std::runtime_error("Path does not exist: " + cluster_descriptor_path);
    }
    if (std::filesystem::is_directory(cluster_descriptor_path)) {
        auto merged = build_from_directory(cluster_descriptor_path, hostnames);
        node_templates_ = std::move(merged.node_templates_);
        root_instance_ = std::move(merged.root_instance_);
        host_id_to_node_ = std::move(merged.host_id_to_node_);
        chip_connections_ = std::move(merged.chip_connections_);
        deployment_hosts_ = std::move(merged.deployment_hosts_);
    } else {
        auto cluster_descriptor = load_cluster_descriptor(cluster_descriptor_path);
        initialize_cluster(cluster_descriptor);
        populate_deployment_hosts_from_hostnames(hostnames, host_id_to_node_, deployment_hosts_);
    }
}

// Constructor with ClusterDescriptor protobuf and hostnames (no file I/O required)
CablingGenerator::CablingGenerator(
    const cabling_generator::proto::ClusterDescriptor& cluster_descriptor, const std::vector<std::string>& hostnames) {
    initialize_cluster(cluster_descriptor);
    populate_deployment_hosts_from_hostnames(hostnames, host_id_to_node_, deployment_hosts_);
}

// Helper to find template key for a node by matching motherboard, board count, architecture, and board type
static std::optional<std::string> find_template_key_for_node(
    const Node& node, const std::unordered_map<std::string, Node>& node_desc_name_to_node) {
    for (const auto& [desc_name, template_node] : node_desc_name_to_node) {
        if (template_node.motherboard == node.motherboard && template_node.boards.size() == node.boards.size()) {
            // Check board architectures and types match
            bool all_match = true;
            for (const auto& [tray_id, board] : node.boards) {
                if (!template_node.boards.contains(tray_id) ||
                    template_node.boards.at(tray_id).get_arch() != board.get_arch() ||
                    template_node.boards.at(tray_id).get_board_type() != board.get_board_type()) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                return desc_name;
            }
        }
    }
    return std::nullopt;
}

// Helper to recreate a node from its template, resetting port availability
// After merging descriptors, nodes may have stale port usage info. This function
// preserves host_id and merged inter_board_connections
static Node create_base_node_from_template(
    const Node& source_node, const std::unordered_map<std::string, Node>& node_templates) {
    // Find the template by matching motherboard and board structure
    auto template_key = find_template_key_for_node(source_node, node_templates);

    if (!template_key) {
        throw std::runtime_error(fmt::format(
            "Could not find template for node with motherboard '{}' and {} boards",
            source_node.motherboard,
            source_node.boards.size()));
    }

    const Node& template_node = node_templates.at(*template_key);

    // Create base node from template - this resets port availability
    // (template only has ports marked as used for inter-board connections from node descriptor)
    Node base_node = template_node;
    base_node.host_id = source_node.host_id;
    // Copy inter_board_connections from source (they may have been merged from multiple files)
    base_node.inter_board_connections = source_node.inter_board_connections;
    // Re-mark ports as used for the merged inter-board connections
    mark_ports_used_for_connections(base_node);
    return base_node;
}

// Helper to merge two ResolvedGraphInstance trees
static void merge_resolved_graph_instances(
    ResolvedGraphInstance& target,
    const ResolvedGraphInstance& source,
    const std::string& existing_source_file,
    const std::string& new_source_file,
    const std::unordered_map<std::string, Node>& node_templates) {
    // Validate template_name matches
    if (target.template_name != source.template_name) {
        throw std::runtime_error(fmt::format(
            "Cannot merge graph instances with different template names: '{}' vs '{}' from {}",
            target.template_name,
            source.template_name,
            get_source_description(new_source_file)));
    }

    // Merge nodes - if same name exists, validate host_id matches
    // (motherboard, board count, and architecture are already validated via templates in merge())
    for (const auto& [name, source_node] : source.nodes) {
        if (target.nodes.contains(name)) {
            // Node exists in both - validate host_id matches (instance-specific, not template)
            if (target.nodes[name].host_id != source_node.host_id) {
                throw std::runtime_error(fmt::format(
                    "Node '{}' has conflicting host_id: {} vs {} from {}",
                    name,
                    target.nodes[name].host_id.get(),
                    source_node.host_id.get(),
                    get_source_description(new_source_file)));
            }
            // Validate inter_board_connections match or are torus-compatible
            // For torus-compatible nodes, we merge the connections
            // For non-torus nodes, we only merge internal_connections, not inter_board_connections

            // Check if this is a torus-compatible merge scenario
            auto target_template_key = find_template_key_for_node(target.nodes[name], node_templates);
            auto source_template_key = find_template_key_for_node(source_node, node_templates);

            bool is_torus_merge =
                (target_template_key && source_template_key &&
                 are_torus_compatible_for_merge(*target_template_key, *source_template_key));

            if (is_torus_merge) {
                // Torus-compatible merge: combine inter_board_connections
                if (existing_source_file.empty() && new_source_file.empty()) {
                    throw std::runtime_error("At least one source file name must be provided for merge error messages");
                }
                log_info(
                    tt::LogDistributed,
                    "Merging torus-compatible node '{}' inter_board_connections from {} and {}",
                    name,
                    get_source_description(existing_source_file),
                    get_source_description(new_source_file));

                // Merge connections from both nodes using helper function
                merge_inter_board_connections(target.nodes[name], source_node);
            } else {
                // Non-torus: validate inter_board_connections match exactly
                // Build normalized sets for comparison (build once per node, not per port type)
                std::map<PortType, std::set<Node::BoardConnection>> target_sets, source_sets;

                // Pre-build all sets for both target and source
                for (const auto& [port_type, connections] : target.nodes[name].inter_board_connections) {
                    target_sets[port_type] = build_normalized_board_connection_set(connections);
                }
                for (const auto& [port_type, connections] : source_node.inter_board_connections) {
                    source_sets[port_type] = build_normalized_board_connection_set(connections);
                }

                // Now compare the sets
                for (const auto& [port_type, target_set] : target_sets) {
                    if (!source_sets.contains(port_type)) {
                        throw std::runtime_error(fmt::format(
                            "Node '{}' has port type {} in {} but not in {} - inconsistent inter_board_connections "
                            "usage",
                            name,
                            enchantum::to_string(port_type),
                            get_source_description(existing_source_file),
                            get_source_description(new_source_file)));
                    }

                    if (target_set != source_sets[port_type]) {
                        throw std::runtime_error(fmt::format(
                            "Node '{}' has conflicting inter_board_connections: {} and {} have different "
                            "inter-board connections (we only merge inter-node connections, not "
                            "inter_board_connections)",
                            name,
                            get_source_description(existing_source_file),
                            get_source_description(new_source_file)));
                    }
                }
                // Also check for port types in source that don't exist in target
                for (const auto& [port_type, connections] : source_node.inter_board_connections) {
                    if (!target.nodes[name].inter_board_connections.contains(port_type)) {
                        throw std::runtime_error(fmt::format(
                            "Node '{}' has inter_board_connections for port type {} in {} but not in {}",
                            name,
                            enchantum::to_string(port_type),
                            get_source_description(new_source_file),
                            get_source_description(existing_source_file)));
                    }
                }
            }
        } else {
            // Node exists in source but not in target
            throw std::runtime_error(fmt::format(
                "Cannot merge graph_template '{}': node '{}' exists in {} but not in {}. "
                "Graph templates must have identical children across all descriptor files.",
                target.template_name,
                name,
                get_source_description(new_source_file),
                get_source_description(existing_source_file)));
        }
    }

    // Merge subgraphs recursively
    for (const auto& [name, source_subgraph] : source.subgraphs) {
        if (target.subgraphs.contains(name)) {
            // Subgraph exists - merge recursively
            merge_resolved_graph_instances(
                *target.subgraphs[name], *source_subgraph, existing_source_file, new_source_file, node_templates);
        } else {
            // Subgraph exists in source but not in target
            throw std::runtime_error(fmt::format(
                "Cannot merge graph_template '{}': subgraph '{}' exists in {} but not in {}. "
                "Graph templates must have identical children across all descriptor files.",
                target.template_name,
                name,
                get_source_description(new_source_file),
                get_source_description(existing_source_file)));
        }
    }

    // Backward pass: Validate that target doesn't have any children that source doesn't have
    // This ensures graph_templates have identical children across all descriptor files
    for (const auto& [name, _] : target.nodes) {
        if (!source.nodes.contains(name)) {
            throw std::runtime_error(fmt::format(
                "Cannot merge graph_template '{}': node '{}' exists in {} but not in {}. "
                "Graph templates must have identical children across all descriptor files.",
                target.template_name,
                name,
                get_source_description(existing_source_file),
                get_source_description(new_source_file)));
        }
    }
    for (const auto& [name, _] : target.subgraphs) {
        if (!source.subgraphs.contains(name)) {
            throw std::runtime_error(fmt::format(
                "Cannot merge graph_template '{}': subgraph '{}' exists in {} but not in {}. "
                "Graph templates must have identical children across all descriptor files.",
                target.template_name,
                name,
                get_source_description(existing_source_file),
                get_source_description(new_source_file)));
        }
    }

    // Merge internal_connections using lookup structures from ResolvedGraphInstance
    for (const auto& [port_type, source_conns] : source.internal_connections) {
        for (const auto& conn : source_conns) {
            // Normalize graph-level connection (smaller endpoint first)
            auto normalized = normalize_graph_connection(conn);

            if (target.connection_pairs.contains(normalized)) {
                // Duplicate - warn but allow
                log_warning(
                    tt::LogDistributed,
                    "Duplicate connection in template '{}' from {}",
                    target.template_name,
                    get_source_description(new_source_file));
            } else {
                // Add new connection
                // Note: For inter-node connections (internal_connections), we allow a port to connect
                // to different destinations across different descriptors. Physical port exhaustion
                // is validated later during FSD generation, not at merge time.
                target.add_connection(port_type, conn);
            }
        }
    }
}

template <typename DeploymentArg>
void CablingGenerator::merge(
    const std::string& new_file_path, const DeploymentArg& deployment_arg, const std::string& existing_sources) {
    // Create CablingGenerator for the new file
    CablingGenerator other(new_file_path, deployment_arg);

    // Validate and merge node_templates_ (must match exactly, except inter_board_connections can differ)
    validate_and_merge_node_templates(node_templates_, other.node_templates_, existing_sources, new_file_path);

    // Merge root_instance_ trees (we know root_instance_ exists since we start with a non-empty CablingGenerator)
    if (!root_instance_ || !other.root_instance_) {
        throw std::runtime_error("Cannot merge: both CablingGenerators must have root_instance_");
    }
    merge_resolved_graph_instances(
        *root_instance_, *other.root_instance_, existing_sources, new_file_path, node_templates_);

    // Rebuild host_id_to_node_ from merged root_instance
    populate_host_id_to_node();

    // Re-validate host_id uniqueness
    validate_host_id_uniqueness();

    // Before processing connections, recreate all nodes from templates to reset port availability
    // This is needed because nodes from individual files may have ports marked as used from
    // graph-level connections, but we need fresh nodes with only inter-board connection ports marked as used.
    recreate_nodes_from_templates(*root_instance_);

    // Regenerate chip_connections_ from merged root_instance (this will mark ports as used)
    generate_logical_chip_connections();

    // Merge deployment_hosts_ (append unique hostnames)
    std::set<std::string> existing_hostnames;
    for (const auto& host : deployment_hosts_) {
        existing_hostnames.insert(host.hostname);
    }
    for (const auto& other_host : other.deployment_hosts_) {
        if (!existing_hostnames.contains(other_host.hostname)) {
            deployment_hosts_.push_back(other_host);
            existing_hostnames.insert(other_host.hostname);
        }
    }
}

// Getters for all data
const std::vector<Host>& CablingGenerator::get_deployment_hosts() const { return deployment_hosts_; }

const std::vector<LogicalChannelConnection>& CablingGenerator::get_chip_connections() const {
    return chip_connections_;
}

// Helper function to build factory system descriptor protobuf (shared between emit and generate_string methods)
static tt::scaleout_tools::fsd::proto::FactorySystemDescriptor build_factory_system_descriptor(
    const std::vector<Host>& deployment_hosts,
    const std::map<HostId, Node*>& host_id_to_node,
    const std::vector<LogicalChannelConnection>& chip_connections) {
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;

    // Add host information from deployment hosts (indexed by host_id)
    for (const auto& deployment_host : deployment_hosts) {
        auto* host = fsd.add_hosts();
        host->set_hostname(deployment_host.hostname);
        host->set_hall(deployment_host.hall);
        host->set_aisle(deployment_host.aisle);
        host->set_rack(deployment_host.rack);
        host->set_shelf_u(deployment_host.shelf_u);
        host->set_motherboard(deployment_host.motherboard);
    }

    // Add board types
    for (const auto& [host_id, node] : host_id_to_node) {
        for (const auto& [tray_id, board] : node->boards) {
            auto* board_location = fsd.mutable_board_types()->add_board_locations();
            board_location->set_host_id(*host_id);
            board_location->set_tray_id(*tray_id);
            board_location->set_board_type(enchantum::to_string(board.get_board_type()).data());
        }
    }

    // Add ASIC connections from chip_connections
    for (const auto& [start, end] : chip_connections) {
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

    return fsd;
}

// Method to emit textproto factory system descriptor
void CablingGenerator::emit_factory_system_descriptor(const std::string& output_path) const {
    auto fsd = build_factory_system_descriptor(deployment_hosts_, host_id_to_node_, chip_connections_);

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
    printer.SetPrintMessageFieldsInIndexOrder(true);

    if (!printer.PrintToString(fsd, &output_string)) {
        throw std::runtime_error("Failed to write textproto to file: " + output_path);
    }

    output_file << output_string;
    output_file.close();
}

// Method to generate factory system descriptor as protobuf object (uses shared helper)
tt::scaleout_tools::fsd::proto::FactorySystemDescriptor CablingGenerator::generate_factory_system_descriptor() const {
    return build_factory_system_descriptor(deployment_hosts_, host_id_to_node_, chip_connections_);
}

// Helper: Compare two nodes for equality (checks motherboard, boards, host_id, inter_board_connections)
static bool compare_nodes(const Node& lhs, const Node& rhs, bool check_host_id = true) {
    // Compare basic fields
    if (lhs.motherboard != rhs.motherboard) {
        return false;
    }
    if (check_host_id && lhs.host_id != rhs.host_id) {
        return false;
    }

    // Compare boards
    if (lhs.boards.size() != rhs.boards.size()) {
        return false;
    }
    for (const auto& [tray_id, board] : lhs.boards) {
        if (!rhs.boards.contains(tray_id)) {
            return false;
        }
        const auto& other_board = rhs.boards.at(tray_id);
        if (board.get_arch() != other_board.get_arch() || board.get_board_type() != other_board.get_board_type()) {
            return false;
        }
    }

    // Compare inter_board_connections (normalize for comparison)
    if (lhs.inter_board_connections.size() != rhs.inter_board_connections.size()) {
        return false;
    }
    for (const auto& [port_type, connections] : lhs.inter_board_connections) {
        if (!rhs.inter_board_connections.contains(port_type)) {
            return false;
        }
        const auto& other_connections = rhs.inter_board_connections.at(port_type);
        std::set<Node::BoardConnection> lhs_set = build_normalized_board_connection_set(connections);
        std::set<Node::BoardConnection> rhs_set = build_normalized_board_connection_set(other_connections);
        if (lhs_set != rhs_set) {
            return false;
        }
    }

    return true;
}

// Helper to compare two ResolvedGraphInstance trees recursively
static bool compare_resolved_graph_instances(const ResolvedGraphInstance& lhs, const ResolvedGraphInstance& rhs) {
    if (lhs.template_name != rhs.template_name || lhs.instance_name != rhs.instance_name) {
        return false;
    }

    // Compare nodes
    if (lhs.nodes.size() != rhs.nodes.size()) {
        return false;
    }
    for (const auto& [name, node] : lhs.nodes) {
        if (!rhs.nodes.contains(name)) {
            return false;
        }
        const auto& other_node = rhs.nodes.at(name);
        // Compare nodes using helper
        if (!compare_nodes(node, other_node, true)) {
            return false;
        }
    }

    // Compare subgraphs recursively
    if (lhs.subgraphs.size() != rhs.subgraphs.size()) {
        return false;
    }
    for (const auto& [name, subgraph] : lhs.subgraphs) {
        if (!rhs.subgraphs.contains(name) || !rhs.subgraphs.at(name)) {
            return false;
        }
        if (!compare_resolved_graph_instances(*subgraph, *rhs.subgraphs.at(name))) {
            return false;
        }
    }

    // Compare internal_connections (normalize for comparison)
    if (lhs.internal_connections.size() != rhs.internal_connections.size()) {
        return false;
    }
    for (const auto& [port_type, connections] : lhs.internal_connections) {
        if (!rhs.internal_connections.contains(port_type)) {
            return false;
        }
        const auto& other_connections = rhs.internal_connections.at(port_type);
        // Build normalized sets for comparison (graph-level internal_connections)
        std::set<PortConnection> lhs_set = build_normalized_graph_connection_set(connections);
        std::set<PortConnection> rhs_set = build_normalized_graph_connection_set(other_connections);
        if (lhs_set != rhs_set) {
            return false;
        }
    }

    return true;
}

// Equality comparison operator
bool CablingGenerator::operator==(const CablingGenerator& other) const {
    // Compare node_templates_
    // NOTE: When merging torus types (e.g., X_TORUS + Y_TORUS), we create aliases,
    // so we might have 2 keys (WH_GALAXY_X_TORUS, WH_GALAXY_Y_TORUS) pointing to the same merged template,
    // while the reference has 1 key (WH_GALAXY_XY_TORUS). We need to compare the unique templates, not the keys.

    // Collect unique templates by content (using motherboard + boards + connections as key)
    auto get_unique_templates = [](const std::unordered_map<std::string, Node>& templates) {
        std::map<std::tuple<std::string, size_t, size_t>, const Node*> unique;
        for (const auto& [name, node] : templates) {
            // Create a content-based key (motherboard, board count, connection count)
            auto key = std::make_tuple(node.motherboard, node.boards.size(), node.inter_board_connections.size());
            unique[key] = &node;
        }
        return unique;
    };

    auto this_unique = get_unique_templates(node_templates_);
    auto other_unique = get_unique_templates(other.node_templates_);

    if (this_unique.size() != other_unique.size()) {
        return false;
    }

    // Compare each unique template using helper
    for (const auto& [key, template_node] : this_unique) {
        if (!other_unique.contains(key)) {
            return false;
        }
        const auto* other_template = other_unique.at(key);
        // Compare template nodes (don't check host_id since templates should have host_id=0)
        if (!compare_nodes(*template_node, *other_template, false)) {
            return false;
        }
    }

    // Compare root_instance_ (recursive tree comparison)
    if (!root_instance_ && !other.root_instance_) {
        // Both null - equal
    } else if (!root_instance_ || !other.root_instance_) {
        return false;
    }
    if (root_instance_ && other.root_instance_ &&
        !compare_resolved_graph_instances(*root_instance_, *other.root_instance_)) {
        return false;
    }

    // Compare chip_connections_ (should be sorted, so direct comparison)
    if (chip_connections_ != other.chip_connections_) {
        return false;
    }

    // Compare deployment_hosts_
    if (deployment_hosts_ != other.deployment_hosts_) {
        return false;
    }

    return true;
}

void CablingGenerator::emit_cabling_guide_csv(const std::string& output_path, bool loc_info) const {
    // Create parent directory if it doesn't exist
    std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    const std::unordered_map<CableLength, std::string> cable_length_str = {
        {CableLength::CABLE_0P5, "0.5m"},
        {CableLength::CABLE_1, "1m"},
        {CableLength::CABLE_2P5, "2.5m"},
        {CableLength::CABLE_3, "3m"},
        {CableLength::CABLE_5, "5m"},
        {CableLength::UNKNOWN, "UNKNOWN"}};

    const std::unordered_map<tt::ARCH, std::string> speed_str = {
        // TODO: BLACKHOLE cable speed 200G in early stages/validation, but should be able to support 800G in the
        // future.
        {tt::ARCH::WORMHOLE_B0, "400G"},
        {tt::ARCH::BLACKHOLE, "400G"},
        {tt::ARCH::Invalid, "UNKNOWN"}};

    // Unknown for lengths unable to be calculated (longer than avaiable cables, cross-aisle/hall, etc.)

    // Vector of (Host,Tray,Port) Connection Pairs
    std::vector<std::pair<std::tuple<HostId, TrayId, PortId>, std::tuple<HostId, TrayId, PortId>>> conn_list;

    get_all_connections_of_type({PortType::QSFP_DD}, conn_list);
    output_file.fill('0');
    if (loc_info) {
        output_file << "Source,,,,,,,,,Destination,,,,,,,,,Cable Length,Cable Type" << std::endl;
        output_file << "Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type,Hostname,Hall,Aisle,Rack,Shelf "
                       "U,Tray,Port,Label,Node Type,,"
                    << std::endl;
    } else {
        output_file << "Source,,,,Destination,,," << std::endl;
        output_file << "Hostname,Tray,Port,Node Type,Hostname,Tray,Port,Node Type" << std::endl;
    }
    for (const auto& [start, end] : conn_list) {
        auto host_id1 = std::get<0>(start).get();
        auto tray_id1 = std::get<1>(start).get();
        auto port_id1 = std::get<2>(start).get();

        auto host_id2 = std::get<0>(end).get();
        auto tray_id2 = std::get<1>(end).get();
        auto port_id2 = std::get<2>(end).get();

        const auto& host1 = deployment_hosts_[host_id1];
        const auto& host2 = deployment_hosts_[host_id2];

        // Create node_type strings with "_DEFAULT" suffix removed if present;
        //  all the default connections are enumerated
        const std::string suffix = "_DEFAULT";
        std::string host1_node_type = host1.node_type;
        if (host1_node_type.size() >= suffix.size() && host1_node_type.ends_with(suffix)) {
            host1_node_type = host1_node_type.substr(0, host1_node_type.size() - suffix.size());
        }
        std::string host2_node_type = host2.node_type;
        if (host2_node_type.size() >= suffix.size() && host2_node_type.ends_with(suffix)) {
            host2_node_type = host2_node_type.substr(0, host2_node_type.size() - suffix.size());
        }

        // Get arch from node
        // Assume arch for start and end are the same
        // This is validated in create_port_connection

        // TODO: Determine better heuristic/specification for cable length and type
        // auto arch = host_id_to_node_.at(std::get<0>(start))->boards.at(std::get<1>(start)).get_arch();
        // CableLength cable_l = calc_cable_length(host1, tray_id1, host2, tray_id2, host1_node_type);

        if (loc_info) {
            output_file << host1.hostname << ",";
            output_file << host1.hall << "," << host1.aisle << "," << std::setw(2) << host1.rack << ",U" << std::setw(2)
                        << host1.shelf_u << "," << tray_id1 << "," << port_id1 << ",";

            output_file << host1.hall << host1.aisle << std::setw(2) << host1.rack << "U" << std::setw(2)
                        << host1.shelf_u << "-" << tray_id1 << "-" << port_id1 << "," << host1_node_type << ",";

            output_file << host2.hostname << ",";
            output_file << host2.hall << "," << host2.aisle << "," << std::setw(2) << host2.rack << ",U" << std::setw(2)
                        << host2.shelf_u << "," << tray_id2 << "," << port_id2 << ",";
            output_file << host2.hall << host2.aisle << std::setw(2) << host2.rack << "U" << std::setw(2)
                        << host2.shelf_u << "-" << tray_id2 << "-" << port_id2 << "," << host2_node_type << ",";

            output_file << ",";        // Length blank, leaving up to technician
            output_file << std::endl;  // Type blank, leaving up to technician
        } else {
            output_file << host1.hostname << "," << tray_id1 << "," << port_id1 << "," << host1_node_type << ",";
            output_file << host2.hostname << "," << tray_id2 << "," << port_id2 << "," << host2_node_type << std::endl;
        }
    }

    output_file.close();
}

// Validate that each host_id is assigned to exactly one node
void CablingGenerator::validate_host_id_uniqueness() {
    std::unordered_map<HostId, std::string> host_to_node_path;
    collect_host_assignments(host_to_node_path);
}

// Collect all host_id assignments (tree structure)
void CablingGenerator::collect_host_assignments(std::unordered_map<HostId, std::string>& host_to_node_path) {
    if (root_instance_) {
        collect_host_assignments_from_resolved_graph(root_instance_, "", host_to_node_path);
    }
}

void CablingGenerator::collect_host_assignments_from_resolved_graph(
    const std::unique_ptr<ResolvedGraphInstance>& graph,
    const std::string& path_prefix,
    std::unordered_map<HostId, std::string>& host_to_node_path) {
    // Check direct nodes in this graph
    for (const auto& [node_name, node] : graph->nodes) {
        HostId host_id = node.host_id;
        std::string full_node_path = path_prefix.empty() ? node_name : path_prefix + "/" + node_name;

        if (host_to_node_path.contains(host_id)) {
            throw std::runtime_error(fmt::format(
                "Host ID is assigned to multiple nodes: {} - '{}' and '{}'",
                *host_id,
                host_to_node_path[host_id],
                full_node_path));
        }
        host_to_node_path[host_id] = full_node_path;
    }

    // Recursively check subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        std::string sub_path = path_prefix.empty() ? subgraph_name : path_prefix + "/" + subgraph_name;
        collect_host_assignments_from_resolved_graph(subgraph, sub_path, host_to_node_path);
    }
}

// Utility function to generate logical chip connections from tree structure
void CablingGenerator::generate_logical_chip_connections() {
    chip_connections_.clear();

    if (root_instance_) {
        generate_connections_from_resolved_graph(root_instance_);
    }
    std::sort(chip_connections_.begin(), chip_connections_.end());
}

void CablingGenerator::generate_connections_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph) {
    // Lambda to add connections between two ports
    auto add_port_connection = [&](PortType port_type,
                                   const Board& start_board,
                                   const Board& end_board,
                                   HostId start_host_id,
                                   TrayId start_tray_id,
                                   PortId start_port_id,
                                   HostId end_host_id,
                                   TrayId end_tray_id,
                                   PortId end_port_id) {
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
                for (const auto& [port_a_id, port_b_id] : connections) {
                    add_port_connection(
                        port_type, board, board, host_id, tray_id, port_a_id, host_id, tray_id, port_b_id);
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
                add_port_connection(
                    port_type,
                    board_a_ref,
                    board_b_ref,
                    host_id,
                    board_a_id,
                    port_a_id,
                    host_id,
                    board_b_id,
                    port_b_id);
            }
        }
    }

    // Process internal connections within this graph
    for (const auto& [port_type, connections] : graph->internal_connections) {
        for (const auto& [conn_a, conn_b] : connections) {
            auto [host_a_id, tray_a_id, port_a_id] = conn_a;
            auto [host_b_id, tray_b_id, port_b_id] = conn_b;

            // Look up nodes using HostId
            if (!host_id_to_node_.contains(host_a_id)) {
                throw std::runtime_error(fmt::format(
                    "Host ID {} referenced in connection but not found in cluster - invalid descriptor",
                    host_a_id.get()));
            }
            if (!host_id_to_node_.contains(host_b_id)) {
                throw std::runtime_error(fmt::format(
                    "Host ID {} referenced in connection but not found in cluster - invalid descriptor",
                    host_b_id.get()));
            }
            Node* node_a = host_id_to_node_.at(host_a_id);
            Node* node_b = host_id_to_node_.at(host_b_id);

            auto& board_a_ref = node_a->boards.at(tray_a_id);
            auto& board_b_ref = node_b->boards.at(tray_b_id);
            create_port_connection(
                board_a_ref, board_b_ref, port_type, host_a_id, host_b_id, tray_a_id, tray_b_id, port_a_id, port_b_id);
            add_port_connection(
                port_type, board_a_ref, board_b_ref, host_a_id, tray_a_id, port_a_id, host_b_id, tray_b_id, port_b_id);
        }
    }

    // Recursively process subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        generate_connections_from_resolved_graph(subgraph);
    }
}

void CablingGenerator::populate_host_id_to_node() {
    host_id_to_node_.clear();
    if (root_instance_) {
        populate_host_id_from_resolved_graph(root_instance_);
    }
}

void CablingGenerator::populate_host_id_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph) {
    // Add direct nodes in this graph
    for (auto& [node_name, node] : graph->nodes) {
        host_id_to_node_[node.host_id] = &node;
    }

    // Recursively process subgraphs
    for (const auto& [subgraph_name, subgraph] : graph->subgraphs) {
        populate_host_id_from_resolved_graph(subgraph);
    }
}

void CablingGenerator::recreate_nodes_from_templates(ResolvedGraphInstance& graph) {
    // Recreate all nodes in this graph from templates
    // This resets port availability (template only has inter-board connection ports marked as used)
    for (auto& [node_name, node] : graph.nodes) {
        node = create_base_node_from_template(node, node_templates_);
    }

    // Recursively process subgraphs
    for (auto& [subgraph_name, subgraph] : graph.subgraphs) {
        recreate_nodes_from_templates(*subgraph);
    }
}

void CablingGenerator::get_all_connections_of_type(
    const std::vector<PortType>& port_types, std::vector<PortConnection>& conn_list) const {
    if (root_instance_) {
        get_all_connections_of_type_from_resolved_graph(root_instance_, port_types, conn_list);
    }
}

void CablingGenerator::get_all_connections_of_type_from_resolved_graph(
    const std::unique_ptr<ResolvedGraphInstance>& instance,
    const std::vector<PortType>& port_types,
    std::vector<PortConnection>& conn_list) const {
    for (auto port_type : port_types) {
        // Add internal connections at this graph level
        auto internal_connections = instance->internal_connections.find(port_type);
        if (internal_connections != instance->internal_connections.end()) {
            conn_list.insert(conn_list.end(), internal_connections->second.begin(), internal_connections->second.end());
        }

        // Add inter-board connections from direct nodes
        for (const auto& [child_name, child_instance] : instance->nodes) {
            auto inter_board_connections = child_instance.inter_board_connections.find(port_type);
            if (inter_board_connections == child_instance.inter_board_connections.end()) {
                continue;
            }
            conn_list.reserve(conn_list.size() + inter_board_connections->second.size());
            for (const auto& [start, end] : inter_board_connections->second) {
                PortEndpoint s_tuple = std::make_tuple(child_instance.host_id, start.first, start.second);
                PortEndpoint e_tuple = std::make_tuple(child_instance.host_id, end.first, end.second);
                conn_list.push_back(std::make_pair(s_tuple, e_tuple));
            }
        }
    }

    // Recursively process subgraphs
    for (const auto& [child_name, child_instance] : instance->subgraphs) {
        get_all_connections_of_type_from_resolved_graph(child_instance, port_types, conn_list);
    }
}

CableLength calc_cable_length(
    const Host& host1, int tray_id1, const Host& host2, int tray_id2, const std::string& node_type) {
    if (host1.hall != host2.hall || host1.aisle != host2.aisle) {
        return CableLength::UNKNOWN;
    }

    int tray_id_0 = tray_id1;
    int tray_id_1 = tray_id2;
    int rack_0 = host1.rack;
    int rack_1 = host2.rack;

    double tray_u_est_0 = host1.shelf_u;
    double tray_u_est_1 = host2.shelf_u;
    if (node_type.find("GALAXY") != std::string::npos) {
        // 1.25 U per tray, 1 U at bottom of 6U shelf, BH_GALAXY has 8U shelves
        tray_u_est_0 += (((4 - tray_id_0) * 1.25) + 1);
        tray_u_est_1 += (((4 - tray_id_1) * 1.25) + 1);
    }

    double standard_rack_w = 600.0;    // mm
    double standard_rack_u_h = 44.45;  // mm

    double rack_distance = std::abs(rack_0 - rack_1) * standard_rack_w;
    double u_distance = std::abs(tray_u_est_0 - tray_u_est_1) * standard_rack_u_h;

    double cable_length = std::sqrt((rack_distance * rack_distance) + (u_distance * u_distance)) + 150;  // 150mm slack

    if (cable_length <= 500.0) {
        return CableLength::CABLE_0P5;
    }
    if (cable_length <= 1000.0) {
        return CableLength::CABLE_1;
    }
    if (cable_length <= 2500.0) {
        return CableLength::CABLE_2P5;
    }
    if (cable_length <= 3000.0) {
        return CableLength::CABLE_3;
    }
    if (cable_length <= 5000.0) {
        return CableLength::CABLE_5;
    }
    return CableLength::UNKNOWN;
}

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelEndpoint& conn) {
    os << "PhysicalChannelEndpoint{hostname='" << conn.hostname << "', tray_id=" << *conn.tray_id
       << ", asic_channel=" << conn.asic_channel << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const PhysicalPortEndpoint& conn) {
    os << "PhysicalPortEndpoint{hostname='" << conn.hostname << "', aisle='" << conn.aisle << "', rack=" << conn.rack
       << ", shelf_u=" << conn.shelf_u << ", tray_id=" << *conn.tray_id
       << ", port_type=" << enchantum::to_string(conn.port_type) << ", port_id=" << *conn.port_id << "}";
    return os;
}

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::LogicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(conn.host_id, conn.tray_id, conn.asic_channel);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(conn.hostname, conn.tray_id, conn.asic_channel);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalPortEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalPortEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.aisle, conn.rack, conn.shelf_u, *conn.tray_id, conn.port_type, *conn.port_id);
    }
};

}  // namespace std
