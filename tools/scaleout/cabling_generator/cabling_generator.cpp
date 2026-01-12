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
#include <google/protobuf/text_format.h>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>

// Add protobuf includes
#include "protobuf/cluster_config.pb.h"
#include "protobuf/deployment.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

namespace {

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
        throw std::runtime_error(
            std::string(enchantum::to_string(port_type)) + " Port " + std::to_string(*port_a_id) +
            " not available on board " + std::to_string(*board_a_id) + " in host " + std::to_string(*host_a_id));
    }
    if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
        throw std::runtime_error(
            std::string(enchantum::to_string(port_type)) + " Port " + std::to_string(*port_b_id) +
            " not available on board " + std::to_string(*board_b_id) + " in host " + std::to_string(*host_b_id));
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
    if (index == path.size() - 1) {
        // Direct node reference - look up in child_mappings
        const std::string& node_name = path[index];
        const auto& child_mapping = graph_instance.child_mappings().at(node_name);

        if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
            return HostId(child_mapping.host_id());
        }
        throw std::runtime_error("Node " + node_name + " is not a leaf node");
    }
    // Multi-level path - descend into subgraph
    const std::string& subgraph_name = path[index];
    const auto& child_mapping = graph_instance.child_mappings().at(subgraph_name);

    if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
        return resolve_path_from_proto(path, child_mapping.sub_instance(), cluster_descriptor, index + 1);
    }
    throw std::runtime_error("Subgraph " + subgraph_name + " is not a graph instance");
}

// Builds a resolved graph instance from a graph instance and deployment descriptor.
// deployment_descriptor is optional - if nullptr, no validation is performed.
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
            resolved->internal_connections[*port_type].emplace_back(
                std::make_tuple(host_a_id, board_a_id, port_a_id), std::make_tuple(host_b_id, board_b_id, port_b_id));
        }
    }

    return resolved;
}

void populate_deployment_hosts(
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::unordered_map<std::string, Node>& node_templates,
    std::vector<Host>& deployment_hosts) {
    // Store deployment hosts
    deployment_hosts.reserve(deployment_descriptor.hosts().size());
    for (const auto& proto_host : deployment_descriptor.hosts()) {
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

std::unique_ptr<ResolvedGraphInstance> build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates) {
    return build_graph_instance_impl(
        graph_instance, cluster_descriptor, &deployment_descriptor, instance_name, node_templates);
}

std::unique_ptr<ResolvedGraphInstance> build_graph_instance(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const std::string& instance_name,
    std::unordered_map<std::string, Node>& node_templates) {
    return build_graph_instance_impl(graph_instance, cluster_descriptor, nullptr, instance_name, node_templates);
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

// Constructor with full deployment descriptor
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path) {
    // Load descriptors from file paths
    auto cluster_descriptor =
        load_descriptor_from_textproto<tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor>(
            cluster_descriptor_path);
    auto deployment_descriptor =
        load_descriptor_from_textproto<tt::scaleout_tools::deployment::proto::DeploymentDescriptor>(
            deployment_descriptor_path);

    initialize_cluster(cluster_descriptor, deployment_descriptor);

    // Populate deployment hosts
    populate_deployment_hosts(deployment_descriptor, node_templates_, deployment_hosts_);
}

// Constructor with just hostnames (no physical location info) - wrapper around protobuf constructor
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::vector<std::string>& hostnames) :
    CablingGenerator(
        load_descriptor_from_textproto<tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor>(
            cluster_descriptor_path),
        hostnames) {}

// Constructor with ClusterDescriptor protobuf and hostnames (no file I/O required)
CablingGenerator::CablingGenerator(
    const cabling_generator::proto::ClusterDescriptor& cluster_descriptor, const std::vector<std::string>& hostnames) {
    initialize_cluster(cluster_descriptor);

    // Populate deployment hosts from hostnames
    populate_deployment_hosts_from_hostnames(hostnames, host_id_to_node_, deployment_hosts_);
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

    CablingGenerator::get_all_connections_of_type(root_instance_, {PortType::QSFP_DD}, conn_list);
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

        if (host_to_node_path.contains(host_id)) {
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

void CablingGenerator::get_all_connections_of_type(
    const std::unique_ptr<ResolvedGraphInstance>& instance,
    const std::vector<PortType>& port_types,
    std::vector<std::pair<std::tuple<HostId, TrayId, PortId>, std::tuple<HostId, TrayId, PortId>>>& conn_list) const {
    for (auto port_type : port_types) {
        auto internal_connections = instance->internal_connections.find(port_type);
        if (internal_connections == instance->internal_connections.end()) {
            continue;
        }
        conn_list.insert(conn_list.end(), internal_connections->second.begin(), internal_connections->second.end());

        for (const auto& [child_name, child_instance] : instance->nodes) {
            auto inter_board_connections = child_instance.inter_board_connections.find(port_type);
            if (inter_board_connections == child_instance.inter_board_connections.end()) {
                continue;
            }
            conn_list.reserve(conn_list.size() + inter_board_connections->second.size());
            for (const auto& [start, end] : inter_board_connections->second) {
                std::tuple<HostId, TrayId, PortId> s_tuple =
                    std::make_tuple(child_instance.host_id, start.first, start.second);

                std::tuple<HostId, TrayId, PortId> e_tuple =
                    std::make_tuple(child_instance.host_id, end.first, end.second);
                conn_list.push_back(std::make_pair(s_tuple, e_tuple));
            }
        }
    }
    for (const auto& [child_name, child_instance] : instance->subgraphs) {
        get_all_connections_of_type(child_instance, port_types, conn_list);
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
