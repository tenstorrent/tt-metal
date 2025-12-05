// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generate_mgd.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string_view>
#include <unistd.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <node/node_type_info.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

namespace {

// Information extracted from cabling descriptor
struct CablingDescriptorInfo {
    size_t num_hosts{};
    std::string node_type;
    std::set<uint32_t> host_ids;
};

// Helper function to recursively collect all host_ids from a GraphInstance
void collect_host_ids(const proto::GraphInstance& instance, std::set<uint32_t>& host_ids) {
    for (const auto& [child_name, child_mapping] : instance.child_mappings()) {
        if (child_mapping.has_host_id()) {
            // This is a leaf node with a host_id
            host_ids.insert(child_mapping.host_id());
        } else if (child_mapping.has_sub_instance()) {
            // This is a nested graph, recurse into it
            collect_host_ids(child_mapping.sub_instance(), host_ids);
        }
    }
}

// Helper function to find node type by traversing the graph template hierarchy
std::string find_node_type_from_template(
    const std::string& template_name, const proto::ClusterDescriptor& cluster_desc) {
    auto it = cluster_desc.graph_templates().find(template_name);
    if (it == cluster_desc.graph_templates().end()) {
        throw std::runtime_error("Graph template '" + template_name + "' not found in cabling descriptor");
    }

    const auto& graph_template = it->second;

    // Traverse children to find a node reference
    for (const auto& child : graph_template.children()) {
        if (child.has_node_ref()) {
            // Found a node reference - this is the node type
            return child.node_ref().node_descriptor();
        } else if (child.has_graph_ref()) {
            // Recurse into nested graph to find node type
            return find_node_type_from_template(child.graph_ref().graph_template(), cluster_desc);
        }
    }

    throw std::runtime_error("No node references found in graph template '" + template_name + "'");
}

std::vector<std::string> generate_hostnames(const size_t num_hosts) {
    std::vector<std::string> hostnames;
    hostnames.reserve(num_hosts);
    for (size_t i = 0; i < num_hosts; ++i) {
        hostnames.push_back("M" + std::to_string(i));
    }
    return hostnames;
}

std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> compute_intermesh_connections(
    const std::vector<std::string>& hostnames, const std::vector<LogicalChannelConnection>& chip_connections) {
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> intermesh_connections;

    for (const auto& connection : chip_connections) {
        if (connection.first.host_id == connection.second.host_id) {
            continue;
        }
        intermesh_connections[hostnames[*connection.first.host_id]][hostnames[*connection.second.host_id]]++;
    }

    return intermesh_connections;
}

// Helper to extract cabling info from ClusterDescriptor protobuf
CablingDescriptorInfo get_cabling_info_from_proto(const proto::ClusterDescriptor& cluster_desc) {
    CablingDescriptorInfo info;

    // Collect all host_ids from the root instance
    if (cluster_desc.has_root_instance()) {
        collect_host_ids(cluster_desc.root_instance(), info.host_ids);
    }

    if (info.host_ids.empty()) {
        throw std::runtime_error("No host_ids found in cabling descriptor");
    }

    // Validate host IDs are contiguous and start from 0
    uint32_t expected_max = info.host_ids.size() - 1;
    uint32_t actual_max = *info.host_ids.rbegin();
    if (actual_max != expected_max) {
        throw std::runtime_error(
            "Host IDs must be contiguous starting from 0. Found " + std::to_string(info.host_ids.size()) +
            " hosts but max host_id is " + std::to_string(actual_max) + " (expected " + std::to_string(expected_max) +
            ")");
    }

    info.num_hosts = info.host_ids.size();

    // Extract node type from the root instance's template
    if (cluster_desc.has_root_instance()) {
        info.node_type = find_node_type_from_template(cluster_desc.root_instance().template_name(), cluster_desc);
    }

    return info;
}

}  // anonymous namespace

static tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_impl(
    const proto::ClusterDescriptor& cluster_desc, bool verbose) {
    // Extract cluster information from protobuf
    auto cabling_info = get_cabling_info_from_proto(cluster_desc);

    if (verbose) {
        std::cout << "Found " << cabling_info.num_hosts << " host(s) with node type: " << cabling_info.node_type
                  << '\n';
        std::cout << "Host IDs: ";
        for (const auto id : cabling_info.host_ids) {
            std::cout << id << ' ';
        }
        std::cout << '\n';
    }

    // Generate hostnames based on the number of hosts
    std::vector<std::string> hostnames = generate_hostnames(cabling_info.num_hosts);

    // Create the cabling generator to get chip connections
    auto cabling_generator = CablingGenerator(cluster_desc, hostnames);
    const auto& chip_connections = cabling_generator.get_chip_connections();

    if (verbose) {
        std::cout << "Node type: " << cabling_info.node_type << '\n';
    }

    const auto intermesh_connections = compute_intermesh_connections(hostnames, chip_connections);

    const size_t total_connections = [&intermesh_connections]() {
        size_t count = 0;
        for (const auto& [_, conn] : intermesh_connections) {
            count += conn.size();
        }
        return count;
    }();

    if (verbose) {
        std::cout << "Computed " << total_connections << " inter-mesh connection(s)\n";
        for (const auto& [src_host, conn] : intermesh_connections) {
            for (const auto& [dst_host, count] : conn) {
                std::cout << "  " << src_host << " <-> " << dst_host << ": " << count << " channel(s)\n";
            }
        }
    }

    tt::tt_fabric::proto::MeshGraphDescriptor mgd;

    // Get node type info from shared lookup (throws if unknown)
    const auto& node_info = get_node_type_info(cabling_info.node_type);
    const auto& device_dims = node_info.device_dims;
    const auto& arch_str = node_info.architecture;
    const auto channel_count = node_info.channel_count;

    // Convert architecture string to proto enum
    tt::tt_fabric::proto::Architecture arch;
    if (!tt::tt_fabric::proto::Architecture_Parse(arch_str, &arch)) {
        throw std::runtime_error("Unknown architecture: " + arch_str);
    }

    if (verbose) {
        std::cout << "Architecture: " << arch_str << '\n';
        std::cout << "Device topology: " << device_dims[0] << "x" << device_dims[1] << '\n';
        std::cout << "Channels per connection: " << channel_count << '\n';
    }

    // Create a single mesh descriptor that will be reused for all instances
    auto* mesh_desc = mgd.add_mesh_descriptors();
    mesh_desc->set_name("Mesh");
    mesh_desc->set_arch(arch);

    // Set device topology
    auto* device_topo = mesh_desc->mutable_device_topology();
    for (int dim : device_dims) {
        device_topo->add_dims(dim);
    }

    // Set host topology
    auto* host_topo = mesh_desc->mutable_host_topology();
    host_topo->add_dims(1);
    host_topo->add_dims(1);

    // Set channels using the value from the lookup table
    auto* channels = mesh_desc->mutable_channels();
    channels->set_count(channel_count);
    channels->set_policy(tt::tt_fabric::proto::Policy::STRICT);

    // Create graph descriptor
    auto* graph_desc = mgd.add_graph_descriptors();
    graph_desc->set_name("G0");
    graph_desc->set_type("FABRIC");

    // Add mesh instances (all using the same mesh descriptor)
    for (size_t i = 0; i < cabling_info.num_hosts; ++i) {
        auto* instance = graph_desc->add_instances();
        auto* mesh_ref = instance->mutable_mesh();
        mesh_ref->set_mesh_descriptor("Mesh");
        mesh_ref->set_mesh_id(i);
    }

    // Add connections between meshes
    for (const auto& [src_host, conn] : intermesh_connections) {
        for (const auto& [dst_host, count] : conn) {
            // Parse mesh ID from hostname (format: "M{id}")
            // Since hostnames are generated as "M0", "M1", etc., we can extract the ID
            int src_idx = std::stoi(src_host.substr(1));
            int dst_idx = std::stoi(dst_host.substr(1));

            auto* connection = graph_desc->add_connections();

            // Add source node
            auto* src_node = connection->add_nodes();
            auto* src_mesh_ref = src_node->mutable_mesh();
            src_mesh_ref->set_mesh_descriptor("Mesh");
            src_mesh_ref->set_mesh_id(src_idx);

            // Add destination node
            auto* dst_node = connection->add_nodes();
            auto* dst_mesh_ref = dst_node->mutable_mesh();
            dst_mesh_ref->set_mesh_descriptor("Mesh");
            dst_mesh_ref->set_mesh_id(dst_idx);

            // Set channels (no policy for cleaner output)
            auto* conn_channels = connection->mutable_channels();
            conn_channels->set_count(count);
        }
    }

    // Set top-level instance
    auto* top_level = mgd.mutable_top_level_instance();
    auto* graph_ref = top_level->mutable_graph();
    graph_ref->set_graph_descriptor("G0");
    graph_ref->set_graph_id(0);

    if (verbose) {
        std::cout << "Mesh Graph Descriptor successfully generated\n";
        std::cout << "  Meshes: " << cabling_info.num_hosts << '\n';
        std::cout << "  Connections: " << total_connections << '\n';
    }

    return mgd;
}

tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const proto::ClusterDescriptor& cluster_desc, bool verbose) {
    return generate_mgd_impl(cluster_desc, verbose);
}

tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const std::filesystem::path& cabling_descriptor_path, const bool verbose) {
    if (verbose) {
        std::cout << "Reading cabling descriptor: " << cabling_descriptor_path << '\n';
    }

    std::ifstream file(cabling_descriptor_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open cabling descriptor file: " + cabling_descriptor_path.string());
    }
    const std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    proto::ClusterDescriptor cluster_desc;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &cluster_desc)) {
        throw std::runtime_error("Failed to parse cabling descriptor protobuf: " + cabling_descriptor_path.string());
    }

    return generate_mgd_from_cabling(cluster_desc, verbose);
}

void write_mgd_to_file(const tt::tt_fabric::proto::MeshGraphDescriptor& mgd, const std::filesystem::path& output_path) {
    if (mgd.mesh_descriptors_size() == 0 || mgd.graph_descriptors_size() == 0) {
        throw std::runtime_error("Invalid MGD: missing mesh or graph descriptors");
    }

    const auto& mesh_desc = mgd.mesh_descriptors(0);
    const auto& device_topo = mesh_desc.device_topology();
    const auto& host_topo = mesh_desc.host_topology();
    const auto& channels = mesh_desc.channels();
    const auto& graph_desc = mgd.graph_descriptors(0);
    const auto& top_level = mgd.top_level_instance();

    std::ofstream mgd_file(output_path);
    if (!mgd_file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + output_path.string());
    }

    // Write Meshes section
    mgd_file << "# --- Meshes ---------------------------------------------------------------\n\n";
    mgd_file << "mesh_descriptors {\n";
    mgd_file << "  name: \"" << mesh_desc.name() << "\"\n";
    mgd_file << "  arch: " << tt::tt_fabric::proto::Architecture_Name(mesh_desc.arch()) << "\n";
    mgd_file << "  device_topology { dims: [ ";
    for (int i = 0; i < device_topo.dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << device_topo.dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  host_topology   { dims: [ ";
    for (int i = 0; i < host_topo.dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << host_topo.dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  channels {\n";
    mgd_file << "    count: " << channels.count() << "\n";
    mgd_file << "    policy: " << tt::tt_fabric::proto::Policy_Name(channels.policy()) << "\n";
    mgd_file << "  }\n";
    mgd_file << "}\n\n";

    // Write Graphs section
    mgd_file << "# --- Graphs ---------------------------------------------------------------\n\n";
    mgd_file << "graph_descriptors {\n";
    mgd_file << "  name: \"" << graph_desc.name() << "\"\n";
    mgd_file << "  type: \"" << graph_desc.type() << "\"\n";

    // Write instances comment
    size_t num_instances = graph_desc.instances_size();
    mgd_file << "  # Instances: mesh ids 0";
    for (size_t i = 1; i < num_instances; ++i) {
        mgd_file << "," << i;
    }
    mgd_file << " (all " << device_topo.dims(0) << "x" << device_topo.dims(1) << ")\n";

    // Write instances
    for (int i = 0; i < graph_desc.instances_size(); ++i) {
        const auto& instance = graph_desc.instances(i);
        mgd_file << "  instances { mesh { mesh_descriptor: \"" << instance.mesh().mesh_descriptor()
                 << "\" mesh_id: " << instance.mesh().mesh_id() << " } }\n";
    }

    // Collect and sort connections for ordered output
    struct ConnectionInfo {
        int src_id;
        int dst_id;
        int channel_count;
    };
    std::vector<ConnectionInfo> sorted_connections;

    for (int i = 0; i < graph_desc.connections_size(); ++i) {
        const auto& connection = graph_desc.connections(i);
        if (connection.nodes_size() == 2) {
            int id1 = connection.nodes(0).mesh().mesh_id();
            int id2 = connection.nodes(1).mesh().mesh_id();
            sorted_connections.push_back({std::min(id1, id2), std::max(id1, id2), connection.channels().count()});
        }
    }

    // Sort connections by source id first, then by destination id
    std::sort(
        sorted_connections.begin(), sorted_connections.end(), [](const ConnectionInfo& a, const ConnectionInfo& b) {
            return (a.src_id != b.src_id) ? (a.src_id < b.src_id) : (a.dst_id < b.dst_id);
        });

    // Write sorted connections
    mgd_file << "\n";
    for (const auto& conn : sorted_connections) {
        mgd_file << "  # M" << conn.src_id << " <-> M" << conn.dst_id << "\n";
        mgd_file << "  connections {\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"" << mesh_desc.name() << "\" mesh_id: " << conn.src_id
                 << " } }\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"" << mesh_desc.name() << "\" mesh_id: " << conn.dst_id
                 << " } }\n";
        mgd_file << "    channels { count: " << conn.channel_count << " }\n";
        mgd_file << "  }\n";
    }

    mgd_file << "}\n\n";

    // Write Instantiation section
    mgd_file << "# --- Instantiation ----------------------------------------------------------\n";
    mgd_file << "top_level_instance { graph { graph_descriptor: \"" << top_level.graph().graph_descriptor()
             << "\" graph_id: " << top_level.graph().graph_id() << " } }\n";

    mgd_file.close();
}

void generate_mesh_graph_descriptor(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    const bool verbose) {
    const auto mgd = generate_mgd_from_cabling(cabling_descriptor_path, verbose);
    write_mgd_to_file(mgd, output_path);

    if (verbose) {
        std::cout << "MGD written to: " << output_path << '\n';
    }
}

}  // namespace tt::scaleout_tools
