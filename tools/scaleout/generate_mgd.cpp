// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generate_mgd.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace tt::scaleout_tools {

// Create lookup table for node types to shapes, architectures, and channel counts
std::unordered_map<std::string, NodeTypeInfo> create_node_type_lookup() {
    return {
        // Wormhole architectures
        {"N300_LB_DEFAULT", {{2, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 2}},     // N300: 2 channels
        {"N300_QB_DEFAULT", {{2, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 2}},     // N300: 2 channels
        {"WH_GALAXY", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},           // WH Galaxy: 4 channels
        {"WH_GALAXY_X_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},   // WH Galaxy: 4 channels
        {"WH_GALAXY_Y_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},   // WH Galaxy: 4 channels
        {"WH_GALAXY_XY_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},  // WH Galaxy: 4 channels

        // Blackhole architectures
        {"P150_LB", {{2, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 4}},             // P150: 4 channels
        {"P150_QB_AE_DEFAULT", {{2, 2}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 4}},  // P150: 4 channels
        {"P300_QB_GE", {{2, 2}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},          // P300: 2 channels
        {"BH_GALAXY", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},           // BH Galaxy: 2 channels
        {"BH_GALAXY_X_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},   // BH Galaxy: 2 channels
        {"BH_GALAXY_Y_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},   // BH Galaxy: 2 channels
        {"BH_GALAXY_XY_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},  // BH Galaxy: 2 channels
    };
}

std::vector<std::string> generate_hostnames(size_t num_hosts) {
    std::vector<std::string> hostnames;
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

void generate_mesh_graph_descriptor(const std::string& cabling_descriptor_path, const std::string& output_path) {
    // Validate input file exists
    std::ifstream test_file(cabling_descriptor_path);
    if (!test_file.is_open()) {
        throw std::runtime_error("Cannot open cabling descriptor file: " + cabling_descriptor_path);
    }
    test_file.close();

    std::cout << "Reading cabling descriptor: " << cabling_descriptor_path << std::endl;

    // First pass: determine number of hosts by parsing the cabling descriptor
    // We'll use a temporary hostname list to figure out the actual number of hosts needed
    std::vector<std::string> temp_hostnames;
    for (int i = 0; i < 256; ++i) {  // Max reasonable cluster size
        temp_hostnames.push_back("M" + std::to_string(i));
    }

    auto temp_cabling = CablingGenerator(cabling_descriptor_path, temp_hostnames);
    auto hosts = temp_cabling.get_deployment_hosts();

    // Now generate the actual hostnames based on the number of hosts discovered
    std::vector<std::string> hostnames = generate_hostnames(hosts.size());

    // Create the final cabling descriptor with the correct number of hostnames
    auto cabling_descriptor = CablingGenerator(cabling_descriptor_path, hostnames);
    hosts = cabling_descriptor.get_deployment_hosts();
    auto chip_connections = cabling_descriptor.get_chip_connections();

    // Log cluster configuration
    std::cout << "Discovered " << hosts.size() << " host(s) in cabling descriptor" << std::endl;
    for (const auto& host : hosts) {
        std::cout << "  " << host.hostname << ": " << host.node_type << std::endl;
    }

    // Compute intermesh connections
    auto intermesh_connections = compute_intermesh_connections(hostnames, chip_connections);

    // Count total connections
    size_t total_connections = 0;
    for (const auto& [src_host, conn] : intermesh_connections) {
        for (const auto& [dst_host, count] : conn) {
            total_connections++;
        }
    }
    std::cout << "Computed " << total_connections << " inter-mesh connection(s)" << std::endl;

    // Create node type lookup
    auto node_type_lookup = create_node_type_lookup();

    // Generate Mesh Graph Descriptor using protobuf
    tt::tt_fabric::proto::MeshGraphDescriptor mgd;

    // Determine common architecture, device topology, and channel count from first host
    if (hosts.empty()) {
        throw std::runtime_error("No hosts found in cabling descriptor");
    }

    const auto& first_host = hosts[0];
    auto it = node_type_lookup.find(first_host.node_type);
    if (it == node_type_lookup.end()) {
        throw std::runtime_error(
            "Unknown node type '" + first_host.node_type + "'. " +
            "Please add this node type to the node_type_lookup table in generate_mgd.cpp");
    }

    std::vector<int> device_dims = it->second.device_dims;
    tt::tt_fabric::proto::Architecture arch = it->second.arch;
    int channel_count = it->second.channel_count;

    std::cout << "Using architecture: " << tt::tt_fabric::proto::Architecture_Name(arch) << std::endl;
    std::cout << "Device topology: " << device_dims[0] << "x" << device_dims[1] << std::endl;
    std::cout << "Channel count: " << channel_count << std::endl;

    // Create a single mesh descriptor that will be reused for all instances
    auto* mesh_desc = mgd.add_mesh_descriptors();
    mesh_desc->set_name("M0");  // Use M0 like in the example
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
    for (size_t i = 0; i < hosts.size(); ++i) {
        auto* instance = graph_desc->add_instances();
        auto* mesh_ref = instance->mutable_mesh();
        mesh_ref->set_mesh_descriptor("M0");
        mesh_ref->set_mesh_id(i);
    }

    // Add connections between meshes
    for (const auto& [src_host, conn] : intermesh_connections) {
        for (const auto& [dst_host, count] : conn) {
            // Find host indices
            int src_idx = -1, dst_idx = -1;
            for (size_t i = 0; i < hosts.size(); ++i) {
                if (hosts[i].hostname == src_host) {
                    src_idx = i;
                }
                if (hosts[i].hostname == dst_host) {
                    dst_idx = i;
                }
            }

            if (src_idx >= 0 && dst_idx >= 0) {
                auto* connection = graph_desc->add_connections();

                // Add source node
                auto* src_node = connection->add_nodes();
                auto* src_mesh_ref = src_node->mutable_mesh();
                src_mesh_ref->set_mesh_descriptor("M0");
                src_mesh_ref->set_mesh_id(src_idx);

                // Add destination node
                auto* dst_node = connection->add_nodes();
                auto* dst_mesh_ref = dst_node->mutable_mesh();
                dst_mesh_ref->set_mesh_descriptor("M0");
                dst_mesh_ref->set_mesh_id(dst_idx);

                // Set channels (no policy for cleaner output)
                auto* conn_channels = connection->mutable_channels();
                conn_channels->set_count(count);
            }
        }
    }

    // Set top-level instance
    auto* top_level = mgd.mutable_top_level_instance();
    auto* graph_ref = top_level->mutable_graph();
    graph_ref->set_graph_descriptor("G0");
    graph_ref->set_graph_id(0);

    // Write to file with custom formatting to match the expected style
    std::ofstream mgd_file(output_path);
    if (!mgd_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << output_path << std::endl;
        throw std::runtime_error("Failed to open MGD output file");
    }

    // Write Meshes section
    mgd_file << "# --- Meshes ---------------------------------------------------------------\n\n";
    mgd_file << "mesh_descriptors {\n";
    mgd_file << "  name: \"" << mesh_desc->name() << "\"\n";
    mgd_file << "  arch: " << tt::tt_fabric::proto::Architecture_Name(mesh_desc->arch()) << "\n";
    mgd_file << "  device_topology { dims: [ ";
    for (int i = 0; i < device_topo->dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << device_topo->dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  host_topology   { dims: [ ";
    for (int i = 0; i < host_topo->dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << host_topo->dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  channels {\n";
    mgd_file << "    count: " << channels->count() << "\n";
    mgd_file << "    policy: " << tt::tt_fabric::proto::Policy_Name(channels->policy()) << "\n";
    mgd_file << "  }\n";
    mgd_file << "}\n\n";

    // Write Graphs section
    mgd_file << "# --- Graphs ---------------------------------------------------------------\n\n";
    mgd_file << "graph_descriptors {\n";
    mgd_file << "  name: \"" << graph_desc->name() << "\"\n";
    mgd_file << "  type: \"" << graph_desc->type() << "\"\n";

    // Write instances in compact format
    mgd_file << "  # Instances: mesh ids 0";
    for (size_t i = 1; i < hosts.size(); ++i) {
        mgd_file << "," << i;
    }
    mgd_file << " (all " << device_dims[0] << "x" << device_dims[1] << ")\n";

    for (int i = 0; i < graph_desc->instances_size(); ++i) {
        const auto& instance = graph_desc->instances(i);
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

    for (int i = 0; i < graph_desc->connections_size(); ++i) {
        const auto& connection = graph_desc->connections(i);
        if (connection.nodes_size() == 2) {
            int id1 = connection.nodes(0).mesh().mesh_id();
            int id2 = connection.nodes(1).mesh().mesh_id();
            // Store with smaller id first for consistent ordering
            sorted_connections.push_back({std::min(id1, id2), std::max(id1, id2), connection.channels().count()});
        }
    }

    // Sort connections by source id first, then by destination id
    std::sort(
        sorted_connections.begin(), sorted_connections.end(), [](const ConnectionInfo& a, const ConnectionInfo& b) {
            if (a.src_id != b.src_id) {
                return a.src_id < b.src_id;
            }
            return a.dst_id < b.dst_id;
        });

    // Write sorted connections
    mgd_file << "\n";
    for (const auto& conn : sorted_connections) {
        mgd_file << "  # M" << conn.src_id << " <-> M" << conn.dst_id << "\n";
        mgd_file << "  connections {\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"M0\" mesh_id: " << conn.src_id << " } }\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"M0\" mesh_id: " << conn.dst_id << " } }\n";
        mgd_file << "    channels { count: " << conn.channel_count << " }\n";
        mgd_file << "  }\n";
    }

    mgd_file << "}\n\n";

    // Write Instantiation section
    mgd_file << "# --- Instantiation ----------------------------------------------------------\n";
    mgd_file << "top_level_instance { graph { graph_descriptor: \"" << top_level->graph().graph_descriptor()
             << "\" graph_id: " << top_level->graph().graph_id() << " } }\n";

    mgd_file.close();

    std::cout << "\n✓ Mesh Graph Descriptor successfully generated" << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  Meshes: " << hosts.size() << std::endl;
    std::cout << "  Connections: " << total_connections << std::endl;
}

}  // namespace tt::scaleout_tools
