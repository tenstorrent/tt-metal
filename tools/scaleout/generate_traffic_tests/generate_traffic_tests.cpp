// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generate_traffic_tests.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include <google/protobuf/text_format.h>
#include <yaml-cpp/yaml.h>

#include <generate_mgd/generate_mgd.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

namespace {

// =============================================================================
// Node Type Information (reuse logic from generate_mgd)
// =============================================================================

struct NodeTypeInfo {
    std::vector<int> device_dims;
    std::string arch;
};

const std::unordered_map<std::string, NodeTypeInfo>& get_node_type_info() {
    static const std::unordered_map<std::string, NodeTypeInfo> lookup = {
        // Wormhole architectures
        {"N300_LB_DEFAULT", {{2, 4}, "WORMHOLE_B0"}},
        {"N300_QB_DEFAULT", {{2, 4}, "WORMHOLE_B0"}},
        {"WH_GALAXY", {{8, 4}, "WORMHOLE_B0"}},
        {"WH_GALAXY_X_TORUS", {{8, 4}, "WORMHOLE_B0"}},
        {"WH_GALAXY_Y_TORUS", {{8, 4}, "WORMHOLE_B0"}},
        {"WH_GALAXY_XY_TORUS", {{8, 4}, "WORMHOLE_B0"}},

        // Blackhole architectures
        {"P150_LB", {{2, 4}, "BLACKHOLE"}},
        {"P150_QB_AE_DEFAULT", {{2, 2}, "BLACKHOLE"}},
        {"P300_QB_GE", {{2, 2}, "BLACKHOLE"}},
        {"BH_GALAXY", {{8, 4}, "BLACKHOLE"}},
        {"BH_GALAXY_X_TORUS", {{8, 4}, "BLACKHOLE"}},
        {"BH_GALAXY_Y_TORUS", {{8, 4}, "BLACKHOLE"}},
        {"BH_GALAXY_XY_TORUS", {{8, 4}, "BLACKHOLE"}},
    };
    return lookup;
}

// Recursively collect all host_ids from a GraphInstance
void collect_host_ids(const proto::GraphInstance& instance, std::set<uint32_t>& host_ids) {
    for (const auto& [child_name, child_mapping] : instance.child_mappings()) {
        if (child_mapping.has_host_id()) {
            host_ids.insert(child_mapping.host_id());
        } else if (child_mapping.has_sub_instance()) {
            collect_host_ids(child_mapping.sub_instance(), host_ids);
        }
    }
}

// Find node type by traversing the graph template hierarchy
std::string find_node_type_from_template(
    const std::string& template_name, const proto::ClusterDescriptor& cluster_desc) {
    auto it = cluster_desc.graph_templates().find(template_name);
    if (it == cluster_desc.graph_templates().end()) {
        throw std::runtime_error("Graph template '" + template_name + "' not found in cabling descriptor");
    }

    const auto& graph_template = it->second;

    for (const auto& child : graph_template.children()) {
        if (child.has_node_ref()) {
            return child.node_ref().node_descriptor();
        } else if (child.has_graph_ref()) {
            return find_node_type_from_template(child.graph_ref().graph_template(), cluster_desc);
        }
    }

    throw std::runtime_error("No node references found in graph template '" + template_name + "'");
}

// Generate hostnames for mesh identification
std::vector<std::string> generate_hostnames(const size_t num_hosts) {
    std::vector<std::string> hostnames;
    hostnames.reserve(num_hosts);
    for (size_t i = 0; i < num_hosts; ++i) {
        hostnames.push_back("M" + std::to_string(i));
    }
    return hostnames;
}

// =============================================================================
// YAML Generation Helpers
// =============================================================================

void emit_fabric_setup(YAML::Emitter& out, const std::filesystem::path& mgd_path) {
    out << YAML::Key << "fabric_setup";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "topology" << YAML::Value << "Mesh";
    if (!mgd_path.empty()) {
        out << YAML::Key << "mesh_descriptor_path" << YAML::Value << mgd_path.string();
    }
    out << YAML::EndMap;
}

void emit_defaults(
    YAML::Emitter& out,
    const std::string& ftype,
    const std::string& ntype,
    uint32_t size,
    uint32_t num_packets) {
    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << ftype;
    out << YAML::Key << "ntype" << YAML::Value << ntype;
    out << YAML::Key << "size" << YAML::Value << size;
    out << YAML::Key << "num_packets" << YAML::Value << num_packets;
    out << YAML::EndMap;
}

void emit_device_coord(YAML::Emitter& out, uint32_t mesh_id, int row, int col) {
    out << YAML::Flow << YAML::BeginSeq << mesh_id << YAML::BeginSeq << row << col << YAML::EndSeq << YAML::EndSeq;
}

void emit_parametrization(YAML::Emitter& out, NocSendTypeSet noc_types, const std::vector<uint32_t>& sizes) {
    out << YAML::Key << "parametrization_params";
    out << YAML::Value << YAML::BeginMap;

    // NoC types based on configuration
    out << YAML::Key << "ntype";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq;
    switch (noc_types) {
        case NocSendTypeSet::BASIC: out << "unicast_write"; break;
        case NocSendTypeSet::STANDARD:
            out << "unicast_write" << "atomic_inc" << "fused_atomic_inc";
            break;
        case NocSendTypeSet::FULL:
            out << "unicast_write" << "atomic_inc" << "fused_atomic_inc" << "unicast_scatter_write";
            break;
    }
    out << YAML::EndSeq;

    // Packet sizes
    out << YAML::Key << "size";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (const auto& s : sizes) {
        out << s;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

void emit_high_level_pattern(YAML::Emitter& out, const std::string& pattern_type, std::optional<uint32_t> iterations) {
    out << YAML::Key << "patterns";
    out << YAML::Value << YAML::BeginSeq;
    out << YAML::BeginMap;
    out << YAML::Key << "type" << YAML::Value << pattern_type;
    if (iterations.has_value()) {
        out << YAML::Key << "iterations" << YAML::Value << iterations.value();
    }
    out << YAML::EndMap;
    out << YAML::EndSeq;
}

void emit_skip_platforms(YAML::Emitter& out, const std::vector<std::string>& platforms) {
    if (!platforms.empty()) {
        out << YAML::Key << "skip";
        out << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (const auto& p : platforms) {
            out << p;
        }
        out << YAML::EndSeq;
    }
}

// =============================================================================
// Test Generation Functions (ordered from easiest to hardest)
// =============================================================================

// 1. Simple intra-mesh unicast (easiest)
void generate_simple_unicast_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "SimpleUnicast");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_defaults(out, "unicast", "unicast_write", 1024, 100);

    // Single sender within first mesh
    out << YAML::Key << "senders";
    out << YAML::Value << YAML::BeginSeq;
    out << YAML::BeginMap;
    out << YAML::Key << "device";
    emit_device_coord(out, 0, 0, 0);
    out << YAML::Key << "patterns";
    out << YAML::Value << YAML::BeginSeq;
    out << YAML::BeginMap;
    out << YAML::Key << "destination";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "device";
    emit_device_coord(out, 0, 0, 1);
    out << YAML::EndMap;
    out << YAML::EndMap;
    out << YAML::EndSeq;
    out << YAML::EndMap;
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

// 2. Inter-mesh unicast (tests mesh-to-mesh connectivity)
void generate_inter_mesh_unicast_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config) {
    if (topology.connected_pairs.empty()) {
        return;  // No inter-mesh connections
    }

    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "InterMeshUnicast");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_defaults(out, "unicast", "unicast_write", 2048, 100);

    // Create sender for each connected mesh pair
    out << YAML::Key << "senders";
    out << YAML::Value << YAML::BeginSeq;

    for (const auto& [src_mesh, dst_mesh] : topology.connected_pairs) {
        out << YAML::BeginMap;
        out << YAML::Key << "device";
        emit_device_coord(out, src_mesh, 0, 0);
        out << YAML::Key << "patterns";
        out << YAML::Value << YAML::BeginSeq;
        out << YAML::BeginMap;
        out << YAML::Key << "destination";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "device";
        emit_device_coord(out, dst_mesh, 0, 0);
        out << YAML::EndMap;
        out << YAML::EndMap;
        out << YAML::EndSeq;
        out << YAML::EndMap;
    }

    out << YAML::EndSeq;
    out << YAML::EndMap;
}

// 3. All-to-all unicast with parametrization
void generate_all_to_all_unicast_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "AllToAllUnicast");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    // Get packet sizes based on config
    std::vector<uint32_t> sizes = config.packet_sizes.value_or(std::vector<uint32_t>{1024, 2048, 4096});
    emit_parametrization(out, config.noc_types, sizes);

    uint32_t num_packets = config.packet_counts.has_value() ? config.packet_counts.value()[0] : 100;
    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << num_packets;
    out << YAML::EndMap;

    emit_high_level_pattern(out, "all_to_all", std::nullopt);

    out << YAML::EndMap;
}

// 4. Random pairing test
void generate_random_pairing_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "RandomPairing");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    uint32_t num_packets = config.packet_counts.has_value() ? config.packet_counts.value()[0] : 100;
    emit_defaults(out, "unicast", "unicast_write", 2048, num_packets);

    uint32_t iterations = config.profile == TestProfile::COVERAGE ? 10 : 3;
    emit_high_level_pattern(out, "full_device_random_pairing", iterations);

    out << YAML::EndMap;
}

// 5. All-to-one convergence test (harder - many senders to single receiver)
void generate_all_to_one_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "AllToOne");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    std::vector<uint32_t> sizes = config.packet_sizes.value_or(std::vector<uint32_t>{1024, 2048});
    emit_parametrization(out, NocSendTypeSet::BASIC, sizes);

    uint32_t num_packets = config.packet_counts.has_value() ? config.packet_counts.value()[0] : 100;
    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << num_packets;
    out << YAML::EndMap;

    emit_high_level_pattern(out, "all_to_one", std::nullopt);

    out << YAML::EndMap;
}

// 6. Flow control test (stress - high packet counts)
void generate_flow_control_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "FlowControlStress");
    out << YAML::Key << "enable_flow_control" << YAML::Value << true;
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    std::vector<uint32_t> sizes = config.packet_sizes.value_or(std::vector<uint32_t>{1024, 2048, 4096});
    emit_parametrization(out, config.noc_types, sizes);

    // High packet count for stress testing
    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << 10000;
    out << YAML::EndMap;

    emit_high_level_pattern(out, "all_to_all", std::nullopt);

    out << YAML::EndMap;
}

// 7. Sequential all-to-all (hardest - stress test, many iterations)
void generate_sequential_all_to_all_test(
    YAML::Emitter& out, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "SequentialAllToAll");
    out << YAML::Key << "enable_flow_control" << YAML::Value << true;
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    std::vector<uint32_t> sizes{1024, 4096};
    emit_parametrization(out, NocSendTypeSet::STANDARD, sizes);

    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << 5000;
    out << YAML::EndMap;

    emit_high_level_pattern(out, "sequential_all_to_all", std::nullopt);

    out << YAML::EndMap;
}

}  // anonymous namespace

// =============================================================================
// Public API Implementation
// =============================================================================

MeshTopologyInfo extract_topology_info(const proto::ClusterDescriptor& cluster_desc, bool verbose) {
    MeshTopologyInfo info;

    // Collect host IDs
    std::set<uint32_t> host_ids;
    if (cluster_desc.has_root_instance()) {
        collect_host_ids(cluster_desc.root_instance(), host_ids);
    }

    if (host_ids.empty()) {
        throw std::runtime_error("No host_ids found in cabling descriptor");
    }

    info.num_meshes = host_ids.size();

    // Get node type and derive device dimensions
    if (cluster_desc.has_root_instance()) {
        info.node_type = find_node_type_from_template(cluster_desc.root_instance().template_name(), cluster_desc);
    }

    const auto& node_info = get_node_type_info();
    auto it = node_info.find(info.node_type);
    if (it != node_info.end()) {
        info.device_dims = it->second.device_dims;
        info.architecture = it->second.arch;
    } else {
        // Default to N300 dimensions
        info.device_dims = {2, 4};
        info.architecture = "WORMHOLE_B0";
    }

    // Generate hostnames and compute inter-mesh connections using CablingGenerator
    std::vector<std::string> hostnames = generate_hostnames(info.num_meshes);
    auto cabling_generator = CablingGenerator(cluster_desc, hostnames);
    const auto& chip_connections = cabling_generator.get_chip_connections();

    // Build mesh connectivity graph
    std::set<std::pair<uint32_t, uint32_t>> unique_pairs;
    for (const auto& connection : chip_connections) {
        uint32_t src_host = *connection.first.host_id;
        uint32_t dst_host = *connection.second.host_id;
        if (src_host != dst_host) {
            info.mesh_connections[src_host][dst_host]++;

            // Track unique pairs (ordered to avoid duplicates)
            auto pair = std::minmax(src_host, dst_host);
            unique_pairs.insert(pair);
        }
    }

    // Convert to ordered vector
    info.connected_pairs.assign(unique_pairs.begin(), unique_pairs.end());

    if (verbose) {
        std::cout << "Topology Info:\n";
        std::cout << "  Meshes: " << info.num_meshes << "\n";
        std::cout << "  Node type: " << info.node_type << "\n";
        std::cout << "  Architecture: " << info.architecture << "\n";
        std::cout << "  Device dims: " << info.device_dims[0] << "x" << info.device_dims[1] << "\n";
        std::cout << "  Connected pairs: " << info.connected_pairs.size() << "\n";
        for (const auto& [a, b] : info.connected_pairs) {
            std::cout << "    M" << a << " <-> M" << b << ": " << info.mesh_connections[a][b] << " channels\n";
        }
    }

    return info;
}

MeshTopologyInfo extract_topology_info(const std::filesystem::path& cabling_descriptor_path, bool verbose) {
    std::ifstream file(cabling_descriptor_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open cabling descriptor file: " + cabling_descriptor_path.string());
    }
    const std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    proto::ClusterDescriptor cluster_desc;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &cluster_desc)) {
        throw std::runtime_error("Failed to parse cabling descriptor protobuf: " + cabling_descriptor_path.string());
    }

    return extract_topology_info(cluster_desc, verbose);
}

std::string generate_traffic_tests_yaml(
    const MeshTopologyInfo& topology, const std::filesystem::path& mgd_path, const TrafficTestConfig& config) {
    YAML::Emitter out;
    out << YAML::Comment("Auto-generated traffic tests");
    out << YAML::Comment("Topology: " + std::to_string(topology.num_meshes) + " meshes, " + topology.architecture);
    out << YAML::Newline;

    out << YAML::BeginMap;
    out << YAML::Key << "Tests";
    out << YAML::Value << YAML::BeginSeq;

    // Generate tests from easiest to hardest based on profile
    // 1. Simple unicast (always included)
    generate_simple_unicast_test(out, mgd_path, config);

    // 2. Inter-mesh unicast (if multi-mesh)
    if (topology.num_meshes > 1) {
        generate_inter_mesh_unicast_test(out, topology, mgd_path, config);
    }

    // 3. All-to-all unicast
    generate_all_to_all_unicast_test(out, mgd_path, config);

    // 4. Random pairing
    if (config.profile != TestProfile::SANITY) {
        generate_random_pairing_test(out, mgd_path, config);
    }

    // 5. All-to-one (convergence stress)
    if (config.profile == TestProfile::STRESS || config.profile == TestProfile::COVERAGE) {
        generate_all_to_one_test(out, mgd_path, config);
    }

    // 6. Flow control tests
    if (config.include_flow_control ||
        config.profile == TestProfile::STRESS ||
        config.profile == TestProfile::COVERAGE) {
        generate_flow_control_test(out, mgd_path, config);
    }

    // 7. Sequential all-to-all (hardest)
    if (config.profile == TestProfile::STRESS || config.profile == TestProfile::COVERAGE) {
        generate_sequential_all_to_all_test(out, mgd_path, config);
    }

    out << YAML::EndSeq;
    out << YAML::EndMap;

    return out.c_str();
}

std::string generate_traffic_tests_yaml(
    const proto::ClusterDescriptor& cluster_desc, const TrafficTestConfig& config, bool verbose) {
    auto topology = extract_topology_info(cluster_desc, verbose);

    std::filesystem::path mgd_path;
    if (config.existing_mgd_path.has_value()) {
        mgd_path = config.existing_mgd_path.value();
    } else if (!config.mgd_output_path.empty()) {
        mgd_path = config.mgd_output_path;
    }

    return generate_traffic_tests_yaml(topology, mgd_path, config);
}

void write_traffic_tests_to_file(const std::string& yaml_content, const std::filesystem::path& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + output_path.string());
    }

    file << "# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC\n";
    file << "#\n";
    file << "# SPDX-License-Identifier: Apache-2.0\n\n";
    file << yaml_content;
}

void generate_traffic_tests(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    const TrafficTestConfig& config,
    bool verbose) {
    if (verbose) {
        std::cout << "Generating traffic tests from: " << cabling_descriptor_path << "\n";
    }

    // Read and parse the cabling descriptor
    std::ifstream file(cabling_descriptor_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open cabling descriptor file: " + cabling_descriptor_path.string());
    }
    const std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    proto::ClusterDescriptor cluster_desc;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &cluster_desc)) {
        throw std::runtime_error("Failed to parse cabling descriptor protobuf: " + cabling_descriptor_path.string());
    }

    // Generate MGD if requested
    std::filesystem::path mgd_path = config.mgd_output_path;
    if (config.generate_mgd && !config.mgd_output_path.empty()) {
        auto mgd = generate_mgd_from_cabling(cluster_desc, verbose);
        write_mgd_to_file(mgd, config.mgd_output_path);
        if (verbose) {
            std::cout << "Generated MGD: " << config.mgd_output_path << "\n";
        }
    } else if (config.existing_mgd_path.has_value()) {
        mgd_path = config.existing_mgd_path.value();
    }

    // Generate traffic tests
    auto topology = extract_topology_info(cluster_desc, verbose);
    auto yaml_content = generate_traffic_tests_yaml(topology, mgd_path, config);
    write_traffic_tests_to_file(yaml_content, output_path);

    if (verbose) {
        std::cout << "Generated traffic tests: " << output_path << "\n";
    }
}

// =============================================================================
// Profile Presets
// =============================================================================

TrafficTestConfig get_sanity_config() {
    return TrafficTestConfig{
        .profile = TestProfile::SANITY,
        .noc_types = NocSendTypeSet::BASIC,
        .fabric_types = FabricTypeSet::UNICAST_ONLY,
        .include_flow_control = false,
        .include_sync = true,
        .packet_sizes = {{1024, 2048}},
        .packet_counts = {{100}},
    };
}

TrafficTestConfig get_stress_config() {
    return TrafficTestConfig{
        .profile = TestProfile::STRESS,
        .noc_types = NocSendTypeSet::STANDARD,
        .fabric_types = FabricTypeSet::UNICAST_ONLY,
        .include_flow_control = true,
        .include_sync = true,
        .packet_sizes = {{1024, 2048, 4096}},
        .packet_counts = {{1000, 5000, 10000}},
    };
}

TrafficTestConfig get_benchmark_config() {
    return TrafficTestConfig{
        .profile = TestProfile::BENCHMARK,
        .noc_types = NocSendTypeSet::BASIC,
        .fabric_types = FabricTypeSet::UNICAST_ONLY,
        .include_flow_control = true,
        .include_sync = true,
        .packet_sizes = {{512, 1024, 2048, 4096, 8192}},
        .packet_counts = {{1000}},
    };
}

TrafficTestConfig get_coverage_config() {
    return TrafficTestConfig{
        .profile = TestProfile::COVERAGE,
        .noc_types = NocSendTypeSet::FULL,
        .fabric_types = FabricTypeSet::ALL,
        .include_flow_control = true,
        .include_sync = true,
        .packet_sizes = {{512, 1024, 2048, 4096}},
        .packet_counts = {{100, 500, 1000}},
    };
}

}  // namespace tt::scaleout_tools
