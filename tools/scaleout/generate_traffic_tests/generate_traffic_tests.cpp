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
#include <node/node_type_info.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

namespace {

void collect_host_ids(const proto::GraphInstance& instance, std::set<uint32_t>& host_ids) {
    for (const auto& [child_name, child_mapping] : instance.child_mappings()) {
        if (child_mapping.has_host_id()) {
            host_ids.insert(child_mapping.host_id());
        } else if (child_mapping.has_sub_instance()) {
            collect_host_ids(child_mapping.sub_instance(), host_ids);
        }
    }
}

std::string find_node_type_from_template(
    const std::string& template_name, const proto::ClusterDescriptor& cluster_desc) {
    auto it = cluster_desc.graph_templates().find(template_name);
    if (it == cluster_desc.graph_templates().end()) {
        throw std::runtime_error("Graph template '" + template_name + "' not found");
    }

    for (const auto& child : it->second.children()) {
        if (child.has_node_ref()) {
            return child.node_ref().node_descriptor();
        }
        if (child.has_graph_ref()) {
            return find_node_type_from_template(child.graph_ref().graph_template(), cluster_desc);
        }
    }

    throw std::runtime_error("No node references found in graph template '" + template_name + "'");
}

std::vector<std::string> generate_hostnames(size_t num_hosts) {
    std::vector<std::string> hostnames;
    hostnames.reserve(num_hosts);
    for (size_t i = 0; i < num_hosts; ++i) {
        hostnames.push_back("M" + std::to_string(i));
    }
    return hostnames;
}

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
    YAML::Emitter& out, const std::string& ftype, const std::string& ntype, uint32_t size, uint32_t num_packets) {
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

void emit_parametrization(
    YAML::Emitter& out, const std::vector<std::string>& noc_types, const std::vector<uint32_t>& sizes) {
    out << YAML::Key << "parametrization_params";
    out << YAML::Value << YAML::BeginMap;

    out << YAML::Key << "ntype";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (const auto& ntype : noc_types) {
        out << ntype;
    }
    out << YAML::EndSeq;

    out << YAML::Key << "size";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (auto s : sizes) {
        out << s;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

void emit_pattern(YAML::Emitter& out, const std::string& pattern_type, std::optional<uint32_t> iterations = {}) {
    out << YAML::Key << "patterns";
    out << YAML::Value << YAML::BeginSeq;
    out << YAML::BeginMap;
    out << YAML::Key << "type" << YAML::Value << pattern_type;
    if (iterations) {
        out << YAML::Key << "iterations" << YAML::Value << *iterations;
    }
    out << YAML::EndMap;
    out << YAML::EndSeq;
}

void emit_skip_platforms(YAML::Emitter& out, const std::vector<std::string>& platforms) {
    if (platforms.empty()) {
        return;
    }
    out << YAML::Key << "skip";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (const auto& p : platforms) {
        out << p;
    }
    out << YAML::EndSeq;
}

void generate_simple_unicast_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "SimpleUnicast");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_defaults(out, "unicast", "unicast_write", sizes[0], num_packets);

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

void generate_inter_mesh_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes) {
    if (topology.connected_pairs.empty()) {
        return;
    }

    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "InterMeshUnicast");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_defaults(out, "unicast", "unicast_write", sizes.size() > 1 ? sizes[1] : sizes[0], num_packets);

    out << YAML::Key << "senders";
    out << YAML::Value << YAML::BeginSeq;
    for (const auto& [src, dst] : topology.connected_pairs) {
        out << YAML::BeginMap;
        out << YAML::Key << "device";
        emit_device_coord(out, src, 0, 0);
        out << YAML::Key << "patterns";
        out << YAML::Value << YAML::BeginSeq;
        out << YAML::BeginMap;
        out << YAML::Key << "destination";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "device";
        emit_device_coord(out, dst, 0, 0);
        out << YAML::EndMap;
        out << YAML::EndMap;
        out << YAML::EndSeq;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

void generate_all_to_all_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes,
    const std::vector<std::string>& noc_types) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "AllToAll");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_parametrization(out, noc_types, sizes);

    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << num_packets;
    out << YAML::EndMap;

    emit_pattern(out, "all_to_all");
    out << YAML::EndMap;
}

void generate_random_pairing_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "RandomPairing");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_defaults(out, "unicast", "unicast_write", sizes.size() > 1 ? sizes[1] : sizes[0], num_packets);

    uint32_t iterations = config.profile == TestProfile::STRESS ? 5 : 3;
    emit_pattern(out, "full_device_random_pairing", iterations);
    out << YAML::EndMap;
}

void generate_all_to_one_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "AllToOne");
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_parametrization(out, {"unicast_write"}, sizes);

    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << num_packets;
    out << YAML::EndMap;

    emit_pattern(out, "all_to_one");
    out << YAML::EndMap;
}

void generate_flow_control_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes,
    const std::vector<std::string>& noc_types) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "FlowControl");
    out << YAML::Key << "enable_flow_control" << YAML::Value << true;
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);
    emit_parametrization(out, noc_types, sizes);

    uint32_t fc_packets = std::max(num_packets, 5000u);
    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << fc_packets;
    out << YAML::EndMap;

    emit_pattern(out, "all_to_all");
    out << YAML::EndMap;
}

void generate_sequential_test(
    YAML::Emitter& out,
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    uint32_t num_packets,
    const std::vector<uint32_t>& sizes,
    const std::vector<std::string>& noc_types) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << (config.test_name_prefix + "SequentialAllToAll");
    out << YAML::Key << "enable_flow_control" << YAML::Value << true;
    if (config.include_sync) {
        out << YAML::Key << "sync" << YAML::Value << true;
    }
    emit_skip_platforms(out, config.skip_platforms);
    emit_fabric_setup(out, mgd_path);

    std::vector<uint32_t> seq_sizes = {sizes[0]};
    if (sizes.size() > 2) {
        seq_sizes.push_back(sizes[sizes.size() / 2]);
    }
    emit_parametrization(out, noc_types, seq_sizes);

    out << YAML::Key << "defaults";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "ftype" << YAML::Value << "unicast";
    out << YAML::Key << "num_packets" << YAML::Value << std::min(num_packets, 2000u);
    out << YAML::EndMap;

    emit_pattern(out, "sequential_all_to_all");
    out << YAML::EndMap;
}

}  // namespace

MeshTopologyInfo extract_topology_info(const proto::ClusterDescriptor& cluster_desc, bool verbose) {
    MeshTopologyInfo info;

    std::set<uint32_t> host_ids;
    if (cluster_desc.has_root_instance()) {
        collect_host_ids(cluster_desc.root_instance(), host_ids);
    }

    if (host_ids.empty()) {
        throw std::runtime_error("No host_ids found in cabling descriptor");
    }

    info.num_meshes = host_ids.size();

    if (cluster_desc.has_root_instance()) {
        info.node_type = find_node_type_from_template(cluster_desc.root_instance().template_name(), cluster_desc);
    }

    if (is_known_node_type(info.node_type)) {
        const auto& node_info = get_node_type_info(info.node_type);
        info.device_dims = node_info.device_dims;
        info.architecture = node_info.architecture;
    } else {
        info.device_dims = {2, 4};
        info.architecture = "WORMHOLE_B0";
    }

    auto hostnames = generate_hostnames(info.num_meshes);
    auto cabling_generator = CablingGenerator(cluster_desc, hostnames);
    const auto& chip_connections = cabling_generator.get_chip_connections();

    std::set<std::pair<uint32_t, uint32_t>> unique_pairs;
    for (const auto& conn : chip_connections) {
        uint32_t src = *conn.first.host_id;
        uint32_t dst = *conn.second.host_id;
        if (src != dst) {
            info.mesh_connections[src][dst]++;
            unique_pairs.insert(std::minmax(src, dst));
        }
    }
    info.connected_pairs.assign(unique_pairs.begin(), unique_pairs.end());

    if (verbose) {
        std::cout << "Topology: " << info.num_meshes << " meshes, " << info.total_devices() << " devices, "
                  << info.architecture << "\n";
        std::cout << "  Node type: " << info.node_type << " (" << info.device_dims[0] << "x" << info.device_dims[1]
                  << ")\n";
        std::cout << "  Inter-mesh connections: " << info.connected_pairs.size() << " pairs\n";
    }

    return info;
}

MeshTopologyInfo extract_topology_info(const std::filesystem::path& path, bool verbose) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + path.string());
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    proto::ClusterDescriptor cluster_desc;
    if (!google::protobuf::TextFormat::ParseFromString(content, &cluster_desc)) {
        throw std::runtime_error("Failed to parse: " + path.string());
    }

    return extract_topology_info(cluster_desc, verbose);
}

void apply_profile_defaults(TrafficTestConfig& config) {
    switch (config.profile) {
        case TestProfile::SANITY:
            if (!config.packet_sizes) {
                config.packet_sizes = {1024, 2048};
            }
            if (!config.num_packets) {
                config.num_packets = 100;
            }
            if (config.noc_types.empty()) {
                config.noc_types = {"unicast_write"};
            }
            break;

        case TestProfile::STRESS:
            config.categories.random_pairing = true;
            config.categories.all_to_one = true;
            config.categories.flow_control = true;
            config.categories.sequential = true;
            if (!config.packet_sizes) {
                config.packet_sizes = {1024, 2048, 4096};
            }
            if (!config.num_packets) {
                config.num_packets = 1000;
            }
            if (config.noc_types.empty()) {
                config.noc_types = {"unicast_write", "atomic_inc", "fused_atomic_inc"};
            }
            break;

        case TestProfile::BENCHMARK:
            if (!config.packet_sizes) {
                config.packet_sizes = {512, 1024, 2048, 4096, 8192};
            }
            if (!config.num_packets) {
                config.num_packets = 1000;
            }
            if (config.noc_types.empty()) {
                config.noc_types = {"unicast_write"};
            }
            break;
    }
}

std::string generate_traffic_tests_yaml(
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config,
    bool verbose) {
    TrafficTestConfig cfg = config;
    apply_profile_defaults(cfg);

    std::vector<uint32_t> sizes = cfg.packet_sizes.value_or(std::vector<uint32_t>{1024, 2048});
    uint32_t num_packets = cfg.num_packets.value_or(100);
    std::vector<std::string> noc_types =
        cfg.noc_types.empty() ? std::vector<std::string>{"unicast_write"} : cfg.noc_types;

    YAML::Emitter out;
    out << YAML::Comment("Auto-generated traffic tests");
    out << YAML::Comment(
        "Topology: " + std::to_string(topology.num_meshes) + " meshes, " + std::to_string(topology.total_devices()) +
        " devices, " + topology.architecture);
    out << YAML::Comment(
        "Profile: " + std::string(
                          cfg.profile == TestProfile::SANITY   ? "sanity"
                          : cfg.profile == TestProfile::STRESS ? "stress"
                                                               : "benchmark"));
    out << YAML::Newline;

    out << YAML::BeginMap;
    out << YAML::Key << "Tests";
    out << YAML::Value << YAML::BeginSeq;

    const auto& cat = cfg.categories;

    if (cat.simple_unicast) {
        generate_simple_unicast_test(out, topology, mgd_path, cfg, num_packets, sizes);
    }
    if (cat.inter_mesh && topology.num_meshes > 1) {
        generate_inter_mesh_test(out, topology, mgd_path, cfg, num_packets, sizes);
    }
    if (cat.all_to_all) {
        generate_all_to_all_test(out, topology, mgd_path, cfg, num_packets, sizes, noc_types);
    }
    if (cat.random_pairing) {
        generate_random_pairing_test(out, topology, mgd_path, cfg, num_packets, sizes);
    }
    if (cat.all_to_one) {
        generate_all_to_one_test(out, topology, mgd_path, cfg, num_packets, sizes);
    }
    if (cat.flow_control) {
        generate_flow_control_test(out, topology, mgd_path, cfg, num_packets, sizes, noc_types);
    }
    if (cat.sequential) {
        generate_sequential_test(out, topology, mgd_path, cfg, num_packets, sizes, noc_types);
    }

    out << YAML::EndSeq;
    out << YAML::EndMap;

    return out.c_str();
}

std::string generate_traffic_tests_yaml(
    const proto::ClusterDescriptor& cluster_desc, const TrafficTestConfig& config, bool verbose) {
    auto topology = extract_topology_info(cluster_desc, verbose);

    std::filesystem::path mgd_path;
    if (config.existing_mgd_path) {
        mgd_path = *config.existing_mgd_path;
    } else if (!config.mgd_output_path.empty()) {
        mgd_path = config.mgd_output_path;
    }

    return generate_traffic_tests_yaml(topology, mgd_path, config, verbose);
}

void write_traffic_tests_to_file(const std::string& yaml_content, const std::filesystem::path& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write to: " + output_path.string());
    }

    file << "# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC\n";
    file << "#\n";
    file << "# SPDX-License-Identifier: Apache-2.0\n\n";
    file << yaml_content;
}

void generate_traffic_tests(
    const std::filesystem::path& cabling_path,
    const std::filesystem::path& output_path,
    const TrafficTestConfig& config,
    bool verbose) {
    if (verbose) {
        std::cout << "Reading cabling descriptor: " << cabling_path << "\n";
    }

    std::ifstream file(cabling_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + cabling_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    proto::ClusterDescriptor cluster_desc;
    if (!google::protobuf::TextFormat::ParseFromString(content, &cluster_desc)) {
        throw std::runtime_error("Failed to parse: " + cabling_path.string());
    }

    std::filesystem::path mgd_path = config.mgd_output_path;
    if (config.generate_mgd && !config.mgd_output_path.empty()) {
        auto mgd = generate_mgd_from_cabling(cluster_desc, verbose);
        write_mgd_to_file(mgd, config.mgd_output_path);
        if (verbose) {
            std::cout << "Generated MGD: " << config.mgd_output_path << "\n";
        }
    } else if (config.existing_mgd_path) {
        mgd_path = *config.existing_mgd_path;
    }

    auto topology = extract_topology_info(cluster_desc, verbose);
    auto yaml = generate_traffic_tests_yaml(topology, mgd_path, config, verbose);
    write_traffic_tests_to_file(yaml, output_path);

    if (verbose) {
        std::cout << "Generated: " << output_path << "\n";
    }
}

TrafficTestConfig get_sanity_config() {
    TrafficTestConfig config;
    config.profile = TestProfile::SANITY;
    config.categories = {
        .simple_unicast = true,
        .inter_mesh = true,
        .all_to_all = true,
        .random_pairing = false,
        .all_to_one = false,
        .flow_control = false,
        .sequential = false};
    return config;
}

TrafficTestConfig get_stress_config() {
    TrafficTestConfig config;
    config.profile = TestProfile::STRESS;
    config.categories = {
        .simple_unicast = true,
        .inter_mesh = true,
        .all_to_all = true,
        .random_pairing = true,
        .all_to_one = true,
        .flow_control = true,
        .sequential = true};
    return config;
}

TrafficTestConfig get_benchmark_config() {
    TrafficTestConfig config;
    config.profile = TestProfile::BENCHMARK;
    config.categories = {
        .simple_unicast = true,
        .inter_mesh = true,
        .all_to_all = true,
        .random_pairing = false,
        .all_to_one = false,
        .flow_control = true,
        .sequential = false};
    return config;
}

}  // namespace tt::scaleout_tools
