// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

namespace proto = cabling_generator::proto;

struct TestCategories {
    bool simple_unicast = true;
    bool inter_mesh = true;
    bool all_to_all = true;
    bool random_pairing = false;
    bool all_to_one = false;
    bool flow_control = false;
    bool sequential = false;
};

enum class TestProfile { SANITY, STRESS, BENCHMARK };

struct TrafficTestConfig {
    TestProfile profile = TestProfile::SANITY;
    TestCategories categories;

    bool generate_mgd = true;
    std::filesystem::path mgd_output_path;
    std::optional<std::filesystem::path> existing_mgd_path;

    std::optional<std::vector<uint32_t>> packet_sizes;
    std::optional<uint32_t> num_packets;
    std::vector<std::string> noc_types;

    bool include_sync = true;
    std::vector<std::string> skip_platforms;
    std::string test_name_prefix;
};

struct MeshTopologyInfo {
    size_t num_meshes = 0;
    std::vector<int> device_dims;
    std::string node_type;
    std::string architecture;
    std::map<uint32_t, std::map<uint32_t, uint32_t>> mesh_connections;
    std::vector<std::pair<uint32_t, uint32_t>> connected_pairs;

    size_t total_devices() const { return num_meshes * device_dims[0] * device_dims[1]; }
};

MeshTopologyInfo extract_topology_info(const proto::ClusterDescriptor& cluster_desc, bool verbose = false);
MeshTopologyInfo extract_topology_info(const std::filesystem::path& cabling_descriptor_path, bool verbose = false);
MeshTopologyInfo extract_topology_info_from_mgd(const std::filesystem::path& mgd_path, bool verbose = false);

std::string generate_traffic_tests_yaml(
    const MeshTopologyInfo& topology,
    const std::filesystem::path& mgd_path,
    const TrafficTestConfig& config = {},
    bool verbose = false);

std::string generate_traffic_tests_yaml(
    const proto::ClusterDescriptor& cluster_desc, const TrafficTestConfig& config = {}, bool verbose = false);

void write_traffic_tests_to_file(const std::string& yaml_content, const std::filesystem::path& output_path);

void generate_traffic_tests(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    const TrafficTestConfig& config = {},
    bool verbose = false);

TrafficTestConfig get_sanity_config();
TrafficTestConfig get_stress_config();
TrafficTestConfig get_benchmark_config();

void apply_profile_defaults(TrafficTestConfig& config);

}  // namespace tt::scaleout_tools
