// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/mesh_graph_descriptor.pb.h"

namespace tt::scaleout_tools {

namespace proto = cabling_generator::proto;

// Node type information structure
struct NodeTypeInfo {
    std::vector<int> device_dims;
    tt::tt_fabric::proto::Architecture arch;
    int channel_count;
};

// Information extracted from cabling descriptor
struct CablingDescriptorInfo {
    size_t num_hosts{};
    std::string node_type;
    std::set<uint32_t> host_ids;
};

// Create lookup table for node types
[[nodiscard]] std::unordered_map<std::string, NodeTypeInfo> create_node_type_lookup();

// Extract cluster information from cabling descriptor
[[nodiscard]] CablingDescriptorInfo get_cabling_descriptor_info(std::string_view cabling_descriptor_path);

// Generate hostnames based on number of hosts
[[nodiscard]] std::vector<std::string> generate_hostnames(size_t num_hosts);

// Compute intermesh connections from chip connections
[[nodiscard]] std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> compute_intermesh_connections(
    const std::vector<std::string>& hostnames, const std::vector<LogicalChannelConnection>& chip_connections);

// Generate MGD from cabling descriptor protobuf object
[[nodiscard]] tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const proto::ClusterDescriptor& cluster_desc, bool verbose = false);

// Generate MGD from cabling descriptor file
[[nodiscard]] tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const std::filesystem::path& cabling_descriptor_path, bool verbose = false);

// Write MGD protobuf to file in textproto format
void write_mgd_to_file(const tt::tt_fabric::proto::MeshGraphDescriptor& mgd, const std::filesystem::path& output_path);

// Convenience function: generate and write MGD in one call
void generate_mesh_graph_descriptor(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    bool verbose = false);

}  // namespace tt::scaleout_tools
