// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/mesh_graph_descriptor.pb.h"

namespace tt::scaleout_tools {

// Node type information structure
struct NodeTypeInfo {
    std::vector<int> device_dims;
    tt::tt_fabric::proto::Architecture arch;
    int channel_count;
};

// Create lookup table for node types
std::unordered_map<std::string, NodeTypeInfo> create_node_type_lookup();

// Generate hostnames based on number of hosts
std::vector<std::string> generate_hostnames(size_t num_hosts);

// Compute intermesh connections from chip connections
std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> compute_intermesh_connections(
    const std::vector<std::string>& hostnames, const std::vector<ChipConnection>& chip_connections);

// Main function to generate Mesh Graph Descriptor from cabling descriptor
void generate_mesh_graph_descriptor(const std::string& cabling_descriptor_path, const std::string& output_path);

}  // namespace tt::scaleout_tools
