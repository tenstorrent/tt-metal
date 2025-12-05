// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/mesh_graph_descriptor.pb.h"

namespace tt::scaleout_tools {

namespace proto = cabling_generator::proto;

// Generate MGD from cabling descriptor protobuf object
tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const proto::ClusterDescriptor& cluster_desc, bool verbose = false);

// Generate MGD from cabling descriptor file
tt::tt_fabric::proto::MeshGraphDescriptor generate_mgd_from_cabling(
    const std::filesystem::path& cabling_descriptor_path, bool verbose = false);

// Write MGD protobuf to file in textproto format
void write_mgd_to_file(const tt::tt_fabric::proto::MeshGraphDescriptor& mgd, const std::filesystem::path& output_path);

// Convenience function: generate and write MGD in one call
void generate_mesh_graph_descriptor(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    bool verbose = false);

}  // namespace tt::scaleout_tools
