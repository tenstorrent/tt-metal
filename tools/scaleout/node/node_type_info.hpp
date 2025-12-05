// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "node_types.hpp"

namespace tt::scaleout_tools {

/// Information about a node type needed for MGD and traffic test generation
struct NodeTypeInfo {
    std::vector<int> device_dims;  // Device topology [rows, cols]
    std::string architecture;      // "WORMHOLE_B0" or "BLACKHOLE"
    int channel_count;             // Number of fabric channels
};

/// Get node type info from NodeType enum
[[nodiscard]] const NodeTypeInfo& get_node_type_info(NodeType node_type);

/// Get node type info from node type string (e.g., "N300_LB_DEFAULT")
[[nodiscard]] const NodeTypeInfo& get_node_type_info(const std::string& node_type_name);

/// Check if a node type string is known/supported
[[nodiscard]] bool is_known_node_type(const std::string& node_type_name);

/// Get a comma-separated list of all supported node type names
[[nodiscard]] std::string get_supported_node_types_string();

}  // namespace tt::scaleout_tools
