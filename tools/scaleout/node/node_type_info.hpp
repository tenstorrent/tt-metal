// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "node_types.hpp"

namespace tt::scaleout_tools {

struct NodeTypeInfo {
    std::vector<int> device_dims;
    std::string architecture;
    int channel_count;
};

const NodeTypeInfo& get_node_type_info(NodeType node_type);
const NodeTypeInfo& get_node_type_info(const std::string& node_type_name);
bool is_known_node_type(const std::string& node_type_name);
std::string get_supported_node_types_string();

}  // namespace tt::scaleout_tools
