// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "node_types.hpp"

#include <stdexcept>
#include <string>
#include <enchantum/enchantum.hpp>
#include <tt_stl/caseless_comparison.hpp>

namespace tt::scaleout_tools {

NodeType get_node_type_from_string(const std::string& node_name) {
    auto node_type = enchantum::cast<NodeType>(node_name, ttsl::ascii_caseless_comp);
    if (!node_type.has_value()) {
        throw std::runtime_error("Invalid node type: " + std::string(node_name));
    }
    return *node_type;
}

}  // namespace tt::scaleout_tools
