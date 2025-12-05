// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "node_type_info.hpp"

#include <stdexcept>
#include <unordered_map>

namespace tt::scaleout_tools {

namespace {

const std::unordered_map<NodeType, NodeTypeInfo>& get_node_type_lookup() {
    static const std::unordered_map<NodeType, NodeTypeInfo> lookup = {
        // Wormhole N300 architectures (2x4 device topology)
        {NodeType::N300_LB, {{2, 4}, "WORMHOLE_B0", 2}},
        {NodeType::N300_LB_DEFAULT, {{2, 4}, "WORMHOLE_B0", 2}},
        {NodeType::N300_QB, {{2, 4}, "WORMHOLE_B0", 2}},
        {NodeType::N300_QB_DEFAULT, {{2, 4}, "WORMHOLE_B0", 2}},

        // Wormhole Galaxy architectures (8x4 device topology)
        {NodeType::WH_GALAXY, {{8, 4}, "WORMHOLE_B0", 4}},
        {NodeType::WH_GALAXY_X_TORUS, {{8, 4}, "WORMHOLE_B0", 4}},
        {NodeType::WH_GALAXY_Y_TORUS, {{8, 4}, "WORMHOLE_B0", 4}},
        {NodeType::WH_GALAXY_XY_TORUS, {{8, 4}, "WORMHOLE_B0", 4}},

        // Blackhole P150 architectures
        {NodeType::P150_LB, {{2, 4}, "BLACKHOLE", 2}},
        {NodeType::P150_QB_AE, {{2, 2}, "BLACKHOLE", 4}},
        {NodeType::P150_QB_AE_DEFAULT, {{2, 2}, "BLACKHOLE", 4}},

        // Blackhole P300 architectures
        {NodeType::P300_QB_GE, {{2, 2}, "BLACKHOLE", 2}},

        // Blackhole Galaxy architectures (8x4 device topology)
        {NodeType::BH_GALAXY, {{8, 4}, "BLACKHOLE", 2}},
        {NodeType::BH_GALAXY_X_TORUS, {{8, 4}, "BLACKHOLE", 2}},
        {NodeType::BH_GALAXY_Y_TORUS, {{8, 4}, "BLACKHOLE", 2}},
        {NodeType::BH_GALAXY_XY_TORUS, {{8, 4}, "BLACKHOLE", 2}},
    };
    return lookup;
}

const std::unordered_map<std::string, NodeType>& get_string_to_node_type() {
    static const std::unordered_map<std::string, NodeType> lookup = {
        {"N300_LB", NodeType::N300_LB},
        {"N300_LB_DEFAULT", NodeType::N300_LB_DEFAULT},
        {"N300_QB", NodeType::N300_QB},
        {"N300_QB_DEFAULT", NodeType::N300_QB_DEFAULT},
        {"WH_GALAXY", NodeType::WH_GALAXY},
        {"WH_GALAXY_X_TORUS", NodeType::WH_GALAXY_X_TORUS},
        {"WH_GALAXY_Y_TORUS", NodeType::WH_GALAXY_Y_TORUS},
        {"WH_GALAXY_XY_TORUS", NodeType::WH_GALAXY_XY_TORUS},
        {"P150_LB", NodeType::P150_LB},
        {"P150_QB_AE", NodeType::P150_QB_AE},
        {"P150_QB_AE_DEFAULT", NodeType::P150_QB_AE_DEFAULT},
        {"P300_QB_GE", NodeType::P300_QB_GE},
        {"BH_GALAXY", NodeType::BH_GALAXY},
        {"BH_GALAXY_X_TORUS", NodeType::BH_GALAXY_X_TORUS},
        {"BH_GALAXY_Y_TORUS", NodeType::BH_GALAXY_Y_TORUS},
        {"BH_GALAXY_XY_TORUS", NodeType::BH_GALAXY_XY_TORUS},
    };
    return lookup;
}

}  // namespace

const NodeTypeInfo& get_node_type_info(NodeType node_type) {
    const auto& lookup = get_node_type_lookup();
    auto it = lookup.find(node_type);
    if (it == lookup.end()) {
        throw std::runtime_error("Unknown node type in get_node_type_info");
    }
    return it->second;
}

const NodeTypeInfo& get_node_type_info(const std::string& node_type_name) {
    const auto& string_lookup = get_string_to_node_type();
    auto it = string_lookup.find(node_type_name);
    if (it == string_lookup.end()) {
        throw std::runtime_error(
            "Unknown node type '" + node_type_name + "'. Supported types: " + get_supported_node_types_string());
    }
    return get_node_type_info(it->second);
}

bool is_known_node_type(const std::string& node_type_name) {
    const auto& string_lookup = get_string_to_node_type();
    return string_lookup.find(node_type_name) != string_lookup.end();
}

std::string get_supported_node_types_string() {
    std::string result;
    const auto& lookup = get_string_to_node_type();
    for (const auto& [name, _] : lookup) {
        if (!result.empty()) {
            result += ", ";
        }
        result += name;
    }
    return result;
}

}  // namespace tt::scaleout_tools
