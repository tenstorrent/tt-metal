// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

// Topology enum for node types
enum class Topology {
    MESH,      // Standard mesh topology
    X_TORUS,   // X-axis torus only
    Y_TORUS,   // Y-axis torus only
    XY_TORUS,  // Both X and Y torus
};

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type);

// Helper function to get topology for a NodeType (uses virtual function from node instances)
Topology get_node_type_topology(NodeType node_type);

// Torus-related helper functions
bool is_a_torus(NodeType node_type);
bool has_same_torus_architecture(NodeType type1, NodeType type2);
bool is_torus_compatible(NodeType type1, NodeType type2);

}  // namespace tt::scaleout_tools
