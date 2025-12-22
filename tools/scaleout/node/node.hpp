// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

enum class Architecture { WORMHOLE, BLACKHOLE };

// Topology enum for node types
enum class Topology {
    MESH,      // Standard mesh topology
    X_TORUS,   // X-axis torus only
    Y_TORUS,   // Y-axis torus only
    XY_TORUS,  // Both X and Y torus
};

// Forward declarations
// Base class for all node types - allows polymorphic access to node properties
class NodeBase {
public:
    virtual ~NodeBase() = default;
    virtual Topology get_topology() const { return Topology::MESH; }
    virtual Architecture get_architecture() const = 0;
    virtual tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const = 0;
};

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type);

// Helper function to get topology for a NodeType (uses virtual function from node instances)
Topology get_node_type_topology(NodeType node_type);

bool is_torus(NodeType node_type);

// Create a node instance (returns base class pointer for polymorphism)
std::unique_ptr<NodeBase> create_node_instance(NodeType node_type);

}  // namespace tt::scaleout_tools
