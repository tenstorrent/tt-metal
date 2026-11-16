// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

enum class Architecture { WORMHOLE, BLACKHOLE };

// Base class for all node types - allows polymorphic access to node properties
class NodeBase {
public:
    virtual ~NodeBase() = default;
    virtual Architecture get_architecture() const = 0;
    virtual tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const = 0;
};

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type);

// Helper function to get BH Galaxy Rev from system, currently only intended
//    for use here
NodeType get_bh_glx_rev_from_system();

// Create a node instance (returns base class pointer for polymorphism)
std::unique_ptr<NodeBase> create_node_instance(NodeType node_type);

}  // namespace tt::scaleout_tools
