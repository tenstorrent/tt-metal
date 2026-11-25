// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

enum class Architecture { WORMHOLE, BLACKHOLE };

// Galaxy board revision named by a node type
enum class BoardRevision {
    UNSPECIFIED,  // Not a Galaxy, or BH_GALAXY (revision resolved from the system)
    WH_GALAXY,
    BH_GALAXY_REV_AB,
    BH_GALAXY_REV_C,
};

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

// These never probe the system, so they are safe to use offline.
Architecture get_node_type_architecture(NodeType node_type);
BoardRevision get_node_type_board_revision(NodeType node_type);

// Create a node instance (returns base class pointer for polymorphism)
std::unique_ptr<NodeBase> create_node_instance(NodeType node_type);

}  // namespace tt::scaleout_tools
