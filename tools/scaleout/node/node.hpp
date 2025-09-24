// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include <memory>

namespace tt::scaleout_tools {

// Forward declaration to hide protobuf implementation details
class NodeDescriptorImpl;

// RAII wrapper for NodeDescriptor that hides protobuf symbols
class NodeDescriptor {
public:
    NodeDescriptor();
    ~NodeDescriptor();

    // Copy constructor and assignment
    NodeDescriptor(const NodeDescriptor& other);
    NodeDescriptor& operator=(const NodeDescriptor& other);

    // Move constructor and assignment
    NodeDescriptor(NodeDescriptor&& other) noexcept;
    NodeDescriptor& operator=(NodeDescriptor&& other) noexcept;

    // Internal access for implementation (used by cabling_generator)
    void* get_internal_proto() const;

private:
    friend NodeDescriptor create_node_descriptor(NodeType node_type);
    explicit NodeDescriptor(std::unique_ptr<NodeDescriptorImpl> impl);

    std::unique_ptr<NodeDescriptorImpl> impl_;
};

// Factory function to create node descriptors by name
NodeDescriptor create_node_descriptor(NodeType node_type);

}  // namespace tt::scaleout_tools
