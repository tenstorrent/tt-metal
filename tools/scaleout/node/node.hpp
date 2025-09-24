// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_types.hpp"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type);

}  // namespace tt::scaleout_tools
