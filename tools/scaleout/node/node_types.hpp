// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::scaleout_tools {

enum class NodeType {
    N300_LB,
    N300_QB,
    WH_GALAXY,
    P150_LB,
    P150_QB_AE,
    P300_QB_GE,
    BH_GALAXY_REV_AB,
    BH_GALAXY_REV_C,
    BH_GALAXY,
};

NodeType get_node_type_from_string(const std::string& node_name);

}  // namespace tt::scaleout_tools
