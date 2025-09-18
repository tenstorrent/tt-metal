// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::scaleout_tools {

enum class NodeType {
    N300_LB,
    N300_QB,
    WH_GALAXY,
    WH_GALAXY_X_TORUS,
    WH_GALAXY_Y_TORUS,
    WH_GALAXY_XY_TORUS,
    P150_QB_GLOBAL,
    P300_QB_AMERICA,
    BH_GALAXY,
    BH_GALAXY_X_TORUS,
    BH_GALAXY_Y_TORUS,
    BH_GALAXY_XY_TORUS,
};

NodeType get_node_type_from_string(const std::string& node_name);

}  // namespace tt::scaleout_tools
