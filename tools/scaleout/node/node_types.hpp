// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::scaleout_tools {

enum class NodeType {
    N300_LB,
    N300_LB_DEFAULT,
    N300_QB,
    N300_QB_DEFAULT,
    WH_GALAXY,
    WH_GALAXY_X_TORUS,
    WH_GALAXY_Y_TORUS,
    WH_GALAXY_XY_TORUS,
    P150_QB_AE,
    P150_QB_AE_DEFAULT,
    P300_QB_GE,
    BH_GALAXY,
    BH_GALAXY_X_TORUS,
    BH_GALAXY_Y_TORUS,
    BH_GALAXY_XY_TORUS,
};

NodeType get_node_type_from_string(const std::string& node_name);

}  // namespace tt::scaleout_tools
