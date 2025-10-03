// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>

#include "tt_fabric_test_interfaces.hpp"  // For IRouteManager
#include <tt-metalium/control_plane.hpp>  // For ControlPlane access
#include "tt-logger/tt-logger.hpp"

namespace tt::tt_fabric::fabric_tests {

// Dump cycles to YAML for debugging
void dump_cycles_to_yaml(
    const std::vector<std::vector<FabricNodeId>>& cycles,
    const std::string& test_name,
    int level,
    const std::string& output_dir = "generated/fabric");

// Main cycle detection function using control plane API
// Detects cycles in inter-mesh traffic (intra-mesh uses dimension-ordered routing and is cycle-free)
// This is the recommended API - delegates to ControlPlane::detect_inter_mesh_cycles()
// Automatically dumps detected cycles to YAML files in generated/fabric/
bool detect_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name);

// Overload for direct control plane access
bool detect_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const ControlPlane& control_plane,
    const std::string& test_name);

}  // namespace tt::tt_fabric::fabric_tests
