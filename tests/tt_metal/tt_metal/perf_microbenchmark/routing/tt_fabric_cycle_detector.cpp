// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_cycle_detector.hpp"
#include <filesystem>
#include <fstream>

namespace tt::tt_fabric::fabric_tests {

// Dump cycles to YAML for debugging
void dump_cycles_to_yaml(
    const std::vector<std::vector<FabricNodeId>>& cycles,
    const std::string& test_name,
    int level,
    const std::string& output_dir) {
    if (cycles.empty()) {
        return;
    }

    std::filesystem::create_directories(output_dir);
    std::string file_path = output_dir + "/cycles_" + test_name + "_level_" + std::to_string(level) + ".yaml";

    std::ofstream fout(file_path);
    if (!fout.is_open()) {
        log_warning(tt::LogTest, "Failed to open file for writing: {}", file_path);
        return;
    }

    fout << "test_name: " << test_name << "\n";
    fout << "level: " << level << "\n";
    fout << "cycles_found: " << cycles.size() << "\n";
    fout << "cycles:\n";

    for (size_t i = 0; i < cycles.size(); ++i) {
        fout << "  - cycle_" << i << ": [";
        for (size_t j = 0; j < cycles[i].size(); ++j) {
            if (j > 0) {
                fout << ", ";
            }
            const auto& node = cycles[i][j];
            fout << "M" << *node.mesh_id << ":C" << node.chip_id;
        }
        fout << "]\n";
    }

    fout.close();
    log_info(tt::LogTest, "Cycles dumped to: {}", file_path);
}

// Main cycle detection function - delegates to control plane API and dumps cycles to YAML
bool detect_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name) {
    // Get control plane from route manager
    const void* cp_ptr = route_manager.get_control_plane();
    if (!cp_ptr) {
        log_warning(tt::LogTest, "Control plane not available for cycle detection in test '{}'", test_name);
        return false;
    }

    const auto* control_plane = static_cast<const ControlPlane*>(cp_ptr);

    // Get cycles for YAML dumping
    std::vector<std::vector<FabricNodeId>> cycles;
    bool has_cycles = control_plane->detect_inter_mesh_cycles(pairs, cycles, test_name);

    if (has_cycles) {
        // Dump cycles to YAML for debugging
        dump_cycles_to_yaml(cycles, test_name, 0, "generated/fabric");
    }

    return has_cycles;
}

// Overload for direct control plane access
bool detect_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const ControlPlane& control_plane,
    const std::string& test_name) {
    // Get cycles for YAML dumping
    std::vector<std::vector<FabricNodeId>> cycles;
    bool has_cycles = control_plane.detect_inter_mesh_cycles(pairs, cycles, test_name);

    if (has_cycles) {
        // Dump cycles to YAML for debugging
        dump_cycles_to_yaml(cycles, test_name, 0, "generated/fabric");
    }

    return has_cycles;
}

}  // namespace tt::tt_fabric::fabric_tests
