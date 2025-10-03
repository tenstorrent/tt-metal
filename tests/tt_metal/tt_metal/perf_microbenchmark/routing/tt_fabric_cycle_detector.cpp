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
    fout << "\n";
    fout << "cycles:\n";

    for (size_t i = 0; i < cycles.size(); ++i) {
        const auto& cycle = cycles[i];

        // Calculate cycle statistics
        std::set<uint32_t> unique_meshes;
        std::set<uint32_t> unique_chips;
        size_t mesh_transitions = 0;
        size_t cycle_start_idx = cycle.size() - 1;

        for (size_t j = 0; j < cycle.size(); ++j) {
            unique_meshes.insert(*cycle[j].mesh_id);
            unique_chips.insert(cycle[j].chip_id);

            if (j < cycle.size() - 1 && cycle[j].mesh_id != cycle[j + 1].mesh_id) {
                mesh_transitions++;
            }

            // Find where the last node appears earlier (actual cycle point)
            if (j < cycle.size() - 1 && cycle[j] == cycle[cycle.size() - 1]) {
                cycle_start_idx = j;
            }
        }

        size_t cycle_length = cycle.size() - cycle_start_idx;
        bool is_bidirectional = (cycle_length == 3);  // A->B->A pattern

        fout << "  - cycle_" << i << ":\n";
        fout << "      # ===== SUMMARY =====\n";
        fout << "      path_length: " << cycle.size() << "\n";
        fout << "      cycle_length: " << cycle_length << "  # Minimal cycle loop size\n";
        fout << "      meshes_involved: [";
        bool first_mesh = true;
        for (auto mesh : unique_meshes) {
            if (!first_mesh) {
                fout << ", ";
            }
            fout << mesh;
            first_mesh = false;
        }
        fout << "]\n";
        fout << "      mesh_transitions: " << mesh_transitions << "\n";
        fout << "      is_bidirectional: " << (is_bidirectional ? "true" : "false");
        if (is_bidirectional) {
            fout << "  # Simple A↔B inter-mesh cycle";
        }
        fout << "\n";

        // Compact path representation
        fout << "      compact_path: [";
        for (size_t j = 0; j < cycle.size(); ++j) {
            if (j > 0) {
                fout << ", ";
            }
            fout << "M" << *cycle[j].mesh_id << ":C" << cycle[j].chip_id;
        }
        fout << "]\n";

        // Detailed path breakdown
        fout << "\n";
        fout << "      # ===== DETAILED PATH =====\n";
        fout << "      path:\n";

        for (size_t j = 0; j < cycle.size(); ++j) {
            const auto& node = cycle[j];
            fout << "        - node_" << j << ": ";
            fout << "{ mesh: " << *node.mesh_id << ", chip: " << node.chip_id << " }";

            // Add annotations
            std::vector<std::string> annotations;

            // Mesh transition indicator
            if (j < cycle.size() - 1) {
                const auto& next_node = cycle[j + 1];
                if (node.mesh_id != next_node.mesh_id) {
                    annotations.push_back("→ MESH TRANSITION →");
                }
            }

            // Mark the cycle start point
            if (j == cycle_start_idx && j < cycle.size() - 1) {
                annotations.push_back("⟲ CYCLE START");
            }

            // Mark the cycle endpoint (where path loops back)
            if (j == cycle.size() - 1) {
                annotations.push_back("⟲ CYCLE BACK TO node_" + std::to_string(cycle_start_idx));
            }

            // Print annotations
            if (!annotations.empty()) {
                fout << "  #";
                for (size_t a = 0; a < annotations.size(); ++a) {
                    if (a > 0) {
                        fout << " |";
                    }
                    fout << " " << annotations[a];
                }
            }

            fout << "\n";
        }
        fout << "\n";
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
