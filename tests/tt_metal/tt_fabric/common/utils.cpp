// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <yaml-cpp/yaml.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <map>
#include <string>
#include <sstream>
#include <vector>

namespace tt::tt_fabric::fabric_router_tests {

// Find a device with enough neighbours in the specified direction
bool find_device_with_neighbor_in_multi_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>& dst_fabric_node_ids_by_dir,
    ChipId& src_physical_device_id,
    std::unordered_map<RoutingDirection, std::vector<ChipId>>& dst_physical_device_ids_by_dir,
    const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops,
    std::optional<RoutingDirection> incoming_direction) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    auto devices = fixture->get_devices();
    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (const auto& device : devices) {
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->get_devices()[0]->id());
        if (incoming_direction.has_value()) {
            if (control_plane.get_intra_chip_neighbors(src_fabric_node_id, incoming_direction.value()).empty()) {
                // This potential source will not have the requested incoming direction, skip
                continue;
            }
        }
        std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> temp_end_fabric_node_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<ChipId>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_fabric_node_ids = temp_end_fabric_node_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            auto curr_fabric_node_id = src_fabric_node_id;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane.get_intra_chip_neighbors(curr_fabric_node_id, routing_direction);
                if (!neighbors.empty()) {
                    temp_end_fabric_node_ids.emplace_back(curr_fabric_node_id.mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane.get_physical_chip_id_from_fabric_node_id(temp_end_fabric_node_ids.back()));
                    curr_fabric_node_id = temp_end_fabric_node_ids.back();
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            src_physical_device_id = device->get_devices()[0]->id();
            dst_fabric_node_ids_by_dir = std::move(temp_end_fabric_node_ids_by_dir);
            dst_physical_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }
    return connection_found;
}

bool find_device_with_neighbor_in_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    FabricNodeId& dst_fabric_node_id,
    ChipId& src_physical_device_id,
    ChipId& dst_physical_device_id,
    RoutingDirection direction) {
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto devices = fixture->get_devices();
    for (const auto& device : devices) {
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->get_devices()[0]->id());

        // Get neighbours within a mesh in the given direction
        auto neighbors = control_plane.get_intra_chip_neighbors(src_fabric_node_id, direction);
        if (!neighbors.empty()) {
            src_physical_device_id = device->get_devices()[0]->id();
            dst_fabric_node_id = FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]);
            dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
            return true;
        }
    }
    return false;
}

std::map<FabricNodeId, ChipId> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<EthCoord>>& mesh_graph_eth_coords) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::map<FabricNodeId, ChipId> physical_chip_ids_mapping;
    for (std::uint32_t mesh_id = 0; mesh_id < mesh_graph_eth_coords.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < mesh_graph_eth_coords[mesh_id].size(); chip_id++) {
            const auto& eth_coord = mesh_graph_eth_coords[mesh_id][chip_id];
            physical_chip_ids_mapping.insert(
                {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
        }
    }
    return physical_chip_ids_mapping;
}

bool compare_asic_mapping_files(const std::filesystem::path& generated_file, const std::filesystem::path& golden_file) {
    if (!std::filesystem::exists(generated_file)) {
        log_error(tt::LogTest, "Generated file does not exist: {}", generated_file.string());
        return false;
    }
    if (!std::filesystem::exists(golden_file)) {
        log_error(tt::LogTest, "Golden file does not exist: {}", golden_file.string());
        return false;
    }

    try {
        YAML::Node generated = YAML::LoadFile(generated_file.string());
        YAML::Node golden = YAML::LoadFile(golden_file.string());

        // Compare the YAML structures
        if (!generated["asic_to_fabric_node_mapping"] || !golden["asic_to_fabric_node_mapping"]) {
            log_error(tt::LogTest, "Missing 'asic_to_fabric_node_mapping' key in one of the files");
            return false;
        }

        auto gen_mapping = generated["asic_to_fabric_node_mapping"];
        auto gold_mapping = golden["asic_to_fabric_node_mapping"];

        if (!gen_mapping["hostnames"] || !gold_mapping["hostnames"]) {
            log_error(tt::LogTest, "Missing 'hostnames' key in one of the files");
            return false;
        }

        auto gen_hostnames = gen_mapping["hostnames"];
        auto gold_hostnames = gold_mapping["hostnames"];

        // Collect all chip mappings from all hostnames and meshes (skip hostname comparison)
        // Key format: "mesh_id:chip_id" (fabric node ID) to uniquely identify each mapping
        // Index by fabric node ID to compare tray assignments for each fabric node
        std::map<std::string, YAML::Node> gen_map_by_fabric_node_id;
        std::map<std::string, YAML::Node> gold_map_by_fabric_node_id;

        // Helper function to collect chips from a host entry (handles both old and new formats)
        // Ignores hostname - collects chips by fabric_node_id (mesh_id:chip_id) for uniqueness
        auto collect_chips_from_host = [](const YAML::Node& host_node, std::map<std::string, YAML::Node>& chip_map) {
            if (host_node.IsMap()) {
                // Check if this is the new format: map with "hostname" and "mesh" keys
                if (host_node["mesh"]) {
                    // New format: single host entry with mesh list
                    // Format: hostname: X
                    //         mesh:
                    //           - mesh: 0
                    //             chips:
                    //               - umd_chip_id: 0
                    auto mesh_list = host_node["mesh"];

                    // Iterate through mesh list - entries can have both mesh and chips keys in the same entry
                    // Format: - mesh: 0
                    //           chips: [...]
                    for (const auto& entry : mesh_list) {
                        if (entry.IsMap()) {
                            // Process chips if chips key exists (can be in same entry as mesh)
                            if (entry["chips"]) {
                                auto chips_list = entry["chips"];
                                for (const auto& chip_entry : chips_list) {
                                    if (chip_entry["fabric_node_id"]) {
                                        uint32_t mesh_id = chip_entry["fabric_node_id"]["mesh_id"].as<uint32_t>();
                                        uint32_t chip_id = chip_entry["fabric_node_id"]["chip_id"].as<uint32_t>();
                                        std::string key = std::to_string(mesh_id) + ":" + std::to_string(chip_id);
                                        chip_map[key] = YAML::Clone(chip_entry);
                                    }
                                }
                            }
                        }
                    }
                    return;  // Processed new format, exit early
                }

                // Old format handling below
                // Old format: could be map of mesh_X -> map of chip_id -> entry
                // OR map of hostname -> sequence of mesh objects with chips
                // Check if it's the old sequence format first
                bool is_old_sequence_format = false;
                for (const auto& entry_pair : host_node) {
                    auto entry_value = entry_pair.second;
                    if (entry_value.IsSequence()) {
                        // Old format: hostname -> sequence of mesh objects
                        is_old_sequence_format = true;
                        for (const auto& mesh_entry : entry_value) {
                            if (mesh_entry.IsMap() && mesh_entry["mesh"]) {
                                if (mesh_entry["chips"]) {
                                    auto chips_list = mesh_entry["chips"];
                                    for (const auto& chip_entry : chips_list) {
                                        if (chip_entry["asic_id"]) {
                                            uint64_t asic_id = chip_entry["asic_id"].as<uint64_t>();
                                            std::string key = std::to_string(asic_id);
                                            chip_map[key] = YAML::Clone(chip_entry);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }

                if (!is_old_sequence_format) {
                    // Old map format: map of mesh_X -> map of chip_id -> entry
                    for (const auto& mesh_pair : host_node) {
                        std::string mesh_key = mesh_pair.first.as<std::string>();
                        if (!mesh_key.starts_with("mesh_")) {
                            continue;  // Skip non-mesh keys
                        }
                        auto mesh_value = mesh_pair.second;
                        if (mesh_value.IsMap()) {
                            // Old map format: chip_id -> entry
                            for (const auto& chip_pair : mesh_value) {
                                auto chip_entry = chip_pair.second;
                                if (chip_entry["fabric_node_id"]) {
                                    uint32_t mesh_id = chip_entry["fabric_node_id"]["mesh_id"].as<uint32_t>();
                                    uint32_t chip_id = chip_entry["fabric_node_id"]["chip_id"].as<uint32_t>();
                                    std::string key = std::to_string(mesh_id) + ":" + std::to_string(chip_id);
                                    chip_map[key] = YAML::Clone(chip_entry);
                                }
                            }
                        }
                    }
                }
            }
        };

        // Collect from generated file
        if (gen_hostnames.IsSequence()) {
            // New format: hostnames is a sequence
            for (const auto& host_entry : gen_hostnames) {
                collect_chips_from_host(host_entry, gen_map_by_fabric_node_id);
            }
        } else if (gen_hostnames.IsMap()) {
            // Old format: hostnames is a map
            for (const auto& host_pair : gen_hostnames) {
                collect_chips_from_host(host_pair.second, gen_map_by_fabric_node_id);
            }
        }

        // Collect from golden file
        if (gold_hostnames.IsSequence()) {
            // New format: hostnames is a sequence
            for (const auto& host_entry : gold_hostnames) {
                collect_chips_from_host(host_entry, gold_map_by_fabric_node_id);
            }
        } else if (gold_hostnames.IsMap()) {
            // Old format: hostnames is a map
            for (const auto& host_pair : gold_hostnames) {
                collect_chips_from_host(host_pair.second, gold_map_by_fabric_node_id);
            }
        }

        // Compare the collected fabric_node_id mappings
        // Compare asic_position (tray_id + asic_location) and fabric_node_id (mesh_id + chip_id), ignore ASIC IDs
        std::vector<std::string> mismatch_details;

        // First, check that we have the same number of entries
        if (gen_map_by_fabric_node_id.size() != gold_map_by_fabric_node_id.size()) {
            std::ostringstream oss;
            oss << "Mismatch in total number of unique fabric node entries: generated="
                << gen_map_by_fabric_node_id.size() << ", golden=" << gold_map_by_fabric_node_id.size();
            mismatch_details.push_back(oss.str());
        }

        // Find missing fabric node entries (entries in generated but not in golden)
        std::vector<std::string> missing_in_golden;
        for (const auto& [fabric_node_key, _] : gen_map_by_fabric_node_id) {
            if (!gold_map_by_fabric_node_id.contains(fabric_node_key)) {
                missing_in_golden.push_back(fabric_node_key);
            }
        }
        if (!missing_in_golden.empty()) {
            std::ostringstream oss2;
            oss2 << "Fabric node entries (mesh_id:chip_id) present in generated but missing in golden: ";
            for (size_t i = 0; i < missing_in_golden.size(); ++i) {
                if (i > 0) {
                    oss2 << ", ";
                }
                oss2 << missing_in_golden[i];
            }
            mismatch_details.push_back(oss2.str());
        }

        // Find missing fabric node entries (entries in golden but not in generated)
        std::vector<std::string> missing_in_generated;
        for (const auto& [fabric_node_key, _] : gold_map_by_fabric_node_id) {
            if (!gen_map_by_fabric_node_id.contains(fabric_node_key)) {
                missing_in_generated.push_back(fabric_node_key);
            }
        }
        if (!missing_in_generated.empty()) {
            std::ostringstream oss2;
            oss2 << "Fabric node entries (mesh_id:chip_id) present in golden but missing in generated: ";
            for (size_t i = 0; i < missing_in_generated.size(); ++i) {
                if (i > 0) {
                    oss2 << ", ";
                }
                oss2 << missing_in_generated[i];
            }
            mismatch_details.push_back(oss2.str());
        }

        // Compare all entries that exist in both files
        // Compare asic_position (tray_id + asic_location) and fabric_node_id, ignore ASIC IDs and other fields
        for (const auto& [fabric_node_key, gen_mapping_node] : gen_map_by_fabric_node_id) {
            if (!gold_map_by_fabric_node_id.contains(fabric_node_key)) {
                // Already reported as missing above, skip detailed comparison
                continue;
            }

            auto gold_mapping_node = gold_map_by_fabric_node_id[fabric_node_key];
            std::vector<std::string> chip_mismatches;

            // Compare tray_id
            uint32_t gen_tray_id = gen_mapping_node["asic_position"]["tray_id"].as<uint32_t>();
            uint32_t gold_tray_id = gold_mapping_node["asic_position"]["tray_id"].as<uint32_t>();
            if (gen_tray_id != gold_tray_id) {
                std::ostringstream oss;
                oss << "tray_id: generated=" << gen_tray_id << ", golden=" << gold_tray_id;
                chip_mismatches.push_back(oss.str());
            }

            // Compare asic_location
            uint32_t gen_asic_location = gen_mapping_node["asic_position"]["asic_location"].as<uint32_t>();
            uint32_t gold_asic_location = gold_mapping_node["asic_position"]["asic_location"].as<uint32_t>();
            if (gen_asic_location != gold_asic_location) {
                std::ostringstream oss;
                oss << "asic_location: generated=" << gen_asic_location << ", golden=" << gold_asic_location;
                chip_mismatches.push_back(oss.str());
            }

            // Compare fabric_node_id (mesh_id + chip_id) - should match since we're indexing by it
            uint32_t gen_mesh_id = gen_mapping_node["fabric_node_id"]["mesh_id"].as<uint32_t>();
            uint32_t gold_mesh_id = gold_mapping_node["fabric_node_id"]["mesh_id"].as<uint32_t>();
            if (gen_mesh_id != gold_mesh_id) {
                std::ostringstream oss;
                oss << "fabric_node_id.mesh_id: generated=" << gen_mesh_id << ", golden=" << gold_mesh_id;
                chip_mismatches.push_back(oss.str());
            }

            uint32_t gen_chip_id = gen_mapping_node["fabric_node_id"]["chip_id"].as<uint32_t>();
            uint32_t gold_chip_id = gold_mapping_node["fabric_node_id"]["chip_id"].as<uint32_t>();
            if (gen_chip_id != gold_chip_id) {
                std::ostringstream oss;
                oss << "fabric_node_id.chip_id: generated=" << gen_chip_id << ", golden=" << gold_chip_id;
                chip_mismatches.push_back(oss.str());
            }

            if (!chip_mismatches.empty()) {
                std::ostringstream oss;
                oss << "Fabric node entry " << fabric_node_key << " (mesh_id:chip_id): ";
                for (size_t i = 0; i < chip_mismatches.size(); ++i) {
                    if (i > 0) {
                        oss << ", ";
                    }
                    oss << chip_mismatches[i];
                }
                mismatch_details.push_back(oss.str());
            }
        }

        if (!mismatch_details.empty()) {
            log_error(tt::LogTest, "Fabric node to tray ID mapping mismatches detected:");
            for (const auto& detail : mismatch_details) {
                log_error(tt::LogTest, "  {}", detail);
            }
            log_error(
                tt::LogTest,
                "\n"
                "================================================================================\n"
                "WARNING: Topology mapping mismatch detected!\n"
                "================================================================================\n"
                "The generated fabric-node-to-tray mappings do not match the golden reference.\n"
                "This indicates a change in how fabric nodes are mapped to tray IDs.\n"
                "\n"
                "IMPORTANT: Changing topology bindings/pinnings is a MAJOR change that can:\n"
                "  - Cause topology errors in other topologies\n"
                "  - Break existing deployments\n"
                "  - Require coordination across multiple teams\n"
                "\n"
                "BEFORE regenerating golden files:\n"
                "  1. Verify this is an intentional change, not a regression\n"
                "  2. Approval from topology users and scaleout team + Umair Cheema, Aditya Saigal, Allan Liu, Joseph "
                "Chu, Ridvan Song\n"
                "  3. Ensure all affected topologies are tested\n"
                "\n"
                "To regenerate golden files manually (ONLY after approval):\n"
                "  See detailed instructions in: tests/tt_metal/tt_fabric/golden_mapping_files/README.md\n"
                "================================================================================\n");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Exception while comparing files: {}", e.what());
        return false;
    }
}

void check_asic_mapping_against_golden(const std::string& test_name, const std::string& golden_name) {
    std::string golden_file_base = golden_name.empty() ? test_name : golden_name;
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    int world_size = *distributed_context->size();
    int rank = *distributed_context->rank();

    std::filesystem::path root_dir = rtoptions.get_root_dir();
    std::filesystem::path generated_dir = root_dir / "generated" / "fabric";
    std::filesystem::path golden_dir = root_dir / "tests" / "tt_metal" / "tt_fabric" / "golden_mapping_files";

    // Check this rank's generated file
    // Ranks are 0-based in distributed_context, but files use 1-based indexing
    std::string generated_filename =
        "asic_to_fabric_node_mapping_rank_" + std::to_string(rank + 1) + "_of_" + std::to_string(world_size) + ".yaml";
    std::string golden_filename = golden_file_base + ".yaml";

    std::filesystem::path generated_file = generated_dir / generated_filename;
    std::filesystem::path golden_file = golden_dir / golden_filename;

    // First check if the generated file exists - if ControlPlane didn't create it, the test must fail
    if (!std::filesystem::exists(generated_file)) {
        FAIL() << "Generated ASIC mapping file does not exist: " << generated_file.string()
               << ". The ControlPlane should have created this file. Test: " << test_name << " on rank " << rank
               << ". This indicates the ControlPlane initialization did not generate the expected mapping file.";
    }

    // If golden file doesn't exist, the test must fail - we need a golden file to compare against
    // This check ensures all tests have corresponding golden files and will fail immediately if missing
    if (!std::filesystem::exists(golden_file)) {
        FAIL() << "Golden file does not exist: " << golden_file.string()
               << ". Expected golden file name: " << golden_filename << ". Test: " << test_name << " on rank " << rank
               << ". See tests/tt_metal/tt_fabric/golden_mapping_files/README.md "
               << "for instructions on generating golden files. "
               << "The test cannot proceed without a golden reference file to compare against.";
    }

    bool comparison_result = compare_asic_mapping_files(generated_file, golden_file);
    EXPECT_TRUE(comparison_result) << "ASIC mapping file mismatch for test " << test_name
                                   << " (golden: " << golden_file_base << ") on rank " << rank;
    if (!comparison_result) {
        FAIL() << "ASIC mapping file mismatch detected on rank " << rank
               << ". Test must fail when mappings don't match golden reference.";
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
