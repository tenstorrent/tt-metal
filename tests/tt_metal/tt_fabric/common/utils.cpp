// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <unordered_map>

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

        struct FabricNodeMappingEntry {
            YAML::Node chip;
            std::string hostname;
        };

        // Collect all chip mappings from all hostnames and meshes.
        // Key format: "mesh_id:chip_id" (fabric node ID) to uniquely identify each mapping.
        std::map<std::string, FabricNodeMappingEntry> gen_map_by_fabric_node_id;
        std::map<std::string, FabricNodeMappingEntry> gold_map_by_fabric_node_id;

        // Helper function to collect chips from a host entry (handles both old and new formats).
        auto collect_chips_from_host = [](const YAML::Node& host_node,
                                          std::map<std::string, FabricNodeMappingEntry>& chip_map) {
            std::string hostname;
            if (host_node["hostname"]) {
                hostname = host_node["hostname"].as<std::string>();
            }

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
                                        chip_map[key] = FabricNodeMappingEntry{YAML::Clone(chip_entry), hostname};
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
                                        if (chip_entry["fabric_node_id"]) {
                                            uint32_t mesh_id = chip_entry["fabric_node_id"]["mesh_id"].as<uint32_t>();
                                            uint32_t chip_id = chip_entry["fabric_node_id"]["chip_id"].as<uint32_t>();
                                            std::string key = std::to_string(mesh_id) + ":" + std::to_string(chip_id);
                                            chip_map[key] = FabricNodeMappingEntry{YAML::Clone(chip_entry), hostname};
                                        } else if (chip_entry["asic_id"]) {
                                            uint64_t asic_id = chip_entry["asic_id"].as<uint64_t>();
                                            std::string key = std::to_string(asic_id);
                                            chip_map[key] = FabricNodeMappingEntry{YAML::Clone(chip_entry), hostname};
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
                                    chip_map[key] = FabricNodeMappingEntry{YAML::Clone(chip_entry), hostname};
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

        // Compare the collected fabric_node_id mappings (tray placement, fabric node, ASIC ID, hostname).
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
        for (const auto& [fabric_node_key, gen_entry] : gen_map_by_fabric_node_id) {
            if (!gold_map_by_fabric_node_id.contains(fabric_node_key)) {
                // Already reported as missing above, skip detailed comparison
                continue;
            }

            const auto& gen_mapping_node = gen_entry.chip;
            const auto& gold_entry = gold_map_by_fabric_node_id[fabric_node_key];
            const auto& gold_mapping_node = gold_entry.chip;
            std::vector<std::string> chip_mismatches;

            if (gen_entry.hostname != gold_entry.hostname) {
                std::ostringstream oss;
                oss << "hostname: generated=\"" << gen_entry.hostname << "\", golden=\"" << gold_entry.hostname << "\"";
                chip_mismatches.push_back(oss.str());
            }

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

            if (gen_mapping_node["asic_id"] && gold_mapping_node["asic_id"]) {
                uint64_t gen_asic_id = gen_mapping_node["asic_id"].as<uint64_t>();
                uint64_t gold_asic_id = gold_mapping_node["asic_id"].as<uint64_t>();
                if (gen_asic_id != gold_asic_id) {
                    std::ostringstream oss;
                    oss << "asic_id: generated=" << gen_asic_id << ", golden=" << gold_asic_id;
                    chip_mismatches.push_back(oss.str());
                }
            } else if (gen_mapping_node["asic_id"] || gold_mapping_node["asic_id"]) {
                chip_mismatches.push_back("asic_id: present in one file but missing in the other");
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
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    // Skip golden file comparison only when not using mock (real devices); check against golden only in mock tests
    if (!rtoptions.get_mock_enabled()) {
        return;
    }
    std::string golden_file_base = golden_name.empty() ? test_name : golden_name;
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

    std::filesystem::path generated_file = generated_dir / generated_filename;

    // Golden path: explicit env (automapper runner), else default golden_mapping_files/<name>.yaml
    std::filesystem::path golden_file;
    if (const char* golden_path_env = std::getenv("TT_METAL_ASIC_MAPPING_GOLDEN_PATH")) {
        if (golden_path_env[0] == '\0') {
            return;
        }
        golden_file = std::filesystem::path(golden_path_env);
    } else {
        std::string golden_filename = golden_file_base + ".yaml";
        golden_file = golden_dir / golden_filename;
        if (!std::filesystem::exists(golden_file) && std::getenv("TT_METAL_ASIC_MAPPING_GOLDEN_OPTIONAL") != nullptr) {
            return;
        }
    }

    // First check if the generated file exists - if ControlPlane didn't create it, the test must fail
    if (!std::filesystem::exists(generated_file)) {
        FAIL() << "Generated ASIC mapping file does not exist: " << generated_file.string()
               << ". The ControlPlane should have created this file. Test: " << test_name << " on rank " << rank
               << ". This indicates the ControlPlane initialization did not generate the expected mapping file.";
    }

    // Regolden mode (TT_METAL_REGOLDEN=1): overwrite the golden with this run's generated mapping instead of
    // comparing. Only rank 0 (which produces rank_1's file) writes the golden, since the golden is rank 1's view
    // and all ranks must produce identical mappings. See golden_mapping_files/README.md.
    if (const char* regolden_env = std::getenv("TT_METAL_REGOLDEN");
        regolden_env != nullptr && regolden_env[0] != '\0') {
        if (rank == 0) {
            std::filesystem::create_directories(golden_file.parent_path());
            std::filesystem::copy_file(generated_file, golden_file, std::filesystem::copy_options::overwrite_existing);
            log_info(tt::LogTest, "Regoldened {} -> {}", generated_file.string(), golden_file.string());
        }
        return;
    }

    // If golden file doesn't exist, the test must fail - we need a golden file to compare against
    // This check ensures all tests have corresponding golden files and will fail immediately if missing
    if (!std::filesystem::exists(golden_file)) {
        FAIL() << "Golden file does not exist: " << golden_file.string() << ". Test: " << test_name << " on rank "
               << rank << ". See tests/tt_metal/tt_fabric/golden_mapping_files/README.md "
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

namespace {

// Host rank for this MPI process as set by tt-run via TT_MESH_HOST_RANK (rank bindings YAML).
// Mirrors ControlPlane::initialize_local_mesh_binding(): unset env defaults to host rank 0
// (single-host / mock runs without tt-run).
MeshHostRankId rank_binding_host_rank_for_local_process() {
    const char* host_rank_str = std::getenv("TT_MESH_HOST_RANK");
    if (host_rank_str != nullptr) {
        return MeshHostRankId{static_cast<unsigned int>(std::stoi(host_rank_str))};
    }
    return MeshHostRankId{0};
}

size_t mesh_graph_total_host_rank_count(const tt::tt_fabric::MeshGraph& mesh_graph) {
    size_t total = 0;
    for (const MeshId mesh_id : mesh_graph.get_mesh_ids()) {
        total += mesh_graph.get_host_ranks(mesh_id).size();
    }
    return total;
}

}  // namespace

void expect_mesh_graph_host_topology_matches_runtime(const ControlPlane& control_plane) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& topology_mapper = control_plane.get_topology_mapper();
    const size_t mpi_size =
        static_cast<size_t>(*tt::tt_metal::MetalContext::instance().full_world_distributed_context().size());
    const size_t mpi_rank =
        static_cast<size_t>(*tt::tt_metal::MetalContext::instance().full_world_distributed_context().rank());

    const auto mesh_ids = mesh_graph.get_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "MGD must define at least one mesh";

    const size_t expected_mpi_ranks = mesh_graph_total_host_rank_count(mesh_graph);
    EXPECT_EQ(mpi_size, expected_mpi_ranks)
        << "MPI world size must match total MGD host rank count (sum of host_topology dims over all mesh instances)";

    // Three independent inputs are cross-checked here:
    //  - mesh_id list: tt-run rank bindings -> TT_MESH_ID -> ControlPlane::local_mesh_binding_
    //  - expected per-host slice: MGD textproto host_topology -> MeshGraph::get_mesh_shape(mesh, host_rank)
    //  - runtime slice: PSD/PGD discovery + topology mapping -> TopologyMapper coord ranges / chip mapping
    for (const MeshId mesh_id : control_plane.get_local_mesh_id_bindings()) {
        const MeshHostRankId rank_binding_host_rank = rank_binding_host_rank_for_local_process();
        const MeshHostRankId discovered_host_rank = control_plane.get_local_host_rank_id_binding();
        EXPECT_EQ(discovered_host_rank, rank_binding_host_rank)
            << "rank " << mpi_rank << " mesh " << *mesh_id
            << " topology-mapper host rank must match tt-run rank binding (TT_MESH_HOST_RANK)";

        const MeshShape expected_local_shape = mesh_graph.get_mesh_shape(mesh_id, rank_binding_host_rank);
        ASSERT_GT(expected_local_shape.mesh_size(), 0u)
            << "rank " << mpi_rank << " mesh " << *mesh_id << " MGD per-host slice must be non-empty";

        const MeshCoordinateRange local_coord_range = control_plane.get_coord_range(mesh_id, MeshScope::LOCAL);
        EXPECT_EQ(local_coord_range.shape(), expected_local_shape)
            << "rank " << mpi_rank << " mesh " << *mesh_id
            << " local coord range shape must match MGD slice for "
               "mesh_host_rank "
            << *rank_binding_host_rank;

        const MeshShape local_physical_shape = control_plane.get_physical_mesh_shape(mesh_id, MeshScope::LOCAL);
        EXPECT_EQ(local_physical_shape, expected_local_shape)
            << "rank " << mpi_rank << " mesh " << *mesh_id << " local physical mesh shape must match MGD slice";

        const auto& local_chip_mapping = topology_mapper.get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();
        size_t mapped_chips_on_mesh = 0;
        for (const auto& [fabric_node_id, physical_chip_id] : local_chip_mapping) {
            (void)physical_chip_id;
            if (fabric_node_id.mesh_id == mesh_id) {
                ++mapped_chips_on_mesh;
            }
        }
        EXPECT_EQ(mapped_chips_on_mesh, expected_local_shape.mesh_size())
            << "rank " << mpi_rank << " mesh " << *mesh_id
            << " mapped device count must match MGD per-host chip "
               "count ("
            << expected_local_shape.mesh_size() << ")";

        const MeshShape full_mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        EXPECT_EQ(full_mesh_shape.dims(), expected_local_shape.dims())
            << "rank " << mpi_rank << " mesh " << *mesh_id << " MGD device topology dimensionality mismatch";
        EXPECT_EQ(full_mesh_shape.mesh_size() % expected_local_shape.mesh_size(), 0u)
            << "rank " << mpi_rank << " mesh " << *mesh_id
            << " full MGD mesh chip count must divide evenly by per-host slice";
    }
}

namespace {

bool rank_group_shape_is(const MeshShape& shape, uint32_t dim0, uint32_t dim1) {
    if (shape.dims() != 2) {
        return false;
    }
    return (shape[0] == dim0 && shape[1] == dim1) || (shape[0] == dim1 && shape[1] == dim0);
}

void expect_rank_group_shape_and_size(
    MeshId mesh_id, MeshHostRankId host_rank, const MeshShape& rank_shape, uint32_t dim0, uint32_t dim1) {
    EXPECT_EQ(rank_shape.dims(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank << " rank group must be 2D";
    EXPECT_TRUE(rank_group_shape_is(rank_shape, dim0, dim1))
        << "mesh " << *mesh_id << " host_rank " << *host_rank << " rank group shape must be " << dim0 << "x" << dim1
        << " or " << dim1 << "x" << dim0;
    EXPECT_EQ(rank_shape.mesh_size(), dim0 * dim1)
        << "mesh " << *mesh_id << " host_rank " << *host_rank << " rank group mesh_size";
}

}  // namespace

void expect_galaxy_rank_group_1x1_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    (void)control_plane;
    (void)mesh_id;
    (void)host_rank;
    // Pass - no rank group checks needed for 1x1 shape
}

void expect_galaxy_rank_group_1x2_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                   << " 1x2 rank group must contain exactly 2 chips";

    std::optional<uint32_t> expected_tray_id;
    std::optional<std::string> expected_hostname;
    std::optional<uint32_t> first_asic_location;
    std::optional<uint32_t> second_asic_location;
    FabricNodeId first_fabric_node_id(mesh_id, 0);
    FabricNodeId second_fabric_node_id(mesh_id, 0);

    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);
        const uint32_t asic_location_value = *psd.get_asic_location(asic_id);
        const std::string hostname = psd.get_host_name_for_asic(asic_id);

        if (!expected_hostname.has_value()) {
            expected_hostname = hostname;
        } else {
            EXPECT_EQ(hostname, *expected_hostname)
                << "mesh " << *mesh_id << " host_rank " << *host_rank
                << " 1x2 rank group fabric nodes must be on the same host; fabric node " << fabric_node_id
                << " hostname=" << hostname << " expected hostname=" << *expected_hostname;
        }

        if (!expected_tray_id.has_value()) {
            expected_tray_id = tray_id_value;
            first_asic_location = asic_location_value;
            first_fabric_node_id = fabric_node_id;
        } else {
            EXPECT_EQ(tray_id_value, *expected_tray_id)
                << "mesh " << *mesh_id << " host_rank " << *host_rank
                << " 1x2 rank group fabric nodes must be on the same tray; fabric node " << fabric_node_id
                << " tray_id=" << tray_id_value << " expected tray_id=" << *expected_tray_id;

            second_asic_location = asic_location_value;
            second_fabric_node_id = fabric_node_id;
        }
    }

    ASSERT_TRUE(first_asic_location.has_value() && second_asic_location.has_value());

    static const std::set<std::pair<uint32_t, uint32_t>> valid_asic_location_pairs = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
        {1, 5},
        {2, 6},
        {3, 7},
        {4, 8},
    };
    uint32_t loc_a = *first_asic_location;
    uint32_t loc_b = *second_asic_location;
    if (loc_a > loc_b) {
        std::swap(loc_a, loc_b);
    }
    EXPECT_TRUE(valid_asic_location_pairs.contains({loc_a, loc_b}))
        << "mesh " << *mesh_id << " host_rank " << *host_rank
        << " 1x2 rank group asic locations must be one of (1,2), (3,4), (5,6), (7,8), (1,5), (2,6), (3,7), (4,8); "
        << "got " << first_fabric_node_id << " asic_location=" << *first_asic_location << " and "
        << second_fabric_node_id << " asic_location=" << *second_asic_location;
}

void expect_galaxy_rank_group_2x2_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 4u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                   << " 2x2 rank group must contain exactly 4 chips";

    std::optional<uint32_t> expected_tray_id;
    std::optional<std::string> expected_hostname;
    std::set<uint32_t> asic_locations;

    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);
        const uint32_t asic_location_value = *psd.get_asic_location(asic_id);
        const std::string hostname = psd.get_host_name_for_asic(asic_id);

        if (!expected_hostname.has_value()) {
            expected_hostname = hostname;
        } else {
            EXPECT_EQ(hostname, *expected_hostname)
                << "mesh " << *mesh_id << " host_rank " << *host_rank
                << " 2x2 rank group fabric nodes must be on the same host; fabric node " << fabric_node_id
                << " hostname=" << hostname << " expected hostname=" << *expected_hostname;
        }

        if (!expected_tray_id.has_value()) {
            expected_tray_id = tray_id_value;
        } else {
            EXPECT_EQ(tray_id_value, *expected_tray_id)
                << "mesh " << *mesh_id << " host_rank " << *host_rank
                << " 2x2 rank group fabric nodes must be on the same tray; fabric node " << fabric_node_id
                << " tray_id=" << tray_id_value << " expected tray_id=" << *expected_tray_id;
        }
        asic_locations.insert(asic_location_value);
    }

    static const std::set<std::set<uint32_t>> valid_asic_location_groups = {
        {1, 2, 5, 6},
        {3, 4, 7, 8},
    };
    EXPECT_TRUE(valid_asic_location_groups.contains(asic_locations))
        << "mesh " << *mesh_id << " host_rank " << *host_rank
        << " 2x2 rank group asic locations must be {1,2,5,6} or {3,4,7,8}";
}

void expect_galaxy_rank_group_4x4_4x4split_check(
    const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    (void)host_rank;
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    ASSERT_EQ(mesh_shape.dims(), 2u) << "mesh " << *mesh_id << " 4x4-split rank group requires 2D device topology";
    ASSERT_TRUE(mesh_shape[0] == 4u && mesh_shape[1] == 4u)
        << "mesh " << *mesh_id << " 4x4-split layout check requires 4x4 device topology";

    static const std::set<uint32_t> all_trays = {1, 2, 3, 4};

    std::set<uint32_t> trays;
    std::set<std::string> hostnames;
    std::map<uint32_t, size_t> chips_per_tray;

    for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);

        hostnames.insert(psd.get_host_name_for_asic(asic_id));
        trays.insert(tray_id_value);
        ++chips_per_tray[tray_id_value];
    }

    EXPECT_TRUE(hostnames.size() == 1u || hostnames.size() == 2u)
        << "mesh " << *mesh_id << " 4x4-split layout must sit on one or two hosts";
    EXPECT_EQ(trays, all_trays) << "mesh " << *mesh_id << " 4x4-split layout must use trays {1,2,3,4}";
    for (uint32_t tray_id = 1; tray_id <= 4; ++tray_id) {
        EXPECT_EQ(chips_per_tray[tray_id], 4u)
            << "mesh " << *mesh_id << " 4x4-split layout tray " << tray_id << " must contain exactly 4 chips";
    }
}

void expect_galaxy_rank_group_2x4_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 8u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                   << " 2x4 rank group must contain exactly 8 chips";

    static const std::set<uint32_t> full_tray_asic_locations = {1, 2, 3, 4, 5, 6, 7, 8};
    static const std::set<uint32_t> half_tray_group_a = {1, 2, 5, 6};
    static const std::set<uint32_t> half_tray_group_b = {3, 4, 7, 8};
    static const std::set<std::set<uint32_t>> valid_half_tray_asic_location_groups = {
        half_tray_group_a, half_tray_group_b};
    static const std::set<std::set<uint32_t>> rev_c_tray_pairs = {{1, 2}, {3, 4}};
    static const std::set<std::set<uint32_t>> rev_ab_tray_pairs = {{1, 3}, {2, 4}};

    std::set<uint32_t> all_asic_locations;
    std::set<uint32_t> trays;
    std::set<std::string> hostnames;
    std::map<uint32_t, std::set<uint32_t>> asic_locations_by_tray;

    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);
        const uint32_t asic_location_value = *psd.get_asic_location(asic_id);

        hostnames.insert(psd.get_host_name_for_asic(asic_id));
        trays.insert(tray_id_value);
        all_asic_locations.insert(asic_location_value);
        asic_locations_by_tray[tray_id_value].insert(asic_location_value);
    }

    EXPECT_EQ(hostnames.size(), 1u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 2x4 rank group fabric nodes must be on the same host";

    if (trays.size() == 1u) {
        EXPECT_EQ(all_asic_locations, full_tray_asic_locations)
            << "mesh " << *mesh_id << " host_rank " << *host_rank
            << " 2x4 rank group on one tray must use asic locations {1,2,3,4,5,6,7,8}";
        return;
    }

    EXPECT_EQ(trays.size(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                << " 2x4 rank group must sit on one tray or exactly two trays";

    const auto& valid_tray_pairs = psd.is_bh_galaxy_rev_c() ? rev_c_tray_pairs : rev_ab_tray_pairs;
    EXPECT_TRUE(valid_tray_pairs.contains(trays))
        << "mesh " << *mesh_id << " host_rank " << *host_rank << " 2x4 rank group tray pair must be "
        << (psd.is_bh_galaxy_rev_c() ? "{1,2} or {3,4}" : "{1,3} or {2,4}");

    for (const auto& [tray_id, asic_locations] : asic_locations_by_tray) {
        EXPECT_EQ(asic_locations.size(), 4u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                             << " 2x4 rank group tray " << tray_id << " must contain exactly 4 chips";
    }

    // 8 chips total; all asic locations must be one half-tray group ({1,2,5,6} or {3,4,7,8}), not a mix of both.
    EXPECT_TRUE(valid_half_tray_asic_location_groups.contains(all_asic_locations))
        << "mesh " << *mesh_id << " host_rank " << *host_rank
        << " 2x4 rank group split across two trays must use asic locations {1,2,5,6} or {3,4,7,8}";

    for (const auto& [tray_id, asic_locations] : asic_locations_by_tray) {
        EXPECT_EQ(asic_locations, all_asic_locations)
            << "mesh " << *mesh_id << " host_rank " << *host_rank << " 2x4 rank group tray " << tray_id
            << " must use the same half-tray asic location group as the rank group";
    }
}

void expect_galaxy_rank_group_2x8_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 16u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 2x8 rank group must contain exactly 16 chips";

    static const std::set<std::set<uint32_t>> rev_c_tray_pairs = {{1, 3}, {2, 4}};
    static const std::set<std::set<uint32_t>> rev_ab_tray_pairs = {{1, 2}, {3, 4}};

    std::set<uint32_t> trays;
    std::set<std::string> hostnames;

    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);

        hostnames.insert(psd.get_host_name_for_asic(asic_id));
        trays.insert(tray_id_value);
    }

    EXPECT_EQ(hostnames.size(), 1u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 2x8 rank group fabric nodes must be on the same host";
    EXPECT_EQ(trays.size(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                << " 2x8 rank group must sit on exactly two trays";

    const auto& valid_tray_pairs = psd.is_bh_galaxy_rev_c() ? rev_c_tray_pairs : rev_ab_tray_pairs;
    EXPECT_TRUE(valid_tray_pairs.contains(trays))
        << "mesh " << *mesh_id << " host_rank " << *host_rank << " 2x8 rank group tray pair must be "
        << (psd.is_bh_galaxy_rev_c() ? "{1,3} or {2,4}" : "{1,2} or {3,4}");
}

void expect_galaxy_rank_group_4x4_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 16u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x4 rank group must contain exactly 16 chips";

    static const std::set<std::set<uint32_t>> rev_c_tray_pairs = {{1, 2}, {3, 4}};
    static const std::set<std::set<uint32_t>> rev_ab_tray_pairs = {{1, 3}, {2, 4}};

    std::set<uint32_t> trays;
    std::set<std::string> hostnames;

    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        const uint32_t tray_id_value = *psd.get_tray_id(asic_id);

        hostnames.insert(psd.get_host_name_for_asic(asic_id));
        trays.insert(tray_id_value);
    }

    EXPECT_EQ(hostnames.size(), 1u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x4 rank group fabric nodes must be on the same host";
    EXPECT_EQ(trays.size(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                << " 4x4 rank group must sit on exactly two trays";

    const auto& valid_tray_pairs = psd.is_bh_galaxy_rev_c() ? rev_c_tray_pairs : rev_ab_tray_pairs;
    EXPECT_TRUE(valid_tray_pairs.contains(trays))
        << "mesh " << *mesh_id << " host_rank " << *host_rank << " 4x4 rank group tray pair must be "
        << (psd.is_bh_galaxy_rev_c() ? "{1,2} or {3,4}" : "{1,3} or {2,4}");
}

void expect_galaxy_rank_group_4x8_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 32u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x8 rank group must contain exactly 32 chips";

    std::set<std::string> hostnames;
    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        hostnames.insert(psd.get_host_name_for_asic(asic_id));
    }

    EXPECT_EQ(hostnames.size(), 1u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x8 rank group fabric nodes must be on the same host";
}

void expect_galaxy_rank_group_4x16_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 64u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x16 rank group must contain exactly 64 chips";

    std::set<std::string> hostnames;
    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        hostnames.insert(psd.get_host_name_for_asic(asic_id));
    }

    EXPECT_EQ(hostnames.size(), 2u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x16 rank group fabric nodes must be on exactly two hosts";
}

void expect_galaxy_rank_group_4x32_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 128u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                     << " 4x32 rank group must contain exactly 128 chips";

    std::set<std::string> hostnames;
    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        hostnames.insert(psd.get_host_name_for_asic(asic_id));
    }

    EXPECT_EQ(hostnames.size(), 4u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 4x32 rank group fabric nodes must be on exactly four hosts";
}

void expect_galaxy_rank_group_8x16_check(const ControlPlane& control_plane, MeshId mesh_id, MeshHostRankId host_rank) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
    ASSERT_EQ(chip_ids.size(), 128u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                     << " 8x16 rank group must contain exactly 128 chips";

    std::set<std::string> hostnames;
    for (const auto chip_id : chip_ids.values()) {
        const FabricNodeId fabric_node_id(mesh_id, static_cast<std::uint32_t>(chip_id));
        const auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        hostnames.insert(psd.get_host_name_for_asic(asic_id));
    }

    EXPECT_EQ(hostnames.size(), 4u) << "mesh " << *mesh_id << " host_rank " << *host_rank
                                    << " 8x16 rank group fabric nodes must be on exactly four hosts";
}

void expect_galaxy_rank_group_checks(const ControlPlane& control_plane) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    for (const MeshId mesh_id : mesh_graph.get_mesh_ids()) {
        for (const auto& [_, host_rank] : mesh_graph.get_host_ranks(mesh_id)) {
            const MeshShape rank_shape = mesh_graph.get_mesh_shape(mesh_id, host_rank);

            if (rank_group_shape_is(rank_shape, 1, 1)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 1, 1);
                expect_galaxy_rank_group_1x1_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 1, 2)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 1, 2);
                expect_galaxy_rank_group_1x2_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 2, 2)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 2, 2);
                expect_galaxy_rank_group_2x2_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 2, 4)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 2, 4);
                expect_galaxy_rank_group_2x4_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 2, 8)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 2, 8);
                expect_galaxy_rank_group_2x8_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 4, 4)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 4, 4);
                expect_galaxy_rank_group_4x4_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 4, 8)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 4, 8);
                expect_galaxy_rank_group_4x8_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 4, 16)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 4, 16);
                expect_galaxy_rank_group_4x16_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 4, 32)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 4, 32);
                expect_galaxy_rank_group_4x32_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 8, 16)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 8, 16);
                expect_galaxy_rank_group_8x16_check(control_plane, mesh_id, host_rank);
            } else {
                ADD_FAILURE() << "mesh " << *mesh_id << " host_rank " << *host_rank << " rank group shape "
                              << rank_shape << " is not tested; please add test";
            }
        }
    }
}

void expect_galaxy_4x4_split_host_mesh_checks(const ControlPlane& control_plane) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    for (const MeshId mesh_id : mesh_graph.get_mesh_ids()) {
        if (!rank_group_shape_is(mesh_graph.get_mesh_shape(mesh_id), 4, 4)) {
            continue;
        }

        bool ran_4x4split_check = false;
        for (const auto& [_, host_rank] : mesh_graph.get_host_ranks(mesh_id)) {
            const MeshShape rank_shape = mesh_graph.get_mesh_shape(mesh_id, host_rank);

            if (rank_group_shape_is(rank_shape, 1, 1)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 1, 1);
                expect_galaxy_rank_group_1x1_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 1, 2)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 1, 2);
                expect_galaxy_rank_group_1x2_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 2, 2)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 2, 2);
                expect_galaxy_rank_group_2x2_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 2, 4)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 2, 4);
                expect_galaxy_rank_group_2x4_check(control_plane, mesh_id, host_rank);
            } else if (rank_group_shape_is(rank_shape, 4, 4)) {
                expect_rank_group_shape_and_size(mesh_id, host_rank, rank_shape, 4, 4);
                if (!ran_4x4split_check) {
                    expect_galaxy_rank_group_4x4_4x4split_check(control_plane, mesh_id, host_rank);
                    ran_4x4split_check = true;
                }
            } else {
                ADD_FAILURE() << "mesh " << *mesh_id << " host_rank " << *host_rank
                              << " split-host 4x4 layout rank shape must be 1x1, 1x2, 2x2, 2x4, or 4x4, got "
                              << rank_shape;
            }
        }
    }
}

void expect_galaxy_corner_folding_check(const ControlPlane& control_plane) {
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    for (const auto& mesh_id : mesh_graph.get_mesh_ids()) {
        const auto chip_ids_container = mesh_graph.get_chip_ids(mesh_id);
        ASSERT_GT(chip_ids_container.size(), 0u);
        const auto& vals = chip_ids_container.values();
        const auto first_chip = vals.front();
        const auto last_chip = vals.back();
        for (auto chip_id : {first_chip, last_chip}) {
            FabricNodeId fn_id(mesh_id, static_cast<std::uint32_t>(chip_id));
            auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fn_id);
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);

            EXPECT_GE(*tray_id, 1u) << "Fabric node (mesh=" << mesh_id << ", chip=" << chip_id
                                    << ") tray_id should be >= 1";
            EXPECT_LE(*tray_id, 4u) << "Fabric node (mesh=" << mesh_id << ", chip=" << chip_id
                                    << ") tray_id should be <= 4";
            EXPECT_EQ(*asic_location, 1u)
                << "Fabric node (mesh=" << mesh_id << ", chip=" << chip_id << ") asic_location should be 1";
        }
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
