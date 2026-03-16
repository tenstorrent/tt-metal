// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <yaml-cpp/yaml.h>

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;
using namespace tt::tt_fabric;

struct RankBindingConfig {
    int rank;               // Sequential MPI rank (0 to N-1, unique and contiguous) for rankfile and rank_bindings.yaml
    int psd_mpi_rank = -1;  // PSD MPI rank from discovery (used for phase2_mock_mapping.yaml lookup)
    int mesh_id;             // Mesh ID this rank belongs to
    int mesh_host_rank = 0;  // Host rank within the mesh (from MeshGraph), defaults to 0
    std::string hostname;    // Physical host for rankfile
    int slot;                // Slot number on host for rankfile (OpenMPI format)
    std::map<std::string, std::string> env_overrides;
};

/**
 * @brief Run Physical System Descriptor discovery via MPI
 *
 * This function:
 * 1. Gets the MetalContext instance (which should already be initialized)
 * 2. Runs physical discovery on all ranks using the distributed context
 * 3. Returns the PhysicalSystemDescriptor
 */
PhysicalSystemDescriptor run_psd_discovery() {
    auto& context = tt::tt_metal::MetalContext::instance();
    auto distributed_context = context.get_distributed_context_ptr();
    const auto& cluster = context.get_cluster();
    const auto& hal = context.hal();
    const auto& rtoptions = context.rtoptions();
    constexpr bool run_discovery = true;

    return tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(), distributed_context, &hal, rtoptions, run_discovery);
}

/**
 * @brief Find and load Physical Grouping Descriptor file with fallback logic
 *
 * If pgd_path is provided, use that path directly.
 * Otherwise, if TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH environment variable is set, use that path directly.
 * Otherwise, search in order:
 * 1. /data/scaleout_configs/<cluster_name>/<cluster_name>_physical_grouping_descriptor.textproto
 * 2. TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/<cluster_name>_physical_grouping_descriptor.textproto
 * 3. Architecture/cluster-type specific default:
 * tests/tt_metal/tt_fabric/physical_groupings/<arch>_<cluster_type>_physical_grouping_descriptor.textproto
 * 4. Generic default: tests/tt_metal/tt_fabric/physical_groupings/default_physical_grouping_descriptor.textproto
 *
 * Cluster name is obtained from TT_CLUSTER_NAME environment variable.
 * Architecture and cluster type are obtained from MetalContext.
 */
PhysicalGroupingDescriptor find_and_load_pgd(const std::optional<std::string>& pgd_path = std::nullopt) {
    // Check for explicit PGD path from argument first
    if (pgd_path.has_value() && !pgd_path->empty()) {
        std::filesystem::path explicit_path(*pgd_path);
        if (std::filesystem::exists(explicit_path) && std::filesystem::is_regular_file(explicit_path)) {
            log_info(
                tt::LogFabric, "Loading Physical Grouping Descriptor from provided path: {}", explicit_path.string());
            return PhysicalGroupingDescriptor(explicit_path);
        } else {
            throw std::runtime_error(
                "Physical Grouping Descriptor path provided but file does not exist: " + explicit_path.string());
        }
    }

    // Check for explicit PGD path from environment variable
    const char* pgd_path_env = std::getenv("TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH");
    if (pgd_path_env && strlen(pgd_path_env) > 0) {
        std::filesystem::path explicit_path(pgd_path_env);
        if (std::filesystem::exists(explicit_path) && std::filesystem::is_regular_file(explicit_path)) {
            log_info(
                tt::LogFabric,
                "Loading Physical Grouping Descriptor from environment variable: {}",
                explicit_path.string());
            return PhysicalGroupingDescriptor(explicit_path);
        } else {
            throw std::runtime_error(
                "TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH is set but file does not exist: " + explicit_path.string());
        }
    }

    // Get cluster name from environment variable
    const char* cluster_name_env = std::getenv("TT_CLUSTER_NAME");
    std::string cluster_name = cluster_name_env ? cluster_name_env : "";

    // Get TT_METAL_HOME for fallback paths
    const char* tt_metal_home_env = std::getenv("TT_METAL_HOME");
    std::string tt_metal_home = tt_metal_home_env ? tt_metal_home_env : ".";

    std::vector<std::filesystem::path> search_paths;

    // Path 1: /data/scaleout_configs/<cluster_name>/<cluster_name>_physical_grouping_descriptor.textproto
    if (!cluster_name.empty()) {
        std::filesystem::path data_path = std::filesystem::path("/data/scaleout_configs") / cluster_name /
                                          (cluster_name + "_physical_grouping_descriptor.textproto");
        search_paths.push_back(data_path);
    }

    // Path 2:
    // TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/<cluster_name>_physical_grouping_descriptor.textproto
    if (!cluster_name.empty()) {
        std::filesystem::path home_path = std::filesystem::path(tt_metal_home) / "tests" / "tt_metal" / "tt_fabric" /
                                          "physical_groupings" /
                                          (cluster_name + "_physical_grouping_descriptor.textproto");
        search_paths.push_back(home_path);
    }

    // Path 3: Architecture/cluster-type specific default and generic default fallback
    // Get cluster type and architecture from MetalContext and select appropriate PGD file
    std::string arch_cluster_filename;

    // Try to get cluster type and architecture, but always fall back to default
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();
    tt::ARCH arch = cluster.arch();

    // Hardcoded if-else if statement for cluster type and architecture combinations
    if (cluster_type == tt::tt_metal::ClusterType::GALAXY && arch == tt::ARCH::WORMHOLE_B0) {
        arch_cluster_filename = "wh_galaxy_physical_grouping_descriptor.textproto";
    } else if (cluster_type == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY && arch == tt::ARCH::BLACKHOLE) {
        arch_cluster_filename = "bh_galaxy_physical_grouping_descriptor.textproto";
    } else {
        arch_cluster_filename = "default_physical_grouping_descriptor.textproto";
    }

    // If we found a specific file, add it to search paths (checked before default)
    std::filesystem::path arch_cluster_path = std::filesystem::path(tt_metal_home) / "tests" / "tt_metal" /
                                              "tt_fabric" / "physical_groupings" / arch_cluster_filename;
    search_paths.push_back(arch_cluster_path);
    log_info(
        tt::LogFabric, "Will check for architecture/cluster-type specific default: {}", arch_cluster_path.string());

    // Try each path in order, but require explicit match (don't just take first existing)
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            log_info(tt::LogFabric, "Loading Physical Grouping Descriptor from: {}", path.string());
            return PhysicalGroupingDescriptor(path);
        }
    }

    // If none found, throw error
    std::string error_msg = "Could not find Physical Grouping Descriptor file. Searched:\n";
    for (const auto& path : search_paths) {
        error_msg += "  - " + path.string() + "\n";
    }
    if (!cluster_name.empty()) {
        error_msg += "Cluster name from TT_CLUSTER_NAME: " + cluster_name + "\n";
    } else {
        error_msg += "TT_CLUSTER_NAME not set\n";
    }
    throw std::runtime_error(error_msg);
}

/**
 * @brief Print logical multi-mesh adjacency map
 */
void print_logical_adjacency_map(const LogicalMultiMeshGraph& multi_mesh_graph) {
    log_info(tt::LogFabric, "Logical Multi-Mesh Adjacency Map:");

    // Print adjacency maps using topology solver's print functions (includes degree histograms)
    multi_mesh_graph.mesh_level_graph_.print_adjacency_map("Logical Mesh-Level Graph", true);
    for (const auto& [mesh_id, graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        graph.print_adjacency_map(fmt::format("Logical Mesh {} Internal Graph", mesh_id.get()), true);
    }
}

/**
 * @brief Print physical multi-mesh adjacency map
 */
void print_physical_adjacency_map(
    const PhysicalMultiMeshGraph& multi_mesh_graph, const PhysicalSystemDescriptor& physical_system_descriptor) {
    log_info(tt::LogFabric, "Physical Multi-Mesh Adjacency Map:");

    // Print adjacency maps using topology solver's print functions (includes degree histograms)
    multi_mesh_graph.mesh_level_graph_.print_adjacency_map("Physical Mesh-Level Graph", true);
    for (const auto& [mesh_id, graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        graph.print_adjacency_map(fmt::format("Physical Mesh {} Internal Graph", mesh_id.get()), true);
    }
}

/**
 * @brief Run topology mapper to map logical meshes to physical ASICs
 *
 * This function:
 * 1. Builds physical multi-mesh graph from PSD, PGD, and MGD
 * 2. Builds logical multi-mesh graph from MGD (via MeshGraph)
 * 3. Configures topology mapping with strict mode and disabled rank bindings
 * 4. Runs map_multi_mesh_to_physical
 * 5. Returns the mapping result
 */
TopologyMappingResult run_topology_mapping(
    const PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor& pgd,
    const MeshGraphDescriptor& mgd,
    const std::filesystem::path& mgd_path) {
    // Build physical multi-mesh graph from PSD, PGD, and MGD
    log_info(tt::LogFabric, "Building physical multi-mesh adjacency graph...");
    PhysicalMultiMeshGraph physical_graph = build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);

    // Build logical multi-mesh graph from MGD
    // Need to create MeshGraph from cluster and MGD path
    log_info(tt::LogFabric, "Building logical multi-mesh adjacency graph...");
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    MeshGraph mesh_graph(cluster, mgd_path.string());
    LogicalMultiMeshGraph logical_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Print adjacency maps
    print_logical_adjacency_map(logical_graph);
    print_physical_adjacency_map(physical_graph, psd);

    // Configure topology mapping
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;  // Pass the rank bindings to make sure there isn't host rank boundary issues

    // Provide hostname_to_asics from PSD so same-host constraint is applied (all ASICs on a host map to one rank)
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }

    // Extract pinnings from MGD and add to config (same as control plane)
    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [pos, fabric_node] : pinnings) {
        config.pinnings.emplace_back(pos, fabric_node);
    }

    // Build ASIC positions map (required if pinnings are used)
    if (!config.pinnings.empty()) {
        const auto& asic_descriptors = psd.get_asic_descriptors();
        for (const auto& [asic_id, _] : asic_descriptors) {
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);
            config.asic_positions[asic_id] = std::make_pair(tray_id, asic_location);
        }
    }

    // Set per-mesh validation modes based on mesh graph policy
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        config.mesh_validation_modes[mesh_id] = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id)
                                                    ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                    : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }

    // Set inter-mesh validation mode based on mesh graph policy
    // TODO: Enable per-connection inter-mesh validation mode. Currently, all inter-mesh connections
    // use the same validation mode based on the mesh graph's global inter-mesh policy. In the future,
    // we should support mixed STRICT and RELAXED policies where some inter-mesh connections are
    // device-level (strict) and others are mesh-level (relaxed).
    config.inter_mesh_validation_mode = mesh_graph.is_inter_mesh_policy_relaxed()
                                            ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                            : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    if (config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
        log_info(tt::LogFabric, "Inter-mesh validation mode: RELAXED");
    } else {
        log_info(tt::LogFabric, "Inter-mesh validation mode: STRICT");
    }

    // Build logical rank bindings from mesh graph: each fabric node gets its mesh_host_rank from the mesh graph
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        const auto& chip_ids = mesh_graph.get_chip_ids(mesh_id);
        for (const auto& [coord, chip_id] : chip_ids) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);
            auto mesh_host_rank = mesh_graph.get_host_rank_for_chip(mesh_id, chip_id);
            if (mesh_host_rank.has_value()) {
                fabric_node_id_to_mesh_rank[mesh_id][fabric_node_id] = mesh_host_rank.value();
            }
        }
    }

    // Physical rank bindings: all ASICs set to UNSET - let topology mapper assign physical ASICs to hosts
    // Build asic_id_to_mesh_rank from physical graph mesh IDs to match the physical graph structure
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank = {};

    // Run mapping with logical rank bindings from mesh graph
    log_info(tt::LogFabric, "Running topology mapping with mesh graph rank bindings...");
    TopologyMappingResult result = map_multi_mesh_to_physical(
        logical_graph, physical_graph, config, asic_id_to_mesh_rank, fabric_node_id_to_mesh_rank);

    return result;
}

/**
 * @brief Get actual hostname (replace "localhost" with resolved name)
 */
std::string get_actual_hostname(const std::string& hostname) {
    if (hostname == "localhost") {
        char buf[256];
        if (gethostname(buf, sizeof(buf)) == 0) {
            return std::string(buf);
        }
        return "localhost";
    }
    return hostname;
}

/**
 * @brief Extract rank bindings from topology mapping result with topology-aware splitting.
 *
 * Splitting rules:
 * - If meshes span multiple hosts: one process per (mesh_id, hostname) pair, each host gets
 *   at most one process per mesh. Total processes = sum over meshes of (hosts per mesh).
 * - If meshes fit on single host (num_meshes > num_hosts): each mesh gets one process with
 *   mesh_host_rank 0. Hosts are split into multiple slots. Total processes = num_meshes.
 *
 * Assigns contiguous MPI ranks 0..N-1 and (hostname, slot) for rankfile generation.
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd, const TopologyMappingResult& mapping_result, const MeshGraph& mesh_graph) {
    // Structure: mesh_id -> hostname -> {ASIC IDs, ChipIds, MeshHostRankId}
    std::map<
        int,
        std::map<std::string, std::tuple<std::vector<AsicID>, std::vector<tt::ChipId>, std::optional<MeshHostRankId>>>>
        mesh_host_asics;

    // Iterate through fabric_node_to_asic mapping
    for (const auto& [fabric_node_id, asic_id] : mapping_result.fabric_node_to_asic) {
        MeshId mesh_id = fabric_node_id.mesh_id;
        tt::ChipId chip_id_from_fabric_node = static_cast<tt::ChipId>(fabric_node_id.chip_id);

        std::optional<MeshHostRankId> mesh_host_rank =
            mesh_graph.get_host_rank_for_chip(mesh_id, chip_id_from_fabric_node);

        if (!mesh_host_rank.has_value()) {
            log_error(
                tt::LogFabric,
                "No mesh host rank found for mesh_id {} and chip_id {}",
                *mesh_id,
                chip_id_from_fabric_node);
            continue;
        }

        std::string hostname = psd.get_host_name_for_asic(asic_id);
        tt::ChipId chip_id = psd.get_umd_unique_id(asic_id);

        int mesh_id_int = static_cast<int>(*mesh_id);
        std::get<0>(mesh_host_asics[mesh_id_int][hostname]).push_back(asic_id);
        std::get<1>(mesh_host_asics[mesh_id_int][hostname]).push_back(chip_id);
        std::get<2>(mesh_host_asics[mesh_id_int][hostname]) = mesh_host_rank;
    }

    // Build flat list of (mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank) for canonical ordering
    // Order: PSD rank first (so output rank i matches topology mapper's mpi_rank_to_host[i]),
    // then mesh_id, mesh_host_rank, hostname - ensures alignment with physical discovery
    using Entry = std::tuple<int, std::string, std::vector<tt::ChipId>, int, int>;
    std::vector<Entry> entries;
    for (const auto& [mesh_id, hostname_map] : mesh_host_asics) {
        for (const auto& [hostname, asic_data] : hostname_map) {
            const auto& [asic_ids, chip_ids, mesh_host_rank] = asic_data;
            if (!mesh_host_rank.has_value()) {
                continue;
            }
            uint32_t psd_rank = 0;
            if (psd.get_host_to_rank_map().contains(hostname)) {
                psd_rank = psd.get_rank_for_hostname(hostname);
            } else {
                log_warning(
                    tt::LogFabric, "Hostname {} not in PSD host_to_rank map, using 0 for rank ordering", hostname);
            }
            entries.emplace_back(
                mesh_id, hostname, chip_ids, static_cast<int>(*mesh_host_rank.value()), static_cast<int>(psd_rank));
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        // Primary: PSD rank - output rank must match topology mapper's mpi_rank_to_host expectation
        if (std::get<4>(a) != std::get<4>(b)) {
            return std::get<4>(a) < std::get<4>(b);
        }
        // Secondary: mesh_id, mesh_host_rank, hostname for deterministic ordering
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        if (std::get<3>(a) != std::get<3>(b)) {
            return std::get<3>(a) < std::get<3>(b);
        }
        return std::get<1>(a) < std::get<1>(b);
    });

    // Assign contiguous ranks 0..N-1 and track slot per host for rankfile
    std::vector<RankBindingConfig> rank_bindings;
    std::map<std::string, int> host_slot_counters;

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& [mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank] = entries[i];

        RankBindingConfig binding;
        // binding.rank is a sequential MPI rank (0, 1, 2, ...) that must be unique and contiguous.
        // It must match: (1) the rank in rank_bindings.yaml, and (2) the rank in rankfile.
        // binding.psd_mpi_rank stores the PSD MPI rank from discovery, used for phase2_mock_mapping.yaml lookup.
        binding.rank = static_cast<int>(i);  // Sequential rank for rankfile and rank_bindings.yaml

        // Store PSD MPI rank separately for phase2_mock_mapping.yaml lookup
        if (psd.get_host_to_rank_map().contains(hostname)) {
            binding.psd_mpi_rank = static_cast<int>(psd.get_rank_for_hostname(hostname));
        } else {
            log_warning(
                tt::LogFabric, "Hostname {} not found in PSD host_to_rank_map, psd_mpi_rank will be -1", hostname);
        }
        binding.mesh_id = mesh_id;
        binding.mesh_host_rank = mesh_host_rank;
        binding.hostname = hostname;
        binding.slot = host_slot_counters[hostname]++;
        binding.env_overrides = {};

        // Build TT_VISIBLE_DEVICES from ChipIds
        std::string visible_devices;
        for (size_t j = 0; j < chip_ids.size(); ++j) {
            if (j > 0) {
                visible_devices += ",";
            }
            visible_devices += std::to_string(chip_ids[j]);
        }
        if (!visible_devices.empty()) {
            binding.env_overrides["TT_VISIBLE_DEVICES"] = visible_devices;
        }

        rank_bindings.push_back(binding);
    }

    return rank_bindings;
}

/**
 * @brief Write rank bindings to YAML file
 *
 * Includes rank, mesh_id, mesh_host_rank, and env_overrides.
 * MPI rank in bindings matches the rank in the rankfile for correct process placement.
 */
void write_rank_bindings_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::string& mesh_graph_desc_path,
    const std::string& output_file) {
    YAML::Node root;

    YAML::Node rank_bindings_node;
    for (const auto& binding : rank_bindings) {
        YAML::Node binding_node;
        binding_node["rank"] = binding.rank;
        binding_node["mesh_id"] = binding.mesh_id;
        binding_node["mesh_host_rank"] = binding.mesh_host_rank;

        if (!binding.env_overrides.empty()) {
            YAML::Node env_overrides_node;
            for (const auto& [key, value] : binding.env_overrides) {
                env_overrides_node[key] = value;
            }
            binding_node["env_overrides"] = env_overrides_node;
        }

        rank_bindings_node.push_back(binding_node);
    }

    root["rank_bindings"] = rank_bindings_node;
    root["mesh_graph_desc_path"] = mesh_graph_desc_path;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    out_file << root << std::endl;
    out_file.close();
}

/**
 * @brief Gather mock cluster descriptor paths from all MPI ranks to rank 0.
 *
 * When TT_METAL_MOCK_CLUSTER_DESC_PATH is set (mock mode), each rank has its own descriptor.
 * This collects rank -> path for use when writing phase2_mock_mapping.yaml.
 *
 * @param distributed_context MPI distributed context
 * @param mpi_rank_to_path Output map (populated only on rank 0): MPI rank -> absolute path
 */
void gather_mock_cluster_desc_paths(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& distributed_context,
    std::map<int, std::string>& mpi_rank_to_path) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr int root_rank = 0;
    auto my_rank = *distributed_context->rank();
    auto world_size = *distributed_context->size();

    const char* my_path_env = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (!my_path_env || std::strlen(my_path_env) == 0) {
        return;
    }
    std::string my_path(my_path_env);
    // Resolve to absolute path for consistent usage
    try {
        my_path = std::filesystem::absolute(std::filesystem::path(my_path)).string();
    } catch (...) {
        // Keep original if resolution fails
    }

    if (world_size == 1) {
        mpi_rank_to_path[0] = my_path;
        return;
    }

    if (my_rank == root_rank) {
        mpi_rank_to_path[root_rank] = my_path;
        for (int r = 1; r < static_cast<int>(world_size); ++r) {
            std::size_t path_size = 0;
            distributed_context->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
                Rank{r},
                Tag{100});  // Use distinct tag for mock path gather
            std::vector<char> path_buf(path_size);
            if (path_size > 0) {
                distributed_context->recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(reinterpret_cast<uint8_t*>(path_buf.data()), path_buf.size())),
                    Rank{r},
                    Tag{100});
            }
            mpi_rank_to_path[r] = std::string(path_buf.data(), path_buf.size());
        }
    } else {
        std::size_t path_size = my_path.size();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
            Rank{root_rank},
            Tag{100});
        std::vector<uint8_t> path_bytes(my_path.begin(), my_path.end());
        distributed_context->send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(path_bytes.data(), path_bytes.size())),
            Rank{root_rank},
            Tag{100});
    }
    distributed_context->barrier();
}

/**
 * @brief Write phase2_mock_mapping.yaml for Phase 2 mock cluster descriptor binding.
 *
 * Maps each sequential rank (binding.rank, matches rank_bindings.yaml) to the cluster descriptor
 * path used during allocation. The cluster descriptor path is looked up using binding.psd_mpi_rank
 * (the PSD MPI rank from discovery) from the mpi_rank_to_path map.
 *
 * Key: sequential rank (matches rank_bindings.yaml and rankfile)
 * Value: cluster descriptor path (looked up using PSD MPI rank from discovery)
 */
void write_phase2_mock_mapping_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const PhysicalSystemDescriptor& psd,
    const std::map<int, std::string>& mpi_rank_to_path,
    const std::string& output_file) {
    if (mpi_rank_to_path.empty()) {
        return;
    }

    YAML::Node root;
    YAML::Node rank_to_desc_node;

    for (const auto& binding : rank_bindings) {
        // Use binding.rank (sequential rank) as the key to match rank_bindings.yaml
        // Use binding.psd_mpi_rank to look up the cluster descriptor path from discovery
        if (binding.psd_mpi_rank < 0) {
            log_warning(
                tt::LogFabric,
                "Rank {} has invalid psd_mpi_rank ({}), skipping phase2 mock mapping",
                binding.rank,
                binding.psd_mpi_rank);
            continue;
        }
        if (!mpi_rank_to_path.contains(binding.psd_mpi_rank)) {
            log_warning(
                tt::LogFabric,
                "PSD MPI rank {} (for sequential rank {}) has no cluster descriptor path, skipping",
                binding.psd_mpi_rank,
                binding.rank);
            continue;
        }
        // Key is sequential rank (matches rank_bindings.yaml), value comes from PSD MPI rank lookup
        rank_to_desc_node[std::to_string(binding.rank)] = mpi_rank_to_path.at(binding.psd_mpi_rank);
    }

    if (rank_to_desc_node.size() == 0) {
        return;
    }

    root["rank_to_cluster_mock_cluster_desc"] = rank_to_desc_node;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file for phase2 mock mapping: " + output_file);
    }
    out_file << root << std::endl;
    out_file.close();
}

/**
 * @brief Write MPI rankfile (OpenMPI format)
 *
 * Format: rank N=hostname slot=X
 * Replaces "localhost" with actual hostname.
 *
 * The rank N in the rankfile is binding.rank, which is a sequential MPI rank (0, 1, 2, ...).
 * It must match: (1) the rank in rank_bindings.yaml, and (2) the rank key in phase2_mock_mapping.yaml.
 * Entries are sorted by rank to ensure correct MPI rank assignment.
 */
void write_rankfile(const std::vector<RankBindingConfig>& rank_bindings, const std::string& output_file) {
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open rankfile for writing: " + output_file);
    }

    // Sort by rank to ensure rankfile entries are in order (MPI expects sequential or ordered ranks)
    std::vector<RankBindingConfig> sorted_bindings = rank_bindings;
    std::sort(
        sorted_bindings.begin(), sorted_bindings.end(), [](const RankBindingConfig& a, const RankBindingConfig& b) {
            return a.rank < b.rank;
        });

    for (const auto& binding : sorted_bindings) {
        std::string hostname = get_actual_hostname(binding.hostname);
        out_file << "rank " << binding.rank << "=" << hostname << " slot=" << binding.slot << "\n";
    }

    out_file.close();
}

struct ProgramArgs {
    std::string mesh_graph_descriptor_path;
    std::optional<std::string> physical_grouping_descriptor_path;
};

/**
 * @brief Parse command line arguments
 */
ProgramArgs parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "generate_rank_bindings",
        "Generate rank bindings YAML file from Physical System Descriptor (PSD) discovery and topology mapping.\n"
        "This tool must be run with an MPI launcher (e.g., mpirun, srun).\n\n"
        "The Mesh Graph Descriptor (MGD) specifies the logical mesh topology.\n"
        "The Physical Grouping Descriptor (PGD) is optional and will be searched using fallback logic if not "
        "provided.");

    options.add_options()(
        "m,mesh-graph-descriptor",
        "Path to Mesh Graph Descriptor file (.textproto) - REQUIRED",
        cxxopts::value<std::string>())(
        "p,physical-grouping-descriptor",
        "Path to Physical Grouping Descriptor file (.textproto) - OPTIONAL",
        cxxopts::value<std::string>())("h,help", "Print usage information");

    try {
        const auto result = options.parse(argc, argv);

        if (result.count("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (!result.count("mesh-graph-descriptor")) {
            throw std::invalid_argument("--mesh-graph-descriptor (-m) is required");
        }

        ProgramArgs args;
        args.mesh_graph_descriptor_path = result["mesh-graph-descriptor"].as<std::string>();

        if (result.count("physical-grouping-descriptor")) {
            args.physical_grouping_descriptor_path = result["physical-grouping-descriptor"].as<std::string>();
        }

        return args;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    // Parse arguments first (before MPI initialization)
    ProgramArgs args = parse_arguments(argc, argv);

    // Check if MPI is initialized (i.e., running under mpirun/srun/etc.)
    // When mpirun launches the program, MPI is already initialized
    // Initialize distributed context - this will detect MPI if available
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    // Verify that we have a valid MPI context
    // Check if context supports fault tolerance (MPI contexts do, SingleHost doesn't)
    // OR check if we have multiple processes (size > 1)
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    try {
        log_info(tt::LogFabric, "Generating rank bindings...");

        // Stage: Run PSD discovery
        log_info(tt::LogFabric, "Stage: Running Physical System Descriptor discovery...");
        PhysicalSystemDescriptor psd = run_psd_discovery();
        log_info(tt::LogFabric, "PSD discovery complete");

        // Gather mock cluster descriptor paths from all ranks (when TT_METAL_MOCK_CLUSTER_DESC_PATH is set)
        std::map<int, std::string> mpi_rank_to_cluster_desc_path;
        gather_mock_cluster_desc_paths(context, mpi_rank_to_cluster_desc_path);

        // Stage: Load Mesh Graph Descriptor
        std::filesystem::path mgd_path(args.mesh_graph_descriptor_path);
        log_info(tt::LogFabric, "Stage: Loading Mesh Graph Descriptor from: {}", mgd_path.string());
        if (!std::filesystem::exists(mgd_path) || !std::filesystem::is_regular_file(mgd_path)) {
            throw std::runtime_error("Mesh Graph Descriptor file does not exist: " + mgd_path.string());
        }
        MeshGraphDescriptor mgd(mgd_path);
        log_info(tt::LogFabric, "Mesh Graph Descriptor loaded");

        // Stage: Load Physical Grouping Descriptor
        log_info(tt::LogFabric, "Stage: Loading Physical Grouping Descriptor...");
        PhysicalGroupingDescriptor pgd = find_and_load_pgd(args.physical_grouping_descriptor_path);
        log_info(tt::LogFabric, "Physical Grouping Descriptor loaded");

        // Get current rank - only rank 0 performs topology mapping and file generation
        auto current_rank = *context->rank();
        if (current_rank == 0) {
            // Stage: Run topology mapping
            log_info(tt::LogFabric, "Stage: Running topology mapping...");
            TopologyMappingResult mapping_result = run_topology_mapping(psd, pgd, mgd, mgd_path);

            if (!mapping_result.success) {
                log_error(tt::LogFabric, "Topology mapping failed: {}", mapping_result.error_message);
                return 1;
            }
            log_info(tt::LogFabric, "Topology mapping complete");

            // Stage: Extract rank bindings
            log_info(tt::LogFabric, "Stage: Extracting rank bindings...");
            // Create MeshGraph for getting host ranks
            auto& context = tt::tt_metal::MetalContext::instance();
            const auto& cluster = context.get_cluster();
            MeshGraph mesh_graph(cluster, mgd_path.string());
            std::vector<RankBindingConfig> rank_bindings = extract_rank_bindings(psd, mapping_result, mesh_graph);
            log_info(tt::LogFabric, "Extracted {} rank binding(s)", rank_bindings.size());

            // Stage: Write YAML file
            log_info(tt::LogFabric, "Stage: Writing rank bindings to YAML...");

            // Create tt-run-generated directory if it doesn't exist
            std::filesystem::path output_dir = "tt-run-generated";
            std::filesystem::create_directories(output_dir);

            std::filesystem::path output_file = output_dir / "rank_bindings.yaml";
            write_rank_bindings_yaml(rank_bindings, args.mesh_graph_descriptor_path, output_file.string());
            log_info(tt::LogFabric, "Successfully wrote: {}", output_file.string());

            std::filesystem::path rankfile_path = output_dir / "rankfile";
            write_rankfile(rank_bindings, rankfile_path.string());
            log_info(tt::LogFabric, "Successfully wrote: {}", rankfile_path.string());

            if (!mpi_rank_to_cluster_desc_path.empty()) {
                std::filesystem::path phase2_mock_path = output_dir / "phase2_mock_mapping.yaml";
                write_phase2_mock_mapping_yaml(
                    rank_bindings, psd, mpi_rank_to_cluster_desc_path, phase2_mock_path.string());
                log_info(
                    tt::LogFabric,
                    "Successfully wrote: {} (cluster descriptors used during allocation)",
                    phase2_mock_path.string());
            }

            log_info(tt::LogFabric, "Rank bindings generation complete!");
        } else {
            log_info(
                tt::LogFabric,
                "Rank {}: Skipping topology mapping and file generation (only rank 0 performs these operations)",
                current_rank);
        }

        // Synchronize all ranks before exiting
        context->barrier();

    } catch (const std::exception& e) {
        log_error(tt::LogFabric, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
