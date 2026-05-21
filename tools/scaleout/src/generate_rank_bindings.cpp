// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

#include "generate_rank_bindings_helpers.hpp"

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;
using namespace tt::tt_fabric;

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
    const auto& rtoptions = context.rtoptions();
    auto& driver_ref = const_cast<tt::umd::Cluster&>(*cluster.get_driver());

    return tt::tt_metal::run_physical_system_discovery(*driver_ref.get_cluster_description(), distributed_context, rtoptions.get_target_device());
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
        }
        TT_THROW("Physical Grouping Descriptor path provided but file does not exist: {}", explicit_path.string());
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
        }
        TT_THROW(
            "TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH is set but file does not exist: {}", explicit_path.string());
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
    } else if (cluster_type == tt::tt_metal::ClusterType::T3K && arch == tt::ARCH::WORMHOLE_B0) {
        arch_cluster_filename = "wh_t3k_physical_grouping_descriptor.textproto";
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
/** Partition used by Phase 1 solve (fabric_node_to_asic); matches PGD mesh granularity. */
std::map<MeshId, std::map<AsicID, MeshHostRankId>> build_asic_id_to_mesh_rank_from_mapping_result(
    const TopologyMappingResult& mapping_result, const MeshGraph& mesh_graph) {
    std::map<MeshId, std::map<AsicID, MeshHostRankId>> mapping;
    for (const auto& [fabric_node_id, asic_id] : mapping_result.fabric_node_to_asic) {
        MeshId mesh_id = fabric_node_id.mesh_id;
        const auto mesh_host_rank =
            mesh_graph.get_host_rank_for_chip(mesh_id, static_cast<tt::ChipId>(fabric_node_id.chip_id));
        if (!mesh_host_rank.has_value()) {
            continue;
        }
        mapping[mesh_id][asic_id] = mesh_host_rank.value();
    }
    return mapping;
}

/**
 * Phase 2 mock partition: PSD MPI rank R owns all ASICs on host R with mesh_id R (TT_MESH_ID on rank R).
 */
std::map<MeshId, std::map<AsicID, MeshHostRankId>> build_asic_id_to_mesh_rank_from_psd_mpi_hosts(
    const PhysicalSystemDescriptor& psd) {
    std::map<MeshId, std::map<AsicID, MeshHostRankId>> mapping;
    for (const auto& host : psd.get_all_hostnames()) {
        const uint32_t mpi_rank = static_cast<uint32_t>(psd.get_rank_for_hostname(host));
        MeshId mesh_id{mpi_rank};
        for (const auto& asic : psd.get_asics_connected_to_host(host)) {
            mapping[mesh_id][asic] = MeshHostRankId{0};
        }
    }
    return mapping;
}

/**
 * Simulates TopologyMapper MPI all_gather: each rank binding row contributes all ASICs on its
 * hostname to mesh_id. Multiple rows with the same mesh_id (multi-host logical mesh) accumulate.
 */
std::map<MeshId, std::map<AsicID, MeshHostRankId>> build_asic_id_to_mesh_rank_from_rank_bindings_mpi_gather(
    const PhysicalSystemDescriptor& psd, const std::vector<RankBindingConfig>& rank_bindings) {
    std::map<MeshId, std::map<AsicID, MeshHostRankId>> mapping;
    for (const auto& binding : rank_bindings) {
        MeshId mesh_id{static_cast<uint32_t>(binding.mesh_id)};
        MeshHostRankId mesh_host_rank{static_cast<unsigned int>(binding.mesh_host_rank)};
        const auto asics = psd.get_asics_connected_to_host(binding.hostname);
        for (const auto& asic : asics) {
            mapping[mesh_id][asic] = mesh_host_rank;
        }
    }
    return mapping;
}

void print_physical_adjacency_map(
    const PhysicalMultiMeshGraph& multi_mesh_graph, const PhysicalSystemDescriptor& /*physical_system_descriptor*/) {
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
    log_info(tt::LogFabric, "Building physical multi-mesh adjacency graph from PGD...");
    PhysicalMultiMeshGraph physical_graph = build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);

    log_info(tt::LogFabric, "Building logical multi-mesh adjacency graph...");
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    MeshGraph mesh_graph(cluster, mgd_path.string());
    LogicalMultiMeshGraph logical_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
        config.asic_positions[asic_id] = std::make_pair(desc.tray_id, desc.asic_location);
    }

    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [pos, fabric_node] : pinnings) {
        config.pinnings.emplace_back(pos, fabric_node);
    }

    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        config.mesh_validation_modes[mesh_id] = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id)
                                                    ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                    : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }

    config.inter_mesh_validation_mode = mesh_graph.is_inter_mesh_policy_relaxed()
                                            ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                            : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    log_info(
        tt::LogFabric,
        "Inter-mesh validation mode: {}",
        config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED ? "RELAXED"
                                                                                                        : "STRICT");

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

    // Physical rank bindings: empty — solver assigns ASICs using PGD physical graph (mock and real cluster).
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;

    // #region agent log
    ::tt::tt_metal::experimental::tt_fabric::agent_debug_ndjson(
        "D", "generate_rank_bindings.cpp:run_topology_mapping", "physical_graph_path", "{\"graph_source\":\"pgd\"}");
    // #endregion

    print_logical_adjacency_map(logical_graph);
    print_physical_adjacency_map(physical_graph, psd);

    log_info(tt::LogFabric, "Running topology mapping...");
    TopologyMappingResult result = map_multi_mesh_to_physical(
        logical_graph, physical_graph, config, asic_id_to_mesh_rank, fabric_node_id_to_mesh_rank);

    return result;
}

/**
 * @brief Extract rank bindings from topology mapping result with topology-aware splitting.
 *
 * Bindings are one row per (mesh_id, PSD hostname, mesh_host_rank), sorted by PSD MPI rank ascending so
 * binding.rank matches Phase 2 MPI rank (hostname_rank in merged PSD); then mesh_id, mesh_host_rank, hostname.
 * Inter-mesh
 * mapping in topology_mapper_utils biases logical mesh 0 toward this process's hostname and toward the physical
 * partition that contains tray id 1 / ASIC location 1 when uniquely resolvable from discovery.
 *
 * - **Multi-process Phase 1** (several MPI ranks / PSD hostnames): each distinct hostname usually owns
 *   ASICs for a single mesh_host_rank, so the per-hostname map has one entry — behavior matches the
 *   historical `(mesh_id, hostname)` grouping.
 * - **Single-process Phase 1** (`mpirun -np 1`, e.g. bare mock cluster YAML): every ASIC shares one
 *   PSD hostname; mesh_host_rank in the key splits logical mesh hosts so MGD `host_topology` still
 *   yields the correct number of Phase 2 MPI ranks.
 *
 * Assigns contiguous ranks 0..N-1 and per-hostname slot indices for the rankfile.
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd, const TopologyMappingResult& mapping_result, const MeshGraph& mesh_graph) {
    struct AsicGrouping {
        std::vector<AsicID> asic_ids;
        std::vector<tt::ChipId> chip_ids;
        std::optional<MeshHostRankId> mesh_host_rank;
    };

    // mesh_id -> hostname -> mesh_host_rank -> AsicGrouping
    std::map<int, std::map<std::string, std::map<int, AsicGrouping>>> mesh_host_asics;

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
        const int mesh_host_rank_int = static_cast<int>(*mesh_host_rank.value());
        auto& bucket = mesh_host_asics[mesh_id_int][hostname][mesh_host_rank_int];
        bucket.asic_ids.push_back(asic_id);
        bucket.chip_ids.push_back(chip_id);
        bucket.mesh_host_rank = mesh_host_rank;
    }

    // Build flat list of (mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank) for canonical ordering
    // Order after sort: mesh_id ascending (rank 0 is first/lowest mesh), then mesh_host_rank, hostname,
    // PSD rank tiebreaker — so sequential MPI rank tracks mesh topology; psd_mpi_rank field keeps PSD identity
    using Entry = std::tuple<int, std::string, std::vector<tt::ChipId>, int, int>;
    std::vector<Entry> entries;
    for (const auto& [mesh_id, hostname_map] : mesh_host_asics) {
        for (const auto& [hostname, rank_map] : hostname_map) {
            for (const auto& [_, asic_data] : rank_map) {
                const auto& chip_ids = asic_data.chip_ids;
                const auto& mesh_host_rank = asic_data.mesh_host_rank;
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
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        // Primary: PSD MPI rank so binding.rank matches Phase 2 MPI rank -> hostname_rank -> ASIC partition.
        if (std::get<4>(a) != std::get<4>(b)) {
            return std::get<4>(a) < std::get<4>(b);
        }
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
        // binding.rank must equal the PSD/MPI rank for this host so Phase 2 maps TT_MESH_ID on MPI rank R to
        // hostname_rank R in the merged PSD (not mesh_id order).
        if (psd.get_host_to_rank_map().contains(hostname)) {
            binding.psd_mpi_rank = static_cast<int>(psd.get_rank_for_hostname(hostname));
            binding.rank = binding.psd_mpi_rank;
        } else {
            log_warning(
                tt::LogFabric, "Hostname {} not found in PSD host_to_rank_map, psd_mpi_rank will be -1", hostname);
            binding.psd_mpi_rank = -1;
            binding.rank = static_cast<int>(i);
        }
        binding.mesh_id = mesh_id;
        binding.mesh_host_rank = mesh_host_rank;
        binding.hostname = hostname;
        binding.slot = host_slot_counters[hostname]++;
        binding.env_overrides = {};
        // TT_VISIBLE_DEVICES filled by gather_per_rank_visible_devices() from each rank's local mock cluster.

        rank_bindings.push_back(binding);
    }

    return rank_bindings;
}

/**
 * @brief Build TT_VISIBLE_DEVICES from the local process cluster (per-rank mock).
 */
std::string build_local_visible_devices_string() {
    const auto& cluster = MetalContext::instance().get_cluster();
    std::set<tt::ChipId> mmio_device_ids;
    for (const auto& [chip_id, _] : cluster.get_unique_chip_ids()) {
        mmio_device_ids.insert(cluster.get_associated_mmio_device(chip_id));
    }
    std::string visible_devices;
    for (auto it = mmio_device_ids.begin(); it != mmio_device_ids.end(); ++it) {
        if (it != mmio_device_ids.begin()) {
            visible_devices += ",";
        }
        visible_devices += std::to_string(*it);
    }
    return visible_devices;
}

/**
 * @brief Gather per-MPI-rank TT_VISIBLE_DEVICES from each rank's local mock cluster to rank 0.
 */
void gather_per_rank_visible_devices(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& distributed_context,
    std::vector<RankBindingConfig>& rank_bindings) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr int root_rank = 0;
    constexpr Tag k_visible_size_tag{110};
    constexpr Tag k_visible_payload_tag{111};

    auto my_rank = *distributed_context->rank();
    auto world_size = *distributed_context->size();
    std::string my_visible = build_local_visible_devices_string();

    if (world_size == 1) {
        if (!rank_bindings.empty() && !my_visible.empty()) {
            rank_bindings[0].env_overrides["TT_VISIBLE_DEVICES"] = my_visible;
        }
        return;
    }

    if (my_rank == root_rank) {
        std::map<int, std::string> rank_to_visible;
        if (!my_visible.empty()) {
            rank_to_visible[root_rank] = my_visible;
        }
        for (int r = 1; r < static_cast<int>(world_size); ++r) {
            std::size_t size = 0;
            distributed_context->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&size), sizeof(size)),
                Rank{r},
                k_visible_size_tag);
            std::vector<char> buf(size);
            if (size > 0) {
                distributed_context->recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(reinterpret_cast<uint8_t*>(buf.data()), buf.size())),
                    Rank{r},
                    k_visible_payload_tag);
            }
            std::string visible(buf.begin(), buf.end());
            if (!visible.empty()) {
                rank_to_visible[r] = std::move(visible);
            }
        }
        for (auto& binding : rank_bindings) {
            if (binding.rank < 0) {
                continue;
            }
            auto it = rank_to_visible.find(binding.rank);
            if (it != rank_to_visible.end()) {
                binding.env_overrides["TT_VISIBLE_DEVICES"] = it->second;
            }
        }
    } else {
        std::size_t size = my_visible.size();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&size), sizeof(size)),
            Rank{root_rank},
            k_visible_size_tag);
        if (size > 0) {
            std::vector<uint8_t> bytes(my_visible.begin(), my_visible.end());
            distributed_context->send(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(bytes.data(), bytes.size())),
                Rank{root_rank},
                k_visible_payload_tag);
        }
    }
    distributed_context->barrier();
}

bool asic_id_to_mesh_rank_maps_equal(
    const std::map<MeshId, std::map<AsicID, MeshHostRankId>>& a,
    const std::map<MeshId, std::map<AsicID, MeshHostRankId>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (const auto& [mesh_id, asic_map] : a) {
        if (!b.contains(mesh_id) || b.at(mesh_id) != asic_map) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Verify rank bindings match Phase 2 TopologyMapper ASIC partition (MPI rank -> host -> mesh_id).
 */
bool validate_rank_bindings_match_phase2_partition(
    const PhysicalSystemDescriptor& psd,
    const TopologyMappingResult& mapping_result,
    const MeshGraph& mesh_graph,
    const std::vector<RankBindingConfig>& rank_bindings) {
    auto from_mapping = build_asic_id_to_mesh_rank_from_mapping_result(mapping_result, mesh_graph);
    auto from_bindings = build_asic_id_to_mesh_rank_from_rank_bindings_mpi_gather(psd, rank_bindings);
    const bool equal = asic_id_to_mesh_rank_maps_equal(from_mapping, from_bindings);

    // #region agent log
    size_t mapping_asics = 0;
    for (const auto& [_, m] : from_mapping) {
        mapping_asics += m.size();
    }
    size_t bindings_asics = 0;
    for (const auto& [_, m] : from_bindings) {
        bindings_asics += m.size();
    }
    ::tt::tt_metal::experimental::tt_fabric::agent_debug_ndjson(
        "C",
        "generate_rank_bindings.cpp:validate_rank_bindings_match_phase2_partition",
        "asic_partition_equal",
        fmt::format(
            "{{\"equal\":{},\"mapping_asic_count\":{},\"bindings_asic_count\":{},\"binding_rows\":{}}}",
            equal,
            mapping_asics,
            bindings_asics,
            rank_bindings.size()));
    // #endregion

    if (!equal) {
        log_error(
            tt::LogFabric,
            "Rank bindings ASIC partition does not match mapping result (Phase 2 TopologyMapper would partition "
            "ASICs differently). Ensure each mesh's ASICs live on the PSD host for MPI rank binding.rank.");
        compare_physical_multi_mesh_graph_paths(
            "MappingResult(partition)",
            build_physical_multi_mesh_adjacency_graph(psd, from_mapping),
            "RankBindings(MPI_gather_sim)",
            build_physical_multi_mesh_adjacency_graph(psd, from_bindings));
        return false;
    }

    for (const auto& binding : rank_bindings) {
        if (binding.rank != binding.psd_mpi_rank) {
            log_error(
                tt::LogFabric,
                "Rank binding rank {} does not match psd_mpi_rank {} (hostname {}). Phase 2 maps ASICs by MPI rank, "
                "not mesh_id order.",
                binding.rank,
                binding.psd_mpi_rank,
                binding.hostname);
            return false;
        }
    }

    // Phase 2 mock: mesh_id must equal MPI rank when one mesh per rank.
    auto phase2_partition = build_asic_id_to_mesh_rank_from_psd_mpi_hosts(psd);
    if (!asic_id_to_mesh_rank_maps_equal(from_bindings, phase2_partition)) {
        log_error(
            tt::LogFabric,
            "Rank bindings partition does not match Phase 2 MPI-rank partition (mesh_id = PSD MPI rank per host).");
        return false;
    }

    return true;
}

/**
 * @brief Gather mock cluster descriptor paths from all MPI ranks to rank 0.
 *
 * When TT_METAL_MOCK_CLUSTER_DESC_PATH is set (mock mode), each rank may have its own descriptor.
 * This collects rank -> path for use when writing phase2_mock_mapping.yaml.
 *
 * For world_size > 1, every rank participates in the exchange and the final barrier even if this
 * rank's env var is unset (empty path), so rank 0 cannot block on recv while another rank returns early.
 *
 * @param distributed_context MPI distributed context
 * @param mpi_rank_to_path Output map (populated only on rank 0): MPI rank -> absolute path (non-empty paths only)
 */
void gather_mock_cluster_desc_paths(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& distributed_context,
    std::map<int, std::string>& mpi_rank_to_path) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr int root_rank = 0;
    constexpr Tag k_mock_path_size_tag{100};
    constexpr Tag k_mock_path_payload_tag{101};
    auto my_rank = *distributed_context->rank();
    auto world_size = *distributed_context->size();

    std::string my_path;
    const char* my_path_env = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (my_path_env && std::strlen(my_path_env) > 0) {
        my_path.assign(my_path_env);
        std::error_code ec;
        auto resolved = std::filesystem::absolute(std::filesystem::path(my_path), ec);
        if (!ec) {
            my_path = resolved.string();
        }
    }

    if (world_size == 1) {
        if (!my_path.empty()) {
            mpi_rank_to_path[0] = std::move(my_path);
        }
        return;
    }

    if (my_rank == root_rank) {
        if (!my_path.empty()) {
            mpi_rank_to_path[root_rank] = my_path;
        }
        for (int r = 1; r < static_cast<int>(world_size); ++r) {
            std::size_t path_size = 0;
            distributed_context->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
                Rank{r},
                k_mock_path_size_tag);
            std::vector<char> path_buf(path_size);
            if (path_size > 0) {
                distributed_context->recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(reinterpret_cast<uint8_t*>(path_buf.data()), path_buf.size())),
                    Rank{r},
                    k_mock_path_payload_tag);
            }
            std::string path(path_buf.begin(), path_buf.end());
            if (!path.empty()) {
                mpi_rank_to_path[r] = std::move(path);
            }
        }
    } else {
        std::size_t path_size = my_path.size();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
            Rank{root_rank},
            k_mock_path_size_tag);
        if (path_size > 0) {
            std::vector<uint8_t> path_bytes(my_path.begin(), my_path.end());
            distributed_context->send(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(path_bytes.data(), path_bytes.size())),
                Rank{root_rank},
                k_mock_path_payload_tag);
        }
    }
    distributed_context->barrier();
}

struct ProgramArgs {
    std::string mesh_graph_descriptor_path;
    std::optional<std::string> physical_grouping_descriptor_path;
    std::optional<std::string> output_dir;
};

/**
 * @brief Parse command line arguments
 */
ProgramArgs parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "generate_rank_bindings",
        "Generate rank bindings YAML file from Physical System Descriptor (PSD) discovery and topology mapping.\n"
        "Requires a Metal build with Open MPI (USE_MPI) enabled; run under an MPI launcher (e.g. mpirun, srun).\n"
        "Single-process runs are allowed (e.g. mpirun -np 1) when mapping a single-rank allocation.\n\n"
        "The Mesh Graph Descriptor (MGD) specifies the logical mesh topology.\n"
        "The Physical Grouping Descriptor (PGD) is optional and will be searched using fallback logic if not "
        "provided.");

    options.add_options()(
        "m,mesh-graph-descriptor",
        "Path to Mesh Graph Descriptor file (.textproto) - REQUIRED",
        cxxopts::value<std::string>())(
        "p,physical-grouping-descriptor",
        "Path to Physical Grouping Descriptor file (.textproto) - OPTIONAL",
        cxxopts::value<std::string>())(
        "o,output-dir",
        "Output directory for rank_bindings.yaml, rankfile, etc. (default: generated/ttrun)",
        cxxopts::value<std::string>())("h,help", "Print usage information");

    try {
        const auto result = options.parse(argc, argv);

        if (result.contains("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (!result.contains("mesh-graph-descriptor")) {
            throw std::invalid_argument("--mesh-graph-descriptor (-m) is required");
        }

        ProgramArgs args;
        args.mesh_graph_descriptor_path = result["mesh-graph-descriptor"].as<std::string>();

        if (result.contains("physical-grouping-descriptor")) {
            args.physical_grouping_descriptor_path = result["physical-grouping-descriptor"].as<std::string>();
        }
        if (result.contains("output-dir")) {
            args.output_dir = result["output-dir"].as<std::string>();
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

    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    const auto& context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    int exit_code = 0;
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
        std::vector<RankBindingConfig> rank_bindings;
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
            rank_bindings = extract_rank_bindings(psd, mapping_result, mesh_graph);
            log_info(tt::LogFabric, "Extracted {} rank binding(s)", rank_bindings.size());

            if (!validate_rank_bindings_match_phase2_partition(psd, mapping_result, mesh_graph, rank_bindings)) {
                exit_code = 1;
            }

            // #region agent log
            for (size_t i = 0; i < rank_bindings.size() && i < 3; ++i) {
                const auto& b = rank_bindings[i];
                ::tt::tt_metal::experimental::tt_fabric::agent_debug_ndjson(
                    "B",
                    "generate_rank_bindings.cpp:main",
                    "rank_binding_row",
                    fmt::format(
                        "{{\"rank\":{},\"mesh_id\":{},\"hostname\":\"{}\",\"psd_mpi_rank\":{},\"slot\":{}}}",
                        b.rank,
                        b.mesh_id,
                        b.hostname,
                        b.psd_mpi_rank,
                        b.slot));
            }
            // #endregion

            // Debug: compare PGD-based vs rank-binding-based physical multi-mesh graphs (phase 2 path).
            if (mesh_graph.get_all_mesh_ids().size() > 1) {
                log_info(
                    tt::LogFabric, "Stage: Comparing PGD vs rank-binding physical multi-mesh graph construction...");
                PhysicalMultiMeshGraph physical_graph_pgd = build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
                auto asic_id_from_mapping = build_asic_id_to_mesh_rank_from_mapping_result(mapping_result, mesh_graph);
                PhysicalMultiMeshGraph physical_graph_from_mapping =
                    build_physical_multi_mesh_adjacency_graph(psd, asic_id_from_mapping);
                compare_physical_multi_mesh_graph_paths(
                    "PGD(generate_rank_bindings)",
                    physical_graph_pgd,
                    "MappingResult(fabric_node_to_asic)",
                    physical_graph_from_mapping);

                auto asic_id_mpi_gather = build_asic_id_to_mesh_rank_from_rank_bindings_mpi_gather(psd, rank_bindings);
                PhysicalMultiMeshGraph physical_graph_mpi_gather =
                    build_physical_multi_mesh_adjacency_graph(psd, asic_id_mpi_gather);
                compare_physical_multi_mesh_graph_paths(
                    "PGD(generate_rank_bindings)",
                    physical_graph_pgd,
                    "RankBinding(MPI_gather_sim)",
                    physical_graph_mpi_gather);
            }

            std::filesystem::path output_dir =
                args.output_dir.has_value() ? std::filesystem::path(*args.output_dir) : "generated/ttrun";
            std::filesystem::create_directories(output_dir);

            // rank_bindings.yaml is written after gather_per_rank_visible_devices (all ranks).

            std::filesystem::path rankfile_path = output_dir / "rankfile";
            const bool mock_cluster_rankfile = !mpi_rank_to_cluster_desc_path.empty();
            write_rankfile(rank_bindings, rankfile_path.string(), mock_cluster_rankfile);
            log_info(tt::LogFabric, "Successfully wrote: {}", rankfile_path.string());

            if (!mpi_rank_to_cluster_desc_path.empty()) {
                std::filesystem::path phase2_mock_path = output_dir / "phase2_mock_mapping.yaml";
                write_phase2_mock_mapping_yaml(rank_bindings, mpi_rank_to_cluster_desc_path, phase2_mock_path.string());
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

        // Per-rank TT_VISIBLE_DEVICES from each rank's local mock cluster (Phase 2 uses the same view).
        gather_per_rank_visible_devices(context, rank_bindings);

        if (current_rank == 0 && !rank_bindings.empty() && exit_code == 0) {
            std::filesystem::path output_dir =
                args.output_dir.has_value() ? std::filesystem::path(*args.output_dir) : "generated/ttrun";
            std::filesystem::create_directories(output_dir);
            std::filesystem::path output_file = output_dir / "rank_bindings.yaml";
            write_rank_bindings_yaml(rank_bindings, args.mesh_graph_descriptor_path, output_file.string());
            log_info(tt::LogFabric, "Successfully wrote: {}", output_file.string());
        }

        // Synchronize all ranks before exiting
        context->barrier();

    } catch (const std::exception& e) {
        log_error(tt::LogFabric, "Error: {}", e.what());
        return 1;
    }

    return exit_code;
}
