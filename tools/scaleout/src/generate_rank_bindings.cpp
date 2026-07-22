// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

#include "generate_rank_bindings_helpers.hpp"

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;
using namespace tt::tt_fabric;

namespace {
struct MeshGraphAndLocalMeshId {
    const MeshGraph* graph = nullptr;
    MeshId local_mesh_id{0};
};

struct TopologyMappingWithLocalMaps {
    TopologyMappingResult mapping;
    std::vector<std::map<MeshId, MeshId>> per_part_local_to_global_mesh_ids;
};

// Resolves a logical global MeshId (after merge) to the owning MeshGraph and MGD-local mesh id using the same
// per-part local -> global maps as \p merge_logical_multi_mesh_adjacency_graphs (out-parameter).
std::optional<MeshGraphAndLocalMeshId> resolve_mesh_graph_for_global_mesh_id(
    const std::vector<MeshGraph>& mesh_graphs,
    MeshId global_mesh_id,
    const std::vector<std::map<MeshId, MeshId>>& per_part_local_to_global_mesh_ids) {
    if (per_part_local_to_global_mesh_ids.empty()) {
        for (const auto& g : mesh_graphs) {
            for (const auto& mid : g.get_all_mesh_ids()) {
                if (mid == global_mesh_id) {
                    return MeshGraphAndLocalMeshId{&g, global_mesh_id};
                }
            }
        }
        return std::nullopt;
    }
    const std::size_t n = std::min(mesh_graphs.size(), per_part_local_to_global_mesh_ids.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& local_to_global = per_part_local_to_global_mesh_ids[i];
        for (MeshId local : mesh_graphs[i].get_all_mesh_ids()) {
            auto it = local_to_global.find(local);
            if (it != local_to_global.end() && it->second == global_mesh_id) {
                return MeshGraphAndLocalMeshId{&mesh_graphs[i], local};
            }
        }
    }
    return std::nullopt;
}

// OpenMPI rankfile `slot` counts per physical host. After merging several MGD extracts, each extract used its own
// per-host counter; reassign slots in global MPI rank order so consecutive global ranks on the same host get
// slot 0,1,2,...
void assign_rankfile_slots_in_global_mpi_order(std::vector<RankBindingConfig>& bindings) {
    if (bindings.empty()) {
        return;
    }
    std::sort(bindings.begin(), bindings.end(), [](const RankBindingConfig& a, const RankBindingConfig& b) {
        return a.rank < b.rank;
    });
    std::map<std::string, int> next_slot_for_host;
    for (RankBindingConfig& b : bindings) {
        b.slot = next_slot_for_host[b.hostname]++;
    }
}

// MGDs with a single mesh or with no inter-mesh (fabric) links do not define inter-mesh STRICT vs RELAXED in a
// meaningful way; skip them in the multi-MGD merge so a sibling MGD (e.g. two meshes + inter-mesh) sets
// the policy and other overlays follow that choice.
bool mesh_graph_includes_intermesh_links(const MeshGraph& g) {
    if (g.get_all_mesh_ids().size() <= 1) {
        return false;
    }
    return !g.get_requested_intermesh_connections().empty() || !g.get_requested_intermesh_ports().empty();
}
}  // namespace

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
 * @brief Run topology mapper to map logical meshes to physical ASICs
 *
 * @param mesh_graph_descriptors  Const reference to the caller's vector of loaded MGDs (the `std::vector`
 *                                is not copied—only a reference is passed). Must match `mgd_paths_in_order`
 *                                in length and order (one path per descriptor for `MeshGraph` host ranks).
 * @param mgd_paths_in_order      Const reference to paths parallel to `mesh_graph_descriptors`.
 *
 * @return Mapping result plus per-MGD local -> global mesh id maps (same order as descriptors / `MeshGraph`s).
 */
TopologyMappingWithLocalMaps run_topology_mapping(
    const PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor& pgd,
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const std::vector<std::filesystem::path>& mgd_paths_in_order) {
    if (mesh_graph_descriptors.size() != mgd_paths_in_order.size() || mesh_graph_descriptors.empty()) {
        throw std::invalid_argument(
            "run_topology_mapping: mesh_graph_descriptors and mgd_paths_in_order size must match and be non-empty");
    }

    log_info(
        tt::LogFabric, "Building physical multi-mesh adjacency graph ({} MGD(s))...", mesh_graph_descriptors.size());
    PhysicalMultiMeshGraph physical_graph = build_physical_multi_mesh_adjacency_graph(psd, pgd, mesh_graph_descriptors);

    log_info(tt::LogFabric, "Building logical multi-mesh adjacency graph from MeshGraphDescriptor(s)...");
    std::vector<LogicalMultiMeshGraph> logical_parts;
    logical_parts.reserve(mesh_graph_descriptors.size());
    for (const MeshGraphDescriptor& mgd : mesh_graph_descriptors) {
        logical_parts.push_back(build_logical_multi_mesh_adjacency_graph(mgd));
    }

    std::vector<std::map<MeshId, MeshId>> per_part_local_to_global_mesh_ids;
    const LogicalMultiMeshGraph logical_graph =
        merge_logical_multi_mesh_adjacency_graphs(logical_parts, &per_part_local_to_global_mesh_ids);

    // Print adjacency maps
    log_logical_multi_mesh_adjacency_histograms(logical_graph);
    log_physical_multi_mesh_adjacency_histograms(physical_graph);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }

    // Extract pinnings from each MGD, remapping the MGD-local mesh id to its merged global mesh id.
    for (std::size_t mgi = 0; mgi < mesh_graph_descriptors.size(); ++mgi) {
        const MeshGraphDescriptor& mgd = mesh_graph_descriptors[mgi];
        for (const auto& [pos, fabric_node] : mgd.get_pinnings()) {
            const MeshId global_mesh = per_part_local_to_global_mesh_ids.at(mgi).at(fabric_node.mesh_id);
            config.pinnings.emplace_back(pos, FabricNodeId(global_mesh, fabric_node.chip_id));
        }
    }

    auto& metal_context = MetalContext::instance();
    const auto& cluster = metal_context.get_cluster();

    if (!config.pinnings.empty()) {
        const auto& asic_descriptors = psd.get_asic_descriptors();
        for (const auto& [asic_id, _] : asic_descriptors) {
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);
            config.asic_positions[asic_id] = std::make_pair(tray_id, asic_location);
        }
    }

    std::vector<MeshGraph> mesh_graphs;
    mesh_graphs.reserve(mgd_paths_in_order.size());
    for (const auto& p : mgd_paths_in_order) {
        mesh_graphs.emplace_back(cluster, p.string());
    }

    for (std::size_t gi = 0; gi < mesh_graphs.size(); ++gi) {
        const auto& mesh_graph = mesh_graphs[gi];
        for (const auto& mid_local : mesh_graph.get_all_mesh_ids()) {
            const MeshId mid_global = per_part_local_to_global_mesh_ids.at(gi).at(mid_local);
            config.mesh_validation_modes[mid_global] = mesh_graph.is_intra_mesh_policy_relaxed(mid_local)
                                                           ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                           : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
        }
        if (!mesh_graph_includes_intermesh_links(mesh_graph)) {
            continue;
        }
        const bool relaxed = mesh_graph.is_inter_mesh_policy_relaxed();
        if (!config.inter_mesh_validation_mode.has_value()) {
            config.inter_mesh_validation_mode = relaxed ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                        : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
        } else {
            const bool existing_relaxed =
                config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
            if (existing_relaxed != relaxed) {
                throw std::runtime_error(
                    "Multi-MGD: conflicting inter-mesh validation policy between MGDs (STRICT vs RELAXED).");
            }
        }

        // Apply the same galaxy corner pinnings as the control plane (Phase 2) so Phase 1 and Phase 2 place
        // the galaxy pins identically. Full galaxies (per-host slice >= 32) pin all four corners; this mirrors
        // main's single-MGD behavior (nw_corner_only=false) under the subtorus topology solver.
        if (cluster.is_ubb_galaxy()) {
            const int world_size = static_cast<int>(
                *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->size());
            for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
                const auto& mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
                const bool is_1d = mesh_shape[0] == 1 || mesh_shape[1] == 1;
                if (!is_1d && mesh_shape.mesh_size() % 32 == 0) {
                    auto mesh_pinnings = get_galaxy_fixed_asic_position_pinnings_for_mesh(
                        mesh_id, mesh_shape, /*hard_pin_node_0=*/world_size == 1, /*nw_corner_only=*/false);
                    for (const auto& [fabric_node, positions] : mesh_pinnings) {
                        for (const auto& position : positions) {
                            config.pinnings.emplace_back(position, fabric_node);
                        }
                    }
                }
            }
        }
    }
    if (!config.inter_mesh_validation_mode.has_value()) {
        config.inter_mesh_validation_mode = ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }
    if (config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
        log_info(tt::LogFabric, "Inter-mesh validation mode: RELAXED");
    } else {
        log_info(tt::LogFabric, "Inter-mesh validation mode: STRICT");
    }

    // Topology mapping rank validation uses merged global MeshId keys. Downstream YAML/rank bindings default to
    // emitting per-MGD **local** mesh ids (generate_rank_bindings extracts with emit_local_mesh_ids default true).
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank;
    for (std::size_t gi = 0; gi < mesh_graphs.size(); ++gi) {
        const auto& mesh_graph = mesh_graphs[gi];
        for (const auto& mesh_id_local : mesh_graph.get_all_mesh_ids()) {
            const MeshId mesh_id_global = per_part_local_to_global_mesh_ids.at(gi).at(mesh_id_local);
            const auto& chip_ids = mesh_graph.get_chip_ids(mesh_id_local);
            for (const auto& [coord, chip_id] : chip_ids) {
                FabricNodeId fabric_node_id(mesh_id_global, chip_id);
                auto mesh_host_rank = mesh_graph.get_host_rank_for_chip(mesh_id_local, chip_id);
                if (mesh_host_rank.has_value()) {
                    fabric_node_id_to_mesh_rank[mesh_id_global][fabric_node_id] = mesh_host_rank.value();
                }
            }
        }
    }

    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank = {};

    log_info(tt::LogFabric, "Running topology mapping with mesh graph rank bindings...");
    TopologyMappingWithLocalMaps out;
    out.mapping = map_multi_mesh_to_physical(
        logical_graph, physical_graph, config, asic_id_to_mesh_rank, fabric_node_id_to_mesh_rank);
    out.per_part_local_to_global_mesh_ids = std::move(per_part_local_to_global_mesh_ids);
    return out;
}

/**
 * @brief Extract rank bindings from topology mapping result with topology-aware splitting.
 *
 * Bindings are one row per (mesh_id, PSD hostname, mesh_host_rank), after sorting by PSD MPI rank
 * (physical host order), then mesh_id, mesh_host_rank, hostname.
 *
 * - **Multi-process Phase 1** (several MPI ranks / PSD hostnames): each distinct hostname usually owns
 *   ASICs for a single mesh_host_rank, so the per-hostname map has one entry — behavior matches the
 *   historical `(mesh_id, hostname)` grouping.
 * - **Single-process Phase 1** (`mpirun -np 1`, e.g. bare mock cluster YAML): every ASIC shares one
 *   PSD hostname; mesh_host_rank in the key splits logical mesh hosts so MGD `host_topology` still
 *   yields the correct number of Phase 2 MPI ranks.
 *
 * Assigns contiguous ranks 0..N-1 and per-hostname slot indices for the rankfile (`binding.rank` is sequential
 * within this output). `binding.mesh_host_rank` always reflects the MGD/MeshGraph value for `(local mesh id, chip)`.
 *
 * @param emit_local_mesh_ids_for_mgd_partition  When true (default), `RankBindingConfig::mesh_id` is the **local**
 * mesh id from the MGD partition map (`per_part_local_to_global_mesh_ids[0]`). Only fabric nodes whose global logical
 * mesh id is a value in that map are included. When false, `mesh_id` is the merged/global logical mesh id (multi-graph
 * callers must pass false). Requires `mesh_graphs.size() == 1` and `per_part_local_to_global_mesh_ids.size() == 1`
 * when true.
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd,
    const TopologyMappingResult& mapping_result,
    const std::vector<MeshGraph>& mesh_graphs,
    const std::vector<std::map<MeshId, MeshId>>& per_part_local_to_global_mesh_ids,
    bool emit_local_mesh_ids_for_mgd_partition = true) {
    if (mesh_graphs.empty()) {
        throw std::invalid_argument("extract_rank_bindings: at least one MeshGraph is required");
    }
    if (emit_local_mesh_ids_for_mgd_partition) {
        if (mesh_graphs.size() != 1 || per_part_local_to_global_mesh_ids.size() != 1) {
            throw std::invalid_argument(
                "extract_rank_bindings: local mesh id mode requires exactly one MeshGraph and one local-to-global map");
        }
    }

    std::unordered_set<MeshId> partition_globals;
    std::unordered_map<MeshId, MeshId> global_to_local_mesh_for_output;
    if (emit_local_mesh_ids_for_mgd_partition) {
        for (const auto& [loc, glob] : per_part_local_to_global_mesh_ids[0]) {
            partition_globals.insert(glob);
            global_to_local_mesh_for_output[glob] = loc;
        }
    }

    struct AsicGrouping {
        std::vector<AsicID> asic_ids;
        std::vector<tt::ChipId> chip_ids;
        std::optional<MeshHostRankId> mesh_host_rank;
    };

    // mesh_id -> hostname -> mesh_host_rank -> AsicGrouping
    std::map<int, std::map<std::string, std::map<int, AsicGrouping>>> mesh_host_asics;

    // Iterate through fabric_node_to_asic mapping
    for (const auto& [fabric_node_id, asic_id] : mapping_result.fabric_node_to_asic) {
        MeshId mesh_id_global = fabric_node_id.mesh_id;
        if (!partition_globals.empty() && !partition_globals.contains(mesh_id_global)) {
            continue;
        }
        tt::ChipId chip_id_from_fabric_node = static_cast<tt::ChipId>(fabric_node_id.chip_id);

        std::optional<MeshGraphAndLocalMeshId> resolved =
            resolve_mesh_graph_for_global_mesh_id(mesh_graphs, mesh_id_global, per_part_local_to_global_mesh_ids);
        if (!resolved.has_value() || resolved->graph == nullptr) {
            log_error(
                tt::LogFabric,
                "No MeshGraph / local mesh for global logical mesh_id {} (check per-MGD local-to-global mesh id map).",
                *mesh_id_global);
            continue;
        }
        const MeshGraph* mesh_graph = resolved->graph;
        const MeshId mesh_id_local = resolved->local_mesh_id;

        std::optional<MeshHostRankId> mesh_host_rank =
            mesh_graph->get_host_rank_for_chip(mesh_id_local, chip_id_from_fabric_node);

        if (!mesh_host_rank.has_value()) {
            log_error(
                tt::LogFabric,
                "No mesh host rank found for mesh_id (global) {} and chip_id {}",
                *mesh_id_global,
                chip_id_from_fabric_node);
            continue;
        }

        std::string hostname = psd.get_host_name_for_asic(asic_id);
        tt::ChipId chip_id = psd.get_umd_unique_id(asic_id);

        const MeshId mesh_id_for_binding =
            emit_local_mesh_ids_for_mgd_partition ? global_to_local_mesh_for_output.at(mesh_id_global) : mesh_id_global;
        int mesh_id_int = static_cast<int>(*mesh_id_for_binding);
        const int mesh_host_rank_int = static_cast<int>(*mesh_host_rank.value());
        auto& bucket = mesh_host_asics[mesh_id_int][hostname][mesh_host_rank_int];
        bucket.asic_ids.push_back(asic_id);
        bucket.chip_ids.push_back(chip_id);
        bucket.mesh_host_rank = mesh_host_rank;
    }

    // Build flat list of (mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank) for canonical ordering
    // Order: PSD rank first (so output rank i matches topology mapper's mpi_rank_to_host[i]),
    // then mesh_id, mesh_host_rank, hostname - ensures alignment with physical discovery
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

        // Build TT_VISIBLE_DEVICES from ChipIds - only include MMIO devices, not remote devices.
        // For each chip, get its associated MMIO device; remote chips share an MMIO device.
        const auto& cluster = MetalContext::instance().get_cluster();
        std::set<tt::ChipId> mmio_device_ids;
        for (tt::ChipId chip_id : chip_ids) {
            tt::ChipId mmio_id = cluster.get_associated_mmio_device(chip_id);
            mmio_device_ids.insert(mmio_id);
        }
        std::string visible_devices;
        for (auto it = mmio_device_ids.begin(); it != mmio_device_ids.end(); ++it) {
            if (it != mmio_device_ids.begin()) {
                visible_devices += ",";
            }
            visible_devices += std::to_string(*it);
        }
        if (!visible_devices.empty()) {
            binding.env_overrides["TT_VISIBLE_DEVICES"] = visible_devices;
        }

        rank_bindings.push_back(binding);
    }

    return rank_bindings;
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
                ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
                Rank{r},
                k_mock_path_size_tag);
            std::vector<char> path_buf(path_size);
            if (path_size > 0) {
                distributed_context->recv(
                    ttsl::as_writable_bytes(
                        ttsl::Span<uint8_t>(reinterpret_cast<uint8_t*>(path_buf.data()), path_buf.size())),
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
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
            Rank{root_rank},
            k_mock_path_size_tag);
        if (path_size > 0) {
            std::vector<uint8_t> path_bytes(my_path.begin(), my_path.end());
            distributed_context->send(
                ttsl::as_writable_bytes(ttsl::Span<uint8_t>(path_bytes.data(), path_bytes.size())),
                Rank{root_rank},
                k_mock_path_payload_tag);
        }
    }
    distributed_context->barrier();
}

struct ProgramArgs {
    /// After parsing, sub-context id -> MGD path. `-m` is normalized to `{0: path}`; `-M` is loaded from YAML.
    std::map<int, std::filesystem::path> subcontext_id_to_mgd_path;
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
        "Provide either -m (single MGD) or -M (YAML map of subcontext_id_to_mesh_graph_descriptor). Not both.\n"
        "The Physical Grouping Descriptor (PGD) is optional and will be searched using fallback logic if not "
        "provided.");

    options.add_options()(
        "m,mesh-graph-descriptor",
        "Path to Mesh Graph Descriptor file (.textproto) — one of -m or -M is required",
        cxxopts::value<std::string>())(
        "M,mesh-graph-descriptor-mapping",
        "Path to YAML mapping: subcontext_id_to_mesh_graph_descriptor: {0: mgd0.textproto, ...} — mutually exclusive "
        "with -m",
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

        const bool has_m = result.contains("mesh-graph-descriptor");
        const bool has_M = result.contains("mesh-graph-descriptor-mapping");
        if (has_m == has_M) {
            throw std::invalid_argument(
                "Exactly one of --mesh-graph-descriptor (-m) or --mesh-graph-descriptor-mapping (-M) is required");
        }

        ProgramArgs args;
        if (has_m) {
            const std::string p = result["mesh-graph-descriptor"].as<std::string>();
            args.subcontext_id_to_mgd_path[0] = std::filesystem::path(p);
        } else {
            const std::string yaml_path = result["mesh-graph-descriptor-mapping"].as<std::string>();
            args.subcontext_id_to_mgd_path =
                load_subcontext_id_to_mesh_graph_descriptor_mapping(std::filesystem::path(yaml_path));
        }

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

    try {
        log_info(tt::LogFabric, "Generating rank bindings...");

        // Stage: Run PSD discovery
        log_info(tt::LogFabric, "Stage: Running Physical System Descriptor discovery...");
        PhysicalSystemDescriptor psd = run_psd_discovery();
        log_info(tt::LogFabric, "PSD discovery complete");

        // Gather mock cluster descriptor paths from all ranks (when TT_METAL_MOCK_CLUSTER_DESC_PATH is set)
        std::map<int, std::string> mpi_rank_to_cluster_desc_path;
        gather_mock_cluster_desc_paths(context, mpi_rank_to_cluster_desc_path);

        // Stage: Load Mesh Graph Descriptor (single MGD: either -m, or -M with one sub-context)
        std::vector<MeshGraphDescriptor> mgds;
        std::vector<std::filesystem::path> mgd_paths_in_order;
        for (const auto& [subctx_id, mgd_path] : args.subcontext_id_to_mgd_path) {
            log_info(
                tt::LogFabric,
                "Stage: Loading Mesh Graph Descriptor subcontext {} from: {}",
                subctx_id,
                mgd_path.string());
            if (!std::filesystem::exists(mgd_path) || !std::filesystem::is_regular_file(mgd_path)) {
                throw std::runtime_error("Mesh Graph Descriptor file does not exist: " + mgd_path.string());
            }
            mgds.emplace_back(MeshGraphDescriptor(mgd_path));
            mgd_paths_in_order.push_back(mgd_path);
        }

        PhysicalGroupingDescriptor pgd = find_and_load_physical_grouping_descriptor(
            args.physical_grouping_descriptor_path.has_value()
                ? std::optional<std::filesystem::path>(*args.physical_grouping_descriptor_path)
                : std::nullopt,
            &psd);

        // Get current rank - only rank 0 performs topology mapping and file generation
        auto current_rank = *context->rank();
        if (current_rank == 0) {
            // Stage: Run topology mapping
            log_info(tt::LogFabric, "Stage: Running topology mapping...");

            TopologyMappingWithLocalMaps topology = run_topology_mapping(psd, pgd, mgds, mgd_paths_in_order);

            if (!topology.mapping.success) {
                log_error(tt::LogFabric, "Topology mapping failed: {}", topology.mapping.error_message);
                return 1;
            }
            log_info(tt::LogFabric, "Topology mapping complete");

            auto& metal_context = tt::tt_metal::MetalContext::instance();
            const auto& cluster = metal_context.get_cluster();
            std::vector<MeshGraph> mesh_graphs_for_extract;
            mesh_graphs_for_extract.reserve(mgd_paths_in_order.size());
            for (const auto& p : mgd_paths_in_order) {
                mesh_graphs_for_extract.emplace_back(cluster, p.string());
            }

            std::vector<int> subcontext_ids_in_order;
            subcontext_ids_in_order.reserve(args.subcontext_id_to_mgd_path.size());
            for (const auto& [sid, _] : args.subcontext_id_to_mgd_path) {
                subcontext_ids_in_order.push_back(sid);
            }

            std::filesystem::path output_dir =
                args.output_dir.has_value() ? std::filesystem::path(*args.output_dir) : "generated/ttrun";
            std::filesystem::create_directories(output_dir);

            const bool mock_cluster_rankfile = !mpi_rank_to_cluster_desc_path.empty();

            const bool multi_mgd = mesh_graphs_for_extract.size() > 1;
            if (multi_mgd) {
                log_info(
                    tt::LogFabric,
                    "Stage: Extracting rank bindings per MGD (per-sub-context YAML + one global rankfile / phase2 "
                    "mock)...");
            } else {
                log_info(tt::LogFabric, "Stage: Extracting rank bindings...");
            }

            std::vector<RankBindingConfig> merged_global_rank_bindings;
            merged_global_rank_bindings.reserve(topology.mapping.fabric_node_to_asic.size());
            std::vector<std::pair<int, std::string>> rank_bindings_mapping_entries;
            std::vector<std::filesystem::path> written_yaml_paths;
            int global_rank_base = 0;

            for (size_t mgi = 0; mgi < mesh_graphs_for_extract.size(); ++mgi) {
                const int subctx_id = subcontext_ids_in_order[mgi];
                std::vector<MeshGraph> one_graph = {mesh_graphs_for_extract[mgi]};
                std::vector<std::map<MeshId, MeshId>> one_map = {topology.per_part_local_to_global_mesh_ids[mgi]};
                std::vector<RankBindingConfig> rank_bindings =
                    extract_rank_bindings(psd, topology.mapping, one_graph, one_map);

                if (rank_bindings.empty()) {
                    if (multi_mgd) {
                        log_info(tt::LogFabric, "Sub-context {}: extracted 0 rank binding(s)", subctx_id);
                    } else {
                        log_info(tt::LogFabric, "Extracted 0 rank binding(s)");
                    }
                } else if (multi_mgd) {
                    log_info(
                        tt::LogFabric,
                        "Sub-context {}: extracted {} rank binding(s) (global mpi rank range [{} , {}))",
                        subctx_id,
                        rank_bindings.size(),
                        global_rank_base,
                        global_rank_base + static_cast<int>(rank_bindings.size()) - 1);
                } else {
                    log_info(tt::LogFabric, "Extracted {} rank binding(s)", rank_bindings.size());
                }

                if (!multi_mgd) {
                    log_info(tt::LogFabric, "Stage: Writing rank bindings to YAML...");
                }
                const std::string mesh_desc_path_for_sub = mgd_paths_in_order[mgi].string();
                const std::filesystem::path yaml_path =
                    multi_mgd ? output_dir / fmt::format("rank_bindings_subctx_{}.yaml", subctx_id)
                              : output_dir / "rank_bindings.yaml";
                if (multi_mgd) {
                    rank_bindings_mapping_entries.emplace_back(subctx_id, yaml_path.filename().string());
                }
                write_rank_bindings_yaml(rank_bindings, mesh_desc_path_for_sub, yaml_path.string());
                written_yaml_paths.push_back(yaml_path);
                log_info(tt::LogFabric, "Successfully wrote: {}", yaml_path.string());

                for (const RankBindingConfig& b : rank_bindings) {
                    RankBindingConfig merged = b;
                    merged.rank = global_rank_base + b.rank;
                    merged_global_rank_bindings.push_back(std::move(merged));
                }
                global_rank_base += static_cast<int>(rank_bindings.size());
            }

            if (multi_mgd) {
                const std::filesystem::path mapping_path = output_dir / "rank_bindings_mapping.yaml";
                write_subcontext_rank_bindings_mapping_yaml(rank_bindings_mapping_entries, mapping_path.string());
                written_yaml_paths.push_back(mapping_path);
                log_info(tt::LogFabric, "Successfully wrote: {}", mapping_path.string());
            }

            assign_rankfile_slots_in_global_mpi_order(merged_global_rank_bindings);

            std::filesystem::path rankfile_path = output_dir / "rankfile";
            write_rankfile(merged_global_rank_bindings, rankfile_path.string(), mock_cluster_rankfile);
            if (global_rank_base == 0) {
                log_info(tt::LogFabric, "Successfully wrote: {} (empty)", rankfile_path.string());
            } else {
                log_info(
                    tt::LogFabric,
                    "Successfully wrote: {} (MPI world size {}, ranks {} .. {})",
                    rankfile_path.string(),
                    global_rank_base,
                    0,
                    global_rank_base - 1);
            }

            if (!mpi_rank_to_cluster_desc_path.empty()) {
                std::filesystem::path phase2_mock_path = output_dir / "phase2_mock_mapping.yaml";
                write_phase2_mock_mapping_yaml(
                    merged_global_rank_bindings, mpi_rank_to_cluster_desc_path, phase2_mock_path.string());
                log_info(
                    tt::LogFabric,
                    "Successfully wrote: {} (cluster descriptors used during allocation)",
                    phase2_mock_path.string());
            }

            // Flush all output files to storage before signaling peers via barrier.
            // std::ofstream::close() only drains the C++ stream buffer to the OS page cache.
            // Without fsync(), NFS peers (and local readers) may see stale or absent files
            // even after generate_rank_bindings exits.  We fsync each file and its parent
            // directory so that both data and directory entries are durable before we call
            // barrier() below — making the barrier the authoritative "writes are visible"
            // signal and allowing ttrun.py to skip any blind sleep after this subprocess.
            auto fsync_path = [](const std::filesystem::path& p) noexcept {
                int fd = ::open(p.c_str(), O_RDONLY);
                if (fd >= 0) {
                    ::fsync(fd);
                    ::close(fd);
                }
                int dir_fd = ::open(p.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
                if (dir_fd >= 0) {
                    ::fsync(dir_fd);
                    ::close(dir_fd);
                }
            };
            for (const auto& yaml_path : written_yaml_paths) {
                fsync_path(yaml_path);
            }
            fsync_path(rankfile_path);
            if (!mpi_rank_to_cluster_desc_path.empty()) {
                fsync_path(output_dir / "phase2_mock_mapping.yaml");
            }
            log_info(tt::LogFabric, "Fsynced output files; barrier will signal peers that writes are visible.");

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
