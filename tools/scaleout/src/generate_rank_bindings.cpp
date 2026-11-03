// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>
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
#include "mesh_pinning_file.hpp"

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;
using namespace tt::tt_fabric;

namespace {
using MeshHostRankKey = std::pair<MeshId, MeshHostRankId>;

struct TopologyMappingParts {
    std::vector<TopologyMappingResult> parts;
    // (mesh_id, mesh_host_rank) -> the hostname + TT_VISIBLE_DEVICES placement chosen from the
    // --mesh-pinning-file. Empty when no pinning file was given. Keyed by the single MGD's local ids.
    std::map<MeshHostRankKey, HostPlacement> selected_mesh_host_rank_pinnings;
};

struct ResolvedHostPlacement {
    HostPlacement host_placement;
    std::set<tt::tt_metal::experimental::PhysicalNodeId> asics;
};

struct ResolvedMeshHostRankPinningGroup {
    std::vector<MeshHostRankKey> mesh_host_ranks;
    std::vector<ResolvedHostPlacement> host_placements;
};

std::string format_visible_devices(const std::vector<int>& devices) {
    std::string value;
    for (const int device : devices) {
        if (!value.empty()) {
            value += ',';
        }
        value += std::to_string(device);
    }
    return value;
}

// Keyed by address, matching the physical graph the solver is handed.
std::map<std::string, std::set<tt::tt_metal::experimental::PhysicalNodeId>> hostname_to_physical_nodes(
    const PhysicalSystemDescriptor& psd) {
    std::map<std::string, std::set<tt::tt_metal::experimental::PhysicalNodeId>> hostname_to_asics;
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        hostname_to_asics[desc.host_name].insert(desc.physical_node_id);
    }
    return hostname_to_asics;
}

// Turn each (hostname, TT_VISIBLE_DEVICES) placement of the pinning file into the exact set of physical
// nodes it exposes, and check every logical (mesh_id, mesh_host_rank) it names exists in the MGD with the
// same node count. Single-MGD only, so the file's mesh ids are the solver's mesh ids.
std::vector<ResolvedMeshHostRankPinningGroup> resolve_mesh_host_rank_pinnings(
    const PhysicalSystemDescriptor& psd,
    const MeshGraph& mesh_graph,
    const std::map<std::string, std::set<tt::tt_metal::experimental::PhysicalNodeId>>& hostname_to_asics,
    const std::vector<MeshHostRankPinningGroup>& pinning_groups) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto all_mesh_ids = mesh_graph.get_all_mesh_ids();
    std::vector<ResolvedMeshHostRankPinningGroup> resolved_groups;
    resolved_groups.reserve(pinning_groups.size());

    for (const MeshHostRankPinningGroup& group : pinning_groups) {
        ResolvedMeshHostRankPinningGroup resolved_group;
        std::vector<std::size_t> logical_node_counts;
        for (const MeshHostRankKey& key : group.mesh_host_ranks) {
            if (std::find(all_mesh_ids.begin(), all_mesh_ids.end(), key.first) == all_mesh_ids.end()) {
                throw std::runtime_error(fmt::format(
                    "Mesh pinning file references mesh_id {}, which the MGD does not define", key.first.get()));
            }
            std::size_t logical_node_count = 0;
            for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(key.first)) {
                const auto host_rank = mesh_graph.get_host_rank_for_chip(key.first, chip_id);
                if (host_rank.has_value() && host_rank.value() == key.second) {
                    ++logical_node_count;
                }
            }
            if (logical_node_count == 0) {
                throw std::runtime_error(fmt::format(
                    "Mesh pinning file references mesh_id {} mesh_host_rank {}, which the MGD does not define",
                    key.first.get(),
                    key.second.get()));
            }
            logical_node_counts.push_back(logical_node_count);
            resolved_group.mesh_host_ranks.push_back(key);
        }

        for (const HostPlacement& placement : group.host_placements) {
            const auto host_it = hostname_to_asics.find(placement.hostname);
            if (host_it == hostname_to_asics.end()) {
                std::string discovered_hosts;
                for (const auto& [hostname, _] : hostname_to_asics) {
                    if (!discovered_hosts.empty()) {
                        discovered_hosts += ", ";
                    }
                    discovered_hosts += hostname;
                }
                throw std::runtime_error(fmt::format(
                    "Mesh pinning file references host '{}', which physical discovery did not find. Discovered "
                    "hosts: [{}]",
                    placement.hostname,
                    discovered_hosts));
            }

            const std::set<int> requested_devices(
                placement.tt_visible_devices.begin(), placement.tt_visible_devices.end());
            std::set<int> discovered_devices;
            ResolvedHostPlacement resolved_placement{placement, {}};
            for (const auto& physical_node_id : host_it->second) {
                const tt::ChipId chip_id = psd.get_umd_unique_id(psd.get_asic_id(physical_node_id));
                const int mmio_device = static_cast<int>(cluster.get_associated_mmio_device(chip_id));
                discovered_devices.insert(mmio_device);
                if (requested_devices.contains(mmio_device)) {
                    resolved_placement.asics.insert(physical_node_id);
                }
            }
            for (const int requested_device : requested_devices) {
                if (!discovered_devices.contains(requested_device)) {
                    throw std::runtime_error(fmt::format(
                        "Mesh pinning file requests TT_VISIBLE_DEVICES={} on host '{}', but device {} was not "
                        "found there",
                        format_visible_devices(placement.tt_visible_devices),
                        placement.hostname,
                        requested_device));
                }
            }
            for (std::size_t logical_node_count : logical_node_counts) {
                if (resolved_placement.asics.size() != logical_node_count) {
                    throw std::runtime_error(fmt::format(
                        "Constraint validation failed: host placement {} with TT_VISIBLE_DEVICES={} exposes {} "
                        "ASICs, but every logical mesh-host-rank in its many-to-many group must contain exactly {} "
                        "nodes",
                        placement.hostname,
                        format_visible_devices(placement.tt_visible_devices),
                        resolved_placement.asics.size(),
                        logical_node_count));
                }
            }
            resolved_group.host_placements.push_back(std::move(resolved_placement));
        }
        resolved_groups.push_back(std::move(resolved_group));
    }
    return resolved_groups;
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

    return tt::tt_metal::run_physical_system_discovery(
        *driver_ref.get_cluster_description(), distributed_context, rtoptions.get_target_device());
}

/**
 * @brief Build the shared MultiMeshSolutionEnumerator for single- and multi-solution mapping.
 *
 * Configures topology mapping (validation modes, MGD + galaxy-corner pinnings, mesh-rank bindings)
 * and constructs the enumerator once. Callers pull solutions with next(): the single-solution path
 * takes one value; --all-solutions keeps this enumerator and loops next() on the same session.
 */
MultiMeshSolutionEnumerator make_topology_mapping_enumerator(
    const PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor* pgd,
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const std::vector<std::filesystem::path>& mgd_paths_in_order,
    bool unique_shapes,
    const std::map<MeshHostRankKey, std::set<tt::tt_metal::experimental::PhysicalNodeId>>& mesh_host_rank_pinnings =
        {}) {
    if (mesh_graph_descriptors.size() != mgd_paths_in_order.size() || mesh_graph_descriptors.empty()) {
        throw std::invalid_argument(
            "make_topology_mapping_enumerator: mesh_graph_descriptors and mgd_paths_in_order size must match and be "
            "non-empty");
    }

    auto& metal_context = MetalContext::instance();
    const auto& cluster = metal_context.get_cluster();

    // MeshGraphs are for galaxy-corner pin shapes, inter-mesh policy, and fabric-node host ranks.
    // Logical adjacency and local->global MeshId remap are owned by the enumerator.
    std::vector<MeshGraph> mesh_graphs;
    mesh_graphs.reserve(mgd_paths_in_order.size());
    for (const auto& p : mgd_paths_in_order) {
        mesh_graphs.emplace_back(cluster, p.string());
    }

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    config.hostname_to_asics = hostname_to_physical_nodes(psd);
    config.mesh_host_rank_pinnings = mesh_host_rank_pinnings;

    std::vector<MultiMeshMappingPart> parts(mesh_graph_descriptors.size());
    const int world_size =
        static_cast<int>(*tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->size());
    for (std::size_t mgi = 0; mgi < mesh_graph_descriptors.size(); ++mgi) {
        parts[mgi].mesh_graph_descriptor = &mesh_graph_descriptors[mgi];
        PinningsByMesh local_pins = mesh_graph_descriptors[mgi].get_pinnings();
        drop_inactive_revision_pinnings(local_pins, psd);
        if (cluster.is_ubb_galaxy()) {
            for (const auto& mesh_id : mesh_graphs[mgi].get_all_mesh_ids()) {
                const auto& mesh_shape = mesh_graphs[mgi].get_mesh_shape(mesh_id);
                const bool is_1d = mesh_shape[0] == 1 || mesh_shape[1] == 1;
                if (!is_1d && mesh_shape.mesh_size() % 32 == 0) {
                    auto corner_pins = get_galaxy_fixed_asic_position_pinnings_for_mesh(
                        mesh_id, mesh_shape, /*hard_pin_node_0=*/world_size == 1, /*nw_corner_only=*/false);
                    merge_pinnings_by_mesh(local_pins, corner_pins);
                }
            }
        }
        if (!local_pins.empty()) {
            parts[mgi].pinnings = std::move(local_pins);
        }
        for (const auto& mesh_id_local : mesh_graphs[mgi].get_all_mesh_ids()) {
            const auto& chip_ids = mesh_graphs[mgi].get_chip_ids(mesh_id_local);
            for (const auto& [coord, chip_id] : chip_ids) {
                FabricNodeId fabric_node_id(mesh_id_local, chip_id);
                auto mesh_host_rank = mesh_graphs[mgi].get_host_rank_for_chip(mesh_id_local, chip_id);
                if (mesh_host_rank.has_value()) {
                    parts[mgi].fabric_node_id_to_mesh_rank[mesh_id_local][fabric_node_id] = mesh_host_rank.value();
                }
            }
        }
    }

    log_info(tt::LogFabric, "Preparing topology mapping ({} MGD(s))...", mesh_graph_descriptors.size());
    for (const auto& mesh_graph : mesh_graphs) {
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
    }
    if (!config.inter_mesh_validation_mode.has_value()) {
        config.inter_mesh_validation_mode = ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }
    if (config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
        log_info(tt::LogFabric, "Inter-mesh validation mode: RELAXED");
    } else {
        log_info(tt::LogFabric, "Inter-mesh validation mode: STRICT");
    }

    log_info(tt::LogFabric, "Building topology mapping enumerator ({} MGD(s))...", mesh_graph_descriptors.size());
    if (pgd != nullptr) {
        return MultiMeshSolutionEnumerator(psd, *pgd, parts, config, unique_shapes);
    }
    return MultiMeshSolutionEnumerator(psd, parts, config, unique_shapes);
}

/**
 * @brief Run topology mapper to map logical meshes to physical ASICs
 *
 * @param mesh_graph_descriptors  Const reference to the caller's vector of loaded MGDs (the `std::vector`
 *                                is not copied—only a reference is passed). Must match `mgd_paths_in_order`
 *                                in length and order (one path per descriptor for `MeshGraph` host ranks).
 * @param mgd_paths_in_order      Const reference to paths parallel to `mesh_graph_descriptors`.
 *
 * @return One local-mesh-id mapping per input MGD. A failed mapping still returns one result
 *         per MGD (success=false, error_message, closest intra-mesh maps). Empty parts means
 *         the enumerator was not initialized.
 */
TopologyMappingParts run_topology_mapping(
    const PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor* pgd,
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const std::vector<std::filesystem::path>& mgd_paths_in_order,
    const std::vector<MeshHostRankPinningGroup>& mesh_pinning_groups = {}) {
    log_info(tt::LogFabric, "Running topology mapping with mesh graph rank bindings...");
    TopologyMappingParts out;

    if (mesh_pinning_groups.empty()) {
        MultiMeshSolutionEnumerator enumerator = make_topology_mapping_enumerator(
            psd, pgd, mesh_graph_descriptors, mgd_paths_in_order, /*unique_shapes=*/false);
        out.parts = enumerator.next();
        return out;
    }

    if (mesh_graph_descriptors.size() != 1) {
        throw std::runtime_error("Mesh-host-rank pinning currently supports exactly one MGD");
    }
    const MeshGraph mesh_graph(MetalContext::instance().get_cluster(), mgd_paths_in_order.front().string());
    const std::vector<ResolvedMeshHostRankPinningGroup> resolved_pinning_groups =
        resolve_mesh_host_rank_pinnings(psd, mesh_graph, hostname_to_physical_nodes(psd), mesh_pinning_groups);

    // Each group is many-to-many: try every injective (mesh-host-rank -> placement) assignment across the
    // groups, depth first, and keep the first one the mapper can complete. The chosen placements become hard
    // per-host-rank ASIC pools for the solve (TopologyMappingConfig::mesh_host_rank_pinnings).
    std::map<MeshHostRankKey, std::set<tt::tt_metal::experimental::PhysicalNodeId>> candidate_constraints;
    std::map<MeshHostRankKey, HostPlacement> candidate_selections;
    std::size_t attempted_assignments = 0;
    std::string last_error;

    std::function<bool(std::size_t)> try_group;
    try_group = [&](std::size_t group_index) -> bool {
        if (group_index == resolved_pinning_groups.size()) {
            ++attempted_assignments;
            try {
                MultiMeshSolutionEnumerator enumerator = make_topology_mapping_enumerator(
                    psd,
                    pgd,
                    mesh_graph_descriptors,
                    mgd_paths_in_order,
                    /*unique_shapes=*/false,
                    candidate_constraints);
                out.parts = enumerator.next();
            } catch (const std::exception& e) {
                out.parts.clear();
                last_error = e.what();
            }
            if (!out.parts.empty() && out.parts.front().success) {
                out.selected_mesh_host_rank_pinnings = candidate_selections;
                return true;
            }
            if (!out.parts.empty()) {
                last_error = out.parts.front().error_message;
            }
            return false;
        }

        const ResolvedMeshHostRankPinningGroup& group = resolved_pinning_groups[group_index];
        std::vector<bool> used_placements(group.host_placements.size(), false);
        std::function<bool(std::size_t)> assign_pair;
        assign_pair = [&](std::size_t pair_index) -> bool {
            if (pair_index == group.mesh_host_ranks.size()) {
                return try_group(group_index + 1);
            }
            const MeshHostRankKey& key = group.mesh_host_ranks[pair_index];
            for (std::size_t placement_index = 0; placement_index < group.host_placements.size(); ++placement_index) {
                if (used_placements[placement_index]) {
                    continue;
                }
                const ResolvedHostPlacement& placement = group.host_placements[placement_index];
                used_placements[placement_index] = true;
                candidate_constraints[key] = placement.asics;
                candidate_selections[key] = placement.host_placement;
                if (assign_pair(pair_index + 1)) {
                    return true;
                }
                candidate_selections.erase(key);
                candidate_constraints.erase(key);
                used_placements[placement_index] = false;
            }
            return false;
        };
        return assign_pair(0);
    };

    const bool mapping_found = try_group(0);
    if (!mapping_found) {
        TopologyMappingResult failed;
        failed.success = false;
        failed.stats.failure_stage = "mesh_host_rank_pinning";
        failed.error_message = fmt::format(
            "No valid topology mapping satisfied the mesh-host-rank pinning groups after {} assignment(s). Last "
            "solver error: {}",
            attempted_assignments,
            last_error.empty() ? std::string("none") : last_error);
        out.parts = {failed};
        return out;
    }

    log_info(
        tt::LogFabric,
        "Selected a valid mesh-host-rank placement after {} candidate assignment(s)",
        attempted_assignments);
    for (const auto& [mesh_host_rank, placement] : out.selected_mesh_host_rank_pinnings) {
        log_info(
            tt::LogFabric,
            "Mesh {} host rank {} -> {} with TT_VISIBLE_DEVICES={}",
            mesh_host_rank.first.get(),
            mesh_host_rank.second.get(),
            placement.hostname,
            format_visible_devices(placement.tt_visible_devices));
    }
    return out;
}

// Post-solve verification; also makes the selected placement explicit in rank_bindings.yaml.
void validate_selected_mesh_host_rank_pinnings(
    const std::map<MeshHostRankKey, HostPlacement>& selected_pinnings, std::vector<RankBindingConfig>& rank_bindings) {
    for (const auto& [mesh_host_rank, placement] : selected_pinnings) {
        auto binding_it = std::find_if(
            rank_bindings.begin(), rank_bindings.end(), [&mesh_host_rank](const RankBindingConfig& binding) {
                return binding.mesh_id == static_cast<int>(mesh_host_rank.first.get()) &&
                       binding.mesh_host_rank == static_cast<int>(mesh_host_rank.second.get());
            });
        if (binding_it == rank_bindings.end()) {
            throw std::runtime_error(fmt::format(
                "Constraint validation failed: no rank binding was produced for mesh {} host rank {}",
                mesh_host_rank.first.get(),
                mesh_host_rank.second.get()));
        }
        if (binding_it->hostname != placement.hostname) {
            throw std::runtime_error(fmt::format(
                "Constraint validation failed: mesh {} host rank {} selected host '{}', but topology mapping "
                "placed it on '{}'",
                mesh_host_rank.first.get(),
                mesh_host_rank.second.get(),
                placement.hostname,
                binding_it->hostname));
        }
        const std::string expected_visible_devices = format_visible_devices(placement.tt_visible_devices);
        const auto visible_it = binding_it->env_overrides.find("TT_VISIBLE_DEVICES");
        const std::string actual_visible_devices =
            visible_it == binding_it->env_overrides.end() ? std::string{} : visible_it->second;
        if (actual_visible_devices != expected_visible_devices) {
            throw std::runtime_error(fmt::format(
                "Constraint validation failed: mesh {} host rank {} selected TT_VISIBLE_DEVICES={}, but mapped "
                "ASICs produced TT_VISIBLE_DEVICES={}",
                mesh_host_rank.first.get(),
                mesh_host_rank.second.get(),
                expected_visible_devices,
                actual_visible_devices));
        }
        binding_it->env_overrides["TT_VISIBLE_DEVICES"] = expected_visible_devices;
    }
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
 * `mapping_result` already uses this MeshGraph's local MeshIds.
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd, const TopologyMappingResult& mapping_result, const MeshGraph& mesh_graph) {
    struct AsicGrouping {
        std::vector<tt::tt_metal::experimental::PhysicalNodeId> physical_node_ids;
        std::vector<tt::ChipId> chip_ids;
        std::optional<MeshHostRankId> mesh_host_rank;
        uint32_t mpi_rank = 0;
    };

    // mesh_id -> hostname -> mesh_host_rank -> AsicGrouping
    std::map<int, std::map<std::string, std::map<int, AsicGrouping>>> mesh_host_asics;

    for (const auto& [fabric_node_id, physical_node_id] : mapping_result.fabric_node_to_physical) {
        const MeshId mesh_id_local = fabric_node_id.mesh_id;
        tt::ChipId chip_id_from_fabric_node = static_cast<tt::ChipId>(fabric_node_id.chip_id);

        std::optional<MeshHostRankId> mesh_host_rank =
            mesh_graph.get_host_rank_for_chip(mesh_id_local, chip_id_from_fabric_node);

        if (!mesh_host_rank.has_value()) {
            log_error(
                tt::LogFabric,
                "No mesh host rank found for mesh_id {} and chip_id {}",
                *mesh_id_local,
                chip_id_from_fabric_node);
            continue;
        }

        // Emit the hostname exactly as the descriptor spells it, not the canonical form carried in the
        // node id. The rankfile uses it to place the process, and the binding exports it as
        // TT_METAL_CLUSTER_ID so phase 2 discovery stamps the same cluster id phase 1 saw.
        const AsicID asic_id = psd.get_asic_id(physical_node_id);
        std::string hostname = psd.get_host_name_for_asic(asic_id);
        tt::ChipId chip_id = psd.get_umd_unique_id(asic_id);

        int mesh_id_int = static_cast<int>(*mesh_id_local);
        const int mesh_host_rank_int = static_cast<int>(*mesh_host_rank.value());
        auto& bucket = mesh_host_asics[mesh_id_int][hostname][mesh_host_rank_int];
        bucket.physical_node_ids.push_back(physical_node_id);
        bucket.chip_ids.push_back(chip_id);
        bucket.mesh_host_rank = mesh_host_rank;
        bucket.mpi_rank = psd.get_mpi_rank_for_asic(asic_id);
    }

    // Build flat list of (mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank) for canonical ordering.
    // Order: mesh_id first (so sequential output rank i tracks mesh order, keeping MPI rank == mesh_id for
    // single-MGD), then mesh_host_rank, hostname, and PSD rank as a deterministic tiebreaker.
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
                const uint32_t psd_rank = asic_data.mpi_rank;
                entries.emplace_back(
                    mesh_id, hostname, chip_ids, static_cast<int>(*mesh_host_rank.value()), static_cast<int>(psd_rank));
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        // Primary: mesh_id ascending, so the sequential binding.rank aligns with mesh order (rank 0 from mesh id 0
        // first). This preserves main's single-MGD invariant that MPI world rank == mesh_id (needed for e.g. the
        // Blitz pipeline's rank_bindings / discovery alignment). PSD identity is not lost: it is stored separately
        // in binding.psd_mpi_rank below, so PSD rank only needs to be a deterministic tiebreaker here.
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        if (std::get<3>(a) != std::get<3>(b)) {
            return std::get<3>(a) < std::get<3>(b);
        }
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<4>(a) < std::get<4>(b);
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

        // The chip's MPI rank. Several ranks share a hostname, so the host map cannot supply this.
        binding.psd_mpi_rank = psd_rank;
        binding.mesh_id = mesh_id;
        binding.mesh_host_rank = mesh_host_rank;
        binding.hostname = hostname;
        binding.slot = host_slot_counters[hostname]++;
        binding.env_overrides = {};
        if (!hostname.empty()) {
            binding.env_overrides["TT_METAL_CLUSTER_ID"] = hostname;
        }

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
    std::optional<std::string> mesh_pinning_file_path;  // --mesh-pinning-file: optional host/TT_VISIBLE_DEVICES pins
    bool all_solutions = false;       // --all-solutions/-a: one artifact set per solution (single-MGD only)
    std::size_t max_solutions = 0;    // --max-solutions/-n: cap (0 = all up to solver cap); implies --all-solutions
    bool distinct_host_sets = false;  // --distinct-host-sets/-d: one solution per unique host set
    bool allow_shape_permutations = false;  // hidden: disable the solver's unique_shapes dedup
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
        cxxopts::value<std::string>())(
        "mesh-pinning-file",
        "Path to optional YAML pinning mesh host ranks to hostname + TT_VISIBLE_DEVICES placements. SINGLE-MGD "
        "only (-m); not combined with --all-solutions.",
        cxxopts::value<std::string>())(
        "a,all-solutions",
        "Enumerate all valid solutions and write one artifact set per solution into per-solution subdirectories "
        "(plus solutions_index.yaml). SINGLE-MGD only (-m). Default: write only the first solution flat in "
        "--output-dir.")(
        "n,max-solutions",
        "Maximum number of solutions to enumerate (0 = all up to the solver safety cap). Implies --all-solutions.",
        cxxopts::value<std::size_t>())(
        "d,distinct-host-sets",
        "Keep only one solution per unique set of HOSTS: after enumeration, solutions that occupy the same hosts "
        "(even if they wire/assign the meshes differently) are collapsed to the first. Only meaningful with "
        "--all-solutions.")("h,help", "Print usage information");

    // Hidden/advanced options (in a separate group so they do NOT appear in --help).
    options.add_options("hidden")(
        "allow-shape-permutations",
        "(advanced) Disable the solver's unique_shapes dedup, which is otherwise always on. When set, the solver "
        "may enumerate multiple automorphic physical footprints (same set of physical meshes, permuted). Rarely "
        "needed; mainly for debugging enumeration completeness.");

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
        if (result.contains("mesh-pinning-file")) {
            args.mesh_pinning_file_path = result["mesh-pinning-file"].as<std::string>();
        }
        if (result.contains("max-solutions")) {
            args.max_solutions = result["max-solutions"].as<std::size_t>();
            args.all_solutions = true;  // --max-solutions implies --all-solutions
        }
        if (result.contains("all-solutions")) {
            args.all_solutions = true;
        }
        if (result.contains("distinct-host-sets")) {
            args.distinct_host_sets = true;
            if (!args.all_solutions) {
                log_warning(
                    tt::LogFabric,
                    "--distinct-host-sets has no effect without --all-solutions/--max-solutions; ignoring it.");
            }
        }
        if (result.contains("allow-shape-permutations")) {
            args.allow_shape_permutations = true;
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

        // Factory System Descriptor (FSD) path comes from RTOptions, populated from the
        // TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH environment variable (set by tt-run and propagated to this
        // MPI worker). NOTE: FSD-driven mapping is not yet implemented; today we always fall back to live PSD
        // discovery below. See issue #52859 for the follow-up that consumes this.
        const auto& fsd_path = tt::tt_metal::MetalContext::instance().rtoptions().get_factory_system_descriptor_path();
        if (!fsd_path.empty()) {
            log_info(
                tt::LogFabric,
                "Factory System Descriptor provided via TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH ({}); FSD-driven "
                "mapping not yet implemented, falling back to live PSD discovery.",
                fsd_path);
        }

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
            // Matches MeshGraph in Phase 2, so what Control Plane rejects is rejected here too.
            mgds.emplace_back(MeshGraphDescriptor(mgd_path, /*backwards_compatible=*/true));
            mgd_paths_in_order.push_back(mgd_path);
        }
        std::optional<PhysicalGroupingDescriptor> pgd = PhysicalGroupingDescriptor::find_and_load(
            MetalContext::instance().get_cluster(),
            args.physical_grouping_descriptor_path.has_value()
                ? std::optional<std::filesystem::path>(*args.physical_grouping_descriptor_path)
                : std::nullopt,
            &psd);
        const PhysicalGroupingDescriptor* pgd_ptr = pgd.has_value() ? &*pgd : nullptr;

        std::vector<MeshHostRankPinningGroup> mesh_pinning_groups;
        if (args.mesh_pinning_file_path.has_value()) {
            if (mgds.size() != 1) {
                throw std::runtime_error(
                    "--mesh-pinning-file supports only a single Mesh Graph Descriptor (-m), not a multi-MGD "
                    "mapping (-M)");
            }
            if (args.all_solutions) {
                throw std::runtime_error("--mesh-pinning-file cannot be combined with --all-solutions");
            }
            log_info(tt::LogFabric, "Stage: Loading mesh pinning file from: {}", *args.mesh_pinning_file_path);
            mesh_pinning_groups = parse_mesh_pinning_file(*args.mesh_pinning_file_path);
            log_info(tt::LogFabric, "Loaded {} mesh-host-rank pinning group(s)", mesh_pinning_groups.size());
        }

        // Get current rank - only rank 0 performs topology mapping and file generation
        auto current_rank = *context->rank();
        if (current_rank == 0) {
            if (args.all_solutions) {
                if (args.subcontext_id_to_mgd_path.size() != 1) {
                    log_error(
                        tt::LogFabric,
                        "--all-solutions is only supported with a single MGD (-m); multi-MGD (-M / multiple "
                        "sub-contexts) is not supported.");
                    return 1;
                }
                const std::filesystem::path& mgd_path = mgd_paths_in_order.front();
                const std::string mgd_path_str = mgd_path.string();
                auto& metal_context = tt::tt_metal::MetalContext::instance();
                const auto& cluster = metal_context.get_cluster();
                // Single MeshGraph in a vector to match main's multi-MGD extract_rank_bindings signature.
                std::vector<MeshGraph> mesh_graphs_for_extract;
                mesh_graphs_for_extract.emplace_back(cluster, mgd_path.string());
                const std::filesystem::path output_dir =
                    args.output_dir.has_value() ? std::filesystem::path(*args.output_dir) : "generated/ttrun";
                std::filesystem::create_directories(output_dir);

                const bool mock_cluster_rankfile = !mpi_rank_to_cluster_desc_path.empty();

                // Flush a file and its parent directory to storage before signaling peers via barrier.
                // std::ofstream::close() only drains the C++ stream buffer to the OS page cache; without fsync(),
                // NFS peers (and local readers) may see stale or absent files even after this process exits. We
                // fsync each file and its parent directory so both data and directory entries are durable before
                // the barrier() below — making the barrier the authoritative "writes are visible" signal.
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

                // Write one solution's artifacts (rank_bindings.yaml, rankfile, optional phase2 mock mapping) into
                // `dir`.
                auto write_solution_artifacts = [&](const std::vector<RankBindingConfig>& rank_bindings,
                                                    const std::filesystem::path& dir) {
                    std::filesystem::create_directories(dir);
                    const std::filesystem::path rank_bindings_file = dir / "rank_bindings.yaml";
                    write_rank_bindings_yaml(rank_bindings, mgd_path_str, rank_bindings_file.string());
                    const std::filesystem::path rankfile_path = dir / "rankfile";
                    write_rankfile(rank_bindings, rankfile_path.string(), mock_cluster_rankfile);
                    fsync_path(rank_bindings_file);
                    fsync_path(rankfile_path);
                    if (!mpi_rank_to_cluster_desc_path.empty()) {
                        const std::filesystem::path phase2_mock_path = dir / "phase2_mock_mapping.yaml";
                        write_phase2_mock_mapping_yaml(
                            rank_bindings, mpi_rank_to_cluster_desc_path, phase2_mock_path.string());
                        fsync_path(phase2_mock_path);
                    }
                    log_info(tt::LogFabric, "Successfully wrote solution artifacts to: {}", dir.string());
                };

                // --all-solutions: one artifact set per solution in a content-hash subdirectory, plus a
                // top-level solutions_index.yaml summarizing them all.
                log_info(tt::LogFabric, "Stage: Enumerating all topology mapping solutions (streaming)...");
                // Solver-level unique_shapes dedup is ALWAYS ON (collapses automorphic physical footprints);
                // only the hidden --allow-shape-permutations turns it off.
                const bool unique_shapes = !args.allow_shape_permutations;
                const std::string enumeration_mode = args.distinct_host_sets ? "distinct-host-sets" : "all";

                // Same enumerator as the single-solution path, built once. Each next() yields one mapping
                // from the live SAT session; this loop is the only extra work for --all-solutions.
                MultiMeshSolutionEnumerator enumerator =
                    make_topology_mapping_enumerator(psd, pgd_ptr, mgds, mgd_paths_in_order, unique_shapes);

                std::vector<SolutionIndexEntry> index_entries;

                // --distinct-host-sets: dedup written solutions by their resolved set of HOSTS. This is coarser
                // than the solver's unique_shapes (which dedups by physical-mesh footprint): for sub-host meshes,
                // several shape-distinct solutions can share one host set, and this keeps only the first.
                std::set<std::set<std::string>> seen_host_sets;
                const std::filesystem::path index_path = output_dir / "solutions_index.yaml";

                // Cap semantics: WITHOUT --distinct-host-sets, the cap bounds raw enumerator outputs
                // (matches the batch --max-solutions semantics). WITH --distinct-host-sets, the cap bounds
                // WRITTEN (distinct) solutions instead: the SAT enumeration explores near-neighbor
                // assignments first, so many consecutive raw solutions share one host set -- counting raw
                // outputs starves the mode of exactly the host-set variety it exists to produce (e.g. 20
                // raw solutions collapsing to 7 distinct sets, all anchored on the first host). The raw
                // safety cap still bounds total solver work in both modes.
                // cap_reached distinguishes "stopped at the cap (more may exist)" from "genuinely exhausted".
                constexpr std::size_t kEnumerationSafetyCap = 500000;
                const std::size_t effective_cap = args.max_solutions != 0 ? args.max_solutions : kEnumerationSafetyCap;
                std::size_t emitted = 0;
                bool cap_reached = false;
                while (true) {
                    const std::size_t capped_count = args.distinct_host_sets ? index_entries.size() : emitted;
                    if (capped_count >= effective_cap || emitted >= kEnumerationSafetyCap) {
                        cap_reached = true;
                        break;
                    }
                    const std::vector<TopologyMappingResult> parts = enumerator.next();
                    if (parts.empty() || !parts.front().success) {
                        break;  // exhausted, or first next() reported no valid mapping
                    }
                    ++emitted;

                    std::vector<RankBindingConfig> rank_bindings =
                        extract_rank_bindings(psd, parts.front(), mesh_graphs_for_extract.front());

                    const std::set<std::string> hosts = solution_host_set(rank_bindings);
                    if (args.distinct_host_sets && !seen_host_sets.insert(hosts).second) {
                        log_info(
                            tt::LogFabric,
                            "--distinct-host-sets: skipping solution on an already-seen host set ({} hosts)",
                            hosts.size());
                        continue;
                    }

                    // Disambiguate short-hash collisions BEFORE writing: if the natural dir already holds a
                    // DIFFERENT solution (its .solution_key differs from this one's canonical signature), pick a
                    // suffixed id instead of overwriting it. Same signature => reuse the dir (idempotent re-run).
                    const std::string signature = compute_solution_signature_string(rank_bindings);
                    const std::string base_id = compute_solution_signature_hash(rank_bindings);
                    std::string solution_id = base_id;
                    std::filesystem::path solution_dir = output_dir / solution_id;
                    for (int suffix = 1;; ++suffix) {
                        std::ifstream existing_key(solution_dir / ".solution_key");
                        std::string existing;
                        if (!existing_key || (std::getline(existing_key, existing), existing == signature)) {
                            break;  // free dir, or the same solution already there -> use it
                        }
                        log_warning(
                            tt::LogFabric,
                            "Short-hash collision on {}: a different solution already occupies it; disambiguating",
                            solution_id);
                        solution_id = base_id + "-" + std::to_string(suffix);
                        solution_dir = output_dir / solution_id;
                    }

                    write_solution_artifacts(rank_bindings, solution_dir);

                    // Full canonical signature for short-hash collision disambiguation.
                    const std::filesystem::path key_path = solution_dir / ".solution_key";
                    {
                        std::ofstream key_file(key_path);
                        key_file << signature << std::endl;
                    }
                    const std::filesystem::path meta_path = solution_dir / "solution_meta.yaml";
                    write_solution_meta_yaml(rank_bindings, solution_id, mgd_path_str, meta_path.string());
                    fsync_path(key_path);
                    fsync_path(meta_path);

                    SolutionIndexEntry entry;
                    entry.id = solution_id;
                    entry.num_ranks = static_cast<int>(rank_bindings.size());
                    entry.num_hosts = static_cast<int>(hosts.size());
                    entry.host_set.assign(hosts.begin(), hosts.end());
                    index_entries.push_back(std::move(entry));

                    // Rewrite the index after every solution so the on-disk index always reflects everything written
                    // so far (crash/timeout resilient). While streaming we mark truncated=true (more may still come);
                    // the definitive flag is written after the enumeration ends below.
                    write_solutions_index_yaml(
                        mgd_path_str,
                        enumeration_mode,
                        args.max_solutions,
                        /*truncated=*/true,
                        index_entries,
                        index_path.string());
                    fsync_path(index_path);
                    log_info(
                        tt::LogFabric, "Streamed solution {} ({} written so far)", solution_id, index_entries.size());
                }

                // Final index rewrite with the definitive truncated flag: true only if a positive cap bounded the
                // output (more solutions may exist), false once the enumeration is genuinely exhausted.
                const bool truncated = cap_reached;
                write_solutions_index_yaml(
                    mgd_path_str, enumeration_mode, args.max_solutions, truncated, index_entries, index_path.string());
                fsync_path(index_path);
                log_info(
                    tt::LogFabric,
                    "Wrote solutions index: {} ({} solution(s), truncated={})",
                    index_path.string(),
                    index_entries.size(),
                    truncated);

                if (index_entries.empty()) {
                    log_error(tt::LogFabric, "No valid topology solutions found");
                    return 1;
                }
            } else {
                // Stage: Run topology mapping
                log_info(tt::LogFabric, "Stage: Running topology mapping...");

                TopologyMappingParts topology =
                    run_topology_mapping(psd, pgd_ptr, mgds, mgd_paths_in_order, mesh_pinning_groups);

                if (topology.parts.empty() || !topology.parts.front().success) {
                    log_error(
                        tt::LogFabric,
                        "Topology mapping failed: {}",
                        topology.parts.empty() ? "no valid placement+intra-mesh mapping found"
                                               : topology.parts.front().error_message);
                    // Peers wait on the exit barrier; a plain return would hang them.
                    context->abort(1);
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
                std::vector<std::pair<int, std::string>> rank_bindings_mapping_entries;
                int global_rank_base = 0;

                for (size_t mgi = 0; mgi < mesh_graphs_for_extract.size(); ++mgi) {
                    const int subctx_id = subcontext_ids_in_order[mgi];
                    if (mgi >= topology.parts.size()) {
                        log_error(
                            tt::LogFabric, "Sub-context {}: no local mapping part from the enumerator", subctx_id);
                        return 1;
                    }
                    std::vector<RankBindingConfig> rank_bindings =
                        extract_rank_bindings(psd, topology.parts[mgi], mesh_graphs_for_extract[mgi]);
                    if (!multi_mgd) {
                        validate_selected_mesh_host_rank_pinnings(
                            topology.selected_mesh_host_rank_pinnings, rank_bindings);
                    }

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
                fsync_path(rankfile_path);
                if (!mpi_rank_to_cluster_desc_path.empty()) {
                    fsync_path(output_dir / "phase2_mock_mapping.yaml");
                }
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
        // Peers wait on the exit barrier; a plain return would hang them.
        context->abort(1);
        return 1;
    }

    return 0;
}
