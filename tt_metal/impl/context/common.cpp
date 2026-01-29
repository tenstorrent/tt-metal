// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace tt::tt_metal {

std::shared_ptr<distributed::multihost::DistributedContext> construct_compute_only_distributed_context(
    const tt::tt_fabric::ControlPlane& control_plane) {
    const auto& global_context = distributed::multihost::DistributedContext::get_current_world();
    if (*global_context->size() == 1) {
        return global_context;
    }

    // Get all compute mesh IDs (excludes switches) from control plane mesh graph
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // If there are no switch meshes, return the global context directly
    if (mesh_graph.get_switch_ids().empty()) {
        return global_context;
    }

    const auto& compute_mesh_ids = mesh_graph.get_mesh_ids();

    // Get global logical bindings to map ranks to mesh IDs
    const auto& global_logical_bindings = control_plane.get_global_logical_bindings();

    // Collect all MPI ranks for compute meshes only
    std::unordered_set<int> compute_mpi_ranks;
    for (const auto& [rank, mesh_binding] : global_logical_bindings) {
        const auto& [mesh_id, _] = mesh_binding;
        // Check if this mesh_id is a compute mesh (not a switch)
        if (std::find(compute_mesh_ids.begin(), compute_mesh_ids.end(), mesh_id) != compute_mesh_ids.end()) {
            compute_mpi_ranks.insert(rank.get());
        }
    }

    // If no compute meshes found, fall back to host_local_context
    if (compute_mpi_ranks.empty()) {
        TT_THROW("No compute meshes found in mesh graph.");
    }

    // Convert to sorted vector for create_sub_context
    std::vector<int> compute_ranks_vec(compute_mpi_ranks.begin(), compute_mpi_ranks.end());
    std::sort(compute_ranks_vec.begin(), compute_ranks_vec.end());

    // Check if current rank is in compute ranks
    int current_rank = *global_context->rank();
    bool is_current_rank_in_compute =
        std::find(compute_ranks_vec.begin(), compute_ranks_vec.end(), current_rank) != compute_ranks_vec.end();

    // If current rank is not in compute ranks (e.g., host only has switches), return host_local_context
    if (!is_current_rank_in_compute) {
        return control_plane.get_host_local_context();
    }

    // Create sub-context with only compute mesh ranks
    return global_context->create_sub_context(compute_ranks_vec);
}

}  // namespace tt::tt_metal
