// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace tt::tt_metal {
class Program;
struct ProgramDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshShape;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

// EXPERIMENTAL / INTERNAL API. Not part of the stable public surface; signature may change.
//
// These APIs separate compile-time concerns (kernel defines) from runtime concerns (semaphores +
// RT args), enabling an eager-compile model where defines are queried before kernel build and
// connections are set up after build without post_build hooks.

// Query the kernel defines required by the current fabric configuration and API type.
// Pure query — no PD mutation, no side effects. Safe to call before kernel compilation.
// Returns defines like {("FABRIC_2D", "1"), ("API_TYPE_Linear", "1")}.
std::vector<std::pair<std::string, std::string>> get_fabric_kernel_defines(
    tt::tt_fabric::FabricApiType api_type = tt::tt_fabric::FabricApiType::Linear);

// Compute fabric connection RT args without any PD mutation.
// Pure computation — resolves routing + assembles RT args using caller-provided semaphore IDs.
// Returns flat vector matching RoutingPlaneConnectionManager::build_from_args() layout.
std::vector<uint32_t> compute_fabric_connection_rt_args(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    const std::vector<tt::tt_fabric::FabricNodeId>& dst_nodes,
    const std::vector<uint32_t>& connection_link_indices,
    const std::vector<uint32_t>& teardown_sem_ids,
    const std::vector<uint32_t>& buffer_index_sem_ids);

// Like append_routing_plane_connection_manager_rt_args but does NOT inject kernel defines.
// Use with get_fabric_kernel_defines() when defines must be set before kernel compilation
// (e.g., blaze eager-compile model). Allocates semaphores and computes RT args only.
template <typename ProgramOrDescriptor>
void append_routing_plane_connection_rt_args_no_defines(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    const std::vector<tt::tt_fabric::FabricNodeId>& dst_nodes,
    const std::vector<uint32_t>& connection_link_indices,
    ProgramOrDescriptor& worker_program_or_desc,
    tt::tt_metal::KernelHandle& kernel_id,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    tt::tt_fabric::FabricApiType api_type = tt::tt_fabric::FabricApiType::Linear,
    CoreType core_type = CoreType::WORKER);

// Returns all compute mesh ids known to the current mesh graph descriptor, including
// peer meshes that are not visible via get_user_physical_mesh_ids() (which only
// returns the local rank's meshes). Intended for inter-mesh topology discovery.
std::vector<tt::tt_fabric::MeshId> get_all_fabric_mesh_ids();

}  // namespace tt::tt_metal::internal
