// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_mapper.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <queue>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_fabric {

TopologyMapper::TopologyMapper(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const LocalMeshBinding& local_mesh_binding) :
    mesh_graph_(mesh_graph),
    physical_system_descriptor_(physical_system_descriptor),
    local_mesh_binding_(local_mesh_binding) {
    build_mapping();
}

void TopologyMapper::build_mapping() {
    log_debug(tt::LogFabric, "TopologyMapper: Building mapping between fabric node IDs and physical ASIC IDs");

    // Use BFS to build the complete host to mesh mapping
    auto mesh_id_host_rank_to_host_name = build_host_mesh_mapping_bfs();

    log_debug(
        tt::LogFabric,
        "TopologyMapper: Built mapping for {} mesh-host-rank pairs",
        mesh_id_host_rank_to_host_name.size());

    log_info(tt::LogFabric, "TopologyMapper: Mesh-host-rank mapping: {}", mesh_id_host_rank_to_host_name);

    // TODO: This currently does not support multiple meshes per host because
    // Graph isomorphism algorithm needs to be used to map multiple meshes per host
}

std::unordered_map<MeshId, HostRank> TopologyMapper::build_host_mesh_mappings() {
    std::unordered_map<MeshId, HostRank> mesh_id_to_host_rank;

    // Get the current host name
    std::string current_host = physical_system_descriptor_.my_host_name();

    // Get the current mesh to start with
    auto mesh_id = local_mesh_binding_.mesh_ids[0];

    // Populate initial current host mapping
    mesh_id_to_host_rank[mesh_id] = {current_host};

    // Discover all hosts
    discover_hosts_dfs(mesh_id, current_host, mesh_id_to_host_rank);

    return mesh_id_to_host_rank;
}

void TopologyMapper::discover_hosts_dfs(
    MeshId mesh_id, HostName& host_name, std::unordered_map<MeshId, HostRank>& mesh_id_to_host_rank) {
    // Get the mesh size
    auto mesh_size = mesh_graph_.get_mesh_shape(mesh_id).mesh_size();

    // Get the host size
    auto host_size = physical_system_descriptor_.get_asics_connected_to_host(host_name).size();

    // Calculate how many ranks are needed to cover the mesh
    auto num_ranks = mesh_size / host_size;

    // If Greater than 1, Big mesh
    if (num_ranks == 1) {
        mesh_id_to_host_rank[mesh_id] = {host_name};
        // TODO: Find adjacent hosts

        else if (num_ranks > 1) {
            mesh_id_to_host_rank[mesh_id].resize(num_ranks);

            // TODO: Find adjacent hosts
        }
        else {  // If Less than 1, many meshes per host
            mesh_id_to_host_rank[mesh_id] = {host_name};

            // TODO: Find adjacent meshes on same host
        }

        // For each host neighbor, discover the hosts
    }

}  // namespace tt::tt_fabric
