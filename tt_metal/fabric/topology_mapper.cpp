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
    auto mesh_id_host_rank_to_host_name = build_host_mesh_mappings();

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

    std::unordered_set<HostName> visited_hosts = {};
    std::unordered_set<MeshId> visited_meshes = {};

    // Discover all hosts
    discover_hosts_dfs(mesh_id, current_host, mesh_id_to_host_rank, visited_hosts, visited_meshes);

    return mesh_id_to_host_rank;
}

bool TopologyMapper::discover_hosts_dfs(
    const MeshId mesh_id,
    const HostName& host_name,
    std::unordered_map<MeshId, HostRank>& mesh_id_to_host_rank,
    std::unordered_set<HostName>& visited_hosts,
    std::unordered_set<MeshId>& visited_meshes) {
    // Mesh Graph related information
    auto mesh_size = mesh_graph_.get_mesh_shape(mesh_id).mesh_size();
    auto host_ranks = mesh_graph_.get_host_ranks(mesh_id);

    // Physical system related information
    auto host_size = physical_system_descriptor_.get_asics_connected_to_host(host_name).size();

    // Rank sizes
    auto rank_size = mesh_size / host_size;

    // Size of the mesh ratio to host size does not match the number of host ranks means the match is incorrect
    if (host_ranks.size() > 1 && host_ranks.size() != rank_size) {
        return false;
    }

    if (host_ranks.size() == 1 && mesh_size > host_size) {
        return false;
    }

    if (visited_hosts.contains(host_name)) {
        return false;
    }
    if (visited_meshes.contains(mesh_id)) {
        return false;
    }

    // Add the chain to the mapping
    visited_hosts.insert(host_name);

    // If Equal to 1, Single mesh per host
    if (rank_size == 1) {
        mesh_id_to_host_rank[mesh_id] = {host_name};
        visited_meshes.insert(mesh_id);
    } else if (rank_size > 1) {
        mesh_id_to_host_rank[mesh_id].resize(host_ranks.size());
        mesh_id_to_host_rank[mesh_id][physical_system_descriptor_.get_rank_for_hostname(host_name)] = host_name;

        // Check if full mesh has been visited
        bool full_mesh_visited = true;
        for (const auto& host_name : mesh_id_to_host_rank[mesh_id]) {
            if (host_name.empty()) {
                full_mesh_visited = false;
                break;
            }
        }
        if (full_mesh_visited) {
            visited_meshes.insert(mesh_id);
        }
    } else {
        mesh_id_to_host_rank[mesh_id] = {host_name};
        visited_meshes.insert(mesh_id);
    }

    // Check if all hosts and meshes have been visited
    if (visited_hosts.size() == physical_system_descriptor_.get_all_hostnames().size() && visited_meshes.size() == mesh_graph_.get_mesh_ids().size()) {
        // FINISH!
        return true;
    }

    std::vector<HostName> adjacent_hosts;
    std::vector<MeshId> adjacent_meshes;

    // If Equal to 1, Single mesh per host
    if (rank_size == 1) {
        adjacent_hosts = physical_system_descriptor_.get_host_neighbors(host_name);
        adjacent_meshes = mesh_graph_.get_adjacent_meshes(mesh_id);
    // If Greater than 1, Big mesh
    } else if (rank_size > 1) {
        adjacent_hosts = physical_system_descriptor_.get_host_neighbors(host_name);
        adjacent_meshes = mesh_graph_.get_adjacent_meshes(mesh_id);
        adjacent_meshes.push_back(mesh_id);
    } else {
        adjacent_hosts = {host_name};
        adjacent_meshes = mesh_graph_.get_adjacent_meshes(mesh_id);
    }

    // Check every combination of adjacent meshes and hosts to see if there is a valid mapping
    for (const auto& adjacent_mesh : adjacent_meshes) {
        for (const auto& adjacent_host : adjacent_hosts) {
            if (discover_hosts_dfs(adjacent_mesh, adjacent_host, mesh_id_to_host_rank, visited_hosts, visited_meshes)) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace tt::tt_fabric
