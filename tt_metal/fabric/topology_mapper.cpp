// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_mapper.hpp"

#include <algorithm>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
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

    // Find corners for each host
    auto host_to_corners = build_corner_mappings();

    // locate the 4 corners for every mesh
    auto mesh_id_corners = build_mesh_corners_mappings(host_to_corners, mesh_id_host_rank_to_host_name);

    // TODO: This currently does not support multiple meshes per host because
    // Graph isomorphism algorithm needs to be used to map multiple meshes per host
}

std::unordered_map<MeshId, MeshContainer<HostName>> TopologyMapper::build_host_mesh_mappings() {
    std::unordered_map<MeshId, MeshContainer<HostName>> mesh_id_to_host_rank;

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
    std::unordered_map<MeshId, MeshContainer<HostName>>& mesh_id_to_host_rank,
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

    // If Equal to 1, Single mesh per host
    if (rank_size == 1) {
        mesh_id_to_host_rank.emplace(mesh_id, MeshContainer<HostName>(MeshShape{1, 1}, {host_name}));
        visited_meshes.insert(mesh_id);
        visited_hosts.insert(host_name);
        // If Greater than 1, Big mesh
    } else if (rank_size > 1) {
        if (!mesh_id_to_host_rank.contains(mesh_id)) {
            std::vector<HostName> empty_values(host_ranks.size(), "");
            mesh_id_to_host_rank.emplace(mesh_id, MeshContainer<HostName>(host_ranks.shape(), empty_values));
        }
        for (const auto& host_rank : host_ranks) {
            if (host_rank.value().get() == physical_system_descriptor_.get_rank_for_hostname(host_name)) {
                mesh_id_to_host_rank.at(mesh_id).at(host_rank.coord()) = host_name;
            }
        }

        // Check if all in container have been visited
        bool all_visited = true;
        for (const auto& host_rank : mesh_id_to_host_rank.at(mesh_id)) {
            if (host_rank.value().empty()) {
                all_visited = false;
            }
        }
        if (all_visited) {
            visited_meshes.insert(mesh_id);
        }
        // Mark this host as visited once it has been placed in the big mesh
        visited_hosts.insert(host_name);

        // If Less than one, multiple meshes per host
    } else {
        mesh_id_to_host_rank.emplace(mesh_id, MeshContainer<HostName>(MeshShape{1, 1}, {host_name}));
        visited_meshes.insert(mesh_id);

        // Check if all meshes expected to be owned by this host have been visited
        // For multi-mesh-per-host, a single host spans multiple meshes. The expected
        // number of meshes for this host equals host_size / mesh_size.
        // Once all those meshes are discovered, mark the host as visited.
        const std::size_t mesh_count_expected_for_host = host_size / mesh_size;
        std::size_t mesh_count_discovered_for_host = 0;
        for (const auto& [mapped_mesh_id, container] : mesh_id_to_host_rank) {
            // Only count single-host containers created in this branch
            // Only one coordinate exists in a 1x1 container: (0,0)
            if (container.shape().mesh_size() == 1 && container.at(MeshCoordinate{0, 0}) == host_name) {
                mesh_count_discovered_for_host++;
            }
        }

        if (mesh_count_discovered_for_host >= mesh_count_expected_for_host) {
            visited_hosts.insert(host_name);
        }
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

std::unordered_map<std::string, std::vector<tt::tt_metal::AsicID>> TopologyMapper::build_corner_mappings() const {
    std::unordered_map<std::string, std::vector<tt::tt_metal::AsicID>> host_to_corners;

    for (const auto& host_name : physical_system_descriptor_.get_all_hostnames()) {
        // Calculate degree
        int max_degree = 0;
        for (const auto& asic_id : physical_system_descriptor_.get_asics_connected_to_host(host_name)) {
            auto asic_neighbors = physical_system_descriptor_.get_asic_neighbors(asic_id);

            int neighbor_count = 0;
            for (const auto& asic_neighbor : asic_neighbors) {
                // If is a local connection, count it
                if (physical_system_descriptor_.get_host_name_for_asic(asic_neighbor) == host_name) {
                    neighbor_count++;
                }
            }
            max_degree = std::max(max_degree, neighbor_count);
        }
        int corner_connection_count = 0;

        int num_corners = 0;
        if (max_degree == 0) {
            // 1x1 case
            host_to_corners[host_name].push_back(physical_system_descriptor_.get_asics_connected_to_host(host_name)[0]);
            continue;
        } else if (max_degree <= 4) {
            corner_connection_count = 2;
            num_corners = 4;
        } else if (max_degree <= 2) {
            corner_connection_count = 1;
            num_corners = 2;
        } else {
            TT_THROW("FATAL Currently not supporting 3D meshes");
        }

        // Find corners
        for (const auto& asic_id : physical_system_descriptor_.get_asics_connected_to_host(host_name)) {
            auto asic_neighbors = physical_system_descriptor_.get_asic_neighbors(asic_id);
            int neighbor_count = 0;
            for (const auto& asic_neighbor : asic_neighbors) {
                if (physical_system_descriptor_.get_host_name_for_asic(asic_neighbor) == host_name) {
                    neighbor_count++;
                }
            }

            if (neighbor_count == corner_connection_count) {
                host_to_corners[host_name].push_back(asic_id);
            }
        }

        TT_FATAL(
            host_to_corners[host_name].size() == num_corners,
            "Host {} does not form a valid uniform mesh, please run ./build/test/tt_metal/tt_fabric/test_system_health "
            "to check connection health",
            host_name);
    }
    return host_to_corners;
}

std::unordered_map<MeshId, MeshContainer<tt::tt_metal::AsicID>> TopologyMapper::build_mesh_corners_mappings(
    std::unordered_map<std::string, std::vector<tt::tt_metal::AsicID>>& host_corners,
    std::unordered_map<MeshId, MeshContainer<HostName>>& mesh_id_to_host_name) const {
    std::unordered_map<MeshId, MeshContainer<tt::tt_metal::AsicID>> mesh_id_to_corners;

    for (const auto& [mesh_id, host_name_container] : mesh_id_to_host_name) {
        log_critical(LogFabric, "Mesh {} Coord Range: {}", *mesh_id, mesh_graph_.get_coord_range(mesh_id));
    }

    return mesh_id_to_corners;
}

}  // namespace tt::tt_fabric
