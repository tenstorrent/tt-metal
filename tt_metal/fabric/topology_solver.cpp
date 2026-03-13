// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric {

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_graph_logical(const MeshGraphDescriptor& mgd) {
    std::map<MeshId, AdjacencyGraph<FabricNodeId>::AdjacencyMap> adjacency_maps;

    // Collect all mesh and switch instances (same order as MeshGraph: meshes then switches)
    std::vector<GlobalNodeId> mesh_and_switch_instances;
    for (GlobalNodeId id : mgd.all_meshes()) {
        mesh_and_switch_instances.push_back(id);
    }
    for (GlobalNodeId id : mgd.all_switches()) {
        mesh_and_switch_instances.push_back(id);
    }

    // Initialize nodes for each mesh/switch
    for (GlobalNodeId mesh_global_id : mesh_and_switch_instances) {
        const auto& instance = mgd.get_instance(mesh_global_id);
        MeshId mesh_id(instance.local_id);
        AdjacencyGraph<FabricNodeId>::AdjacencyMap& adj = adjacency_maps[mesh_id];
        for (GlobalNodeId device_global_id : instance.sub_instances) {
            const auto& device_inst = mgd.get_instance(device_global_id);
            adj[FabricNodeId(mesh_id, device_inst.local_id)] = std::vector<FabricNodeId>();
        }
    }

    // Add intra-mesh edges from MESH connections (device-device within a mesh, populated by descriptor)
    if (mgd.has_connections_of_type("MESH")) {
        for (ConnectionId conn_id : mgd.connections_by_type("MESH")) {
            const auto& conn = mgd.get_connection(conn_id);
            const auto& src_inst = mgd.get_instance(conn.nodes[0]);
            const auto& dst_inst = mgd.get_instance(conn.nodes[1]);
            if (src_inst.kind != NodeKind::Device || dst_inst.kind != NodeKind::Device) {
                continue;
            }
            GlobalNodeId src_mesh_global = src_inst.hierarchy.back();
            GlobalNodeId dst_mesh_global = dst_inst.hierarchy.back();
            if (src_mesh_global != dst_mesh_global) {
                continue;
            }
            const auto& mesh_inst = mgd.get_instance(src_mesh_global);
            MeshId mesh_id(mesh_inst.local_id);
            FabricNodeId src_node(mesh_id, src_inst.local_id);
            FabricNodeId dst_node(mesh_id, dst_inst.local_id);
            if (src_node == dst_node) {
                continue;
            }
            auto it = adjacency_maps.find(mesh_id);
            if (it != adjacency_maps.end()) {
                for (uint32_t i = 0; i < conn.count; ++i) {
                    it->second[src_node].push_back(dst_node);
                }
            }
        }
    }

    std::map<MeshId, AdjacencyGraph<FabricNodeId>> result;
    for (auto& [mesh_id, adj_map] : adjacency_maps) {
        result[mesh_id] = AdjacencyGraph<FabricNodeId>(adj_map);
    }
    return result;
}

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_graph_logical(const MeshGraph& mesh_graph) {
    std::map<MeshId, AdjacencyGraph<FabricNodeId>> adjacency_map;

    auto get_local_adjacents = [&](FabricNodeId fabric_node_id, MeshId mesh_id) {
        auto adjacent_map = mesh_graph.get_intra_mesh_connectivity()[*mesh_id][fabric_node_id.chip_id];

        std::vector<FabricNodeId> adjacents;
        for (const auto& [neighbor_chip_id, edge] : adjacent_map) {
            // Skip self-connections
            if (neighbor_chip_id == fabric_node_id.chip_id) {
                continue;
            }
            for (size_t i = 0; i < edge.connected_chip_ids.size(); ++i) {
                adjacents.push_back(FabricNodeId(mesh_id, neighbor_chip_id));
            }
        }
        return adjacents;
    };

    // Iterate over all mesh IDs from the mesh graph (including switches)
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        AdjacencyGraph<FabricNodeId>::AdjacencyMap logical_adjacency_map;
        for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            auto fabric_node_id = FabricNodeId(mesh_id, chip_id);
            logical_adjacency_map[fabric_node_id] = get_local_adjacents(fabric_node_id, mesh_id);
        }
        adjacency_map[mesh_id] = AdjacencyGraph<FabricNodeId>(logical_adjacency_map);
    }

    return adjacency_map;
}

std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_graph_physical(
    tt::tt_metal::ClusterType /*cluster_type*/,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> adjacency_map;

    // Build a set of ASIC IDs for each mesh based on mesh rank mapping
    std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> mesh_asic_ids;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            mesh_asic_ids[mesh_id].insert(asic_id);
        }
    }

    for (const auto& [mesh_id, mesh_asics] : mesh_asic_ids) {
        auto get_local_adjacents = [&](tt::tt_metal::AsicID asic_id,
                                       const std::unordered_set<tt::tt_metal::AsicID>& mesh_asics) {
            std::vector<tt::tt_metal::AsicID> adjacents;

            for (const auto& neighbor : physical_system_descriptor.get_asic_neighbors(asic_id)) {
                // Skip self-connections
                if (neighbor == asic_id) {
                    continue;
                }
                // Make sure that the neighbor is in the mesh
                if (mesh_asics.contains(neighbor)) {
                    // Add each neighbor multiple times based on number of ethernet connections
                    auto eth_connections = physical_system_descriptor.get_eth_connections(asic_id, neighbor);
                    adjacents.insert(adjacents.end(), eth_connections.size(), neighbor);
                }
            }
            return adjacents;
        };

        AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap physical_adjacency_map;
        for (const auto& asic_id : mesh_asics) {
            physical_adjacency_map[asic_id] = get_local_adjacents(asic_id, mesh_asics);
        }
        adjacency_map[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adjacency_map);
    }

    return adjacency_map;
}

}  // namespace tt::tt_fabric
