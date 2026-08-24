// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <functional>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/device_types.hpp>

namespace tt::tt_fabric {
class TopologyMapper;
// Defined in tt_metal/fabric/axis_route_topology.hpp. Held by pointer so the internal ring/domain
// representation stays out of this public header.
struct AxisRouteTopology;

using RoutingTable =
    std::vector<std::vector<std::vector<RoutingDirection>>>;  // [mesh_id][chip_id][target_chip_or_mesh_id]

class RoutingTableGenerator {
public:
    explicit RoutingTableGenerator(const TopologyMapper& topology_mapper);
    ~RoutingTableGenerator();  // out of line: express_rings_ holds an incomplete type

    void dump_to_yaml();
    void load_from_yaml();

    RoutingTable get_intra_mesh_table() const { return this->intra_mesh_table_; }
    RoutingTable get_inter_mesh_table() const { return this->inter_mesh_table_; }

    // Ring decomposition of a mesh's express axis, or nullptr when the mesh declares no express links.
    // The type stays incomplete here; only tt_metal/fabric consumers include its definition.
    const AxisRouteTopology* get_express_rings(MeshId mesh_id) const;

    // The ordinary X ring, or nullptr when the mesh has no express rings or that dimension does not close.
    const AxisRouteTopology* get_x_rings(MeshId mesh_id) const;

    // The topology governing `axis` of this mesh: express chords where declared for that axis, the
    // ordinary ring where the axis closes, else the plain line. Unlike get_express_rings/get_x_rings
    // this is NEVER null for a 2D mesh -- the line fallback guarantees an answer. Prefer it: a null
    // axis topology silently disables multicast, because the encoder needs a per-axis tree on both
    // dimensions.
    const AxisRouteTopology* get_axis_topology(MeshId mesh_id, int axis) const;

    void print_routing_tables() const;
    // Return a list of all exit nodes, across all meshes that are connected to the requested
    // MeshID.
    const std::vector<FabricNodeId>& get_exit_nodes_routing_to_mesh(MeshId mesh_id) const;
    // Return the single exit node (chip in src_mesh_id) for a given src chip and dst mesh
    FabricNodeId get_exit_node_from_mesh_to_mesh(MeshId src_mesh_id, ChipId src_chip_id, MeshId dst_mesh_id) const;

    // Load Inter-Mesh Connectivity into the Routing Table Generator
    void load_intermesh_connections(const AnnotatedIntermeshConnections& intermesh_connections);

private:
    const TopologyMapper& topology_mapper_;
    // configurable in future architectures
    const uint32_t max_nodes_in_mesh_ = 1024;
    const uint32_t max_num_meshes_ = 1024;

    RoutingTable intra_mesh_table_;
    RoutingTable inter_mesh_table_;
    // Per mesh, null when the mesh declares no express links.
    std::vector<std::unique_ptr<AxisRouteTopology>> express_rings_;
    // Per mesh, null when the X dimension does not close.
    std::vector<std::unique_ptr<AxisRouteTopology>> x_rings_;
    // Per mesh, per axis (0 = Y, 1 = X). Always populated for a 2D mesh: express chords, else the
    // ordinary ring, else the plain line. This is the one that must never be null.
    std::vector<std::array<std::unique_ptr<AxisRouteTopology>, 2>> axis_topologies_;
    std::unordered_map<MeshId, std::vector<FabricNodeId>> mesh_to_exit_nodes_;
    // Direct lookup table: [src_mesh][src_chip][dst_mesh] -> exit chip_id in src_mesh
    std::vector<std::vector<std::vector<ChipId>>> exit_node_lut_;

    std::vector<std::vector<std::pair<ChipId, MeshId>>> get_first_hops_to_all_meshes(
        MeshId src, const InterMeshConnectivity& inter_mesh_connectivity) const;
    void generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity);
    // Setup validation for a mesh whose express rings were derived: walks every ordered pair of the
    // generated table and rejects the configuration on any violation. No-op for other meshes.
    void validate_express_ring_routes(std::uint32_t mesh_id_val, const IntraMeshConnectivity& intra_mesh_connectivity);
    // when generating intermesh routing table, we use the intramesh connectivity table to find the shortest path to
    // the exit chip
    void generate_intermesh_routing_table(
        const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& intra_mesh_connectivity);
};

}  // namespace tt::tt_fabric
