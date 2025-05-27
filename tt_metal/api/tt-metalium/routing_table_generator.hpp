// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <magic_enum/magic_enum.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_graph.hpp>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_fabric {

using RoutingTable =
    std::vector<std::vector<std::vector<RoutingDirection>>>;  // [mesh_id][chip_id][target_chip_or_mesh_id]

// TODO: first pass at switching over mesh_id_t/chip_id_t to proper struct
// Need to update the usage in routing table generator
class FabricNodeId {
public:
    explicit FabricNodeId(std::uint32_t mesh_id, std::uint32_t chip_id);
    std::uint32_t mesh_id = 0;
    std::uint32_t chip_id = 0;
};

bool operator==(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator!=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>=(const FabricNodeId& lhs, const FabricNodeId& rhs);
std::ostream& operator<<(std::ostream& os, const FabricNodeId& fabric_node_id);

class RoutingTableGenerator {
public:
    explicit RoutingTableGenerator(const std::string& mesh_graph_desc_yaml_file);
    ~RoutingTableGenerator() = default;

    void dump_to_yaml();
    void load_from_yaml();

    RoutingTable get_intra_mesh_table() const { return this->intra_mesh_table_; }
    RoutingTable get_inter_mesh_table() const { return this->inter_mesh_table_; }

    void print_routing_tables() const;

    std::unique_ptr<MeshGraph> mesh_graph;

private:
    // configurable in future architectures
    const uint32_t max_nodes_in_mesh_ = 1024;
    const uint32_t max_num_meshes_ = 1024;

    RoutingTable intra_mesh_table_;
    RoutingTable inter_mesh_table_;

    std::vector<std::vector<std::vector<std::pair<chip_id_t, mesh_id_t>>>> get_paths_to_all_meshes(
        mesh_id_t src, const InterMeshConnectivity& inter_mesh_connectivity);
    void generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity);
    // when generating intermesh routing table, we use the intramesh connectivity table to find the shortest path to
    // the exit chip
    void generate_intermesh_routing_table(
        const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& intra_mesh_connectivity);
};

}  // namespace tt::tt_fabric

namespace std {
template <>
struct hash<tt::tt_fabric::FabricNodeId> {
    size_t operator()(const tt::tt_fabric::FabricNodeId& fabric_node_id) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(fabric_node_id.mesh_id, fabric_node_id.chip_id);
    }
};
}  // namespace std
