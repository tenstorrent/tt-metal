// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "magic_enum.hpp"
#include "tt_fabric/mesh_graph.hpp"

namespace tt::tt_fabric {

using RoutingTable =
    std::vector<std::vector<std::vector<RoutingDirection>>>;  // [mesh_id][chip_id][target_chip_or_mesh_id]

class RoutingTableGenerator {
   public:
    RoutingTableGenerator(const std::string& mesh_graph_desc_yaml_file);
    ~RoutingTableGenerator() = default;

    void dump_to_yaml();
    void load_from_yaml();

    void print_connectivity() const { this->mesh_graph_->print_connectivity(); }

    const IntraMeshConnectivity& get_intra_mesh_connectivity() const {
        return this->mesh_graph_->get_intra_mesh_connectivity();
    }
    const InterMeshConnectivity& get_inter_mesh_connectivity() const {
        return this->mesh_graph_->get_inter_mesh_connectivity();
    }
    const ChipSpec& get_chip_spec() const { return this->mesh_graph_->get_chip_spec(); }

    RoutingTable get_intra_mesh_table() const { return this->intra_mesh_table_; }
    RoutingTable get_inter_mesh_table() const { return this->inter_mesh_table_; }

    void print_routing_tables() const;

   private:
    std::unique_ptr<MeshGraph> mesh_graph_;
    ;
    // configurable in future architectures
    const uint32_t max_nodes_in_mesh_ = 1024;
    const uint32_t max_num_meshes_ = 1024;

    std::vector<uint32_t> mesh_sizes;

    RoutingTable intra_mesh_table_;
    RoutingTable inter_mesh_table_;

    std::vector<std::vector<std::vector<std::pair<chip_id_t, mesh_id_t>>>> get_paths_to_all_meshes(
        mesh_id_t src, const InterMeshConnectivity& inter_mesh_connectivity);
    void generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity);
    // when generating intermesh routing table, we use the intramesh connectivity table to find the shortest path to the
    // exit chip
    void generate_intermesh_routing_table(
        const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& intra_mesh_connectivity);
};

}  // namespace tt::tt_fabric
