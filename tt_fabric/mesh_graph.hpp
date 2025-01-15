// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt::tt_fabric {
struct ChipSpec {
    tt::ARCH arch;
    std::uint32_t num_eth_ports_per_direction;
    std::uint32_t num_z_ports;
};

enum class FabricType {
    MESH = 0,
    TORUS = 1,
};

enum class RoutingDirection {
    N = 0,
    E = 2,
    S = 4,
    W = 8,
    C = 16,  // Centre, means that destination is same as source
};

struct RouterEdge {
    // TODO: change this to be port_id_t
    RoutingDirection port_direction;  // Assume all ports in one direction connect to the same chip
    std::vector<chip_id_t>
        connected_chip_ids;  // One per port to the connected chip, used by ControlPlane to map to physical links
    std::uint32_t weight;    // Assume all chip to chip communication is equal weight, but in reality there may be less
                             // intermesh traffic for example
};
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        return tt::stl::hash::hash_objects(std::hash<T1>{}(p.first), std::hash<T2>{}(p.second));
    }
};

using port_id_t = std::pair<RoutingDirection, uint32_t>;
using mesh_id_t = uint32_t;
using InterMeshConnectivity = std::vector<std::vector<std::unordered_map<mesh_id_t, RouterEdge>>>;
using IntraMeshConnectivity = std::vector<std::vector<std::unordered_map<chip_id_t, RouterEdge>>>;

class MeshGraph {
   public:
       explicit MeshGraph(const std::string& mesh_graph_desc_file_path);
       MeshGraph() = delete;
       ~MeshGraph() = default;

       void print_connectivity() const;

       const IntraMeshConnectivity& get_intra_mesh_connectivity() const { return intra_mesh_connectivity_; }
       const InterMeshConnectivity& get_inter_mesh_connectivity() const { return inter_mesh_connectivity_; }

       const ChipSpec& get_chip_spec() const { return chip_spec_; }

       std::uint32_t get_mesh_ns_size(mesh_id_t mesh_id) const { return mesh_shapes_[mesh_id].first; }
       std::uint32_t get_mesh_ew_size(mesh_id_t mesh_id) const { return mesh_shapes_[mesh_id].second; }

   private:
       std::unordered_map<chip_id_t, RouterEdge> get_valid_connections(
           chip_id_t src_chip_id,
           std::uint32_t row_size,
           std::uint32_t num_chips_in_mesh,
           FabricType fabric_type) const;
       void initialize_from_yaml(const std::string& mesh_graph_desc_file_path);

       void add_to_connectivity(
           mesh_id_t src_mesh_id,
           chip_id_t src_chip_id,
           chip_id_t dest_mesh_id,
           chip_id_t dest_chip_id,
           RoutingDirection port_direction);

       ChipSpec chip_spec_;
       std::vector<std::pair<std::uint32_t, std::uint32_t>> mesh_shapes_;
       IntraMeshConnectivity intra_mesh_connectivity_;
       InterMeshConnectivity inter_mesh_connectivity_;
};
}  // namespace tt::tt_fabric
