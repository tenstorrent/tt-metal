// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt_stl/reflection.hpp>
#include <umd/device/types/arch.h>                      // tt::ARCH
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include <vector>
namespace tt {
enum class ARCH;
}  // namespace tt

namespace tt::tt_fabric {

using tt::tt_metal::distributed::MeshContainer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;

struct ChipSpec {
    tt::ARCH arch;
    std::uint32_t num_eth_ports_per_direction;
    std::uint32_t num_z_ports;
};

enum class FabricType {
    MESH = 1 << 0,
    TORUS_X = 1 << 1,  // Connections along mesh_coord[1]
    TORUS_Y = 1 << 2,  // Connections along mesh_coord[0]
    TORUS_XY = (TORUS_X | TORUS_Y),
};

FabricType operator|(FabricType lhs, FabricType rhs);
FabricType operator&(FabricType lhs, FabricType rhs);
bool has_flag(FabricType flags, FabricType test_flag);

enum class RoutingDirection {
    N = 0,
    E = 2,
    S = 4,
    W = 8,
    C = 16,     // Centre, means that destination is same as source
    NONE = 32,  // No direction, means that destination is not reachable
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
using InterMeshConnectivity = std::vector<std::vector<std::unordered_map<MeshId, RouterEdge>>>;
using IntraMeshConnectivity = std::vector<std::vector<std::unordered_map<chip_id_t, RouterEdge>>>;

class MeshGraph {
public:
    explicit MeshGraph(const std::string& mesh_graph_desc_file_path);
    MeshGraph() = delete;
    ~MeshGraph() = default;

    void print_connectivity() const;

    const IntraMeshConnectivity& get_intra_mesh_connectivity() const;
    const InterMeshConnectivity& get_inter_mesh_connectivity() const;

    const ChipSpec& get_chip_spec() const { return chip_spec_; }

    // Get the host ranks for a given mesh_id
    // Returned MeshContainer has a shape denoting the shape of how the "board" are arranged
    const MeshContainer<HostRankId>& get_host_ranks(MeshId mesh_id) const;

    // Get the shape of the mesh, or the shape of the submesh for a given host rank if provided
    MeshShape get_mesh_shape(MeshId mesh_id, std::optional<HostRankId> host_rank = std::nullopt) const;

    // Get the coordinate range of the mesh, or the coordinate range of the submesh for a given host rank if provided
    MeshCoordinateRange get_coord_range(MeshId mesh_id, std::optional<HostRankId> host_rank = std::nullopt) const;

    std::vector<MeshId> get_mesh_ids() const;

    // Get the chip ids for a given mesh_id
    // If host_rank is provided, return the chip ids for the submesh for that host rank
    // Otherwise, return the chip ids for the entire mesh
    MeshContainer<chip_id_t> get_chip_ids(MeshId mesh_id, std::optional<HostRankId> host_rank = std::nullopt) const;

    // Get the host rank that owns a given chip in a mesh
    std::optional<HostRankId> get_host_rank_for_chip(MeshId mesh_id, chip_id_t chip_id) const;

    // Translation functions for chip_id and coordinate using RM-convention
    MeshCoordinate chip_to_coordinate(MeshId mesh_id, chip_id_t chip_id) const;
    chip_id_t coordinate_to_chip(MeshId mesh_id, MeshCoordinate coordinate) const;

private:
    void validate_mesh_id(MeshId mesh_id) const;
    std::unordered_map<chip_id_t, RouterEdge> get_valid_connections(
        const MeshCoordinate& src_mesh_coord, const MeshCoordinateRange& mesh_coord_range, FabricType fabric_type) const;
    void initialize_from_yaml(const std::string& mesh_graph_desc_file_path);

    void add_to_connectivity(
        MeshId src_mesh_id,
        chip_id_t src_chip_id,
        MeshId dest_mesh_id,
        chip_id_t dest_chip_id,
        RoutingDirection port_direction);

    ChipSpec chip_spec_;
    std::map<MeshId, MeshContainer<chip_id_t>> mesh_to_chip_ids_;
    IntraMeshConnectivity intra_mesh_connectivity_;
    InterMeshConnectivity inter_mesh_connectivity_;

    // For distributed context, bookkeeping of host ranks and their shapes
    std::vector<MeshContainer<HostRankId>> mesh_host_ranks_;
    std::unordered_map<std::pair<MeshId, HostRankId>, MeshCoordinateRange, hash_pair> mesh_host_rank_coord_ranges_;
};
}  // namespace tt::tt_fabric
