// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

#include <enchantum/enchantum.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>

namespace tt::tt_fabric {

RoutingTableGenerator::RoutingTableGenerator(const TopologyMapper& topology_mapper) :
    topology_mapper_(topology_mapper) {
    // Use IntraMeshConnectivity to size all variables
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = mesh_graph.get_inter_mesh_connectivity();
    this->intra_mesh_table_.resize(intra_mesh_connectivity.size());
    this->inter_mesh_table_.resize(intra_mesh_connectivity.size());
    this->exit_node_lut_.resize(intra_mesh_connectivity.size());
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); mesh_id_val++) {
        this->intra_mesh_table_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        this->inter_mesh_table_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        this->exit_node_lut_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        for (auto& devices_in_mesh : this->intra_mesh_table_[mesh_id_val]) {
            // intra_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other chips in the mesh
            devices_in_mesh.resize(intra_mesh_connectivity[mesh_id_val].size());
        }
        for (auto& devices_in_mesh : this->inter_mesh_table_[mesh_id_val]) {
            // inter_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other meshes
            devices_in_mesh.resize(intra_mesh_connectivity.size());
        }
        for (auto& devices_in_mesh : this->exit_node_lut_[mesh_id_val]) {
            // exit_node_lut_[mesh_id][chip_id] holds exit chip per destination mesh
            devices_in_mesh.resize(intra_mesh_connectivity.size());
        }
    }
    // Generate the intra mesh routing table
    this->generate_intramesh_routing_table(intra_mesh_connectivity);

    // Generate the inter mesh routing table
    this->generate_intermesh_routing_table(inter_mesh_connectivity, intra_mesh_connectivity);
}

void RoutingTableGenerator::generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity) {
    const auto get_shorter_direction_on_row_or_col = [&](std::uint32_t mesh_id_val,
                                                         std::uint32_t src_chip_id,
                                                         std::uint32_t dst_chip_id,
                                                         RoutingDirection a,
                                                         RoutingDirection b) -> RoutingDirection {
        // Loop through intra_mesh_connectivity starting with a or b direction and return direction that matches
        // dst_chip_id_first In case of tie, this function is returning a
        std::uint32_t curr_a = src_chip_id, curr_b = src_chip_id;
        bool a_valid = true, b_valid = true;
        while (a_valid or b_valid) {
            if (intra_mesh_connectivity[mesh_id_val][curr_a].contains(dst_chip_id) and
                intra_mesh_connectivity[mesh_id_val][curr_a].at(dst_chip_id).port_direction == a) {
                return a;
            }
            if (intra_mesh_connectivity[mesh_id_val][curr_b].contains(dst_chip_id) and
                intra_mesh_connectivity[mesh_id_val][curr_b].at(dst_chip_id).port_direction == b) {
                return b;
            }
            a_valid = false;
            b_valid = false;
            for (const auto& [next_chip_id, edge] : intra_mesh_connectivity[mesh_id_val][curr_a]) {
                if (edge.port_direction == a) {
                    curr_a = next_chip_id;
                    a_valid = true;
                    break;
                }
            }
            for (const auto& [next_chip_id, edge] : intra_mesh_connectivity[mesh_id_val][curr_b]) {
                if (edge.port_direction == b) {
                    curr_b = next_chip_id;
                    b_valid = true;
                    break;
                }
            }
        }
        TT_ASSERT(
            false,
            "No valid direction found for src_chip_id {} and dst_chip_id {} in mesh_id {}. "
            "This should not happen, check the intra_mesh_connectivity.",
            src_chip_id,
            dst_chip_id,
            mesh_id_val);
        return RoutingDirection::NONE;  // This line should never be reached
    };
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_table_.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        for (ChipId src_chip_id = 0; src_chip_id < this->intra_mesh_table_[mesh_id_val].size(); src_chip_id++) {
            for (ChipId dst_chip_id = 0; dst_chip_id < this->intra_mesh_table_[mesh_id_val].size(); dst_chip_id++) {
                auto src_mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, src_chip_id);
                auto dst_mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, dst_chip_id);
                // X first routing, traverse rows first
                if (src_mesh_coord[0] != dst_mesh_coord[0]) {
                    // If source and destination are in different rows, we need to move in the X direction first
                    // Move North or South
                    MeshCoordinate target_coord_on_column(dst_mesh_coord[0], src_mesh_coord[1]);
                    auto target_chip_id = mesh_graph.coordinate_to_chip(mesh_id, target_coord_on_column);
                    auto direction = get_shorter_direction_on_row_or_col(
                        mesh_id_val, src_chip_id, target_chip_id, RoutingDirection::N, RoutingDirection::S);
                    this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = direction;
                    // TODO: today we are not updating the weight of the edge, should we use weight to balance
                    //  routing traffic?
                    //  intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else if (src_mesh_coord[1] != dst_mesh_coord[1]) {
                    // Move East or West
                    auto direction = get_shorter_direction_on_row_or_col(
                        mesh_id_val, src_chip_id, dst_chip_id, RoutingDirection::E, RoutingDirection::W);
                    this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = direction;
                    // intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else {
                    // No movement
                    // TODO: what value do we put for this entry? If we pack table entries to 4 bits
                    // any number is a valid port id. Do we assume FW will never try to access table entry to itself?
                    this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::C;
                }
            }
        }
    }
}

// Returns first hop to reach each destination mesh from source mesh.
// first_hops[dst_mesh] = vector of (exit_chip_in_source_mesh, immediate_next_mesh) pairs
// This is more memory efficient than storing full paths (exponential space complexity) since we only need the first hop.
std::vector<std::vector<std::pair<ChipId, MeshId>>> RoutingTableGenerator::get_first_hops_to_all_meshes(
    MeshId src, const InterMeshConnectivity& inter_mesh_connectivity) const {
    std::uint32_t num_meshes = inter_mesh_connectivity.size();
    std::vector<std::uint8_t> visited(num_meshes, false);

    // first_hops[target_mesh_id] = vector of (exit_chip, next_mesh) pairs representing first hop options
    std::vector<std::vector<std::pair<ChipId, MeshId>>> first_hops(num_meshes);

    std::vector<std::uint32_t> dist(num_meshes, std::numeric_limits<std::uint32_t>::max());
    dist[*src] = 0;

    std::queue<MeshId> q;
    q.push(src);
    visited[*src] = true;

    // BFS to find shortest paths
    while (!q.empty()) {
        MeshId current_mesh_id = q.front();
        q.pop();

        for (ChipId chip_in_mesh = 0; chip_in_mesh < inter_mesh_connectivity[*current_mesh_id].size(); chip_in_mesh++) {
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[*current_mesh_id][chip_in_mesh]) {
                if (!visited[*connected_mesh_id]) {
                    q.push(connected_mesh_id);
                    visited[*connected_mesh_id] = true;
                }

                std::uint32_t new_dist = dist[*current_mesh_id] + 1;
                if (dist[*connected_mesh_id] > new_dist) {
                    // Found shorter path - update distance and first hops
                    dist[*connected_mesh_id] = new_dist;
                    if (current_mesh_id == src) {
                        // Direct neighbor: first hop is (chip_in_source, connected_mesh)
                        first_hops[*connected_mesh_id] = {{chip_in_mesh, connected_mesh_id}};
                    } else {
                        // Multi-hop: inherit first hops from intermediate mesh
                        first_hops[*connected_mesh_id] = first_hops[*current_mesh_id];
                    }
                } else if (dist[*connected_mesh_id] == new_dist) {
                    // Same distance - add alternative first hops
                    if (current_mesh_id == src) {
                        first_hops[*connected_mesh_id].push_back({chip_in_mesh, connected_mesh_id});
                    } else {
                        // Only inherit from each intermediate mesh once.
                        // Multiple chips in the same intermediate mesh connecting to the same destination
                        // would give identical first hops - avoid adding duplicates.
                        // Check if we've already inherited from current_mesh by checking if its first hop
                        // is already present in connected_mesh's first hops.
                        bool already_inherited = false;
                        if (!first_hops[*current_mesh_id].empty() && !first_hops[*connected_mesh_id].empty()) {
                            const auto& hop_to_check = first_hops[*current_mesh_id][0];
                            already_inherited = std::find(
                                                    first_hops[*connected_mesh_id].begin(),
                                                    first_hops[*connected_mesh_id].end(),
                                                    hop_to_check) != first_hops[*connected_mesh_id].end();
                        }
                        if (!already_inherited) {
                            for (const auto& hop : first_hops[*current_mesh_id]) {
                                first_hops[*connected_mesh_id].push_back(hop);
                            }
                        }
                    }
                }
            }
        }
    }
    return first_hops;
}

void RoutingTableGenerator::generate_intermesh_routing_table(
    const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& /*intra_mesh_connectivity*/) {
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();

    for (std::uint32_t src_mesh_id_val = 0; src_mesh_id_val < this->inter_mesh_table_.size(); src_mesh_id_val++) {
        MeshId src_mesh_id{src_mesh_id_val};
        auto first_hops = get_first_hops_to_all_meshes(src_mesh_id, inter_mesh_connectivity);
        MeshShape mesh_shape = mesh_graph.get_mesh_shape(src_mesh_id);
        std::uint32_t ew_size = mesh_shape[1];
        for (ChipId src_chip_id = 0; src_chip_id < this->inter_mesh_table_[src_mesh_id_val].size(); src_chip_id++) {
            for (std::uint32_t dst_mesh_id_val = 0; dst_mesh_id_val < this->inter_mesh_table_.size();
                 dst_mesh_id_val++) {
                MeshId dst_mesh_id{dst_mesh_id_val};
                if (dst_mesh_id == src_mesh_id) {
                    // inter mesh table entry from mesh to itself
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = RoutingDirection::C;
                    this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = src_chip_id;
                    continue;
                }
                auto& candidate_first_hops = first_hops[dst_mesh_id_val];
                std::uint32_t min_load = std::numeric_limits<std::uint32_t>::max();
                std::uint32_t min_distance = std::numeric_limits<std::uint32_t>::max();
                if (candidate_first_hops.empty()) {
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = RoutingDirection::NONE;
                    this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        std::numeric_limits<ChipId>::max();
                    continue;
                }
                // Initialize with first candidate
                ChipId exit_chip_id = candidate_first_hops[0].first;
                MeshId next_mesh_id = candidate_first_hops[0].second;
                for (const auto& hop : candidate_first_hops) {
                    ChipId candidate_exit_chip_id = hop.first;
                    MeshId candidate_next_mesh_id = hop.second;
                    if (candidate_exit_chip_id == src_chip_id) {
                        // optimization for latency, always use src chip if it is an exit chip to next mesh, regardless
                        // of load on the edge
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                        break;
                    }
                    // TODO: Ideally this should take into account the shortest path through all of the meshes to get to
                    // the target mesh This is a simple implementation that only considers the shortest path to the next
                    // mesh
                    std::uint32_t ew_distance = std::abs(
                        static_cast<std::int32_t>(src_chip_id % ew_size) -
                        static_cast<std::int32_t>(candidate_exit_chip_id % ew_size));
                    std::uint32_t ns_distance = std::abs(
                        static_cast<std::int32_t>(src_chip_id / ew_size) -
                        static_cast<std::int32_t>(candidate_exit_chip_id / ew_size));
                    std::uint32_t distance = ew_distance + ns_distance;
                    if (distance < min_distance) {
                        // optimization for latency, always use the shortest path to next mesh, regardless of load on
                        // the edge
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                        min_distance = distance;
                    } else if (distance == min_distance) {
                        const auto& edge =
                            inter_mesh_connectivity[*src_mesh_id][candidate_exit_chip_id].at(candidate_next_mesh_id);
                        if (edge.weight < min_load) {
                            min_load = edge.weight;
                            exit_chip_id = candidate_exit_chip_id;
                            next_mesh_id = candidate_next_mesh_id;
                        }
                    }
                }

                if (exit_chip_id == src_chip_id) {
                    // If src is already exit chip, use port directions to next mesh
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        inter_mesh_connectivity[*src_mesh_id][src_chip_id].at(next_mesh_id).port_direction;
                    // TODO: today we are not updating the weight of the edge, should we use weight to balance
                    //  routing traffic?
                    //  inter_mesh_connectivity[src_mesh_id][src_chip_id][next_mesh_id].weight += 1;
                } else {
                    // Use direction to exit chip from the intermesh routing table
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        this->intra_mesh_table_[src_mesh_id_val][src_chip_id][exit_chip_id];
                    // Update weight for exit chip to next mesh and src chip to exit chip
                    // inter_mesh_connectivity[src_mesh_id][exit_chip_id][next_mesh_id].weight += 1;
                    // for (auto& edge: intra_mesh_connectivity[src_mesh_id][src_chip_id]) {
                    //   if (edge.second.port_direction ==
                    //   this->inter_mesh_table_[src_mesh_id][src_chip_id][dst_mesh_id]) {
                    //       edge.second.weight += 1;
                    //       break;
                    //    }
                    //  }
                }
                this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = exit_chip_id;
                mesh_to_exit_nodes_[dst_mesh_id].push_back(FabricNodeId(MeshId{src_mesh_id}, exit_chip_id));
            }
        }
    }

    // Deduplicate mesh_to_exit_nodes_ - many chips share the same exit chip
    for (auto& [mesh_id, exit_nodes] : mesh_to_exit_nodes_) {
        std::sort(exit_nodes.begin(), exit_nodes.end(), [](const FabricNodeId& a, const FabricNodeId& b) {
            if (a.mesh_id != b.mesh_id) {
                return *a.mesh_id < *b.mesh_id;
            }
            return a.chip_id < b.chip_id;
        });
        exit_nodes.erase(std::unique(exit_nodes.begin(), exit_nodes.end()), exit_nodes.end());
    }
}

void RoutingTableGenerator::load_intermesh_connections(const AnnotatedIntermeshConnections& intermesh_connections) {
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    const_cast<MeshGraph&>(mesh_graph).load_intermesh_connections(intermesh_connections);
    this->generate_intermesh_routing_table(
        mesh_graph.get_inter_mesh_connectivity(), mesh_graph.get_intra_mesh_connectivity());
}

void RoutingTableGenerator::print_routing_tables() const {
    std::stringstream ss;
    ss << "Routing Table Generator: IntraMesh Routing Tables" << std::endl;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_table_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (ChipId src_chip_id = 0; src_chip_id < this->intra_mesh_table_[mesh_id_val].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (ChipId dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->intra_mesh_table_[mesh_id_val][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->intra_mesh_table_[mesh_id_val][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << enchantum::to_string(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Routing Table Generator: InterMesh Routing Tables" << std::endl;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->inter_mesh_table_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (ChipId src_chip_id = 0; src_chip_id < this->inter_mesh_table_[mesh_id_val].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (ChipId dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->inter_mesh_table_[mesh_id_val][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->inter_mesh_table_[mesh_id_val][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << enchantum::to_string(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

const std::vector<FabricNodeId>& RoutingTableGenerator::get_exit_nodes_routing_to_mesh(MeshId mesh_id) const {
    auto it = this->mesh_to_exit_nodes_.find(mesh_id);
    if (it != this->mesh_to_exit_nodes_.end()) {
        return it->second;
    }
    TT_THROW("No exit nodes found for mesh_id {}", *mesh_id);
}

FabricNodeId RoutingTableGenerator::get_exit_node_from_mesh_to_mesh(
    MeshId src_mesh_id, ChipId src_chip_id, MeshId dst_mesh_id) const {
    TT_FATAL(*src_mesh_id < this->exit_node_lut_.size(), "src_mesh_id out of range");
    TT_FATAL(src_chip_id < this->exit_node_lut_[*src_mesh_id].size(), "src_chip_id out of range");
    TT_FATAL(*dst_mesh_id < this->exit_node_lut_[*src_mesh_id][src_chip_id].size(), "dst_mesh_id out of range");

    ChipId exit_chip = this->exit_node_lut_[*src_mesh_id][src_chip_id][*dst_mesh_id];
    if (src_mesh_id == dst_mesh_id) {
        return FabricNodeId(src_mesh_id, src_chip_id);
    }
    TT_FATAL(
        exit_chip != std::numeric_limits<ChipId>::max(),
        "No exit chip mapped from M{}D{} to M{}",
        *src_mesh_id,
        src_chip_id,
        *dst_mesh_id);
    return FabricNodeId(src_mesh_id, exit_chip);
}
}  // namespace tt::tt_fabric
