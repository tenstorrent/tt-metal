// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    // GATED OVERLAY (skip-link sub-torus routing). Populated per-mesh in the loop below.
    // chord_family[chip] = the base-hop span of that chip's skip (Z) chord, i.e. its ring family
    // (ex4 -> span 3, ex8 -> span 7; 0 = no chord). mesh_has_skip gates the entire overlay: with
    // no intra-mesh Z edges it stays false and N/S routing is byte-identical to the base policy.
    // (Intra-mesh Z edges are added only from declared skip_links -- see mesh_graph.cpp -- so this
    //  is an exact gate on "skip_links exist".)
    std::vector<int> chord_family;
    bool mesh_has_skip = false;
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
    // Shortest hop-distance from `start` to `goal` traversing only edges whose port_direction is in
    // `allowed`. Returns -1 if unreachable. Used to compare base-ring-only vs skip-inclusive distance.
    const auto bfs_dist = [&](std::uint32_t mesh_id_val,
                              ChipId start,
                              ChipId goal,
                              std::initializer_list<RoutingDirection> allowed) -> int {
        if (start == goal) {
            return 0;
        }
        const auto is_allowed = [&](RoutingDirection d) {
            for (auto a : allowed) {
                if (a == d) {
                    return true;
                }
            }
            return false;
        };
        std::vector<int> dist(intra_mesh_connectivity[mesh_id_val].size(), -1);
        std::queue<ChipId> q;
        dist[start] = 0;
        q.push(start);
        while (!q.empty()) {
            ChipId cur = q.front();
            q.pop();
            for (const auto& [next_chip_id, edge] : intra_mesh_connectivity[mesh_id_val][cur]) {
                if (!is_allowed(edge.port_direction) || dist[next_chip_id] != -1) {
                    continue;
                }
                dist[next_chip_id] = dist[cur] + 1;
                if (next_chip_id == goal) {
                    return dist[next_chip_id];
                }
                q.push(next_chip_id);
            }
        }
        return -1;
    };
    // DEADLOCK-FREE skip-link policy along one axis: first-hop direction on a SHORTEST path that
    // spends at most ONE ring crossover. Layered BFS over (chip, crossovers_used in {0,1}), scoped
    // to {a, b, Z}. A crossover is a base hop (dir a/b) between two express nodes of DIFFERENT ring
    // families (a contact link); chords (Z) and same-family / leaf base hops are free. This gives
    // ring containment (I1) + at most one crossover (I5); it is loop-free because the safe distance
    // to `goal` strictly decreases every hop, and memoryless because the decision depends only on
    // (current, goal). Reduces to a plain shortest path when no crossover is ever possible
    // (single ring / skip-free axis). Returns NONE if `goal` is unreachable within the budget --
    // that only happens for configs outside the proven envelope (>1 accelerator ring family).
    const auto safe_first_hop_along_axis = [&](std::uint32_t mesh_id_val,
                                               ChipId src,
                                               ChipId goal,
                                               RoutingDirection a,
                                               RoutingDirection b) -> RoutingDirection {
        const std::size_t num_chips = intra_mesh_connectivity[mesh_id_val].size();
        // state = chip * 2 + crossovers_used
        std::vector<int> dist(num_chips * 2, -1);
        std::vector<RoutingDirection> first_dir(num_chips * 2, RoutingDirection::NONE);
        std::queue<int> q;
        const int start_state = static_cast<int>(src) * 2;
        dist[start_state] = 0;
        q.push(start_state);
        while (!q.empty()) {
            const int state = q.front();
            q.pop();
            const ChipId u = state / 2;
            const int c = state % 2;
            if (u == goal) {
                return first_dir[state];
            }
            // Deterministic neighbour order: base (a/b) hops before the chord (Z), then by chip id.
            // intra_mesh_connectivity is an unordered_map, so without this the tie-break between two
            // equal-length safe paths would follow hash order (non-reproducible tables). Preferring
            // the base hop keeps equal-length ties on the ring rather than taking a chord.
            std::vector<std::pair<int, RoutingDirection>> neighbours;  // (chip, dir)
            for (const auto& [v, edge] : intra_mesh_connectivity[mesh_id_val][u]) {
                const RoutingDirection d = edge.port_direction;
                if (d == a || d == b || d == RoutingDirection::Z) {
                    neighbours.emplace_back(static_cast<int>(v), d);
                }
            }
            std::sort(neighbours.begin(), neighbours.end(), [](const auto& x, const auto& y) {
                const bool x_is_z = x.second == RoutingDirection::Z;
                const bool y_is_z = y.second == RoutingDirection::Z;
                if (x_is_z != y_is_z) {
                    return y_is_z;  // base hops first
                }
                return x.first < y.first;  // then lower chip id
            });
            for (const auto& [v, d] : neighbours) {
                const bool is_crossover = (d != RoutingDirection::Z) && chord_family[u] > 0 && chord_family[v] > 0 &&
                                          chord_family[u] != chord_family[v];
                // Directional deadlock rule: a crossover into the SPARSER ring (dense->sparse, e.g.
                // ex4->ex8, where the chord span / family INCREASES) is only allowed as a TERMINAL
                // delivery hop -- the sparser node must be the destination. Sparse->dense (ex8->ex4)
                // is a free injection. This keeps the cross-ring dependency one-directional so the two
                // per-ring bubble domains cannot close a cycle. (Without it the BFS takes shorter but
                // UNSAFE non-terminal ex4->ex8 shortcuts.)
                if (is_crossover && chord_family[v] > chord_family[u] && v != static_cast<int>(goal)) {
                    continue;
                }
                const int nc = c + (is_crossover ? 1 : 0);
                if (nc > 1) {
                    continue;  // crossover budget exhausted
                }
                const int next_state = v * 2 + nc;
                if (dist[next_state] != -1) {
                    continue;
                }
                dist[next_state] = dist[state] + 1;
                first_dir[next_state] = (state == start_state) ? d : first_dir[state];
                q.push(next_state);
            }
        }
        return RoutingDirection::NONE;
    };
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_table_.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};

        // Ring decomposition depends on whether the row axis (dim 0) closes into a torus. Detect the
        // wrap edge (a base N/S hop between the first and last row):
        //   * dim-0 WRAPS: ex4 and ex8 each close their OWN ring -> DISJOINT rings. Family = the chord's
        //     base-hop span (ex4 -> 3, ex8 -> 7), so a base hop between the two families is a rationed,
        //     one-directional crossover (the full 4x32 case).
        //   * dim-0 does NOT wrap (partial column, e.g. 4x16 / 4x24): ex8 cannot close its own ring, so
        //     ex4 + ex8 form a SINGLE ring for the whole column. All chords get ONE family -> no
        //     ex4<->ex8 crossover -> routing is shortest path on that one ring. Its only bubble-gated
        //     injections are hops leaving a leaf node, which device flow-control handles; the table just
        //     needs shortest / contained / loop-free paths, which the merged single family produces.
        const int L0 = static_cast<int>(mesh_graph.get_mesh_shape(mesh_id)[0]);
        bool row_axis_wraps = false;
        for (ChipId u = 0; u < intra_mesh_connectivity[mesh_id_val].size() && !row_axis_wraps; u++) {
            const int ru = mesh_graph.chip_to_coordinate(mesh_id, u)[0];
            if (ru != 0 && ru != L0 - 1) {
                continue;
            }
            for (const auto& [v, edge] : intra_mesh_connectivity[mesh_id_val][u]) {
                if (edge.port_direction != RoutingDirection::N && edge.port_direction != RoutingDirection::S) {
                    continue;
                }
                const int rv = mesh_graph.chip_to_coordinate(mesh_id, v)[0];
                if ((ru == 0 && rv == L0 - 1) || (ru == L0 - 1 && rv == 0)) {
                    row_axis_wraps = true;
                    break;
                }
            }
        }

        // Recover the skip-link structure from the intra-mesh Z edges (gated overlay). Each express
        // node has exactly one chord; leaves have no chord (family 0). No Z edges -> mesh_has_skip
        // stays false -> N/S routing is unchanged.
        chord_family.assign(intra_mesh_connectivity[mesh_id_val].size(), 0);
        mesh_has_skip = false;
        for (ChipId u = 0; u < intra_mesh_connectivity[mesh_id_val].size(); u++) {
            for (const auto& [v, edge] : intra_mesh_connectivity[mesh_id_val][u]) {
                if (edge.port_direction == RoutingDirection::Z) {
                    mesh_has_skip = true;
                    chord_family[u] = row_axis_wraps
                                          ? bfs_dist(mesh_id_val, u, v, {RoutingDirection::N, RoutingDirection::S})
                                          : 1;  // merged single ring: one family -> no ex4<->ex8 crossover
                }
            }
        }

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
                    // GATED OVERLAY: a skip-link mesh uses the deadlock-free policy along the row
                    // (N/S) axis; every other case uses the base dimension-order policy, byte-identical
                    // to main. This is the only place the skip overlay changes routing behaviour.
                    auto direction =
                        mesh_has_skip
                            ? safe_first_hop_along_axis(
                                  mesh_id_val, src_chip_id, target_chip_id, RoutingDirection::N, RoutingDirection::S)
                            : get_shorter_direction_on_row_or_col(
                                  mesh_id_val, src_chip_id, target_chip_id, RoutingDirection::N, RoutingDirection::S);
                    this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = direction;
                    // TODO: today we are not updating the weight of the edge, should we use weight to balance
                    //  routing traffic?
                    //  intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else if (src_mesh_coord[1] != dst_mesh_coord[1]) {
                    // Move East or West. The E/W axis carries no chords, so it always uses the base
                    // policy -- identical to main.
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
