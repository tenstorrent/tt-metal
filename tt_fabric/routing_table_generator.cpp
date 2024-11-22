// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric/routing_table_generator.hpp"

#include <queue>
#include <memory>

namespace tt::tt_fabric {
RoutingTableGenerator::RoutingTableGenerator(const std::string& mesh_graph_desc_yaml_file) {
    this->mesh_graph_ = std::make_unique<MeshGraph>(mesh_graph_desc_yaml_file);
    // Use IntraMeshConnectivity to size all variables
    const auto& intra_mesh_connectivity = this->mesh_graph_->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->mesh_graph_->get_inter_mesh_connectivity();
    this->intra_mesh_table_.resize(intra_mesh_connectivity.size());
    this->inter_mesh_table_.resize(intra_mesh_connectivity.size());
    for (mesh_id_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
        this->intra_mesh_table_[mesh_id].resize(intra_mesh_connectivity[mesh_id].size());
        this->inter_mesh_table_[mesh_id].resize(intra_mesh_connectivity[mesh_id].size());
        for (auto& devices_in_mesh : this->intra_mesh_table_[mesh_id]) {
            // intra_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other chips in the mesh
            devices_in_mesh.resize(intra_mesh_connectivity[mesh_id].size());
        }
        for (auto& devices_in_mesh : this->inter_mesh_table_[mesh_id]) {
            // inter_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other meshes
            devices_in_mesh.resize(intra_mesh_connectivity.size());
        }
    }
    // Generate the intra mesh routing table
    this->generate_intramesh_routing_table(intra_mesh_connectivity);

    // Generate the inter mesh routing table
    this->generate_intermesh_routing_table(inter_mesh_connectivity, intra_mesh_connectivity);
}

void RoutingTableGenerator::generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity) {
    for (mesh_id_t mesh_id = 0; mesh_id < this->intra_mesh_table_.size(); mesh_id++) {
        for (chip_id_t src_chip_id = 0; src_chip_id < this->intra_mesh_table_[mesh_id].size(); src_chip_id++) {
            for (chip_id_t dst_chip_id = 0; dst_chip_id < this->intra_mesh_table_[mesh_id].size(); dst_chip_id++) {
                int row_size = 8;  // TODO: get this from mesh descriptor
                uint32_t src_x = src_chip_id / row_size;
                uint32_t src_y = src_chip_id % row_size;
                uint32_t dst_x = dst_chip_id / row_size;
                uint32_t dst_y = dst_chip_id % row_size;

                uint32_t next_chip_id;
                // X first routing, traverse rows first
                if (src_x > dst_x) {
                    // Move North
                    next_chip_id = src_chip_id - row_size;
                    this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_id] =
                        intra_mesh_connectivity[mesh_id][src_chip_id].at(next_chip_id).port_direction;
                    // intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else if (src_x < dst_x) {
                    // Move South
                    next_chip_id = src_chip_id + row_size;
                    this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_id] =
                        intra_mesh_connectivity[mesh_id][src_chip_id].at(next_chip_id).port_direction;
                    // intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else if (src_y < dst_y) {
                    // Move East
                    next_chip_id = src_chip_id + 1;
                    this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_id] =
                        intra_mesh_connectivity[mesh_id][src_chip_id].at(next_chip_id).port_direction;
                    // intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else if (src_y > dst_y) {
                    // Move West
                    next_chip_id = src_chip_id - 1;
                    this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_id] =
                        intra_mesh_connectivity[mesh_id][src_chip_id].at(next_chip_id).port_direction;
                    // intra_mesh_connectivity[mesh_id][src_chip_id][next_chip_id].weight += 1;
                } else {
                    // No movement
                    // TODO: what value do we put for this entry? If we pack table entries to 4 bits
                    // any number is a valid port id. Do we assume FW will never try to access table entry to itself?
                    this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::C;
                }
            }
        }
    }
}

// Shortest Path
// TODO: Put into mesh algorithms?
std::vector<std::vector<std::vector<std::pair<chip_id_t, mesh_id_t>>>> RoutingTableGenerator::get_paths_to_all_meshes(
    mesh_id_t src, const InterMeshConnectivity& inter_mesh_connectivity) {
    // TODO: add more tests for this
    uint32_t num_meshes = inter_mesh_connectivity.size();
    bool visited[num_meshes];
    std::fill_n(visited, num_meshes, false);

    // paths[target_mesh_id][path_count][next_chip and next_mesh];
    std::vector<std::vector<std::vector<std::pair<chip_id_t, mesh_id_t>>>> paths;
    paths.resize(num_meshes);
    paths[src] = {{{}}};

    uint32_t dist[num_meshes];
    std::fill_n(dist, num_meshes, std::numeric_limits<uint32_t>::max());
    dist[src] = 0;

    std::queue<mesh_id_t> q;
    q.push(src);
    visited[src] = true;
    // BFS
    while (!q.empty()) {
        mesh_id_t current_mesh_id = q.front();
        q.pop();

        // Captures paths at the chip level
        for (chip_id_t chip_in_mesh = 0; chip_in_mesh < inter_mesh_connectivity[current_mesh_id].size();
             chip_in_mesh++) {
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[current_mesh_id][chip_in_mesh]) {
                if (!visited[connected_mesh_id]) {
                    q.push(connected_mesh_id);
                    visited[connected_mesh_id] = true;
                }
                if (dist[connected_mesh_id] > dist[current_mesh_id] + 1) {
                    dist[connected_mesh_id] = dist[current_mesh_id] + 1;
                    paths[connected_mesh_id] = paths[current_mesh_id];
                    for (auto& path : paths[connected_mesh_id]) {
                        path.push_back({chip_in_mesh, connected_mesh_id});
                    }
                } else if (dist[connected_mesh_id] == dist[current_mesh_id] + 1) {
                    // another possible path discovered
                    for (auto& path : paths[current_mesh_id]) {
                        paths[connected_mesh_id].push_back(path);
                    }

                    paths[connected_mesh_id][paths[connected_mesh_id].size() - 1].push_back(
                        {chip_in_mesh, connected_mesh_id});
                }
            }
        }
    }
    /*
    for (mesh_id_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
        std::cout << "Path: from src " << src << " to " << mesh_id << " : ";
        for (const auto& path: paths[mesh_id]) {
          std::cout << std::endl;
          for (const auto& id: path) {
              std::cout << " chip " << id.first << " mesh " << id.second << " ";
          }
        }
        std::cout << std::endl;
    }*/
    return paths;
}

void RoutingTableGenerator::generate_intermesh_routing_table(
    const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& intra_mesh_connectivity) {
    for (mesh_id_t src_mesh_id = 0; src_mesh_id < this->inter_mesh_table_.size(); src_mesh_id++) {
        auto paths = get_paths_to_all_meshes(src_mesh_id, inter_mesh_connectivity);
        for (chip_id_t src_chip_id = 0; src_chip_id < this->inter_mesh_table_[src_mesh_id].size(); src_chip_id++) {
            for (mesh_id_t dst_mesh_id = 0; dst_mesh_id < this->inter_mesh_table_.size(); dst_mesh_id++) {
                if (dst_mesh_id == src_mesh_id) {
                    // inter mesh table entry from mesh to itself
                    this->inter_mesh_table_[src_mesh_id][src_chip_id][dst_mesh_id] = RoutingDirection::C;
                    continue;
                }
                auto& candidate_paths = paths[dst_mesh_id];
                std::uint32_t min_load = std::numeric_limits<std::uint32_t>::max();
                TT_ASSERT(candidate_paths.size() > 0, "Expecting at least one path to target mesh");
                chip_id_t exit_chip_id = candidate_paths[0][1].first;
                mesh_id_t next_mesh_id = candidate_paths[0][1].second;
                for (auto& path : candidate_paths) {
                    // First element is itself, second is next mesh
                    TT_ASSERT(path.size() > 0, "Expecting at least two entries in path");
                    chip_id_t candidate_exit_chip_id = path[1].first;  // first element is the first hop to target mesh
                    mesh_id_t candidate_next_mesh_id = path[1].second;
                    if (candidate_exit_chip_id == src_chip_id) {
                        // optimization for latency, always use src chip if it is an exit chip to next mesh, regardless
                        // of load on the edge
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                        break;
                    }
                    const auto& edge =
                        inter_mesh_connectivity[src_mesh_id][candidate_exit_chip_id].at(candidate_next_mesh_id);
                    if (edge.weight < min_load) {
                        min_load = edge.weight;
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                    }
                }

                if (exit_chip_id == src_chip_id) {
                    // If src is already exit chip, use port directions to next mesh
                    this->inter_mesh_table_[src_mesh_id][src_chip_id][dst_mesh_id] =
                        inter_mesh_connectivity[src_mesh_id][src_chip_id].at(next_mesh_id).port_direction;
                    // inter_mesh_connectivity[src_mesh_id][src_chip_id][next_mesh_id].weight += 1;
                } else {
                    // Use direction to exit chip from the intermesh routing table
                    this->inter_mesh_table_[src_mesh_id][src_chip_id][dst_mesh_id] =
                        this->intra_mesh_table_[src_mesh_id][src_chip_id].at(exit_chip_id);
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
            }
        }
    }
}

void RoutingTableGenerator::print_routing_tables() const {
    std::stringstream ss;
    ss << "Routing Table Generator: IntraMesh Routing Tables" << std::endl;
    for (mesh_id_t mesh_id = 0; mesh_id < this->intra_mesh_table_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (chip_id_t src_chip_id = 0; src_chip_id < this->intra_mesh_table_[mesh_id].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (chip_id_t dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->intra_mesh_table_[mesh_id][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->intra_mesh_table_[mesh_id][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << magic_enum::enum_name(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Routing Table Generator: InterMesh Routing Tables" << std::endl;
    for (mesh_id_t mesh_id = 0; mesh_id < this->inter_mesh_table_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (chip_id_t src_chip_id = 0; src_chip_id < this->inter_mesh_table_[mesh_id].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (chip_id_t dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->inter_mesh_table_[mesh_id][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->inter_mesh_table_[mesh_id][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << magic_enum::enum_name(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}
}  // namespace tt::tt_fabric
