// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_mapper.hpp"

#include <algorithm>
#include <unordered_set>
#include <limits>
#include <queue>

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_types.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {

namespace {
// Encodes a MeshId and MeshHostRankId into a single 64-bit value for transport.
std::uint64_t encode_mesh_id_and_rank(MeshId mesh_id, MeshHostRankId host_rank) {
    return (static_cast<std::uint64_t>(mesh_id.get()) << 32) | static_cast<std::uint64_t>(host_rank.get());
}

std::pair<MeshId, MeshHostRankId> decode_mesh_id_and_rank(std::uint64_t encoded_value) {
    return {
        MeshId{static_cast<std::uint32_t>(encoded_value >> 32)},
        MeshHostRankId{static_cast<std::uint32_t>(encoded_value & 0xFFFFFFFF)}};
}

// Encodes/decodes a FabricNodeId (mesh_id, chip_id) into/from a 64-bit value.
std::uint64_t encode_fabric_node_id(const FabricNodeId& fabric_node_id) {
    return (static_cast<std::uint64_t>(fabric_node_id.mesh_id.get()) << 32) |
           static_cast<std::uint64_t>(fabric_node_id.chip_id);
}

FabricNodeId decode_fabric_node_id(std::uint64_t encoded_value) {
    return FabricNodeId(
        MeshId{static_cast<std::uint32_t>(encoded_value >> 32)},
        static_cast<std::uint32_t>(encoded_value & 0xFFFFFFFF));
}
}  // namespace

FabricNodeId TopologyMapper::get_fabric_node_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
    return asic_id_to_fabric_node_id_.at(asic_id);
}

FabricNodeId TopologyMapper::get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id) const {
    auto it = physical_chip_id_to_asic_id_.find(physical_chip_id);
    TT_FATAL(it != physical_chip_id_to_asic_id_.end(), "Physical chip id {} not found in mapping", physical_chip_id);
    return asic_id_to_fabric_node_id_.at(it->second);
}

chip_id_t TopologyMapper::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    auto asic_id = fabric_node_id_to_asic_id_.at(fabric_node_id);
    auto it = asic_id_to_physical_chip_id_.find(asic_id);
    TT_FATAL(
        it != asic_id_to_physical_chip_id_.end(), "Physical chip id not found for fabric node id {}", fabric_node_id);
    return it->second;
}

tt::tt_metal::AsicID TopologyMapper::get_asic_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    return fabric_node_id_to_asic_id_.at(fabric_node_id);
}

TopologyMapper::TopologyMapper(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const LocalMeshBinding& local_mesh_binding) :
    mesh_graph_(mesh_graph),
    physical_system_descriptor_(physical_system_descriptor),
    local_mesh_binding_(local_mesh_binding) {
    build_asic_physical_chip_id_mappings();
    build_mapping();
}

chip_id_t TopologyMapper::get_physical_chip_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
    return asic_id_to_physical_chip_id_.at(asic_id);
}

void TopologyMapper::build_asic_physical_chip_id_mappings() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
        tt::tt_metal::AsicID asic_id{unique_id};
        asic_id_to_physical_chip_id_.emplace(asic_id, physical_chip_id);
        physical_chip_id_to_asic_id_.emplace(physical_chip_id, asic_id);
    }
}

void TopologyMapper::build_mapping() {
    log_debug(tt::LogFabric, "TopologyMapper: Building mapping between fabric node IDs and physical ASIC IDs");

    // Check that this is not a multi-mesh-per-host system not supported by this algorithm
    TT_FATAL(
        local_mesh_binding_.mesh_ids.size() == 1,
        "Multi-mesh-per-host systems are not supported by this algorithm, please use custom fabric topology via "
        "MetalContext::set_custom_fabric_topology");

    // Build host-to-mesh mapping via distributed all-gather of local bindings.
    auto mesh_id_host_names = build_cross_host_mesh_mappings();

    // Only 1 host builds the mapping the rest will wait and use the mapping from the 1st host
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        // Find corners per host: map<host, set<AsicID>>
        auto host_corners_map = build_host_corner_mappings();

        // Locate mesh corners per mesh
        auto mesh_corners_map = build_mesh_corners_mappings(host_corners_map, mesh_id_host_names);

        // Populate fabric_node_id_to_asic_id mapping for each mesh
        populate_fabric_node_id_to_asic_id_mappings(mesh_corners_map, mesh_id_host_names);

        // Broadcast the mapping to all hosts
        broadcast_mapping_to_all_hosts();
    } else {
        // Wait for the 1st host to build the mapping
        receive_mapping_from_host(0);
    }
}

std::unordered_map<MeshId, std::unordered_set<HostName>> TopologyMapper::build_cross_host_mesh_mappings() {
    std::unordered_map<MeshId, std::unordered_set<HostName>> mesh_id_to_hosts;

    // Gather (mesh_id, host_rank) for ALL meshes owned by each rank.
    auto global_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const std::size_t world_size = *global_context->size();

    // Build MPI rank -> host name map using PhysicalSystemDescriptor's rank mapping.
    std::vector<HostName> rank_to_host(world_size);
    for (const auto& host : physical_system_descriptor_.get_all_hostnames()) {
        auto rank = physical_system_descriptor_.get_rank_for_hostname(host);
        if (rank < rank_to_host.size()) {
            rank_to_host[rank] = host;
        }
    }

    // 1) All-gather counts (how many meshes each rank owns)
    const std::uint32_t local_count = static_cast<std::uint32_t>(local_mesh_binding_.mesh_ids.size());
    std::vector<std::uint32_t> counts(world_size, 0);
    global_context->all_gather(
        ttsl::Span<std::byte>(
            reinterpret_cast<std::byte*>(const_cast<std::uint32_t*>(&local_count)), sizeof(std::uint32_t)),
        ttsl::as_writable_bytes(ttsl::Span<std::uint32_t>(counts.data(), counts.size())));

    const std::uint32_t max_count = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());

    // 2) All-gather fixed-width list of encoded (mesh_id, host_rank) per rank
    const std::uint64_t sentinel = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> send_values(max_count, sentinel);
    for (std::uint32_t i = 0; i < local_count; ++i) {
        send_values[i] = encode_mesh_id_and_rank(local_mesh_binding_.mesh_ids[i], local_mesh_binding_.host_rank);
    }

    std::vector<std::uint64_t> gathered(static_cast<std::size_t>(world_size) * max_count, sentinel);
    if (max_count > 0) {
        global_context->all_gather(
            ttsl::Span<std::byte>(
                reinterpret_cast<std::byte*>(send_values.data()), send_values.size() * sizeof(std::uint64_t)),
            ttsl::as_writable_bytes(ttsl::Span<std::uint64_t>(gathered.data(), gathered.size())));
    }

    // 3) Populate mesh_id_to_hosts using gathered data and counts
    for (std::size_t mpi_rank = 0; mpi_rank < world_size; ++mpi_rank) {
        const auto entries_for_rank = counts[mpi_rank];
        for (std::uint32_t j = 0; j < entries_for_rank; ++j) {
            const auto encoded = gathered[mpi_rank * max_count + j];
            if (encoded == sentinel) {
                continue;
            }
            const auto [mesh_id, host_rank] = decode_mesh_id_and_rank(encoded);
            const auto& host_name = rank_to_host.at(mpi_rank);
            mesh_id_to_hosts[mesh_id].insert(host_name);
        }
    }

    return mesh_id_to_hosts;
}

std::unordered_map<std::string, std::unordered_set<tt::tt_metal::AsicID>> TopologyMapper::build_host_corner_mappings()
    const {
    std::unordered_map<std::string, std::unordered_set<tt::tt_metal::AsicID>> host_corner_map;

    for (const auto& host_name : physical_system_descriptor_.get_all_hostnames()) {
        // Build local adjacency degrees (count neighbors on the same host only)
        const auto& local_asics = physical_system_descriptor_.get_asics_connected_to_host(host_name);
        if (local_asics.empty()) {
            continue;
        }

        std::unordered_map<tt::tt_metal::AsicID, int> local_degree;
        local_degree.reserve(local_asics.size());
        for (const auto& asic_id : local_asics) {
            auto asic_neighbors = physical_system_descriptor_.get_asic_neighbors(asic_id);
            int neighbor_count = 0;
            for (const auto& asic_neighbor : asic_neighbors) {
                if (physical_system_descriptor_.get_host_name_for_asic(asic_neighbor) == host_name) {
                    neighbor_count++;
                }
            }
            local_degree.emplace(asic_id, neighbor_count);
        }

        // 1x1 slice on this host
        if (local_asics.size() == 1) {
            host_corner_map[host_name].insert(local_asics[0]);
            continue;
        }

        // Classify: 1D if exactly two degree-1 endpoints and max degree <= 2, else 2D
        int max_deg = 0;
        int deg1_count = 0;
        for (const auto& [_, d] : local_degree) {
            max_deg = std::max(max_deg, d);
            if (d == 1) {
                deg1_count++;
            }
        }
        const bool is_1d = (deg1_count == 2) && (max_deg <= 2);

        if (is_1d) {
            // 1D: corners are the two endpoints (degree 1)
            for (const auto& [asic, d] : local_degree) {
                if (d == 1) {
                    host_corner_map[host_name].insert(asic);
                }
            }
        } else {
            // 2D: corners locally have degree 2
            for (const auto& [asic, d] : local_degree) {
                if (d == 2) {
                    host_corner_map[host_name].insert(asic);
                }
            }
        }
    }
    return host_corner_map;
}

std::unordered_map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> TopologyMapper::build_mesh_corners_mappings(
    const std::unordered_map<std::string, std::unordered_set<tt::tt_metal::AsicID>>& host_corners,
    const std::unordered_map<MeshId, std::unordered_set<HostName>>& mesh_id_to_host_names) const {
    std::unordered_map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> mesh_corner_map;

    // Populate corners per mesh
    for (const auto& [mesh_id, mesh_hostnames] : mesh_id_to_host_names) {
        // Get the corners for each host
        for (const auto& host_name : mesh_hostnames) {
            for (const auto& corner : host_corners.at(host_name)) {
                bool is_mesh_corner = true;
                // Check if the corner is a mesh corner
                // The mesh corner is the one that is not connected to any other host in the mesh
                for (const auto& adj_asic : physical_system_descriptor_.get_asic_neighbors(corner)) {
                    const auto& adj_host = physical_system_descriptor_.get_host_name_for_asic(adj_asic);
                    if (adj_host != host_name && mesh_hostnames.contains(adj_host)) {
                        is_mesh_corner = false;
                        break;
                    }
                }
                if (is_mesh_corner) {
                    mesh_corner_map[mesh_id].insert(corner);
                }
            }
        }
    }

    return mesh_corner_map;
}

void TopologyMapper::populate_fabric_node_id_to_asic_id_mappings(
    const std::unordered_map<MeshId, std::unordered_set<tt::tt_metal::AsicID>>& mesh_corners_map,
    const std::unordered_map<MeshId, std::unordered_set<HostName>>& mesh_id_to_host_names) {
    // Helper: gather all ASICs for hosts in this mesh
    auto gather_mesh_asics = [&](const std::unordered_set<HostName>& hostnames) {
        std::unordered_set<tt::tt_metal::AsicID> asics;
        for (const auto& host : hostnames) {
            const auto& host_asics = physical_system_descriptor_.get_asics_connected_to_host(host);
            asics.insert(host_asics.begin(), host_asics.end());
        }
        return asics;
    };

    // Helper: build adjacency within mesh (exclude remote chips)
    auto build_adjacency = [&](const std::unordered_set<tt::tt_metal::AsicID>& in_mesh_asics,
                               const std::unordered_set<HostName>& mesh_hosts) {
        std::unordered_map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>> adj;
        for (const auto& asic : in_mesh_asics) {
            std::vector<tt::tt_metal::AsicID> nbrs;
            for (const auto& n : physical_system_descriptor_.get_asic_neighbors(asic)) {
                // exclude remote chips (hosts not in this mesh)
                if (in_mesh_asics.find(n) == in_mesh_asics.end()) {
                    continue;
                }
                const auto& host = physical_system_descriptor_.get_host_name_for_asic(n);
                if (!mesh_hosts.contains(host)) {
                    continue;
                }
                nbrs.push_back(n);
            }
            adj.emplace(asic, std::move(nbrs));
        }
        return adj;
    };

    auto bfs_dist = [&](const std::unordered_map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>>& adj,
                        tt::tt_metal::AsicID start) {
        std::unordered_map<tt::tt_metal::AsicID, std::uint32_t> dist;
        std::queue<tt::tt_metal::AsicID> queue;
        dist[start] = 0;
        queue.push(start);
        while (!queue.empty()) {
            auto cur = queue.front();
            queue.pop();
            auto it = adj.find(cur);
            if (it == adj.end()) {
                continue;
            }
            for (auto nbr : it->second) {
                if (!dist.contains(nbr)) {
                    dist[nbr] = dist[cur] + 1;
                    queue.push(nbr);
                }
            }
        }
        return dist;
    };

    auto pick_canonical_corner = [&](const std::unordered_set<tt::tt_metal::AsicID>& corner_asics) {
        // Pick deterministically by (host_name, asic_id value)
        return *std::min_element(corner_asics.begin(), corner_asics.end(), [&](const auto& a, const auto& b) {
            const auto& ha = physical_system_descriptor_.get_host_name_for_asic(a);
            const auto& hb = physical_system_descriptor_.get_host_name_for_asic(b);
            if (ha != hb) {
                return ha < hb;
            }
            return *a < *b;
        });
    };

    for (const auto& [mesh_id, corner_set] : mesh_corners_map) {
        const auto& mesh_hosts = mesh_id_to_host_names.at(mesh_id);
        const auto mesh_shape = mesh_graph_.get_mesh_shape(mesh_id);
        const std::uint32_t ns = mesh_shape[0];
        const std::uint32_t ew = mesh_shape[1];
        const std::size_t num_nodes = static_cast<std::size_t>(ns) * static_cast<std::size_t>(ew);

        // Collect ASICs belonging to this mesh and build adjacency (intra-mesh only)
        auto in_mesh_asics = gather_mesh_asics(mesh_hosts);
        auto adj = build_adjacency(in_mesh_asics, mesh_hosts);

        // Special case 1x1
        if (ns == 1 && ew == 1) {
            TT_FATAL(!in_mesh_asics.empty(), "No ASICs found for 1x1 mesh {}", mesh_id);
            auto single_asic = *in_mesh_asics.begin();
            // Logical chips from mesh graph
            auto logical_container = mesh_graph_.get_chip_ids(mesh_id);
            auto logical_chip_id = logical_container.at(MeshCoordinate{0, 0});
            fabric_node_id_to_asic_id_.emplace(FabricNodeId(mesh_id, logical_chip_id), single_asic);
            asic_id_to_fabric_node_id_.emplace(single_asic, FabricNodeId(mesh_id, logical_chip_id));
            continue;
        }

        // Determine if 1D
        bool is_1d = (ns == 1) || (ew == 1);

        if (is_1d) {
            // Pick one endpoint deterministically
            TT_FATAL(
                corner_set.size() == 2,
                "Missing connections to form a uniform 1D mesh, run build/test/tt_metal/tt_fabric/test_system_health "
                "to check if all chips are connected",
                mesh_id,
                corner_set.size());
            auto start = pick_canonical_corner(corner_set);

            // DFS to produce linear ordering
            std::vector<tt::tt_metal::AsicID> path;
            path.reserve(num_nodes);
            std::unordered_set<tt::tt_metal::AsicID> visited;
            std::function<bool(tt::tt_metal::AsicID)> dfs = [&](tt::tt_metal::AsicID cur) -> bool {
                visited.insert(cur);
                path.push_back(cur);
                if (path.size() == num_nodes) {
                    return true;
                }
                for (auto nbr : adj[cur]) {
                    if (!visited.contains(nbr)) {
                        if (dfs(nbr)) {
                            return true;
                        }
                    }
                }
                path.pop_back();
                visited.erase(cur);
                return false;
            };
            bool ok = dfs(start);
            TT_FATAL(ok && path.size() == num_nodes, "Failed to generate 1D ordering for mesh {}", mesh_id);

            // Map to logical chips
            auto logical_container = mesh_graph_.get_chip_ids(mesh_id);
            std::size_t idx = 0;
            for (const auto& [_, logical_chip_id] : logical_container) {
                auto asic = path[idx++];
                fabric_node_id_to_asic_id_.emplace(FabricNodeId(mesh_id, logical_chip_id), asic);
                asic_id_to_fabric_node_id_.emplace(asic, FabricNodeId(mesh_id, logical_chip_id));
            }
            continue;
        }

        // 2D case
        TT_FATAL(
            corner_set.size() == 4,
            "Missing connections to form a uniform 2D mesh, run build/test/tt_metal/tt_fabric/test_system_health to "
            "check if all chips are connected",
            mesh_id,
            corner_set.size());
        auto nw = pick_canonical_corner(corner_set);
        auto dist_from_nw = bfs_dist(adj, nw);

        // Identify NE as the corner at distance (ew - 1) from NW
        tt::tt_metal::AsicID ne_candidate = nw;
        for (const auto& c : corner_set) {
            if (c == nw) {
                continue;
            }
            if (dist_from_nw.contains(c) && dist_from_nw[c] == (ew - 1)) {
                ne_candidate = c;
                break;
            }
        }
        TT_FATAL(ne_candidate != nw, "Failed to identify NE corner for mesh {}", mesh_id);
        auto dist_from_ne = bfs_dist(adj, ne_candidate);

        // Assign coordinates by Manhattan distance identities
        std::unordered_map<tt::tt_metal::AsicID, std::pair<std::uint32_t, std::uint32_t>> coord_of_asic;
        for (const auto& asic : in_mesh_asics) {
            TT_FATAL(
                dist_from_nw.contains(asic) && dist_from_ne.contains(asic),
                "Distance maps incomplete for mesh {}",
                mesh_id);
            auto a = dist_from_nw[asic];
            auto b = dist_from_ne[asic];
            // i = (a + b - (ew - 1)) / 2; j = a - i
            TT_FATAL((a + b) >= (ew - 1), "Invalid distances for node in mesh {}", mesh_id);
            auto i2 = static_cast<std::int64_t>(a) + static_cast<std::int64_t>(b) - static_cast<std::int64_t>(ew - 1);
            TT_FATAL(i2 % 2 == 0, "Non-integer row index derived for mesh {}", mesh_id);
            std::uint32_t i = static_cast<std::uint32_t>(i2 / 2);
            std::uint32_t j = static_cast<std::uint32_t>(a - i);
            TT_FATAL(i < ns && j < ew, "Derived coordinate out of bounds for mesh {}: ({}, {})", mesh_id, i, j);
            coord_of_asic[asic] = {i, j};
        }

        // Map to logical chips using MeshGraph's logical layout
        auto logical_container = mesh_graph_.get_chip_ids(mesh_id);
        for (const auto& [coord, logical_chip_id] : logical_container) {
            std::uint32_t i = coord[0];
            std::uint32_t j = coord[1];
            // find asic with this coordinate
            tt::tt_metal::AsicID found_asic{0};
            bool found = false;
            for (const auto& [asic, ij] : coord_of_asic) {
                if (ij.first == i && ij.second == j) {
                    found_asic = asic;
                    found = true;
                    break;
                }
            }
            TT_FATAL(found, "No ASIC found for logical coordinate ({}, {}) in mesh {}", i, j, mesh_id);
            fabric_node_id_to_asic_id_.emplace(FabricNodeId(mesh_id, logical_chip_id), found_asic);
            asic_id_to_fabric_node_id_.emplace(found_asic, FabricNodeId(mesh_id, logical_chip_id));
        }
    }
}

void TopologyMapper::broadcast_mapping_to_all_hosts() {
    using namespace tt::tt_metal::distributed::multihost;
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    const std::size_t world_size = *distributed_context.size();
    if (world_size <= 1) {
        return;  // single-host, nothing to broadcast
    }

    // Only controller broadcasts
    constexpr std::size_t CONTROLLER_RANK = 0;
    auto my_rank = *distributed_context.rank();
    if (my_rank != CONTROLLER_RANK) {
        return;
    }

    // Streaming format:
    // [u32 count]
    // repeated 'count' times send a fixed-size record:
    //   record = [u64 asic_id][u64 encoded_fabric_node_id]
    auto serialize_u64 = [](std::vector<uint8_t>& buf, std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
        }
    };

    const std::uint32_t count = static_cast<std::uint32_t>(fabric_node_id_to_asic_id_.size());

    for (std::size_t peer = 0; peer < world_size; ++peer) {
        if (peer == CONTROLLER_RANK) {
            continue;
        }

        // Send count first (synchronous send to ensure receiver posted recv)
        std::uint32_t count_copy = count;
        distributed_context.ssend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&count_copy), sizeof(count_copy)),
            Rank{static_cast<uint32_t>(peer)},
            Tag{0});

        // Send one record at a time using synchronous send
        for (const auto& [fabric_node_id, asic_id] : fabric_node_id_to_asic_id_) {
            std::vector<uint8_t> record;
            record.reserve(16);
            serialize_u64(record, *asic_id);
            const std::uint64_t encoded_fn = encode_fabric_node_id(fabric_node_id);
            serialize_u64(record, encoded_fn);

            distributed_context.ssend(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(record.data(), record.size())),
                Rank{static_cast<uint32_t>(peer)},
                Tag{0});
        }
    }
}

void TopologyMapper::receive_mapping_from_host(int rank) {
    using namespace tt::tt_metal::distributed::multihost;
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    auto my_rank = *distributed_context.rank();
    if (static_cast<int>(my_rank) == rank) {
        return;  // sender does not receive
    }

    // Receive count, then 'count' fixed-size records
    std::uint32_t count = 0;
    distributed_context.recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&count), sizeof(count)),
        Rank{static_cast<uint32_t>(rank)},
        Tag{0});

    fabric_node_id_to_asic_id_.clear();
    asic_id_to_fabric_node_id_.clear();

    auto read_u64_from = [&](const std::vector<uint8_t>& buf, std::size_t& idx) -> std::uint64_t {
        TT_FATAL(idx + 8 <= buf.size(), "Deserializer overflow reading u64");
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (static_cast<std::uint64_t>(buf[idx++]) << (8 * i));
        }
        return v;
    };

    for (std::uint32_t i = 0; i < count; ++i) {
        std::vector<uint8_t> record(16);
        distributed_context.recv(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(record.data(), record.size())),
            Rank{static_cast<uint32_t>(rank)},
            Tag{0});

        std::size_t idx = 0;
        const auto asic_val = read_u64_from(record, idx);
        const auto encoded_fn = read_u64_from(record, idx);
        tt::tt_metal::AsicID asic_id{asic_val};

        FabricNodeId fn = decode_fabric_node_id(encoded_fn);
        fabric_node_id_to_asic_id_.emplace(fn, asic_id);
        asic_id_to_fabric_node_id_.emplace(asic_id, fn);
    }

    TT_FATAL(
        fabric_node_id_to_asic_id_.size() == count && asic_id_to_fabric_node_id_.size() == count,
        "Topology mapping size mismatch after streaming receive");
}

std::map<FabricNodeId, chip_id_t> TopologyMapper::get_local_logical_mesh_chip_id_to_physical_chip_id_mapping() const {
    std::map<FabricNodeId, chip_id_t> mapping;
    // Only return the mapping for local logical mesh ids
    for (const auto& asic_id :
         physical_system_descriptor_.get_asics_connected_to_host(physical_system_descriptor_.my_host_name())) {
        mapping[get_fabric_node_id_from_asic_id(asic_id)] = get_physical_chip_id_from_asic_id(asic_id);
    }
    return mapping;
}

}  // namespace tt::tt_fabric
