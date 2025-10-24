// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_mapper.hpp"

#include <algorithm>
#include <unordered_set>
#include <limits>
#include <functional>

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_types.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdlib>

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

// Helper function to get timeout duration for topology mapping operations
std::chrono::duration<float> get_topology_mapping_timeout() {
    auto timeout = tt::tt_metal::MetalContext::instance().rtoptions().get_timeout_duration_for_operations();
    if (timeout.count() <= 0.0f) {
        timeout = std::chrono::duration<float>(60.0f);
    }
    return timeout;
}

// Generic timeout mechanism that can handle different types of operations
template <typename OperationType, typename... Args>
void execute_with_timeout(OperationType&& operation, const std::string& operation_description, Args&&... args) {
    auto timeout = get_topology_mapping_timeout();
    std::atomic<bool> operation_completed{false};
    std::atomic<bool> operation_failed{false};
    std::exception_ptr exception_ptr{nullptr};

    // Run operation in a separate thread
    std::thread operation_thread([&]() {
        try {
            operation(std::forward<Args>(args)...);
            operation_completed = true;
        } catch (...) {
            exception_ptr = std::current_exception();
            operation_failed = true;
        }
    });

    // Wait for completion or timeout
    auto start = std::chrono::steady_clock::now();
    while (!operation_completed && !operation_failed) {
        std::this_thread::yield();
        if (timeout.count() > 0.0f) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - start).count();
            if (elapsed >= timeout.count()) {
                // Timeout occurred - detach the thread and throw an error
                operation_thread.detach();
                TT_THROW(
                    "Timeout while waiting for {} operation. One or more hosts may have failed.",
                    operation_description);
            }
        }
    }

    // Wait for thread to complete
    if (operation_thread.joinable()) {
        operation_thread.join();
    }

    // Re-throw any exception that occurred in the thread
    if (operation_failed && exception_ptr) {
        std::rethrow_exception(exception_ptr);
    }
}

// Specialized wrapper for request-based operations (like irecv)
template<typename RequestType>
void wait_for_request_with_timeout(RequestType& req, const std::string& operation_description, int rank) {
    auto timeout = get_topology_mapping_timeout();
    auto start = std::chrono::steady_clock::now();

    while (!req->test()) {
        std::this_thread::yield();
        if (timeout.count() > 0.0f) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - start).count();
            if (elapsed >= timeout.count()) {
                req->cancel();
                TT_THROW(
                    "Timeout while waiting for {} from rank {}. Controller likely failed.",
                    operation_description,
                    rank);
            }
        }
    }
}

// Wrapper for all_gather operations
void all_gather_with_timeout(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::stl::Span<std::byte> send_buf,
    tt::stl::Span<std::byte> recv_buf,
    const std::string& operation_description) {
    execute_with_timeout(
        [&context](tt::stl::Span<std::byte> send, tt::stl::Span<std::byte> recv) {
            context->all_gather(send, recv);
        },
        operation_description,
        send_buf, recv_buf);
}
}  // namespace

FabricNodeId TopologyMapper::get_fabric_node_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
    return asic_id_to_fabric_node_id_.at(asic_id);
}

FabricNodeId TopologyMapper::get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const {
    auto it = physical_chip_id_to_asic_id_.find(physical_chip_id);
    TT_FATAL(it != physical_chip_id_to_asic_id_.end(), "Physical chip id {} not found in mapping", physical_chip_id);
    return asic_id_to_fabric_node_id_.at(it->second);
}

ChipId TopologyMapper::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
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
    // Initialize containers; population will occur during build_mapping
    mesh_host_ranks_.clear();
    mesh_host_rank_coord_ranges_.clear();
    build_asic_physical_chip_id_mappings();
    build_mapping();
}

ChipId TopologyMapper::get_physical_chip_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
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
    auto mesh_id_host_names = build_host_mesh_mapping();

    // Only 1 host builds the mapping the rest will wait and use the mapping from the 1st host
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        // Build logical and physical adjacency maps
        auto adjacency_map_logical = build_adjacency_map_logical(mesh_id_host_names);
        auto adjacency_map_physical = build_adjacency_map_physical(mesh_id_host_names);

        // Use sat solver algo to preserve the logical connectivity in the physical topology
        populate_fabric_node_id_to_asic_id_mappings(adjacency_map_physical, adjacency_map_logical);

        // Broadcast the mapping to all hosts
        broadcast_mapping_to_all_hosts();
    } else {
        // Wait for the 1st host to build the mapping
        receive_mapping_from_host(0);
    }

    // Build host rank containers now that mapping is complete
    rebuild_host_rank_structs_from_mapping();
}

std::unordered_map<MeshId, std::unordered_set<HostName>> TopologyMapper::build_host_mesh_mapping() {
    std::unordered_map<MeshId, std::unordered_set<HostName>> mesh_id_to_hosts;

    // Gather (mesh_id, host_rank) for ALL meshes owned by each rank, but only if multi-host.
    auto global_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const std::size_t world_size = *global_context->size();

    // Single-host or uninitialized distributed context: compute mapping locally without any collectives
    if (world_size <= 1) {
        for (const auto& mesh_id : local_mesh_binding_.mesh_ids) {
            mesh_id_to_hosts[mesh_id].insert(physical_system_descriptor_.my_host_name());
        }
        return mesh_id_to_hosts;
    }

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
    all_gather_with_timeout(
        global_context,
        ttsl::Span<std::byte>(
            reinterpret_cast<std::byte*>(const_cast<std::uint32_t*>(&local_count)), sizeof(std::uint32_t)),
        ttsl::as_writable_bytes(ttsl::Span<std::uint32_t>(counts.data(), counts.size())),
        "mesh count all_gather");

    const std::uint32_t max_count = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());

    // 2) All-gather fixed-width list of encoded (mesh_id, host_rank) per rank
    const std::uint64_t sentinel = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> send_values(max_count, sentinel);
    for (std::uint32_t i = 0; i < local_count; ++i) {
        send_values[i] = encode_mesh_id_and_rank(local_mesh_binding_.mesh_ids[i], local_mesh_binding_.host_rank);
    }

    std::vector<std::uint64_t> gathered(static_cast<std::size_t>(world_size) * max_count, sentinel);
    if (max_count > 0) {
        all_gather_with_timeout(
            global_context,
            ttsl::Span<std::byte>(
                reinterpret_cast<std::byte*>(send_values.data()), send_values.size() * sizeof(std::uint64_t)),
            ttsl::as_writable_bytes(ttsl::Span<std::uint64_t>(gathered.data(), gathered.size())),
            "mesh binding all_gather");
    }

    // 3) Populate mesh_id_to_hosts using gathered data and counts
    for (std::size_t mpi_rank = 0; mpi_rank < world_size; ++mpi_rank) {
        const auto entries_for_rank = counts[mpi_rank];
        for (std::uint32_t j = 0; j < entries_for_rank; ++j) {
            const auto encoded = gathered[(mpi_rank * max_count) + j];
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

std::unordered_map<MeshId, LogicalAdjacencyMap> TopologyMapper::build_adjacency_map_logical(
    HostMeshMapping& mesh_id_to_host_names) const {
    std::unordered_map<MeshId, LogicalAdjacencyMap> adjacency_map;

    auto get_local_adjacents = [&](tt::tt_fabric::FabricNodeId fabric_node_id, MeshId mesh_id) {
        auto adjacent_map = mesh_graph_.get_intra_mesh_connectivity()[*mesh_id][fabric_node_id.chip_id];

        std::vector<tt::tt_fabric::FabricNodeId> adjacents;
        bool relaxed = mesh_graph_.is_intra_mesh_policy_relaxed(mesh_id);
        for (const auto& [neighbor_chip_id, edge] : adjacent_map) {
            size_t repeat_count = relaxed ? 1 : edge.connected_chip_ids.size();
            for (size_t i = 0; i < repeat_count; ++i) {
                adjacents.push_back(tt::tt_fabric::FabricNodeId(mesh_id, neighbor_chip_id));
            }
        }
        return adjacents;
    };

    for (const auto& [mesh_id, _] : mesh_id_to_host_names) {
        LogicalAdjacencyMap logical_adjacency_map;
        for (const auto& [_, chip_id] : mesh_graph_.get_chip_ids(mesh_id)) {
            auto fabric_node_id = tt::tt_fabric::FabricNodeId(mesh_id, chip_id);
            logical_adjacency_map[fabric_node_id] = get_local_adjacents(fabric_node_id, mesh_id);
        }
        adjacency_map[mesh_id] = logical_adjacency_map;
    }

    return adjacency_map;
}

std::unordered_map<MeshId, PhysicalAdjacencyMap> TopologyMapper::build_adjacency_map_physical(
    HostMeshMapping& mesh_id_to_host_names) const {
    std::unordered_map<MeshId, PhysicalAdjacencyMap> adjacency_map;

    auto get_local_adjacents =
        [&](tt::tt_metal::AsicID asic_id, MeshId mesh_id, const std::unordered_set<HostName>& mesh_hostnames) {
            std::vector<tt::tt_metal::AsicID> adjacents;
            for (const auto& neighbor : physical_system_descriptor_.get_asic_neighbors(asic_id)) {
                // Make sure that the neighbor is in the mesh
                if (mesh_hostnames.contains(physical_system_descriptor_.get_host_name_for_asic(neighbor))) {
                    auto connections = physical_system_descriptor_.get_eth_connections(asic_id, neighbor);
                    for (size_t i = 0; i < connections.size(); ++i) {
                        adjacents.push_back(neighbor);
                    }
                }
            }
            return adjacents;
        };

    for (const auto& [mesh_id, mesh_hostnames] : mesh_id_to_host_names) {
        PhysicalAdjacencyMap physical_adjacency_map;
        for (const auto& host_name : mesh_hostnames) {
            for (const auto& asic_id : physical_system_descriptor_.get_asics_connected_to_host(host_name)) {
                physical_adjacency_map[asic_id] = get_local_adjacents(asic_id, mesh_id, mesh_hostnames);
            }
        }
        adjacency_map[mesh_id] = physical_adjacency_map;
    }

    return adjacency_map;
}

void TopologyMapper::populate_fabric_node_id_to_asic_id_mappings(
    const std::unordered_map<MeshId, PhysicalAdjacencyMap>& adjacency_map_physical,
    const std::unordered_map<MeshId, LogicalAdjacencyMap>& adjacency_map_logical) {
    for (const auto& [mesh_id, log_adj] : adjacency_map_logical) {
        auto& phys_adj = adjacency_map_physical.at(mesh_id);

        std::vector<FabricNodeId> log_nodes;
        for (const auto& p : log_adj) {
            log_nodes.push_back(p.first);
        }

        std::vector<tt::tt_metal::AsicID> phys_nodes;
        for (const auto& p : phys_adj) {
            phys_nodes.push_back(p.first);
        }

        size_t n_log = log_nodes.size();
        size_t n_phys = phys_nodes.size();

        TT_FATAL(
            n_log <= n_phys,
            "Graph specified in MGD is larger than the discovered physical topology for mesh {}, please modify your "
            "MGD or use ./build/test/tt_metal/tt_fabric/test_system_health to check if all chips are connected",
            mesh_id.get());

        std::unordered_map<FabricNodeId, size_t> log_to_idx;
        for (size_t i = 0; i < n_log; ++i) {
            log_to_idx[log_nodes[i]] = i;
        }

        std::unordered_map<tt::tt_metal::AsicID, size_t> phys_to_idx;
        for (size_t j = 0; j < n_phys; ++j) {
            phys_to_idx[phys_nodes[j]] = j;
        }

        std::vector<std::unordered_map<size_t, size_t>> log_adj_count(n_log);
        for (size_t i = 0; i < n_log; ++i) {
            const auto& vec = log_adj.at(log_nodes[i]);
            auto& count = log_adj_count[i];
            for (const auto& neigh : vec) {
                count[log_to_idx.at(neigh)]++;
            }
        }

        std::vector<std::unordered_map<size_t, size_t>> phys_adj_count(n_phys);
        for (size_t j = 0; j < n_phys; ++j) {
            const auto& vec = phys_adj.at(phys_nodes[j]);
            auto& count = phys_adj_count[j];
            for (const auto& neigh : vec) {
                count[phys_to_idx.at(neigh)]++;
            }
        }

        bool relaxed = mesh_graph_.is_intra_mesh_policy_relaxed(mesh_id);

        std::vector<size_t> log_deg(n_log, 0);
        for (size_t i = 0; i < n_log; ++i) {
            const auto& m = log_adj_count[i];
            for (const auto& p : m) {
                log_deg[i] += relaxed ? 1 : p.second;
            }
        }

        std::vector<size_t> phys_deg(n_phys, 0);
        for (size_t j = 0; j < n_phys; ++j) {
            const auto& m = phys_adj_count[j];
            for (const auto& p : m) {
                phys_deg[j] += relaxed ? 1 : p.second;
            }
        }

        // Sort log_order by decreasing log_deg
        std::vector<size_t> log_order(n_log);
        std::iota(log_order.begin(), log_order.end(), 0);
        std::stable_sort(
            log_order.begin(), log_order.end(), [&](size_t a, size_t b) { return log_deg[a] > log_deg[b]; });

        // Sort phys_order by decreasing phys_deg
        std::vector<size_t> phys_order(n_phys);
        std::iota(phys_order.begin(), phys_order.end(), 0);
        std::stable_sort(
            phys_order.begin(), phys_order.end(), [&](size_t a, size_t b) { return phys_deg[a] > phys_deg[b]; });

        // mapping[logical_index] = physical_index
        std::vector<int> mapping(n_log, -1);
        std::vector<bool> used(n_phys, false);

        std::function<bool(size_t)> dfs = [&](size_t pos) -> bool {
            if (pos == n_log) {
                return true;
            }

            size_t li = log_order[pos];

            for (size_t jj = 0; jj < n_phys; ++jj) {
                size_t j = phys_order[jj];
                if (used[j]) {
                    continue;
                }
                if (phys_deg[j] < log_deg[li]) {
                    continue;
                }

                bool ok = true;
                for (const auto& p : log_adj_count[li]) {
                    size_t lk = p.first;
                    size_t log_count = p.second;
                    int pk_i = mapping[lk];
                    if (pk_i == -1) {
                        continue;
                    }
                    size_t pk = static_cast<size_t>(pk_i);
                    size_t phys_count = phys_adj_count[j][pk];  // defaults to 0

                    if (relaxed) {
                        if (log_count > 0 && phys_count == 0) {
                            ok = false;
                            break;
                        }
                    } else {
                        if (log_count > phys_count) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (!ok) {
                    continue;
                }

                used[j] = true;
                mapping[li] = static_cast<int>(j);
                if (dfs(pos + 1)) {
                    return true;
                }
                mapping[li] = -1;
                used[j] = false;
            }
            return false;
        };

        bool found = dfs(0);
        TT_FATAL(
            found,
            "Graph specified in MGD could not fit in the discovered physical topology for mesh {}, please modify your "
            "MGD or use ./build/test/tt_metal/tt_fabric/test_system_health to check if all chips are connected",
            mesh_id.get());

        for (size_t i = 0; i < n_log; ++i) {
            FabricNodeId fn = log_nodes[i];
            size_t j = static_cast<size_t>(mapping[i]);
            tt::tt_metal::AsicID asic = phys_nodes[j];
            fabric_node_id_to_asic_id_.emplace(fn, asic);
            asic_id_to_fabric_node_id_.emplace(asic, fn);
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

    // If not in distributed context, nothing to receive
    if (*distributed_context.size() <= 1) {
        return;
    }

    auto my_rank = *distributed_context.rank();
    if (static_cast<int>(my_rank) == rank) {
        return;  // sender does not receive
    }

    // Receive count, then 'count' fixed-size records
    std::uint32_t count = 0;
    {
        auto req = distributed_context.irecv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&count), sizeof(count)),
            Rank{static_cast<uint32_t>(rank)},
            Tag{0});

        wait_for_request_with_timeout(req, "topology mapping header", rank);
    }

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
        auto req = distributed_context.irecv(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(record.data(), record.size())),
            Rank{static_cast<uint32_t>(rank)},
            Tag{0});

        wait_for_request_with_timeout(
            req, "topology mapping record " + std::to_string(i + 1) + " of " + std::to_string(count), rank);

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

std::map<FabricNodeId, ChipId> TopologyMapper::get_local_logical_mesh_chip_id_to_physical_chip_id_mapping() const {
    std::map<FabricNodeId, ChipId> mapping;
    const auto& my_host = physical_system_descriptor_.my_host_name();
    // Only include ASICs that are part of the current fabric mapping and reside on this host
    for (const auto& [asic_id, fabric_node_id] : asic_id_to_fabric_node_id_) {
        if (physical_system_descriptor_.get_host_name_for_asic(asic_id) == my_host) {
            mapping[fabric_node_id] = get_physical_chip_id_from_asic_id(asic_id);
        }
    }
    return mapping;
}

// Replacement MeshGraph-like APIs backed by TopologyMapper
const MeshContainer<MeshHostRankId>& TopologyMapper::get_host_ranks(MeshId mesh_id) const {
    TT_FATAL(*mesh_id < mesh_host_ranks_.size(), "TopologyMapper: mesh_id {} not found", mesh_id);
    return mesh_host_ranks_[*mesh_id];
}

MeshShape TopologyMapper::get_mesh_shape(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    if (host_rank.has_value()) {
        auto it = mesh_host_rank_coord_ranges_.find(std::make_pair(mesh_id, *host_rank));
        TT_FATAL(
            it != mesh_host_rank_coord_ranges_.end(),
            "TopologyMapper: host_rank {} not found for mesh {}",
            *host_rank,
            *mesh_id);
        return it->second.shape();
    }
    return mesh_graph_.get_mesh_shape(mesh_id);
}

MeshCoordinateRange TopologyMapper::get_coord_range(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    if (host_rank.has_value()) {
        auto it = mesh_host_rank_coord_ranges_.find(std::make_pair(mesh_id, *host_rank));
        TT_FATAL(
            it != mesh_host_rank_coord_ranges_.end(),
            "TopologyMapper: host_rank {} not found for mesh {}",
            *host_rank,
            *mesh_id);
        return it->second;
    }
    return mesh_graph_.get_coord_range(mesh_id);
}

std::optional<MeshHostRankId> TopologyMapper::get_host_rank_for_chip(MeshId mesh_id, ChipId chip_id) const {
    // Compute coord and check which host range contains it
    MeshCoordinate coord = mesh_graph_.chip_to_coordinate(mesh_id, chip_id);
    for (const auto& [key, range] : mesh_host_rank_coord_ranges_) {
        if (key.first == mesh_id && range.contains(coord)) {
            return key.second;
        }
    }
    return std::nullopt;
}

MeshContainer<ChipId> TopologyMapper::get_chip_ids(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    // Return global or submesh chip ids using the same indexing convention as MeshGraph.
    if (!host_rank.has_value()) {
        auto shape = mesh_graph_.get_mesh_shape(mesh_id);
        std::vector<ChipId> chip_ids(shape.mesh_size());
        std::iota(chip_ids.begin(), chip_ids.end(), 0);
        return MeshContainer<ChipId>(shape, chip_ids);
    }

    // Submesh: iterate over coord range and collect logical chip ids
    MeshCoordinateRange coord_range = get_coord_range(mesh_id, host_rank);
    MeshShape sub_shape = coord_range.shape();
    std::vector<ChipId> sub_chip_ids;
    sub_chip_ids.reserve(sub_shape.mesh_size());
    for (const auto& coord : coord_range) {
        // Convert coordinate to logical chip id using global mesh shape
        auto chip = mesh_graph_.coordinate_to_chip(mesh_id, coord);
        sub_chip_ids.push_back(chip);
    }
    return MeshContainer<ChipId>(sub_shape, sub_chip_ids);
}

void TopologyMapper::rebuild_host_rank_structs_from_mapping() {
    // Derive per-mesh host sets and per-host coord ranges from current mapping
    std::unordered_map<MeshId, std::unordered_set<HostName>> mesh_to_hosts;
    std::unordered_map<MeshId, std::unordered_map<HostName, MeshCoordinateRange>> mesh_host_to_range;
    // For wraparound-aware construction, accumulate coordinates per host then compute minimal circular ranges.
    std::unordered_map<MeshId, std::unordered_map<HostName, std::vector<MeshCoordinate>>> mesh_host_to_coords;

    // Precompute coordinate per chip from MeshGraph
    std::unordered_map<MeshId, std::unordered_map<ChipId, MeshCoordinate>> per_mesh_chip_to_coord;
    for (const auto& [fabric_node_id, _] : fabric_node_id_to_asic_id_) {
        auto& m = per_mesh_chip_to_coord[fabric_node_id.mesh_id];
        if (m.find(fabric_node_id.chip_id) == m.end()) {
            m.emplace(
                fabric_node_id.chip_id, mesh_graph_.chip_to_coordinate(fabric_node_id.mesh_id, fabric_node_id.chip_id));
        }
    }

    // Accumulate coordinates per host
    for (const auto& [fabric_node_id, asic_id] : fabric_node_id_to_asic_id_) {
        const auto host = physical_system_descriptor_.get_host_name_for_asic(asic_id);
        mesh_to_hosts[fabric_node_id.mesh_id].insert(host);
        const auto coord = per_mesh_chip_to_coord[fabric_node_id.mesh_id].at(fabric_node_id.chip_id);
        mesh_host_to_coords[fabric_node_id.mesh_id][host].push_back(coord);
    }

    // Build minimal wraparound-aware ranges per host
    for (const auto& [mesh_id, host_coords_map] : mesh_host_to_coords) {
        const auto shape = mesh_graph_.get_mesh_shape(mesh_id);
        auto& range_map = mesh_host_to_range[mesh_id];
        for (const auto& [host, coords] : host_coords_map) {
            // Compute unique values per dimension
            std::vector<uint32_t> unique_r;
            std::vector<uint32_t> unique_c;
            unique_r.reserve(coords.size());
            unique_c.reserve(coords.size());
            for (const auto& c : coords) {
                unique_r.push_back(c[0]);
                unique_c.push_back(c[1]);
            }
            auto uniq = [](std::vector<uint32_t>& v) {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            };
            uniq(unique_r);
            uniq(unique_c);

            auto minimal_circular_span = [](const std::vector<uint32_t>& values, uint32_t dim_size) {
                // Returns pair(start, end) in circular sense; start may be > end to indicate wrap.
                if (values.empty()) {
                    return std::pair<uint32_t, uint32_t>(0, 0);
                }
                if (values.size() == 1) {
                    return std::pair<uint32_t, uint32_t>(values[0], values[0]);
                }
                if (values.size() >= dim_size) {
                    return std::pair<uint32_t, uint32_t>(0u, dim_size - 1);
                }
                // values must be sorted unique
                std::vector<uint32_t> v = values;
                // compute maximum gap between consecutive values on circle
                uint32_t max_gap = 0;
                size_t max_gap_idx = 0;  // gap between v[i] and v[i+1] (wrapping at end)
                for (size_t i = 0; i < v.size(); ++i) {
                    uint32_t a = v[i];
                    uint32_t b = (i + 1 < v.size()) ? v[i + 1] : v[0];
                    uint32_t gap = (i + 1 < v.size()) ? (b - a) : ((dim_size - a) + b);
                    if (gap > max_gap) {
                        max_gap = gap;
                        max_gap_idx = i;
                    }
                }
                // minimal arc excludes the largest gap; start is next value, end is current value
                uint32_t start = (max_gap_idx + 1 < v.size()) ? v[max_gap_idx + 1] : v[0];
                uint32_t end = v[max_gap_idx];
                return std::make_pair(start, end);
            };

            auto [row_start, row_end] = minimal_circular_span(unique_r, shape[0]);
            auto [col_start, col_end] = minimal_circular_span(unique_c, shape[1]);
            MeshCoordinate start(row_start, col_start);
            MeshCoordinate end(row_end, col_end);

            bool wraparound = row_start > row_end || col_start > col_end;
            if (wraparound) {
                range_map.emplace(host, MeshCoordinateRange(start, end, shape));
            } else {
                range_map.emplace(host, MeshCoordinateRange(start, end));
            }
        }
    }

    // Build MeshContainer<MeshHostRankId> by row-major ordering of host tile ranges
    // Determine host grid using unique start rows/cols
    mesh_host_ranks_.clear();
    mesh_host_rank_coord_ranges_.clear();
    std::size_t max_mesh_index = 0;
    for (const auto& [mid, _] : mesh_to_hosts) {
        max_mesh_index = std::max<std::size_t>(max_mesh_index, *mid + 1);
    }
    mesh_host_ranks_.resize(max_mesh_index, MeshContainer<MeshHostRankId>(MeshShape{1, 1}, MeshHostRankId{0}));
    for (const auto& [mesh_id, hosts] : mesh_to_hosts) {
        const auto& range_map = mesh_host_to_range.at(mesh_id);
        std::set<std::uint32_t> rows;
        std::set<std::uint32_t> cols;
        for (const auto& [host, range] : range_map) {
            rows.insert(range.start_coord()[0]);
            cols.insert(range.start_coord()[1]);
        }
        MeshShape host_grid_shape(rows.size(), cols.size());
        std::vector<MeshHostRankId> host_rank_values(host_grid_shape.mesh_size(), MeshHostRankId{0});
        std::vector<std::uint32_t> row_list(rows.begin(), rows.end());
        std::vector<std::uint32_t> col_list(cols.begin(), cols.end());
        auto row_index = [&](std::uint32_t r) {
            return std::distance(row_list.begin(), std::find(row_list.begin(), row_list.end(), r));
        };
        auto col_index = [&](std::uint32_t c) {
            return std::distance(col_list.begin(), std::find(col_list.begin(), col_list.end(), c));
        };
        // Map host -> normalized host-rank id aligned with distributed ranks for this mesh
        std::unordered_map<HostName, std::uint32_t> host_to_rank;
        std::uint32_t base_rank = std::numeric_limits<std::uint32_t>::max();
        for (const auto& [host, _] : range_map) {
            std::uint32_t rank = physical_system_descriptor_.get_rank_for_hostname(host);
            host_to_rank[host] = rank;
            base_rank = std::min(base_rank, rank);
        }
        for (std::uint32_t r : row_list) {
            for (std::uint32_t c : col_list) {
                // find host whose range starts at (r,c)
                for (const auto& [host, range] : range_map) {
                    if (range.start_coord()[0] == r && range.start_coord()[1] == c) {
                        std::size_t idx = (row_index(r) * host_grid_shape[1]) + col_index(c);
                        std::uint32_t norm_rank = host_to_rank[host] - base_rank;
                        MeshHostRankId host_rank_val{norm_rank};
                        if (idx < host_rank_values.size()) {
                            host_rank_values[idx] = host_rank_val;
                        }
                        mesh_host_rank_coord_ranges_.insert({{mesh_id, host_rank_val}, range});
                        break;
                    }
                }
            }
        }
        mesh_host_ranks_[*mesh_id] = MeshContainer<MeshHostRankId>(host_grid_shape, host_rank_values);
    }
}

}  // namespace tt::tt_fabric
