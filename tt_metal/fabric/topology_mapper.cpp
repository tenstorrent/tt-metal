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
    local_mesh_binding_(local_mesh_binding),
    fixed_asic_position_pinnings_({}) {
    // Initialize containers; population will occur during build_mapping
    mesh_host_ranks_.clear();
    mesh_host_rank_coord_ranges_.clear();
    build_asic_physical_chip_id_mappings();
    build_mapping();
}

// Removed bus-id pinning constructor

TopologyMapper::TopologyMapper(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const LocalMeshBinding& local_mesh_binding,
    const std::vector<std::pair<AsicPosition, FabricNodeId>>& fixed_asic_position_pinnings) :
    mesh_graph_(mesh_graph),
    physical_system_descriptor_(physical_system_descriptor),
    local_mesh_binding_(local_mesh_binding),
    fixed_asic_position_pinnings_(fixed_asic_position_pinnings) {
    mesh_host_ranks_.clear();
    mesh_host_rank_coord_ranges_.clear();
    build_asic_physical_chip_id_mappings();
    build_mapping();
}

ChipId TopologyMapper::get_physical_chip_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
    auto asic_id_it = asic_id_to_physical_chip_id_.find(asic_id);
    TT_FATAL(asic_id_it != asic_id_to_physical_chip_id_.end(), "Physical chip id not found for ASIC id {}", asic_id);
    return asic_id_it->second;
}

void TopologyMapper::build_asic_physical_chip_id_mappings() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
        tt::tt_metal::AsicID asic_id{unique_id};
        asic_id_to_physical_chip_id_.emplace(asic_id, physical_chip_id);
        physical_chip_id_to_asic_id_.emplace(physical_chip_id, asic_id);
    }

    // Check the physical chip asic ids from UMD cluster with the physical chip asic ids from the physical system descriptor
    for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
        tt::tt_metal::AsicID asic_id{unique_id};
        auto asic_ids_for_host = physical_system_descriptor_.get_asics_connected_to_host(physical_system_descriptor_.my_host_name());
        TT_FATAL(std::find(asic_ids_for_host.begin(), asic_ids_for_host.end(), asic_id) != asic_ids_for_host.end(), "Asic id {} in UMD cluster not found for in Physical System {}", asic_id, physical_system_descriptor_.my_host_name());
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

    auto asic_id_to_mesh_rank = build_asic_id_to_mesh_rank_mapping();
    auto fabric_node_id_to_mesh_rank = build_fabric_node_id_to_mesh_rank_mapping();

    // Only 1 host builds the mapping the rest will wait and use the mapping from the 1st host
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        // Build logical and physical adjacency maps
        auto adjacency_map_logical = build_adjacency_map_logical(mesh_id_host_names);
        auto adjacency_map_physical = build_adjacency_map_physical(mesh_id_host_names);

        print_logical_adjacency_map(adjacency_map_logical);
        print_physical_adjacency_map(adjacency_map_physical);

        // Use sat solver algo to preserve the logical connectivity in the physical topology
        for (const auto& mesh_id : mesh_graph_.get_mesh_ids()) {
            populate_fabric_node_id_to_asic_id_mappings(
                mesh_id,
                adjacency_map_physical.at(mesh_id),
                adjacency_map_logical.at(mesh_id),
                asic_id_to_mesh_rank.at(mesh_id),
                fabric_node_id_to_mesh_rank.at(mesh_id));
        }

        // Broadcast the mapping to all hosts
        broadcast_mapping_to_all_hosts();
    } else {
        // Wait for the 1st host to build the mapping
        receive_mapping_from_host(0);
    }

    // Build host rank containers now that mapping is complete
    rebuild_host_rank_structs_from_mapping(asic_id_to_mesh_rank);
}

std::unordered_map<MeshId, std::unordered_set<HostName>> TopologyMapper::build_host_mesh_mapping() const {
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

std::unordered_map<MeshId, std::unordered_map<FabricNodeId, MeshHostRankId>>
TopologyMapper::build_fabric_node_id_to_mesh_rank_mapping() const {
    std::unordered_map<MeshId, std::unordered_map<FabricNodeId, MeshHostRankId>> mapping;
    for (const auto& mesh_id : mesh_graph_.get_mesh_ids()) {
        for (const auto& [_, chip_id] : mesh_graph_.get_chip_ids(mesh_id)) {
            auto host_rank = mesh_graph_.get_host_rank_for_chip(mesh_id, chip_id);
            TT_FATAL(host_rank.has_value(), "Fabric node id {} not found", FabricNodeId(mesh_id, chip_id));
            mapping[mesh_id][FabricNodeId(mesh_id, chip_id)] = host_rank.value();
        }
    }
    return mapping;
}

std::unordered_map<MeshId, std::unordered_map<tt::tt_metal::AsicID, MeshHostRankId>>
TopologyMapper::build_asic_id_to_mesh_rank_mapping() const {
    std::unordered_map<MeshId, std::unordered_map<tt::tt_metal::AsicID, MeshHostRankId>> mapping;
    auto global_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const std::size_t world_size = *global_context->size();

    if (world_size <= 1) {
        for (const auto& mesh_id : local_mesh_binding_.mesh_ids) {
            for (const auto& asic_id :
                 physical_system_descriptor_.get_asics_connected_to_host(physical_system_descriptor_.my_host_name())) {
                mapping[mesh_id][asic_id] = local_mesh_binding_.host_rank;
            }
        }
        return mapping;
    }

    std::vector<HostName> rank_to_host(world_size);
    for (const auto& host : physical_system_descriptor_.get_all_hostnames()) {
        auto rank = physical_system_descriptor_.get_rank_for_hostname(host);
        if (rank < rank_to_host.size()) {
            rank_to_host[rank] = host;
        }
    }

    const std::uint32_t local_count = static_cast<std::uint32_t>(local_mesh_binding_.mesh_ids.size());
    std::vector<std::uint32_t> counts(world_size, 0);
    all_gather_with_timeout(
        global_context,
        ttsl::Span<std::byte>(
            reinterpret_cast<std::byte*>(const_cast<std::uint32_t*>(&local_count)), sizeof(std::uint32_t)),
        ttsl::as_writable_bytes(ttsl::Span<std::uint32_t>(counts.data(), counts.size())),
        "mesh count all_gather");

    const std::uint32_t max_count = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());

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

    for (std::size_t mpi_rank = 0; mpi_rank < world_size; ++mpi_rank) {
        const auto entries_for_rank = counts[mpi_rank];
        for (std::uint32_t j = 0; j < entries_for_rank; ++j) {
            const auto encoded = gathered[(mpi_rank * max_count) + j];
            if (encoded == sentinel) {
                continue;
            }
            const auto [mesh_id, host_rank] = decode_mesh_id_and_rank(encoded);
            const auto& host_name = rank_to_host.at(mpi_rank);
            auto asics = physical_system_descriptor_.get_asics_connected_to_host(host_name);
            for (const auto& asic : asics) {
                mapping[mesh_id][asic] = host_rank;
            }
        }
    }
    return mapping;
}

std::unordered_map<MeshId, LogicalAdjacencyMap> TopologyMapper::build_adjacency_map_logical(
    HostMeshMapping& mesh_id_to_host_names) const {
    std::unordered_map<MeshId, LogicalAdjacencyMap> adjacency_map;

    auto get_local_adjacents = [&](tt::tt_fabric::FabricNodeId fabric_node_id, MeshId mesh_id) {
        auto adjacent_map = mesh_graph_.get_intra_mesh_connectivity()[*mesh_id][fabric_node_id.chip_id];

        std::vector<tt::tt_fabric::FabricNodeId> adjacents;
        for (const auto& [neighbor_chip_id, edge] : adjacent_map) {
            adjacents.push_back(tt::tt_fabric::FabricNodeId(mesh_id, neighbor_chip_id));
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
                    adjacents.push_back(neighbor);
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

// NOTE: This mapping algorithm uses nested lambdas and deep control flow for
// pruning and search. Refactoring would be non-trivial and risks regressions,
// so we suppress the cognitive-complexity check for this function.
// NOLINTBEGIN(readability-function-cognitive-complexity)
void TopologyMapper::populate_fabric_node_id_to_asic_id_mappings(
    const MeshId mesh_id,
    const PhysicalAdjacencyMap& adjacency_map_physical,
    const LogicalAdjacencyMap& adjacency_map_logical,
    const std::unordered_map<tt::tt_metal::AsicID, MeshHostRankId>& asic_id_to_mesh_rank,
    const std::unordered_map<FabricNodeId, MeshHostRankId>& fabric_node_id_to_mesh_rank) {
    auto& phys_adj = adjacency_map_physical;
    auto& log_adj = adjacency_map_logical;

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

    std::vector<std::vector<size_t>> log_adj_idx(n_log);
    for (size_t i = 0; i < n_log; ++i) {
        for (const auto& neigh : log_adj.at(log_nodes[i])) {
            log_adj_idx[i].push_back(log_to_idx.at(neigh));
        }
        std::sort(log_adj_idx[i].begin(), log_adj_idx[i].end());
    }

    std::unordered_map<tt::tt_metal::AsicID, size_t> phys_to_idx;
    for (size_t i = 0; i < n_phys; ++i) {
        phys_to_idx[phys_nodes[i]] = i;
    }

    std::vector<std::vector<size_t>> phys_adj_idx(n_phys);
    for (size_t i = 0; i < n_phys; ++i) {
        for (const auto& neigh : phys_adj.at(phys_nodes[i])) {
            auto it = phys_to_idx.find(neigh);
            if (it != phys_to_idx.end()) {
                phys_adj_idx[i].push_back(it->second);
            }
        }
        std::sort(phys_adj_idx[i].begin(), phys_adj_idx[i].end());
    }

    // mapping[logical_index] = physical_index
    std::vector<int> mapping(n_log, -1);
    std::vector<bool> used(n_phys, false);

    // Precompute degrees for pruning (needed for early checks on pinned nodes)
    std::vector<size_t> log_deg(n_log, 0);
    for (size_t i = 0; i < n_log; ++i) {
        log_deg[i] = log_adj_idx[i].size();
    }
    std::vector<size_t> phys_deg(n_phys, 0);
    for (size_t j = 0; j < n_phys; ++j) {
        phys_deg[j] = phys_adj_idx[j].size();
    }

    // Emit initial stats for debugging
    auto emit_degree_hist = [&](const std::vector<size_t>& degs) {
        std::map<size_t, size_t> hist;
        for (auto d : degs) {
            hist[d]++;
        }
        std::string s = "{";
        bool first = true;
        for (const auto& [d, c] : hist) {
            if (!first) {
                s += ", ";
            }
            first = false;
            s += std::to_string(d) + ":" + std::to_string(c);
        }
        s += "}";
        return s;
    };
    log_info(
        tt::LogFabric,
        "TopologyMapper mapping start (mesh={}): n_log={}, n_phys={}, log_deg_hist={}, phys_deg_hist={}",
        mesh_id.get(),
        n_log,
        n_phys,
        emit_degree_hist(log_deg),
        emit_degree_hist(phys_deg));

    // Candidate restrictions for logical indices pinned by ASIC position (tray, location).
    // If entry is empty, no restriction; otherwise, only listed physical indices are allowed.
    std::vector<std::vector<size_t>> restricted_phys_indices_for_logical(n_log);
    if (!fixed_asic_position_pinnings_.empty()) {
        // Validate uniqueness of pins for this mesh and apply
        std::unordered_map<FabricNodeId, AsicPosition> first_pinnings;

        for (const auto& [pos, fabric_node] : fixed_asic_position_pinnings_) {
            if (fabric_node.mesh_id != mesh_id) {
                continue;  // pin for another mesh
            }

            TT_FATAL(
                log_to_idx.find(fabric_node) != log_to_idx.end(),
                "Pinned fabric node {} not found in logical mesh {}",
                fabric_node,
                mesh_id.get());

            auto [it, inserted] = first_pinnings.try_emplace(fabric_node, pos);
            if (!inserted) {
                const auto& prev_pos = it->second;
                TT_THROW(
                    "Fabric node {} in mesh {} is pinned to multiple ASIC positions: (tray {}, loc {}) and (tray "
                    "{}, loc {})",
                    fabric_node,
                    mesh_id.get(),
                    *prev_pos.first,
                    *prev_pos.second,
                    *pos.first,
                    *pos.second);
            }

            // Find matching physical indices in this mesh for the pinned ASIC position (across any host)
            std::vector<size_t> matches;
            for (size_t j = 0; j < n_phys; ++j) {
                auto asic = phys_nodes[j];
                auto tray = physical_system_descriptor_.get_tray_id(asic);
                auto loc = physical_system_descriptor_.get_asic_location(asic);
                if (tray == pos.first && loc == pos.second) {
                    matches.push_back(j);
                }
            }

            if (matches.empty()) {
                TT_THROW(
                    "Pinned ASIC position (tray {}, loc {}) not found among physical ASICs participating in mesh "
                    "{}.",
                    *pos.first,
                    *pos.second,
                    mesh_id.get());
            }

            size_t li = log_to_idx.at(fabric_node);
            restricted_phys_indices_for_logical[li] = std::move(matches);
        }
    }

    // Degrees already computed above

    // Fast path: if logical graph is a single path (two endpoints with degree 1; all others degree <=2),
    // map it using a linear path-extension DFS over the physical graph to avoid heavy general search.
    auto try_fast_path_for_logical_chain = [&]() -> bool {
        std::vector<size_t> endpoints;
        for (size_t i = 0; i < n_log; ++i) {
            if (log_deg[i] == 1) {
                endpoints.push_back(i);
            }
            if (log_deg[i] > 2) {
                return false;
            }
        }
        if (endpoints.size() != 2) {
            return false;
        }

        // Build ordered logical path indices from one endpoint
        std::vector<size_t> log_order;
        log_order.reserve(n_log);
        std::vector<bool> seen(n_log, false);
        size_t curr = endpoints[0];
        size_t prev = n_log;  // sentinel
        for (size_t k = 0; k < n_log; ++k) {
            log_order.push_back(curr);
            seen[curr] = true;
            size_t next_candidate = n_log;
            for (size_t nb : log_adj_idx[curr]) {
                if (nb != prev && !seen[nb]) {
                    next_candidate = nb;
                    break;
                }
            }
            prev = curr;
            curr = next_candidate;
            if (k + 1 < n_log && curr == n_log) {
                return false;  // disconnected
            }
        }

        // Reachability check helper to prevent dead-ends
        auto reachable_unused_count = [&](size_t start_phys) -> size_t {
            std::vector<char> vis(n_phys, 0);
            std::vector<size_t> q;
            q.reserve(n_phys);
            if (used[start_phys]) {
                return 0;
            }
            q.push_back(start_phys);
            vis[start_phys] = 1;
            size_t qi = 0;
            size_t cnt = 0;
            while (qi < q.size()) {
                size_t u = q[qi++];
                cnt++;
                for (size_t v : phys_adj_idx[u]) {
                    if (!vis[v] && !used[v]) {
                        vis[v] = 1;
                        q.push_back(v);
                    }
                }
            }
            return cnt;
        };

        std::function<bool(size_t, size_t)> place = [&](size_t idx_in_path, size_t prev_phys) -> bool {
            if (idx_in_path == n_log) {
                return true;
            }
            size_t li = log_order[idx_in_path];
            if (idx_in_path == 0) {
                // Symmetry break: iterate physical starts in deterministic order
                for (size_t pj = 0; pj < n_phys; ++pj) {
                    if (used[pj]) {
                        continue;
                    }
                    if (phys_deg[pj] < log_deg[li]) {
                        continue;
                    }
                    used[pj] = true;
                    mapping[li] = static_cast<int>(pj);
                    bool ok = place(idx_in_path + 1, pj);
                    if (ok) {
                        return true;
                    }
                    mapping[li] = -1;
                    used[pj] = false;
                }
                return false;
            } else {
                // Next must be an unused neighbor of prev_phys
                // Early capacity check: remaining logicals must fit in reachable component from some neighbor
                size_t remain = n_log - idx_in_path;
                for (size_t pj : phys_adj_idx[prev_phys]) {
                    if (used[pj]) {
                        continue;
                    }
                    if (phys_deg[pj] < log_deg[li]) {
                        continue;
                    }
                    // Reachability pruning
                    size_t reach = reachable_unused_count(pj);
                    if (reach < remain) {
                        continue;
                    }
                    used[pj] = true;
                    mapping[li] = static_cast<int>(pj);
                    if (place(idx_in_path + 1, pj)) {
                        return true;
                    }
                    mapping[li] = -1;
                    used[pj] = false;
                }
                return false;
            }
        };

        bool ok = place(0, n_phys);
        if (ok) {
            log_info(tt::LogFabric, "Fast-path path-graph mapping succeeded for mesh {}", mesh_id.get());
        } else {
            log_debug(tt::LogFabric, "Fast-path path-graph mapping failed; falling back to general DFS");
        }
        return ok;
    };

    if (try_fast_path_for_logical_chain()) {
        // mapping already populated; build maps
        for (size_t i = 0; i < n_log; ++i) {
            TT_FATAL(mapping[i] >= 0, "Internal error: fast-path produced incomplete mapping");
            FabricNodeId fn = log_nodes[i];
            tt::tt_metal::AsicID asic = phys_nodes[static_cast<size_t>(mapping[i])];
            fabric_node_id_to_asic_id_.emplace(fn, asic);
            asic_id_to_fabric_node_id_.emplace(asic, fn);
        }
        return;  // next mesh
    }

    // We'll select the next logical node dynamically: pick the unmapped node
    // with the most already-mapped neighbors (most-constraining). Tie-break by MRV.
    // Additional tie-break: when no neighbors are mapped yet, prefer lower-degree (endpoints first)
    auto select_next_logical = [&](const std::vector<int>& mapping_ref, const std::vector<bool>& used_ref) {
        size_t best_li = n_log;
        size_t best_mapped_neigh = 0;
        size_t best_cand_count = (std::numeric_limits<size_t>::max)();
        size_t best_log_deg = (std::numeric_limits<size_t>::max)();

        for (size_t li = 0; li < n_log; ++li) {
            if (mapping_ref[li] != -1) {
                continue;
            }
            size_t mapped_neigh_count = 0;
            for (size_t v : log_adj_idx[li]) {
                if (mapping_ref[v] != -1) {
                    mapped_neigh_count++;
                }
            }
            // Compute candidate count under current partial assignment
            size_t cand_count = 0;
            for (size_t j = 0; j < n_phys; ++j) {
                if (used_ref[j] || phys_deg[j] < log_deg[li]) {
                    continue;
                }
                bool ok_local = true;
                for (size_t v : log_adj_idx[li]) {
                    int pj = mapping_ref[v];
                    if (pj == -1) {
                        continue;
                    }
                    // logical edge must be present physically
                    if (!std::binary_search(phys_adj_idx[j].begin(), phys_adj_idx[j].end(), static_cast<size_t>(pj))) {
                        ok_local = false;
                        break;
                    }
                }
                if (ok_local) {
                    cand_count++;
                }
            }
            if (best_li == n_log || mapped_neigh_count > best_mapped_neigh ||
                (mapped_neigh_count == best_mapped_neigh &&
                 ((best_mapped_neigh == 0 && log_deg[li] < best_log_deg) ||
                  (best_mapped_neigh != 0 && cand_count < best_cand_count)))) {
                best_li = li;
                best_mapped_neigh = mapped_neigh_count;
                best_cand_count = cand_count;
                best_log_deg = log_deg[li];
            }
        }
        return best_li;
    };

    // Memoization of failed states: include prefix mapping to avoid false negatives
    std::unordered_set<std::uint64_t> failed_states;
    auto hash_state = [&](size_t /*pos*/) -> std::uint64_t {
        const std::uint64_t fnv_offset = 1469598103934665603ull;
        const std::uint64_t fnv_prime = 1099511628211ull;
        std::uint64_t h = fnv_offset;
        for (size_t li = 0; li < n_log; ++li) {
            if (mapping[li] != -1) {
                std::uint64_t pairh =
                    (static_cast<std::uint64_t>(li) << 32) ^ static_cast<std::uint64_t>(mapping[li] + 1);
                h ^= pairh;
                h *= fnv_prime;
            }
        }
        return h;
    };

    // Debug counters and timing for visibility into search behavior
    std::size_t dfs_calls = 0;
    auto dfs_start = std::chrono::steady_clock::now();

    std::function<bool(size_t)> dfs = [&](size_t pos) -> bool {
        if (pos == n_log) {
            return true;
        }

        // Periodic progress logging to help diagnose search blow-ups
        dfs_calls++;
        if ((dfs_calls & ((1u << 18) - 1)) == 0) {  // every ~262k calls
            std::size_t assigned = 0;
            for (auto v : mapping) {
                assigned += (v != -1);
            }
            auto now = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - dfs_start).count();
            log_info(
                tt::LogFabric,
                "TopologyMapper DFS progress: calls={}, assigned={}/{}, failed_states={}, elapsed_ms={}",
                dfs_calls,
                assigned,
                n_log,
                failed_states.size(),
                ms);
        }

        std::uint64_t key = hash_state(pos);
        if (failed_states.find(key) != failed_states.end()) {
            return false;
        }

        // Select next logical node dynamically
        size_t li = select_next_logical(mapping, used);
        if (li == n_log) {
            return false;
        }

        // Candidate generation with symmetry breaking for first two assignments
        std::vector<size_t> candidates;
        candidates.reserve(n_phys);

        // If this logical node has restricted candidates from pinning, use those; otherwise try all viable anchors
        if (!restricted_phys_indices_for_logical[li].empty()) {
            for (size_t j : restricted_phys_indices_for_logical[li]) {
                if (j < n_phys && !used[j] && phys_deg[j] >= log_deg[li]) {
                    candidates.push_back(j);
                }
            }
            if (candidates.empty()) {
                return false;
            }
        } else {
            for (size_t j = 0; j < n_phys; ++j) {
                if (used[j] || phys_deg[j] < log_deg[li]) {
                    continue;
                }
                candidates.push_back(j);
            }
        }

        // Order candidates by degree gap, then index for determinism
        std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
            size_t da =
                (phys_deg[a] >= log_deg[li]) ? (phys_deg[a] - log_deg[li]) : (std::numeric_limits<size_t>::max)();
            size_t db =
                (phys_deg[b] >= log_deg[li]) ? (phys_deg[b] - log_deg[li]) : (std::numeric_limits<size_t>::max)();
            if (da != db) {
                return da < db;
            }
            return a < b;
        });

        // Periodic selection summary
        if ((dfs_calls & ((1u << 16) - 1)) == 0) {
            size_t mapped_neigh_count = 0;
            for (size_t v : log_adj_idx[li]) {
                if (mapping[v] != -1) {
                    mapped_neigh_count++;
                }
            }
            log_info(
                tt::LogFabric,
                "DFS select li={}, log_deg={}, mapped_neigh={}, candidates={}",
                li,
                log_deg[li],
                mapped_neigh_count,
                candidates.size());
        }

        for (size_t j : candidates) {
            if (fabric_node_id_to_mesh_rank.at(log_nodes[li]) != asic_id_to_mesh_rank.at(phys_nodes[j])) {
                continue;
            }
            // Debug: occasionally emit candidate summary for selected logical index
            if ((dfs_calls & ((1u << 18) - 1)) == 1) {
                log_debug(
                    tt::LogFabric,
                    "DFS step: li={}, log_deg={}, candidate_phys_idx=j={}, phys_deg[j]={}, cand_count={}",
                    li,
                    log_deg[li],
                    j,
                    phys_deg[j],
                    candidates.size());
            }
            // Local consistency: enforce that logical edges are present physically (allow extra phys edges)
            bool ok = true;
            for (size_t lk = 0; lk < n_log; ++lk) {
                int pk_i = mapping[lk];
                if (pk_i == -1) {
                    continue;
                }
                size_t pk = static_cast<size_t>(pk_i);
                bool log_connected = std::binary_search(log_adj_idx[li].begin(), log_adj_idx[li].end(), lk);
                bool phys_connected = std::binary_search(phys_adj_idx[j].begin(), phys_adj_idx[j].end(), pk);
                if (log_connected && !phys_connected) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                if ((dfs_calls & ((1u << 17) - 1)) == 0) {
                    log_debug(tt::LogFabric, "Prune: local consistency failed for li={}, phys_j={}", li, j);
                }
                continue;
            }

            // Forward checking: ensure candidate has enough unused neighbors to satisfy future edges
            // Count unassigned logical neighbors of li
            std::vector<size_t> unassigned_neighbors;
            for (size_t v : log_adj_idx[li]) {
                if (mapping[v] == -1) {
                    unassigned_neighbors.push_back(v);
                }
            }
            // Collect unused physical neighbors of j
            std::vector<size_t> unused_phys_neighbors;
            for (size_t pj : phys_adj_idx[j]) {
                if (!used[pj]) {
                    unused_phys_neighbors.push_back(pj);
                }
            }
            if (unused_phys_neighbors.size() < unassigned_neighbors.size()) {
                if ((dfs_calls & ((1u << 17) - 1)) == 0) {
                    log_debug(
                        tt::LogFabric,
                        "Prune: capacity check failed li={}, phys_j={}, unused_phys_neighbors={}, "
                        "unassigned_neighbors={}",
                        li,
                        j,
                        unused_phys_neighbors.size(),
                        unassigned_neighbors.size());
                }
                continue;  // not enough capacity to satisfy pending logical edges
            }
            // For each future logical neighbor v, verify there exists at least one viable unused physical neighbor
            for (size_t v : unassigned_neighbors) {
                bool has_candidate = false;
                for (size_t pj : unused_phys_neighbors) {
                    // Degree feasibility
                    if (phys_deg[pj] < log_deg[v]) {
                        continue;
                    }
                    // Check consistency with already assigned neighbors of v
                    bool consistent = true;
                    for (size_t lv = 0; lv < n_log; ++lv) {
                        int pk2_i = mapping[lv];
                        if (pk2_i == -1) {
                            continue;
                        }
                        size_t pk2 = static_cast<size_t>(pk2_i);
                        bool log_conn2 = std::binary_search(log_adj_idx[v].begin(), log_adj_idx[v].end(), lv);
                        bool phys_conn2 = std::binary_search(phys_adj_idx[pj].begin(), phys_adj_idx[pj].end(), pk2);
                        if (log_conn2 && !phys_conn2) {
                            consistent = false;
                            break;
                        }
                    }
                    if (consistent) {
                        has_candidate = true;
                        break;
                    }
                }
                if (!has_candidate) {
                    ok = false;
                    if ((dfs_calls & ((1u << 17) - 1)) == 0) {
                        log_debug(
                            tt::LogFabric,
                            "Prune: future neighbor viability failed for li={}, neighbor_v={}, trying phys_j={}",
                            li,
                            v,
                            j);
                    }
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            log_debug(tt::LogFabric, "Assigning fabric_node: {} to asic: {}", log_nodes[li], phys_nodes[j]);

            used[j] = true;
            mapping[li] = static_cast<int>(j);
            if ((dfs_calls & ((1u << 16) - 1)) == 0) {
                log_info(
                    tt::LogFabric,
                    "Assign: li={} -> phys_j={}, log_deg={}, phys_deg={}",
                    li,
                    j,
                    log_deg[li],
                    phys_deg[j]);
            }
            if (dfs(pos + 1)) {
                return true;
            }
            mapping[li] = -1;
            used[j] = false;
            if ((dfs_calls & ((1u << 16) - 1)) == 0) {
                log_debug(tt::LogFabric, "Backtrack: li={} from phys_j={}", li, j);
            }
        }

        failed_states.insert(key);
        return false;
    };

    // Start DFS from the number of already assigned pinned nodes
    size_t assigned_count = 0;
    for (auto v : mapping) {
        if (v != -1) {
            assigned_count++;
        }
    }
    bool found = dfs(assigned_count);
    TT_FATAL(
        found,
        "Graph specified in MGD could not fit in the discovered physical topology for mesh {} under the given "
        "pinning constraints. Either relax pinnings or modify the MGD. If this is unexpected, run "
        "./build/test/tt_metal/tt_fabric/test_system_health to check connectivity.",
        mesh_id.get());

    for (size_t i = 0; i < n_log; ++i) {
        FabricNodeId fn = log_nodes[i];
        tt::tt_metal::AsicID asic = phys_nodes[mapping[i]];
        fabric_node_id_to_asic_id_.emplace(fn, asic);
        asic_id_to_fabric_node_id_.emplace(asic, fn);
    }
}

// NOLINTEND(readability-function-cognitive-complexity)

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
    return get_host_rank_for_coord(mesh_id, coord);
}

std::optional<MeshHostRankId> TopologyMapper::get_host_rank_for_coord(MeshId mesh_id, const MeshCoordinate& coord) const {
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

void TopologyMapper::rebuild_host_rank_structs_from_mapping(
    const std::unordered_map<MeshId, std::unordered_map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Derive per-mesh host sets and per-host coord ranges from current mapping
    std::unordered_map<MeshId, std::unordered_set<MeshHostRankId>> mesh_to_hosts;
    std::unordered_map<MeshId, std::unordered_map<MeshHostRankId, MeshCoordinateRange>> mesh_host_to_range;
    // For wraparound-aware construction, accumulate coordinates per host then compute minimal circular ranges.
    std::unordered_map<MeshId, std::unordered_map<MeshHostRankId, std::vector<MeshCoordinate>>> mesh_host_to_coords;

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
        const auto mesh_id_val = fabric_node_id.mesh_id;
        const auto host_rank = asic_id_to_mesh_rank.at(mesh_id_val).at(asic_id);
        mesh_to_hosts[mesh_id_val].insert(host_rank);
        const auto coord = per_mesh_chip_to_coord[mesh_id_val].at(fabric_node_id.chip_id);
        mesh_host_to_coords[mesh_id_val][host_rank].push_back(coord);
    }

    // Build minimal wraparound-aware ranges per host
    for (const auto& [mesh_id, host_coords_map] : mesh_host_to_coords) {
        const auto shape = mesh_graph_.get_mesh_shape(mesh_id);
        auto& range_map = mesh_host_to_range[mesh_id];
        for (const auto& [host_rank, coords] : host_coords_map) {
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
                range_map.emplace(host_rank, MeshCoordinateRange(start, end, shape));
            } else {
                range_map.emplace(host_rank, MeshCoordinateRange(start, end));
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
        for (const auto& [host_rank, range] : range_map) {
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
        // Compute base_rank as min over host_ranks
        std::uint32_t base_rank = std::numeric_limits<std::uint32_t>::max();
        for (const auto& [host_rank, _] : range_map) {
            base_rank = std::min(base_rank, host_rank.get());
        }
        for (std::uint32_t r : row_list) {
            for (std::uint32_t c : col_list) {
                // find host_rank whose range starts at (r,c)
                for (const auto& [original_host_rank, range] : range_map) {
                    if (range.start_coord()[0] == r && range.start_coord()[1] == c) {
                        std::size_t idx = (row_index(r) * host_grid_shape[1]) + col_index(c);
                        std::uint32_t norm_rank = original_host_rank.get() - base_rank;
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

void TopologyMapper::print_logical_adjacency_map(const std::unordered_map<MeshId, LogicalAdjacencyMap>& adj_map) const {
    log_debug(tt::LogFabric, "TopologyMapper: Logical Adjacency Map:");
    for (const auto& [mesh_id, node_map] : adj_map) {
        log_debug(tt::LogFabric, "  Mesh ID: {}", *mesh_id);
        for (const auto& [node, neighbors] : node_map) {
            std::string neigh_str;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                neigh_str += fmt::format("{}", neighbors[i]);
                if (i < neighbors.size() - 1) {
                    neigh_str += ", ";
                }
            }
            log_debug(tt::LogFabric, "    Node {} connected to: [{}]", node, neigh_str);
        }
    }
}

void TopologyMapper::print_physical_adjacency_map(
    const std::unordered_map<MeshId, PhysicalAdjacencyMap>& adj_map) const {
    log_debug(tt::LogFabric, "TopologyMapper: Physical Adjacency Map:");
    for (const auto& [mesh_id, node_map] : adj_map) {
        log_debug(tt::LogFabric, "  Mesh ID: {}", *mesh_id);
        for (const auto& [node, neighbors] : node_map) {
            std::string neigh_str;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                neigh_str += fmt::format("{}", neighbors[i].get());
                if (i < neighbors.size() - 1) {
                    neigh_str += ", ";
                }
            }
            log_debug(tt::LogFabric, "    Node {} connected to: [{}]", node.get(), neigh_str);
            log_debug(tt::LogFabric, "    Host_name = {}", physical_system_descriptor_.get_host_name_for_asic(node));
        }
    }
}

}  // namespace tt::tt_fabric
