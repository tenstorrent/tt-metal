// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/topology_mapper.hpp>

#include <algorithm>
#include <climits>
#include <unordered_set>
#include <limits>
#include <functional>
#include <optional>
#include <tuple>
#include <map>

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "experimental/fabric/routing_table_generator.hpp"
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric {

namespace {
// Encodes MPI rank, MeshId and MeshHostRankId into a single 64-bit value for transport.
// Format: mpi_rank (16 bits) | mesh_id (16 bits) | host_rank (32 bits)
std::uint64_t encode_mpi_rank_mesh_id_and_rank(int mpi_rank, MeshId mesh_id, MeshHostRankId host_rank) {
    return (static_cast<std::uint64_t>(mpi_rank & 0xFFFF) << 48) |
           (static_cast<std::uint64_t>(mesh_id.get() & 0xFFFF) << 32) | static_cast<std::uint64_t>(host_rank.get());
}

std::tuple<int, MeshId, MeshHostRankId> decode_mpi_rank_mesh_id_and_rank(std::uint64_t encoded_value) {
    return {
        static_cast<int>((encoded_value >> 48) & 0xFFFF),
        MeshId{static_cast<std::uint32_t>((encoded_value >> 32) & 0xFFFF)},
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
    auto it = asic_id_to_mapping_.find(asic_id);
    TT_FATAL(it != asic_id_to_mapping_.end(), "ASIC id {} not found in mapping", asic_id);
    TT_FATAL(it->second->is_mapped, "Fabric node ID not yet assigned for ASIC id {}", asic_id);
    return it->second->fabric_node_id;
}

FabricNodeId TopologyMapper::get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const {
    auto it = physical_chip_id_to_mapping_.find(physical_chip_id);
    TT_FATAL(it != physical_chip_id_to_mapping_.end(), "Physical chip id {} not found in mapping", physical_chip_id);
    TT_FATAL(it->second->is_mapped, "Fabric node ID not yet assigned for physical chip id {}", physical_chip_id);
    return it->second->fabric_node_id;
}

ChipId TopologyMapper::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    auto it = fabric_node_id_to_mapping_.find(fabric_node_id);
    TT_FATAL(it != fabric_node_id_to_mapping_.end(), "Fabric node id {} not found in mapping", fabric_node_id);
    return it->second->physical_chip_id;
}

tt::tt_metal::AsicID TopologyMapper::get_asic_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    auto it = fabric_node_id_to_mapping_.find(fabric_node_id);
    TT_FATAL(it != fabric_node_id_to_mapping_.end(), "Fabric node id {} not found in mapping", fabric_node_id);
    return it->second->asic_id;
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
    mesh_host_rank_to_mpi_rank_.clear();
    build_asic_physical_chip_id_mappings();
    initialize_chip_topology_mapping_map();
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
    mesh_host_rank_to_mpi_rank_.clear();
    build_asic_physical_chip_id_mappings();
    initialize_chip_topology_mapping_map();
    build_mapping();
}

// Constructor that skips discovery and builds mapping directly from provided logical to physical chip mapping
TopologyMapper::TopologyMapper(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const LocalMeshBinding& local_mesh_binding,
    const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) :
    mesh_graph_(mesh_graph),
    physical_system_descriptor_(physical_system_descriptor),
    local_mesh_binding_(local_mesh_binding),
    fixed_asic_position_pinnings_({}) {
    log_debug(
        tt::LogFabric,
        "TopologyMapper: Building mapping directly from provided logical to physical chip mapping (skipping "
        "discovery)");

    mesh_host_ranks_.clear();
    mesh_host_rank_coord_ranges_.clear();
    mesh_host_rank_to_mpi_rank_.clear();

    // Initialize chip_topology_mapping_ with all ASICs
    initialize_chip_topology_mapping_map();

    // Build asic to physical chip id mappings first (needed for conversion)
    build_asic_physical_chip_id_mappings();

    // Build fabric node id to asic id mapping directly from the provided logical to physical chip mapping
    // Update chip_topology_mapping_ entries with the mapping information
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& [fabric_node_id, physical_chip_id] : logical_mesh_chip_id_to_physical_chip_id_mapping) {
        // Convert physical chip id to asic id
        // First try to find it in the local cluster (for local chips)
        tt::tt_metal::AsicID asic_id{0};
        bool found_asic_id = false;
        for (const auto& [chip_id, unique_id] : cluster.get_unique_chip_ids()) {
            if (chip_id == physical_chip_id) {
                asic_id = tt::tt_metal::AsicID{unique_id};
                found_asic_id = true;
                break;
            }
        }

        // If not found in local cluster, try to find it in physical_chip_id_to_mapping_
        // (for local chips that were already set during initialization)
        MappedChipInfo* info = nullptr;
        if (!found_asic_id) {
            auto mapping_it = physical_chip_id_to_mapping_.find(physical_chip_id);
            if (mapping_it != physical_chip_id_to_mapping_.end() && mapping_it->second != nullptr) {
                info = mapping_it->second;
                asic_id = info->asic_id;
                found_asic_id = true;
            }
        }

        // If still not found, search through chip_topology_mapping_ to find an entry
        // that might have this physical_chip_id set (for remote chips that were received via broadcast)
        if (!found_asic_id) {
            for (auto& entry : chip_topology_mapping_) {
                if (entry.physical_chip_id == physical_chip_id) {
                    asic_id = entry.asic_id;
                    found_asic_id = true;
                    break;
                }
            }
        }

        // If we found the ASIC ID, look up the entry by ASIC ID
        if (found_asic_id) {
            auto asic_it = asic_id_to_mapping_.find(asic_id);
            TT_FATAL(asic_it != asic_id_to_mapping_.end(), "ASIC id {} not found in chip_topology_mapping_", asic_id);
            info = asic_it->second;
        } else {
            // Last resort: this shouldn't happen, but provide a helpful error message
            TT_FATAL(
                false,
                "Physical chip id {} not found in chip_topology_mapping_. "
                "This may happen if the physical chip is not in the cluster or physical system descriptor.",
                physical_chip_id);
        }

        TT_FATAL(info != nullptr, "Null pointer in lookup for physical_chip_id {}", physical_chip_id);

        // Update the MappedChipInfo entry with mapping information
        info->fabric_node_id = fabric_node_id;
        info->physical_chip_id = physical_chip_id;
        info->mesh_coord = mesh_graph_.chip_to_coordinate(fabric_node_id.mesh_id, fabric_node_id.chip_id);

        // Get host rank from mesh graph
        auto host_rank = mesh_graph_.get_host_rank_for_chip(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        TT_FATAL(host_rank.has_value(), "Fabric node id {} not found in mesh graph", fabric_node_id);
        info->mesh_host_rank = host_rank.value();

        // Get hostname from physical system descriptor
        info->hostname = physical_system_descriptor_.get_host_name_for_asic(info->asic_id);

        info->is_mapped = true;
    }

    // Rebuild lookup maps after updating entries
    rebuild_lookup_maps();

    // Build asic_id_to_mesh_rank mapping needed for rebuild_host_rank_structs_from_mapping
    // This maps each asic to its mesh host rank based on the fabric node it's mapped to
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [fabric_node_id, info_ptr] : fabric_node_id_to_mapping_) {
        TT_FATAL(info_ptr != nullptr, "Null pointer in fabric_node_id_to_mapping_");
        const auto& info = *info_ptr;
        TT_FATAL(info.is_mapped, "Entry in fabric_node_id_to_mapping_ is not mapped");
        asic_id_to_mesh_rank[fabric_node_id.mesh_id][info.asic_id] = info.mesh_host_rank;
    }

    // Populate mesh_host_rank_to_mpi_rank_ mapping
    // For each fabric node in the mapping, determine which MPI rank owns it
    for (const auto& [fabric_node_id, info_ptr] : fabric_node_id_to_mapping_) {
        TT_FATAL(info_ptr != nullptr, "Null pointer in fabric_node_id_to_mapping_");
        const auto& info = *info_ptr;
        TT_FATAL(info.is_mapped, "Entry in fabric_node_id_to_mapping_ is not mapped");
        int mpi_rank = static_cast<int>(physical_system_descriptor_.get_rank_for_hostname(info.hostname));
        mesh_host_rank_to_mpi_rank_[std::make_pair(fabric_node_id.mesh_id, info.mesh_host_rank)] = mpi_rank;
    }

    // Build host rank structures from the mapping
    rebuild_host_rank_structs_from_mapping(asic_id_to_mesh_rank);

    // For custom fabric topology, we also need to gather mesh bindings from all ranks to populate
    // mesh_host_rank_to_mpi_rank_ for meshes this rank doesn't participate in
    auto global_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const std::size_t world_size = *global_context->size();
    if (world_size > 1) {
        // Gather mesh_id and host_rank from all ranks
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
        auto my_mpi_rank = static_cast<int>(*global_context->rank());
        for (std::uint32_t i = 0; i < local_count; ++i) {
            send_values[i] = encode_mpi_rank_mesh_id_and_rank(
                my_mpi_rank, local_mesh_binding_.mesh_ids[i], local_mesh_binding_.host_rank);
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

        // Decode and populate mesh_host_rank_to_mpi_rank_ from gathered data
        for (const auto encoded : gathered) {
            if (encoded == sentinel) {
                continue;
            }
            const auto [gathered_mpi_rank, mesh_id, host_rank] = decode_mpi_rank_mesh_id_and_rank(encoded);
            mesh_host_rank_to_mpi_rank_[std::make_pair(mesh_id, host_rank)] = gathered_mpi_rank;
        }
    }
}

ChipId TopologyMapper::get_physical_chip_id_from_asic_id(tt::tt_metal::AsicID asic_id) const {
    auto it = asic_id_to_mapping_.find(asic_id);
    TT_FATAL(it != asic_id_to_mapping_.end(), "ASIC id {} not found in mapping", asic_id);
    return it->second->physical_chip_id;
}

void TopologyMapper::build_asic_physical_chip_id_mappings() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Check the physical chip asic ids from UMD cluster with the physical chip asic ids from the physical system
    // descriptor
    for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
        tt::tt_metal::AsicID asic_id{unique_id};
        auto asic_ids_for_host =
            physical_system_descriptor_.get_asics_connected_to_host(physical_system_descriptor_.my_host_name());
        TT_FATAL(
            std::find(asic_ids_for_host.begin(), asic_ids_for_host.end(), asic_id) != asic_ids_for_host.end(),
            "Asic id {} in UMD cluster not found for in Physical System {}",
            asic_id,
            physical_system_descriptor_.my_host_name());
    }
}

void TopologyMapper::initialize_chip_topology_mapping_map() {
    log_debug(tt::LogFabric, "TopologyMapper: Initializing chip topology info map for all ASICs");

    chip_topology_mapping_.clear();

    // Get all ASICs from physical system descriptor
    const auto& asic_descriptors = physical_system_descriptor_.get_asic_descriptors();

    // Get local cluster for physical_chip_id lookup
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& my_host = physical_system_descriptor_.my_host_name();

    // Create MappedChipInfo entry for each ASIC
    for (const auto& [asic_id, asic_descriptor] : asic_descriptors) {
        // Fill with available information

        MappedChipInfo info;
        info.asic_id = asic_id;
        info.hostname = asic_descriptor.host_name;

        // Fill in physical_chip_id if this ASIC is on the local host
        if (asic_descriptor.host_name == my_host) {
            // Look up physical_chip_id from cluster
            for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
                if (unique_id == *asic_id) {
                    info.physical_chip_id = physical_chip_id;
                    break;
                }
            }
        }

        // Otherwise physical_chip_id remains 0 (will be filled later by the owning host)
        chip_topology_mapping_.push_back(info);
    }

    // Build lookup map indexed by ASIC ID
    rebuild_lookup_maps();

    log_debug(
        tt::LogFabric, "TopologyMapper: Initialized {} chip topology info entries", chip_topology_mapping_.size());
}

void TopologyMapper::build_mapping() {
    log_debug(tt::LogFabric, "TopologyMapper: Building mapping between fabric node IDs and physical ASIC IDs");

    // Check that this is not a multi-mesh-per-host system not supported by this algorithm
    TT_FATAL(
        local_mesh_binding_.mesh_ids.size() == 1,
        "Multi-mesh-per-host systems are not supported by this algorithm, please use custom fabric topology via "
        "MetalContext::set_custom_fabric_topology");

    generate_mapping_locally_ = (mesh_graph_.get_all_mesh_ids().size() == 1) &&
                                (mesh_graph_.get_host_ranks(local_mesh_binding_.mesh_ids[0]).size() == 1);

    // Build ASIC ID to mesh rank mapping using the gathered mesh bindings directly
    // This function gathers mesh_id and host_rank from all MPI ranks and maps them to ASICs
    auto asic_id_to_mesh_rank = build_asic_id_to_mesh_rank_mapping();
    auto fabric_node_id_to_mesh_rank = build_fabric_node_id_to_mesh_rank_mapping();

    // Only 1 host builds the mapping the rest will wait and use the mapping from the 1st host
    if (generate_mapping_locally_ ||
        *tt::tt_metal::MetalContext::instance().full_world_distributed_context().rank() == 0) {
        // Build logical and physical adjacency maps
        auto adjacency_map_logical = tt::tt_metal::experimental::tt_fabric::build_adjacency_map_logical(mesh_graph_);
        auto adjacency_map_physical = tt::tt_metal::experimental::tt_fabric::build_adjacency_map_physical(
            physical_system_descriptor_, asic_id_to_mesh_rank);

        print_logical_adjacency_map(adjacency_map_logical);
        print_physical_adjacency_map(adjacency_map_physical);

        // Use sat solver algo to preserve the logical connectivity in the physical topology
        // Note: physical_chip_id is filled in during populate_fabric_node_id_to_asic_id_mappings
        // for ASICs that belong to this host, so no separate loop is needed here
        for (const auto& mesh_id : mesh_graph_.get_all_mesh_ids()) {
            populate_fabric_node_id_to_asic_id_mappings(
                mesh_id,
                adjacency_map_physical.at(mesh_id),
                adjacency_map_logical.at(mesh_id),
                asic_id_to_mesh_rank.at(mesh_id),
                fabric_node_id_to_mesh_rank.at(mesh_id));
        }

        // Broadcast the mapping to all hosts
        if (!generate_mapping_locally_) {
            broadcast_mapping_to_all_hosts();
        }
    } else {
        // Wait for the 1st host to build the mapping
        receive_mapping_from_host(0);
    }

    // Rebuild lookup maps from container
    rebuild_lookup_maps();

    // Build host rank containers now that mapping is complete
    rebuild_host_rank_structs_from_mapping(asic_id_to_mesh_rank);
}

std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> TopologyMapper::build_fabric_node_id_to_mesh_rank_mapping()
    const {
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> mapping;
    for (const auto& mesh_id : mesh_graph_.get_all_mesh_ids()) {
        for (const auto& [_, chip_id] : mesh_graph_.get_chip_ids(mesh_id)) {
            auto host_rank = mesh_graph_.get_host_rank_for_chip(mesh_id, chip_id);
            TT_FATAL(host_rank.has_value(), "Fabric node id {} not found", FabricNodeId(mesh_id, chip_id));
            mapping[mesh_id][FabricNodeId(mesh_id, chip_id)] = host_rank.value();
        }
    }
    return mapping;
}

std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> TopologyMapper::build_asic_id_to_mesh_rank_mapping() {
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> mapping;
    auto global_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const std::size_t world_size = *global_context->size();

    if (generate_mapping_locally_ || world_size <= 1) {
        auto host_rank = local_mesh_binding_.host_rank;
        auto mpi_rank = static_cast<int>(*host_rank);

        // Get asics on current host
        auto asics =
            physical_system_descriptor_.get_asics_connected_to_host(physical_system_descriptor_.my_host_name());
        for (const auto& mesh_id : mesh_graph_.get_all_mesh_ids()) {
            for (const auto& asic : asics) {
                mapping[mesh_id][asic] = host_rank;
                mesh_host_rank_to_mpi_rank_[std::make_pair(mesh_id, host_rank)] = mpi_rank;
            }
        }
        return mapping;
    }

    // Step 1: Build MPI rank -> hostname mapping (rank from PSD is the MPI rank)
    std::map<int, HostName> mpi_rank_to_host;
    for (const auto& host : physical_system_descriptor_.get_all_hostnames()) {
        auto mpi_rank = physical_system_descriptor_.get_rank_for_hostname(host);
        mpi_rank_to_host[static_cast<int>(mpi_rank)] = host;
    }

    // Step 2: Gather mesh_id and host_rank from all ranks to know which mesh_ids each MPI rank participates in
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
    auto my_mpi_rank = static_cast<int>(*global_context->rank());
    for (std::uint32_t i = 0; i < local_count; ++i) {
        // Encode MPI rank along with mesh_id and host_rank so we can map correctly
        send_values[i] = encode_mpi_rank_mesh_id_and_rank(
            my_mpi_rank, local_mesh_binding_.mesh_ids[i], local_mesh_binding_.host_rank);
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

    // Step 3: Use the gathered mesh bindings directly to build the mapping
    // Decode MPI rank, mesh_id and host_rank from gathered data and map to ASICs
    // Build an ordered map from MPI rank to (mesh_id, host_rank) pairs for deterministic iteration
    std::map<int, std::map<MeshId, MeshHostRankId>> mpi_rank_to_mesh_bindings;
    mesh_host_rank_to_mpi_rank_.clear();
    for (const auto encoded : gathered) {
        if (encoded == sentinel) {
            continue;
        }
        const auto [gathered_mpi_rank, mesh_id, host_rank] = decode_mpi_rank_mesh_id_and_rank(encoded);
        mpi_rank_to_mesh_bindings[gathered_mpi_rank][mesh_id] = host_rank;
        // Store reverse mapping for quick lookups
        mesh_host_rank_to_mpi_rank_[std::make_pair(mesh_id, host_rank)] = gathered_mpi_rank;
    }

    // Step 4: For each MPI rank in the gathered data, assign mesh host rank to ASICs
    for (const auto& [gathered_mpi_rank, mesh_bindings] : mpi_rank_to_mesh_bindings) {
        // Get the hostname for this MPI rank
        auto host_it = mpi_rank_to_host.find(gathered_mpi_rank);
        if (host_it == mpi_rank_to_host.end()) {
            TT_FATAL(false, "MPI rank {} not found in mpi_rank_to_host mapping", gathered_mpi_rank);
        }

        const auto& host_name = host_it->second;
        auto asics = physical_system_descriptor_.get_asics_connected_to_host(host_name);

        // For each mesh_id this MPI rank participates in, use the host_rank from gathered data
        for (const auto& [mesh_id, host_rank] : mesh_bindings) {
            // Use the host_rank directly from the gathered data (which comes from local_mesh_binding_.host_rank)
            // This is the mesh host rank set via TT_MESH_HOST_RANK environment variable
            for (const auto& asic : asics) {
                mapping[mesh_id][asic] = host_rank;
            }
        }
    }

    return mapping;
}

void TopologyMapper::populate_fabric_node_id_to_asic_id_mappings(
    const MeshId mesh_id,
    const PhysicalAdjacencyMap& adjacency_map_physical,
    const LogicalAdjacencyMap& adjacency_map_logical,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_id_to_mesh_rank,
    const std::map<FabricNodeId, MeshHostRankId>& fabric_node_id_to_mesh_rank) {
    // Build configuration for the utility function
    tt::tt_metal::experimental::tt_fabric::TopologyMappingConfig config;
    config.strict_mode = !mesh_graph_.is_intra_mesh_policy_relaxed(mesh_id);

    // Build pinning constraints if any
    for (const auto& [pos, fabric_node] : fixed_asic_position_pinnings_) {
        if (fabric_node.mesh_id == mesh_id) {
            config.pinnings.emplace_back(pos, fabric_node);
        }
    }

    // Build AsicPositionMap if pinnings are non-empty
    if (!config.pinnings.empty()) {
        for (const auto& [asic_id, _] : adjacency_map_physical) {
            auto tray = physical_system_descriptor_.get_tray_id(asic_id);
            auto loc = physical_system_descriptor_.get_asic_location(asic_id);
            config.asic_positions.emplace(asic_id, std::make_pair(tray, loc));
        }
    }

    // Call the utility function
    auto result = tt::tt_metal::experimental::tt_fabric::map_mesh_to_physical(
        mesh_id,
        adjacency_map_logical,
        adjacency_map_physical,
        fabric_node_id_to_mesh_rank,
        asic_id_to_mesh_rank,
        config);

    TT_FATAL(
        result.success,
        "Graph specified in MGD could not fit in the discovered physical topology for mesh {}. {}. "
        "Either relax pinnings or modify the MGD. If this is unexpected, run "
        "./build/test/tt_metal/tt_fabric/test_system_health to check connectivity.",
        mesh_id.get(),
        result.error_message);

    // Update MappedChipInfo entries from the result
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        auto it = asic_id_to_mapping_.find(asic);
        TT_FATAL(it != asic_id_to_mapping_.end(), "ASIC id {} not found in chip_topology_mapping_", asic);
        MappedChipInfo& info = *it->second;

        info.fabric_node_id = fabric_node;
        info.mesh_coord = mesh_graph_.chip_to_coordinate(mesh_id, fabric_node.chip_id);
        if (asic_id_to_mesh_rank.contains(asic)) {
            info.mesh_host_rank = asic_id_to_mesh_rank.at(asic);
        }
        info.is_mapped = true;
    }

    // Rebuild lookup maps after updating entries
    rebuild_lookup_maps();
}

void TopologyMapper::broadcast_mapping_to_all_hosts() {
    using namespace tt::tt_metal::distributed::multihost;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();

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

    // Serialization helpers
    auto serialize_u32 = [](std::vector<uint8_t>& buf, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
        }
    };

    auto serialize_u64 = [](std::vector<uint8_t>& buf, std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
        }
    };

    auto serialize_string = [&](std::vector<uint8_t>& buf, const std::string& s) {
        serialize_u32(buf, static_cast<std::uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    };

    // Collect all mapped entries for broadcasting
    std::vector<const MappedChipInfo*> mapped_entries;
    for (const auto& info : chip_topology_mapping_) {
        if (info.is_mapped) {
            mapped_entries.push_back(&info);
        }
    }
    std::uint32_t count = static_cast<std::uint32_t>(mapped_entries.size());

    for (std::size_t peer = 0; peer < world_size; ++peer) {
        if (peer == CONTROLLER_RANK) {
            continue;
        }

        // Send count first (synchronous send to ensure receiver posted recv)
        std::uint32_t count_copy = count;
        distributed_context.ssend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&count_copy), sizeof(count_copy)),
            Rank{static_cast<int>(peer)},
            Tag{0});

        // Send one record at a time using synchronous send
        // First send the record size, then send the record data
        for (const auto* info_ptr : mapped_entries) {
            const auto& info = *info_ptr;

            std::vector<uint8_t> record;

            // fabric_node_id (encoded as u64)
            const std::uint64_t encoded_fn = encode_fabric_node_id(info.fabric_node_id);
            serialize_u64(record, encoded_fn);

            // asic_id (u64)
            serialize_u64(record, *info.asic_id);

            // physical_chip_id (u32)
            serialize_u32(record, info.physical_chip_id);

            // mesh_coord (u32 dims, then u32 values per dim)
            serialize_u32(record, static_cast<std::uint32_t>(info.mesh_coord.dims()));
            for (size_t d = 0; d < info.mesh_coord.dims(); ++d) {
                serialize_u32(record, info.mesh_coord[d]);
            }

            // mesh_host_rank (u32)
            serialize_u32(record, info.mesh_host_rank.get());

            // hostname (string, or empty string if not present)
            if (!info.hostname.empty()) {
                serialize_string(record, info.hostname);
            } else {
                serialize_u32(record, 0);  // empty string
            }

            // Send size first, then data
            std::uint32_t record_size = static_cast<std::uint32_t>(record.size());
            distributed_context.ssend(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&record_size), sizeof(record_size)),
                Rank{static_cast<int>(peer)},
                Tag{1});  // Use Tag{1} for size messages

            distributed_context.ssend(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(record.data(), record.size())),
                Rank{static_cast<int>(peer)},
                Tag{0});  // Use Tag{0} for data messages
        }
    }
}

void TopologyMapper::receive_mapping_from_host(int rank) {
    using namespace tt::tt_metal::distributed::multihost;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();

    // If not in distributed context, nothing to receive
    if (*distributed_context.size() <= 1) {
        return;
    }

    auto my_rank = *distributed_context.rank();
    if (static_cast<int>(my_rank) == rank) {
        return;  // sender does not receive
    }

    // Receive count, then 'count' variable-size records
    std::uint32_t count = 0;
    {
        auto req = distributed_context.irecv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&count), sizeof(count)),
            Rank{static_cast<int>(rank)},
            Tag{0});

        wait_for_request_with_timeout(req, "topology mapping header", rank);
    }

    // Don't clear chip_topology_mapping_ - we want to keep initialized entries and update them
    // The count is the number of mapped entries (with fabric_node_id), not total entries

    auto read_u32_from = [&](const std::vector<uint8_t>& buf, std::size_t& idx) -> std::uint32_t {
        TT_FATAL(idx + 4 <= buf.size(), "Deserializer overflow reading u32");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= (static_cast<std::uint32_t>(buf[idx++]) << (8 * i));
        }
        return v;
    };

    auto read_u64_from = [&](const std::vector<uint8_t>& buf, std::size_t& idx) -> std::uint64_t {
        TT_FATAL(idx + 8 <= buf.size(), "Deserializer overflow reading u64");
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (static_cast<std::uint64_t>(buf[idx++]) << (8 * i));
        }
        return v;
    };

    for (std::uint32_t i = 0; i < count; ++i) {
        // Receive size first, then receive data of that size
        std::uint32_t record_size = 0;
        {
            auto req = distributed_context.irecv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&record_size), sizeof(record_size)),
                Rank{static_cast<int>(rank)},
                Tag{1});  // Use Tag{1} for size messages

            wait_for_request_with_timeout(
                req, "topology mapping record size " + std::to_string(i + 1) + " of " + std::to_string(count), rank);
        }

        TT_FATAL(
            record_size > 0 && record_size < 1000000,
            "Invalid message size {} for topology mapping record {} from rank {} (suspiciously large, possible "
            "corruption)",
            record_size,
            i + 1,
            rank);

        // Allocate buffer of exact size
        std::vector<uint8_t> record(record_size);
        auto req = distributed_context.irecv(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(record.data(), record.size())),
            Rank{static_cast<int>(rank)},
            Tag{0});  // Use Tag{0} for data messages

        wait_for_request_with_timeout(
            req, "topology mapping record " + std::to_string(i + 1) + " of " + std::to_string(count), rank);

        std::size_t idx = 0;

        // fabric_node_id
        const auto encoded_fn = read_u64_from(record, idx);
        FabricNodeId fn = decode_fabric_node_id(encoded_fn);

        // asic_id
        const auto asic_val = read_u64_from(record, idx);
        tt::tt_metal::AsicID asic_id{asic_val};

        // physical_chip_id
        ChipId physical_chip_id = read_u32_from(record, idx);

        // Find existing entry by ASIC ID and update it
        auto it = asic_id_to_mapping_.find(asic_id);
        TT_FATAL(
            it != asic_id_to_mapping_.end(), "ASIC id {} not found in chip_topology_mapping_ during receive", asic_id);
        MappedChipInfo& info = *it->second;

        // Update fields with received information
        info.fabric_node_id = fn;
        if (physical_chip_id != 0) {
            info.physical_chip_id = physical_chip_id;
        }

        // mesh_coord (always present when mapped)
        std::uint32_t coord_dims = read_u32_from(record, idx);
        std::vector<uint32_t> coord_values(coord_dims);
        for (std::uint32_t d = 0; d < coord_dims; ++d) {
            coord_values[d] = read_u32_from(record, idx);
        }
        info.mesh_coord = MeshCoordinate(tt::stl::Span<const uint32_t>(coord_values));

        // mesh_host_rank (always present when mapped)
        std::uint32_t host_rank_val = read_u32_from(record, idx);
        info.mesh_host_rank = MeshHostRankId{host_rank_val};

        // hostname (string, or empty string if not present)
        std::uint32_t hostname_len = read_u32_from(record, idx);
        if (hostname_len > 0) {
            TT_FATAL(idx + hostname_len <= record.size(), "Deserializer overflow reading hostname");
            std::string hostname_str(reinterpret_cast<const char*>(record.data() + idx), hostname_len);
            idx += hostname_len;
            info.hostname = hostname_str;
        }

        info.is_mapped = true;
    }

    // Rebuild lookup maps after receiving and updating entries
    rebuild_lookup_maps();

    // Fill in physical_chip_id for ASICs that belong to this host
    // (The controller may have set it to 0 for ASICs on other hosts)
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& my_host = physical_system_descriptor_.my_host_name();
    for (auto& info : chip_topology_mapping_) {
        if (info.physical_chip_id == 0 && !info.hostname.empty() && info.hostname == my_host) {
            // This ASIC belongs to this host, look up its physical chip ID
            for (const auto& [physical_chip_id, unique_id] : cluster.get_unique_chip_ids()) {
                if (unique_id == *info.asic_id) {
                    info.physical_chip_id = physical_chip_id;
                    break;
                }
            }
        }
    }

    // Rebuild lookup maps after filling in physical_chip_id
    rebuild_lookup_maps();
}

void TopologyMapper::rebuild_lookup_maps() {
    fabric_node_id_to_mapping_.clear();
    asic_id_to_mapping_.clear();
    physical_chip_id_to_mapping_.clear();

    for (auto& info : chip_topology_mapping_) {
        // Only add to fabric_node_id map if entry is mapped
        if (info.is_mapped) {
            fabric_node_id_to_mapping_[info.fabric_node_id] = &info;
            physical_chip_id_to_mapping_[info.physical_chip_id] = &info;
        }
        asic_id_to_mapping_[info.asic_id] = &info;
    }
}

std::map<FabricNodeId, ChipId> TopologyMapper::get_local_logical_mesh_chip_id_to_physical_chip_id_mapping() const {
    std::map<FabricNodeId, ChipId> mapping;
    const auto& my_host = physical_system_descriptor_.my_host_name();
    // Use chip_topology_mapping_ for centralized access
    for (const auto& info : chip_topology_mapping_) {
        if (info.is_mapped && !info.hostname.empty() && info.hostname == my_host) {
            mapping[info.fabric_node_id] = info.physical_chip_id;
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

std::optional<MeshHostRankId> TopologyMapper::get_host_rank_for_coord(
    MeshId mesh_id, const MeshCoordinate& coord) const {
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
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& /* asic_id_to_mesh_rank */) {
    // Derive per-mesh host sets and per-host coord ranges from current mapping
    std::map<MeshId, std::unordered_set<MeshHostRankId>> mesh_to_hosts;
    std::map<MeshId, std::map<MeshHostRankId, MeshCoordinateRange>> mesh_host_to_range;
    // For wraparound-aware construction, accumulate coordinates per host then compute minimal circular ranges.
    std::map<MeshId, std::map<MeshHostRankId, std::vector<MeshCoordinate>>> mesh_host_to_coords;

    // Accumulate coordinates per host from chip_topology_mapping_
    // Only process mapped entries - use fabric_node_id_to_mapping_ which only contains mapped entries
    for (const auto& [fabric_node_id, info_ptr] : fabric_node_id_to_mapping_) {
        TT_FATAL(
            info_ptr != nullptr,
            "fabric_node_id_to_mapping_ contains null pointer for fabric_node_id {}",
            fabric_node_id);
        const auto& info = *info_ptr;
        // All entries in fabric_node_id_to_mapping_ should be mapped, but verify to fail fast if not
        TT_FATAL(
            info.is_mapped,
            "MappedChipInfo entry for ASIC {} in fabric_node_id_to_mapping_ is not mapped",
            info.asic_id);
        const auto mesh_id_val = info.fabric_node_id.mesh_id;
        const auto host_rank = info.mesh_host_rank;
        const auto coord = info.mesh_coord;
        mesh_to_hosts[mesh_id_val].insert(host_rank);
        mesh_host_to_coords[mesh_id_val][host_rank].push_back(coord);
    }

    // Build minimal wraparound-aware ranges per host
    // Convert to ordered maps for deterministic iteration across hosts
    std::map<MeshId, std::map<MeshHostRankId, std::vector<MeshCoordinate>>> ordered_mesh_host_coords;
    for (const auto& [mesh_id, host_coords_map] : mesh_host_to_coords) {
        for (const auto& [host_rank, coords] : host_coords_map) {
            ordered_mesh_host_coords[mesh_id][host_rank] = coords;
        }
    }
    for (const auto& [mesh_id, host_coords_map] : ordered_mesh_host_coords) {
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
    mesh_host_rank_to_mpi_rank_.clear();
    std::size_t max_mesh_index = 0;
    // Convert to ordered map for deterministic iteration
    std::map<MeshId, std::unordered_set<MeshHostRankId>> ordered_mesh_to_hosts(
        mesh_to_hosts.begin(), mesh_to_hosts.end());
    for (const auto& [mid, _] : ordered_mesh_to_hosts) {
        max_mesh_index = std::max<std::size_t>(max_mesh_index, *mid + 1);
    }
    mesh_host_ranks_.resize(max_mesh_index, MeshContainer<MeshHostRankId>(MeshShape{1, 1}, MeshHostRankId{0}));
    // Convert range_map to ordered map for deterministic iteration
    std::map<MeshId, std::map<MeshHostRankId, MeshCoordinateRange>> ordered_mesh_host_to_range;
    for (const auto& [mesh_id, range_map] : mesh_host_to_range) {
        for (const auto& [host_rank, range] : range_map) {
            ordered_mesh_host_to_range[mesh_id].emplace(host_rank, range);
        }
    }
    for (const auto& [mesh_id, hosts] : ordered_mesh_to_hosts) {
        const auto& range_map = ordered_mesh_host_to_range.at(mesh_id);
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

    // If there's only one host rank per mesh in the mesh graph, ensure all meshes have host_rank entries
    // even if the current rank doesn't participate in all meshes. This is needed for initialize_distributed_contexts()
    // to be able to look up MPI ranks for all (mesh_id, host_rank) pairs.
    for (const auto& mesh_id : mesh_graph_.get_all_mesh_ids()) {
        const auto& host_ranks = mesh_graph_.get_host_ranks(mesh_id);
        // If there's only one host rank in the mesh graph and we don't have an entry for it, add it
        if (host_ranks.size() == 1) {
            MeshHostRankId mesh_host_rank{*host_ranks.values().front()};
            auto key = std::make_pair(mesh_id, mesh_host_rank);
            if (!mesh_host_rank_coord_ranges_.contains(key)) {
                // Get the full coordinate range for this mesh from the mesh graph
                MeshCoordinateRange coord_range = mesh_graph_.get_coord_range(mesh_id);
                mesh_host_rank_coord_ranges_.insert({key, coord_range});
            }
        }
    }
}

HostName TopologyMapper::get_hostname_for_mesh(MeshId mesh_id) const {
    // Get all hosts for this mesh_id from the fabric node mapping
    // Meshes can be multi-host, so we collect all unique hostnames and return the first one
    std::unordered_set<HostName> mesh_hosts;
    for (const auto& [fabric_node_id, info_ptr] : fabric_node_id_to_mapping_) {
        if (fabric_node_id.mesh_id == mesh_id && info_ptr != nullptr && info_ptr->is_mapped) {
            mesh_hosts.insert(info_ptr->hostname);
        }
    }

    TT_FATAL(!mesh_hosts.empty(), "Mesh mesh_id {} not found in fabric node mapping", *mesh_id);

    // Return the first hostname (for multi-host meshes, this represents one of the hosts)
    return *mesh_hosts.begin();
}

HostName TopologyMapper::get_hostname_for_switch(SwitchId switch_id) const {
    // Verify that the switch exists in the mesh graph
    const auto& switch_ids = mesh_graph_.get_switch_ids();
    bool switch_exists = false;
    for (const auto& existing_switch_id : switch_ids) {
        if (*existing_switch_id == *switch_id) {
            switch_exists = true;
            break;
        }
    }
    TT_FATAL(switch_exists, "Switch ID {} not found in mesh graph", *switch_id);

    // Convert SwitchId to MeshId and use the consolidated mesh hostname function
    return get_hostname_for_mesh(MeshId(*switch_id));
}

HostName TopologyMapper::get_hostname_for_fabric_node_id(FabricNodeId fabric_node_id) const {
    // Direct lookup in the fabric node to mapping
    auto it = fabric_node_id_to_mapping_.find(fabric_node_id);
    TT_FATAL(it != fabric_node_id_to_mapping_.end(), "Fabric node id {} not found in mapping", fabric_node_id);
    TT_FATAL(it->second != nullptr, "Null pointer in fabric_node_id_to_mapping_");
    TT_FATAL(it->second->is_mapped, "Fabric node id {} is not mapped", fabric_node_id);

    // Get the hostname from the MappedChipInfo
    return it->second->hostname;
}

int TopologyMapper::get_mpi_rank_for_mesh_host_rank(MeshId mesh_id, MeshHostRankId host_rank) const {
    // First, try to use the direct mapping if available (from gathered mesh bindings)
    auto direct_mapping_it = mesh_host_rank_to_mpi_rank_.find(std::make_pair(mesh_id, host_rank));
    if (direct_mapping_it != mesh_host_rank_to_mpi_rank_.end()) {
        return direct_mapping_it->second;
    }

    // Fallback: Find a fabric node with this mesh_id and host_rank
    // Use the coordinate range to find a chip_id, then get the fabric node id
    auto coord_range_it = mesh_host_rank_coord_ranges_.find(std::make_pair(mesh_id, host_rank));
    TT_FATAL(
        coord_range_it != mesh_host_rank_coord_ranges_.end(),
        "TopologyMapper: host_rank {} not found for mesh {}",
        host_rank.get(),
        mesh_id.get());

    // Get a chip_id from the coordinate range (use the start coordinate)
    const auto& coord_range = coord_range_it->second;
    MeshCoordinate start_coord = coord_range.start_coord();
    ChipId chip_id = mesh_graph_.coordinate_to_chip(mesh_id, start_coord);
    FabricNodeId fabric_node_id(mesh_id, chip_id);

    // Try to get the hostname for this fabric node
    auto fabric_node_it = fabric_node_id_to_mapping_.find(fabric_node_id);
    if (fabric_node_it != fabric_node_id_to_mapping_.end() && fabric_node_it->second != nullptr &&
        fabric_node_it->second->is_mapped) {
        // Fabric node exists in mapping, use it
        HostName hostname = fabric_node_it->second->hostname;
        return static_cast<int>(physical_system_descriptor_.get_rank_for_hostname(hostname));
    }

    // Fabric node not found in mapping (current rank doesn't participate in this mesh)
    // Try to find any fabric node for this mesh to get the hostname
    for (const auto& [fnode_id, info_ptr] : fabric_node_id_to_mapping_) {
        if (fnode_id.mesh_id == mesh_id && info_ptr != nullptr && info_ptr->is_mapped) {
            HostName hostname = info_ptr->hostname;
            return static_cast<int>(physical_system_descriptor_.get_rank_for_hostname(hostname));
        }
    }

    // If we still can't find it, this is an error
    TT_FATAL(
        false, "TopologyMapper: Cannot determine MPI rank for mesh {} host_rank {}", mesh_id.get(), host_rank.get());
    return -1;  // Unreachable
}

void TopologyMapper::print_logical_adjacency_map(const std::map<MeshId, LogicalAdjacencyMap>& adj_map) const {
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

void TopologyMapper::print_physical_adjacency_map(const std::map<MeshId, PhysicalAdjacencyMap>& adj_map) const {
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

IntraMeshConnectivity TopologyMapper::get_intra_mesh_connectivity(MeshId mesh_id) const {
    // Passthrough to mesh graph - return the full intra-mesh connectivity structure
    // The mesh_id parameter is validated by checking bounds against the connectivity structure
    const auto& connectivity = mesh_graph_.get_intra_mesh_connectivity();
    TT_FATAL(
        *mesh_id < connectivity.size(),
        "TopologyMapper: mesh_id {} not found in mesh graph (connectivity size: {})",
        mesh_id,
        connectivity.size());
    return connectivity;
}

InterMeshConnectivity TopologyMapper::get_inter_mesh_connectivity(MeshId mesh_id) const {
    // Passthrough to mesh graph - return the full inter-mesh connectivity structure
    // The mesh_id parameter is validated by checking bounds against the connectivity structure
    const auto& connectivity = mesh_graph_.get_inter_mesh_connectivity();
    TT_FATAL(
        *mesh_id < connectivity.size(),
        "TopologyMapper: mesh_id {} not found in mesh graph (connectivity size: {})",
        mesh_id,
        connectivity.size());
    return connectivity;
}

namespace {
/**
 * @brief Determines the maximum number of local Ethernet connections per direction between ASICs in the system.
 *
 * For each ASIC in the provided PhysicalSystemDescriptor, this function examines all neighboring ASICs and counts
 * the number of Ethernet connections to each neighbor that are marked as local (i.e., connection.is_local is true).
 * It returns the maximum number of such local connections found in any direction for any ASIC.
 *
 * @param psd The PhysicalSystemDescriptor representing the system's ASICs and their interconnections.
 * @return The maximum number of local Ethernet connections per direction between any two ASICs.
 */
std::uint32_t get_num_connections_per_direction(const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    // Check the number of connections per direction for each asic
    std::uint32_t num_connections_per_direction = 1;  // Default to 1 connection per direction
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& [asic_id, asic_descriptor] : psd.get_asic_descriptors()) {
        auto neighbors = psd.get_asic_neighbors(asic_id);
        for (const auto& neighbor : neighbors) {
            auto connections = psd.get_eth_connections(asic_id, neighbor);
            std::uint32_t num_local_connections = 0;
            for (const auto& connection : connections) {
                if (cluster.get_cluster_type() == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY &&
                    (connection.src_chan == 8 || connection.src_chan == 9)) {
                    // Skip internally connected Z links, since these are not to be used by any single node
                    // Fabric Topologies.
                    // TODO: Ridvan - need a better way of handling internally connected Z links.
                    // Relying on ports 8 and 9 is not sustainable
                    continue;
                }
                if (connection.is_local) {
                    num_local_connections++;
                }
            }
            num_connections_per_direction = std::max(num_connections_per_direction, num_local_connections);
        }
    }
    return num_connections_per_direction;
}

/**
 * @brief Generate all possible mesh shapes that can be formed from a given number of chips
 *
 * This function generates all possible rectangular mesh shapes (MxN) where M*N equals the total number of chips.
 * It tries shapes from total_number_of_chips down to 1, prioritizing 2D shapes over 1D line topologies.
 *
 * @param total_number_of_chips The total number of chips to form a mesh from
 * @return std::vector<MeshShape> Vector of possible mesh shapes, ordered by preference
 */
std::vector<MeshShape> generate_possible_cluster_shapes(std::uint32_t total_number_of_chips) {
    // Come up with all possible mesh shapes that can be formed from the given number of chips
    // Try shapes for total_number_of_chips first, then total_number_of_chips - 1, etc., down to 1
    // All 1D cases (where one dimension is 1) are saved for the end
    std::vector<MeshShape> mesh_shapes_to_try;
    std::vector<MeshShape> one_d_shapes;

    // Try from total_number_of_chips down to 1
    for (std::uint32_t num_chips = total_number_of_chips; num_chips > 0; num_chips--) {
        // Find all divisor pairs (x, y) where x * y = num_chips
        // Only check divisors up to sqrt(num_chips) for efficiency
        std::uint32_t sqrt_num_chips = static_cast<std::uint32_t>(std::sqrt(num_chips));

        for (std::uint32_t i = sqrt_num_chips; i > 0; i--) {
            if (num_chips % i == 0) {
                auto x = num_chips / i;
                auto y = i;
                // Normalize: always put larger dimension first to avoid duplicates like (x,y) and (y,x)
                // This ensures (x,y) and (y,x) are treated as the same shape
                auto larger_dim = std::max(x, y);
                auto smaller_dim = std::min(x, y);
                MeshShape shape(larger_dim, smaller_dim);

                // NOTE: Special case for t3k 4x2 mesh shape, change it to 2x4 to avoid performance issues with mesh
                // device shape
                if (larger_dim == 4 && smaller_dim == 2) {
                    shape = MeshShape(2, 4);
                }

                // if odd shape then skip
                if ((larger_dim % 2 != 0 && larger_dim != 1) || (smaller_dim % 2 != 0 && smaller_dim != 1)) {
                    continue;
                }

                // Save 1D cases (where one dimension is 1) to be added at the end
                if (smaller_dim == 1) {
                    // Avoid duplicates by checking if this shape is already in one_d_shapes
                    if (std::find(one_d_shapes.begin(), one_d_shapes.end(), shape) == one_d_shapes.end()) {
                        one_d_shapes.push_back(shape);
                    }
                } else {
                    // Avoid duplicates by checking if this shape is already in mesh_shapes_to_try
                    if (std::find(mesh_shapes_to_try.begin(), mesh_shapes_to_try.end(), shape) ==
                        mesh_shapes_to_try.end()) {
                        mesh_shapes_to_try.push_back(shape);
                    }
                }
            }
        }
    }

    // Append all 1D shapes at the end
    mesh_shapes_to_try.insert(mesh_shapes_to_try.end(), one_d_shapes.begin(), one_d_shapes.end());

    return mesh_shapes_to_try;
}
}  // namespace

MeshGraph TopologyMapper::generate_mesh_graph_from_physical_system_descriptor(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor, FabricConfig fabric_config) {
    // Come up with the biggest mesh that can be formed by the physical system descriptor based on number of chips
    FabricType fabric_type = get_fabric_type(fabric_config);

    // Detect the number of connections per direction using the psd
    const auto number_of_connections = get_num_connections_per_direction(physical_system_descriptor);

    // Get the total number of chips in the physical system descriptor
    const auto total_number_of_chips = physical_system_descriptor.get_asic_descriptors().size();

    // Extract ASIC IDs from the descriptors map
    std::vector<tt::tt_metal::AsicID> all_asic_ids;
    for (const auto& [asic_id, _] : physical_system_descriptor.get_asic_descriptors()) {
        all_asic_ids.push_back(asic_id);
    }

    // Form physical adjacency matrix from physical system descriptor
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}] = std::map<tt::tt_metal::AsicID, MeshHostRankId>();
    for (const auto& asic_id : all_asic_ids) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = MeshHostRankId{0};
    }
    auto physical_adjacency_matrix = tt::tt_metal::experimental::tt_fabric::build_adjacency_map_physical(
        physical_system_descriptor, asic_id_to_mesh_rank);

    // Generate possible mesh shapes
    std::vector<MeshShape> mesh_shapes_to_try = generate_possible_cluster_shapes(total_number_of_chips);

    // Try all possible mesh shapes
    const MeshId mesh_id{0};
    for (const auto& mesh_shape : mesh_shapes_to_try) {
        auto mesh_graph = MeshGraph::generate_mesh_graph_of_shape(mesh_shape, fabric_type, number_of_connections);
        auto logical_adjacency_matrix = tt::tt_metal::experimental::tt_fabric::build_adjacency_map_logical(mesh_graph);

        // Extract adjacency maps for this mesh_id
        if (!logical_adjacency_matrix.contains(mesh_id) || !physical_adjacency_matrix.contains(mesh_id)) {
            continue;
        }

        const auto& logical_adj = logical_adjacency_matrix.at(mesh_id);
        const auto& physical_adj = physical_adjacency_matrix.at(mesh_id);

        // Build node_to_host_rank map - assume single mesh, all nodes on same host rank
        std::map<FabricNodeId, MeshHostRankId> node_to_host_rank;
        auto chip_ids = mesh_graph.get_chip_ids(mesh_id);
        const MeshHostRankId single_host_rank{0};
        for (const auto& chip_id : chip_ids.values()) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);
            node_to_host_rank[fabric_node_id] = single_host_rank;
        }

        // Extract asic_to_host_rank for this mesh_id
        const auto& asic_to_host_rank = asic_id_to_mesh_rank.at(mesh_id);

        // Do the mapping and see if its successful
        tt::tt_metal::experimental::tt_fabric::TopologyMappingConfig config;
        config.strict_mode = false;  // Use relaxed mode for initial matching

        auto mapping_result = tt::tt_metal::experimental::tt_fabric::map_mesh_to_physical(
            mesh_id, logical_adj, physical_adj, node_to_host_rank, asic_to_host_rank, config);

        // Return mesh_graph if mapping is successful
        if (mapping_result.success) {
            // Check if the final mesh size doesn't match the number of physical chips
            size_t final_mesh_size = mesh_shape.mesh_size();
            if (final_mesh_size < total_number_of_chips) {
                // Format mesh shape as "2x4" style string
                std::string mesh_shape_str;
                for (size_t i = 0; i < mesh_shape.dims(); ++i) {
                    if (i > 0) {
                        mesh_shape_str += "x";
                    }
                    mesh_shape_str += std::to_string(mesh_shape[i]);
                }

                log_warning(
                    tt::LogFabric,
                    "TopologyMapper auto-discovery: Downgrading to mesh shape {} ({} total nodes) for {} physical "
                    "chips. "
                    "Some physical chips may not be used. This may indicate connectivity issues, topology mismatches, "
                    "or insufficient fabric links between chips. Verify your physical chip connectivity and ensure "
                    "that the fabric links are correctly configured.",
                    mesh_shape_str,
                    final_mesh_size,
                    total_number_of_chips);
            }
            return mesh_graph;
        }
    }
    // Throw if no possible mesh shape is found to match, this means there are no devices! This should never happen
    TT_THROW("No possible mesh shape found to match physical adjacency matrix");
}

}  // namespace tt::tt_fabric
