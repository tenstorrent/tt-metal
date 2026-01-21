// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/hw/inc/hostdev/socket.h"

using namespace tt::tt_metal::distributed::multihost;

namespace tt::tt_metal::distributed {

namespace {

struct SocketSenderSize {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t md_size_bytes = tt::align(sizeof(sender_socket_md), l1_alignment);
    const uint32_t ack_size_bytes = tt::align(sizeof(uint32_t), l1_alignment);
    const uint32_t enc_size_bytes = tt::align(sizeof(sender_downstream_encoding), l1_alignment);
};

// Need to index the connections to properly read the FabricNodeId from peer descriptor.
// This will get cleaned up with the improved socket APIs. See Issue #27207
std::unordered_map<MeshCoordinate, std::unordered_map<CoreCoord, std::vector<std::pair<uint32_t, SocketConnection>>>>
group_socket_connections(const SocketConfig& config, SocketEndpoint socket_endpoint) {
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    // Group by endpoint device coordinate, then by endpoint core
    std::
        unordered_map<MeshCoordinate, std::unordered_map<CoreCoord, std::vector<std::pair<uint32_t, SocketConnection>>>>
            grouped_connections;
    uint32_t conn_idx = 0;
    for (const auto& connection : config.socket_connection_config) {
        const auto& core = is_sender ? connection.sender_core : connection.receiver_core;
        grouped_connections[core.device_coord][core.core_coord].push_back(std::make_pair(conn_idx++, connection));
    }
    return grouped_connections;
}

std::unordered_map<SocketConnection, uint32_t> get_receiver_ids_per_sender(const SocketConfig& config) {
    std::unordered_map<SocketConnection, uint32_t> connection_to_receiver_id;
    std::unordered_map<MeshCoreCoord, uint32_t> sender_counter;

    // Assign unique IDs starting from 0 for each sender
    for (const auto& connection : config.socket_connection_config) {
        uint32_t receiver_id = sender_counter[connection.sender_core]++;
        connection_to_receiver_id[connection] = receiver_id;
    }

    return connection_to_receiver_id;
}

// Get the maximum number of downstreams per sender core to calcalate size of the metadata buffer.
// This will get cleaned up along with improved socket APIs. See Issue #27207
uint32_t get_max_num_downstreams_per_core(const SocketConfig& config) {
    std::unordered_map<MeshCoreCoord, uint32_t> num_downstreams_per_core;
    uint32_t max_num_downstreams = 0;
    for (const auto& connection : config.socket_connection_config) {
        num_downstreams_per_core[connection.sender_core]++;
        max_num_downstreams = std::max(max_num_downstreams, num_downstreams_per_core[connection.sender_core]);
    }
    return max_num_downstreams;
}

void validate_fabric_config_for_sockets(
    tt_fabric::FabricConfig fabric_config, tt_fabric::FabricNodeId sender_node, tt_fabric::FabricNodeId recv_node) {
    if (sender_node != recv_node) {
        TT_FATAL(
            fabric_config != tt_fabric::FabricConfig::DISABLED,
            "Can only create multi-device sockets with fabric enabled.");
    }

    static const std::unordered_set<tt_fabric::FabricConfig> supported_fabrics = {
        tt_fabric::FabricConfig::FABRIC_1D,
        tt_fabric::FabricConfig::FABRIC_1D_RING,
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricConfig::DISABLED  // Fabric can be disabled as long as socket endpoints are on the same
                                           // physical device
    };

    bool fabric_config_supported = supported_fabrics.contains(fabric_config);
    TT_FATAL(fabric_config_supported, "Unsupported Fabric Config for Sockets specified {}", fabric_config);
}

// This does not return a FabricNodeId because for 1D fabric, we return a distance between the sender and receiver
// instead of a chip id (FabricNodeId also stores its chip_id as uint32_t)
std::pair<tt_fabric::MeshId, uint32_t> get_sender_receiver_chip_fabric_encoding(
    tt_fabric::FabricNodeId sender_node_id,
    tt_fabric::FabricNodeId recv_node_id,
    tt_fabric::FabricConfig fabric_config,
    SocketEndpoint socket_endpoint) {
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;

    validate_fabric_config_for_sockets(fabric_config, sender_node_id, recv_node_id);

    if (fabric_config == tt_fabric::FabricConfig::FABRIC_1D or
        fabric_config == tt_fabric::FabricConfig::FABRIC_1D_RING) {
        // 1D Fabric requires passing in the number of hops between the sender and receiver
        // Assume 1D is a single mesh
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        TT_FATAL(
            sender_node_id.mesh_id == recv_node_id.mesh_id,
            "1D Fabric requires sender and receiver to be on the same mesh");
        auto mesh_id = is_sender ? sender_node_id.mesh_id : recv_node_id.mesh_id;
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh_id);
        TT_FATAL(mesh_shape.dims() == 2, "1D Fabric requires a 2D mesh");
        MeshCoordinate sender_global_coord =
            MeshCoordinate(sender_node_id.chip_id / mesh_shape[1], sender_node_id.chip_id % mesh_shape[1]);
        MeshCoordinate recv_global_coord =
            MeshCoordinate(recv_node_id.chip_id / mesh_shape[1], recv_node_id.chip_id % mesh_shape[1]);
        TT_FATAL(
            sender_global_coord[0] == recv_global_coord[0] || sender_global_coord[1] == recv_global_coord[1],
            "Sender and receiver chips must be in the same row or column when using 1D Line Fabric");

        // Calculate the number of hops between the sender and receiver needed for 1D Fabric
        // mesh_id is a don't care value for 1D Fabric
        return std::make_pair(
            mesh_id,
            std::abs(static_cast<int>(sender_global_coord[0]) - static_cast<int>(recv_global_coord[0])) +
                std::abs(static_cast<int>(sender_global_coord[1]) - static_cast<int>(recv_global_coord[1])));
    }  // 2D/Mesh Fabric requires looking up "logical" encodings from the control plane
    if (is_sender) {
        return {recv_node_id.mesh_id, recv_node_id.chip_id};
    }
    return {sender_node_id.mesh_id, sender_node_id.chip_id};
}

// Validate the remote descriptor received from the peer against the local descriptor.
// This is done when:
// 1. Consolidating descriptors across hosts in a Big Mesh (validate validate_buffer_addresses = true in this case),
//    due the lock-step allocation of buffers across a mesh.
// 2. Exchanging descriptors between the controller host and all peers.
void validate_remote_desc(
    const SocketPeerDescriptor& local_desc,
    const SocketPeerDescriptor& remote_desc,
    bool validate_buffer_addresses = true) {
    // Verify that socket connection config matches
    TT_FATAL(
        local_desc.config.socket_connection_config.size() == remote_desc.config.socket_connection_config.size(),
        "Mismatch in number of socket connections during handshake.");
    for (size_t i = 0; i < local_desc.config.socket_connection_config.size(); ++i) {
        const auto& local_conn = local_desc.config.socket_connection_config[i];
        const auto& remote_conn = remote_desc.config.socket_connection_config[i];
        TT_FATAL(local_conn.sender_core == remote_conn.sender_core, "Mismatch in sender core during handshake.");
        TT_FATAL(local_conn.receiver_core == remote_conn.receiver_core, "Mismatch in receiver core during handshake.");
    }
    // Verify that socket memory config matches
    TT_FATAL(
        local_desc.config.socket_mem_config.socket_storage_type ==
            remote_desc.config.socket_mem_config.socket_storage_type,
        "Mismatch in socket storage type during handshake.");
    TT_FATAL(
        local_desc.config.socket_mem_config.fifo_size == remote_desc.config.socket_mem_config.fifo_size,
        "Mismatch in socket FIFO size during handshake.");

    TT_FATAL(
        local_desc.config.socket_mem_config.sender_sub_device == remote_desc.config.socket_mem_config.sender_sub_device,
        "Mismatch in sender sub-device during handshake.");

    TT_FATAL(
        local_desc.config.socket_mem_config.receiver_sub_device ==
            remote_desc.config.socket_mem_config.receiver_sub_device,
        "Mismatch in receiver sub-device during handshake.");
    if (validate_buffer_addresses) {
        TT_FATAL(
            local_desc.config_buffer_address == remote_desc.config_buffer_address,
            "Mismatch in config buffer address during handshake.");
        TT_FATAL(
            local_desc.data_buffer_address == remote_desc.data_buffer_address,
            "Mismatch in data buffer address during handshake.");
    }
    // Verify that the mesh IDs match
    TT_FATAL(
        local_desc.config.sender_mesh_id.value() == remote_desc.config.sender_mesh_id.value(),
        "Mismatch in sender mesh ID during handshake.");
    TT_FATAL(
        local_desc.config.receiver_mesh_id.value() == remote_desc.config.receiver_mesh_id.value(),
        "Mismatch in receiver mesh ID during handshake.");
}

Tag generate_descriptor_exchange_tag(tt_fabric::MeshId peer_mesh_id, std::optional<DistributedContextId> context_id) {
    // Generate a unique id to tag the exchange of socket peer
    // descriptors between the sender and receiver.
    // This is used to ensure that the sender and receiver are
    // exchanging the correct descriptors.
    static std::unordered_map<DistributedContextId, std::unordered_map<tt_fabric::MeshId, uint32_t>> exchange_tags;
    DistributedContextId unique_context_id = context_id.value_or(DistributedContext::get_current_world()->id());
    return Tag{static_cast<int>(exchange_tags[unique_context_id][peer_mesh_id]++)};
}
}  // namespace

std::shared_ptr<MeshBuffer> create_socket_config_buffer(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, SocketEndpoint socket_endpoint) {
    const auto& socket_connections = config.socket_connection_config;
    const auto& socket_mem_config = config.socket_mem_config;
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    uint32_t config_buffer_size = 0;
    if (is_sender) {
        const auto max_num_downstreams = get_max_num_downstreams_per_core(config);
        const SocketSenderSize sender_size;
        config_buffer_size =
            sender_size.md_size_bytes + max_num_downstreams * (sender_size.ack_size_bytes + sender_size.enc_size_bytes);
    } else {
        config_buffer_size = sizeof(receiver_socket_md);
    }
    std::set<CoreRange> all_cores_set;
    std::unordered_map<MeshCoordinate, std::set<CoreRange>> socket_cores_per_device;
    for (const auto& connection : socket_connections) {
        const auto& socket_device =
            is_sender ? connection.sender_core.device_coord : connection.receiver_core.device_coord;
        const auto& socket_core = is_sender ? connection.sender_core.core_coord : connection.receiver_core.core_coord;
        TT_FATAL(
            is_sender || socket_cores_per_device[socket_device].insert(socket_core).second,
            "Cannot reuse receiver cores in a single socket.");
        all_cores_set.insert(socket_core);
    }

    auto all_cores = CoreRangeSet(all_cores_set);
    auto num_cores = all_cores_set.size();
    auto total_config_buffer_size = num_cores * config_buffer_size;

    auto shard_params =
        ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {static_cast<uint32_t>(num_cores), 1});

    DeviceLocalBufferConfig buffer_specs = {
        .page_size = config_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = is_sender ? socket_mem_config.sender_sub_device : socket_mem_config.receiver_sub_device,
    };

    if (buffer_specs.sub_device_id.has_value()) {
        auto sub_device_id = buffer_specs.sub_device_id.value();
        auto device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        auto device_cores_set = CoreRangeSet(device_cores);
        TT_FATAL(device_cores_set.contains(all_cores), "Socket cores must be contained in the specified sub device");
    }

    MeshBufferConfig mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_config_buffer_size,
    };

    return MeshBuffer::create(mesh_buffer_specs, buffer_specs, device.get());
}

std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    const std::shared_ptr<MeshDevice>& receiver, const SocketConfig& config) {
    const auto& socket_mem_config = config.socket_mem_config;

    uint32_t num_data_cores = 0;
    CoreRangeSet shard_grid;
    if (socket_mem_config.socket_storage_type == BufferType::DRAM) {
        // Allocate DRAM Sharded Buffer
        shard_grid =
            CoreRange(CoreCoord(0, 0), CoreCoord(receiver->dram_grid_size().x - 1, receiver->dram_grid_size().y - 1));
        num_data_cores = receiver->dram_grid_size().x * receiver->dram_grid_size().y;
    } else {
        // Allocate Sharded buffer on worker cores
        auto receiver_sub_device_id = socket_mem_config.receiver_sub_device.value_or(SubDeviceId{0});
        shard_grid = receiver->worker_cores(HalProgrammableCoreType::TENSIX, receiver_sub_device_id);
        num_data_cores = receiver->num_worker_cores(HalProgrammableCoreType::TENSIX, receiver_sub_device_id);
    }
    // Allocate a shard of size fifo_size on each data core. User decides how these data-cores must be used
    const auto total_data_buffer_size = num_data_cores * socket_mem_config.fifo_size;
    DeviceLocalBufferConfig socket_data_buffer_specs = {
        .page_size = socket_mem_config.fifo_size,
        .buffer_type = socket_mem_config.socket_storage_type,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = socket_mem_config.socket_storage_type == BufferType::DRAM
                             ? std::nullopt
                             : socket_mem_config.receiver_sub_device,
    };
    MeshBufferConfig socket_data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };

    return MeshBuffer::create(socket_data_mesh_buffer_specs, socket_data_buffer_specs, receiver.get());
}

void write_socket_configs(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const SocketPeerDescriptor& local_descriptor,
    const SocketPeerDescriptor& peer_descriptor,
    SocketEndpoint socket_endpoint,
    const std::shared_ptr<MeshDevice>& peer_device) {
    auto* mesh_device = config_buffer->device();
    const auto& core_to_core_id = config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    const auto& config = peer_descriptor.config;
    auto grouped_connections = group_socket_connections(config, socket_endpoint);
    auto peer_config_buf_addr = peer_descriptor.config_buffer_address;
    const SocketSenderSize sender_size;
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    const auto receiver_ids_per_sender = get_receiver_ids_per_sender(config);
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    auto get_fabric_node_from_coord = [&](const MeshCoordinate& device_coord,
                                          const std::shared_ptr<MeshDevice>& peer_device,
                                          tt_fabric::MeshId peer_mesh_id) -> tt_fabric::FabricNodeId {
        if (peer_device) {
            return peer_device->get_fabric_node_id(device_coord);
        }
        return tt_fabric::FabricNodeId(peer_mesh_id, mesh_graph.coordinate_to_chip(peer_mesh_id, device_coord));
    };

    if (is_sender) {
        const auto max_num_downstreams = get_max_num_downstreams_per_core(config);
        const auto local_coord_range =
            control_plane.get_coord_range(local_descriptor.config.sender_mesh_id.value(), tt_fabric::MeshScope::LOCAL);
        const auto sender_total_size_bytes =
            sender_size.md_size_bytes +
            (max_num_downstreams * (sender_size.ack_size_bytes + sender_size.enc_size_bytes));

        std::vector<uint32_t> config_data(config_buffer->size() / sizeof(uint32_t), 0);

        for (const auto& [device_coord, cores_map] : grouped_connections) {
            if (!local_coord_range.contains(device_coord)) {
                continue;
            }
            if (cores_map.size() > 1) {
                log_warning(
                    tt::LogAlways,
                    "Multiple sender cores on a single device may lead to errors due to Fabric limitations.");
            }
            for (const auto& [sender_core_coord, connections] : cores_map) {
                MeshCoreCoord sender_core = {device_coord, sender_core_coord};

                uint32_t idx = core_to_core_id.at(sender_core.core_coord);
                // write sender_socket_md (only once per sender core)
                uint32_t md_offset = idx * sender_total_size_bytes / sizeof(uint32_t);
                config_data[md_offset++] = connections.size();                   // num_downstreams
                config_data[md_offset++] = peer_descriptor.data_buffer_address;  // write_ptr
                config_data[md_offset++] = 0;                                    // bytes_sent
                config_data[md_offset++] = peer_config_buf_addr;                 // downstream_bytes_sent_addr
                config_data[md_offset++] = peer_descriptor.data_buffer_address;  // downstream_fifo_addr
                config_data[md_offset++] = config.socket_mem_config.fifo_size;   // downstream_fifo_total_size
                config_data[md_offset++] = is_sender;                            // is_sender

                // Write downstream encodings for each receiver of this sender core
                uint32_t enc_offset = (idx * sender_total_size_bytes + sender_size.md_size_bytes +
                                       sender_size.ack_size_bytes * connections.size()) /
                                      sizeof(uint32_t);

                // Write one encoding per receiver, ordered by receiver ID
                for (const auto& [conn_idx, connection] : connections) {
                    MeshCoordinate recv_device_coord = connection.receiver_core.device_coord;
                    auto recv_virtual_core =
                        mesh_device->worker_core_from_logical_core(connection.receiver_core.core_coord);
                    tt_fabric::FabricNodeId recv_fabric_node_id =
                        get_fabric_node_from_coord(recv_device_coord, peer_device, config.receiver_mesh_id.value());
                    uint32_t receiver_id = receiver_ids_per_sender.at(connection);
                    auto [downstream_mesh_id, downstream_chip_id] = get_sender_receiver_chip_fabric_encoding(
                        mesh_device->get_fabric_node_id(sender_core.device_coord),
                        recv_fabric_node_id,
                        fabric_config,
                        SocketEndpoint::SENDER);
                    // Write to the correct slot based on receiver ID
                    uint32_t receiver_enc_offset =
                        enc_offset + (receiver_id * (sender_size.enc_size_bytes / sizeof(uint32_t)));
                    config_data[receiver_enc_offset] = *downstream_mesh_id;      // downstream_mesh_id
                    config_data[receiver_enc_offset + 1] = downstream_chip_id;   // downstream_chip_id
                    config_data[receiver_enc_offset + 2] = recv_virtual_core.y;  // downstream_noc_y
                    config_data[receiver_enc_offset + 3] = recv_virtual_core.x;  // downstream_noc_x
                }
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    } else {
        std::vector<receiver_socket_md> config_data(
            config_buffer->size() / sizeof(receiver_socket_md), receiver_socket_md());
        const auto local_coord_range = control_plane.get_coord_range(
            local_descriptor.config.receiver_mesh_id.value(), tt_fabric::MeshScope::LOCAL);

        for (const auto& [device_coord, cores_map] : grouped_connections) {
            if (!local_coord_range.contains(device_coord)) {
                continue;
            }

            for (const auto& [recv_core_coord, indexed_connections] : cores_map) {
                const auto& [conn_idx, connection] =
                    indexed_connections.front();  // Only one connection per receiver core for now
                MeshCoordinate sender_device_coord = connection.sender_core.device_coord;
                auto sender_virtual_core =
                    mesh_device->worker_core_from_logical_core(connection.sender_core.core_coord);
                tt_fabric::FabricNodeId sender_fabric_node_id =
                    get_fabric_node_from_coord(sender_device_coord, peer_device, config.sender_mesh_id.value());
                MeshCoreCoord recv_core = {device_coord, recv_core_coord};

                auto [upstream_mesh_id, upstream_chip_id] = get_sender_receiver_chip_fabric_encoding(
                    sender_fabric_node_id,
                    mesh_device->get_fabric_node_id(recv_core.device_coord),
                    fabric_config,
                    SocketEndpoint::RECEIVER);

                uint32_t idx = core_to_core_id.at(recv_core.core_coord);
                auto& md = config_data[idx];
                md.bytes_sent = 0;
                md.bytes_acked = 0;
                md.read_ptr = local_descriptor.data_buffer_address;
                md.fifo_addr = local_descriptor.data_buffer_address;
                md.fifo_total_size = config.socket_mem_config.fifo_size;
                md.upstream_mesh_id = *upstream_mesh_id;
                md.upstream_chip_id = upstream_chip_id;
                md.upstream_noc_y = sender_virtual_core.y;
                md.upstream_noc_x = sender_virtual_core.x;
                md.upstream_bytes_acked_addr = peer_config_buf_addr + sender_size.md_size_bytes +
                                               sender_size.ack_size_bytes * receiver_ids_per_sender.at(connection);
                md.is_sender = is_sender;
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    }
}

SocketPeerDescriptor generate_local_endpoint_descriptor(
    const MeshSocket& socket_endpoint, std::optional<DistributedContextId> context_id) {
    const auto& config = socket_endpoint.get_config();
    bool is_sender = socket_endpoint.get_socket_endpoint_type() == SocketEndpoint::SENDER;

    auto peer_mesh_id = is_sender ? config.receiver_mesh_id.value() : config.sender_mesh_id.value();
    SocketPeerDescriptor local_endpoint_desc = {
        .config = config,
        .config_buffer_address = socket_endpoint.get_config_buffer()->address(),
        .data_buffer_address = is_sender ? 0 : socket_endpoint.get_data_buffer()->address(),
        .exchange_tag = generate_descriptor_exchange_tag(peer_mesh_id, context_id)  // Unique tag for this exchange
    };
    return local_endpoint_desc;
}

void validate_subordinate_descriptors(
    const SocketPeerDescriptor& desc,
    std::vector<uint8_t>& serialized_local_desc,
    const tt_metal::distributed::multihost::Rank& controller_rank,
    const std::vector<tt_metal::distributed::multihost::Rank>& ranks,
    const std::shared_ptr<const multihost::DistributedContext>& context) {
    // Send size of serialized descriptor first, so that the peer knows the amount of data to expect
    int local_descriptor_size_bytes = serialized_local_desc.size();
    SocketPeerDescriptor subordinate_desc = {};
    for (const auto& rank : ranks) {
        if (rank == controller_rank) {
            continue;
        }
        if (context->rank() == controller_rank) {
            int expected_subordinate_descriptor_size_bytes = 0;
            context->recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&expected_subordinate_descriptor_size_bytes),
                    sizeof(expected_subordinate_descriptor_size_bytes)),
                rank,
                desc.exchange_tag);
            auto subordinate_descriptor_size_bytes = context->snoop_incoming_msg_size(Rank{rank}, desc.exchange_tag);
            TT_FATAL(
                subordinate_descriptor_size_bytes == expected_subordinate_descriptor_size_bytes,
                "Expected {} bytes in the subordinate descriptor, but got {} bytes during multi-host handshake.",
                expected_subordinate_descriptor_size_bytes,
                subordinate_descriptor_size_bytes);
            std::vector<uint8_t> serialized_subordinate_desc(subordinate_descriptor_size_bytes);
            context->recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_subordinate_desc.data(), serialized_subordinate_desc.size())),
                Rank{rank},
                desc.exchange_tag);
            subordinate_desc = deserialize_from_bytes(serialized_subordinate_desc);
            validate_remote_desc(desc, subordinate_desc, true);
        } else {
            context->send(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&local_descriptor_size_bytes), sizeof(local_descriptor_size_bytes)),
                controller_rank,
                desc.exchange_tag);
            context->send(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_local_desc.data(), serialized_local_desc.size())),
                controller_rank,
                desc.exchange_tag);
        }
    }
}

void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context,
    const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table) {
    const auto& config = desc.config;
    bool is_sender = socket_endpoint_type == SocketEndpoint::SENDER;
    auto my_mesh_id = is_sender ? config.sender_mesh_id.value() : config.receiver_mesh_id.value();
    auto peer_mesh_id = is_sender ? config.receiver_mesh_id.value() : config.sender_mesh_id.value();

    std::vector<tt_metal::distributed::multihost::Rank> my_mesh_id_ranks =
        get_ranks_for_mesh_id(my_mesh_id, rank_translation_table);
    std::vector<tt_metal::distributed::multihost::Rank> peer_mesh_id_ranks =
        get_ranks_for_mesh_id(peer_mesh_id, rank_translation_table);
    tt_metal::distributed::multihost::Rank controller_rank =
        *std::min_element(my_mesh_id_ranks.begin(), my_mesh_id_ranks.end());

    // Forward descriptor to controller host. This host is on the same mesh with the lowest host rank.
    std::vector<uint8_t> serialized_local_desc = serialize_to_bytes(desc);
    execute_with_timeout([&]() {
        validate_subordinate_descriptors(desc, serialized_local_desc, controller_rank, my_mesh_id_ranks, context);
    });
    int local_descriptor_size_bytes = serialized_local_desc.size();
    // Once all descriptors are validated, forward the socket descriptor from the controller to all peers.
    if (context->rank() == controller_rank) {
        for (const auto& peer_rank : peer_mesh_id_ranks) {
            execute_with_timeout([&]() {
                context->send(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&local_descriptor_size_bytes),
                        sizeof(local_descriptor_size_bytes)),
                    Rank{peer_rank},
                    desc.exchange_tag  // Forward this descriptor over the specified tag
                );
            });
            // Send the serialized descriptor
            execute_with_timeout([&]() {
                context->send(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(serialized_local_desc.data(), serialized_local_desc.size())),
                    Rank{peer_rank},
                    desc.exchange_tag  // Forward this descriptor over the specified tag
                );
            });
        }
    }
}

SocketPeerDescriptor receive_and_verify_descriptor_from_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context,
    const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table) {
    const auto& config = desc.config;
    bool is_sender = socket_endpoint_type == SocketEndpoint::SENDER;
    auto peer_mesh_id = is_sender ? config.receiver_mesh_id.value() : config.sender_mesh_id.value();
    auto peer_ranks = get_ranks_for_mesh_id(peer_mesh_id, rank_translation_table);
    tt_metal::distributed::multihost::Rank peer_controller_rank =
        *std::min_element(peer_ranks.begin(), peer_ranks.end());

    // Query the size of the serialized descriptor first (this is the only element in the header)
    std::size_t msg_header_size = 0;
    execute_with_timeout(
        [&]() { msg_header_size = context->snoop_incoming_msg_size(Rank{peer_controller_rank}, desc.exchange_tag); });
    TT_FATAL(
        msg_header_size == sizeof(int),
        "Expected {} bytes in the header for socket descriptor, but got {} bytes during multi-host handshake.",
        sizeof(int),
        msg_header_size);

    int expected_descriptor_size_bytes = 0;
    execute_with_timeout([&]() {
        context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&expected_descriptor_size_bytes), sizeof(expected_descriptor_size_bytes)),
            Rank{peer_controller_rank},
            desc.exchange_tag  // Read the descriptor over the specified tag
        );
    });
    // Validate that the size in the header matches the descriptor message size
    auto descriptor_size_bytes = context->snoop_incoming_msg_size(Rank{peer_controller_rank}, desc.exchange_tag);
    TT_FATAL(
        descriptor_size_bytes == expected_descriptor_size_bytes,
        "Expected {} bytes in the socket descriptor, but got {} bytes during multi-host handshake.",
        expected_descriptor_size_bytes,
        descriptor_size_bytes);

    // Allocate a buffer to receive the serialized descriptor
    std::vector<uint8_t> serialized_remote_desc(descriptor_size_bytes);
    // Receive the serialized descriptor
    execute_with_timeout([&]() {
        context->recv(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_remote_desc.data(), serialized_remote_desc.size())),
            Rank{peer_controller_rank},
            desc.exchange_tag  // Read the descriptor over the specified tag
        );
    });
    // Deserialize the received descriptor
    auto remote_desc = deserialize_from_bytes(serialized_remote_desc);
    validate_remote_desc(desc, remote_desc, false);
    return remote_desc;
}

std::array<std::unordered_map<MeshCoordinate, tt::tt_fabric::FabricNodeId>, 2> generate_fabric_node_id_map(
    const SocketConfig& config,
    const std::shared_ptr<MeshDevice>& sender_device,
    const std::shared_ptr<MeshDevice>& receiver_device) {
    std::array<std::unordered_map<MeshCoordinate, tt::tt_fabric::FabricNodeId>, 2> fabric_node_id_map;
    const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();

    for (uint32_t i = 0; i < config.socket_connection_config.size(); ++i) {
        const auto& connection = config.socket_connection_config[i];
        if (sender_device) {
            fabric_node_id_map[static_cast<std::underlying_type_t<SocketEndpoint>>(SocketEndpoint::SENDER)].emplace(
                connection.sender_core.device_coord,
                sender_device->get_fabric_node_id(connection.sender_core.device_coord));
        } else {
            TT_FATAL(config.sender_mesh_id.has_value(), "Sender mesh id is not set.");
            fabric_node_id_map[static_cast<std::underlying_type_t<SocketEndpoint>>(SocketEndpoint::SENDER)].emplace(
                connection.sender_core.device_coord,
                tt::tt_fabric::FabricNodeId(
                    config.sender_mesh_id.value(),
                    mesh_graph.coordinate_to_chip(config.sender_mesh_id.value(), connection.sender_core.device_coord)));
        }
        if (receiver_device) {
            fabric_node_id_map[static_cast<std::underlying_type_t<SocketEndpoint>>(SocketEndpoint::RECEIVER)].emplace(
                connection.receiver_core.device_coord,
                receiver_device->get_fabric_node_id(connection.receiver_core.device_coord));
        } else {
            TT_FATAL(config.receiver_mesh_id.has_value(), "Receiver mesh id is not set.");
            fabric_node_id_map[static_cast<std::underlying_type_t<SocketEndpoint>>(SocketEndpoint::RECEIVER)].emplace(
                connection.receiver_core.device_coord,
                tt::tt_fabric::FabricNodeId(
                    config.receiver_mesh_id.value(),
                    mesh_graph.coordinate_to_chip(
                        config.receiver_mesh_id.value(), connection.receiver_core.device_coord)));
        }
    }
    return fabric_node_id_map;
}

std::vector<multihost::Rank> get_ranks_for_mesh_id(
    tt_fabric::MeshId mesh_id, const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table) {
    const auto& global_logical_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    std::vector<multihost::Rank> ranks;

    for (const auto& [rank, mesh_id_and_host_rank] : global_logical_bindings) {
        if (std::get<0>(mesh_id_and_host_rank) == mesh_id && rank_translation_table.contains(rank)) {
            ranks.push_back(rank_translation_table.at(rank));
        }
    }

    return ranks;
}

}  // namespace tt::tt_metal::distributed
