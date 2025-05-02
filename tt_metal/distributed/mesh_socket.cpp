// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/socket.h"

namespace tt::tt_metal::distributed {

std::shared_ptr<MeshBuffer> create_sender_socket_config_buffer(
    std::shared_ptr<MeshDevice> sender, const socket_config_t& config) {
    const auto& socket_connections = config.socket_connection_config;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t sender_config_buffer_size = sizeof(sender_socket_md);
    std::vector<CoreRange> all_sender_cores_vec = {};
    for (const auto& connection : socket_connections) {
        all_sender_cores_vec.push_back(connection.sender_core.second);
    }
    auto all_sender_cores = CoreRangeSet(all_sender_cores_vec);
    auto num_sender_cores = all_sender_cores_vec.size();
    auto total_sender_config_buffer_size = num_sender_cores * sender_config_buffer_size;
    auto sender_config_shard_params =
        ShardSpecBuffer(all_sender_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_sender_cores, 1});
    DeviceLocalBufferConfig sender_config_buffer_specs = {
        .page_size = sender_config_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_config_shard_params,
        .bottom_up = std::nullopt,
    };

    MeshBufferConfig sender_config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_sender_config_buffer_size,
    };
    return MeshBuffer::create(sender_config_mesh_buffer_specs, sender_config_buffer_specs, sender.get());
}

std::shared_ptr<MeshBuffer> create_receiver_socket_config_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    const auto& socket_connections = config.socket_connection_config;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t recv_config_buffer_size = sizeof(receiver_socket_md);
    std::vector<CoreRange> all_recv_cores_vec = {};
    for (const auto& connection : socket_connections) {
        all_recv_cores_vec.push_back(connection.receiver_core.second);
    }
    auto all_recv_cores = CoreRangeSet(all_recv_cores_vec);
    auto num_recv_cores = all_recv_cores_vec.size();
    auto total_recv_config_buffer_size = num_recv_cores * recv_config_buffer_size;

    auto recv_config_shard_params =
        ShardSpecBuffer(all_recv_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_recv_cores, 1});

    DeviceLocalBufferConfig recv_config_buffer_specs = {
        .page_size = recv_config_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_config_shard_params,
        .bottom_up = std::nullopt,
    };

    MeshBufferConfig recv_config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_recv_config_buffer_size,
    };

    return MeshBuffer::create(recv_config_mesh_buffer_specs, recv_config_buffer_specs, receiver.get());
}

std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    const auto& socket_mem_config = config.socket_mem_config;
    // TODO: Support DRAM
    TT_FATAL(socket_mem_config.socket_type == BufferType::L1, "Socket data buffer must be L1");
    auto num_worker_cores = receiver->num_worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    const auto total_data_buffer_size = num_worker_cores * socket_mem_config.fifo_size;
    DeviceLocalBufferConfig socket_data_buffer_specs = {
        .page_size = socket_mem_config.fifo_size,
        .buffer_type = socket_mem_config.socket_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            receiver->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            {1, 1},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {num_worker_cores, 1}),
        .bottom_up = std::nullopt,
    };
    MeshBufferConfig socket_data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };

    return MeshBuffer::create(socket_data_mesh_buffer_specs, socket_data_buffer_specs, receiver.get());
}

void populate_sender_socket_config_buffer(
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config) {
    auto recv_bytes_sent_addr = recv_config_buffer->address();
    std::vector<sender_socket_md> sender_config_buffer_data(
        sender_config_buffer->size() / sizeof(sender_socket_md), sender_socket_md());
    const auto& sender_core_to_core_id =
        sender_config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;
    auto sender_mesh_device = sender_config_buffer->device();
    for (const auto& [sender, receiver] : config.socket_connection_config) {
        auto recv_virtual_core = sender_mesh_device->worker_core_from_logical_core(receiver.second);
        uint32_t sender_idx = sender_core_to_core_id.at(sender.second);
        auto& curr_sender_md = sender_config_buffer_data[sender_idx];
        curr_sender_md.write_ptr = socket_data_buffer->address();
        curr_sender_md.downstream_mesh_id =
            0;  // TODO: Get physical mesh ID from receiver f(receiver mesh_device + receiver mesh_coord)
        curr_sender_md.downstream_chip_id =
            0;  // TODO: Get physical chip ID from receiver f(receiver mesh_device + receiver mesh_coord)
        curr_sender_md.downstream_noc_y = recv_virtual_core.y;
        curr_sender_md.downstream_noc_x = recv_virtual_core.x;
        curr_sender_md.downstream_bytes_sent_addr = recv_bytes_sent_addr;
        curr_sender_md.downstream_fifo_addr = socket_data_buffer->address();
        curr_sender_md.downstream_fifo_total_size = config.socket_mem_config.fifo_size;
        curr_sender_md.is_sender = 1;
    }

    distributed::WriteShard(
        sender_mesh_device->mesh_command_queue(0),
        sender_config_buffer,
        sender_config_buffer_data,
        MeshCoordinate(0, 0),
        true);
}

void populate_receiver_socket_config_buffer(
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config) {
    auto sender_bytes_acked_addr = sender_config_buffer->address();
    std::vector<receiver_socket_md> recv_config_buffer_data(
        recv_config_buffer->size() / sizeof(receiver_socket_md), receiver_socket_md());
    const auto& recv_core_to_core_id =
        recv_config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;
    auto receiver_mesh_device = recv_config_buffer->device();
    for (const auto& [sender, receiver] : config.socket_connection_config) {
        auto sender_virtual_core = receiver_mesh_device->worker_core_from_logical_core(sender.second);
        uint32_t recv_idx = recv_core_to_core_id.at(receiver.second);
        auto& curr_recv_md = recv_config_buffer_data[recv_idx];
        curr_recv_md.read_ptr = socket_data_buffer->address();
        curr_recv_md.fifo_addr = socket_data_buffer->address();
        curr_recv_md.fifo_total_size = config.socket_mem_config.fifo_size;
        curr_recv_md.upstream_mesh_id =
            0;  // TODO: Get physical mesh ID from sender f(sender mesh_device + sender mesh_coord)
        curr_recv_md.upstream_chip_id =
            0;  // TODO: Get physical chip ID from sender f(sender mesh_device + sender mesh_coord)
        curr_recv_md.upstream_noc_y = sender_virtual_core.y;
        curr_recv_md.upstream_noc_x = sender_virtual_core.x;
        curr_recv_md.upstream_bytes_acked_addr = sender_bytes_acked_addr;
        curr_recv_md.is_sender = 0;
    }
    distributed::WriteShard(
        receiver_mesh_device->mesh_command_queue(0),
        recv_config_buffer,
        recv_config_buffer_data,
        MeshCoordinate(0, 0),
        true);
}

std::pair<mesh_socket_t, mesh_socket_t> create_sockets(
    std::shared_ptr<MeshDevice> sender, std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    auto sender_config_buffer = create_sender_socket_config_buffer(sender, config);
    auto recv_config_buffer = create_receiver_socket_config_buffer(receiver, config);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);
    populate_sender_socket_config_buffer(sender_config_buffer, recv_config_buffer, socket_data_buffer, config);
    populate_receiver_socket_config_buffer(recv_config_buffer, sender_config_buffer, socket_data_buffer, config);
    auto sender_socket = mesh_socket_t{
        .data_buffer = socket_data_buffer,
        .config_buffer = sender_config_buffer,
    };
    auto receiver_socket = mesh_socket_t{
        .data_buffer = socket_data_buffer,
        .config_buffer = recv_config_buffer,
    };
    return {sender_socket, receiver_socket};
}

}  // namespace tt::tt_metal::distributed
