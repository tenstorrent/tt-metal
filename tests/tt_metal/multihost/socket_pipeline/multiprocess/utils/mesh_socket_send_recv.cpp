// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_socket_send_recv.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/cluster.hpp>

using namespace tt::constants;

namespace tt::tt_metal {

namespace {

// Extract program creation logic from send_async_op_program_factory.cpp
tt::tt_metal::Program create_send_async_program(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    distributed::MeshDevice* mesh_device,
    uint32_t latency_measurement_address,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    TT_FATAL(
        tt::tt_metal::GetClusterType() == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY,
        "Socket pipeline send/recv only supports BLACKHOLE_GALAXY cluster");

    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;
    TT_FATAL(socket_connection_config.size() == 1, "Socket send/recv expects exactly one connection");

    const auto& connection = socket_connection_config[0];
    CoreCoord sender_core_coord = connection.sender_core.core_coord;
    CoreCoord receiver_core_coord = connection.receiver_core.core_coord;
    tt::tt_fabric::FabricNodeId sender_fabric_node_id =
        mesh_device->get_fabric_node_id(connection.sender_core.device_coord);
    tt::tt_fabric::FabricNodeId receiver_fabric_node_id = mesh_socket.get_fabric_node_id(
        tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord);

    const auto& recv_conn = recv_socket.get_config().socket_connection_config[0];
    tt::tt_fabric::FabricNodeId upstream_fabric_node_id = recv_socket.get_fabric_node_id(
        tt::tt_metal::distributed::SocketEndpoint::SENDER, recv_conn.sender_core.device_coord);

    tt::tt_metal::Program program{};

    // Use mesh_device allocator - in practice all devices in a mesh share the same allocator configuration
    auto max_alignment = std::max(
        mesh_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_buffer->alignment());
    auto input_page_size = input_buffer->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);
    TT_FATAL(link_indices.size() > 1, "Single core multi link version of SendAsync only supports multiple links");
    uint32_t num_links = 2;

    uint32_t fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    uint32_t num_whole_packets = input_page_size / fabric_max_payload_size;
    uint32_t partial_packet_size = input_page_size % fabric_max_payload_size;
    uint32_t num_whole_packets_link_0 = 0;
    uint32_t num_whole_packets_link_1 = 0;
    if (num_whole_packets > 0U) {
        // Distribute whole packets across links, biasing link 0 by one whole packet when a partial packet exists.
        num_whole_packets_link_0 = (num_whole_packets / num_links) + static_cast<uint32_t>(partial_packet_size > 0);
        if (num_whole_packets_link_0 > num_whole_packets) {
            num_whole_packets_link_0 = num_whole_packets;
        }
        num_whole_packets_link_1 = num_whole_packets - num_whole_packets_link_0;
    }

    uint32_t socket_block_size = socket_aligned_page_size;
    uint32_t socket_fifo_size_in_pages =
        mesh_socket.get_config().socket_mem_config.fifo_size / socket_aligned_page_size;
    // Notify upstream every half buffer to increase fabric utilization (compile-time ack granularity).
    uint32_t notify_sender_every_n_iterations = socket_fifo_size_in_pages / 2;

    uint32_t cb_num_pages = socket_fifo_size_in_pages;
    uint32_t cb_page_size = socket_block_size;

    tt::DataFormat df = input_data_format;

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_src0_config);

    // Writer kernel always assumes an upstream connection + a third packet header (recv_socket is required).
    uint32_t packet_header_cb_num_pages = num_links + 1;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_packet_header_config);

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_buffer);
    auto compile_time_args = input_accessor_args.get_compile_time_args();

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                // cb0_id
        packet_header_cb_index,       // fabric_packet_header_cb_id
        socket_block_size,            // socket_block_size
        partial_packet_size,          // partial_packet_size
        fabric_max_payload_size,      // whole_packet_size (fabric_max_payload_size)
        num_whole_packets_link_0,     // num_whole_packets_link_0
        num_whole_packets_link_1,     // num_whole_packets_link_1
        input_page_size,              // input_page_size
        latency_measurement_address,  // credit_address (reused for credit sync and latency measurements)
        num_iterations,               // num_iterations
        static_cast<uint32_t>(enable_correctness_check),  // enable_correctness_check
        notify_sender_every_n_iterations,  // notify_sender_every_n_iterations (ack granularity, e.g. fifo_pages/2)
    };
    writer_compile_args.insert(writer_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/multihost/socket_pipeline/multiprocess/utils/kernels/sender_writer.cpp",
        sender_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // Use mesh_device allocator - in practice all devices in a mesh share the same allocator configuration
    uint32_t bank_id = mesh_device->allocator()->get_bank_ids_from_logical_core(
        mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];

    std::vector<uint32_t> writer_rt_args = {
        input_buffer->address(),
        mesh_socket.get_config_buffer()->address(),
        recv_socket.get_config_buffer()->address(),
        bank_id};

    for (uint32_t i = 0; i < num_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            link_indices[i],
            program,
            sender_core_coord,
            writer_rt_args);
    }

    auto bwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, upstream_fabric_node_id);
    tt::tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id,
        upstream_fabric_node_id,
        bwd_link_indices[0],
        program,
        sender_core_coord,
        writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord, writer_rt_args);

    return program;
}

}  // namespace

void send_async(
    distributed::MeshDevice* mesh_device,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    const distributed::MeshSocket& mesh_socket,
    const distributed::MeshSocket& recv_socket,
    uint32_t latency_measurement_address,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    const auto& socket_connections = mesh_socket.get_config().socket_connection_config;
    TT_FATAL(socket_connections.size() == 1, "Socket send/recv expects exactly one connection");

    const auto& device_coord = socket_connections[0].sender_core.device_coord;
    TT_FATAL(
        mesh_device->get_device(device_coord) != nullptr,
        "Sender device for socket connection is not local to this mesh device");

    distributed::MeshWorkload workload;
    auto program = create_send_async_program(
        mesh_socket,
        recv_socket,
        input_buffer,
        input_data_format,
        mesh_device,
        latency_measurement_address,
        num_iterations,
        enable_correctness_check);
    workload.add_program(distributed::MeshCoordinateRange(device_coord, device_coord), std::move(program));

    // Enqueue the workload
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

}  // namespace tt::tt_metal
