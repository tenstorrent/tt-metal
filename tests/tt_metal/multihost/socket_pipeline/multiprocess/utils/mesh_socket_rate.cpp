// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_socket_rate.hpp"

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

// ─── send_async_rate program ─────────────────────────────────────────────────

tt::tt_metal::Program create_send_async_rate_program(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    distributed::MeshDevice* mesh_device,
    uint32_t num_iterations) {
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

    tt::tt_metal::Program program{};

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
        num_whole_packets_link_0 = (num_whole_packets / num_links) + static_cast<uint32_t>(partial_packet_size > 0);
        num_whole_packets_link_0 = std::min(num_whole_packets_link_0, num_whole_packets);
        num_whole_packets_link_1 = num_whole_packets - num_whole_packets_link_0;
    }

    uint32_t socket_block_size = socket_aligned_page_size;
    uint32_t socket_fifo_size_in_pages =
        mesh_socket.get_config().socket_mem_config.fifo_size / socket_aligned_page_size;

    uint32_t cb_num_pages = socket_fifo_size_in_pages;
    uint32_t cb_page_size = socket_block_size;

    tt::DataFormat df = input_data_format;

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_src0_config);

    // Only 2 packet headers needed (no upstream connection for rate-mode sender)
    uint32_t packet_header_cb_num_pages = num_links;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_packet_header_config);

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_buffer);
    auto compile_time_args = input_accessor_args.get_compile_time_args();

    // num_iterations is NOT a compile-time arg — it is a runtime arg so warmup shares the binary
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,             // data_cb_id
        packet_header_cb_index,    // fabric_packet_header_cb_id
        socket_block_size,         // socket_block_size
        partial_packet_size,       // aligned_partial_packet_size
        fabric_max_payload_size,   // whole_packet_size
        num_whole_packets_link_0,  // num_whole_packets_link_0
        num_whole_packets_link_1,  // num_whole_packets_link_1
        input_page_size,           // input_page_size
    };
    writer_compile_args.insert(writer_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/multihost/socket_pipeline/multiprocess/utils/kernels/sender_writer_rate.cpp",
        sender_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    uint32_t bank_id = mesh_device->allocator()->get_bank_ids_from_logical_core(
        mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];

    std::vector<uint32_t> writer_rt_args = {
        input_buffer->address(), mesh_socket.get_config_buffer()->address(), bank_id};

    for (uint32_t i = 0; i < num_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            link_indices[i],
            program,
            sender_core_coord,
            writer_rt_args);
    }

    // Append num_iterations as the last runtime arg
    writer_rt_args.push_back(num_iterations);

    tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord, writer_rt_args);

    return program;
}

// ─── socket_forward_rate program ─────────────────────────────────────────────

tt::tt_metal::Program create_socket_forward_rate_program(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    std::size_t num_bytes,
    distributed::MeshDevice* mesh_device,
    uint32_t num_iterations) {
    TT_FATAL(
        tt::tt_metal::GetClusterType() == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY,
        "Socket forward only supports BLACKHOLE_GALAXY cluster");

    tt::tt_metal::Program program{};

    const auto& send_socket_conn = send_socket.get_config().socket_connection_config[0];
    const auto& recv_socket_conn = recv_socket.get_config().socket_connection_config[0];

    TT_FATAL(
        send_socket_conn.sender_core.device_coord == recv_socket_conn.receiver_core.device_coord,
        "Sender and receiver cores must be on the same device.");
    TT_FATAL(
        send_socket_conn.sender_core.core_coord == recv_socket_conn.receiver_core.core_coord,
        "Sender and receiver must be on the same core.");

    distributed::MeshCoordinate my_device_coord = send_socket_conn.sender_core.device_coord;
    distributed::MeshCoordinate upstream_device_coord = recv_socket_conn.sender_core.device_coord;
    distributed::MeshCoordinate downstream_device_coord = send_socket_conn.receiver_core.device_coord;

    CoreCoord my_core_coord = send_socket_conn.sender_core.core_coord;
    CoreCoord receiver_core_coord = send_socket_conn.receiver_core.core_coord;

    tt::tt_fabric::FabricNodeId my_fabric_node_id = mesh_device->get_fabric_node_id(my_device_coord);
    tt::tt_fabric::FabricNodeId upstream_fabric_node_id =
        recv_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::SENDER, upstream_device_coord);
    tt::tt_fabric::FabricNodeId downstream_fabric_node_id =
        send_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::RECEIVER, downstream_device_coord);

    auto max_alignment = std::max(
        mesh_device->allocator()->get_alignment(send_socket.get_config().socket_mem_config.socket_storage_type),
        mesh_device->allocator()->get_alignment(recv_socket.get_config().socket_mem_config.socket_storage_type));

    auto socket_aligned_page_size = tt::align(num_bytes, max_alignment);

    auto bwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(my_fabric_node_id, upstream_fabric_node_id);
    auto fwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(my_fabric_node_id, downstream_fabric_node_id);
    TT_FATAL(
        fwd_link_indices.size() > 1, "Single core multi link version of SocketForward only supports multiple links.");
    TT_FATAL(!bwd_link_indices.empty(), "No link indices found from downstream to upstream core in SocketForward.");
    uint32_t num_fwd_links = 2;
    uint32_t num_bwd_links = 1;

    auto fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t partial_packet_size = num_bytes % fabric_max_payload_size;
    uint32_t socket_block_size = socket_aligned_page_size;

    uint32_t num_whole_packets = num_bytes / fabric_max_payload_size;
    uint32_t num_whole_packets_link_0 = 0;
    uint32_t num_whole_packets_link_1 = 0;
    if (num_whole_packets > 0U) {
        num_whole_packets_link_0 = (num_whole_packets / num_fwd_links) + static_cast<uint32_t>(partial_packet_size > 0);
        num_whole_packets_link_0 = std::min(num_whole_packets_link_0, num_whole_packets);
        num_whole_packets_link_1 = num_whole_packets - num_whole_packets_link_0;
    }

    uint32_t packet_header_cb_num_pages = num_fwd_links + num_bwd_links;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, my_core_coord, cb_packet_header_config);

    // Notify upstream every half buffer to increase fabric utilization
    uint32_t socket_fifo_size_in_pages =
        send_socket.get_config().socket_mem_config.fifo_size / socket_aligned_page_size;
    uint32_t notify_sender_every_n_iterations = socket_fifo_size_in_pages / 2;

    // num_iterations is NOT a compile-time arg — it is a runtime arg so warmup shares the binary
    std::vector<uint32_t> compile_args = {
        packet_header_cb_index,
        socket_block_size,
        partial_packet_size,
        fabric_max_payload_size,
        num_whole_packets_link_0,
        num_whole_packets_link_1,
        notify_sender_every_n_iterations,
    };

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/multihost/socket_pipeline/multiprocess/utils/kernels/socket_forward_rate.cpp",
        my_core_coord,
        tt::tt_metal::WriterDataMovementConfig(compile_args));

    uint32_t downstream_bank_id = mesh_device->allocator()->get_bank_ids_from_logical_core(
        send_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];

    std::vector<uint32_t> rt_args = {
        send_socket.get_config_buffer()->address(), recv_socket.get_config_buffer()->address(), downstream_bank_id};

    for (uint32_t i = 0; i < num_bwd_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            my_fabric_node_id, upstream_fabric_node_id, bwd_link_indices[i], program, my_core_coord, rt_args);
    }
    for (uint32_t i = 0; i < num_fwd_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            my_fabric_node_id, downstream_fabric_node_id, fwd_link_indices[i], program, my_core_coord, rt_args);
    }

    // Append num_iterations as the last runtime arg
    rt_args.push_back(num_iterations);

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, my_core_coord, rt_args);

    return program;
}

// ─── recv_async_rate program ─────────────────────────────────────────────────

tt::tt_metal::Program create_recv_async_rate_program(
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    std::size_t num_bytes,
    distributed::MeshDevice* mesh_device,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    TT_FATAL(
        tt::tt_metal::GetClusterType() == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY,
        "Socket pipeline recv only supports BLACKHOLE_GALAXY cluster");

    tt::tt_metal::Program program{};

    const auto& recv_socket_conn = recv_socket.get_config().socket_connection_config[0];

    distributed::MeshCoordinate my_device_coord = recv_socket_conn.receiver_core.device_coord;
    distributed::MeshCoordinate upstream_device_coord = recv_socket_conn.sender_core.device_coord;

    CoreCoord my_core_coord = recv_socket_conn.receiver_core.core_coord;

    tt::tt_fabric::FabricNodeId my_fabric_node_id = mesh_device->get_fabric_node_id(my_device_coord);
    tt::tt_fabric::FabricNodeId upstream_fabric_node_id =
        recv_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::SENDER, upstream_device_coord);

    auto max_alignment =
        mesh_device->allocator()->get_alignment(recv_socket.get_config().socket_mem_config.socket_storage_type);
    auto socket_aligned_page_size = tt::align(num_bytes, max_alignment);

    auto bwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(my_fabric_node_id, upstream_fabric_node_id);
    TT_FATAL(!bwd_link_indices.empty(), "No link indices found from receiver to upstream core.");
    uint32_t num_bwd_links = 1;

    uint32_t socket_block_size = socket_aligned_page_size;

    // Notify upstream every half buffer
    uint32_t socket_fifo_size_in_pages =
        recv_socket.get_config().socket_mem_config.fifo_size / socket_aligned_page_size;
    uint32_t notify_sender_every_n_iterations = socket_fifo_size_in_pages / 2;

    // Packet header CB for upstream acks
    uint32_t packet_header_cb_num_pages = num_bwd_links;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, my_core_coord, cb_packet_header_config);

    // num_iterations is NOT a compile-time arg — it is a runtime arg so warmup shares the binary
    std::vector<uint32_t> compile_args = {
        packet_header_cb_index,
        socket_block_size,
        notify_sender_every_n_iterations,
        static_cast<uint32_t>(enable_correctness_check),
    };

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/multihost/socket_pipeline/multiprocess/utils/kernels/receiver_rate.cpp",
        my_core_coord,
        tt::tt_metal::WriterDataMovementConfig(compile_args));

    std::vector<uint32_t> rt_args = {recv_socket.get_config_buffer()->address()};

    for (uint32_t i = 0; i < num_bwd_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            my_fabric_node_id, upstream_fabric_node_id, bwd_link_indices[i], program, my_core_coord, rt_args);
    }

    // Append num_iterations as the last runtime arg
    rt_args.push_back(num_iterations);

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, my_core_coord, rt_args);

    return program;
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────────

void send_async_rate(
    distributed::MeshDevice* mesh_device,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    const distributed::MeshSocket& mesh_socket,
    uint32_t num_iterations) {
    const auto& socket_connections = mesh_socket.get_config().socket_connection_config;
    TT_FATAL(socket_connections.size() == 1, "Socket send/recv expects exactly one connection");

    const auto& device_coord = socket_connections[0].sender_core.device_coord;
    TT_FATAL(
        mesh_device->get_device(device_coord) != nullptr,
        "Sender device for socket connection is not local to this mesh device");

    distributed::MeshWorkload workload;
    auto program =
        create_send_async_rate_program(mesh_socket, input_buffer, input_data_format, mesh_device, num_iterations);
    workload.add_program(distributed::MeshCoordinateRange(device_coord, device_coord), std::move(program));

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

void socket_forward_rate(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    const distributed::MeshSocket& send_socket,
    std::size_t num_bytes,
    uint32_t num_iterations) {
    const auto& send_socket_connections = send_socket.get_config().socket_connection_config;
    const auto& recv_socket_connections = recv_socket.get_config().socket_connection_config;

    TT_FATAL(send_socket_connections.size() == 1, "SocketForward only supports one sender and one receiver core.");
    TT_FATAL(recv_socket_connections.size() == 1, "SocketForward only supports one sender and one receiver core.");

    const auto& device_coord = send_socket_connections[0].sender_core.device_coord;
    auto* target_device = mesh_device->get_device(device_coord);
    TT_FATAL(target_device != nullptr, "SocketForward device coordinate not found in mesh device");

    distributed::MeshWorkload workload;
    auto program = create_socket_forward_rate_program(send_socket, recv_socket, num_bytes, mesh_device, num_iterations);
    workload.add_program(distributed::MeshCoordinateRange(device_coord, device_coord), std::move(program));

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

void recv_async_rate(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    std::size_t num_bytes,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    const auto& recv_socket_connections = recv_socket.get_config().socket_connection_config;
    TT_FATAL(recv_socket_connections.size() == 1, "recv_async_rate only supports one connection.");

    const auto& device_coord = recv_socket_connections[0].receiver_core.device_coord;
    auto* target_device = mesh_device->get_device(device_coord);
    TT_FATAL(target_device != nullptr, "Receiver device coordinate not found in mesh device");

    distributed::MeshWorkload workload;
    auto program =
        create_recv_async_rate_program(recv_socket, num_bytes, mesh_device, num_iterations, enable_correctness_check);
    workload.add_program(distributed::MeshCoordinateRange(device_coord, device_coord), std::move(program));

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

}  // namespace tt::tt_metal
