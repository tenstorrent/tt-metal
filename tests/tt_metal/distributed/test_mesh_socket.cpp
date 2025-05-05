// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "gmock/gmock.h"

#include "tt_metal/hw/inc/socket.h"

namespace tt::tt_metal::distributed {

using MeshSocketTest = T3000MeshDeviceFixture;

// Sanity test with a single connection
TEST_F(MeshSocketTest, SingleConnectionSingleDeviceConfig) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
    std::size_t socket_fifo_size = 1024;

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    socket_connection_t socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord},
    };

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    std::vector<sender_socket_md> sender_config_readback;
    std::vector<receiver_socket_md> recv_config_readback;

    ReadShard(md0->mesh_command_queue(), sender_config_readback, send_socket.config_buffer, MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_config_readback, recv_socket.config_buffer, MeshCoordinate(0, 0));

    EXPECT_EQ(sender_config_readback.size(), 1);
    EXPECT_EQ(recv_config_readback.size(), 1);

    const auto& sender_config = sender_config_readback[0];
    const auto& recv_config = recv_config_readback[0];

    // Validate Sender Config
    EXPECT_EQ(sender_config.bytes_acked, 0);
    EXPECT_EQ(sender_config.write_ptr, send_socket.data_buffer->address());
    EXPECT_EQ(sender_config.bytes_sent, 0);
    EXPECT_EQ(sender_config.downstream_mesh_id, 0);
    EXPECT_EQ(sender_config.downstream_chip_id, current_device_id);
    EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
    EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
    EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket.config_buffer->address());
    EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket.data_buffer->address());
    EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
    EXPECT_EQ(sender_config.is_sender, 1);
    EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

    // Validate Receiver Config
    EXPECT_EQ(recv_config.bytes_sent, 0);
    EXPECT_EQ(recv_config.bytes_acked, 0);
    EXPECT_EQ(recv_config.read_ptr, recv_socket.data_buffer->address());
    EXPECT_EQ(recv_config.fifo_addr, recv_socket.data_buffer->address());
    EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
    EXPECT_EQ(recv_config.upstream_mesh_id, 0);
    EXPECT_EQ(recv_config.upstream_chip_id, current_device_id);
    EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
    EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
    EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket.config_buffer->address());
    EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
}

// Test multiple connections
TEST_F(MeshSocketTest, MultiConnectionSingleDeviceTest) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    std::size_t socket_fifo_size = 1024;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const auto& worker_grid = md0->compute_with_storage_grid_size();
    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;

    for (std::size_t x = 0; x < worker_grid.x; x += 2) {
        if (x + 1 >= worker_grid.x) {
            continue;
        }
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x + 1, y));
        }
    }

    std::vector<socket_connection_t> socket_connections;

    for (std::size_t core_idx = 0; core_idx < sender_logical_coords.size(); core_idx++) {
        socket_connections.push_back(socket_connection_t{
            .sender_core = {MeshCoordinate(0, 0), sender_logical_coords[core_idx]},
            .receiver_core = {MeshCoordinate(0, 0), recv_logical_coords[core_idx]}});
    }

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = socket_connections,
        .socket_mem_config = socket_mem_config,
    };

    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    std::vector<sender_socket_md> sender_configs;
    std::vector<receiver_socket_md> recv_configs;

    ReadShard(md0->mesh_command_queue(), sender_configs, send_socket.config_buffer, MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_configs, recv_socket.config_buffer, MeshCoordinate(0, 0));

    EXPECT_EQ(sender_configs.size(), sender_logical_coords.size());
    EXPECT_EQ(recv_configs.size(), recv_logical_coords.size());

    const auto& sender_core_to_core_id =
        send_socket.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    const auto& recv_core_to_core_id =
        recv_socket.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    for (const auto& connection : socket_connections) {
        const auto& sender = connection.sender_core;
        const auto& recv = connection.receiver_core;
        auto sender_idx = sender_core_to_core_id.at(sender.second);
        auto recv_idx = recv_core_to_core_id.at(recv.second);

        const auto& sender_config = sender_configs[sender_idx];
        const auto& recv_config = recv_configs[recv_idx];

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender.second);
        auto recv_virtual_coord = md0->worker_core_from_logical_core(recv.second);

        // Validate Sender Configs
        EXPECT_EQ(sender_config.bytes_acked, 0);
        EXPECT_EQ(sender_config.write_ptr, send_socket.data_buffer->address());
        EXPECT_EQ(sender_config.bytes_sent, 0);
        EXPECT_EQ(sender_config.downstream_mesh_id, 0);
        EXPECT_EQ(sender_config.downstream_chip_id, current_device_id);
        EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
        EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket.config_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket.data_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
        EXPECT_EQ(sender_config.is_sender, 1);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

        // Validate Recv Configs
        EXPECT_EQ(recv_config.bytes_sent, 0);
        EXPECT_EQ(recv_config.bytes_acked, 0);
        EXPECT_EQ(recv_config.read_ptr, recv_socket.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_addr, recv_socket.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
        EXPECT_EQ(recv_config.upstream_mesh_id, 0);
        EXPECT_EQ(recv_config.upstream_chip_id, current_device_id);
        EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
        EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket.config_buffer->address());
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
    }
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocket) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
    std::size_t socket_fifo_size = 1024;

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    socket_connection_t socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord},
    };

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = socket_fifo_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = socket_fifo_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(socket_fifo_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto send_recv_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket.config_buffer->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                32,
                static_cast<uint32_t>(socket_fifo_size)}});

    auto recv_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                32,
                static_cast<uint32_t>(socket_fifo_size)}});

    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(send_recv_program), devices);

    std::cout << "D" << std::endl;
    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = socket_fifo_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_data_shard_params,
        .bottom_up = false};

    auto recv_data_buffer =
        MeshBuffer::create(buffer_config, recv_device_local_config, md0.get(), recv_socket.data_buffer->address());

    std::vector<uint32_t> recv_data_readback;

    std::cout << "C" << std::endl;
    ReadShard(md0->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    std::cout << "A" << std::endl;
    Finish(md0->mesh_command_queue());
    std::cout << "B" << std::endl;
    EXPECT_EQ(src_vec, recv_data_readback);
}

}  // namespace tt::tt_metal::distributed
