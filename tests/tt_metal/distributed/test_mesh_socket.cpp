// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <random>
#include "gmock/gmock.h"
#include <tt-metalium/fabric.hpp>
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
TEST_F(MeshSocketTest, MultiConnectionSingleDeviceConfig) {
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

TEST_F(MeshSocketTest, MultiConnectionMultiDeviceTest) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));
    std::unordered_map<MeshCoordinate, chip_id_t> sender_device_coord_to_id;
    std::unordered_map<MeshCoordinate, chip_id_t> receiver_device_coord_to_id;

    for (const auto& coord : MeshCoordinateRange(md0->shape())) {
        sender_device_coord_to_id[coord] = md0->get_device(coord)->id();
    }

    for (const auto& coord : MeshCoordinateRange(md1->shape())) {
        receiver_device_coord_to_id[coord] = md1->get_device(coord)->id();
    }
    std::size_t socket_fifo_size = 1024;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const auto& worker_grid = md0->compute_with_storage_grid_size();

    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;
    std::vector<MeshCoordinate> sender_device_coords;
    std::vector<MeshCoordinate> recv_device_coords;
    uint32_t core_idx = 0;
    for (std::size_t x = 0; x < worker_grid.x; x++) {
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x, y));
            sender_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            recv_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            core_idx++;
        }
    }

    // Shuffle core coordinates to randomize the connections
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(sender_logical_coords.begin(), sender_logical_coords.end(), generator);
    std::shuffle(recv_logical_coords.begin(), recv_logical_coords.end(), generator);
    std::shuffle(sender_device_coords.begin(), sender_device_coords.end(), generator);
    std::shuffle(recv_device_coords.begin(), recv_device_coords.end(), generator);

    std::vector<socket_connection_t> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        std::cout << "Create Connection: "
                  << "Sender: (" << sender_device_coords[coord_idx] << ", " << sender_logical_coords[coord_idx].str()
                  << ") "
                  << "Receiver: (" << recv_device_coords[coord_idx] << ", " << recv_logical_coords[coord_idx].str()
                  << ")" << std::endl;
        socket_connection_t socket_connection = {
            .sender_core = {sender_device_coords[coord_idx], sender_logical_coords[coord_idx]},
            .receiver_core = {recv_device_coords[coord_idx], recv_logical_coords[coord_idx]}};
        socket_connections.push_back(socket_connection);
    }

    socket_config_t socket_config_l1 = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {
                .socket_type = BufferType::L1,
                .fifo_size = socket_fifo_size,
            },
    };
    socket_config_t socket_config_dram = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {
                .socket_type = BufferType::DRAM,
                .fifo_size = socket_fifo_size,
            },
    };

    auto [send_socket_l1, recv_socket_l1] = create_sockets(md0, md1, socket_config_l1);
    auto [send_socket_dram, recv_socket_dram] = create_sockets(md0, md1, socket_config_dram);

    const auto& sender_core_to_core_id =
        send_socket_l1.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    const auto& recv_core_to_core_id =
        recv_socket_l1.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    std::unordered_map<MeshCoordinate, std::vector<sender_socket_md>> sender_configs_per_dev_coord;
    std::unordered_map<MeshCoordinate, std::vector<receiver_socket_md>> recv_configs_per_dev_coord;

    for (const auto& device_coord : MeshCoordinateRange(md0->shape())) {
        std::vector<sender_socket_md> sender_configs;
        std::vector<receiver_socket_md> recv_configs;

        ReadShard(md0->mesh_command_queue(), sender_configs, send_socket_l1.config_buffer, device_coord);
        ReadShard(md1->mesh_command_queue(), recv_configs, recv_socket_l1.config_buffer, device_coord);

        sender_configs_per_dev_coord[device_coord] = std::move(sender_configs);
        recv_configs_per_dev_coord[device_coord] = std::move(recv_configs);
    }

    for (const auto& connection : socket_connections) {
        const auto& sender_core = connection.sender_core;
        const auto& recv_core = connection.receiver_core;
        const auto& sender_device_coord = sender_core.first;
        const auto& recv_device_coord = recv_core.first;
        const auto& sender_core_coord = sender_core.second;
        const auto& recv_core_coord = recv_core.second;

        auto sender_idx = sender_core_to_core_id.at(sender_core_coord);
        auto recv_idx = recv_core_to_core_id.at(recv_core_coord);

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_core_coord);
        auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_core_coord);
        auto sender_device_id = sender_device_coord_to_id[sender_device_coord];
        auto receiver_device_id = receiver_device_coord_to_id[recv_device_coord];

        const auto& sender_config = sender_configs_per_dev_coord[sender_device_coord][sender_idx];
        const auto& recv_config = recv_configs_per_dev_coord[recv_device_coord][recv_idx];

        // Validate Sender Configs
        EXPECT_EQ(sender_config.bytes_acked, 0);
        EXPECT_EQ(sender_config.write_ptr, send_socket_l1.data_buffer->address());
        EXPECT_EQ(sender_config.bytes_sent, 0);
        EXPECT_EQ(sender_config.downstream_mesh_id, 0);
        EXPECT_EQ(sender_config.downstream_chip_id, receiver_device_id);
        EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
        EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket_l1.config_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket_l1.data_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
        EXPECT_EQ(sender_config.is_sender, 1);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

        // Validate Recv Configs
        EXPECT_EQ(recv_config.bytes_sent, 0);
        EXPECT_EQ(recv_config.bytes_acked, 0);
        EXPECT_EQ(recv_config.read_ptr, recv_socket_l1.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_addr, recv_socket_l1.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
        EXPECT_EQ(recv_config.upstream_mesh_id, 0);
        EXPECT_EQ(recv_config.upstream_chip_id, sender_device_id);
        EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
        EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket_l1.config_buffer->address());
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
    }
}

void test_single_connection_single_device_socket(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);

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
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
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
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto recv_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(send_recv_program), devices);

    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    std::vector<uint32_t> recv_data_readback;
    ReadShard(md0->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocket) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    // No wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 1024);
    // Even wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 2048);
    // Uneven wrap
    test_single_connection_single_device_socket(md0, 4096, 1088, 9792);
}

void test_single_connection_single_device_socket_with_workers(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto worker_logical_coord = CoreCoord(0, 2);
    auto output_logical_coord = CoreCoord(0, 3);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
    auto worker_virtual_coord = md0->worker_core_from_logical_core(worker_logical_coord);
    auto output_virtual_coord = md0->worker_core_from_logical_core(output_logical_coord);

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
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    auto output_shard_params =
        ShardSpecBuffer(CoreRangeSet(output_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = output_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto output_buffer = MeshBuffer::create(buffer_config, output_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
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
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    CoreRangeSet recv_worker_crs =
        CoreRangeSet(std::array{CoreRange(recv_logical_coord), CoreRange(worker_logical_coord)}).merge_ranges();

    // Create CB on both receiver and worker so that receiver knows the address
    auto config_cb_index = tt::CBIndex::c_0;
    auto config_cb_config =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config_cb_index, sizeof(receiver_socket_md));
    auto config_cb = CreateCircularBuffer(send_recv_program, recv_worker_crs, config_cb_config);

    auto data_cb_index = tt::CBIndex::c_1;
    auto data_cb_config = CircularBufferConfig(2 * page_size, {{data_cb_index, tt::DataFormat::UInt32}})
                              .set_page_size(data_cb_index, page_size);
    // No need to create on recv core, but better dispatch to do so
    auto data_cb = CreateCircularBuffer(send_recv_program, recv_worker_crs, data_cb_config);

    auto config_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);
    auto credits_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);

    auto recv_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_final_ack.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                static_cast<uint32_t>(config_cb_index),
                static_cast<uint32_t>(config_sem),
                static_cast<uint32_t>(credits_sem),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(worker_virtual_coord.x),
                static_cast<uint32_t>(worker_virtual_coord.y),
                static_cast<uint32_t>(worker_virtual_coord.x),
                static_cast<uint32_t>(worker_virtual_coord.y),
                static_cast<uint32_t>(1)}});

    auto worker_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_final_ack.cpp",
        worker_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(config_cb_index),
                static_cast<uint32_t>(config_sem),
                static_cast<uint32_t>(credits_sem),
                static_cast<uint32_t>(data_cb_index),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(recv_virtual_coord.x),
                static_cast<uint32_t>(recv_virtual_coord.y),
                static_cast<uint32_t>(recv_virtual_coord.x),
                static_cast<uint32_t>(recv_virtual_coord.y),
                static_cast<uint32_t>(output_virtual_coord.x),
                static_cast<uint32_t>(output_virtual_coord.y),
                static_cast<uint32_t>(output_buffer->address())}});

    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(send_recv_program), devices);

    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    std::vector<uint32_t> recv_data_readback;
    ReadShard(md0->mesh_command_queue(), recv_data_readback, output_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithWorkers) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    // No wrap
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 1024);
}

void test_single_connection_multi_device_socket(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md1,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    // Used to setup fabric connections
    const uint32_t sender_physical_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t recv_physical_device_id = md1->get_device(MeshCoordinate(0, 0))->id();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // Create Socket between Sender and Receiver
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
    auto [send_socket, recv_socket] = create_sockets(md0, md1, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());
    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, md1.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    const auto reserved_packet_header_CB_index = tt::CB::c_in0;

    tt::tt_metal::CircularBufferConfig sender_cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            2 * packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);

    auto sender_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket.config_buffer->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto sender_packet_header_CB_handle =
        CreateCircularBuffer(sender_program, sender_logical_coord, sender_cb_reserved_packet_header_config);

    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_physical_device_id, recv_physical_device_id, 0, sender_program, {sender_logical_coord}, sender_rtas);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    tt::tt_metal::CircularBufferConfig recv_cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(tt::CB::c_in0, packet_header_size_bytes);

    auto recv_program = CreateProgram();
    auto recv_kernel = CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto recv_packet_header_CB_handle =
        CreateCircularBuffer(recv_program, recv_logical_coord, recv_cb_packet_header_config);

    std::vector<uint32_t> recv_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        recv_physical_device_id, sender_physical_device_id, 0, recv_program, {recv_logical_coord}, recv_rtas);
    tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_logical_coord, recv_rtas);

    auto sender_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());
    AddProgramToMeshWorkload(sender_mesh_workload, std::move(sender_program), devices);

    auto recv_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_recv(md1->shape());
    AddProgramToMeshWorkload(recv_mesh_workload, std::move(recv_program), devices_recv);

    EnqueueMeshWorkload(md0->mesh_command_queue(), sender_mesh_workload, false);
    EnqueueMeshWorkload(md1->mesh_command_queue(), recv_mesh_workload, false);
    std::vector<uint32_t> recv_data_readback;
    ReadShard(md1->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

void test_single_connection_multi_device_socket_with_workers(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md1,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    // Used to setup fabric connections
    const uint32_t sender_physical_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t recv_physical_device_id = md1->get_device(MeshCoordinate(0, 0))->id();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);
    auto worker_logical_coord = CoreCoord(0, 2);
    auto output_logical_coord = CoreCoord(0, 3);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_logical_coord);
    auto worker_virtual_coord = md1->worker_core_from_logical_core(worker_logical_coord);
    auto output_virtual_coord = md1->worker_core_from_logical_core(output_logical_coord);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // Create Socket between Sender and Receiver
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
    auto [send_socket, recv_socket] = create_sockets(md0, md1, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    auto output_shard_params =
        ShardSpecBuffer(CoreRangeSet(output_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = output_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto output_buffer = MeshBuffer::create(buffer_config, output_device_local_config, md1.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    const auto reserved_packet_header_CB_index = tt::CB::c_in0;

    tt::tt_metal::CircularBufferConfig sender_cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            2 * packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);

    auto sender_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket.config_buffer->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto sender_packet_header_CB_handle =
        CreateCircularBuffer(sender_program, sender_logical_coord, sender_cb_reserved_packet_header_config);

    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_physical_device_id, recv_physical_device_id, 0, sender_program, {sender_logical_coord}, sender_rtas);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    auto recv_program = CreateProgram();

    CoreRangeSet recv_worker_crs =
        CoreRangeSet(std::array{CoreRange(recv_logical_coord), CoreRange(worker_logical_coord)}).merge_ranges();

    tt::tt_metal::CircularBufferConfig recv_cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);

    auto recv_packet_header_CB_handle =
        CreateCircularBuffer(recv_program, recv_logical_coord, recv_cb_packet_header_config);

    // Create CB on both receiver and worker so that receiver knows the address
    auto config_cb_index = tt::CBIndex::c_1;
    auto config_cb_config =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config_cb_index, sizeof(receiver_socket_md));
    auto config_cb = CreateCircularBuffer(recv_program, recv_worker_crs, config_cb_config);

    auto data_cb_index = tt::CBIndex::c_2;
    auto data_cb_config = CircularBufferConfig(2 * page_size, {{data_cb_index, tt::DataFormat::UInt32}})
                              .set_page_size(data_cb_index, page_size);
    // No need to create on recv core, but better dispatch to do so
    auto data_cb = CreateCircularBuffer(recv_program, recv_worker_crs, data_cb_config);

    auto config_sem = CreateSemaphore(recv_program, recv_worker_crs, 0);
    auto credits_sem = CreateSemaphore(recv_program, recv_worker_crs, 0);

    auto recv_kernel = CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_final_ack.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                static_cast<uint32_t>(config_cb_index),
                static_cast<uint32_t>(config_sem),
                static_cast<uint32_t>(credits_sem),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(worker_virtual_coord.x),
                static_cast<uint32_t>(worker_virtual_coord.y),
                static_cast<uint32_t>(worker_virtual_coord.x),
                static_cast<uint32_t>(worker_virtual_coord.y),
                static_cast<uint32_t>(1)}});

    auto worker_kernel = CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_final_ack.cpp",
        worker_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(config_cb_index),
                static_cast<uint32_t>(config_sem),
                static_cast<uint32_t>(credits_sem),
                static_cast<uint32_t>(data_cb_index),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(recv_virtual_coord.x),
                static_cast<uint32_t>(recv_virtual_coord.y),
                static_cast<uint32_t>(recv_virtual_coord.x),
                static_cast<uint32_t>(recv_virtual_coord.y),
                static_cast<uint32_t>(output_virtual_coord.x),
                static_cast<uint32_t>(output_virtual_coord.y),
                static_cast<uint32_t>(output_buffer->address())}});

    std::vector<uint32_t> recv_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        recv_physical_device_id, sender_physical_device_id, 0, recv_program, {recv_logical_coord}, recv_rtas);
    tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_logical_coord, recv_rtas);

    auto sender_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());
    AddProgramToMeshWorkload(sender_mesh_workload, std::move(sender_program), devices);

    auto recv_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_recv(md1->shape());
    AddProgramToMeshWorkload(recv_mesh_workload, std::move(recv_program), devices_recv);

    EnqueueMeshWorkload(md0->mesh_command_queue(), sender_mesh_workload, false);
    EnqueueMeshWorkload(md1->mesh_command_queue(), recv_mesh_workload, false);
    std::vector<uint32_t> recv_data_readback;
    ReadShard(md1->mesh_command_queue(), recv_data_readback, output_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

TEST_F(MeshSocketTest, SingleConnectionMultiDeviceSocketWithWorkers) {
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(1, 0));
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    test_single_connection_multi_device_socket_with_workers(md0, md1, 1024, 64, 1024);
}

TEST_F(MeshSocketTest, SingleConnectionMultiDeviceSocket) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(1, 0));
    test_single_connection_multi_device_socket(md0, md1, 1024, 64, 1024);
    test_single_connection_multi_device_socket(md0, md1, 1024, 64, 2048);
    test_single_connection_multi_device_socket(md0, md1, 4096, 1088, 9792);
}

std::shared_ptr<Program> create_sender_program(
    const mesh_socket_t& sender_socket,
    std::shared_ptr<MeshBuffer> sender_data_buffer,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& sender_logical_coord,
    chip_id_t sender_physical_device_id,
    chip_id_t recv_physical_device_id) {
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    const auto reserved_packet_header_CB_index = tt::CB::c_in0;
    auto sender_program = std::make_shared<Program>();
    auto sender_kernel = CreateKernel(
        *sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(sender_socket.config_buffer->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    tt::tt_metal::CircularBufferConfig sender_cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            2 * packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto sender_packet_header_CB_handle =
        CreateCircularBuffer(*sender_program, sender_logical_coord, sender_cb_reserved_packet_header_config);
    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_physical_device_id, recv_physical_device_id, 0, *sender_program, {sender_logical_coord}, sender_rtas);
    tt_metal::SetRuntimeArgs(*sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    return sender_program;
}

std::shared_ptr<Program> create_recv_program(
    const mesh_socket_t& recv_socket_0,
    const mesh_socket_t& recv_socket_1,
    std::shared_ptr<MeshBuffer> recv_data_buffer,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& recv_logical_coord_0,
    const CoreCoord& recv_logical_coord_1,
    const CoreCoord& reduce_logical_coord,
    chip_id_t sender0_physical_device_id,
    chip_id_t sender1_physical_device_id,
    chip_id_t recv_physical_device_id) {
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);

    auto reserved_packet_header_CB_index = tt::CB::c_in0;
    auto config0_cb_index = tt::CBIndex::c_1;
    auto config1_cb_index = tt::CBIndex::c_2;
    auto in0_cb_index = tt::CBIndex::c_3;
    auto in1_cb_index = tt::CBIndex::c_4;

    auto recv_virtual_coord_0 = recv_data_buffer->device()->worker_core_from_logical_core(recv_logical_coord_0);
    auto recv_virtual_coord_1 = recv_data_buffer->device()->worker_core_from_logical_core(recv_logical_coord_1);
    auto reduce_virtual_core = recv_data_buffer->device()->worker_core_from_logical_core(reduce_logical_coord);

    auto recv_program = std::make_shared<Program>();

    auto recv_cb_packet_header_config =
        CircularBufferConfig(packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(tt::CB::c_in0, packet_header_size_bytes);
    auto config_cb_config0 =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config0_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config0_cb_index, sizeof(receiver_socket_md));
    auto config_cb_config1 =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config1_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config1_cb_index, sizeof(receiver_socket_md));
    auto in0_cb_config = CircularBufferConfig(2 * page_size, {{in0_cb_index, tt::DataFormat::UInt32}})
                             .set_page_size(in0_cb_index, page_size);
    auto in1_cb_config = CircularBufferConfig(2 * page_size, {{in1_cb_index, tt::DataFormat::UInt32}})
                             .set_page_size(in1_cb_index, page_size);
    CoreRangeSet recv_crs =
        CoreRangeSet(std::array{CoreRange(recv_logical_coord_0), CoreRange(recv_logical_coord_1)}).merge_ranges();
    CoreRangeSet recv_worker_crs =
        CoreRangeSet(
            std::array{
                CoreRange(recv_logical_coord_0), CoreRange(recv_logical_coord_1), CoreRange(reduce_logical_coord)})
            .merge_ranges();
    // Fabric header CB
    auto recv_packet_header_CB_handle = CreateCircularBuffer(*recv_program, recv_crs, recv_cb_packet_header_config);
    // Socket Config CB
    auto config_cb_handle0 = CreateCircularBuffer(*recv_program, recv_worker_crs, config_cb_config0);
    auto config_cb_handle1 = CreateCircularBuffer(*recv_program, recv_worker_crs, config_cb_config1);
    // Data CBs
    auto in0_cb_handle = CreateCircularBuffer(*recv_program, reduce_logical_coord, in0_cb_config);
    auto in1_cb_handle = CreateCircularBuffer(*recv_program, reduce_logical_coord, in1_cb_config);

    auto config0_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto credits0_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto config1_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto credits1_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);

    auto recv_kernel_0 = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord_0,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket_0.config_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(config0_cb_index),
                static_cast<uint32_t>(config0_sem),
                static_cast<uint32_t>(credits0_sem),
                static_cast<uint32_t>(reduce_virtual_core.x),
                static_cast<uint32_t>(reduce_virtual_core.y)}});

    auto recv_kernel_1 = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord_1,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket_1.config_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(config1_cb_index),
                static_cast<uint32_t>(config1_sem),
                static_cast<uint32_t>(credits1_sem),
                static_cast<uint32_t>(reduce_virtual_core.x),
                static_cast<uint32_t>(reduce_virtual_core.y)}});

    auto reduce_kernel = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/reduce_worker.cpp",
        reduce_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(config0_cb_index),
                static_cast<uint32_t>(config0_sem),
                static_cast<uint32_t>(config1_cb_index),
                static_cast<uint32_t>(config1_sem),
                static_cast<uint32_t>(credits0_sem),
                static_cast<uint32_t>(credits1_sem),
                static_cast<uint32_t>(in0_cb_index),
                static_cast<uint32_t>(in1_cb_index),
                static_cast<uint32_t>(recv_virtual_coord_0.x),
                static_cast<uint32_t>(recv_virtual_coord_0.y),
                static_cast<uint32_t>(recv_virtual_coord_1.x),
                static_cast<uint32_t>(recv_virtual_coord_1.y),
                static_cast<uint32_t>(reduce_virtual_core.x),  // Output buf core
                static_cast<uint32_t>(reduce_virtual_core.y),  // Output buf core
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    std::vector<uint32_t> recv_rtas_0;
    std::vector<uint32_t> recv_rtas_1;

    tt_fabric::append_fabric_connection_rt_args(
        recv_physical_device_id, sender0_physical_device_id, 0, *recv_program, {recv_logical_coord_0}, recv_rtas_0);
    tt_fabric::append_fabric_connection_rt_args(
        recv_physical_device_id, sender1_physical_device_id, 0, *recv_program, {recv_logical_coord_1}, recv_rtas_1);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel_0, recv_logical_coord_0, recv_rtas_0);
    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel_1, recv_logical_coord_1, recv_rtas_1);

    return recv_program;
}

void test_multi_sender_single_recv(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> sender_0,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> sender_1,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> receiver,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    // Used to setup fabric connections
    const uint32_t sender_0_physical_device_id = sender_0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t sender_1_physical_device_id = sender_1->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t receiver_physical_device_id = receiver->get_device(MeshCoordinate(0, 0))->id();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv0_logical_coord = CoreCoord(0, 0);
    auto recv1_logical_coord = CoreCoord(0, 1);
    auto reduce_core = CoreCoord(0, 2);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    socket_connection_t socket_connection_0 = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv0_logical_coord},
    };
    socket_connection_t socket_connection_1 = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv1_logical_coord},
    };
    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config_0 = {
        .socket_connection_config = {socket_connection_0},
        .socket_mem_config = socket_mem_config,
    };
    socket_config_t socket_config_1 = {
        .socket_connection_config = {socket_connection_1},
        .socket_mem_config = socket_mem_config,
    };

    auto [send_socket_0, recv_socket_0] = create_sockets(sender_0, receiver, socket_config_0);
    auto [send_socket_1, recv_socket_1] = create_sockets(sender_1, receiver, socket_config_1);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    // CoreRangeSet recv_crs =
    //     CoreRangeSet(std::array{CoreRange(recv0_logical_coord), CoreRange(recv1_logical_coord)}).merge_ranges();
    auto reduce_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(reduce_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig reduce_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = reduce_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig sender_buffer_config{.size = data_size};
    const ReplicatedBufferConfig reduce_buffer_config{.size = data_size};
    auto sender_data_buffer_0 = MeshBuffer::create(sender_buffer_config, sender_device_local_config, sender_0.get());
    auto sender_data_buffer_1 = MeshBuffer::create(sender_buffer_config, sender_device_local_config, sender_1.get());
    auto reduce_data_buffer = MeshBuffer::create(reduce_buffer_config, reduce_device_local_config, receiver.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    // Write data to both senders
    WriteShard(sender_0->mesh_command_queue(), sender_data_buffer_0, src_vec, MeshCoordinate(0, 0));
    WriteShard(sender_1->mesh_command_queue(), sender_data_buffer_1, src_vec, MeshCoordinate(0, 0));

    auto sender_program_0 = create_sender_program(
        send_socket_0,
        sender_data_buffer_0,
        page_size,
        data_size,
        sender_logical_coord,
        sender_0_physical_device_id,
        receiver_physical_device_id);
    auto sender_program_1 = create_sender_program(
        send_socket_1,
        sender_data_buffer_1,
        page_size,
        data_size,
        sender_logical_coord,
        sender_1_physical_device_id,
        receiver_physical_device_id);
    auto recv_program = create_recv_program(
        recv_socket_0,
        recv_socket_1,
        reduce_data_buffer,
        page_size,
        data_size,
        recv0_logical_coord,
        recv1_logical_coord,
        reduce_core,
        sender_0_physical_device_id,
        sender_1_physical_device_id,
        receiver_physical_device_id);

    auto sender_0_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_0(sender_0->shape());
    AddProgramToMeshWorkload(sender_0_mesh_workload, std::move(*sender_program_0), devices_0);

    auto sender_1_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_1(sender_1->shape());
    AddProgramToMeshWorkload(sender_1_mesh_workload, std::move(*sender_program_1), devices_1);

    auto recv_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_recv(receiver->shape());
    AddProgramToMeshWorkload(recv_mesh_workload, std::move(*recv_program), devices_recv);

    EnqueueMeshWorkload(sender_0->mesh_command_queue(), sender_0_mesh_workload, false);
    EnqueueMeshWorkload(sender_1->mesh_command_queue(), sender_1_mesh_workload, false);
    EnqueueMeshWorkload(receiver->mesh_command_queue(), recv_mesh_workload, false);

    std::vector<uint32_t> reduce_data_readback;
    ReadShard(receiver->mesh_command_queue(), reduce_data_readback, reduce_data_buffer, MeshCoordinate(0, 0));
    for (size_t i = 0; i < src_vec.size(); ++i) {
        EXPECT_EQ(2 * src_vec[i], reduce_data_readback[i]);
    }
}

TEST_F(MeshSocketTest, MultiSenderSingleRecv) {
    auto sender_0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(1, 0));
    auto sender_1 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1));
    auto receiver = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));

    log_info(LogTest, "Sender 0 ID: {}", sender_0->get_device(MeshCoordinate(0, 0))->id());
    log_info(LogTest, "Sender 1 ID: {}", sender_1->get_device(MeshCoordinate(0, 0))->id());
    log_info(LogTest, "Receiver ID: {}", receiver->get_device(MeshCoordinate(0, 0))->id());

    test_multi_sender_single_recv(sender_0, sender_1, receiver, 1024, 64, 1024);
    test_multi_sender_single_recv(sender_0, sender_1, receiver, 1024, 64, 2048);
    test_multi_sender_single_recv(sender_0, sender_1, receiver, 4096, 1088, 9792);
}

}  // namespace tt::tt_metal::distributed
