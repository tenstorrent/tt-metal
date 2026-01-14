// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include "gmock/gmock.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <cstring>
#include <tt-metalium/tt_align.hpp>

namespace tt::tt_metal::distributed {

using MeshSocketTest = MeshDevice2x4Fixture;
using MeshSocketTest1DFabric = MeshDevice2x4Fabric1DFixture;
using MeshSocketTest2DFabric = MeshDevice2x4Fabric2DFixture;

struct SocketCoreMapping {
    CoreCoord sender_core;
    CoreCoord receiver_core;
    CoreRange worker_cores;
    CoreRangeSet data_cores;
    CoreRangeSet output_cores;
};

struct ParsedSenderPage {
    sender_socket_md md;
    std::vector<uint32_t> bytes_acked;
    std::vector<sender_downstream_encoding> encodings;
};

static ParsedSenderPage parse_sender_page(
    const uint8_t* page_base, uint32_t l1_alignment, uint32_t max_num_downstreams) {
    ParsedSenderPage parsed;
    // copy the static md
    EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(page_base) % l1_alignment);
    std::memcpy(&parsed.md, page_base, sizeof(sender_socket_md));

    const uint32_t md_size_aligned = tt::align(sizeof(sender_socket_md), l1_alignment);
    const uint32_t ack_stride = tt::align(sizeof(uint32_t), l1_alignment);
    const uint32_t ack_base = md_size_aligned;

    // copy each of the acked bytes
    parsed.bytes_acked.resize(parsed.md.num_downstreams);
    for (uint32_t i = 0; i < parsed.md.num_downstreams; ++i) {
        uint32_t v = 0;
        const auto* bytes_acked_addr = page_base + ack_base + (i * ack_stride);
        EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(bytes_acked_addr) % l1_alignment);
        std::memcpy(&v, bytes_acked_addr, sizeof(uint32_t));
        parsed.bytes_acked[i] = v;
    }

    // copy each of the encodings
    const uint32_t enc_stride = tt::align(sizeof(sender_downstream_encoding), l1_alignment);
    const uint32_t enc_base = ack_base + (max_num_downstreams * ack_stride);
    parsed.encodings.resize(parsed.md.num_downstreams);
    for (uint32_t i = 0; i < parsed.md.num_downstreams; ++i) {
        sender_downstream_encoding enc{};
        const auto* encoding_addr = page_base + enc_base + (i * enc_stride);
        EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(encoding_addr) % l1_alignment);
        std::memcpy(&enc, encoding_addr, sizeof(sender_downstream_encoding));
        parsed.encodings[i] = enc;
    }
    return parsed;
}

void verify_socket_configs(
    const ParsedSenderPage& sender_page,
    const receiver_socket_md& recv_config,
    const MeshSocket& send_socket,
    const MeshSocket& recv_socket,
    uint32_t downstream_device_id,
    uint32_t upstream_device_id,
    const CoreCoord& sender_virtual_coord,
    const CoreCoord& recv_virtual_coord,
    uint32_t socket_fifo_size) {
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // Sender md checks
    EXPECT_EQ(sender_page.md.write_ptr, recv_socket.get_data_buffer()->address());
    EXPECT_EQ(sender_page.md.bytes_sent, 0);
    EXPECT_EQ(sender_page.md.downstream_bytes_sent_addr, recv_socket.get_config_buffer()->address());
    EXPECT_EQ(sender_page.md.downstream_fifo_addr, recv_socket.get_data_buffer()->address());
    EXPECT_EQ(sender_page.md.downstream_fifo_total_size, socket_fifo_size);
    EXPECT_EQ(sender_page.md.is_sender, 1);
    EXPECT_EQ(sender_page.md.downstream_bytes_sent_addr % l1_alignment, 0);
    // Bytes acks are zero-initialized
    for (auto v : sender_page.bytes_acked) {
        EXPECT_EQ(v, 0);
    }
    // At least one downstream encoding matches the expected recv info
    bool found_match = false;
    for (const auto& enc : sender_page.encodings) {
        if (enc.downstream_chip_id == downstream_device_id && enc.downstream_noc_y == recv_virtual_coord.y &&
            enc.downstream_noc_x == recv_virtual_coord.x) {
            found_match = true;
            break;
        }
    }
    EXPECT_TRUE(found_match);

    // Receiver checks
    EXPECT_EQ(recv_config.bytes_sent, 0);
    EXPECT_EQ(recv_config.bytes_acked, 0);
    EXPECT_EQ(recv_config.read_ptr, recv_socket.get_data_buffer()->address());
    EXPECT_EQ(recv_config.fifo_addr, recv_socket.get_data_buffer()->address());
    EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
    EXPECT_EQ(recv_config.upstream_mesh_id, 0);
    EXPECT_EQ(recv_config.upstream_chip_id, upstream_device_id);
    EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
    EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
    EXPECT_EQ(
        recv_config.upstream_bytes_acked_addr,
        send_socket.get_config_buffer()->address() + tt::align(sizeof(sender_socket_md), l1_alignment));
    EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
}

void test_single_connection_single_device_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    bool use_cbs) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);

    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config({socket_connection}, socket_mem_config);
    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md0, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto send_recv_program = CreateProgram();
    CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket.get_config_buffer()->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    if (use_cbs) {
        auto data_format = tt::DataFormat::UInt32;
        auto tile_size_bytes = tile_size(data_format);
        if (page_size % tile_size_bytes != 0) {
            GTEST_SKIP() << "Page size must be a multiple of tile size";
        }
        if (page_size / tile_size_bytes > 8) {
            GTEST_SKIP() << "Page size must be less than or equal to 8 tiles";
        }
        uint32_t num_tiles_per_page = page_size / tile_size_bytes;
        // Total size does not matter here as it will get reconfigured
        auto input_cb_index = CBIndex::c_0;
        auto input_cb_config = CircularBufferConfig(page_size, {{input_cb_index, data_format}})
                                   .set_page_size(input_cb_index, tile_size_bytes);
        CreateCircularBuffer(send_recv_program, recv_logical_coord, input_cb_config);
        auto output_cb_index = CBIndex::c_1;
        auto output_cb_config = CircularBufferConfig(2 * page_size, {{output_cb_index, data_format}})
                                    .set_page_size(output_cb_index, tile_size_bytes);
        CreateCircularBuffer(send_recv_program, recv_logical_coord, output_cb_config);
        CreateKernel(
            send_recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_cb_compute.cpp",
            recv_logical_coord,
            ComputeConfig{
                .dst_full_sync_en = true,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(input_cb_index),
                    static_cast<uint32_t>(output_cb_index),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(num_tiles_per_page)}});
        CreateKernel(
            send_recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_cb_writer.cpp",
            recv_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(output_cb_index),
                    static_cast<uint32_t>(recv_data_buffer->address()),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(num_tiles_per_page)}});
    } else {
        CreateKernel(
            send_recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
            recv_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(recv_data_buffer->address()),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size)}});
    }

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    mesh_workload.add_program(devices, std::move(send_recv_program));
    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);
    std::vector<uint32_t> recv_data_readback;
    ReadShard(md0->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

void test_single_device_socket_with_workers(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    tt::stl::Span<SocketCoreMapping> socket_core_mappings,
    bool final_ack) {
    CoreRangeSet used_cores;
    uint32_t num_used_cores = 0;
    for (const auto& mapping : socket_core_mappings) {
        if (mapping.worker_cores.size() != mapping.output_cores.num_cores() ||
            mapping.worker_cores.size() != mapping.output_cores.num_cores()) {
            GTEST_SKIP() << "Worker and data/output core ranges must be the same size";
        }
        // TODO: Update this check for multi device
        used_cores = used_cores.merge(CoreRangeSet(mapping.sender_core));
        used_cores = used_cores.merge(CoreRangeSet(mapping.receiver_core));
        used_cores = used_cores.merge(CoreRangeSet(mapping.worker_cores));
        num_used_cores += mapping.worker_cores.size() + 2;
        if (used_cores.num_cores() != num_used_cores) {
            GTEST_SKIP() << "Socket core ranges must not overlap" << used_cores.num_cores() << " != " << num_used_cores;
        }
    }

    if (final_ack && socket_fifo_size < data_size) {
        GTEST_SKIP() << "Socket FIFO size must be greater than data size for final ack";
    }
    if (!final_ack && socket_fifo_size < 2 * page_size) {
        GTEST_SKIP() << "Socket FIFO size must be greater than 2 * page size for loop ack";
    }

    CoreCoord sender_logical_data_core = CoreCoord(0, 0);
    CoreCoord sender_virtual_data_core = md0->worker_core_from_logical_core(sender_logical_data_core);

    std::vector<SocketConnection> socket_connections;
    socket_connections.reserve(socket_core_mappings.size());

    CoreRangeSet sender_crs;
    CoreRangeSet recv_worker_crs;
    CoreRangeSet output_crs;
    uint32_t num_data_cores = 0;
    uint32_t num_output_cores = 0;
    {
        std::vector<CoreRange> sender_logical_cr;
        sender_logical_cr.reserve(socket_core_mappings.size());
        std::vector<CoreRange> recv_worker_logical_cr;
        recv_worker_logical_cr.reserve(socket_core_mappings.size() * 2);
        std::vector<CoreRange> output_logical_cr;

        for (const auto& mapping : socket_core_mappings) {
            SocketConnection socket_connection(
                MeshCoreCoord(MeshCoordinate(0, 0), mapping.sender_core),
                MeshCoreCoord(MeshCoordinate(0, 0), mapping.receiver_core));
            socket_connections.push_back(socket_connection);

            sender_logical_cr.push_back(CoreRange(mapping.sender_core));
            recv_worker_logical_cr.push_back(CoreRange(mapping.receiver_core));
            recv_worker_logical_cr.push_back(mapping.worker_cores);

            output_logical_cr.insert(
                output_logical_cr.end(), mapping.output_cores.ranges().begin(), mapping.output_cores.ranges().end());
            num_data_cores += mapping.data_cores.num_cores();
            num_output_cores += mapping.output_cores.num_cores();
        }
        sender_crs = CoreRangeSet(sender_logical_cr);
        recv_worker_crs = CoreRangeSet(recv_worker_logical_cr);
        output_crs = CoreRangeSet(output_logical_cr);
    }

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config(socket_connections, socket_mem_config);

    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md0, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_data_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size * num_data_cores,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig sender_buffer_config{.size = data_size * num_data_cores};

    // Only used for allocation
    auto output_shard_params = ShardSpecBuffer(output_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(output_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig output_buffer_config{.size = data_size * num_output_cores};

    auto sender_data_buffer = MeshBuffer::create(sender_buffer_config, sender_device_local_config, md0.get());

    auto output_buffer = MeshBuffer::create(output_buffer_config, output_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(sender_buffer_config.size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto send_recv_program = CreateProgram();

    auto sender_cb_index = tt::CBIndex::c_0;
    auto sender_cb_config = CircularBufferConfig(data_size, {{sender_cb_index, tt::DataFormat::UInt32}})
                                .set_page_size(sender_cb_index, data_size);
    CreateCircularBuffer(send_recv_program, sender_crs, sender_cb_config);

    // Create CB on both receiver and worker so that receiver knows the address
    auto config_cb_index = tt::CBIndex::c_0;
    auto config_cb_config =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config_cb_index, sizeof(receiver_socket_md));
    CreateCircularBuffer(send_recv_program, recv_worker_crs, config_cb_config);

    auto data_cb_index = tt::CBIndex::c_1;
    auto data_cb_config = CircularBufferConfig(2 * page_size, {{data_cb_index, tt::DataFormat::UInt32}})
                              .set_page_size(data_cb_index, page_size);
    // No need to create on recv core, but better dispatch to do so
    CreateCircularBuffer(send_recv_program, recv_worker_crs, data_cb_config);

    auto config_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);
    auto credits0_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);

    uint32_t data_offset = 0;
    for (const auto& mapping : socket_core_mappings) {
        const auto& sender_logical_coord = mapping.sender_core;
        const auto& recv_logical_coord = mapping.receiver_core;
        auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
        auto worker_logical_coords = corerange_to_cores(mapping.worker_cores, std::nullopt, true);
        auto worker_virtual_coords = md0->worker_cores_from_logical_cores(worker_logical_coords);
        auto data_logical_coords = corerange_to_cores(mapping.data_cores, std::nullopt, true);
        auto data_virtual_coords = md0->worker_cores_from_logical_cores(data_logical_coords);
        auto output_logical_coords = corerange_to_cores(mapping.output_cores, std::nullopt, true);
        auto output_virtual_coords = md0->worker_cores_from_logical_cores(output_logical_coords);
        std::vector<uint32_t> data_virtual_xys;
        std::vector<uint32_t> output_virtual_xys;
        data_virtual_xys.reserve(data_virtual_coords.size() * 2);
        output_virtual_xys.reserve(output_virtual_coords.size() * 2);
        for (uint32_t j = 0; j < data_virtual_coords.size(); ++j) {
            data_virtual_xys.push_back(data_virtual_coords[j].x);
            data_virtual_xys.push_back(data_virtual_coords[j].y);
            output_virtual_xys.push_back(output_virtual_coords[j].x);
            output_virtual_xys.push_back(output_virtual_coords[j].y);
        }

        auto sender_kernel = CreateKernel(
            send_recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender_multi_data.cpp",
            sender_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(send_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(sender_data_buffer->address() + data_offset),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(data_logical_coords.size()),
                    static_cast<uint32_t>(sender_virtual_data_core.x),
                    static_cast<uint32_t>(sender_virtual_data_core.y),
                    static_cast<uint32_t>(sender_cb_index),
                }});

        const auto& sender_rtas = data_virtual_xys;
        SetRuntimeArgs(send_recv_program, sender_kernel, sender_logical_coord, sender_rtas);
        data_offset += data_logical_coords.size() * data_size;

        if (final_ack) {
            CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_final_ack.cpp",
                recv_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.size()),
                    }});
            auto worker_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_final_ack.cpp",
                mapping.worker_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(data_cb_index),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(recv_virtual_coord.x),
                        static_cast<uint32_t>(recv_virtual_coord.y),
                        static_cast<uint32_t>(output_buffer->address()),
                    }});
            for (uint32_t j = 0; j < worker_logical_coords.size(); ++j) {
                std::vector<uint32_t> worker_rtas = {
                    static_cast<uint32_t>(data_virtual_coords[j].x),
                    static_cast<uint32_t>(data_virtual_coords[j].y),
                    static_cast<uint32_t>(output_virtual_coords[j].x),
                    static_cast<uint32_t>(output_virtual_coords[j].y),
                };
                SetRuntimeArgs(send_recv_program, worker_kernel, worker_logical_coords[j], worker_rtas);
            }
        } else {
            auto credits1_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);
            CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_loop_ack.cpp",
                recv_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(credits1_sem),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.size()),
                    }});

            auto worker_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_loop_ack.cpp",
                mapping.worker_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(credits1_sem),
                        static_cast<uint32_t>(data_cb_index),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(recv_virtual_coord.x),
                        static_cast<uint32_t>(recv_virtual_coord.y),
                        static_cast<uint32_t>(output_buffer->address())}});
            for (uint32_t j = 0; j < worker_logical_coords.size(); ++j) {
                std::vector<uint32_t> worker_rtas = {
                    static_cast<uint32_t>(data_virtual_coords[j].x),
                    static_cast<uint32_t>(data_virtual_coords[j].y),
                    static_cast<uint32_t>(output_virtual_coords[j].x),
                    static_cast<uint32_t>(output_virtual_coords[j].y),
                };
                SetRuntimeArgs(send_recv_program, worker_kernel, worker_logical_coords[j], worker_rtas);
            }
        }
    }

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    mesh_workload.add_program(devices, std::move(send_recv_program));

    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    uint8_t* src_ptr = reinterpret_cast<uint8_t*>(src_vec.data());
    uint32_t pages_per_core = data_size / page_size;
    for (const auto& mapping : socket_core_mappings) {
        std::vector<uint32_t> output_data_readback;
        uint32_t num_local_output_cores = mapping.output_cores.num_cores();
        auto local_output_shard_params = ShardSpecBuffer(
            mapping.output_cores,
            {pages_per_core, 1},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {pages_per_core, num_local_output_cores});

        const DeviceLocalBufferConfig local_output_device_local_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(local_output_shard_params, TensorMemoryLayout::WIDTH_SHARDED),
            .bottom_up = false};

        const ReplicatedBufferConfig local_buffer_config{.size = data_size * num_local_output_cores};

        auto local_output_buffer = MeshBuffer::create(
            local_buffer_config, local_output_device_local_config, md0.get(), output_buffer->address());

        ReadShard(md0->mesh_command_queue(), output_data_readback, local_output_buffer, MeshCoordinate(0, 0));

        EXPECT_EQ(std::memcmp(src_ptr, output_data_readback.data(), local_buffer_config.size), 0);
        src_ptr += local_buffer_config.size;
    }
}

void test_single_connection_multi_device_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md1,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    bool use_cbs) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);

    auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_logical_coord);

    auto fabric_max_packet_size = tt_fabric::get_tt_fabric_max_payload_size_bytes();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Used to setup fabric connections
    const uint32_t sender_physical_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t recv_physical_device_id = md1->get_device(MeshCoordinate(0, 0))->id();
    const auto sender_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender_physical_device_id);
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);

    // Create Socket between Sender and Receiver
    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config({socket_connection}, socket_mem_config);
    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md1, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());
    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, md1.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto sender_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {static_cast<uint32_t>(send_socket.get_config_buffer()->address()),
                 static_cast<uint32_t>(sender_data_buffer->address()),
                 static_cast<uint32_t>(page_size),
                 static_cast<uint32_t>(data_size)},
            .defines = {{"FABRIC_MAX_PACKET_SIZE", std::to_string(fabric_max_packet_size)}}});

    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id, recv_fabric_node_id, 0, sender_program, {sender_logical_coord}, sender_rtas);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    auto recv_program = CreateProgram();

    KernelHandle recv_kernel;
    if (use_cbs) {
        auto data_format = tt::DataFormat::UInt32;
        auto tile_size_bytes = tile_size(data_format);
        if (page_size % tile_size_bytes != 0) {
            GTEST_SKIP() << "Page size must be a multiple of tile size";
        }
        if (page_size / tile_size_bytes > 8) {
            GTEST_SKIP() << "Page size must be less than or equal to 8 tiles";
        }
        uint32_t num_tiles_per_page = page_size / tile_size_bytes;

        auto input_cb_index = CBIndex::c_1;
        auto input_cb_config = CircularBufferConfig(page_size, {{input_cb_index, data_format}})
                                   .set_page_size(input_cb_index, tile_size_bytes);
        CreateCircularBuffer(recv_program, recv_logical_coord, input_cb_config);

        auto output_cb_index = CBIndex::c_2;
        auto output_cb_config = CircularBufferConfig(2 * page_size, {{output_cb_index, data_format}})
                                    .set_page_size(output_cb_index, tile_size_bytes);
        CreateCircularBuffer(recv_program, recv_logical_coord, output_cb_config);

        CreateKernel(
            recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_cb_compute.cpp",
            recv_logical_coord,
            ComputeConfig{
                .dst_full_sync_en = true,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(input_cb_index),
                    static_cast<uint32_t>(output_cb_index),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(num_tiles_per_page)}});
        recv_kernel = CreateKernel(
            recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_cb_writer.cpp",
            recv_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(output_cb_index),
                    static_cast<uint32_t>(recv_data_buffer->address()),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(num_tiles_per_page)}});
    } else {
        recv_kernel = CreateKernel(
            recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp",
            recv_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(recv_virtual_coord.x),
                    static_cast<uint32_t>(recv_virtual_coord.y),
                    static_cast<uint32_t>(recv_data_buffer->address())}});
    }

    std::vector<uint32_t> recv_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        recv_fabric_node_id, sender_fabric_node_id, 0, recv_program, {recv_logical_coord}, recv_rtas);
    tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_logical_coord, recv_rtas);

    auto sender_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(md0->shape());
    sender_mesh_workload.add_program(devices, std::move(sender_program));

    auto recv_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_recv(md1->shape());
    recv_mesh_workload.add_program(devices_recv, std::move(recv_program));

    EnqueueMeshWorkload(md0->mesh_command_queue(), sender_mesh_workload, false);
    EnqueueMeshWorkload(md1->mesh_command_queue(), recv_mesh_workload, false);
    std::vector<uint32_t> recv_data_readback;
    ReadShard(md1->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

void test_single_connection_multi_device_socket_with_workers(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md1,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);
    auto worker_logical_coord = CoreCoord(0, 2);
    auto output_logical_coord = CoreCoord(0, 3);
    auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_logical_coord);
    auto worker_virtual_coord = md1->worker_core_from_logical_core(worker_logical_coord);
    auto output_virtual_coord = md1->worker_core_from_logical_core(output_logical_coord);

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    auto fabric_max_packet_size = tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Used to setup fabric connections
    const uint32_t sender_physical_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t recv_physical_device_id = md1->get_device(MeshCoordinate(0, 0))->id();
    const auto sender_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender_physical_device_id);
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);
    // Create Socket between Sender and Receiver
    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config({socket_connection}, socket_mem_config);
    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md1, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    auto output_shard_params =
        ShardSpecBuffer(CoreRangeSet(output_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(output_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto output_buffer = MeshBuffer::create(buffer_config, output_device_local_config, md1.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), std::chrono::system_clock::now().time_since_epoch().count());

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto sender_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {static_cast<uint32_t>(send_socket.get_config_buffer()->address()),
                 static_cast<uint32_t>(sender_data_buffer->address()),
                 static_cast<uint32_t>(page_size),
                 static_cast<uint32_t>(data_size)},
            .defines = {{"FABRIC_MAX_PACKET_SIZE", std::to_string(fabric_max_packet_size)}}});

    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id, recv_fabric_node_id, 0, sender_program, {sender_logical_coord}, sender_rtas);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    auto recv_program = CreateProgram();

    CoreRangeSet recv_worker_crs =
        CoreRangeSet(std::array{CoreRange(recv_logical_coord), CoreRange(worker_logical_coord)}).merge_ranges();

    // Create CB on both receiver and worker so that receiver knows the address
    auto config_cb_index = tt::CBIndex::c_1;
    auto config_cb_config =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config_cb_index, sizeof(receiver_socket_md));
    CreateCircularBuffer(recv_program, recv_worker_crs, config_cb_config);

    auto data_cb_index = tt::CBIndex::c_2;
    auto data_cb_config = CircularBufferConfig(2 * page_size, {{data_cb_index, tt::DataFormat::UInt32}})
                              .set_page_size(data_cb_index, page_size);
    // No need to create on recv core, but better dispatch to do so
    CreateCircularBuffer(recv_program, recv_worker_crs, data_cb_config);

    auto config_sem = CreateSemaphore(recv_program, recv_worker_crs, 0);
    auto credits_sem = CreateSemaphore(recv_program, recv_worker_crs, 0);

    auto recv_kernel = CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_final_ack.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
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

    std::vector<uint32_t> recv_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        recv_fabric_node_id, sender_fabric_node_id, 0, recv_program, {recv_logical_coord}, recv_rtas);
    tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_logical_coord, recv_rtas);

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
                static_cast<uint32_t>(output_buffer->address())}});

    std::vector<uint32_t> worker_rtas = {
        static_cast<uint32_t>(recv_virtual_coord.x),
        static_cast<uint32_t>(recv_virtual_coord.y),
        static_cast<uint32_t>(output_virtual_coord.x),
        static_cast<uint32_t>(output_virtual_coord.y),
    };
    tt_metal::SetRuntimeArgs(recv_program, worker_kernel, worker_logical_coord, worker_rtas);

    auto sender_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(md0->shape());
    sender_mesh_workload.add_program(devices, std::move(sender_program));

    auto recv_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_recv(md1->shape());
    recv_mesh_workload.add_program(devices_recv, std::move(recv_program));

    EnqueueMeshWorkload(md0->mesh_command_queue(), sender_mesh_workload, false);
    EnqueueMeshWorkload(md1->mesh_command_queue(), recv_mesh_workload, false);
    std::vector<uint32_t> recv_data_readback;
    ReadShard(md1->mesh_command_queue(), recv_data_readback, output_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

std::shared_ptr<Program> create_sender_program(
    const MeshSocket& sender_socket,
    const std::shared_ptr<MeshBuffer>& sender_data_buffer,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& sender_logical_coord,
    ChipId sender_physical_device_id,
    ChipId recv_physical_device_id,
    uint32_t sender_link_idx) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Used to setup fabric connections
    const auto sender_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender_physical_device_id);
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);

    auto fabric_max_packet_size = tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto sender_program = std::make_shared<Program>();
    auto sender_kernel = CreateKernel(
        *sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {static_cast<uint32_t>(sender_socket.get_config_buffer()->address()),
                 static_cast<uint32_t>(sender_data_buffer->address()),
                 static_cast<uint32_t>(page_size),
                 static_cast<uint32_t>(data_size)},
            .defines = {{"FABRIC_MAX_PACKET_SIZE", std::to_string(fabric_max_packet_size)}}});

    std::vector<uint32_t> sender_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id,
        recv_fabric_node_id,
        sender_link_idx,
        *sender_program,
        {sender_logical_coord},
        sender_rtas);
    tt_metal::SetRuntimeArgs(*sender_program, sender_kernel, sender_logical_coord, sender_rtas);

    return sender_program;
}

std::shared_ptr<Program> create_split_reduce_program(
    const MeshSocket& recv_socket_0,
    const MeshSocket& recv_socket_1,
    const std::shared_ptr<MeshBuffer>& recv_data_buffer,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& recv_logical_coord_0,
    const CoreCoord& recv_logical_coord_1,
    const CoreCoord& reduce_logical_coord,
    ChipId sender0_physical_device_id,
    ChipId sender1_physical_device_id,
    ChipId recv_physical_device_id,
    uint32_t sender0_link_idx,
    uint32_t sender1_link_idx) {
    auto config0_cb_index = tt::CBIndex::c_1;
    auto config1_cb_index = tt::CBIndex::c_2;
    auto in0_cb_index = tt::CBIndex::c_3;
    auto in1_cb_index = tt::CBIndex::c_4;

    // Used to setup fabric connections
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);
    const auto sender0_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender0_physical_device_id);
    const auto sender1_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender1_physical_device_id);

    auto recv_virtual_coord_0 = recv_data_buffer->device()->worker_core_from_logical_core(recv_logical_coord_0);
    auto recv_virtual_coord_1 = recv_data_buffer->device()->worker_core_from_logical_core(recv_logical_coord_1);
    auto reduce_virtual_core = recv_data_buffer->device()->worker_core_from_logical_core(reduce_logical_coord);

    auto recv_program = std::make_shared<Program>();

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
    // Socket Config CB
    CreateCircularBuffer(*recv_program, recv_worker_crs, config_cb_config0);
    CreateCircularBuffer(*recv_program, recv_worker_crs, config_cb_config1);
    // Data CBs
    CreateCircularBuffer(*recv_program, reduce_logical_coord, in0_cb_config);
    CreateCircularBuffer(*recv_program, reduce_logical_coord, in1_cb_config);

    auto config0_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto credits0_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto config1_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);
    auto credits1_sem = CreateSemaphore(*recv_program, recv_worker_crs, 0);

    auto recv_kernel_0 = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp",
        recv_logical_coord_0,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket_0.get_config_buffer()->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(config0_cb_index),
                static_cast<uint32_t>(config0_sem),
                static_cast<uint32_t>(credits0_sem),
                static_cast<uint32_t>(reduce_virtual_core.x),
                static_cast<uint32_t>(reduce_virtual_core.y)}});

    auto recv_kernel_1 = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp",
        recv_logical_coord_1,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket_1.get_config_buffer()->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(config1_cb_index),
                static_cast<uint32_t>(config1_sem),
                static_cast<uint32_t>(credits1_sem),
                static_cast<uint32_t>(reduce_virtual_core.x),
                static_cast<uint32_t>(reduce_virtual_core.y)}});

    CreateKernel(
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
        recv_fabric_node_id,
        sender0_fabric_node_id,
        sender0_link_idx,
        *recv_program,
        {recv_logical_coord_0},
        recv_rtas_0);
    tt_fabric::append_fabric_connection_rt_args(
        recv_fabric_node_id,
        sender1_fabric_node_id,
        sender1_link_idx,
        *recv_program,
        {recv_logical_coord_1},
        recv_rtas_1);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel_0, recv_logical_coord_0, recv_rtas_0);
    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel_1, recv_logical_coord_1, recv_rtas_1);

    return recv_program;
}

std::shared_ptr<Program> create_reduce_program(
    const MeshSocket& recv_socket_0,
    const MeshSocket& recv_socket_1,
    const MeshSocket& send_socket_2,
    const std::shared_ptr<MeshDevice>& /*reducer*/,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& reduce_logical_coord,
    ChipId sender0_physical_device_id,
    ChipId sender1_physical_device_id,
    ChipId reducer_physical_device_id,
    ChipId recv_physical_device_id,
    uint32_t sender0_link_idx,
    uint32_t sender1_link_idx,
    uint32_t recv_link_idx) {
    auto out_cb_index = tt::CBIndex::c_2;

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    // Used to setup fabric connections
    const auto sender0_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender0_physical_device_id);
    const auto sender1_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender1_physical_device_id);
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);
    const auto reducer_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(reducer_physical_device_id);

    auto reduce_program = std::make_shared<Program>();

    auto output_cb_config = CircularBufferConfig(2 * page_size, {{out_cb_index, tt::DataFormat::UInt32}})
                                .set_page_size(out_cb_index, page_size);
    CoreRangeSet reduce_crs = CoreRangeSet(reduce_logical_coord).merge_ranges();

    // Data CBs
    CreateCircularBuffer(*reduce_program, reduce_crs, output_cb_config);

    auto recv_kernel = CreateKernel(
        *reduce_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_reduce_receiver.cpp",
        reduce_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket_0.get_config_buffer()->address()),
                static_cast<uint32_t>(recv_socket_1.get_config_buffer()->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(out_cb_index)}});

    auto send_kernel = CreateKernel(
        *reduce_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_reduce_sender.cpp",
        reduce_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket_2.get_config_buffer()->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(out_cb_index)}});

    std::vector<uint32_t> recv_rtas;

    tt_fabric::append_fabric_connection_rt_args(
        reducer_fabric_node_id,
        sender0_fabric_node_id,
        sender0_link_idx,
        *reduce_program,
        reduce_logical_coord,
        recv_rtas);
    tt_fabric::append_fabric_connection_rt_args(
        reducer_fabric_node_id,
        sender1_fabric_node_id,
        sender1_link_idx,
        *reduce_program,
        reduce_logical_coord,
        recv_rtas);

    tt_metal::SetRuntimeArgs(*reduce_program, recv_kernel, reduce_logical_coord, recv_rtas);

    std::vector<uint32_t> send_rtas;
    tt_fabric::append_fabric_connection_rt_args(
        reducer_fabric_node_id, recv_fabric_node_id, recv_link_idx, *reduce_program, reduce_logical_coord, send_rtas);
    tt_metal::SetRuntimeArgs(*reduce_program, send_kernel, reduce_logical_coord, send_rtas);

    return reduce_program;
}

std::shared_ptr<Program> create_recv_program(
    const MeshSocket& recv_socket,
    const std::shared_ptr<MeshBuffer>& output_data_buffer,
    std::size_t page_size,
    std::size_t data_size,
    const CoreCoord& recv_logical_coord,
    const CoreCoord& output_logical_coord,
    ChipId sender_physical_device_id,
    ChipId recv_physical_device_id,
    uint32_t recv_link_idx) {
    // Used to setup fabric connections
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto sender_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(sender_physical_device_id);
    const auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id);

    auto output_virtual_coord = output_data_buffer->device()->worker_core_from_logical_core(output_logical_coord);

    auto recv_program = std::make_shared<Program>();

    CoreRangeSet recv_crs = CoreRangeSet(std::array{CoreRange(recv_logical_coord)}).merge_ranges();

    auto recv_kernel = CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.get_config_buffer()->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(output_virtual_coord.x),
                static_cast<uint32_t>(output_virtual_coord.y),
                static_cast<uint32_t>(output_data_buffer->address()),
            }});

    std::vector<uint32_t> recv_rtas;

    tt_fabric::append_fabric_connection_rt_args(
        recv_fabric_node_id, sender_fabric_node_id, recv_link_idx, *recv_program, {recv_logical_coord}, recv_rtas);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, recv_logical_coord, recv_rtas);

    return recv_program;
}

void test_multi_sender_single_recv(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& sender_0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& sender_1,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& reducer,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& receiver,
    const std::vector<uint32_t>& link_indices,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    std::size_t num_interations,
    bool split_reducer) {
    TT_FATAL(link_indices.size() == 3, "Link indices must be of size 3");
    // Used to setup fabric connections
    const uint32_t sender_0_physical_device_id = sender_0->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t sender_1_physical_device_id = sender_1->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t reducer_physical_device_id = reducer->get_device(MeshCoordinate(0, 0))->id();
    const uint32_t receiver_physical_device_id = receiver->get_device(MeshCoordinate(0, 0))->id();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv0_logical_coord = CoreCoord(0, 0);
    auto recv1_logical_coord = CoreCoord(0, 1);
    auto reduce_logical_coord = CoreCoord(0, 2);
    auto output_logical_coord = CoreCoord(0, 1);

    CoreCoord reduce_recv0_coord;
    CoreCoord reduce_recv1_coord;
    if (split_reducer) {
        reduce_recv0_coord = recv0_logical_coord;
        reduce_recv1_coord = recv1_logical_coord;
    } else {
        reduce_recv0_coord = reduce_logical_coord;
        reduce_recv1_coord = reduce_logical_coord;
    }

    SocketConnection socket_connection_0(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), reduce_recv0_coord));
    SocketConnection socket_connection_1(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), reduce_recv1_coord));
    SocketConnection socket_connection_2(
        MeshCoreCoord(MeshCoordinate(0, 0), reduce_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv0_logical_coord));
    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config_0({socket_connection_0}, socket_mem_config);
    SocketConfig socket_config_1({socket_connection_1}, socket_mem_config);
    SocketConfig socket_config_2({socket_connection_2}, socket_mem_config);

    auto [send_socket_0, recv_socket_0] = MeshSocket::create_socket_pair(sender_0, reducer, socket_config_0);
    auto [send_socket_1, recv_socket_1] = MeshSocket::create_socket_pair(sender_1, reducer, socket_config_1);
    auto [send_socket_2, recv_socket_2] = MeshSocket::create_socket_pair(reducer, receiver, socket_config_2);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    auto output_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(output_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(output_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig sender_buffer_config{.size = data_size};
    const ReplicatedBufferConfig output_buffer_config{.size = data_size};

    auto sender_data_buffer_0 = MeshBuffer::create(sender_buffer_config, sender_device_local_config, sender_0.get());
    auto sender_data_buffer_1 = MeshBuffer::create(sender_buffer_config, sender_device_local_config, sender_1.get());
    auto output_data_buffer = MeshBuffer::create(output_buffer_config, output_device_local_config, receiver.get());

    std::shared_ptr<MeshBuffer> reduce_data_buffer = nullptr;
    if (split_reducer) {
        auto reduce_data_shard_params =
            ShardSpecBuffer(CoreRangeSet(reduce_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

        const DeviceLocalBufferConfig reduce_device_local_config{
            .page_size = data_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(reduce_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = false};
        const ReplicatedBufferConfig reduce_buffer_config{.size = data_size};
        reduce_data_buffer = MeshBuffer::create(reduce_buffer_config, reduce_device_local_config, reducer.get());
    }

    auto sender_program_0 = create_sender_program(
        send_socket_0,
        sender_data_buffer_0,
        page_size,
        data_size,
        sender_logical_coord,
        sender_0_physical_device_id,
        reducer_physical_device_id,
        0);
    auto sender_program_1 = create_sender_program(
        send_socket_1,
        sender_data_buffer_1,
        page_size,
        data_size,
        sender_logical_coord,
        sender_1_physical_device_id,
        reducer_physical_device_id,
        0);
    std::shared_ptr<Program> reduce_program = nullptr;
    std::shared_ptr<Program> sender_program_2 = nullptr;
    if (split_reducer) {
        reduce_program = create_split_reduce_program(
            recv_socket_0,
            recv_socket_1,
            reduce_data_buffer,
            page_size,
            data_size,
            recv0_logical_coord,
            recv1_logical_coord,
            reduce_logical_coord,
            sender_0_physical_device_id,
            sender_1_physical_device_id,
            reducer_physical_device_id,
            link_indices[0],
            link_indices[1]);
        sender_program_2 = create_sender_program(
            send_socket_2,
            reduce_data_buffer,
            page_size,
            data_size,
            reduce_logical_coord,
            reducer_physical_device_id,
            receiver_physical_device_id,
            link_indices[2]);

    } else {
        reduce_program = create_reduce_program(
            recv_socket_0,
            recv_socket_1,
            send_socket_2,
            reducer,
            page_size,
            data_size,
            reduce_logical_coord,
            sender_0_physical_device_id,
            sender_1_physical_device_id,
            reducer_physical_device_id,
            receiver_physical_device_id,
            link_indices[0],
            link_indices[1],
            link_indices[2]);
    }
    auto recv_program = create_recv_program(
        recv_socket_2,
        output_data_buffer,
        page_size,
        data_size,
        recv0_logical_coord,
        output_logical_coord,
        reducer_physical_device_id,
        receiver_physical_device_id,
        0);

    auto sender_0_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_0(sender_0->shape());
    sender_0_mesh_workload.add_program(devices_0, std::move(*sender_program_0));

    auto sender_1_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_1(sender_1->shape());
    sender_1_mesh_workload.add_program(devices_1, std::move(*sender_program_1));

    auto reduce_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_reduce(reducer->shape());
    reduce_mesh_workload.add_program(devices_reduce, std::move(*reduce_program));
    MeshWorkload sender_2_mesh_workload;
    if (split_reducer) {
        sender_2_mesh_workload = MeshWorkload();
        sender_2_mesh_workload.add_program(devices_reduce, std::move(*sender_program_2));
    }

    auto recv_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_recv(receiver->shape());
    recv_mesh_workload.add_program(devices_recv, std::move(*recv_program));

    for (std::size_t i = 0; i < num_interations; ++i) {
        std::vector<uint32_t> src_vec =
            tt::test_utils::generate_uniform_random_vector<uint32_t>(0, UINT16_MAX, data_size / sizeof(uint32_t));
        // Write data to both senders
        WriteShard(sender_0->mesh_command_queue(), sender_data_buffer_0, src_vec, MeshCoordinate(0, 0));
        WriteShard(sender_1->mesh_command_queue(), sender_data_buffer_1, src_vec, MeshCoordinate(0, 0));

        EnqueueMeshWorkload(sender_0->mesh_command_queue(), sender_0_mesh_workload, false);
        EnqueueMeshWorkload(sender_1->mesh_command_queue(), sender_1_mesh_workload, false);
        EnqueueMeshWorkload(reducer->mesh_command_queue(), reduce_mesh_workload, false);
        if (split_reducer) {
            EnqueueMeshWorkload(reducer->mesh_command_queue(), sender_2_mesh_workload, false);
        }
        EnqueueMeshWorkload(receiver->mesh_command_queue(), recv_mesh_workload, false);

        std::vector<uint32_t> output_data_readback;
        if (split_reducer) {
            ReadShard(reducer->mesh_command_queue(), output_data_readback, reduce_data_buffer, MeshCoordinate(0, 0));
            for (size_t i = 0; i < src_vec.size(); ++i) {
                EXPECT_EQ(2 * src_vec[i], output_data_readback[i]);
            }
        }
        output_data_readback.clear();
        ReadShard(receiver->mesh_command_queue(), output_data_readback, output_data_buffer, MeshCoordinate(0, 0));
        for (size_t i = 0; i < src_vec.size(); ++i) {
            EXPECT_EQ(2 * src_vec[i], output_data_readback[i]);
        }
    }
}

void test_multi_connection_multi_device_data_copy(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& recv_mesh,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    auto sender_logical_core = CoreCoord(0, 0);
    auto recv_logical_core = CoreCoord(0, 0);

    std::vector<SocketConnection> socket_connections;

    socket_connections.reserve(4);
    for (std::size_t x = 0; x < 4; x++) {
        socket_connections.push_back(SocketConnection(
            MeshCoreCoord(MeshCoordinate(0, x), sender_logical_core),
            MeshCoreCoord(MeshCoordinate(0, 4 - x - 1), recv_logical_core)));
    }

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    auto [send_socket, recv_socket] =
        MeshSocket::create_socket_pair(sender_mesh, recv_mesh, SocketConfig(socket_connections, socket_mem_config));

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    const ReplicatedBufferConfig global_buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(global_buffer_config, sender_device_local_config, sender_mesh.get());
    auto recv_data_buffer = MeshBuffer::create(global_buffer_config, recv_device_local_config, recv_mesh.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    EnqueueWriteMeshBuffer(sender_mesh->mesh_command_queue(), sender_data_buffer, src_vec);

    auto sender_mesh_workload = MeshWorkload();
    auto recv_mesh_workload = MeshWorkload();

    for (const auto& connection : socket_connections) {
        auto sender_physical_id = sender_mesh->get_device(connection.sender_core.device_coord)->id();
        auto recv_physical_id = recv_mesh->get_device(connection.receiver_core.device_coord)->id();

        auto sender_program = create_sender_program(
            send_socket,
            sender_data_buffer,
            page_size,
            data_size,
            sender_logical_core,
            sender_physical_id,
            recv_physical_id,
            0);
        auto recv_program = create_recv_program(
            recv_socket,
            recv_data_buffer,
            page_size,
            data_size,
            recv_logical_core,
            recv_logical_core,
            sender_physical_id,
            recv_physical_id,
            0);

        sender_mesh_workload.add_program(
            MeshCoordinateRange(connection.sender_core.device_coord), std::move(*sender_program));
        recv_mesh_workload.add_program(
            MeshCoordinateRange(connection.receiver_core.device_coord), std::move(*recv_program));
    }
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_mesh_workload, false);
    EnqueueMeshWorkload(recv_mesh->mesh_command_queue(), recv_mesh_workload, false);
    for (std::size_t x = 0; x < 4; x++) {
        std::vector<uint32_t> output_data_readback;
        ReadShard(recv_mesh->mesh_command_queue(), output_data_readback, recv_data_buffer, MeshCoordinate(0, x));
        EXPECT_EQ(output_data_readback, src_vec);
    }
}

std::pair<MeshCoordinate, MeshCoordinate> get_random_mesh_coordinates(const MeshShape& mesh_shape) {
    std::srand(std::time(nullptr));  // Seed the RNG
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_2d_fabric_config(fabric_config)) {
        auto coord0 = MeshCoordinate(rand() % mesh_shape[0], rand() % mesh_shape[1]);
        auto coord1 = coord0;
        while (coord1 == coord0) {
            coord1 = MeshCoordinate(rand() % mesh_shape[0], rand() % mesh_shape[1]);
        }
        log_info(LogTest, "Random mesh coordinates: {} {}", coord0, coord1);
        return {coord0, coord1};
    }  // 1D fabric config requires neighboring devices for now
    auto coord0 = MeshCoordinate(0, 0);
    auto coord1 = MeshCoordinate(1, 0);
    return {coord0, coord1};
}

template <typename FixtureT>
void run_single_connection_multi_device_socket_with_workers(FixtureT* fixture) {
    auto [coord0, coord1] = get_random_mesh_coordinates(fixture->get_mesh_device()->shape());
    auto md0 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord0);
    auto md1 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord1);
    test_single_connection_multi_device_socket_with_workers(md0, md1, 1024, 64, 1024);
}

template <typename FixtureT>
void run_single_connection_multi_device_socket(FixtureT* fixture) {
    auto [coord0, coord1] = get_random_mesh_coordinates(fixture->get_mesh_device()->shape());
    auto md0 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord0);
    auto md1 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord1);
    test_single_connection_multi_device_socket(md0, md1, 1024, 64, 1024, false);
    test_single_connection_multi_device_socket(md0, md1, 1024, 64, 2048, false);
    test_single_connection_multi_device_socket(md0, md1, 4096, 1088, 9792, false);
}

template <typename FixtureT>
void run_single_connection_multi_device_socket_with_cbs(FixtureT* fixture) {
    constexpr auto tile_size_bytes = tile_size(tt::DataFormat::UInt32);
    auto [coord0, coord1] = get_random_mesh_coordinates(fixture->get_mesh_device()->shape());
    auto md0 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord0);
    auto md1 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coord1);

    test_single_connection_multi_device_socket(
        md0, md1, 2 * tile_size_bytes, tile_size_bytes, 4 * tile_size_bytes, true);
    test_single_connection_multi_device_socket(
        md0, md1, 6 * tile_size_bytes, 3 * tile_size_bytes, 15 * tile_size_bytes, true);
    test_single_connection_multi_device_socket(
        md0, md1, 5 * tile_size_bytes, 3 * tile_size_bytes, 27 * tile_size_bytes, true);
    test_single_connection_multi_device_socket(
        md0, md1, 9 * tile_size_bytes, 4 * tile_size_bytes, 28 * tile_size_bytes, true);
    test_single_connection_multi_device_socket(
        md0, md1, 6 * tile_size_bytes, 5 * tile_size_bytes, 25 * tile_size_bytes, true);
}

template <typename FixtureT>
void run_multi_sender_single_recv(FixtureT* fixture, bool split_reducer) {
    std::vector<MeshCoordinate> coordinates;
    std::vector<uint32_t> link_indices;
    if (tt_fabric::is_2d_fabric_config(tt::tt_metal::MetalContext::instance().get_fabric_config())) {
        auto mesh_device = fixture->get_mesh_device();
        auto mesh_shape = mesh_device->shape();
        coordinates.resize(mesh_shape.mesh_size(), MeshCoordinate(0, 0));
        int idx = 0;
        std::generate(coordinates.begin(), coordinates.end(), [&idx, mesh_shape]() mutable {
            int x = idx % mesh_shape[0];
            int y = idx / mesh_shape[0];
            ++idx;
            return MeshCoordinate(x, y);
        });
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        while (true) {
            link_indices.clear();
            std::shuffle(coordinates.begin(), coordinates.end(), std::mt19937(std::random_device()()));
            auto sender_0_physical_chip_id = mesh_device->get_device(coordinates[0])->id();
            auto sender_1_physical_chip_id = mesh_device->get_device(coordinates[1])->id();
            auto reducer_physical_chip_id = mesh_device->get_device(coordinates[2])->id();
            auto receiver_physical_chip_id = mesh_device->get_device(coordinates[3])->id();
            auto reducer_fabric_node_id =
                control_plane.get_fabric_node_id_from_physical_chip_id(reducer_physical_chip_id);

            std::unordered_set<tt_fabric::chan_id_t> used_channels;
            for (auto dst_chip_id : {sender_0_physical_chip_id, sender_1_physical_chip_id, receiver_physical_chip_id}) {
                auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_chip_id);
                auto forwarding_direction =
                    control_plane.get_forwarding_direction(reducer_fabric_node_id, dst_fabric_node_id).value();
                const auto candidate_eth_chans = control_plane.get_active_fabric_eth_channels_in_direction(
                    reducer_fabric_node_id, forwarding_direction);

                auto forwarding_links = get_forwarding_link_indices_in_direction(
                    reducer_fabric_node_id, dst_fabric_node_id, forwarding_direction);
                // Cannot use the last link which might already have a fabric router on it or used by dispatch
                // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
                if (!forwarding_links.empty()) {
                    forwarding_links.pop_back();
                }
                for (auto link_idx : forwarding_links) {
                    if (used_channels.find(candidate_eth_chans[link_idx]) == used_channels.end()) {
                        used_channels.insert(candidate_eth_chans[link_idx]);
                        link_indices.push_back(link_idx);
                        break;
                    }
                }
            }
            if (used_channels.size() == 3) {
                break;
            }
        }

    } else {
        coordinates = {MeshCoordinate(0, 0), MeshCoordinate(0, 2), MeshCoordinate(0, 1), MeshCoordinate(1, 1)};
        link_indices = {0, 0, 0};
    }

    auto sender_0 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coordinates[0]);
    auto sender_1 = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coordinates[1]);
    auto reducer = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coordinates[2]);
    auto receiver = fixture->get_mesh_device()->create_submesh(MeshShape(1, 1), coordinates[3]);

    log_info(LogTest, "Sender 0 ID: {}", sender_0->get_device(MeshCoordinate(0, 0))->id());
    log_info(LogTest, "Sender 1 ID: {}", sender_1->get_device(MeshCoordinate(0, 0))->id());
    log_info(LogTest, "Reduce ID: {}", reducer->get_device(MeshCoordinate(0, 0))->id());
    log_info(LogTest, "Receiver ID: {}", receiver->get_device(MeshCoordinate(0, 0))->id());

    uint32_t num_interations = 10;
    test_multi_sender_single_recv(
        sender_0, sender_1, reducer, receiver, link_indices, 1024, 64, 1024, num_interations, split_reducer);
    test_multi_sender_single_recv(
        sender_0, sender_1, reducer, receiver, link_indices, 2048, 64, 5120, num_interations, split_reducer);
    test_multi_sender_single_recv(
        sender_0, sender_1, reducer, receiver, link_indices, 4096, 1088, 9792, num_interations, split_reducer);
}

template <typename FixtureT>
void run_multi_connection_multi_device_data_copy(FixtureT* fixture) {
    fixture->get_mesh_device()->reshape(MeshShape(1, 8));

    auto sender_mesh = fixture->get_mesh_device()->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto recv_mesh = fixture->get_mesh_device()->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 4));

    test_multi_connection_multi_device_data_copy(sender_mesh, recv_mesh, 1024, 64, 1024);
    test_multi_connection_multi_device_data_copy(sender_mesh, recv_mesh, 1024, 64, 2048);
    test_multi_connection_multi_device_data_copy(sender_mesh, recv_mesh, 4096, 1088, 9792);
}

// ========= Config Validation Tests =========

// Sanity test with a single connection
TEST_F(MeshSocketTest, SingleConnectionSingleDeviceConfig) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    auto current_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(current_device_id);
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
    std::size_t socket_fifo_size = 1024;

    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config({socket_connection}, socket_mem_config);
    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md0, socket_config);

    std::vector<uint8_t> sender_config_bytes;
    std::vector<receiver_socket_md> recv_config_readback;

    ReadShard(md0->mesh_command_queue(), sender_config_bytes, send_socket.get_config_buffer(), MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_config_readback, recv_socket.get_config_buffer(), MeshCoordinate(0, 0));

    const uint32_t sender_page_size = send_socket.get_config_buffer()->page_size();
    EXPECT_EQ(sender_config_bytes.size(), sender_page_size);
    EXPECT_EQ(recv_config_readback.size(), 1);

    const auto& recv_config = recv_config_readback[0];

    // Parse single sender page (page index 0)
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // Single connection => max_num_downstreams = 1
    ParsedSenderPage sender_page =
        parse_sender_page(sender_config_bytes.data(), l1_alignment, /*max_num_downstreams*/ 1);

    verify_socket_configs(
        sender_page,
        recv_config,
        send_socket,
        recv_socket,
        current_fabric_node_id.chip_id,
        current_fabric_node_id.chip_id,
        sender_virtual_coord,
        recv_virtual_coord,
        socket_fifo_size);
}

// Test multiple connections
TEST_F(MeshSocketTest, MultiConnectionSingleDeviceConfig) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    auto current_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(current_device_id);
    std::size_t socket_fifo_size = 1024;
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

    std::vector<SocketConnection> socket_connections;

    socket_connections.reserve(sender_logical_coords.size());
    for (std::size_t core_idx = 0; core_idx < sender_logical_coords.size(); core_idx++) {
        socket_connections.push_back(SocketConnection(
            MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coords[core_idx]),
            MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coords[core_idx])));
    }

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config(socket_connections, socket_mem_config);

    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(md0, md0, socket_config);

    std::vector<uint8_t> sender_config_bytes;
    std::vector<receiver_socket_md> recv_configs;

    ReadShard(md0->mesh_command_queue(), sender_config_bytes, send_socket.get_config_buffer(), MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_configs, recv_socket.get_config_buffer(), MeshCoordinate(0, 0));

    const uint32_t sender_page_size = send_socket.get_config_buffer()->page_size();
    EXPECT_EQ(sender_config_bytes.size(), sender_page_size * sender_logical_coords.size());
    EXPECT_EQ(recv_configs.size(), recv_logical_coords.size());

    const auto& sender_core_to_core_id =
        send_socket.get_config_buffer()->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    const auto& recv_core_to_core_id =
        recv_socket.get_config_buffer()->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    const uint32_t max_num_downstreams = 1;
    const auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    for (const auto& connection : socket_connections) {
        const auto& sender = connection.sender_core;
        const auto& recv = connection.receiver_core;
        auto sender_idx = sender_core_to_core_id.at(sender.core_coord);
        auto recv_idx = recv_core_to_core_id.at(recv.core_coord);

        const uint8_t* page_ptr = sender_config_bytes.data() + (sender_idx * sender_page_size);
        ParsedSenderPage sender_page = parse_sender_page(page_ptr, l1_alignment, max_num_downstreams);
        const auto& recv_config = recv_configs[recv_idx];

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender.core_coord);
        auto recv_virtual_coord = md0->worker_core_from_logical_core(recv.core_coord);
        verify_socket_configs(
            sender_page,
            recv_config,
            send_socket,
            recv_socket,
            current_fabric_node_id.chip_id,
            current_fabric_node_id.chip_id,
            sender_virtual_coord,
            recv_virtual_coord,
            socket_fifo_size);
    }
}

// Test random connections across multiple devices
TEST_F(MeshSocketTest2DFabric, MultiConnectionMultiDeviceTest) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));
    std::unordered_map<MeshCoordinate, ChipId> sender_device_coord_to_id;
    std::unordered_map<MeshCoordinate, ChipId> receiver_device_coord_to_id;

    for (const auto& coord : MeshCoordinateRange(md0->shape())) {
        sender_device_coord_to_id[coord] = md0->get_device(coord)->id();
    }

    for (const auto& coord : MeshCoordinateRange(md1->shape())) {
        receiver_device_coord_to_id[coord] = md1->get_device(coord)->id();
    }
    std::size_t socket_fifo_size = 1024;
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

    std::vector<SocketConnection> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        SocketConnection socket_connection(
            MeshCoreCoord(sender_device_coords[coord_idx], sender_logical_coords[coord_idx]),
            MeshCoreCoord(recv_device_coords[coord_idx], recv_logical_coords[coord_idx]));
        socket_connections.push_back(socket_connection);
    }

    SocketMemoryConfig socket_mem_config_l1(BufferType::L1, socket_fifo_size);
    SocketConfig socket_config_l1(socket_connections, socket_mem_config_l1);
    SocketMemoryConfig socket_mem_config_dram(BufferType::DRAM, socket_fifo_size);
    SocketConfig socket_config_dram(socket_connections, socket_mem_config_dram);

    auto [send_socket_l1, recv_socket_l1] = MeshSocket::create_socket_pair(md0, md1, socket_config_l1);
    auto [send_socket_dram, recv_socket_dram] = MeshSocket::create_socket_pair(md0, md1, socket_config_dram);

    const auto& sender_core_to_core_id =
        send_socket_l1.get_config_buffer()->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    const auto& recv_core_to_core_id =
        recv_socket_l1.get_config_buffer()->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    std::unordered_map<MeshCoordinate, std::vector<uint8_t>> sender_bytes_per_dev_coord;
    std::unordered_map<MeshCoordinate, std::vector<receiver_socket_md>> recv_configs_per_dev_coord;

    for (const auto& device_coord : MeshCoordinateRange(md0->shape())) {
        std::vector<uint8_t> sender_bytes;
        std::vector<receiver_socket_md> recv_configs;

        ReadShard(md0->mesh_command_queue(), sender_bytes, send_socket_l1.get_config_buffer(), device_coord);
        ReadShard(md1->mesh_command_queue(), recv_configs, recv_socket_l1.get_config_buffer(), device_coord);

        sender_bytes_per_dev_coord[device_coord] = std::move(sender_bytes);
        recv_configs_per_dev_coord[device_coord] = std::move(recv_configs);
    }

    const uint32_t max_num_downstreams = 1;
    const uint32_t sender_page_size = send_socket_l1.get_config_buffer()->page_size();
    const auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    for (const auto& connection : socket_connections) {
        const auto& sender_core = connection.sender_core;
        const auto& recv_core = connection.receiver_core;
        const auto& sender_device_coord = sender_core.device_coord;
        const auto& recv_device_coord = recv_core.device_coord;
        const auto& sender_core_coord = sender_core.core_coord;
        const auto& recv_core_coord = recv_core.core_coord;

        auto sender_idx = sender_core_to_core_id.at(sender_core_coord);
        auto recv_idx = recv_core_to_core_id.at(recv_core_coord);

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_core_coord);
        auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_core_coord);
        auto sender_device_id = sender_device_coord_to_id[sender_device_coord];
        auto receiver_device_id = receiver_device_coord_to_id[recv_device_coord];

        const auto& sender_bytes = sender_bytes_per_dev_coord[sender_device_coord];
        const uint8_t* page_ptr = sender_bytes.data() + (sender_idx * sender_page_size);
        ParsedSenderPage sender_page = parse_sender_page(page_ptr, l1_alignment, max_num_downstreams);
        const auto& recv_config = recv_configs_per_dev_coord[recv_device_coord][recv_idx];

        auto sender_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(sender_device_id);
        auto receiver_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(receiver_device_id);
        verify_socket_configs(
            sender_page,
            recv_config,
            send_socket_l1,
            recv_socket_l1,
            receiver_fabric_node_id.chip_id,
            sender_fabric_node_id.chip_id,
            sender_virtual_coord,
            recv_virtual_coord,
            socket_fifo_size);
    }
}

// Verify that sockets are correctly created on different sub devices
TEST_F(MeshSocketTest2DFabric, SocketsOnSubDevice) {
    auto [coord0, coord1] = get_random_mesh_coordinates(mesh_device_->shape());
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    constexpr uint32_t socket_fifo_size = 1024;

    // Create sockets in global memory space. This socket is persistent, it lives regardless
    // of the sub device config loaded on the mesh_device
    SocketConnection global_socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)));
    SocketMemoryConfig global_socket_mem_cfg(BufferType::L1, socket_fifo_size);
    SocketConfig global_socket_config({global_socket_connection}, global_socket_mem_cfg);
    auto [send_socket_global, recv_socket_global] = MeshSocket::create_socket_pair(md0, md1, global_socket_config);

    SubDevice sub_device_0(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({1, 1}, {1, 1}))});

    // Create and load sub-device managers on both mesh devices
    auto sub_device_manager_0 = md0->create_sub_device_manager({sub_device_0, sub_device_1}, 3200);
    auto sub_device_manager_1 = md1->create_sub_device_manager({sub_device_0, sub_device_1}, 3200);

    md0->load_sub_device_manager(sub_device_manager_0);
    md1->load_sub_device_manager(sub_device_manager_1);

    {
        // Socket on sub device 0
        SocketConnection socket_0_connection(
            MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)));
        SocketMemoryConfig socket_mem_config_0(
            BufferType::L1, socket_fifo_size, md0->get_sub_device_ids()[0], md1->get_sub_device_ids()[0]);

        // Socket on sub device 1
        SocketConnection socket_1_connection(
            MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(1, 1)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(1, 1)));

        SocketMemoryConfig socket_mem_config_1(
            BufferType::L1, socket_fifo_size, md1->get_sub_device_ids()[1], md0->get_sub_device_ids()[1]);

        auto [send_socket_0, recv_socket_0] =
            MeshSocket::create_socket_pair(md0, md1, SocketConfig({socket_0_connection}, socket_mem_config_0));
        auto [send_socket_1, recv_socket_1] =
            MeshSocket::create_socket_pair(md1, md0, SocketConfig({socket_1_connection}, socket_mem_config_1));
        // Assert exppected: Socket cores don't match sub device
        EXPECT_THROW(
            MeshSocket::create_socket_pair(md0, md1, SocketConfig({socket_1_connection}, socket_mem_config_0)),
            std::exception);

        // Ensure that sockets were allocated using the sub device alloactor
        EXPECT_EQ(send_socket_0.get_config_buffer()->address(), send_socket_1.get_config_buffer()->address());
        EXPECT_EQ(recv_socket_0.get_config_buffer()->address(), recv_socket_1.get_config_buffer()->address());
        EXPECT_EQ(recv_socket_0.get_data_buffer()->address(), recv_socket_1.get_data_buffer()->address());
        // Try clearing the sub devices while sockets are still allocated - this should fail
        EXPECT_THROW(md0->clear_loaded_sub_device_manager(), std::exception);
        EXPECT_THROW(md1->clear_loaded_sub_device_manager(), std::exception);
    }
    // This should not fail - sub device sockets are now deallocated
    md0->clear_loaded_sub_device_manager();
    md1->clear_loaded_sub_device_manager();
}

TEST_F(MeshSocketTest, AssertOnDuplicateRecvCores) {
    auto [coord0, coord1] = get_random_mesh_coordinates(mesh_device_->shape());
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    constexpr uint32_t socket_fifo_size = 1024;

    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)));

    SocketConnection duplicate_socket_connection_0(
        MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(1, 0)));

    SocketConnection duplicate_socket_connection_1(
        MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(1, 0)), MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0)));

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config_0({socket_connection, duplicate_socket_connection_0}, socket_mem_config);

    SocketConfig socket_config_1({socket_connection, duplicate_socket_connection_1}, socket_mem_config);

    SocketConfig socket_config_2({socket_connection}, socket_mem_config);

    EXPECT_NO_THROW(MeshSocket::create_socket_pair(md0, md1, socket_config_0));
    EXPECT_THROW(MeshSocket::create_socket_pair(md0, md1, socket_config_1), std::exception);
    // Having the sender and receiver on the same core is valid. Ensure that this doesn't fail.
    EXPECT_NO_THROW(MeshSocket::create_socket_pair(md0, md0, socket_config_2));
}

void verify_socket_configs_match(const SocketConfig& config_a, const SocketConfig& config_b) {
    EXPECT_EQ(config_a.socket_connection_config.size(), config_b.socket_connection_config.size());

    // Make sure connections match
    for (size_t i = 0; i < config_a.socket_connection_config.size(); ++i) {
        const auto& local_conn = config_a.socket_connection_config[i];
        const auto& peer_conn = config_b.socket_connection_config[i];
        EXPECT_EQ(local_conn.sender_core, peer_conn.sender_core);
        EXPECT_EQ(local_conn.receiver_core, peer_conn.receiver_core);
    }
    // make sure socket memory config matches
    EXPECT_EQ(config_a.socket_mem_config.socket_storage_type, config_b.socket_mem_config.socket_storage_type);
    EXPECT_EQ(config_a.socket_mem_config.fifo_size, config_b.socket_mem_config.fifo_size);
    EXPECT_EQ(config_a.socket_mem_config.sender_sub_device, config_b.socket_mem_config.sender_sub_device);
    EXPECT_EQ(config_a.socket_mem_config.receiver_sub_device, config_b.socket_mem_config.receiver_sub_device);
    EXPECT_EQ(config_a.sender_mesh_id, config_b.sender_mesh_id);
    EXPECT_EQ(config_a.receiver_mesh_id, config_b.receiver_mesh_id);
    EXPECT_EQ(config_a.sender_rank, config_b.sender_rank);
    EXPECT_EQ(config_a.receiver_rank, config_b.receiver_rank);
}

// ========= Single Device Data Movement Tests =========

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocket) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    // No wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 1024, false);
    // Even wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 2048, false);
    // Uneven wrap
    test_single_connection_single_device_socket(md0, 4096, 1088, 9792, false);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithCBs) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto tile_size_bytes = tile_size(tt::DataFormat::UInt32);
    test_single_connection_single_device_socket(
        md0, 2 * tile_size_bytes, 1 * tile_size_bytes, 4 * tile_size_bytes, true);
    test_single_connection_single_device_socket(
        md0, 6 * tile_size_bytes, 3 * tile_size_bytes, 15 * tile_size_bytes, true);
    test_single_connection_single_device_socket(
        md0, 5 * tile_size_bytes, 2 * tile_size_bytes, 10 * tile_size_bytes, true);
    test_single_connection_single_device_socket(
        md0, 9 * tile_size_bytes, 4 * tile_size_bytes, 28 * tile_size_bytes, true);
    test_single_connection_single_device_socket(
        md0, 6 * tile_size_bytes, 5 * tile_size_bytes, 25 * tile_size_bytes, true);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithWorkersFinalAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<SocketCoreMapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {3, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {3, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {3, 5}))},
    };
    // These tests must not wrap and continue sending data
    test_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, true);
    test_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, true);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithWorkersLoopAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<SocketCoreMapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {3, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {3, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {3, 5}))},
    };
    // No wrap
    test_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, false);
    test_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, false);
    // Even wrap
    test_single_device_socket_with_workers(md0, 1024, 64, 2048, socket_core_mappings, false);
    // Uneven wrap
    test_single_device_socket_with_workers(md0, 4096, 1088, 9792, socket_core_mappings, false);
}

TEST_F(MeshSocketTest, MultiConnectionSingleDeviceSocketWithWorkersFinalAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<SocketCoreMapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {1, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {1, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {1, 5}))},
        {.sender_core = {2, 0},
         .receiver_core = {3, 0},
         .worker_cores = CoreRange({2, 1}, {4, 2}),
         .data_cores = CoreRangeSet(CoreRange({2, 3}, {4, 4})),
         .output_cores = CoreRangeSet(CoreRange({2, 4}, {4, 5}))},
    };
    // These tests must not wrap and continue sending data
    test_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, true);
    test_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, true);
}

TEST_F(MeshSocketTest, MultiConnectionSingleDeviceSocketWithWorkersLoopAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<SocketCoreMapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {1, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {1, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {1, 5}))},
        {.sender_core = {2, 0},
         .receiver_core = {3, 0},
         .worker_cores = CoreRange({2, 1}, {4, 2}),
         .data_cores = CoreRangeSet(CoreRange({2, 3}, {4, 4})),
         .output_cores = CoreRangeSet(CoreRange({2, 4}, {4, 5}))},
    };
    // No wrap
    test_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, false);
    test_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, false);
    // Even wrap
    test_single_device_socket_with_workers(md0, 1024, 64, 2048, socket_core_mappings, false);
    // Uneven wrap
    test_single_device_socket_with_workers(md0, 4096, 1088, 9792, socket_core_mappings, false);
}

// ========= Multi Device Data Movement Tests (1D Fabric) =========

TEST_F(MeshSocketTest1DFabric, SingleConnectionMultiDeviceSocketWithWorkers) {
    run_single_connection_multi_device_socket_with_workers(this);
}

TEST_F(MeshSocketTest1DFabric, SingleConnectionMultiDeviceSocket) { run_single_connection_multi_device_socket(this); }

TEST_F(MeshSocketTest1DFabric, SingleConnectionMultiDeviceSocketWithCBs) {
    run_single_connection_multi_device_socket_with_cbs(this);
}

TEST_F(MeshSocketTest1DFabric, MultiSenderSingleRecv) { run_multi_sender_single_recv(this, false); }

TEST_F(MeshSocketTest1DFabric, MultiSenderSingleRecvSplitReducer) { run_multi_sender_single_recv(this, true); }

TEST_F(MeshSocketTest1DFabric, MultiConnectionMultiDeviceDataCopy) {
    run_multi_connection_multi_device_data_copy(this);
}

// ========= Multi Device Data Movement Tests (2D Fabric with Dynamic Routing) =========

TEST_F(MeshSocketTest2DFabric, SingleConnectionMultiDeviceSocketWithWorkers) {
    run_single_connection_multi_device_socket_with_workers(this);
}

TEST_F(MeshSocketTest2DFabric, SingleConnectionMultiDeviceSocket) { run_single_connection_multi_device_socket(this); }

TEST_F(MeshSocketTest2DFabric, SingleConnectionMultiDeviceSocketWithCBs) {
    run_single_connection_multi_device_socket_with_cbs(this);
}

TEST_F(MeshSocketTest2DFabric, MultiSenderSingleRecv) { run_multi_sender_single_recv(this, false); }

TEST_F(MeshSocketTest2DFabric, MultiSenderSingleRecvSplitReducer) { run_multi_sender_single_recv(this, true); }

TEST_F(MeshSocketTest2DFabric, MultiConnectionMultiDeviceDataCopy) {
    run_multi_connection_multi_device_data_copy(this);
}

// ================== (De)Serialization Tests ==================

// Verify that serialization and deserialization of socket peer descriptors works correctly (this is needed for
// Multi-Host Sockets)
TEST(SocketSerializationTest, PeerDesc) {
    std::size_t socket_fifo_size = 1024;
    const auto worker_grid = CoreCoord(8, 8);

    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;
    std::vector<uint32_t> sender_chip_ids;
    std::vector<uint32_t> recv_chip_ids;
    std::vector<MeshCoordinate> sender_device_coords;
    std::vector<MeshCoordinate> recv_device_coords;
    uint32_t core_idx = 0;
    for (std::size_t x = 0; x < worker_grid.x; x++) {
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x, y));
            sender_chip_ids.push_back(core_idx % 4);
            recv_chip_ids.push_back(4 + (core_idx % 4));
            sender_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            recv_device_coords.push_back(MeshCoordinate(1, core_idx % 4));
            core_idx++;
        }
    }

    // Shuffle core coordinates to randomize the connections
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(sender_logical_coords.begin(), sender_logical_coords.end(), generator);
    std::shuffle(recv_logical_coords.begin(), recv_logical_coords.end(), generator);
    std::shuffle(sender_chip_ids.begin(), sender_chip_ids.end(), generator);
    std::shuffle(recv_chip_ids.begin(), recv_chip_ids.end(), generator);
    std::shuffle(sender_device_coords.begin(), sender_device_coords.end(), generator);
    std::shuffle(recv_device_coords.begin(), recv_device_coords.end(), generator);
    std::vector<SocketConnection> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        SocketConnection socket_connection(
            MeshCoreCoord(sender_device_coords[coord_idx], sender_logical_coords[coord_idx]),
            MeshCoreCoord(recv_device_coords[coord_idx], recv_logical_coords[coord_idx]));
        socket_connections.push_back(socket_connection);
    }

    SocketMemoryConfig socket_mem_config_l1_mh(BufferType::L1, socket_fifo_size);
    SocketConfig socket_config_l1 =
        SocketConfig(socket_connections, socket_mem_config_l1_mh, tt_fabric::MeshId{0}, tt_fabric::MeshId{1});

    // Populate sender size peer descriptor based on config, addresses and device coordinates
    SocketPeerDescriptor send_socket_peer_desc_l1 = SocketPeerDescriptor{
        .config = socket_config_l1,
        .config_buffer_address = 1 << 20,  // Assuming a dummy address for the config buffer at 1 MB
        .data_buffer_address = 0,          /* Sender Endpoint has no data buffer allocated. */
    };

    // Populate receiver size peer descriptor based on config, addresses and device coordinates
    SocketPeerDescriptor recv_socket_peer_desc_l1 = SocketPeerDescriptor{
        .config = socket_config_l1,
        .config_buffer_address = 1 << 21,  // Assuming a dummy address for the config buffer at 2 MB
        .data_buffer_address = 1 << 22,    // Assuming a dummy address for the data buffer at 4 MB
    };

    // Serialize and deserialize the socket peer descriptors
    auto serialized_send_socket_desc = serialize_to_bytes(send_socket_peer_desc_l1);
    SocketPeerDescriptor deserialized_send_socket_desc = deserialize_from_bytes(serialized_send_socket_desc);
    auto serialized_recv_socket_desc = serialize_to_bytes(recv_socket_peer_desc_l1);
    SocketPeerDescriptor deserialized_recv_socket_desc = deserialize_from_bytes(serialized_recv_socket_desc);
    // Validate configs are preserved after serialization
    verify_socket_configs_match(deserialized_send_socket_desc.config, send_socket_peer_desc_l1.config);
    verify_socket_configs_match(deserialized_recv_socket_desc.config, recv_socket_peer_desc_l1.config);
    verify_socket_configs_match(deserialized_send_socket_desc.config, deserialized_recv_socket_desc.config);
    // Validate that all other attributes of the peer descriptors are preserved
    EXPECT_EQ(deserialized_send_socket_desc.config_buffer_address, send_socket_peer_desc_l1.config_buffer_address);
    EXPECT_EQ(deserialized_recv_socket_desc.config_buffer_address, recv_socket_peer_desc_l1.config_buffer_address);
    EXPECT_EQ(deserialized_send_socket_desc.data_buffer_address, send_socket_peer_desc_l1.data_buffer_address);
    EXPECT_EQ(deserialized_recv_socket_desc.data_buffer_address, recv_socket_peer_desc_l1.data_buffer_address);
}

}  // namespace tt::tt_metal::distributed
