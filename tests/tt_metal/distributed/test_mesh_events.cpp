// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt_stl/span.hpp>
#include "tests/tt_metal/distributed/utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/util.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

using MeshEventsTestT3000 = T3000MultiCQMeshDeviceFixture;
using MeshEventsTestSuite = GenericMultiCQMeshDeviceFixture;

TEST_F(MeshEventsTestSuite, ReplicatedAsyncIO) {
    uint32_t NUM_TILES = 1000;
    uint32_t num_iterations = 20;
    int32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    ReplicatedBufferConfig global_buffer_config = {
        .size = NUM_TILES * single_tile_size,
    };

    std::shared_ptr<MeshBuffer> buf =
        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec(NUM_TILES * single_tile_size / sizeof(uint32_t), 0);
        std::iota(src_vec.begin(), src_vec.end(), i);

        std::vector<std::vector<uint32_t>> readback_vecs = {};
        // Writes on CQ 0
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(0), buf, src_vec);
        // Device to Device Synchronization
        auto write_event = EnqueueRecordEvent(mesh_device_->mesh_command_queue(0));
        EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), write_event);

        // Reads on CQ 1
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            readback_vecs.push_back({});
            auto shard = buf->get_device_buffer(coord);
            ReadShard(mesh_device_->mesh_command_queue(1), readback_vecs.back(), buf, coord);
        }

        for (auto& vec : readback_vecs) {
            EXPECT_EQ(vec, src_vec);
        }
    }
}

TEST_F(MeshEventsTestT3000, ShardedAsyncIO) {
    uint32_t num_iterations = 20;
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    Shape2D global_buffer_shape = {2048, 2048};
    Shape2D shard_shape = {512, 1024};

    uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

    ShardedBufferConfig sharded_config{
        .global_size = global_buffer_size,
        .global_buffer_shape = global_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), i);
        // Writes on CQ 0
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(0), mesh_buffer, src_vec);
        if (i % 2) {
            // Test Host <-> Device synchronization
            auto write_event = EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(0));
            EventSynchronize(write_event);
        } else {
            // Test Device <-> Device synchronization
            auto write_event = EnqueueRecordEvent(mesh_device_->mesh_command_queue(0));
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), write_event);
        }
        // Reads on CQ 1
        std::vector<uint32_t> dst_vec = {};
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(1), dst_vec, mesh_buffer);

        EXPECT_EQ(dst_vec, src_vec);
    }
}

TEST_F(MeshEventsTestSuite, AsyncWorkloadAndIO) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    uint32_t num_iters = 5;
    std::vector<std::shared_ptr<MeshBuffer>> src0_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> src1_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    auto programs = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, src0_bufs, src1_bufs, output_bufs);
    uint32_t num_rows_in_workload = mesh_device_->num_rows() / 2;
    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices_0(
        MeshCoordinate{0, 0}, MeshCoordinate{num_rows_in_workload - 1, mesh_device_->num_cols() - 1});
    MeshCoordinateRange devices_1(
        MeshCoordinate{num_rows_in_workload, 0},
        MeshCoordinate{mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1});

    AddProgramToMeshWorkload(mesh_workload, std::move(*programs[0]), devices_0);
    AddProgramToMeshWorkload(mesh_workload, std::move(*programs[1]), devices_1);

    for (int iter = 0; iter < num_iters; iter++) {
        std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), iter + 2);
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src1_bufs[0]->size(), iter + 3);

        // Issue writes on MeshCQ 1
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteMeshBuffer(
                    mesh_device_->mesh_command_queue(1), src0_bufs[col_idx * worker_grid_size.y + row_idx], src0_vec);
                EnqueueWriteMeshBuffer(
                    mesh_device_->mesh_command_queue(1), src1_bufs[col_idx * worker_grid_size.y + row_idx], src1_vec);
            }
        }
        if (iter % 2) {
            // Test Host <-> Device Synchronization
            auto write_event = EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(1));
            EventSynchronize(write_event);
        } else {
            // Test Device <-> Device Synchronization
            auto write_event = EnqueueRecordEvent(mesh_device_->mesh_command_queue(1));
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(0), write_event);
        }
        // Issue workloads on MeshCQ 0
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(0), mesh_workload, false);
        if (iter % 2) {
            // Test Device <-> Device Synchronization
            auto op_event = EnqueueRecordEvent(mesh_device_->mesh_command_queue(0));
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), op_event);
        } else {
            // Test Host <-> Device Synchronization
            auto op_event = EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(0));
            EventSynchronize(op_event);
        }

        // Issue reads on MeshCQ 1
        for (const auto& device_coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<bfloat16> dst_vec = {};
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    std::vector<bfloat16> dst_vec = {};
                    ReadShard(
                        mesh_device_->mesh_command_queue(1),
                        dst_vec,
                        output_bufs[col_idx * worker_grid_size.y + row_idx],
                        device_coord);
                    if (device_coord[0] <= num_rows_in_workload - 1) {
                        for (int i = 0; i < dst_vec.size(); i++) {
                            EXPECT_EQ(dst_vec[i].to_float(), (2 * iter + 5));
                        }
                    } else {
                        for (int i = 0; i < dst_vec.size(); i++) {
                            EXPECT_EQ(dst_vec[i].to_float(), (iter + 2) * (iter + 3));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MeshEventsTestSuite, CustomDeviceRanges) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    uint32_t NUM_TILES = 1000;
    uint32_t num_iterations = 20;
    int32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    ReplicatedBufferConfig global_buffer_config = {
        .size = NUM_TILES * single_tile_size,
    };

    std::shared_ptr<MeshBuffer> buf =
        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec(NUM_TILES * single_tile_size / sizeof(uint32_t), i);
        std::iota(src_vec.begin(), src_vec.end(), i);
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{0, mesh_device_->num_cols() - 1});
        MeshCoordinateRange devices_1(MeshCoordinate{1, 0}, MeshCoordinate{1, mesh_device_->num_cols() - 1});

        std::vector<std::vector<uint32_t>> readback_vecs = {};

        mesh_device_->mesh_command_queue(1).enqueue_write_shard_to_sub_grid(*buf, src_vec.data(), devices_0, false);
        auto event0 = EnqueueRecordEvent(mesh_device_->mesh_command_queue(1), {}, devices_0);
        EnqueueWaitForEvent(mesh_device_->mesh_command_queue(0), event0);

        for (const auto& coord : devices_0) {
            readback_vecs.push_back({});
            auto shard = buf->get_device_buffer(coord);
            ReadShard(mesh_device_->mesh_command_queue(0), readback_vecs.back(), buf, coord);
        }

        mesh_device_->mesh_command_queue(1).enqueue_write_shard_to_sub_grid(*buf, src_vec.data(), devices_1, false);
        auto event1 = EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(1), {}, devices_1);
        EventSynchronize(event1);

        for (const auto& coord : devices_1) {
            readback_vecs.push_back({});
            auto shard = buf->get_device_buffer(coord);
            ReadShard(mesh_device_->mesh_command_queue(0), readback_vecs.back(), buf, coord);
        }
        for (auto& vec : readback_vecs) {
            EXPECT_EQ(vec, src_vec);
        }
    }
    Finish(mesh_device_->mesh_command_queue(0));
    Finish(mesh_device_->mesh_command_queue(1));
}

TEST_F(MeshEventsTestSuite, MultiCQNonBlockingReads) {
    // Reads and writes on 2 CQs
    auto& write_cq = mesh_device_->mesh_command_queue(0);
    auto& read_cq = mesh_device_->mesh_command_queue(1);

    uint32_t num_tiles = 1024;
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    constexpr uint32_t NUM_ITERS = 500;

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};
    ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};
    MeshCoordinateRange devices_0(mesh_device_->shape());

    uint32_t num_devices = mesh_device_->num_devices();

    // Read and write different data from the same buffer across iterations
    auto buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    // Initialize containers to store input and output data
    std::vector<std::vector<uint32_t>> input_shard_data = {};
    std::vector<std::vector<MeshCommandQueue::ShardDataTransfer>> read_shards = {};
    std::vector<std::vector<uint32_t>> output_shard_data = {};

    for (int i = 0; i < NUM_ITERS; i++) {
        // Initialize different input data across iterations
        input_shard_data.push_back(std::vector<uint32_t>(dram_buffer_size / sizeof(uint32_t)));
        std::iota(input_shard_data.back().begin(), input_shard_data.back().end(), i);
        // Initialize ShardDataTransfer objects for reads across iterations and allocate
        // output buffers on host
        read_shards.push_back({});
        for (const auto& device_coord : devices_0) {
            output_shard_data.push_back(std::vector<uint32_t>(input_shard_data.back().size()));
            read_shards.back().push_back(MeshCommandQueue::ShardDataTransfer{
                .shard_coord = device_coord,
                .host_data = output_shard_data.back().data(),
            });
        }
    }

    // Events signalling read and write completion
    std::vector<MeshEvent> write_events;
    std::vector<MeshEvent> read_events;

    for (int i = 0; i < NUM_ITERS; i++) {
        if (i > 0) {
            // Wait for read to complete before writing, since the same
            // buffer is used across iterations
            EnqueueWaitForEvent(write_cq, read_events.back());
        }
        EnqueueWriteMeshBuffer(write_cq, buffer, input_shard_data[i], true);
        write_events.push_back(EnqueueRecordEventToHost(write_cq));
        // Wait for write to complete before reading
        EnqueueWaitForEvent(read_cq, write_events.back());
        read_cq.enqueue_read_shards(read_shards[i], buffer, false);
        read_events.push_back(EnqueueRecordEventToHost(read_cq));
    }

    // Stall on read and write CQs before data verification
    Finish(write_cq);
    Finish(read_cq);

    uint32_t idx = 0;
    for (auto& dst_vec : output_shard_data) {
        EXPECT_EQ(dst_vec, input_shard_data[idx / num_devices]);
        idx++;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
