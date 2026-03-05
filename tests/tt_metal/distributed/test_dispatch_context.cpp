// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/distributed/utils.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::distributed::test {

TEST(DispatchContext, RuntimeDispatchModeTransitions) {
    // Merged test covering all dispatch context scenarios:
    // 1. Error handling (terminate without init)
    // 2. SD -> FD for writes
    // 3. FD mode with L1 sharded buffers
    // 4. FD -> SD for workloads and reads
    // 5. SD mode buffer operations after FD->SD transition
    // 6. Error handling (re-init after terminate)
    //
    // Note: DispatchContext only allows ONE FD init/terminate cycle per process,
    // so all scenarios must be tested in a single test.

    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP()
            << "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.";
    }

    // === Part 1: Test error handling - terminating without initializing should throw ===
    EXPECT_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get()), std::runtime_error);

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // === Part 2: Initialize FD and test DRAM buffer writes ===
    DeviceLocalBufferConfig dram_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 512;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig dram_global_config{.size = bytes_per_device};
    auto dram_buffer = MeshBuffer::create(dram_global_config, dram_buffer_config, mesh_device_.get());

    std::vector<uint32_t> dram_src_vec(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(dram_src_vec.begin(), dram_src_vec.end(), 0);

    // Turn on Fast Dispatch for issuing writes
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

    for (std::size_t logical_x = 0; logical_x < dram_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < dram_buffer->device()->num_rows(); logical_y++) {
            WriteShard(
                mesh_device_->mesh_command_queue(), dram_buffer, dram_src_vec, MeshCoordinate(logical_y, logical_x));
        }
    }
    Finish(mesh_device_->mesh_command_queue());

    // === Part 3: Test L1 sharded buffer in FD mode ===
    const uint32_t num_tiles = 64;
    CoreRangeSet shard_grid(CoreRange({0, 0}, {1, 1}));
    const uint32_t num_cores = 4;
    const uint32_t tiles_per_shard = num_tiles / num_cores;
    std::array<uint32_t, 2> shard_shape = {tiles_per_shard * single_tile_size, 1};
    std::array<uint32_t, 2> page_shape = {single_tile_size, 1};
    std::array<uint32_t, 2> tensor2d_shape = {num_tiles, 1};
    ShardSpecBuffer shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceLocalBufferConfig l1_sharded_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = true};
    ReplicatedBufferConfig l1_sharded_global_config{.size = num_tiles * single_tile_size};
    auto l1_sharded_buf = MeshBuffer::create(l1_sharded_global_config, l1_sharded_config, mesh_device_.get());

    std::vector<uint32_t> l1_src_vec(num_tiles * single_tile_size / sizeof(uint32_t), 0);
    std::iota(l1_src_vec.begin(), l1_src_vec.end(), 100);
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), l1_sharded_buf, l1_src_vec);
    Finish(mesh_device_->mesh_command_queue());

    // Verify sharded buffer readback in FD mode
    std::vector<uint32_t> l1_dst_vec = {};
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        ReadShard(mesh_device_->mesh_command_queue(), l1_dst_vec, l1_sharded_buf, coord);
        EXPECT_EQ(l1_dst_vec, l1_src_vec) << "L1 sharded buffer readback failed in FD mode";
    }

    // === Part 4: Turn off FD and test workloads + DRAM buffer reads in SD mode ===
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    const uint32_t num_programs = 5;
    auto seed = 0;
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);

    for (uint32_t i = 0; i < num_programs; i++) {
        auto random_workload = std::make_shared<MeshWorkload>();
        random_workload->add_program(
            MeshCoordinateRange(
                MeshCoordinate{0, 0},
                MeshCoordinate{dram_buffer->device()->num_rows() - 1, dram_buffer->device()->num_cols() - 1}),
            std::move(*programs[i]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, true);
    }

    // Verify DRAM buffer written in FD mode can be read in SD mode
    for (std::size_t logical_x = 0; logical_x < dram_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < dram_buffer->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dram_dst_vec = {};
            ReadShard(
                mesh_device_->mesh_command_queue(), dram_dst_vec, dram_buffer, MeshCoordinate(logical_y, logical_x));
            EXPECT_EQ(dram_dst_vec, dram_src_vec) << "DRAM buffer readback failed in SD mode after FD->SD transition";
        }
    }

    // === Part 5: Verify L1 sharded buffer still works after FD->SD transition ===
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        std::vector<uint32_t> l1_buf_readback_in_sd = {};
        ReadShard(mesh_device_->mesh_command_queue(), l1_buf_readback_in_sd, l1_sharded_buf, coord);
        EXPECT_EQ(l1_buf_readback_in_sd, l1_src_vec) << "L1 sharded buffer data mismatch after FD->SD transition";
    }

    // === Part 6: Test interleaved L1 buffer operations in SD mode ===
    DeviceLocalBufferConfig l1_interleaved_config{
        .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig l1_interleaved_global_config{.size = num_tiles * single_tile_size};
    auto l1_interleaved_buf =
        MeshBuffer::create(l1_interleaved_global_config, l1_interleaved_config, mesh_device_.get());

    std::vector<uint32_t> l1_interleaved_src_vec(num_tiles * single_tile_size / sizeof(uint32_t), 0);
    std::iota(l1_interleaved_src_vec.begin(), l1_interleaved_src_vec.end(), 200);
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), l1_interleaved_buf, l1_interleaved_src_vec);
    Finish(mesh_device_->mesh_command_queue());

    std::vector<uint32_t> l1_interleaved_dst_vec = {};
    EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), l1_interleaved_dst_vec, l1_interleaved_buf, true);
    EXPECT_EQ(l1_interleaved_dst_vec, l1_interleaved_src_vec) << "L1 interleaved buffer verification failed in SD mode";

    // === Part 7: Test error handling - initializing again after terminate should throw ===
    EXPECT_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get()), std::runtime_error);
}

}  // namespace tt::tt_metal::distributed::test
