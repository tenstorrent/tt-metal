// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

class DispatchContextFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::DispatchContext::get().reset(); }
};

TEST_F(DispatchContextFixture, TestWritesAndWorkloads) {
    // Test using DispatchContext to turn FD on and off during runtime.
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    // Terminating without initializing should throw
    EXPECT_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get()), std::runtime_error);

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 512;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;
    const uint32_t num_programs = 5;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src_vec(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    // Turn on Fast Dispatch for issuing writes.
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

    for (std::size_t logical_x = 0; logical_x < mesh_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < mesh_buffer->device()->num_rows(); logical_y++) {
            WriteShard(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec, MeshCoordinate(logical_y, logical_x));
        }
    }
    Finish(mesh_device_->mesh_command_queue());

    // Turn off FD for running a workload and issuing reads.
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    auto seed = 0;
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);

    for (uint32_t i = 0; i < num_programs; i++) {
        auto random_workload = std::make_shared<MeshWorkload>();
        random_workload->add_program(
            MeshCoordinateRange(
                MeshCoordinate{0, 0},
                MeshCoordinate{mesh_buffer->device()->num_rows() - 1, mesh_buffer->device()->num_cols() - 1}),
            std::move(*programs[i]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, true);
    }

    for (std::size_t logical_x = 0; logical_x < mesh_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < mesh_buffer->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer, MeshCoordinate(logical_y, logical_x));
            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

TEST(DispatchContext, DoubleInitWithoutTerminateShouldThrow) {
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

    // Initialize fast dispatch
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

    // Double init without terminate should throw
    EXPECT_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get()), std::runtime_error);

    // Clean up
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
}

// Stress test repeated SD <-> FD round-trips: verify buffer I/O and workload dispatch remain correct across cycles
TEST_F(DispatchContextFixture, RepeatedFdSdTransitionStress) {
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

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const uint32_t num_tiles = 64;
    const uint32_t num_programs = 5;

    CoreRangeSet shard_grid(CoreRange({0, 0}, {1, 1}));
    const uint32_t num_cores = 4;
    const uint32_t tiles_per_shard = num_tiles / num_cores;
    std::array<uint32_t, 2> shard_shape = {tiles_per_shard * single_tile_size, 1};
    std::array<uint32_t, 2> page_shape = {single_tile_size, 1};
    std::array<uint32_t, 2> tensor2d_shape = {num_tiles, 1};
    ShardSpecBuffer shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceLocalBufferConfig fd_l1_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};
    ReplicatedBufferConfig fd_l1_global{.size = num_tiles * single_tile_size};

    DeviceLocalBufferConfig sd_l1_config{
        .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig sd_l1_global{.size = num_tiles * single_tile_size};

    DeviceLocalBufferConfig fd_dram_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig fd_dram_global{.size = num_tiles * single_tile_size};

    constexpr uint32_t num_cycles = 5;
    for (uint32_t cycle = 0; cycle < num_cycles; cycle++) {
        const uint32_t base = cycle * 10000;

        // FD phase 1: sharded L1 buffer
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

        auto fd_buf = MeshBuffer::create(fd_l1_global, fd_l1_config, mesh_device_.get());
        std::vector<uint32_t> fd_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(fd_src_vec.begin(), fd_src_vec.end(), base + 100);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), fd_buf, fd_src_vec);
        Finish(mesh_device_->mesh_command_queue());

        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, fd_buf, coord);
            EXPECT_EQ(dst, fd_src_vec) << "Cycle " << cycle << ": sharded L1 readback failed in FD mode at " << coord;
        }

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

        // SD phase
        // Verify FD-written sharded buffer is still readable after FD->SD transition.
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, fd_buf, coord);
            EXPECT_EQ(dst, fd_src_vec) << "Cycle " << cycle << ": sharded L1 data mismatch after FD->SD transition at "
                                       << coord;
        }

        // Write and verify interleaved L1 buffer in SD mode. Validated shard-by-shard because
        // EnqueueReadMeshBuffer is only defined for sharded global layouts on multi-device meshes.
        auto sd_buf = MeshBuffer::create(sd_l1_global, sd_l1_config, mesh_device_.get());
        std::vector<uint32_t> sd_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(sd_src_vec.begin(), sd_src_vec.end(), base + 200);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), sd_buf, sd_src_vec);
        Finish(mesh_device_->mesh_command_queue());

        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, sd_buf, coord);
            EXPECT_EQ(dst, sd_src_vec) << "Cycle " << cycle << ": SD interleaved L1 verification failed at " << coord;
        }

        // Run random workloads to stress the dispatch path and dirty compute state.
        auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
            num_programs, mesh_device_->compute_with_storage_grid_size(), 0);
        for (uint32_t i = 0; i < num_programs; i++) {
            auto random_workload = std::make_shared<MeshWorkload>();
            random_workload->add_program(
                MeshCoordinateRange(
                    MeshCoordinate{0, 0}, MeshCoordinate{mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1}),
                std::move(*programs[i]));
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, true);
        }
        Finish(mesh_device_->mesh_command_queue());

        // Verify SD buffer is uncorrupted after running workloads.
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, sd_buf, coord);
            EXPECT_EQ(dst, sd_src_vec) << "Cycle " << cycle << ": SD buffer corrupted after running workloads at "
                                       << coord;
        }

        // FD phase 2: DRAM buffer
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

        auto fd2_buf = MeshBuffer::create(fd_dram_global, fd_dram_config, mesh_device_.get());
        std::vector<uint32_t> fd2_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(fd2_src_vec.begin(), fd2_src_vec.end(), base + 300);
        for (std::size_t y = 0; y < mesh_device_->num_rows(); y++) {
            for (std::size_t x = 0; x < mesh_device_->num_cols(); x++) {
                WriteShard(mesh_device_->mesh_command_queue(), fd2_buf, fd2_src_vec, MeshCoordinate(y, x));
            }
        }
        Finish(mesh_device_->mesh_command_queue());

        for (std::size_t y = 0; y < mesh_device_->num_rows(); y++) {
            for (std::size_t x = 0; x < mesh_device_->num_cols(); x++) {
                std::vector<uint32_t> dst;
                ReadShard(mesh_device_->mesh_command_queue(), dst, fd2_buf, MeshCoordinate(y, x));
                EXPECT_EQ(dst, fd2_src_vec)
                    << "Cycle " << cycle << ": DRAM readback failed in FD mode at (" << y << "," << x << ")";
            }
        }

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
    }
}

TEST_F(DispatchContextFixture, AsyncSdStatePreservedAcrossFdTransition) {
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

    // Enable async slow dispatch before FD transition
    experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh_device_.get());
    EXPECT_TRUE(experimental::DispatchContext::get().is_asynchronous_slow_dispatch_enabled(mesh_device_.get()));

    // SD -> FD -> SD round-trip
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    // Verify async SD state survived the round-trip
    EXPECT_TRUE(experimental::DispatchContext::get().is_asynchronous_slow_dispatch_enabled(mesh_device_.get()));
}

}  // namespace tt::tt_metal::distributed::test
