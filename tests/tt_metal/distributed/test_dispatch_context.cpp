// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
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

TEST(DispatchContext, TestWritesAndWorkloads) {
    // Test using DispatchContext to turn FD on and off during runtime.
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    // SD→FD transition requires dispatch core reallocation. On single-device setups with SD's
    // expanded grid, all dispatch cores are reclaimed for compute, leaving none available for FD.
    if (system_shape.mesh_size() == 1) {
        GTEST_SKIP() << "SD→FD→SD transition requires multi-device cluster (single-device lacks dispatch cores for FD "
                        "after SD grid expansion).";
    }

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

    // Initializing again should throw
    EXPECT_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get()), std::runtime_error);
}

// After SD -> enable FD -> disable FD, verify NOC/L1 bank tables by using an L1 buffer across the mesh.
TEST(DispatchContext, SdEnableFdDisableFdThenL1Buffer) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP()
            << "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.";
    }

    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());
    Finish(mesh_device_->mesh_command_queue());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const uint32_t num_tiles = 64;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig global_buffer_config{.size = num_tiles * single_tile_size};
    auto l1_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src_vec(num_tiles * single_tile_size / sizeof(uint32_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 42);
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), l1_buf, src_vec);
    Finish(mesh_device_->mesh_command_queue());

    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        std::vector<uint32_t> dst_vec = {};
        ReadShard(mesh_device_->mesh_command_queue(), dst_vec, l1_buf, coord);
        EXPECT_EQ(dst_vec, src_vec);
    }
}

}  // namespace tt::tt_metal::distributed::test
