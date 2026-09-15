// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/distributed/utils.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::distributed::test {

class DispatchContextFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::DispatchContext::get().reset(); }
};

namespace {

std::optional<std::string> fd_preflight_skip_reason() {
    const auto& context = MetalContext::instance();
    if (context.rtoptions().get_fast_dispatch()) {
        return "This test can only be run with Slow Dispatch mode.";
    }
    const auto& cluster = context.get_cluster();
    if (cluster.is_mock_or_emulated()) {
        return "This test requires real hardware.";
    }
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        return "Manual Fast Dispatch setup is supported only on Galaxy and Blackhole clusters.";
    }
    return std::nullopt;
}

bool has_expected_dispatch_column(const MeshDevice& mesh) {
    // DispatchCoreConfig's axis is resolved only when MeshDevice::create
    // initializes the MetalContext, so this validation must happen afterwards.
    const auto& context = MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& dispatch_config = context.get_dispatch_core_config();
    if (cluster.arch() != tt::ARCH::BLACKHOLE || dispatch_config.get_dispatch_core_type() != DispatchCoreType::WORKER ||
        dispatch_config.get_dispatch_core_axis() != DispatchCoreAxis::COL) {
        return false;
    }
    for (ChipId chip : cluster.all_chip_ids()) {
        if (cluster.get_associated_mmio_device(chip) != chip) {
            return false;
        }
    }
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    return grid.x > 12 && grid.y > 1;
}

std::string capture_default_fd_refusal(MeshDevice* mesh) {
    try {
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh);
    } catch (const std::runtime_error& error) {
        return error.what();
    }

    // Keep the process usable if a regression makes the expected refusal
    // unexpectedly succeed.
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh);
    return {};
}

BufferShardingArgs two_dispatch_core_sharding_args(uint32_t page_size) {
    CoreRangeSet shard_grid(CoreRange({12, 0}, {12, 1}));
    ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{page_size, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{page_size, 1},
        /*tensor2d_shape_in_pages=*/{2, 1});
    return BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
}

}  // namespace

TEST_F(DispatchContextFixture, RefusesWhenResidentL1InsideDispatchFootprint) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{.page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "Expected resident L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("tt-blaze #2019"), std::string::npos);
    EXPECT_NE(error.find("[prefetch]"), std::string::npos);
    EXPECT_NE(error.find("[dispatch]"), std::string::npos);
    for (IDevice* device : mesh->get_devices()) {
        EXPECT_NE(error.find("chip " + std::to_string(device->id()) + " "), std::string::npos)
            << "Missing conflict for chip " << device->id();
    }

    // A refusal must leave the original Slow Dispatch queue usable.
    DeviceLocalBufferConfig dram{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig dram_global{.size = page_size};
    auto probe = MeshBuffer::create(dram_global, dram, mesh.get());
    std::vector<uint32_t> src(page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 7);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), probe, src);
    Finish(mesh->mesh_command_queue());
    std::vector<uint32_t> dst;
    ReadShard(mesh->mesh_command_queue(), dst, probe, MeshCoordinate(0, 0));
    EXPECT_EQ(dst, src);

    // A later clean session must succeed.
    resident.reset();
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get()));
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));
}

TEST_F(DispatchContextFixture, AllowDestructiveProceedsAndOverwritesResidentL1) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());

    std::vector<uint32_t> src(2 * page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0x12340000);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), resident, src);
    Finish(mesh->mesh_command_queue());

    experimental::FastDispatchSetupOptions options{.allow_destructive = true};
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get(), options));
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));

    std::vector<uint32_t> dst;
    ReadShard(mesh->mesh_command_queue(), dst, resident, MeshCoordinate(0, 0));
    EXPECT_NE(dst, src) << "The destructive override did not expose the expected dispatch-core overwrite.";
}

TEST_F(DispatchContextFixture, RefusesPerCoreResidentL1InsideDispatchFootprint) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "Per-core L1 allocation requires TT_METAL_ALLOCATOR_MODE_HYBRID=1.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    auto sharding_args = two_dispatch_core_sharding_args(page_size);
    experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = sharding_args, .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "Expected per-core resident L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("chip ledger"), std::string::npos);
    EXPECT_NE(error.find("[dispatch]"), std::string::npos);
}

TEST_F(DispatchContextFixture, WriteOnlyStillChecksPinnedWriteScratchRegion) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "Per-core L1 allocation requires TT_METAL_ALLOCATOR_MODE_HYBRID=1.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch core (12,0).";
    }

    // Top-down placement gives 0x180000 - 0x110000 = 0x70000, which is
    // above cmddat end (0x5CFC0) but inside pinned-write scratch (to 0x7CFC0).
    constexpr uint32_t page_size = 4096;
    constexpr uint32_t per_core_size = 0x110000;
    CoreRangeSet shard_grid(CoreRange({12, 0}));
    ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{per_core_size, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{page_size, 1},
        /*tensor2d_shape_in_pages=*/{per_core_size / page_size, 1});
    auto sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    DeviceLocalBufferConfig scratch_l1{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = sharding_args, .bottom_up = false};
    ReplicatedBufferConfig scratch_l1_global{.size = per_core_size};
    auto resident = MeshBuffer::create(scratch_l1_global, scratch_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    experimental::FastDispatchSetupOptions options{.write_only = true};
    std::string error;
    try {
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get(), options);
    } catch (const std::runtime_error& exception) {
        error = exception.what();
    }
    if (error.empty()) {
        experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get());
    }
    ASSERT_FALSE(error.empty()) << "write_only preflight failed to include the pinned-write scratch region.";
    EXPECT_NE(error.find("[prefetch]"), std::string::npos);
    EXPECT_EQ(error.find("[dispatch]"), std::string::npos);
}

TEST_F(DispatchContextFixture, RefusesPersistentArenaResidentL1InsideDispatchFootprint) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch core (12,1).";
    }

    auto& arena = mesh->allocator_impl()->persistent_l1();
    const auto allocation = arena.allocate(CoreRangeSet(CoreRange({12, 1})), /*size=*/4096, /*alignment=*/64);

    const std::string error = capture_default_fd_refusal(mesh.get());
    arena.deallocate(allocation.id);
    ASSERT_FALSE(error.empty()) << "Expected persistent arena L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("mesh arena ledger"), std::string::npos);
    EXPECT_NE(error.find("[dispatch]"), std::string::npos);
}

TEST_F(DispatchContextFixture, AllowsResidentL1AboveDispatchFootprint) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig high_l1{.page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig high_l1_global{.size = page_size};
    auto resident = MeshBuffer::create(high_l1_global, high_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get()));
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));
}

TEST_F(DispatchContextFixture, TestWritesAndWorkloads) {
    // Test using DispatchContext to turn FD on and off during runtime.
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    if (MetalContext::instance().get_cluster().is_mock_or_emulated()) {
        GTEST_SKIP() << "Mock/emulated devices cannot validate real data movement; see "
                        "MockDeviceFdSdToggleIsNoOp for the mock-specific no-op behavior.";
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

// Regression test for https://github.com/tenstorrent/tt-metal/issues/50634:
// A SD->FD->SD toggle must be a safe no-op on mock/emulated devices. These targets never create
// hardware command queues, so the FD teardown previously dereferenced an empty command-queue vector
// and segfaulted. Verify the round-trip completes and the device remains usable.
TEST_F(DispatchContextFixture, MockDeviceFdSdToggleIsNoOp) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    if (!MetalContext::instance().get_cluster().is_mock_or_emulated()) {
        GTEST_SKIP() << "This test only applies to mock/emulated devices.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    // SD -> FD -> SD toggle should not crash and should leave the device usable.
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
    Finish(mesh_device_->mesh_command_queue());
}

TEST(DispatchContext, DoubleInitWithoutTerminateShouldThrow) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.is_mock_or_emulated()) {
        GTEST_SKIP() << "FD/SD toggle is a no-op on mock/emulated devices; the throw invariants do not apply. See "
                        "MockDeviceFdSdToggleIsNoOp.";
    }
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
    if (cluster.is_mock_or_emulated()) {
        GTEST_SKIP() << "Mock/emulated devices cannot validate real data movement; see "
                        "MockDeviceFdSdToggleIsNoOp for the mock-specific no-op behavior.";
    }
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
