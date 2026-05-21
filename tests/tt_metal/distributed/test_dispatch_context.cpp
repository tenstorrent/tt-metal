// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/experimental/service_core_claims.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/llrt.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
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

// Verify that claiming all free FD dispatch cores as service cores:
//   (a) caps compute_with_storage_grid_size() to the FD grid in SD mode (conservative cap),
//   (b) leaves no idle dispatch cores when FD is re-initialized.
TEST_F(DispatchContextFixture, ServiceCoreGridCap) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test requires Slow Dispatch mode.";
    }
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Service-core APIs require BH or UBB Galaxy.";
    }

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    // Phase 1: record full SD grid (no services active).
    const CoreCoord sd_grid = device->compute_with_storage_grid_size();

    // Phase 2: init FD, record FD grid, claim all free dispatch cores.
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
    const CoreCoord fd_grid = device->compute_with_storage_grid_size();
    ASSERT_LT(fd_grid.x * fd_grid.y, sd_grid.x * sd_grid.y)
        << "FD grid must be smaller than SD grid (dispatch column excluded by YAML)";

    const auto free_cores = experimental::service::ServiceCoreClaims::get().get_claimable_cores(device);
    ASSERT_FALSE(free_cores.empty()) << "Expected at least one free dispatch core";

    experimental::service::ServiceCoreClaims::get().claim(device, free_cores);

    // Attempting to claim the same cores again must fail — claim guard is independent
    // of the pool filter and catches double-claim regardless of FD state.
    EXPECT_THROW(experimental::service::ServiceCoreClaims::get().claim(device, free_cores), std::exception);

    // Phase 3: terminate + re-init FD — pool filter must exclude all claimed cores,
    // leaving no idle dispatch cores for new FD kernels.
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());

    EXPECT_TRUE(experimental::service::ServiceCoreClaims::get().get_claimable_cores(device).empty())
        << "All free dispatch cores were claimed; none should be available in FD";

    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

    // Phase 4: back in SD — grid must be capped to the FD grid (conservative cap).
    const CoreCoord capped_grid = device->compute_with_storage_grid_size();
    EXPECT_EQ(capped_grid.x, fd_grid.x) << "SD grid x must be capped to FD grid x when service cores are claimed";
    EXPECT_EQ(capped_grid.y, fd_grid.y) << "SD grid y must be capped to FD grid y when service cores are claimed";

    // Cleanup.
    experimental::service::ServiceCoreClaims::get().release(device, free_cores);
    experimental::service::ServiceCoreClaims::get().on_device_close(device->id());
}

// Persistent service pipecleaner: test a Persistent kernel on FD cores (survive multiple FD init + teardown cycles)
// Kernel increments a counter in L1 until a stop flag is set
// We verify the counter is still incrementing after each FD lifecycle transition
TEST_F(DispatchContextFixture, PersistentServiceMultiCycle) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test requires Slow Dispatch mode.";
    }
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Service-core APIs require BH or UBB Galaxy.";
    }

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const uint32_t num_tiles = 64;
    const uint32_t num_programs = 1;

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

    constexpr int num_cycles = 5;
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        // Phase 1: FD up, claim a free core, drop FD.
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());

        auto free = experimental::service::ServiceCoreClaims::get().get_claimable_cores(device);
        ASSERT_FALSE(free.empty()) << "No free dispatch cores on cycle " << cycle;
        CoreCoord svc_core = free[free.size() / 2];

        experimental::service::ServiceCoreClaims::get().claim(device, free);
        DeviceAddr stop_addr =
            experimental::service::ServiceCoreClaims::get().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr counter_addr =
            experimental::service::ServiceCoreClaims::get().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr service_done_addr =
            experimental::service::ServiceCoreClaims::get().allocate_l1(device, svc_core, sizeof(uint32_t));

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

        // Initialise all three words to 0.
        CoreCoord phys_core = device->worker_core_from_logical_core(svc_core);
        tt_cxy_pair phys_cxy(device->id(), phys_core.x, phys_core.y);
        const uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, stop_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, counter_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, service_done_addr);

        // Phase 2: launch persistent kernel (SD, non-blocking).
        // The kernel handles FD dispatch go signals internally via notify_dispatch_core_done.
        Program prog;
        auto kernel = CreateKernel(
            prog,
            "tests/tt_metal/tt_metal/test_kernels/misc/service_counter.cpp",
            svc_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(
            prog, kernel, svc_core, {(uint32_t)stop_addr, (uint32_t)counter_addr, (uint32_t)service_done_addr});

        tt::tt_metal::detail::CompileProgram(device, prog, /*force_slow_dispatch=*/true);
        tt::tt_metal::detail::WriteRuntimeArgsToDevice(device, prog, /*force_slow_dispatch=*/true);
        tt::tt_metal::detail::LaunchProgram(
            device, prog, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

        auto read_counter = [&]() -> uint32_t {
            uint32_t val = 0;
            cluster.read_core(&val, sizeof(uint32_t), phys_cxy, counter_addr);
            return val;
        };

        auto assert_incrementing = [&](const std::string& label) {
            uint32_t a = read_counter();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            uint32_t b = read_counter();
            EXPECT_GT(b, a) << label << " (cycle " << cycle << "): counter stalled (" << a << " -> " << b << ")";
        };

        assert_incrementing("SD baseline");

        // Phase 3: FD re-init
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
        assert_incrementing("after FD init");

        // Sharded L1 buffer
        auto fd_buf = MeshBuffer::create(fd_l1_global, fd_l1_config, mesh_device.get());
        std::vector<uint32_t> fd_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(fd_src_vec.begin(), fd_src_vec.end(), 100);
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), fd_buf, fd_src_vec);
        Finish(mesh_device->mesh_command_queue());

        for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device->mesh_command_queue(), dst, fd_buf, coord);
            EXPECT_EQ(dst, fd_src_vec) << "Cycle " << cycle << ": sharded L1 readback failed in FD mode at " << coord;
        }
        assert_incrementing("FD steady-state");

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
        assert_incrementing("after FD terminate");
        CoreCoord grid = device->compute_with_storage_grid_size();
        // Run random workloads to stress the dispatch path and dirty compute state.
        auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(num_programs, grid, 0);
        for (uint32_t i = 0; i < num_programs; i++) {
            std::cout << "creating program " << i << '\n';
            auto random_workload = std::make_shared<MeshWorkload>();
            random_workload->add_program(
                MeshCoordinateRange(
                    MeshCoordinate{0, 0}, MeshCoordinate{mesh_device->num_rows() - 1, mesh_device->num_cols() - 1}),
                std::move(*programs[i]));
            EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *random_workload, true);
        }
        std::cout << "entering finish SD programs " << '\n';
        Finish(mesh_device->mesh_command_queue());
        std::cout << "Done" << '\n';

        // Phase 4: stop kernel, wait via service_done_addr
        const uint32_t one = 1;
        cluster.write_core(&one, sizeof(uint32_t), phys_cxy, stop_addr);

        constexpr int kTimeoutMs = 5000;
        constexpr int kPollMs = 1;
        for (int elapsed = 0; elapsed < kTimeoutMs; elapsed += kPollMs) {
            uint32_t done = 0;
            cluster.read_core(&done, sizeof(uint32_t), phys_cxy, service_done_addr);
            if (done) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
            ASSERT_LT(elapsed + kPollMs, kTimeoutMs) << "Service kernel stop timed out on cycle " << cycle;
        }

        experimental::service::ServiceCoreClaims::get().release(device, free);
    }
}

// Launch a persistent service kernel on a claimed FD-idle core while simultaneously
// running standard FD workloads on the normal worker grid. Verifies both paths are
// independent: the service counter keeps incrementing while FD programs dispatch.
TEST(DispatchContext, FDWorkloadAndServiceKernelConcurrent) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (!rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test requires Fast Dispatch mode.";
    }
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Service-core APIs require BH or UBB Galaxy.";
    }

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    auto claimable = experimental::service::ServiceCoreClaims::get().get_claimable_cores(device);
    ASSERT_FALSE(claimable.empty()) << "No claimable service cores available";
    experimental::service::ServiceCoreClaims::get().claim(device, claimable);

    // Allocate L1 communication words on each service core.
    struct CoreAddrs {
        DeviceAddr stop, counter, service_done;
    };
    std::unordered_map<CoreCoord, CoreAddrs> core_addrs;
    for (const auto& core : claimable) {
        core_addrs[core] = {
            .stop = experimental::service::ServiceCoreClaims::get().allocate_l1(device, core, sizeof(uint32_t)),
            .counter = experimental::service::ServiceCoreClaims::get().allocate_l1(device, core, sizeof(uint32_t)),
            .service_done = experimental::service::ServiceCoreClaims::get().allocate_l1(device, core, sizeof(uint32_t)),
        };
        const uint32_t zero = 0;
        CoreCoord phys = device->worker_core_from_logical_core(core);
        tt_cxy_pair cxy(device->id(), phys.x, phys.y);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].stop);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].counter);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].service_done);
    }

    // Build one program targeting all service cores with per-core runtime args.
    {
        CoreRangeSet svc_cores(claimable);
        Program svc_prog;
        auto kernel = CreateKernel(
            svc_prog,
            "tests/tt_metal/tt_metal/test_kernels/misc/service_counter.cpp",
            svc_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        for (const auto& core : claimable) {
            SetRuntimeArgs(
                svc_prog,
                kernel,
                core,
                {(uint32_t)core_addrs[core].stop,
                 (uint32_t)core_addrs[core].counter,
                 (uint32_t)core_addrs[core].service_done});
        }
        auto svc_workload = std::make_shared<MeshWorkload>();
        svc_workload->add_program(MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}), std::move(svc_prog));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *svc_workload, false);
    }

    // Verify all service cores are running by checking one representative counter.
    const CoreCoord rep_core = claimable[0];
    CoreCoord rep_phys = device->worker_core_from_logical_core(rep_core);
    tt_cxy_pair rep_cxy(device->id(), rep_phys.x, rep_phys.y);

    auto read_counter = [&]() -> uint32_t {
        uint32_t val = 0;
        cluster.read_core(&val, sizeof(uint32_t), rep_cxy, core_addrs[rep_core].counter);
        return val;
    };
    auto assert_incrementing = [&](const std::string& label) {
        uint32_t a = read_counter();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        uint32_t b = read_counter();
        EXPECT_GT(b, a) << label << ": counter stalled (" << a << " -> " << b << ")";
    };

    assert_incrementing("after service launch");

    // Launch standard 5x5 FD workloads on the regular worker grid concurrently.
    const uint32_t num_programs = 5;
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(num_programs, CoreCoord{5, 5}, 0);
    for (uint32_t i = 0; i < num_programs; i++) {
        auto workload = std::make_shared<MeshWorkload>();
        workload->add_program(MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}), std::move(*programs[i]));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false);
        assert_incrementing("during FD workload " + std::to_string(i));
    }
    Finish(mesh_device->mesh_command_queue());
    assert_incrementing("after all FD workloads complete");

    // Stop all service kernels and wait for clean exit on each.
    const uint32_t one = 1;
    for (const auto& core : claimable) {
        CoreCoord phys = device->worker_core_from_logical_core(core);
        tt_cxy_pair cxy(device->id(), phys.x, phys.y);
        cluster.write_core(&one, sizeof(uint32_t), cxy, core_addrs[core].stop);
    }
    constexpr int kTimeoutMs = 5000;
    constexpr int kPollMs = 1;
    for (const auto& core : claimable) {
        CoreCoord phys = device->worker_core_from_logical_core(core);
        tt_cxy_pair cxy(device->id(), phys.x, phys.y);
        for (int elapsed = 0; elapsed < kTimeoutMs; elapsed += kPollMs) {
            uint32_t done = 0;
            cluster.read_core(&done, sizeof(uint32_t), cxy, core_addrs[core].service_done);
            if (done) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
            ASSERT_LT(elapsed + kPollMs, kTimeoutMs)
                << "Service kernel stop timed out on core (" << core.x << "," << core.y << ")";
        }
    }

    experimental::service::ServiceCoreClaims::get().release(device, claimable);
}

// Verify allocator correctness: alignment, non-overlap, bytes_available accounting, and deallocation.
TEST(DispatchContext, ServiceCoreAllocatorCorrectness) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (!rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test requires Fast Dispatch mode.";
    }
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Service-core APIs require BH or UBB Galaxy.";
    }

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    auto claimable = experimental::service::ServiceCoreClaims::get().get_claimable_cores(device);
    ASSERT_FALSE(claimable.empty());
    CoreCoord core = claimable[0];
    experimental::service::ServiceCoreClaims::get().claim(device, {core});

    const DeviceAddr dram_align = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    // Allocate several buffers of varying sizes and verify properties.
    const std::vector<size_t> sizes = {64, 128, 32, 256};
    std::vector<DeviceAddr> addrs;
    size_t initial_available = experimental::service::ServiceCoreClaims::get().bytes_available(device, core);

    for (size_t sz : sizes) {
        DeviceAddr addr = experimental::service::ServiceCoreClaims::get().allocate_l1(device, core, sz);
        EXPECT_EQ(addr % dram_align, 0u) << "Allocation not DRAM-aligned";
        addrs.push_back(addr);
    }

    // No pair of allocations should overlap.
    for (size_t i = 0; i < addrs.size(); i++) {
        for (size_t j = i + 1; j < addrs.size(); j++) {
            DeviceAddr a_start = addrs[i], a_end = addrs[i] + sizes[i];
            DeviceAddr b_start = addrs[j], b_end = addrs[j] + sizes[j];
            EXPECT_TRUE(a_end <= b_start || b_end <= a_start) << "Allocations " << i << " and " << j << " overlap";
        }
    }

    // bytes_available should have decreased.
    size_t after_alloc = experimental::service::ServiceCoreClaims::get().bytes_available(device, core);
    EXPECT_LT(after_alloc, initial_available);

    // Deallocate one allocation and verify space is recovered.
    experimental::service::ServiceCoreClaims::get().deallocate_l1(device, core, addrs[1]);
    size_t after_dealloc = experimental::service::ServiceCoreClaims::get().bytes_available(device, core);
    EXPECT_GT(after_dealloc, after_alloc);

    // OOM: requesting more than the full L1 space must TT_FATAL.
    constexpr size_t k1p5MB = 1536 * 1024;
    EXPECT_THROW(experimental::service::ServiceCoreClaims::get().allocate_l1(device, core, k1p5MB), std::runtime_error);

    experimental::service::ServiceCoreClaims::get().release(device, {core});
}

}  // namespace tt::tt_metal::distributed::test
