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
#include <internal/service/service_core_manager.hpp>
#include "impl/internal/service/service_core_manager_impl.hpp"
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

    const CoreCoord sd_grid = device->compute_with_storage_grid_size();

    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
    const CoreCoord fd_grid = device->compute_with_storage_grid_size();
    ASSERT_LT(fd_grid.x * fd_grid.y, sd_grid.x * sd_grid.y)
        << "FD grid must be smaller than SD grid (dispatch column excluded by YAML)";

    const auto free_cores = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    ASSERT_FALSE(free_cores.empty()) << "Expected at least one free dispatch core";

    MetalContext::instance().get_service_core_manager().claim(device, free_cores);

    // Double-claim must fail; the claim guard is independent of the pool filter.
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().claim(device, free_cores), std::exception);

    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());

    EXPECT_TRUE(MetalContext::instance().get_service_core_manager().get_claimable_cores(device).empty())
        << "All free dispatch cores were claimed; none should be available in FD";

    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

    const CoreCoord capped_grid = device->compute_with_storage_grid_size();
    EXPECT_EQ(capped_grid.x, fd_grid.x) << "SD grid x must be capped to FD grid x when service cores are claimed";
    EXPECT_EQ(capped_grid.y, fd_grid.y) << "SD grid y must be capped to FD grid y when service cores are claimed";

    MetalContext::instance().get_service_core_manager().release(device, free_cores);
    MetalContext::instance().get_service_core_manager().impl().on_device_close(device->id());
}

// A persistent service kernel must keep running across repeated FD init/teardown cycles.
// The kernel increments an L1 counter until a stop flag is set; we check it keeps
// incrementing after each FD lifecycle transition.
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
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());

        auto free = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
        ASSERT_FALSE(free.empty()) << "No free dispatch cores on cycle " << cycle;
        CoreCoord svc_core = free[free.size() / 2];

        MetalContext::instance().get_service_core_manager().claim(device, free);
        DeviceAddr stop_addr = MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr counter_addr = MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr service_done_addr =
            MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

        CoreCoord phys_core = device->worker_core_from_logical_core(svc_core);
        tt_cxy_pair phys_cxy(device->id(), phys_core.x, phys_core.y);
        const uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, stop_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, counter_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, service_done_addr);

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

        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
        assert_incrementing("after FD init");

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
            auto random_workload = std::make_shared<MeshWorkload>();
            random_workload->add_program(
                MeshCoordinateRange(
                    MeshCoordinate{0, 0}, MeshCoordinate{mesh_device->num_rows() - 1, mesh_device->num_cols() - 1}),
                std::move(*programs[i]));
            EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *random_workload, true);
        }
        Finish(mesh_device->mesh_command_queue());

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

        MetalContext::instance().get_service_core_manager().release(device, free);
    }
}

// Run a persistent service kernel on a claimed FD-idle core concurrently with standard FD workloads.
TEST_F(DispatchContextFixture, FDWorkloadAndServiceKernelConcurrent) {
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

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    ASSERT_FALSE(claimable.empty()) << "No claimable service cores available";
    MetalContext::instance().get_service_core_manager().claim(device, claimable);

    struct CoreAddrs {
        DeviceAddr stop, counter, service_done;
    };
    std::unordered_map<CoreCoord, CoreAddrs> core_addrs;
    for (const auto& core : claimable) {
        core_addrs[core] = {
            .stop = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
            .counter = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
            .service_done = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
        };
        const uint32_t zero = 0;
        CoreCoord phys = device->worker_core_from_logical_core(core);
        tt_cxy_pair cxy(device->id(), phys.x, phys.y);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].stop);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].counter);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].service_done);
    }

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

    // One core's counter is representative of all service cores.
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

    // Run FD workloads on the regular worker grid concurrently with the service kernels.
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

    MetalContext::instance().get_service_core_manager().release(device, claimable);
}

// Verify allocator correctness: alignment, non-overlap, bytes_available accounting, and deallocation.
TEST_F(DispatchContextFixture, ServiceCoreAllocatorCorrectness) {
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

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    ASSERT_FALSE(claimable.empty());
    CoreCoord core = claimable[0];
    MetalContext::instance().get_service_core_manager().claim(device, {core});

    const DeviceAddr dram_align = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    const std::vector<size_t> sizes = {64, 128, 32, 256};
    std::vector<DeviceAddr> addrs;
    size_t initial_available = MetalContext::instance().get_service_core_manager().bytes_available(device, core);

    for (size_t sz : sizes) {
        DeviceAddr addr = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sz);
        EXPECT_EQ(addr % dram_align, 0u) << "Allocation not DRAM-aligned";
        addrs.push_back(addr);
    }

    for (size_t i = 0; i < addrs.size(); i++) {
        for (size_t j = i + 1; j < addrs.size(); j++) {
            DeviceAddr a_start = addrs[i], a_end = addrs[i] + sizes[i];
            DeviceAddr b_start = addrs[j], b_end = addrs[j] + sizes[j];
            EXPECT_TRUE(a_end <= b_start || b_end <= a_start) << "Allocations " << i << " and " << j << " overlap";
        }
    }

    size_t after_alloc = MetalContext::instance().get_service_core_manager().bytes_available(device, core);
    EXPECT_LT(after_alloc, initial_available);

    MetalContext::instance().get_service_core_manager().deallocate_l1(device, core, addrs[1]);
    size_t after_dealloc = MetalContext::instance().get_service_core_manager().bytes_available(device, core);
    EXPECT_GT(after_dealloc, after_alloc);

    // Requesting more than the full L1 space must TT_FATAL.
    constexpr size_t k1p5MB = 1536 * 1024;
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().allocate_l1(device, core, k1p5MB), std::runtime_error);

    MetalContext::instance().get_service_core_manager().release(device, {core});
}

}  // namespace tt::tt_metal::distributed::test
