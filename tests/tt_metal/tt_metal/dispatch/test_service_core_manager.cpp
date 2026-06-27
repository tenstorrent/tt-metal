// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
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
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/llrt.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal::distributed::test {

static void assert_counter_incrementing(const std::function<uint32_t()>& read_fn, const std::string& label) {
    uint32_t a = read_fn();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    uint32_t b = read_fn();
    EXPECT_GT(b, a) << label << ": counter stalled (" << a << " -> " << b << ")";
}

// Builds a trivial workload to exercise the FD/SD dispatch pipeline
static std::shared_ptr<MeshWorkload> make_trivial_workload(IDevice* device) {
    const uint32_t l1_scratch = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    Program prog;
    CreateKernel(
        prog,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_scratch, 0u}});
    auto workload = std::make_shared<MeshWorkload>();
    workload->add_program(MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}), std::move(prog));
    return workload;
}

// Base fixture: arch guard + dispatch context cleanup on top of MeshDispatchFixture.
class ServiceCoreFixture : public tt::tt_metal::MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        const auto& cluster = MetalContext::instance().get_cluster();
        if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Service-core APIs require BH or UBB Galaxy.";
        }
    }
    void TearDown() override {
        experimental::DispatchContext::get().reset();
        MeshDispatchFixture::TearDown();
    }
};

// For tests that require Slow Dispatch mode.
class ServiceCoreSdFixture : public ServiceCoreFixture {
protected:
    void SetUp() override {
        ServiceCoreFixture::SetUp();
        if (!this->slow_dispatch_) {
            GTEST_SKIP() << "This test requires Slow Dispatch mode.";
        }
    }
};

// For tests that require Fast Dispatch mode.
class ServiceCoreFdFixture : public ServiceCoreFixture {
protected:
    void SetUp() override {
        ServiceCoreFixture::SetUp();
        if (this->slow_dispatch_) {
            GTEST_SKIP() << "This test requires Fast Dispatch mode.";
        }
    }
};

// Sanity test: Verify TT_FATAL fires on every documented misuse path:
//   1. get_claimable_cores / claim before FD is active
//   2. allocate_l1 / deallocate_l1 / bytes_available on an unclaimed core
TEST_F(ServiceCoreSdFixture, ServiceCoreFatalGuards) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    // Before FD is active, both query and claim must fatal
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().get_claimable_cores(device), std::exception);
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().claim(device, {{0, 0}}), std::exception);

    // Bring FD up and claim one core so we can test the unclaimed core guards
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
    auto free = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    CoreCoord svc_core = free[0];
    MetalContext::instance().get_service_core_manager().claim(device, {svc_core});

    // {0,0} is a worker-grid core. Never a service core — so all three guards must fatal
    CoreCoord unclaimed{0, 0};
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().allocate_l1(device, unclaimed, 4), std::exception);
    EXPECT_THROW(
        MetalContext::instance().get_service_core_manager().deallocate_l1(device, unclaimed, 0), std::exception);
    EXPECT_THROW(
        MetalContext::instance().get_service_core_manager().bytes_available(device, unclaimed), std::exception);

    MetalContext::instance().get_service_core_manager().release(device, {svc_core});
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
}

// Sanity test: Verify that claiming all free FD dispatch cores as service cores:
//   1. caps compute_with_storage_grid_size() to the FD grid in SD mode so we don't perturb service cores
//   2. leaves no idle dispatch cores when FD is re-initialized.
TEST_F(ServiceCoreSdFixture, ServiceCoreGridCap) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    // Phase 1: record full SD grid (no services active)
    const CoreCoord sd_grid = device->compute_with_storage_grid_size();

    // Phase 2: init FD, record FD grid, claim all free dispatch cores
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
    const CoreCoord fd_grid = device->compute_with_storage_grid_size();
    ASSERT_LT(fd_grid.x * fd_grid.y, sd_grid.x * sd_grid.y)
        << "FD grid must be smaller than SD grid (dispatch column excluded by YAML)";

    const auto free_cores = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    MetalContext::instance().get_service_core_manager().claim(device, free_cores);

    // Attempting to claim the same cores again must fail
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().claim(device, free_cores), std::exception);

    // Phase 3: terminate + re-init FD must exclude all claimed cores
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());

    // All cores claimed — get_claimable_cores() must fatal rather than return an empty vector
    EXPECT_THROW(MetalContext::instance().get_service_core_manager().get_claimable_cores(device), std::exception);

    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

    // Phase 4: back in SD — grid must be capped to the FD grid so we don't disturb service cores
    const CoreCoord capped_grid = device->compute_with_storage_grid_size();
    EXPECT_EQ(capped_grid.x, fd_grid.x) << "SD grid x must be capped to FD grid x when service cores are claimed";
    EXPECT_EQ(capped_grid.y, fd_grid.y) << "SD grid y must be capped to FD grid y when service cores are claimed";

    // Phase 5: release service cores and verify the full SD grid is restored
    MetalContext::instance().get_service_core_manager().release(device, free_cores);

    const CoreCoord restored_grid = device->compute_with_storage_grid_size();
    EXPECT_EQ(restored_grid.x, sd_grid.x) << "SD grid x must be restored to original after release";
    EXPECT_EQ(restored_grid.y, sd_grid.y) << "SD grid y must be restored to original after release";

    MetalContext::instance().get_service_core_manager().impl().on_device_close(device->id());
}

// Test a Persistent kernel on FD cores can survive multiple FD init + teardown cycles
// Kernel increments a counter in L1 until a stop flag is set
// We verify the counter is still incrementing after each FD lifecycle transition
TEST_F(ServiceCoreSdFixture, PersistentServiceMultiCycle) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));
    const auto& cluster = MetalContext::instance().get_cluster();

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const uint32_t num_tiles = 64;

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

        auto free = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
        CoreCoord svc_core = free[free.size() / 2];

        MetalContext::instance().get_service_core_manager().claim(device, free);
        DeviceAddr stop_addr =
            MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr counter_addr =
            MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));
        DeviceAddr service_done_addr =
            MetalContext::instance().get_service_core_manager().allocate_l1(device, svc_core, sizeof(uint32_t));

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());

        // Initialise all three words to 0.
        CoreCoord phys_core = device->worker_core_from_logical_core(svc_core);
        tt_cxy_pair phys_cxy(device->id(), phys_core.x, phys_core.y);
        const uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, stop_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, counter_addr);
        cluster.write_core(&zero, sizeof(uint32_t), phys_cxy, service_done_addr);

        // Phase 2: launch persistent kernel (SD, non-blocking).
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
        assert_counter_incrementing(read_counter, "SD baseline (cycle " + std::to_string(cycle) + ")");

        // Phase 3: FD re-init
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device.get());
        assert_counter_incrementing(read_counter, "after FD init (cycle " + std::to_string(cycle) + ")");

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
        assert_counter_incrementing(read_counter, "FD steady-state (cycle " + std::to_string(cycle) + ")");

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device.get());
        assert_counter_incrementing(read_counter, "after FD terminate (cycle " + std::to_string(cycle) + ")");

        // Dispatch a trivial workload to stress the dispatch path
        auto stressor = make_trivial_workload(device);
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *stressor, true);
        Finish(mesh_device->mesh_command_queue());

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

        MetalContext::instance().get_service_core_manager().release(device, free);
    }
}

// Per-core Allocator sanity check: verify alignment, non-overlap, bytes_available accounting, and deallocation
TEST_F(ServiceCoreFdFixture, ServiceCoreAllocatorCorrectness) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
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

    // No pair of allocations should overlap.
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

    // OOM: requesting more than the full L1 space must TT_FATAL.
    constexpr size_t k1p5MB = 1536 * 1024;
    EXPECT_THROW(
        MetalContext::instance().get_service_core_manager().allocate_l1(device, core, k1p5MB), std::runtime_error);

    MetalContext::instance().get_service_core_manager().release(device, {core});
}

// Actual use case: Launch a persistent service kernel on a claimed FD idle core while simultaneously
// running standard FD workloads on the normal worker grid. Verifies both paths are
// independent. Needed for prefill in Blitz.
TEST_F(ServiceCoreFdFixture, FDWorkloadAndServiceKernelConcurrent) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));
    const auto& cluster = MetalContext::instance().get_cluster();

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    MetalContext::instance().get_service_core_manager().claim(device, claimable);

    // Allocate L1 and zero-init communication words on each service core.
    struct CoreAddrs {
        DeviceAddr stop, counter, service_done;
    };
    std::unordered_map<CoreCoord, CoreAddrs> core_addrs;
    for (const auto& core : claimable) {
        core_addrs[core] = {
            .stop = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
            .counter = MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
            .service_done =
                MetalContext::instance().get_service_core_manager().allocate_l1(device, core, sizeof(uint32_t)),
        };
        const uint32_t zero = 0;
        CoreCoord phys = device->worker_core_from_logical_core(core);
        tt_cxy_pair cxy(device->id(), phys.x, phys.y);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].stop);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].counter);
        cluster.write_core(&zero, sizeof(uint32_t), cxy, core_addrs[core].service_done);
    }

    // Build one program targeting all service cores with per-core runtime args.
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

    // Launch-once: re-enqueuing the same (already-launched) service workload must TT_FATAL.
    EXPECT_THROW(EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *svc_workload, false), std::exception);

    // Pick a single representative service workload core for sanity checks
    const CoreCoord rep_core = claimable[0];
    CoreCoord rep_phys = device->worker_core_from_logical_core(rep_core);
    tt_cxy_pair rep_cxy(device->id(), rep_phys.x, rep_phys.y);

    auto read_counter = [&]() -> uint32_t {
        uint32_t val = 0;
        cluster.read_core(&val, sizeof(uint32_t), rep_cxy, core_addrs[rep_core].counter);
        return val;
    };
    assert_counter_incrementing(read_counter, "after service launch");

    // Launch FD workloads on the regular worker grid while the service kernel runs.
    auto fd_workload = make_trivial_workload(device);
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *fd_workload, false);
    assert_counter_incrementing(read_counter, "during FD workload");
    Finish(mesh_device->mesh_command_queue());
    assert_counter_incrementing(read_counter, "after all FD workloads complete");

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

    MetalContext::instance().get_service_core_manager().release(device, claimable);
}

// Launch-once contract (service_core_manager.hpp): a claimed service core accepts one service enqueue;
// a second TT_FATALs until the core is released and re-claimed, after which a fresh enqueue succeeds.
TEST_F(ServiceCoreFdFixture, ServiceWorkloadReenqueueRequiresRelease) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    ASSERT_GE(claimable.size(), 1u) << "Need at least one claimable service core for this test.";
    const CoreCoord svc_core = claimable[0];

    auto build_service_workload = [&]() {
        Program prog;
        CreateKernel(
            prog,
            "tt_metal/kernels/dataflow/blank.cpp",
            CoreRangeSet(CoreRange(svc_core)),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto workload = std::make_shared<MeshWorkload>();
        workload->add_program(MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}), std::move(prog));
        return workload;
    };

    MetalContext::instance().get_service_core_manager().claim(device, {svc_core});

    auto workload = build_service_workload();
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false);

    // Re-enqueue without releasing: launch-once TT_FATALs.
    EXPECT_THROW(EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false), std::exception);

    // Release and re-claim, then enqueue a fresh workload: now allowed (the core's launched flag was
    // cleared by release, and a new workload classifies from scratch).
    MetalContext::instance().get_service_core_manager().release(device, {svc_core});
    MetalContext::instance().get_service_core_manager().claim(device, {svc_core});
    auto workload2 = build_service_workload();
    EXPECT_NO_THROW(EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload2, false));

    MetalContext::instance().get_service_core_manager().release(device, {svc_core});
}

// No-mixing contract (service_core_manager.hpp): a single program must target only claimed service
// cores or only worker-grid cores, never both. EnqueueMeshWorkload TT_FATALs on a program that spans
// a claimed service core and an unclaimed worker core.
TEST_F(ServiceCoreFdFixture, ServiceWorkloadMixingFatal) {
    auto& mesh_device = this->devices_[0];
    IDevice* device = mesh_device->get_device(MeshCoordinate(0, 0));

    auto claimable = MetalContext::instance().get_service_core_manager().get_claimable_cores(device);
    ASSERT_GE(claimable.size(), 1u) << "Need at least one claimable service core for this test.";
    const CoreCoord svc_core = claimable[0];
    const CoreCoord worker_core{0, 0};  // worker-grid core, never a dispatch-column service core
    ASSERT_NE(svc_core, worker_core);
    MetalContext::instance().get_service_core_manager().claim(device, {svc_core});

    // One program spanning the claimed service core and an unclaimed worker core -> mixed -> TT_FATAL.
    Program prog;
    CreateKernel(
        prog,
        "tt_metal/kernels/dataflow/blank.cpp",
        CoreRangeSet(std::vector<CoreCoord>{svc_core, worker_core}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto workload = std::make_shared<MeshWorkload>();
    workload->add_program(MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}), std::move(prog));
    EXPECT_THROW(EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false), std::exception);

    MetalContext::instance().get_service_core_manager().release(device, {svc_core});
}

}  // namespace tt::tt_metal::distributed::test
