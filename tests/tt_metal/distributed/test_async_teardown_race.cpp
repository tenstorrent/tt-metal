// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Targeted tests for the async-dispatch + device-teardown race condition.
//
// Background (see AI-JOURNAL.md for full root cause analysis):
//   When EnqueueMeshWorkload is called with blocking=false, the workload may still
//   be executing on device when the MeshDevice destructor tears down ETH dispatch
//   infrastructure. On T3K, this leaves stale ERISC firmware on non-MMIO chips
//   (devices 1,3,4,5) that corrupts subsequent re-initialization.
//
// Why the existing AsyncExecutionWorksCQ0 test is insufficient:
//   1. It requires a killed predecessor test to leave stale ETH state — cannot
//      self-contain the dirty setup, making CI runs noisy and non-deterministic.
//   2. It uses heavy CCL AllGather ops that have their own flakiness, conflating
//      the race-condition signal with unrelated CCL bugs.
//   3. The only failure signal is a 5-minute hang (exit=124), giving zero
//      diagnostic information about which component failed.
//   4. It cannot distinguish between: (a) our async teardown race, (b) dirty
//      device state from predecessor, (c) CCL op flakiness.
//
// These tests isolate the specific race condition using lightweight void kernels
// and provide clear pass/fail signal in under 60 seconds.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture: minimal mesh device with a 30-second watchdog.
// Uses the system mesh shape (adapts to N300 / T3K / single-chip).
// No fabric config — we are testing the dispatch + teardown path, not CCL.
// ---------------------------------------------------------------------------
class AsyncTeardownRaceFixture : public MeshDeviceFixtureBase {
protected:
    AsyncTeardownRaceFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .test_budget_ms = 30000,  // 30s hard kill — never wait 5 minutes
          }) {}
};

// Helper: create a Program with 3 blank kernels (BRISC, NCRISC, compute) on a
// small core range. This is the lightest possible workload that exercises the
// full dispatch path (compile, upload binary to L1, launch, signal completion).
static Program create_blank_program(const CoreRange& cores) {
    Program program;

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cores, ComputeConfig{});

    return program;
}

// ---------------------------------------------------------------------------
// Scenario A: Basic async dispatch + immediate teardown + clean re-init.
//
// Steps:
//   1. Open mesh device
//   2. Dispatch a blank workload with blocking=false
//   3. Immediately close the mesh device (without calling Finish)
//   4. Re-open the mesh device
//   5. Dispatch + Finish a second workload (blocking=true)
//   6. Verify no hang, no crash, clean completion
//
// What this tests:
//   - The device teardown path correctly waits for (or cancels) in-flight
//     async workloads before tearing down ETH dispatch cores.
//   - Re-initialization after an async-interrupted session succeeds without
//     hanging on stale ERISC firmware.
//
// Pass = completes in <30s. Fail = hang (watchdog kills at 30s) or crash.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownRaceFixture, AsyncDispatchThenImmediateTeardown) {
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    // Phase 1: async dispatch without waiting
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));

        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario A] Dispatching blank workload (blocking=false)");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        // Intentionally NOT calling Finish() — this is the race condition trigger.
    }

    // Phase 2: close the mesh device immediately
    log_info(tt::LogTest, "[Scenario A] Closing mesh device without waiting for async workload");
    auto mesh_shape = mesh_device_->shape();
    mesh_device_->close();
    mesh_device_.reset();

    // Phase 3: re-open the mesh device
    log_info(tt::LogTest, "[Scenario A] Re-opening mesh device");
    mesh_device_ = MeshDevice::create(
        MeshDeviceConfig(mesh_shape),
        config_.l1_small_size,
        config_.trace_region_size,
        config_.num_cqs,
        DispatchCoreConfig{},
        {},
        config_.worker_l1_size);

    // Phase 4: dispatch a blocking workload to verify clean state
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        auto new_device_range = MeshCoordinateRange(mesh_device_->shape());
        workload.add_program(new_device_range, std::move(program));

        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario A] Dispatching verification workload (blocking=true)");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        log_info(tt::LogTest, "[Scenario A] Verification workload completed — device re-init is clean");
    }

    // Phase 5: buffer round-trip to verify data path integrity
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;  // 1 KB page
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto shard_shape = Shape2D{1, 1};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = shard_shape};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());

        // Write a known pattern
        std::vector<uint32_t> src(page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols());
        for (size_t i = 0; i < src.size(); i++) {
            src[i] = static_cast<uint32_t>(0xDEAD0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);

        // Read it back
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);

        ASSERT_EQ(dst.size(), src.size()) << "Buffer readback size mismatch after device re-init";
        for (size_t i = 0; i < src.size(); i++) {
            ASSERT_EQ(dst[i], src[i])
                << "Buffer corruption at index " << i << " after async teardown + re-init";
        }
        log_info(tt::LogTest, "[Scenario A] Buffer round-trip verified — no data corruption");
    }
}

// ---------------------------------------------------------------------------
// Scenario B: Multiple rapid open/close cycles with async dispatch.
//
// Stress test: repeat the async-dispatch + teardown + reopen cycle N times.
// This amplifies any cumulative state leaks (e.g., un-freed ETH channels,
// ERISC firmware refcount issues, L1 corruption from stale NOC writes).
//
// Pass = all iterations complete in <30s. Fail = hang or crash on any iteration.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownRaceFixture, RepeatedAsyncTeardownCycles) {
    constexpr int kCycles = 3;
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        log_info(tt::LogTest, "[Scenario B] Cycle {}/{}", cycle + 1, kCycles);

        // Dispatch async
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }

        // Tear down
        auto mesh_shape = mesh_device_->shape();
        mesh_device_->close();
        mesh_device_.reset();

        // Re-open
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            DispatchCoreConfig{},
            {},
            config_.worker_l1_size);
    }

    // Final verification: blocking dispatch must succeed
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        Finish(cq);
        log_info(tt::LogTest, "[Scenario B] All {} cycles completed — no cumulative state corruption", kCycles);
    }
}

// ---------------------------------------------------------------------------
// Scenario C: Async dispatch on multiple command queues + immediate teardown.
//
// Requires 2 CQs. Dispatches async on both CQ0 and CQ1 simultaneously,
// then tears down without waiting. Re-opens and verifies clean state.
// This targets the multi-CQ variant of the race (the same path that
// AsyncExecutionWorksCQ0CQ1 exercises, but without the heavy CCL ops).
//
// If the system only has 1 CQ, skips gracefully.
// ---------------------------------------------------------------------------
class AsyncTeardownMultiCQFixture : public MeshDeviceFixtureBase {
protected:
    AsyncTeardownMultiCQFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 2,
              .test_budget_ms = 30000,
          }) {}
};

TEST_F(AsyncTeardownMultiCQFixture, MultiCQAsyncDispatchThenTeardown) {
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    // Dispatch async on CQ0
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq0 = mesh_device_->mesh_command_queue(0);
        log_info(tt::LogTest, "[Scenario C] Dispatching on CQ0 (blocking=false)");
        EnqueueMeshWorkload(cq0, workload, /*blocking=*/false);
    }

    // Dispatch async on CQ1
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq1 = mesh_device_->mesh_command_queue(1);
        log_info(tt::LogTest, "[Scenario C] Dispatching on CQ1 (blocking=false)");
        EnqueueMeshWorkload(cq1, workload, /*blocking=*/false);
    }

    // Tear down without waiting
    log_info(tt::LogTest, "[Scenario C] Closing mesh device with both CQs having in-flight work");
    auto mesh_shape = mesh_device_->shape();
    mesh_device_->close();
    mesh_device_.reset();

    // Re-open with the same dispatch core type that SetUp() would have chosen.
    // MeshDeviceFixtureBase::SetUp() uses ETH dispatch for >=2 CQs on T3K/N300.
    // We replicate that logic here inline to avoid depending on ClusterType enum.
    log_info(tt::LogTest, "[Scenario C] Re-opening mesh device");
    {
        auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
        bool need_eth = (config_.num_cqs >= 2) &&
                        (cluster_type == tt::tt_metal::ClusterType::T3K ||
                         cluster_type == tt::tt_metal::ClusterType::N300);
        auto core_type = need_eth ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            core_type,
            {},
            config_.worker_l1_size);
    }

    // Verify clean state on both CQs
    for (uint8_t cq_id = 0; cq_id < 2; cq_id++) {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        auto new_range = MeshCoordinateRange(mesh_device_->shape());
        workload.add_program(new_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue(cq_id);
        log_info(tt::LogTest, "[Scenario C] Verifying CQ{} with blocking dispatch", cq_id);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        Finish(cq);
    }
    log_info(tt::LogTest, "[Scenario C] Both CQs clean after multi-CQ async teardown");
}

// ---------------------------------------------------------------------------
// Fixture: async dispatch with FABRIC_2D active, to test the ETH-router
// teardown poll ("Fix B" in FabricFirmwareInitializer::teardown()).
//
// Why FabricConfig::DISABLED is insufficient for Scenarios A/B/C:
//   MeshDevice::close() calls FabricFirmwareInitializer::teardown(). With
//   DISABLED config, teardown() exits early (lines 84-90 of
//   fabric_firmware_initializer.cpp) before reaching the ETH-router
//   TERMINATED poll or the Tensix MUX termination logic. So Scenarios A/B/C
//   never exercise the code paths that matter for the CI failures (which all
//   use FABRIC_2D via unit_tests_ttnn_ccl_ops as predecessor).
//
// This fixture uses FABRIC_2D so teardown() exercises:
//   1. Tensix MUX TERMINATED poll (with force-reset on timeout)
//   2. Master ETH router TERMINATED poll (with force-reset on timeout)
//   3. compile_and_configure_fabric() → terminate_stale_erisc_routers() on re-init
//   4. wait_for_fabric_router_sync() — full ERISC handshake verification
//
// Requires >= 2 devices (FABRIC_2D requires a multi-chip topology).
// Skips gracefully on single-chip systems.
// ---------------------------------------------------------------------------
class AsyncTeardownFabric2DFixture : public MeshDeviceFixtureBase {
protected:
    AsyncTeardownFabric2DFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 30000,
          }) {}

    void SetUp() override {
        // FABRIC_2D requires >= 2 devices; skip gracefully on single-chip CI runners.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "AsyncTeardownFabric2DFixture requires >= 2 devices (FABRIC_2D)";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Scenario D: async dispatch with FABRIC_2D active → immediate close → re-init.
//
// Exercises code paths that Scenarios A/B/C (FabricConfig::DISABLED) skip:
//   - FabricFirmwareInitializer::teardown() ETH router TERMINATED poll
//   - Force-reset of master ETH router on teardown timeout
//   - terminate_stale_erisc_routers() on re-init (detects leftover EDM state)
//   - wait_for_fabric_router_sync() ERISC handshake verification
//
// This is the closest deterministic proxy for the actual CI failure:
//   predecessor unit_tests_ttnn_ccl_ops (FABRIC_2D) → async dispatch →
//   SIGKILL → stale ERISCs on non-MMIO chips → AsyncExecutionWorksCQ0 hangs.
//
// Pass = device re-opens cleanly, blocking dispatch completes in < 30s.
// Fail = hang (watchdog kills), crash during re-init, or TT_FATAL.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric2DFixture, Fabric2DAsyncDispatchThenReinit) {
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());
    auto mesh_shape = mesh_device_->shape();

    // Phase 1: async dispatch with FABRIC_2D ERISCs active.
    // No Finish() — ERISC fabric ops may still be draining when close() is called.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario D] Dispatching blank workload on FABRIC_2D mesh (blocking=false)");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    }

    // Phase 2: close mesh device — exercises FabricFirmwareInitializer::teardown():
    //   (a) Tensix MUX TERMINATED poll (skipped when FabricTensixConfig::DISABLED)
    //   (b) Master ETH router TERMINATED poll with force-reset on timeout
    log_info(tt::LogTest, "[Scenario D] Closing FABRIC_2D mesh device without waiting for async workload");
    mesh_device_->close();
    mesh_device_.reset();
    // post_teardown() resets MetalContext fabric config to DISABLED internally.

    // Phase 3: re-open with FABRIC_2D — exercises compile_and_configure_fabric()
    //   → terminate_stale_erisc_routers() on any residual ERISC state
    //   → wait_for_fabric_router_sync() full handshake verification.
    log_info(tt::LogTest, "[Scenario D] Re-opening FABRIC_2D mesh device");
    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    mesh_device_ = MeshDevice::create(
        MeshDeviceConfig(mesh_shape),
        config_.l1_small_size,
        config_.trace_region_size,
        config_.num_cqs,
        DispatchCoreConfig{},
        {},
        config_.worker_l1_size);

    // Phase 4: blocking dispatch verifies freshly loaded ERISCs are operational.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        auto new_range = MeshCoordinateRange(mesh_device_->shape());
        workload.add_program(new_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario D] Dispatching verification workload (blocking=true)");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        log_info(tt::LogTest, "[Scenario D] FABRIC_2D re-init + dispatch completed cleanly");
    }
}

// ---------------------------------------------------------------------------
// Fixture: same as AsyncTeardownFabric2DFixture but with a 90-second watchdog
// for multi-cycle tests. FABRIC_2D init + teardown per cycle takes ~10-15s on
// T3K; 2 cycles + final verification needs ~40s worst-case, which exceeds the
// 30s budget used for single-cycle tests.
// ---------------------------------------------------------------------------
class AsyncTeardownFabric2DRepeatFixture : public MeshDeviceFixtureBase {
protected:
    AsyncTeardownFabric2DRepeatFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 90000,  // 90s for multi-cycle FABRIC_2D stress
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "AsyncTeardownFabric2DRepeatFixture requires >= 2 devices (FABRIC_2D)";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Scenario E: Multiple FABRIC_2D open/async-dispatch/close cycles.
//
// Stress test for accumulated ERISC state across consecutive FABRIC_2D sessions.
// CI Iterations 3-5 showed that FABRIC_2D sessions (each using ETH channels for
// fabric routing) left stale edm_status values on ETH channels after teardown,
// causing the next session to hang during fabric bring-up. This test runs N
// back-to-back FABRIC_2D open/async-dispatch/close cycles to surface any
// session-to-session state leak.
//
// Each cycle:
//   1. Opens a FABRIC_2D mesh device (loads ERISC EDM firmware on all ETH chans)
//   2. Dispatches a blank workload (blocking=false — workload may still be in
//      flight when close() is called)
//   3. Closes the device; FabricFirmwareInitializer::teardown() sends TERMINATE
//      to master ETH routers and polls for EDMStatus::TERMINATED (5s timeout,
//      force-reset on timeout)
//   4. If not the last cycle, re-opens with FABRIC_2D for the next iteration
//
// Final cycle: re-opens FABRIC_2D, dispatches a blocking workload, and performs
// a buffer round-trip to verify no ERISC L1 corruption accumulated across
// the N sessions.
//
// Note: blank kernels do NOT actually route data through ERISC fabric channels,
// so this test checks the teardown/reinit state-machine cleanness, not fabric
// correctness. See AsyncExecutionWorksCQ0 for the full end-to-end signal.
//
// Pass = all cycles complete + buffer matches in <30s.
// Fail = hang (watchdog), crash on any re-open, or data corruption.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric2DRepeatFixture, RepeatedFabric2DTeardownCycles) {
    constexpr int kCycles = 2;
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        log_info(tt::LogTest, "[Scenario E] FABRIC_2D cycle {}/{} — async dispatch + close", cycle + 1, kCycles);
        auto mesh_shape = mesh_device_->shape();

        // Phase 1: async dispatch (no Finish) — ERISC firmware is live when close() runs.
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }

        // Phase 2: close — teardown() sends TERMINATE to master ETH routers and polls.
        mesh_device_->close();
        mesh_device_.reset();

        // Phase 3: re-open with FABRIC_2D for the next cycle or final verification.
        // SetFabricConfig must be called before MeshDevice::create; post_teardown()
        // already reset it to DISABLED so we must set it again here.
        log_info(tt::LogTest, "[Scenario E] Re-opening FABRIC_2D mesh device (cycle {})", cycle + 1);
        tt_fabric::SetFabricConfig(
            tt_fabric::FabricConfig::FABRIC_2D,
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            DispatchCoreConfig{},
            {},
            config_.worker_l1_size);
    }

    // Final verification: blocking dispatch confirms ERISC state is clean after kCycles.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario E] Final blocking dispatch after {} FABRIC_2D cycles", kCycles);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Buffer round-trip: detect DRAM/L1 corruption from stale ERISC NOC traffic.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto shard_shape = Shape2D{1, 1};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = shard_shape};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());

        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xBEEF0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);

        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);

        ASSERT_EQ(dst.size(), src.size()) << "Buffer size mismatch after " << kCycles << " FABRIC_2D cycles";
        for (size_t i = 0; i < src.size(); i++) {
            ASSERT_EQ(dst[i], src[i])
                << "Corruption at index " << i << " after " << kCycles << " FABRIC_2D async teardown cycles";
        }
        log_info(
            tt::LogTest,
            "[Scenario E] Buffer round-trip clean after {} FABRIC_2D cycles — no accumulated ERISC state corruption",
            kCycles);
    }
}

// ---------------------------------------------------------------------------
// Scenario F: Slow-kernel async teardown — guaranteed real race condition.
//
// Problem with Scenarios A/B/C: blank kernels finish in ~1µs. By the time
// host calls close() (~10µs of C++ overhead after EnqueueMeshWorkload returns),
// the kernel is already done. There is no actual race — it's a clean teardown.
//
// This test dispatches a busy-spin kernel that runs for ~8ms on device,
// which is long enough to guarantee it is still executing when close() fires.
// This creates a deterministic async-dispatch teardown race without needing
// an external SIGKILL.
//
// Pass = re-open succeeds + blocking workload completes + buffer round-trip
//        matches in < 30s.
// Fail = hang (watchdog), crash on re-open, or data corruption.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownRaceFixture, SlowKernelAsyncTeardownRace) {
    // 1M volatile iterations ≈ 8ms on WH BRISC @1.2 GHz.
    // Far longer than the ~10µs host overhead between EnqueueMeshWorkload and close().
    constexpr uint32_t kSpinIters = 1'000'000;
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    // Phase 1: dispatch slow kernel — guaranteed to be running when close() fires.
    {
        Program program;
        auto kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/busy_spin.cpp",
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, kernel_id, CoreCoord{0, 0}, {kSpinIters});

        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(
            tt::LogTest,
            "[Scenario F] Dispatching busy-spin kernel ({} iters ≈ 8ms) — will still be running at close()",
            kSpinIters);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        // NO Finish() — close() fires while kernel is still spinning.
    }

    // Phase 2: close immediately while kernel is guaranteed to still be running.
    log_info(tt::LogTest, "[Scenario F] Closing mesh device — kernel still executing (real race)");
    auto mesh_shape = mesh_device_->shape();
    mesh_device_->close();
    mesh_device_.reset();

    // Phase 3: re-open — exercises terminate_stale_erisc_routers / force-reset paths.
    log_info(tt::LogTest, "[Scenario F] Re-opening mesh device after slow-kernel async teardown");
    mesh_device_ = MeshDevice::create(
        MeshDeviceConfig(mesh_shape),
        config_.l1_small_size,
        config_.trace_region_size,
        config_.num_cqs,
        DispatchCoreConfig{},
        {},
        config_.worker_l1_size);

    // Phase 4: blocking dispatch verifies the device re-initialised cleanly.
    {
        Program program;
        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario F] Verification blocking dispatch");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Phase 5: buffer round-trip to detect data-path corruption.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};
        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xFACE0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size()) << "[Scenario F] Buffer size mismatch after slow-kernel teardown";
        for (size_t i = 0; i < src.size(); i++) {
            ASSERT_EQ(dst[i], src[i]) << "[Scenario F] Corruption at index " << i;
        }
        log_info(tt::LogTest, "[Scenario F] Buffer round-trip clean — no data corruption after guaranteed race");
    }
}

// ---------------------------------------------------------------------------
// Scenario G: Event recording on WORKER dispatch (1CQ) does not issue a
// DISPATCH_S WAIT, verifying the regression fix for commit f1caf807b7.
//
// Background:
//   Commit 9f260fb8d2 extended `issue_record_event_commands` to send a
//   DISPATCH_S WAIT for all `dispatch_s_enabled` configs — including WORKER
//   dispatch on T3K/N300 (1CQ, dispatch_s + dispatch_d on same Tensix core).
//   On WORKER dispatch, dispatch_d's CLEAR_STREAM clears the stream register
//   before dispatch_s reads it, so dispatch_s hangs forever at the WAIT.
//
//   Fix (f1caf807b7): gate the DISPATCH_S WAIT on `distributed_dispatcher`
//   (true only for ETH 1CQ, where dispatch_s and dispatch_d are on DIFFERENT
//   cores). For WORKER dispatch (DISTRIBUTED_DISPATCHER=0), no WAIT is issued.
//
// This test exercises `enqueue_record_event_to_host()` — the host-side call
// that eventually calls `issue_record_event_commands` — on a 1CQ WORKER
// dispatch mesh device. It must complete within 30s; any hang indicates the
// DISPATCH_S WAIT guard is broken.
//
// Uses `AsyncTeardownRaceFixture` (1CQ) which selects WORKER dispatch on T3K.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownRaceFixture, WorkerDispatchEventRecordingDoesNotHang) {
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());
    auto& cq = mesh_device_->mesh_command_queue();

    // Round 1: dispatch async, then record event to host.
    // enqueue_record_event_to_host() → issue_record_event_commands() on the
    // host side. On WORKER dispatch, the dispatch_s guard must NOT emit a WAIT.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        log_info(tt::LogTest, "[Scenario G] Round 1: async dispatch + enqueue_record_event_to_host");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

        // This call triggers issue_record_event_commands. Pre-fix: hangs on WORKER
        // dispatch because DISPATCH_S receives a WAIT after dispatch_d already
        // cleared the stream. Post-fix: no DISPATCH_S WAIT issued for WORKER dispatch.
        auto event = cq.enqueue_record_event_to_host();
        (void)event;  // host already waited inside enqueue_record_event_to_host
        log_info(tt::LogTest, "[Scenario G] Round 1: event completed — no DISPATCH_S hang");
    }

    // Round 2: a second independent program + event to check no residual state.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        log_info(tt::LogTest, "[Scenario G] Round 2: second async dispatch + event record");
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        auto event2 = cq.enqueue_record_event_to_host();
        (void)event2;
        log_info(tt::LogTest, "[Scenario G] Round 2: event completed — no accumulated state");
    }

    // Final: blocking dispatch + buffer round-trip to confirm clean CQ state.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};
        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xC0DE0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size()) << "[Scenario G] Buffer size mismatch";
        for (size_t i = 0; i < src.size(); i++) {
            ASSERT_EQ(dst[i], src[i]) << "[Scenario G] Data corruption at index " << i;
        }
        log_info(tt::LogTest, "[Scenario G] Buffer round-trip clean — WORKER dispatch event recording is safe");
    }
}

// ---------------------------------------------------------------------------
// Scenario H: quiesce_devices() with FABRIC_2D active — exercises Phase 2.5 ERISC termination.
//
// CRITICAL GAP FILLED: Scenarios A-G test FabricFirmwareInitializer::teardown() (device close
// path). NONE test quiesce_and_restart_fabric_workers() — the Phase 2.5 fix that stopped the
// AllGatherEthTxqTeardownRace hang.
//
// Root cause of the CI failure (AI-JOURNAL iter10):
//   AllGather iter1 passes; iter2 hangs because quiesce_and_restart_fabric_workers() Phase 3
//   calls configure_fabric_cores() which overwrites every active ERISC's L1 BEFORE the ERISC
//   has finished draining its ETH TXQ. The ERISC then runs corrupted firmware, generates
//   invalid NOC traffic at 0x880030060, and the next AllGather hangs in completion queue wait.
//
// Fix (Phase 2.5 in device.cpp): before Phase 3 (L1 overwrite), send TERMINATE to each active
// ERISC channel and poll for EDMStatus::TERMINATED (500ms timeout). Only then overwrite L1.
//
// This test verifies Phase 2.5 directly:
//   1. Open FABRIC_2D mesh device (ERISC EDM firmware running on all active channels).
//   2. Dispatch a blank workload (async) — ERISC channels are live.
//   3. Call quiesce_devices() — triggers quiesce_and_restart_fabric_workers() which MUST
//      terminate ERISC channels (Phase 2.5) before overwriting their L1 (Phase 3).
//   4. Repeat N times to amplify any race in the termination poll.
//   5. Final blocking dispatch + buffer round-trip confirm no accumulated state corruption.
//
// Pass = all quiesce cycles complete + buffer matches in <30s.
// Fail = hang (watchdog), crash in quiesce_and_restart_fabric_workers, or data corruption.
//
// Note: device stays OPEN between quiesce cycles (unlike Scenarios D/E which close + reopen).
// This mirrors the actual AllGather use-case where a single MeshDevice session runs multiple
// CCL iterations separated by quiesce_devices().
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric2DRepeatFixture, QuiesceDevicesExercisesPhase25ERISCTermination) {
    constexpr int kCycles = 3;
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        log_info(
            tt::LogTest,
            "[Scenario H] Cycle {}/{}: async dispatch + quiesce_devices() — exercises Phase 2.5",
            cycle + 1,
            kCycles);

        // Dispatch async: ERISC EDM firmware is actively running.
        // No Finish() — quiesce_devices() must wait internally AND terminate ERISC before L1 clear.
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }

        // quiesce_devices() → quiesce_internal() → quiesce_and_restart_fabric_workers():
        //   Phase 1: send TERMINATE to Tensix MUX cores (skipped if FabricTensixConfig::DISABLED)
        //   Phase 2: poll Tensix MUX TERMINATED
        //   Phase 2.5 (FIX): send TERMINATE to each ERISC channel and poll TERMINATED
        //   Phase 3: configure_fabric_cores() — overwrite ERISC L1 with new firmware
        //   Phase 4: wait for ERISC READY_FOR_TRAFFIC
        //
        // Pre-fix: Phase 3 ran without Phase 2.5 guard → ERISC still running when L1 cleared
        //          → corrupted program → invalid NOC traffic → next dispatch hangs.
        mesh_device_->quiesce_devices();

        log_info(tt::LogTest, "[Scenario H] Cycle {}/{}: quiesce_devices() returned cleanly", cycle + 1, kCycles);
    }

    // Final verification: blocking dispatch must complete after N quiesce cycles.
    // If Phase 2.5 is broken, one of the quiesce cycles above would have corrupted ERISC state
    // and this dispatch would hang in completion_queue_wait_front (exit=124 in CI).
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario H] Final blocking dispatch after {} quiesce cycles", kCycles);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Buffer round-trip: detect DRAM corruption from stale ERISC NOC writes.
    // Without Phase 2.5, a zombie ERISC can write stale packet data to device DRAM at any
    // time during Phase 3 or after, corrupting the buffer we're about to read back.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xA11E0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size())
            << "[Scenario H] Buffer size mismatch after " << kCycles << " quiesce cycles";
        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[Scenario H] Data corruption at index " << i << " after " << kCycles << " quiesce cycles";
        }
        log_info(
            tt::LogTest,
            "[Scenario H] Buffer round-trip clean — Phase 2.5 ERISC termination working across {} quiesce cycles",
            kCycles);
    }
}

// ---------------------------------------------------------------------------
// Scenario I: High-iteration quiesce stress with per-cycle timing bound.
//
// GAPS FILLED:
//   1. Scenario H only runs 3 quiesce cycles. 8 cycles amplify the Phase 2.5
//      ERISC termination race window and catch cumulative state leaks.
//   2. Scenario H has no timing assertion. This test asserts each quiesce
//      cycle completes in < 6000ms, catching Phase 2.5 poll hangs early —
//      well before the 90s test budget expires.
//
// What the timing bound catches:
//   - Phase 2.5 polls ERISC channels for TERMINATED (50ms timeout each).
//     T3K has ~8 devices × ~4 active ETH channels = ~32 polls per quiesce.
//     A regressed poll stuck at 50ms per channel = ~1600ms overhead/cycle.
//     The 6s limit catches this before the test times out silently.
//
// Pass = 8 cycles complete, each in < 6s, final buffer matches.
// Fail = any cycle > 6s (Phase 2.5 regression), hang (watchdog), or corruption.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric2DRepeatFixture, QuiesceDevicesTimingBoundStressTest) {
    constexpr int kCycles = 8;
    constexpr int64_t kMaxCycleMs = 6000;  // 50ms × 32 channels + 4.4s margin
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        // Dispatch async — ERISC channels are live during quiesce.
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            // No Finish() — quiesce_devices() must drain + terminate ERISCs.
        }

        // Time the quiesce. Assertion catches Phase 2.5 poll regression before
        // the 90s budget expires.
        const auto t0 = std::chrono::steady_clock::now();
        mesh_device_->quiesce_devices();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

        log_info(
            tt::LogTest,
            "[Scenario I] Cycle {}/{}: quiesce_devices() in {}ms (limit {}ms)",
            cycle + 1,
            kCycles,
            elapsed_ms,
            kMaxCycleMs);

        ASSERT_LT(elapsed_ms, kMaxCycleMs)
            << "[Scenario I] Cycle " << (cycle + 1) << "/" << kCycles
            << " exceeded " << kMaxCycleMs << "ms — Phase 2.5 ERISC poll may be hanging";
    }

    // Final blocking dispatch must succeed after 8 quiesce cycles.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario I] Final blocking dispatch after {} quiesce cycles", kCycles);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Buffer round-trip: detect DRAM corruption from stale ERISC NOC writes
    // that could accumulate over 8 quiesce cycles.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};
        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0x51CE0000 | (i & 0xFFFF));
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size())
            << "[Scenario I] Buffer size mismatch after " << kCycles << " quiesce cycles";
        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[Scenario I] Data corruption at index " << i << " after " << kCycles << " quiesce cycles";
        }
        log_info(
            tt::LogTest,
            "[Scenario I] Buffer round-trip clean — Phase 2.5 timing bounded across {} quiesce cycles",
            kCycles);
    }
}

// ---------------------------------------------------------------------------
// Fixture: quiesce_devices() with FABRIC_1D active — tests the ETH-only path
// through quiesce_and_restart_fabric_workers().
//
// WHY THIS FIXTURE EXISTS (critical gap vs AsyncTeardownFabric2DRepeatFixture):
//   Scenarios H & I use FABRIC_2D (FabricTensixConfig::ENABLED → has_tensix_mux=true),
//   which exercises Phases 1/2/2.5/3/4 in quiesce_and_restart_fabric_workers().
//
//   FABRIC_1D uses FabricTensixConfig::DISABLED → has_tensix_mux=false.
//   Phases 1/2/4 (Tensix MUX) are SKIPPED — only Phase 2.5 (ERISC TERMINATE poll)
//   and Phase 3 (re-configure ERISC L1) run.
//
//   This is exactly the code path that regressed in iter12 (commit b7b19dc905):
//   an early-return guard at FabricTensixConfig::DISABLED prevented Phase 2.5
//   from running → ERISCs not terminated → stale firmware ran concurrently with
//   new L1 load → AllGather hung at iteration 2.
//
//   Without a FABRIC_1D-specific test, that regression would be invisible to
//   Scenarios H/I (which pass with or without the FABRIC_1D fix).
//
// Requires >= 2 devices (FABRIC_1D needs ETH routing between chips).
// ---------------------------------------------------------------------------
class AsyncTeardownFabric1DQuiesceFixture : public MeshDeviceFixtureBase {
protected:
    AsyncTeardownFabric1DQuiesceFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
              .test_budget_ms = 90000,  // 90s: FABRIC_1D init ~5-10s, 5 cycles max ~60s
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "AsyncTeardownFabric1DQuiesceFixture requires >= 2 devices (FABRIC_1D needs ETH routing)";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Scenario J: FABRIC_1D quiesce_devices() — Phase 2.5 runs in ETH-only path.
//
// GAPS FILLED vs Scenarios H/I (FABRIC_2D quiesce):
//   1. Exercises quiesce_and_restart_fabric_workers() with has_tensix_mux=false
//      (FabricTensixConfig::DISABLED). Phases 1/2/4 are skipped; only Phase 2.5
//      and Phase 3 run. This is the FABRIC_1D code path.
//   2. Would catch a regression of the iter12 bug: if someone re-introduces an
//      early-return at the FabricTensixConfig::DISABLED guard, FABRIC_2D tests
//      (H/I) would still pass but this test would hang.
//   3. Mirrors the exact dispatch pattern of MeshDevice1x4FabricFixture (uses
//      FABRIC_1D + blank kernels + quiesce_devices between iterations), which is
//      the test that exposed the original AllGather hang.
//
// The test dispatches async → calls quiesce_devices() 3 times without blocking
// Finish(). Each quiesce cycle must complete Phase 2.5 (ERISC TERMINATE poll)
// in FABRIC_1D mode, then re-configure ERISC L1 (Phase 3). A final blocking
// dispatch and buffer round-trip verify no ERISC state was corrupted.
//
// Pass = 3 quiesce cycles complete, final dispatch succeeds, buffer matches.
// Fail = hang (90s watchdog), crash in quiesce_and_restart_fabric_workers, or
//        data corruption indicating stale ERISC NOC traffic.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric1DQuiesceFixture, QuiesceDevicesPhase25ERISCTerminationFabric1D) {
    constexpr int kCycles = 3;
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        log_info(
            tt::LogTest,
            "[Scenario J] Cycle {}/{}: FABRIC_1D async dispatch + quiesce_devices() — Phase 2.5 ETH-only path",
            cycle + 1,
            kCycles);

        // Dispatch async — ERISCs are live in FABRIC_1D mode during quiesce.
        // No Finish() — quiesce_devices() must: poll ERISC TERMINATED (Phase 2.5),
        // then overwrite ERISC L1 with new firmware (Phase 3). If Phase 2.5 is
        // skipped (has_tensix_mux=false guard regression), Phase 3 races with
        // still-running ERISC firmware → L1 corruption → next dispatch hangs.
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }

        // quiesce_and_restart_fabric_workers (FABRIC_1D / has_tensix_mux=false):
        //   Phase 1: SKIPPED (no Tensix MUX)
        //   Phase 2: SKIPPED (no Tensix MUX)
        //   Phase 2.5 (FIX): send TERMINATE to each ERISC channel, poll TERMINATED
        //   Phase 3: configure_fabric_cores() — overwrite ERISC L1 with new firmware
        //   Phase 4: SKIPPED (no Tensix MUX READY_FOR_TRAFFIC)
        mesh_device_->quiesce_devices();

        log_info(tt::LogTest, "[Scenario J] Cycle {}/{}: quiesce_devices() returned cleanly", cycle + 1, kCycles);
    }

    // Final blocking dispatch must complete — if any quiesce cycle left a zombie ERISC
    // (Phase 2.5 missing), this dispatch hangs in completion_queue_wait_front (exit=124).
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario J] Final blocking dispatch after {} FABRIC_1D quiesce cycles", kCycles);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Buffer round-trip: detect DRAM corruption from stale ERISC NOC writes after
    // incomplete Phase 2.5 termination in FABRIC_1D mode.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xFA1D0000 | (i & 0xFFFF));  // 0xFA1D = "FA1D" (FABRIC_1D)
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size())
            << "[Scenario J] Buffer size mismatch after " << kCycles << " FABRIC_1D quiesce cycles";
        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[Scenario J] Data corruption at index " << i << " after " << kCycles
                << " FABRIC_1D quiesce cycles — stale ERISC NOC write corrupted DRAM";
        }
        log_info(
            tt::LogTest,
            "[Scenario J] Buffer round-trip clean — FABRIC_1D Phase 2.5 ERISC termination working across {} cycles",
            kCycles);
    }
}

// ---------------------------------------------------------------------------
// Scenario K: FABRIC_1D quiesce timing-bound stress — per-cycle bound detects
// Phase 2.5 poll regressions in the has_tensix_mux=false path.
//
// GAPS FILLED vs Scenario J:
//   1. Scenario J runs 3 quiesce cycles with no timing assertion. This test runs
//      5 cycles with a per-cycle bound (5000ms) to catch Phase 2.5 regressions
//      early — before the 90s test budget expires silently.
//   2. FABRIC_1D has fewer active channels per device than FABRIC_2D (ring-only
//      ETH routing vs full mesh), so the timing characteristics differ. An
//      independent bound tuned to FABRIC_1D is needed.
//
// Per-cycle timing bound:
//   FABRIC_1D T3K: ~4 devices × ~2 active ETH channels = ~8 polls per quiesce.
//   Phase 2.5 poll timeout = 50ms/channel → worst-case regressed = ~400ms overhead.
//   5000ms limit gives 12.5× headroom over worst-case normal (50ms × 8 channels)
//   while catching a stuck poll before the 90s test budget expires.
//
// Pass = 5 cycles complete, each in < 5s, final buffer matches.
// Fail = any cycle > 5s (Phase 2.5 poll regression), hang (watchdog), or corruption.
// ---------------------------------------------------------------------------
TEST_F(AsyncTeardownFabric1DQuiesceFixture, QuiesceDevicesTimingBoundFabric1D) {
    constexpr int kCycles = 5;
    constexpr int64_t kMaxCycleMs = 5000;  // 50ms × 8 channels + 4.6s margin
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        // Dispatch async — FABRIC_1D ERISCs are live during quiesce.
        {
            auto program = create_blank_program(cores);
            auto workload = MeshWorkload();
            workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            // No Finish() — quiesce_devices() must drain + terminate ERISCs.
        }

        // Time the quiesce cycle. Assertion catches Phase 2.5 FABRIC_1D regression
        // before the 90s test budget expires.
        const auto t0 = std::chrono::steady_clock::now();
        mesh_device_->quiesce_devices();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

        log_info(
            tt::LogTest,
            "[Scenario K] Cycle {}/{}: FABRIC_1D quiesce_devices() in {}ms (limit {}ms)",
            cycle + 1,
            kCycles,
            elapsed_ms,
            kMaxCycleMs);

        ASSERT_LT(elapsed_ms, kMaxCycleMs)
            << "[Scenario K] Cycle " << (cycle + 1) << "/" << kCycles
            << " exceeded " << kMaxCycleMs << "ms — FABRIC_1D Phase 2.5 ERISC poll may be hanging";
    }

    // Final blocking dispatch must succeed after 5 FABRIC_1D quiesce cycles.
    {
        auto program = create_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        log_info(tt::LogTest, "[Scenario K] Final blocking dispatch after {} FABRIC_1D quiesce cycles", kCycles);
        EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    }

    // Buffer round-trip: detect DRAM corruption from stale ERISC NOC writes
    // that could accumulate over 5 FABRIC_1D quiesce cycles.
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = Shape2D{1, 1}};
        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words = page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0x1D1D0000 | (i & 0xFFFF));  // "1D1D" for FABRIC_1D stress
        }
        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);
        ASSERT_EQ(dst.size(), src.size())
            << "[Scenario K] Buffer size mismatch after " << kCycles << " FABRIC_1D quiesce cycles";
        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[Scenario K] Data corruption at index " << i << " after " << kCycles
                << " FABRIC_1D quiesce cycles";
        }
        log_info(
            tt::LogTest,
            "[Scenario K] Buffer round-trip clean — FABRIC_1D timing-bound quiesce across {} cycles",
            kCycles);
    }
}

}  // namespace tt::tt_metal::distributed::test
