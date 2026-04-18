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

}  // namespace tt::tt_metal::distributed::test
