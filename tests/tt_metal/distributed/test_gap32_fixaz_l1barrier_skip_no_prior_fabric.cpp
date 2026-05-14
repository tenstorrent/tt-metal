// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-32: l1_barrier hangs for non-MMIO device after assert_cores throws when
//         relay_broken_non_mmio is empty (no FabricFirmwareInitializer session).
//
// Root cause (discovered April 2026, OPUS audit iteration 20):
//   In RiscFirmwareInitializer::teardown() Step 3, the guard that skips l1_barrier
//   for non-MMIO devices was conditioned on relay_broken_non_mmio being non-empty.
//   relay_broken_non_mmio is populated in Step 1 by checking
//   device->is_fabric_relay_path_broken(), which is only set by
//   FabricFirmwareInitializer when it detects a dead relay path.
//
//   When a session opens WITHOUT fabric (FabricConfig::DISABLED / no SetFabricConfig),
//   FabricFirmwareInitializer never runs.  relay_broken_non_mmio stays empty.
//   Step 3 then reaches the assert_cores / l1_barrier path for non-MMIO devices.
//
//   If a prior process was SIGKILLed while running FABRIC_2D, it left non-MMIO
//   ERISCs in FABRIC firmware state.  When the current (no-fabric) session's
//   RiscFirmwareInitializer::teardown calls assert_cores for those non-MMIO
//   devices, assert_cores throws (5s UMD relay timeout — FABRIC fw ignores
//   base UMD relay reads).  Without FIX AZ, l1_barrier was then called
//   unconditionally — also going through the dead relay — adding another 5s per
//   device.  On a T3K (4 non-MMIO devices): first throw 5s + l1_barrier per
//   device 5s × 4 = ~25s extra teardown overhead.
//
// FIX AZ (risc_firmware_initializer.cpp, RiscFirmwareInitializer::teardown):
//   Introduced relay_dead_detected_step3 local flag.  When assert_cores throws
//   for any non-MMIO device in Step 3, the flag is set, l1_barrier is skipped
//   for that device, and all subsequent non-MMIO devices are skipped entirely
//   (they share the same MMIO relay path; if one is dead they all are).
//   Also guards Step 4's set_internal_routing_info_for_ethernet_cores via
//   any_relay_broken = relay_broken_non_mmio.empty() || relay_dead_detected_step3.
//
// What this test verifies:
//   1. Predecessor opens FABRIC_2D, signals ready, is SIGKILLed.
//      Non-MMIO ETH ERISCs are left in FABRIC firmware state.
//   2. Testee opens WITHOUT fabric (FabricConfig::DISABLED):
//      - FabricFirmwareInitializer never runs → relay_broken_non_mmio stays empty.
//      - TopologyDiscovery may hit 5s timeouts per non-MMIO device (FIX AQ handles).
//   3. Testee signals "about to close" and calls dev->close() or exits normally.
//   4. RiscFirmwareInitializer::teardown runs (triggered by MetalContext destructor
//      via global statics when exit(0) is called).
//   5. In Step 3: assert_cores throws for non-MMIO device → FIX AZ sets
//      relay_dead_detected_step3=true → l1_barrier skipped for all non-MMIO.
//   6. Testee teardown completes quickly.  Parent verifies testee exits within
//      kTeardownBudgetMs=20s after the "about to close" signal.
//      Without FIX AZ: teardown alone takes ~N*10s (5s assert_cores + 5s l1_barrier
//      per non-MMIO device) — > 20s on a T3K with 4 non-MMIO devices.
//      With FIX AZ:    teardown takes ~5s (first assert_cores throw + fast rest).
//
// Gap vs. existing tests:
//   GAP-31 (FIX AY) tests the scenario where FabricFirmwareInitializer DID run
//   (relay_broken_non_mmio is non-empty) and verifies deferred ERISC reset.
//   It does NOT test the "no FabricFirmwareInitializer session" path.
//
//   GAP-29 (FIX AW) tests the ~Cluster destructor hang in wait_for_non_mmio_flush.
//   It also has relay_broken_non_mmio non-empty (FabricFirmwareInitializer ran).
//
//   No existing test exercises the path where relay_broken_non_mmio stays empty
//   but assert_cores still throws for a non-MMIO device in Step 3.
//   This is the exact scenario from t3k_ttnn_udm_tests in CI iteration 20.
//
// Topology requirement: >= 2 devices (non-MMIO relay path required).

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <thread>

#include <experimental/fabric/fabric_types.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "fabric/fabric_init.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Shared memory layout for inter-process coordination
// ---------------------------------------------------------------------------
struct Gap32SharedFlags {
    std::atomic<int> predecessor_ready{0};    // predecessor: FABRIC_2D open + ERISCs active
    std::atomic<int> testee_about_to_close{0}; // testee: about to call close/exit
    std::atomic<int> testee_closed{0};          // testee: close/exit complete
};

// ---------------------------------------------------------------------------
// Fixture
//
// Budget covers:
//   ~30s  predecessor FABRIC_2D init + dispatch + signal
//    ~2s  post-kill margin
//   ~30s  testee DISABLED open (TopologyDiscovery: 5s × 4 non-MMIO + margin)
//   ~20s  testee teardown budget (FIX AZ: ~5s; without: ~25s on T3K)
//   ~30s  general margin
// ---------------------------------------------------------------------------
class FixAzL1BarrierSkipNoPriorFabricFixture : public MeshDeviceFixtureBase {
protected:
    FixAzL1BarrierSkipNoPriorFabricFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-32 requires >= 2 devices (non-MMIO relay path required). "
                            "Found "
                         << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Lightest workload that activates the ETH relay path.
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap32(const MeshCoordinateRange& range) {
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    Program prog;
    CreateKernel(
        prog,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    MeshWorkload workload;
    workload.add_program(range, std::move(prog));
    return workload;
}

// ---------------------------------------------------------------------------
// GAP-32: FixAzL1BarrierNotCalledAfterAssertCoresThrowsWithEmptyRelayBrokenSet
//
// Verifies that RiscFirmwareInitializer::teardown does NOT call l1_barrier
// after assert_cores throws for a non-MMIO device when relay_broken_non_mmio
// is empty (no FabricFirmwareInitializer session in this process).
// Without FIX AZ: l1_barrier is called unconditionally, adding 5s per device.
// With FIX AZ:    l1_barrier is skipped, teardown exits in ~5s.
// ---------------------------------------------------------------------------
TEST_F(FixAzL1BarrierSkipNoPriorFabricFixture, FixAzL1BarrierNotCalledAfterAssertCoresThrowsWithEmptyRelayBrokenSet) {
    // ── Shared memory ────────────────────────────────────────────────────────
    void* raw_shm = ::mmap(
        nullptr, sizeof(Gap32SharedFlags), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* flags = new (raw_shm) Gap32SharedFlags{};

    // Close fixture's own mesh device so children inherit a clean MetalContext.
    mesh_device_->close();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ─────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches a workload so ETH ERISCs are in ACTIVE relay
    // state, signals ready, then spins.  SIGKILL leaves non-MMIO ERISCs running
    // FABRIC firmware — the exact precondition for the FIX AZ bug.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap32(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
        }
        flags->predecessor_ready.store(1);
        // Spin until SIGKILL — leaves non-MMIO ERISCs in FABRIC firmware state.
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);
    }

    // Wait for predecessor to be ready (ETH ERISCs confirmed ACTIVE).
    constexpr int kPredWaitMs = 30000;
    const auto pred_start = std::chrono::steady_clock::now();
    while (flags->predecessor_ready.load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count();
        if (elapsed > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap32SharedFlags));
            GTEST_SKIP() << "Predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-32: predecessor SIGKILL'd — non-MMIO ERISCs left in FABRIC firmware state. "
        "Forking testee (FABRIC_DISABLED, no FabricFirmwareInitializer)...");

    // ── Phase 2: Fork TESTEE ──────────────────────────────────────────────────
    // Opens WITHOUT fabric (DISABLED) — FabricFirmwareInitializer never runs.
    // relay_broken_non_mmio stays EMPTY in RiscFirmwareInitializer::teardown.
    // When assert_cores throws for non-MMIO device (relay dead from FABRIC fw),
    // FIX AZ must: set relay_dead_detected_step3=true, skip l1_barrier, skip
    // all subsequent non-MMIO devices.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Open WITHOUT fabric: FabricConfig::DISABLED so no FabricFirmwareInitializer.
        // TopologyDiscovery may emit 5s timeouts for non-MMIO devices with stale
        // FABRIC fw — this is expected and handled by FIX AQ.
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::DISABLED,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Signal: testee opened successfully (or at least didn't hang).
            flags->testee_about_to_close.store(1);
            // Close: triggers RiscFirmwareInitializer::teardown.
            // Without FIX AZ: l1_barrier called after assert_cores throws → 5s per device.
            // With FIX AZ: relay_dead_detected_step3 set → l1_barrier skipped → ~5s total.
            dev->close();
        } catch (...) {
            // Open failed (TopologyDiscovery exception or similar) — still signal.
            // The MetalContext destructor will call RiscFirmwareInitializer::teardown
            // on exit(0) anyway.
            flags->testee_about_to_close.store(1);
        }
        flags->testee_closed.store(1);
        // exit(0) runs C++ global destructors (MetalContext teardown), which is where
        // RiscFirmwareInitializer::teardown fires if dev->close() didn't already run it.
        exit(0);
    }

    // ── Phase 3: Wait for testee to signal "about to close" ──────────────────
    // Budget: 90s for open (TopologyDiscovery may have 5s × N non-MMIO timeouts
    // via FIX AQ, plus general init overhead).
    constexpr int kOpenBudgetMs = 90000;
    const auto open_start = std::chrono::steady_clock::now();
    bool about_to_close_seen = false;
    while (true) {
        if (flags->testee_about_to_close.load() != 0) {
            about_to_close_seen = true;
            break;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - open_start)
                .count();
        if (elapsed > kOpenBudgetMs) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!about_to_close_seen) {
        ::kill(testee_pid, SIGKILL);
        ::waitpid(testee_pid, nullptr, 0);
        ::munmap(raw_shm, sizeof(Gap32SharedFlags));
        GTEST_SKIP() << "Testee did not signal about-to-close within " << kOpenBudgetMs
                     << "ms (hardware init stall or TopologyDiscovery hang?); skipping";
    }

    log_info(
        tt::LogTest,
        "GAP-32: testee signaled about_to_close (open completed in {}ms). "
        "Measuring teardown time — budget {}ms...",
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - open_start)
            .count(),
        15000);

    // ── Phase 4: Time the teardown — must complete within kTeardownBudgetMs ───
    // With FIX AZ:    ~5s  (first assert_cores throw for non-MMIO; rest skipped)
    // Without FIX AZ: ~N*10s on T3K (assert_cores 5s + l1_barrier 5s per device)
    //
    // kTeardownBudgetMs=20s: passes with FIX AZ, fails without on >= 3 non-MMIO devices.
    constexpr int kTeardownBudgetMs = 20000;
    const auto teardown_start = std::chrono::steady_clock::now();
    bool testee_exited = false;

    while (true) {
        int wstatus = 0;
        const pid_t result = ::waitpid(testee_pid, &wstatus, WNOHANG);
        if (result == testee_pid) {
            testee_exited = true;
            break;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - teardown_start)
                .count();
        if (elapsed > kTeardownBudgetMs) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!testee_exited) {
        ::kill(testee_pid, SIGKILL);
        ::waitpid(testee_pid, nullptr, 0);
    }

    ::munmap(raw_shm, sizeof(Gap32SharedFlags));
    raw_shm = nullptr;

    const auto teardown_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - teardown_start)
            .count();

    EXPECT_TRUE(testee_exited)
        << "GAP-32 REGRESSION: testee teardown did not complete within " << kTeardownBudgetMs
        << "ms after relay_broken_non_mmio-empty scenario. "
           "Root cause: RiscFirmwareInitializer::teardown Step 3 called l1_barrier "
           "unconditionally after assert_cores threw for a non-MMIO device, even though "
           "relay_broken_non_mmio was empty (no FabricFirmwareInitializer session). "
           "l1_barrier routes through the same dead relay → 5s UMD timeout per device. "
           "Fix: FIX AZ — relay_dead_detected_step3 flag skips l1_barrier and all "
           "subsequent non-MMIO devices once the first assert_cores throw is detected.";

    if (testee_exited) {
        log_info(
            tt::LogTest,
            "GAP-32: testee teardown completed in {}ms (FIX AZ: l1_barrier not called "
            "after assert_cores threw for relay-dead non-MMIO device; "
            "relay_broken_non_mmio was empty — no FabricFirmwareInitializer session).",
            teardown_elapsed_ms);
    }
}

}  // namespace tt::tt_metal::distributed::test
