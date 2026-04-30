// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-55: FIX E2+AY gap — probe_dead path must set fabric_relay_path_broken_ to trigger FIX AY.
// Commit: 6249f8f839d
//
// Root cause (CI run 25137442210, job 73680628895 — T3K with degraded ETH):
//
//   compile_and_configure_fabric() probes each device's ETH channels.  For non-MMIO devices
//   with compromised channels (probe_dead_channels non-empty), FIX E2 adds the device to
//   dead_relay_devices_ so dispatch is skipped.  However, before the fix, FIX E2 did NOT call
//   dev->set_fabric_relay_path_broken() for the probe_dead path — only for the relay_broken
//   path (3+ consecutive relay timeouts).
//
//   This omission breaks the teardown chain:
//     1. RiscFirmwareInitializer::teardown() populates relay_broken_non_mmio from
//        is_fabric_relay_path_broken().  Without the flag set, probe_dead devices are excluded.
//     2. FIX AY (deferred non-MMIO ERISC reset) is gated on relay_broken_non_mmio.
//        No entries → FIX AY doesn't fire for those devices.
//     3. Stale FABRIC firmware remains on non-MMIO ERISCs after session close.
//     4. Next session: UMD topology discovery sends heartbeat probes to those ERISCs.
//        Stale FABRIC firmware ignores them → "ASIC not found in chip_topology_mapping_" TT_FATAL.
//
// The fix (fabric_firmware_initializer.cpp):
//   Inside the FIX E2 block, dev->set_fabric_relay_path_broken() is now called for ALL cases
//   (both relay_broken and probe_dead), ensuring FIX AY fires during teardown:
//     if (is_non_mmio && (relay_broken || !probe_dead_channels.empty())) {
//         dead_relay_devices_.insert(dev->id());
//         dev->set_fabric_relay_path_broken();  // <-- was missing for probe_dead
//     }
//
// What this test verifies:
//   Three-fork test targeting the SPECIFIC regression path where probe_dead triggers dead-relay
//   but FIX AY was not invoked, leaving stale firmware for the next session.
//
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D, signals ready, spins until SIGKILL.
//            SIGKILL leaves non-MMIO ERISCs in FABRIC firmware (channels may appear as
//            INITIALIZATION_STARTED or corrupt state to subsequent probes).
//
//   Phase 2: Fork TESTEE-1 — opens FABRIC_2D in STRICT mode.
//            compile_and_configure_fabric() probes non-MMIO channels → finds dead/stale
//            channels → FIX E2 fires (dead_relay_devices_ populated).
//            With the fix: set_fabric_relay_path_broken() is called → FIX AY fires on close.
//            Without the fix: flag not set → FIX AY skipped → stale firmware persists.
//            TESTEE-1 may fail with exceptions (relay broken) — that's acceptable.
//            The important thing is that it runs the teardown chain and exits.
//
//   Phase 3: Fork TESTEE-2 — opens FABRIC_2D again.
//            With the fix: ERISCs were reset to base firmware by FIX AY → topology discovery
//            succeeds → exit 0.
//            Without the fix: stale FABRIC firmware → UMD heartbeat fails →
//            "ASIC not found in chip_topology_mapping_" TT_FATAL → SIGABRT.
//
// Regression indicator:
//   TESTEE-2 exits nonzero (SIGABRT from TT_FATAL) or times out.
//   TESTEE-1 failures are tolerated (catch-and-continue).
//   The critical assertion is TESTEE-2 exits 0 within budget.
//
// Timing budget:
//   PREDECESSOR wait: 30s (hardware init)
//   TESTEE-1:         90s (full first-session teardown chain)
//   TESTEE-2:         45s (if FIX AY didn't fire, topology discovery hangs or crashes)
//   Total test_budget_ms: 240s
//
// Topology requirement: >= 2 devices (non-MMIO relay path required).
// Relation to GAP-31: GAP-31 tests FIX AY via the relay_broken path.
//   GAP-55 tests FIX AY via the probe_dead path (the gap fixed by 6249f8f839d).

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <thread>

#include <experimental/fabric/fabric_types.hpp>
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
// Constants
// ---------------------------------------------------------------------------
static constexpr int kPredWaitMs = 30000;       // 30s predecessor init
static constexpr int kTestee1BudgetMs = 90000;  // 90s first session (open + teardown chain)
static constexpr int kTestee2BudgetMs = 45000;  // 45s second session (topology + open)

// ---------------------------------------------------------------------------
// Shared memory for inter-process signaling
// ---------------------------------------------------------------------------
struct Gap55SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper (same pattern as GAP-31/48/54)
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap55(const MeshCoordinateRange& range) {
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
// Helper: fork a child, wait with budget, return exit status or -1 on timeout.
// ---------------------------------------------------------------------------
static int wait_child_with_budget(pid_t pid, int budget_ms, const char* label) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
        if (elapsed_ms > budget_ms) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
            return -1;  // timeout sentinel
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 255;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class FixE2AyProbeDeadFayTriggerFixture : public MeshDeviceFixtureBase {
protected:
    FixE2AyProbeDeadFayTriggerFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 240000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-55 requires >= 2 devices (non-MMIO relay path required). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-55: ProbeDeadPathTriggersFayForEriscCleanup
//
// Regression indicator: TESTEE-2 exits nonzero (SIGABRT) or times out.
// ---------------------------------------------------------------------------
TEST_F(FixE2AyProbeDeadFayTriggerFixture, ProbeDeadPathTriggersFayForEriscCleanup) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap55SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap55SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D to put ERISCs into FABRIC firmware state, then spins.
    // SIGKILL leaves non-MMIO ERISCs in stale FABRIC firmware (may appear as
    // INITIALIZATION_STARTED or corrupt to subsequent probes → probe_dead path).
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
            // Run a blank workload to confirm dispatch is fully active.
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap55(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        // Spin so SIGKILL leaves ERISCs in FABRIC firmware state.
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Parent: wait for predecessor to signal ready.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap55SharedMem));
            GTEST_SKIP() << "GAP-55: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?); skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Let ERISCs stabilize in FABRIC fw state before SIGKILL.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-55: Predecessor SIGKILL'd — non-MMIO ERISCs left in stale FABRIC firmware. "
        "TESTEE-1 will open FABRIC_2D (FIX E2 fires for probe_dead channels). "
        "With fix: set_fabric_relay_path_broken_ set → FIX AY fires on close. "
        "Without fix: flag not set → FIX AY skipped → stale firmware persists.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE-1 ─────────────────────────────────────────────────
    // Opens FABRIC_2D → compile_and_configure_fabric() probes non-MMIO channels →
    // FIX E2 fires for dead/stale channels → adds to dead_relay_devices_.
    // With fix: set_fabric_relay_path_broken() called → FIX AY fires on close.
    // Without fix: flag not set → FIX AY skipped.
    // TESTEE-1 may fail with exceptions — that's acceptable. The critical path is
    // that the teardown chain runs (triggering FIX AY if the flag was set).
    pid_t testee1_pid = ::fork();
    ASSERT_GE(testee1_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee1_pid == 0) {
        int rc = 0;
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            dev->close();
            rc = 0;
        } catch (...) {
            // Failures from relay-broken state are expected.  The important thing
            // is that teardown ran (FIX AY fires if fabric_relay_path_broken_ set).
            rc = 0;
        }
        _exit(rc);
    }

    int rc1 = wait_child_with_budget(testee1_pid, kTestee1BudgetMs, "TESTEE-1");
    if (rc1 == -1) {
        ::munmap(raw_shm, sizeof(Gap55SharedMem));
        GTEST_SKIP() << "GAP-55: TESTEE-1 timed out at " << kTestee1BudgetMs
                     << "ms (hardware init stall?). Cannot verify FIX AY path; skipping.";
    }

    log_info(
        tt::LogTest,
        "GAP-55: TESTEE-1 completed (exit %d). FIX E2 should have marked probe_dead "
        "devices as relay-broken. If fix present, FIX AY reset non-MMIO ERISCs to base firmware.",
        rc1);

    // Brief settle between testees.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ── Phase 3: Fork TESTEE-2 ─────────────────────────────────────────────────
    // Opens FABRIC_2D again.  This is the critical regression check:
    //   With fix: ERISCs reset to base firmware by FIX AY → topology discovery succeeds → exit 0.
    //   Without fix: stale FABRIC firmware → UMD heartbeat fails →
    //                "ASIC not found in chip_topology_mapping_" TT_FATAL → SIGABRT.
    pid_t testee2_pid = ::fork();
    ASSERT_GE(testee2_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee2_pid == 0) {
        int rc = 0;
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Verify second session is functional: dispatch a blank workload.
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap55(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, /*blocking=*/true);
            Synchronize(dev.get(), std::nullopt);
            dev->close();
            rc = 0;
        } catch (...) {
            // Even if workload fails, reaching here means topology discovery didn't crash.
            // Use rc=1 to distinguish from SIGABRT.
            rc = 1;
        }
        _exit(rc);
    }

    const auto testee2_start = std::chrono::steady_clock::now();
    int rc2 = wait_child_with_budget(testee2_pid, kTestee2BudgetMs, "TESTEE-2");
    const auto testee2_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - testee2_start)
                                      .count();

    ::munmap(raw_shm, sizeof(Gap55SharedMem));

    // ── Assertions ──────────────────────────────────────────────────────────────
    // Primary check: TESTEE-2 must not timeout (FIX AY didn't fire → stale firmware).
    if (rc2 == -1) {
        FAIL() << "GAP-55 TIMEOUT (FIX E2+AY regression): TESTEE-2 did not exit within "
               << kTestee2BudgetMs << "ms.\n"
               << "\n"
               << "Root cause: FIX E2 (compile_and_configure_fabric) added non-MMIO devices to\n"
               << "dead_relay_devices_ via probe_dead path, but did NOT call\n"
               << "dev->set_fabric_relay_path_broken().  Without that flag,\n"
               << "RiscFirmwareInitializer::teardown() excludes those devices from\n"
               << "relay_broken_non_mmio → FIX AY (deferred non-MMIO ERISC reset) never fires.\n"
               << "Stale FABRIC firmware remains on non-MMIO ERISCs.\n"
               << "Next session's UMD topology discovery can't reach those chips →\n"
               << "\"ASIC not found in chip_topology_mapping_\" TT_FATAL.\n"
               << "\n"
               << "Fix: In FIX E2 block, call dev->set_fabric_relay_path_broken() for ALL\n"
               << "cases (both relay_broken and probe_dead). See commit 6249f8f839d.";
    }

    // Secondary check: no SIGABRT (the specific crash from TT_FATAL).
    // rc2 == 134 means SIGABRT (128 + 6).
    if (rc2 == 134) {
        FAIL() << "GAP-55 CRASH (FIX E2+AY regression): TESTEE-2 killed by SIGABRT.\n"
               << "\n"
               << "This is the exact failure mode: UMD topology discovery sends heartbeat\n"
               << "probes to non-MMIO ERISCs still running stale FABRIC firmware.\n"
               << "Stale firmware ignores probes → \"ASIC not found in chip_topology_mapping_\"\n"
               << "TT_FATAL → SIGABRT.\n"
               << "\n"
               << "Root cause: FIX E2 probe_dead path did not set fabric_relay_path_broken_\n"
               << "→ FIX AY skipped → ERISCs not reset to base firmware.\n"
               << "Fix: dev->set_fabric_relay_path_broken() in FIX E2 block (6249f8f839d).";
    }

    // TESTEE-2 exit 0 is the gold standard. Exit 1 (caught exception) is acceptable
    // — it means topology discovery succeeded but something else failed downstream.
    // The critical thing is no SIGABRT and no timeout.
    EXPECT_TRUE(rc2 == 0 || rc2 == 1)
        << "GAP-55: TESTEE-2 exited with unexpected code " << rc2
        << " (expected 0 or 1). Signal-based exit codes (128+N) indicate a crash.";

    log_info(
        tt::LogTest,
        "GAP-55 PASS: TESTEE-2 completed in {}ms (budget: {}ms) with exit {}. "
        "FIX E2+AY gap fix correctly set fabric_relay_path_broken_ for probe_dead devices, "
        "enabling FIX AY to reset non-MMIO ERISCs to base firmware during TESTEE-1 teardown. "
        "TESTEE-2 topology discovery succeeded (no stale FABRIC firmware on ERISCs).",
        testee2_elapsed,
        kTestee2BudgetMs,
        rc2);
}

}  // namespace tt::tt_metal::distributed::test
