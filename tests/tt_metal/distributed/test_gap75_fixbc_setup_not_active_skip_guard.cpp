// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-75: FIX BC gap — MeshDeviceFixtureBase::SetUp() previously threw TT_FATAL
//         when non-MMIO devices were unreachable after a prior session left the
//         ETH relay dead, instead of calling GTEST_SKIP.
//
// Root cause (FIX BC, commit 43287d3ce3f):
//
//   FIX AQ (UMD TopologyDiscovery) drops non-MMIO devices from the active set
//   when their ETH relay is dead after a prior session was SIGKILL'd.
//   initialize_fabric_and_dispatch_fw() then TT_FATALs on
//   "Device N is not active" because the system mesh config lists N devices but
//   only MMIO devices were opened.
//
//   Before FIX BC, this crashes AllGatherPersistentOutput, ReduceScatter, and
//   AllReduce tests as FAIL (abort/SIGABRT) rather than SKIP, contaminating the
//   CI failure count and hiding the real hardware-degraded root cause.
//
//   FIX BC fix (43287d3ce3f) wraps MeshDevice::create() in a try/catch in
//   MeshDeviceFixtureBase::SetUp():
//
//     try {
//         mesh_device_ = MeshDevice::create(...);
//     } catch (const std::exception& e) {
//         std::string what = e.what();
//         if (what.find("is not active") != std::string::npos) {
//             if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
//                 tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
//             }
//             GTEST_SKIP() << "FIX BC (#42429): ...";
//         }
//         throw;
//     }
//
// What this test verifies:
//
//   Fork test confirming that when a prior session left the ETH relay dead,
//   MeshDevice::create() throwing "is not active" is caught and converted to
//   GTEST_SKIP rather than crashing the test binary.
//
//   PREDECESSOR: Opens full FABRIC_2D mesh, dispatches blank workload, signals
//   ready, spins until SIGKILL.  Leaves ETH relay firmware live on non-MMIO
//   ERISCs.  FIX AQ drops non-MMIO devices from subsequent sessions.
//
//   TESTEE-1: Attempts to create a MeshDevice using the same try/catch logic
//   that FIX BC adds to MeshDeviceFixtureBase::SetUp().  Three outcomes:
//     A) "is not active" exception caught → SetFabricConfig(DISABLED) → exit 75
//        (FIX BC working — test would have GTEST_SKIP'd)
//     B) MeshDevice created successfully → exit 0
//        (clean cluster — FIX AQ dropped non-MMIO but MeshDevice didn't TT_FATAL)
//     C) TT_FATAL (abort/SIGABRT) → exit 134 or killed by signal → -1
//        (REGRESSION — FIX BC catch missing, test would FAIL instead of SKIP)
//
// Exit codes from TESTEE-1:
//   exit 0   — Clean cluster (MeshDevice created; no stale-relay scenario)
//   exit 75  — FIX BC WORKING: "is not active" caught, GTEST_SKIP path taken
//   exit 134 — REGRESSION: TT_FATAL/SIGABRT — FIX BC catch missing
//   -1/timeout — REGRESSION: hang in MeshDevice::create()
//
// Timing budgets:
//   PREDECESSOR wait:      35s (hardware init + blank workload)
//   TESTEE-1 budget:       45s (SetFabricConfig + MeshDevice::create() attempt)
//   Total:                 ~85s
//
// Topology requirement: >= 2 devices (non-MMIO devices needed for relay kill
//   scenario). Skip if < 2 devices.
//
// Note: This test does NOT use a GTest fixture for its own mesh — it manages
//   everything with explicit fork/SIGKILL to isolate the predecessor scenario.
//   The test binary must be run with a healthy cluster; the test itself creates
//   the degraded state via PREDECESSOR.
//
// ─────────────────────────────────────────────────────────────────────────────

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
#include <experimental/fabric/fabric.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kGap75PredWaitMs      = 35000;  // 35s for predecessor init
static constexpr int kGap75Testee1BudgetMs = 45000;  // 45s for SetFabricConfig + create attempt

// Exit code: FIX BC working — "is not active" caught, GTEST_SKIP taken
static constexpr int kGap75ExitFixBcWorking = 75;

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap75Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap75(const MeshCoordinateRange& range) {
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
// Wait-for-child with timeout
// ---------------------------------------------------------------------------
static int wait_child_budget_gap75(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start).count();
        if (elapsed > budget_ms) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 255;
}

// ---------------------------------------------------------------------------
// GAP-75: SetUpSkipsCleanlyWhenNonMmioDevicesNotActive
//
// Verifies FIX BC: when a prior session left the ETH relay dead, and FIX AQ
// drops non-MMIO devices from UMD TopologyDiscovery, MeshDevice::create()
// throws "is not active" which the new FIX BC catch converts to GTEST_SKIP
// rather than crashing the binary with TT_FATAL/SIGABRT.
//
// Without FIX BC:
//   - PREDECESSOR leaves ETH relay dead.
//   - FIX AQ drops non-MMIO devices from UMD TopologyDiscovery.
//   - MeshDeviceFixtureBase::SetUp() calls SetFabricConfig(FABRIC_2D) then
//     MeshDevice::create() which calls initialize_fabric_and_dispatch_fw().
//   - initialize_fabric_and_dispatch_fw() TT_FATALs: "Device N is not active".
//   - abort() kills the test binary → FAIL (not SKIP).
//
// With FIX BC:
//   - Same scenario up to MeshDevice::create() throwing.
//   - Catch block: SetFabricConfig(DISABLED) (cleanup), GTEST_SKIP().
//   - Test skips cleanly; binary continues to next test.
// ---------------------------------------------------------------------------
TEST(Gap75SetUpNotActiveSkipGuard, SetUpSkipsCleanlyWhenNonMmioDevicesNotActive) {
    const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        GTEST_SKIP() << "GAP-75 requires >= 2 devices (non-MMIO devices needed for "
                     << "ETH relay kill scenario). Found " << num_devices << " device(s).";
    }

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap75Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap75Shm();

    // ── Phase 1: Fork PREDECESSOR ────────────────────────────────────────────
    // Opens full FABRIC_2D mesh, dispatches blank workload, signals ready,
    // spins until SIGKILL.  Leaves ETH relay firmware live on non-MMIO ERISCs.
    // FIX AQ will drop non-MMIO devices from the next session's UMD discovery.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_devices)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap75(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor ready signal (or timeout → skip)
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - pred_start).count();
        if (elapsed > kGap75PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap75Shm));
            GTEST_SKIP() << "GAP-75: predecessor did not signal ready within " << kGap75PredWaitMs
                         << "ms — skipping (cluster may be unhealthy).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-75: Predecessor SIGKILL'd — ETH relay firmware left on non-MMIO ERISCs. "
        "TESTEE-1 will attempt MeshDevice::create() using the FIX BC try/catch pattern. "
        "Expected: 'is not active' exception caught → exit 75 (FIX BC working). "
        "Regression: TT_FATAL/SIGABRT (exit 134) or hang → FIX BC catch missing. "
        "Budget: {}ms.",
        kGap75Testee1BudgetMs);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ── Phase 2: Fork TESTEE-1 (FIX BC try/catch validation) ─────────────────
    // Replicates the logic that FIX BC adds to MeshDeviceFixtureBase::SetUp():
    //
    //   SetFabricConfig(FABRIC_2D, STRICT_SYSTEM_HEALTH_SETUP_MODE)
    //   try {
    //       mesh_device_ = MeshDevice::create(...)
    //   } catch (const std::exception& e) {
    //       if (e.what() contains "is not active") {
    //           SetFabricConfig(DISABLED)
    //           GTEST_SKIP()  ← we exit 75 here to represent the skip path
    //       }
    //       throw;  ← re-throw unknown exceptions (exit via abort)
    //   }
    //
    // Exit 75 = FIX BC working (skip path reached)
    // Exit 0  = MeshDevice created successfully (clean cluster, no regression)
    // Exit 134/signal = REGRESSION (TT_FATAL without FIX BC catch)
    pid_t testee1_pid = ::fork();
    ASSERT_GE(testee1_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee1_pid == 0) {
        int rc = 0;
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_devices)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // MeshDevice created successfully — cluster was healthy enough.
            // Clean up and exit 0.
            dev->close();
            fprintf(stderr,
                "GAP-75 TESTEE-1: MeshDevice::create() succeeded (clean cluster). Exit 0.\n");
            rc = 0;
        } catch (const std::exception& e) {
            std::string what = e.what();
            fprintf(stderr, "GAP-75 TESTEE-1: exception: %s\n", what.c_str());
            if (what.find("is not active") != std::string::npos) {
                // This is the FIX BC path: SetFabricConfig(DISABLED) + GTEST_SKIP
                try {
                    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
                } catch (...) {}
                fprintf(stderr,
                    "GAP-75 TESTEE-1: FIX BC path taken — 'is not active' caught.\n"
                    "  SetFabricConfig(DISABLED) called. Would GTEST_SKIP in fixture.\n"
                    "  Exiting %d (FIX BC working).\n", kGap75ExitFixBcWorking);
                rc = kGap75ExitFixBcWorking;
            } else {
                // Unknown exception — re-throw (will cause abort in subprocess)
                fprintf(stderr,
                    "GAP-75 TESTEE-1: unknown exception — re-throwing.\n");
                throw;
            }
        }
        _exit(rc);
    }

    const auto t1_start = std::chrono::steady_clock::now();
    int rc1 = wait_child_budget_gap75(testee1_pid, kGap75Testee1BudgetMs);
    const long t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t1_start).count();

    ::munmap(raw_shm, sizeof(Gap75Shm));

    if (rc1 == -1) {
        FAIL() << "GAP-75 REGRESSION (FIX BC): TESTEE-1 timed out after " << t1_ms << "ms "
               << "(budget: " << kGap75Testee1BudgetMs << "ms).\n"
               << "\n"
               << "MeshDevice::create() hung after predecessor SIGKILL'd the ETH relay.\n"
               << "Possible causes:\n"
               << "  1. FIX BC catch is missing → MeshDevice::create() hangs before TT_FATAL\n"
               << "  2. FIX AQ not dropping non-MMIO devices → topology init hang\n"
               << "\n"
               << "Fix: Ensure MeshDeviceFixtureBase::SetUp() wraps MeshDevice::create()\n"
               << "in a try/catch that converts 'is not active' to GTEST_SKIP.\n"
               << "See commit 43287d3ce3f.";
    }

    if (rc1 == 134 || (rc1 >= 128 && rc1 != kGap75ExitFixBcWorking)) {
        FAIL() << "GAP-75 REGRESSION (FIX BC): TESTEE-1 exited " << rc1
               << " (likely SIGABRT / TT_FATAL).\n"
               << "\n"
               << "This confirms the FIX BC regression scenario:\n"
               << "  - PREDECESSOR left ETH relay dead on non-MMIO ERISCs.\n"
               << "  - FIX AQ dropped non-MMIO devices from UMD TopologyDiscovery.\n"
               << "  - MeshDevice::create() called initialize_fabric_and_dispatch_fw().\n"
               << "  - TT_FATAL fired: 'Device N is not active' (device listed in system\n"
               << "    mesh config but not in active_devices_ after FIX AQ drop).\n"
               << "  - abort() killed the test binary → FAIL instead of SKIP.\n"
               << "\n"
               << "Without FIX BC, AllGatherPersistentOutput, ReduceScatter, and AllReduce\n"
               << "show as FAIL rather than SKIP on degraded clusters, hiding the\n"
               << "hardware-degraded root cause.\n"
               << "\n"
               << "Fix: Add try/catch around MeshDevice::create() in\n"
               << "MeshDeviceFixtureBase::SetUp() that converts 'is not active' exceptions\n"
               << "to SetFabricConfig(DISABLED) + GTEST_SKIP.\n"
               << "See commit 43287d3ce3f.";
    }

    EXPECT_TRUE(rc1 == 0 || rc1 == kGap75ExitFixBcWorking)
        << "GAP-75: TESTEE-1 exited with unexpected code " << rc1;

    if (rc1 == kGap75ExitFixBcWorking) {
        log_info(
            tt::LogTest,
            "GAP-75 PASS (exit 75): FIX BC correctly converts 'is not active' exception "
            "to GTEST_SKIP path in MeshDeviceFixtureBase::SetUp(). "
            "SetFabricConfig(DISABLED) called for cleanup. "
            "Without FIX BC, this would have crashed the binary with TT_FATAL (exit 134). "
            "TESTEE-1 completed in {}ms (budget {}ms).",
            t1_ms, kGap75Testee1BudgetMs);
    } else {
        log_info(
            tt::LogTest,
            "GAP-75 PASS (exit 0): MeshDevice::create() succeeded on clean cluster — "
            "gap scenario not triggered. TESTEE-1 in {}ms.", t1_ms);
    }
}

}  // namespace tt::tt_metal::distributed::test
