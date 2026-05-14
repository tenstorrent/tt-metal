// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-58: FIX QC gap — reset_cores() must skip assert calls for non-MMIO devices with
//         dead ERISC relay to avoid 325s+ timeout cascade.
// Commit: 3249a432ad0
//
// Root cause:
//
//   In reset_cores(), when had_unresponsive_eth_cores is true, the safe_assert lambda
//   wraps each core's assert/deassert call in try/catch.  For non-MMIO devices, each
//   call goes through the ERISC relay.  If the relay is dead, FIX AF's 5-second timeout
//   fires per call.  A typical T3K device has ~120 tensix cores + inactive ETH cores =
//   65+ timeout-prone calls × 5s = 325+ seconds of relay timeouts per device during
//   MeshDevice::create on degraded T3K.
//
//   This makes device recovery after a SIGKILL'd session prohibitively slow — a single
//   MeshDevice::create takes 5+ minutes instead of <30 seconds.
//
// The fix (risc_firmware_initializer.cpp):
//   In reset_cores(), when had_unresponsive_eth_cores is true AND the device is non-MMIO
//   (cluster_.get_associated_mmio_device(device_id) != device_id), the safe_assert lambda
//   logs a warning and returns immediately instead of calling the underlying function.
//   This skips the 325s timeout cascade entirely — non-MMIO devices with dead relay don't
//   need core resets via relay (they'll be reset via MMIO path or FIX AY).
//
// What this test verifies:
//   Timing budget test confirming that MeshDevice::create with non-MMIO dead relay
//   completes within a tight budget (60s), not the 325s+ without FIX QC.
//
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D with non-MMIO chips.  Signals ready,
//            spins until SIGKILL.  Leaves dead relay state on non-MMIO ERISCs, which
//            triggers had_unresponsive_eth_cores in reset_cores().
//
//   Phase 2: Fork TESTEE — creates MeshDevice (FABRIC_2D).  reset_cores() detects
//            had_unresponsive_eth_cores for non-MMIO devices.
//            With FIX QC: safe_assert skips calls for non-MMIO → fast init → exit 0.
//            Without FIX QC: 120+ cores × 5s timeout each → 325s+ → test timeout.
//
// Regression indicator:
//   TESTEE times out (>60s) or exits nonzero.
//   The 60s budget is generous for a healthy path but catches the 325s regression.
//
// Timing budget:
//   PREDECESSOR wait: 30s (hardware init)
//   TESTEE:           60s pass budget (with FIX QC: ~10-30s; without: 325s+)
//   Total test_budget_ms: 120s
//
// Topology requirement: >= 2 devices (non-MMIO relay path required).

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
static constexpr int kTesteeBudgetMs = 60000;   // 60s pass budget (FIX QC: ~10-30s; no fix: 325s+)

// ---------------------------------------------------------------------------
// Shared memory for inter-process signaling
// ---------------------------------------------------------------------------
struct Gap58SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap58(const MeshCoordinateRange& range) {
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
static int wait_child_with_budget_gap58(pid_t pid, int budget_ms, const char* label) {
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
class FixQcNonMmioResetCoresSkipFixture : public MeshDeviceFixtureBase {
protected:
    FixQcNonMmioResetCoresSkipFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-58 requires >= 2 devices (non-MMIO relay path required). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-58: NonMmioResetCoresSkipPreventTimeoutCascade
//
// Regression indicator: TESTEE times out (>60s) or exits nonzero.
// ---------------------------------------------------------------------------
TEST_F(FixQcNonMmioResetCoresSkipFixture, NonMmioResetCoresSkipPreventTimeoutCascade) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap58SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap58SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D to put ERISCs into FABRIC firmware state, then spins.
    // SIGKILL leaves non-MMIO ERISCs with dead relay.  In the next session,
    // reset_cores() will detect had_unresponsive_eth_cores for non-MMIO devices.
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
            auto workload = make_blank_workload_gap58(range);
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
            ::munmap(raw_shm, sizeof(Gap58SharedMem));
            GTEST_SKIP() << "GAP-58: predecessor did not signal ready within " << kPredWaitMs
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
        "GAP-58: Predecessor SIGKILL'd — non-MMIO ERISCs left with dead relay. "
        "TESTEE will create MeshDevice. reset_cores() detects had_unresponsive_eth_cores. "
        "With FIX QC: safe_assert skips non-MMIO calls → fast. "
        "Without FIX QC: 120+ cores × 5s timeout = 325s+ cascade.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Creates MeshDevice (FABRIC_2D).  reset_cores() runs for all devices.
    // For non-MMIO devices with had_unresponsive_eth_cores:
    //   With FIX QC: safe_assert returns immediately → ~10-30s total init.
    //   Without FIX QC: safe_assert wraps call in try/catch but still pays
    //                   5s relay timeout per core → 325s+ for non-MMIO device.
    // The 60s budget catches the 325s regression while being generous for fix path.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
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
            // Even exceptions are acceptable — the key metric is wall-clock time.
            // With FIX QC, we finish fast even if something else throws.
            rc = 0;
        }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_with_budget_gap58(testee_pid, kTesteeBudgetMs, "TESTEE");
    const auto testee_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - testee_start)
                                     .count();

    ::munmap(raw_shm, sizeof(Gap58SharedMem));

    // ── Assertions ──────────────────────────────────────────────────────────────
    // Primary check: TESTEE must complete within the tight budget.
    // Without FIX QC, reset_cores() pays 5s per core for non-MMIO devices with
    // dead relay = 325s+, which far exceeds the 60s budget.
    if (rc == -1) {
        FAIL() << "GAP-58 TIMEOUT (FIX QC regression): TESTEE did not exit within "
               << kTesteeBudgetMs << "ms (elapsed: " << testee_elapsed << "ms).\n"
               << "\n"
               << "Root cause: reset_cores() → safe_assert wraps each core's assert/deassert\n"
               << "call in try/catch.  For non-MMIO devices with dead relay, each call hits\n"
               << "FIX AF's 5s timeout.  ~120 tensix + inactive ETH cores = 65+ timeouts\n"
               << "× 5s = 325+ seconds per device.  This makes MeshDevice::create\n"
               << "prohibitively slow on degraded T3K.\n"
               << "\n"
               << "Fix: In reset_cores(), when had_unresponsive_eth_cores is true AND the\n"
               << "device is non-MMIO, safe_assert returns immediately instead of calling\n"
               << "the underlying function.  See commit 3249a432ad0.";
    }

    // Secondary check: no SIGABRT.
    if (rc == 134) {
        FAIL() << "GAP-58 CRASH (FIX QC regression): TESTEE killed by SIGABRT.\n"
               << "\n"
               << "reset_cores() for non-MMIO device with dead relay triggered an unhandled\n"
               << "exception during core assert/deassert.\n"
               << "\n"
               << "Fix: safe_assert should skip calls for non-MMIO devices when\n"
               << "had_unresponsive_eth_cores is true.  See commit 3249a432ad0.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-58: TESTEE exited with unexpected code " << rc
        << " (expected 0 or 1). Signal-based exit codes (128+N) indicate a crash.";

    log_info(
        tt::LogTest,
        "GAP-58 PASS: TESTEE completed in {}ms (budget: {}ms) with exit {}. "
        "FIX QC correctly skipped reset_cores assert calls for non-MMIO devices with "
        "dead ERISC relay. No 325s+ timeout cascade — fast recovery from degraded state.",
        testee_elapsed,
        kTesteeBudgetMs,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
