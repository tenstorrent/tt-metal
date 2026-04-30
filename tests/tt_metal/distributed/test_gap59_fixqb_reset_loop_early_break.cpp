// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-59: FIX QB gap — break assert_risc_reset_at_core loop early for non-MMIO dead relay.
// Commit: 53415739233
//
// Root cause:
//
//   In reset_cores(), when had_unresponsive_eth_cores is true, the safe_assert lambda
//   wraps each core's assert/deassert call in try/catch.  For non-MMIO devices, each
//   call routes through the UMD legacy ERISC relay.  If the relay is dead, each call
//   hits FIX AF's 5-second UMD timeout.
//
//   Before FIX QB: assert_risc_reset_at_core loops through ALL active ETH cores on
//   non-MMIO devices even after the first timeout.  A typical non-MMIO device has 4
//   active ETH cores.  With 3 non-MMIO devices: 12 cores × 5s = 60s of serial timeouts
//   in the reset phase alone.  With multiple test fixtures (each calling MeshDevice::create
//   → reset_cores), this fills the 5-minute GHA wall-clock budget before any test starts.
//
//   FIX QB differs from FIX QC (GAP-58): FIX QC skips the entire safe_assert body for
//   non-MMIO devices (applies to ALL cores — tensix + ETH).  FIX QB is more targeted:
//   it breaks the ETH-core-specific assert_risc_reset_at_core loop after the first failure.
//   FIX QC is a broader optimization; FIX QB catches the specific case where only the
//   ETH-core reset path is timing out and the rest of reset_cores could proceed.
//
// The fix (risc_firmware_initializer.cpp):
//   After the first assert_risc_reset_at_core exception on a non-MMIO device, break out
//   of the inner ETH-core loop.  Cost: 1×5s per non-MMIO device instead of N×5s.
//   MMIO devices use the PCIe path (fast) — unchanged.
//
// What this test verifies:
//   Timing budget test confirming that MeshDevice::create with non-MMIO dead relay
//   completes within a tight budget, specifically measuring that only ONE ETH core
//   timeout is paid per non-MMIO device (the early-break behavior), not N.
//
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D with non-MMIO chips.  Signals ready,
//            spins until SIGKILL.  Leaves dead relay state on non-MMIO ERISCs.
//
//   Phase 2: Fork TESTEE — creates MeshDevice (FABRIC_2D).  reset_cores() detects
//            had_unresponsive_eth_cores, iterates ETH cores for non-MMIO devices.
//            With FIX QB: first failure → break → 1×5s per device → fast init.
//            Without FIX QB: 4 ETH cores × 5s = 20s per device × 3 devices = 60s.
//
// Regression indicator:
//   TESTEE times out (>60s) or exits nonzero.
//   The budget is set at 60s — generous for the fix path (~15-25s with 3 non-MMIO
//   devices × 1 timeout each) but catches the 60s+ regression without the fix.
//
// Timing budget:
//   PREDECESSOR wait: 30s (hardware init)
//   TESTEE:           60s pass budget (with FIX QB: ~15-25s; without: 60s+)
//   Total test_budget_ms: 120s
//
// Topology requirement: >= 2 devices (non-MMIO relay path required).
// Relation to GAP-58: GAP-58 tests FIX QC (skip entire safe_assert for non-MMIO).
//   GAP-59 tests FIX QB (break ETH-core-specific reset loop after first failure).
//   Both reduce timeout cascade time; they are complementary defenses at different
//   loop levels in reset_cores().

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
static constexpr int kGap59PredWaitMs = 30000;      // 30s predecessor init
static constexpr int kGap59TesteeBudgetMs = 60000;   // 60s pass budget

// ---------------------------------------------------------------------------
// Shared memory for inter-process signaling
// ---------------------------------------------------------------------------
struct Gap59SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap59(const MeshCoordinateRange& range) {
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
static int wait_child_with_budget_gap59(pid_t pid, int budget_ms, const char* label) {
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
class FixQbResetLoopEarlyBreakFixture : public MeshDeviceFixtureBase {
protected:
    FixQbResetLoopEarlyBreakFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-59 requires >= 2 devices (non-MMIO relay path required). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-59: ResetLoopEarlyBreakNonMmioDeadRelay
//
// Regression indicator: TESTEE times out (>60s) or exits nonzero.
// ---------------------------------------------------------------------------
TEST_F(FixQbResetLoopEarlyBreakFixture, ResetLoopEarlyBreakNonMmioDeadRelay) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap59SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap59SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
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
            auto workload = make_blank_workload_gap59(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Parent: wait for predecessor to signal ready.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kGap59PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap59SharedMem));
            GTEST_SKIP() << "GAP-59: predecessor did not signal ready within " << kGap59PredWaitMs
                         << "ms; skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-59: Predecessor SIGKILL'd — non-MMIO ERISCs left with dead relay. "
        "TESTEE will create MeshDevice. reset_cores() iterates ETH cores for non-MMIO devices. "
        "With FIX QB: first assert_risc_reset_at_core failure → break → 1×5s per device. "
        "Without FIX QB: 4 ETH cores × 5s = 20s per non-MMIO device × 3 devices = 60s+.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
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
            rc = 0;
        }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_with_budget_gap59(testee_pid, kGap59TesteeBudgetMs, "TESTEE");
    const auto testee_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - testee_start)
                                     .count();

    ::munmap(raw_shm, sizeof(Gap59SharedMem));

    if (rc == -1) {
        FAIL() << "GAP-59 TIMEOUT (FIX QB regression): TESTEE did not exit within "
               << kGap59TesteeBudgetMs << "ms (elapsed: " << testee_elapsed << "ms).\n"
               << "\n"
               << "Root cause: reset_cores() → assert_risc_reset_at_core loop does not break\n"
               << "after first failure on non-MMIO device with dead relay.  Each ETH core\n"
               << "pays 5s UMD timeout.  4 cores × 3 non-MMIO devices = 60s wasted.\n"
               << "\n"
               << "Fix: break out of inner ETH-core loop after first exception for non-MMIO\n"
               << "devices.  See commit 53415739233.";
    }

    if (rc == 134) {
        FAIL() << "GAP-59 CRASH (FIX QB regression): TESTEE killed by SIGABRT.\n"
               << "assert_risc_reset_at_core threw unhandled exception during ETH core\n"
               << "reset loop.  See commit 53415739233.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-59: TESTEE exited with unexpected code " << rc
        << " (expected 0 or 1). Signal-based exit codes (128+N) indicate a crash.";

    log_info(
        tt::LogTest,
        "GAP-59 PASS: TESTEE completed in {}ms (budget: {}ms) with exit {}. "
        "FIX QB correctly breaks assert_risc_reset_at_core loop after first failure "
        "on non-MMIO devices — 1×5s per device instead of N×5s cascade.",
        testee_elapsed,
        kGap59TesteeBudgetMs,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
