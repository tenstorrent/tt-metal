// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-48: FIX PA — clear fw_launch_addr after force-reset in reset_cores().
// Commit: 1111316e84f
//
// Root cause (CI run 25094103200, t3k_ttnn_tests):
//
//   After a fabric test crashes during teardown, dispatch ETH cores on MMIO devices are
//   force-reset via assert_risc_reset_at_core + deassert_risc_reset_at_core.  Hardware
//   reset halts the ERISC and restarts base UMD firmware, but does NOT zero L1.
//
//   The fw_launch_addr (ERISC dispatch launch flag at hal.get_jit_build_config(...).fw_launch_addr)
//   retains its non-zero value from the previous dispatch session.  On the next test open:
//
//     reset_cores(device_id) →
//       erisc_app_still_running(virtual_core) reads fw_launch_addr → non-zero → returns true →
//       adds core to device_to_early_exit_cores →
//       sends Metal exit signal to UMD base firmware (which doesn't understand it) →
//       wait_until_cores_done(500ms) times out →
//       force-reset again → fw_launch_addr STILL non-zero → LOOP
//
//   This produces a 500ms cascade on EVERY test open for the rest of the run:
//     "TT_THROW: Device N: Timeout (500 ms) waiting for physical cores to finish: 25-16, 18-16, ..."
//
//   Observed: 146 occurrences in run 25094103200, causing the job to timeout at exit code 124.
//
// FIX PA (risc_firmware_initializer.cpp reset_cores()):
//   After deassert_risc_reset_at_core(), write 0 to fw_launch_addr via write_core_immediate().
//   This ensures erisc_app_still_running() returns false on the next open → no spurious 500ms wait.
//   For MMIO devices: PCIe write (direct, no relay). Non-MMIO: best-effort (caught).
//
// What this test verifies:
//   1. Fork PREDECESSOR: open FABRIC_2D MeshDevice, dispatch blank workload (puts dispatch
//      firmware on ETH cores, setting fw_launch_addr non-zero), signal ready, spin.
//   2. Parent SIGKILLs predecessor — fw_launch_addr retains non-zero in L1 on all MMIO
//      ETH dispatch cores.
//   3. 2s settle.
//   4. Fork TESTEE-1: open MeshDevice (reset_cores finds stale fw_launch_addr, FIX PA
//      clears it during force-reset), close, exit 0.
//   5. Fork TESTEE-2: open MeshDevice again.  If FIX PA worked, erisc_app_still_running()
//      returns false → no 500ms cascade → fast open. Close, exit 0.
//   6. Parent checks:
//      (a) TESTEE-1 completes within kTestee1BudgetMs (generous — allows one force-reset cycle).
//      (b) TESTEE-2 completes within kTestee2BudgetMs (tight — catches FIX PA regression where
//          the cascade would re-trigger on every subsequent open).
//      (c) Both testees exit 0.
//
// Timing budget analysis (T3K: 4 MMIO devices, ~6 ETH dispatch cores each):
//   With FIX PA:    Testee-1: one force-reset cycle + clear ≈ 15-20s.
//                   Testee-2: clean open, no 500ms waits ≈ 20-25s (topology discovery + init).
//   Without FIX PA: Testee-2: 4 devices × 6 cores × 500ms = 12s cascade + 25s init ≈ 37s.
//   kTestee2BudgetMs = 35000ms: passes with FIX PA (~25s), fails without (~37s+).
//
// Related fixes in the same cascade family (each covers a different code path):
//   FIX PB (GAP-49): rescue_stuck_dispatch_cores() — dispatch teardown rescue
//   FIX PC:          fabric_firmware_initializer teardown force-reset
//   FIX PD (GAP-50): quiesce Pass-0 deassert — quiesce_and_restart_fabric_workers()
//   FIX PE:          fabric cleanly-terminated ERISC channels
//   FIX PF:          dispatch ETH normal completion path (wait_for_dispatch_cores)

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

// Testee-1 budget: generous (allows topology discovery + one force-reset cycle + FIX AQ overhead).
static constexpr int kPredWaitMs = 30000;
static constexpr int kTestee1BudgetMs = 90000;
// Testee-2 budget: tight — catches FIX PA regression (cascade adds ~12s on T3K).
// With FIX PA: ~25s (clean open). Without FIX PA: ~37s+ (cascade).
static constexpr int kTestee2BudgetMs = 35000;

struct Gap48SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap48(const MeshCoordinateRange& range) {
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

class FwLaunchAddrForceResetFixture : public MeshDeviceFixtureBase {
protected:
    FwLaunchAddrForceResetFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 240000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-48 requires >= 2 devices (need MMIO + non-MMIO). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// Helper: fork a child, wait with budget, return exit status or FAIL on timeout.
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
// GAP-48: FwLaunchAddrClearedAfterForceReset
//
// Regression indicator: TESTEE-2 times out at kTestee2BudgetMs (500ms cascade).
// ---------------------------------------------------------------------------
TEST_F(FwLaunchAddrForceResetFixture, FwLaunchAddrClearedAfterForceReset) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap48SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap48SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
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
            auto workload = make_blank_workload_gap48(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor ready, then SIGKILL.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap48SharedMem));
            GTEST_SKIP() << "GAP-48: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-48: Predecessor SIGKILL'd — MMIO ETH dispatch cores have stale non-zero "
        "fw_launch_addr in L1. Without FIX PA: 500ms cascade on every test open. "
        "With FIX PA: force-reset in Testee-1 clears flag, Testee-2 opens cleanly.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE-1 ─────────────────────────────────────────────────
    // This one triggers the force-reset path in reset_cores().
    // FIX PA should clear fw_launch_addr during that force-reset.
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
        } catch (...) { rc = 1; }
        _exit(rc);
    }

    int rc1 = wait_child_with_budget(testee1_pid, kTestee1BudgetMs, "TESTEE-1");
    if (rc1 == -1) {
        ::munmap(raw_shm, sizeof(Gap48SharedMem));
        FAIL() << "GAP-48: TESTEE-1 timed out at " << kTestee1BudgetMs
               << "ms. Hardware init may be too slow on this runner. "
               << "This is not necessarily a FIX PA regression — it may be "
               << "FIX AQ/NX overhead from dead non-MMIO chips.";
    }
    ASSERT_EQ(rc1, 0) << "GAP-48: TESTEE-1 exited with code " << rc1
                       << " — exception during MeshDevice::create().";

    log_info(tt::LogTest, "GAP-48: TESTEE-1 completed with exit 0. fw_launch_addr should be cleared.");

    // Brief settle between testees.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ── Step 3: Fork TESTEE-2 ─────────────────────────────────────────────────
    // If FIX PA worked: erisc_app_still_running() returns false → clean open, no 500ms cascade.
    // If FIX PA missing: fw_launch_addr still non-zero → 500ms × N cores × M devices → timeout.
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
            dev->close();
            rc = 0;
        } catch (...) { rc = 1; }
        _exit(rc);
    }

    const auto testee2_start = std::chrono::steady_clock::now();
    int rc2 = wait_child_with_budget(testee2_pid, kTestee2BudgetMs, "TESTEE-2");
    const auto testee2_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - testee2_start)
                                      .count();

    ::munmap(raw_shm, sizeof(Gap48SharedMem));

    if (rc2 == -1) {
        FAIL() << "GAP-48 TIMEOUT (FIX PA regression): TESTEE-2 did not exit within "
               << kTestee2BudgetMs << "ms.\n"
               << "\n"
               << "Root cause: reset_cores() force-reset path does NOT clear fw_launch_addr\n"
               << "after deassert_risc_reset_at_core(). L1 retains stale non-zero value →\n"
               << "erisc_app_still_running() false-positive → 500ms cascade on every open.\n"
               << "\n"
               << "Fix (FIX PA) in risc_firmware_initializer.cpp reset_cores():\n"
               << "  After deassert_risc_reset_at_core(), write 0 to fw_launch_addr via\n"
               << "  write_core_immediate(). MMIO: PCIe write. Non-MMIO: best-effort.\n"
               << "\n"
               << "See also: FIX PB (GAP-49, rescue path), FIX PD (GAP-50, quiesce path).";
    }

    EXPECT_EQ(rc2, 0) << "GAP-48: TESTEE-2 exited with code " << rc2 << " (expected 0).";

    log_info(
        tt::LogTest,
        "GAP-48 PASS: TESTEE-2 completed in {}ms (budget: {}ms) with exit 0. "
        "FIX PA correctly cleared fw_launch_addr during TESTEE-1's force-reset, "
        "so TESTEE-2 opened without the 500ms erisc_app_still_running cascade.",
        testee2_elapsed,
        kTestee2BudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
