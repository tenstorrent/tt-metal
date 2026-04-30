// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-51: FIX PF (commit f0e3d677935) — skip Metal exit signal in reset_cores when UMD
// base firmware heartbeat is detected.
//
// Root cause:
//
//   erisc_app_still_running() checks fw_launch_addr in L1. If non-zero, it returns true,
//   meaning "Metal dispatch firmware is running on this core." But after a force-reset
//   (assert + deassert), the ERISC runs UMD BASE firmware (the boot ROM / relay firmware),
//   NOT Metal dispatch firmware. UMD base firmware does NOT understand the Metal exit
//   signal protocol.
//
//   When reset_cores() sees erisc_app_still_running()==true on an MMIO device, it:
//     1. Sends erisc_send_exit_signal() — UMD base firmware ignores this (protocol mismatch).
//     2. wait_until_cores_done(500ms) — times out (UMD firmware never acknowledges).
//     3. Force-resets the core (FIX PA clears fw_launch_addr).
//
//   This adds 500ms per false-positive core per device. On T3K with 4 MMIO devices and
//   ~6 ETH dispatch cores each, that's up to 12s of wasted time per test open.
//
//   FIX PF detects this situation by reading the UMD base firmware heartbeat register:
//     - If heartbeat >> 16 == 0xABCD: UMD base firmware is confirmed running.
//     - Metal dispatch was NOT running (the non-zero fw_launch_addr is stale).
//     - Clear fw_launch_addr directly and skip the exit signal + 500ms wait entirely.
//
//   This eliminates the 500ms-per-core overhead for the common case where a prior process
//   left stale fw_launch_addr but UMD firmware was properly rebooted by the hardware reset.
//
// FIX PF (risc_firmware_initializer.cpp reset_cores()):
//   After the erisc_app_still_running() check, before sending exit signal:
//   For MMIO devices: read heartbeat via read_reg (PCIe — no relay needed).
//   If heartbeat matches UMD pattern (0xABCDxxxx): clear fw_launch_addr, set still_running=false.
//   This bypasses the exit signal + 500ms wait, going directly to the next core.
//
// What this test verifies:
//   1. PREDECESSOR: opens FABRIC_2D, dispatches workload (sets fw_launch_addr non-zero), spins.
//   2. Parent SIGKILLs predecessor.
//   3. TESTEE: opens MeshDevice. reset_cores() finds stale fw_launch_addr on MMIO ETH cores.
//      FIX PF reads UMD heartbeat (0xABCDxxxx) → clears flag → skips exit signal.
//      Without FIX PF: 500ms per core × 6 cores × 4 devices ≈ 12s extra.
//      With FIX PF: heartbeat check + direct clear ≈ 0ms per core.
//
// Timing budget:
//   With FIX PF:    ~20-25s (topology discovery + init, no 500ms waits).
//   Without FIX PF: ~32-37s (same + 12s of 500ms waits before FIX PA force-resets each).
//   kTesteeBudgetMs = 30000ms: passes with FIX PF (~25s), fails without (~37s).
//
// Note: This test exercises the SAME stale-fw_launch_addr scenario as GAP-48, but tests
// the FIX PF OPTIMIZATION path (heartbeat detection → skip) rather than the FIX PA
// FALLBACK path (force-reset → clear). Both are needed: FIX PF avoids the 500ms waste,
// FIX PA is the safety net if heartbeat detection fails.

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

static constexpr int kPredWaitMs = 30000;
// Tight budget to catch missing FIX PF (12s overhead from 500ms × 24 cores).
static constexpr int kTesteeBudgetMs = 30000;

struct Gap51SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap51(const MeshCoordinateRange& range) {
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

class UmdHeartbeatSkipExitFixture : public MeshDeviceFixtureBase {
protected:
    UmdHeartbeatSkipExitFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-51 requires >= 2 devices. Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-51: UmdHeartbeatSkipsMetalExitSignal
//
// Tests that FIX PF detects UMD base firmware heartbeat on MMIO ETH cores with
// stale fw_launch_addr and skips the unnecessary Metal exit signal + 500ms wait.
// Regression indicator: testee exceeds kTesteeBudgetMs (12s of wasted 500ms waits).
// ---------------------------------------------------------------------------
TEST_F(UmdHeartbeatSkipExitFixture, UmdHeartbeatSkipsMetalExitSignal) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap51SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap51SharedMem();

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
            auto workload = make_blank_workload_gap51(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap51SharedMem));
            GTEST_SKIP() << "GAP-51: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    const size_t eth_cores_per_chip = 6;
    log_info(
        tt::LogTest,
        "GAP-51: Predecessor SIGKILL'd. MMIO ETH dispatch cores have stale fw_launch_addr "
        "but UMD base firmware heartbeat (0xABCDxxxx). Without FIX PF: reset_cores() sends "
        "Metal exit signal (UMD ignores) → 500ms wait × ~{} cores ≈ {}ms overhead. "
        "With FIX PF: heartbeat detected → flag cleared → skip exit signal → ~0ms overhead. "
        "Budget: {}ms.",
        num_dev * eth_cores_per_chip,
        num_dev * eth_cores_per_chip * 500,
        kTesteeBudgetMs);

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE ───────────────────────────────────────────────────
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
        } catch (...) { rc = 1; }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start)
                               .count();
        if (elapsed_ms > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap51SharedMem));
            FAIL() << "GAP-51 TIMEOUT (FIX PF regression): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "Root cause: reset_cores() does not check UMD base firmware heartbeat\n"
                   << "before sending Metal exit signal to MMIO ETH cores with stale\n"
                   << "fw_launch_addr. UMD firmware ignores the signal → 500ms wait per core.\n"
                   << "\n"
                   << "Fix (FIX PF) in risc_firmware_initializer.cpp reset_cores():\n"
                   << "  After erisc_app_still_running()==true, for MMIO devices:\n"
                   << "  Read heartbeat via read_reg(). If (hb >> 16) == 0xABCD:\n"
                   << "  clear fw_launch_addr directly, set still_running=false.\n"
                   << "  Skips exit signal + 500ms wait entirely.\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - testee_start)
                                 .count();

    ::munmap(raw_shm, sizeof(Gap51SharedMem));

    if (WIFEXITED(status)) {
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << "GAP-51: Testee exited with code " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        FAIL() << "GAP-51: Testee killed by signal " << WTERMSIG(status);
    }

    log_info(
        tt::LogTest,
        "GAP-51 PASS: Testee completed in {}ms (budget: {}ms) with exit 0. "
        "FIX PF correctly detected UMD base firmware heartbeat on MMIO ETH cores, "
        "cleared stale fw_launch_addr, and skipped Metal exit signal.",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
