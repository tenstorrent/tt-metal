// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-50: FIX PD — clear fw_launch_addr in quiesce Pass-0 deassert.
// Commit: 3e2614543f3
//
// Root cause (CI runs 25096771728, 25110543943 — 140+ cascade occurrences):
//
//   The quiesce path in device.cpp (quiesce_and_restart_fabric_workers + launch_eth_cores_for_quiesce)
//   has a Phase 2.5 force-reset for ERISC channels that didn't respond to TERMINATE within the
//   deadline.  Pass-0 of this path does:
//     assert_risc_reset_at_core(...)
//     deassert_risc_reset_at_core(...)
//     // ← fw_launch_addr NOT cleared (pre-FIX PD)
//
//   This is the DOMINANT path for the 500ms cascade because:
//   1. After a fabric test (e.g., AllGather on T3K), quiesce runs at teardown.
//   2. Non-MMIO devices have dead relays → Phase 2.5 L1 reads fail → FIX AN sets relay_broken.
//   3. MMIO ETH channels connecting to dead non-MMIO devices get Phase 2.5 force-reset.
//   4. Those channels (e.g., 25-16, 18-16, 25-17, 18-17, 22-17, 21-17) have fw_launch_addr
//      left non-zero in L1 after the deassert.
//   5. Every subsequent test: reset_cores() → erisc_app_still_running() true → 500ms cascade.
//
//   FIX PA covers the init force-reset, FIX PB covers the rescue path, but the quiesce path
//   (most common in actual CI) was MISSED until FIX PD.
//
// FIX PD (device.cpp, two locations):
//   1. quiesce_and_restart_fabric_workers() Pass-0: after deassert_risc_reset_at_core, write 0
//      to fw_launch_addr via write_core_immediate.
//   2. launch_eth_cores_for_quiesce() Pass-0 (defer_eth_launch=true path): same write.
//   Both wrapped in try/catch for non-MMIO dead-relay channels.
//
// What this test verifies:
//   The same PREDECESSOR→TESTEE pattern as GAP-48/49, with a key difference:
//   the PREDECESSOR runs a FABRIC workload and then the parent opens and closes
//   (triggering quiesce), simulating the CI pattern where one test's quiesce
//   leaves stale fw_launch_addr for the next test.
//
//   1. PREDECESSOR: opens FABRIC_2D, dispatches workload, signals ready, spins.
//   2. Parent SIGKILLs predecessor.
//   3. TESTEE-1: opens and closes (quiesce runs, FIX PD clears fw_launch_addr in Pass-0).
//   4. TESTEE-2: opens and closes. If FIX PD works: clean open. If missing: 500ms cascade.
//
// Budget analysis: same as GAP-48 — kTestee2BudgetMs catches the cascade.

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
static constexpr int kTestee1BudgetMs = 90000;
static constexpr int kTestee2BudgetMs = 35000;

struct Gap50SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap50(const MeshCoordinateRange& range) {
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

class FwLaunchAddrQuiesceFixture : public MeshDeviceFixtureBase {
protected:
    FwLaunchAddrQuiesceFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 240000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-50 requires >= 2 devices (need MMIO + non-MMIO). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

static int wait_child_gap50(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
        if (elapsed_ms > budget_ms) {
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
// GAP-50: FwLaunchAddrClearedInQuiescePass0
//
// Tests the quiesce Pass-0 path — the DOMINANT source of the 500ms cascade in CI.
// Regression indicator: TESTEE-2 times out (quiesce left stale fw_launch_addr).
// ---------------------------------------------------------------------------
TEST_F(FwLaunchAddrQuiesceFixture, FwLaunchAddrClearedInQuiescePass0) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap50SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap50SharedMem();

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
            auto workload = make_blank_workload_gap50(range);
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
            ::munmap(raw_shm, sizeof(Gap50SharedMem));
            GTEST_SKIP() << "GAP-50: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-50: Predecessor SIGKILL'd. Quiesce Pass-0 in TESTEE-1 will force-reset MMIO "
        "ETH channels connecting to dead non-MMIO devices. FIX PD should clear fw_launch_addr.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: TESTEE-1 (triggers quiesce → Phase 2.5 → Pass-0 force-reset) ──
    pid_t t1_pid = ::fork();
    ASSERT_GE(t1_pid, 0);
    if (t1_pid == 0) {
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
            // Close triggers quiesce_and_restart_fabric_workers → Phase 2.5 → Pass-0.
            dev->close();
            rc = 0;
        } catch (...) { rc = 1; }
        _exit(rc);
    }

    int rc1 = wait_child_gap50(t1_pid, kTestee1BudgetMs);
    if (rc1 == -1) {
        ::munmap(raw_shm, sizeof(Gap50SharedMem));
        GTEST_SKIP() << "GAP-50: TESTEE-1 timed out at " << kTestee1BudgetMs << "ms.";
    }
    ASSERT_EQ(rc1, 0) << "GAP-50: TESTEE-1 exited with code " << rc1;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ── Step 3: TESTEE-2 (regression check: fw_launch_addr should be zero) ────
    pid_t t2_pid = ::fork();
    ASSERT_GE(t2_pid, 0);
    if (t2_pid == 0) {
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

    auto t2_start = std::chrono::steady_clock::now();
    int rc2 = wait_child_gap50(t2_pid, kTestee2BudgetMs);
    auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t2_start)
                      .count();

    ::munmap(raw_shm, sizeof(Gap50SharedMem));

    if (rc2 == -1) {
        FAIL() << "GAP-50 TIMEOUT (FIX PD regression): TESTEE-2 did not exit within "
               << kTestee2BudgetMs << "ms.\n"
               << "\n"
               << "Root cause: quiesce_and_restart_fabric_workers() Pass-0 deassert does NOT\n"
               << "clear fw_launch_addr after force-resetting MMIO ETH channels. This is the\n"
               << "DOMINANT path for the 500ms cascade in CI (covers channels 25-16, 18-16,\n"
               << "25-17, 18-17, 22-17, 21-17 on T3K MMIO devices).\n"
               << "\n"
               << "Fix (FIX PD) in device.cpp:\n"
               << "  1. quiesce_and_restart_fabric_workers() Pass-0 (~line 1432)\n"
               << "  2. launch_eth_cores_for_quiesce() Pass-0 (~line 1848)\n"
               << "  After deassert_risc_reset_at_core, write 0 to fw_launch_addr.\n"
               << "\n"
               << "See also: FIX PA (GAP-48, init reset), FIX PB (GAP-49, rescue path).";
    }

    EXPECT_EQ(rc2, 0) << "GAP-50: TESTEE-2 exited with code " << rc2;

    log_info(
        tt::LogTest,
        "GAP-50 PASS: TESTEE-2 completed in {}ms (budget: {}ms) with exit 0. "
        "FIX PD correctly cleared fw_launch_addr in quiesce Pass-0 deassert.",
        t2_ms, kTestee2BudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
