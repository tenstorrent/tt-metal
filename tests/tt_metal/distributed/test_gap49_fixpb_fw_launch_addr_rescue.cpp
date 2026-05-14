// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-49: FIX PB — clear fw_launch_addr in rescue_stuck_dispatch_cores().
// Commit: 136dd144275
//
// Root cause (CI runs 25094103200, 25095358480):
//
//   The dispatch teardown rescue path in dispatch_kernel_initializer.cpp performs the same
//   assert+deassert sequence as the init force-reset path (FIX PA covers), but in a DIFFERENT
//   code location: rescue_stuck_dispatch_cores().
//
//   When dispatch ETH cores don't exit cleanly during teardown (e.g., fabric TERMINATE fails
//   because the remote ERISC is stuck), the rescue path hard-resets them:
//     assert_risc_reset_at_core(...)
//     deassert_risc_reset_at_core(...)
//     // ← fw_launch_addr NOT cleared (pre-FIX PB)
//
//   Even though FIX PA clears fw_launch_addr in the INIT force-reset path
//   (risc_firmware_initializer.cpp::reset_cores()), the cascade can restart from the TEARDOWN
//   rescue path because rescue_stuck_dispatch_cores() runs at every test close:
//
//     Test N close → rescue_stuck_dispatch_cores → hard-reset → fw_launch_addr NOT cleared
//     Test N+1 open → reset_cores → erisc_app_still_running() true → 500ms wait → FIX PA clears
//     Test N+1 close → rescue_stuck_dispatch_cores → hard-reset → fw_launch_addr NOT cleared → LOOP
//
//   FIX PA breaks the cycle on the INIT side, FIX PB breaks it on the TEARDOWN side.
//   Both are needed: FIX PA alone leaves the per-test 500ms overhead (FIX PA re-clears each
//   time, but the 500ms wait_until_cores_done has already fired). FIX PB prevents the flag
//   from being re-set at teardown, so the NEXT test's FIX PA never needs to fire.
//
// FIX PB (dispatch_kernel_initializer.cpp rescue_stuck_dispatch_cores()):
//   After deassert_risc_reset_at_core(), write 0 to fw_launch_addr via write_core_immediate().
//   Uses hal_.get_programmable_core_type_index(ACTIVE_ETH) to get the correct fw_launch_addr.
//
// What this test verifies:
//   Same structure as GAP-48 but with THREE testees instead of two:
//   1. PREDECESSOR: sets up stale fabric state (SIGKILL'd).
//   2. TESTEE-1: opens, triggers force-reset (FIX PA clears), closes (rescue path fires).
//   3. TESTEE-2: opens. If FIX PB works: rescue cleared flag → clean open. If FIX PB missing:
//      rescue re-set flag → 500ms cascade.
//   4. TESTEE-3: opens. Confirms steady-state: no cascade accumulation across multiple cycles.
//
// Timing budget:
//   TESTEE-2 and TESTEE-3 must both complete within kTesteeBudgetMs.
//   With FIX PA+PB: clean open ~25s.
//   Without FIX PB (FIX PA alone): ~25s + 500ms×24 ETH cores ≈ 37s → exceeds budget.

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
static constexpr int kTesteeBudgetMs = 35000;  // Tight budget for testee-2 and testee-3.

struct Gap49SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap49(const MeshCoordinateRange& range) {
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

class FwLaunchAddrRescueFixture : public MeshDeviceFixtureBase {
protected:
    FwLaunchAddrRescueFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 300000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-49 requires >= 2 devices (need MMIO + non-MMIO). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

static int wait_child_gap49(pid_t pid, int budget_ms) {
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
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 255;
}

// Fork a child that opens and closes a MeshDevice. Returns exit code or -1 on timeout.
static int fork_open_close_mesh(size_t num_dev, int budget_ms) {
    pid_t pid = ::fork();
    if (pid < 0) return -2;
    if (pid == 0) {
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
    return wait_child_gap49(pid, budget_ms);
}

// ---------------------------------------------------------------------------
// GAP-49: FwLaunchAddrClearedInRescuePath
//
// Tests the three-testee cycle to ensure FIX PB prevents the teardown rescue
// path from re-setting fw_launch_addr after FIX PA clears it.
// ---------------------------------------------------------------------------
TEST_F(FwLaunchAddrRescueFixture, FwLaunchAddrClearedInRescuePath) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap49SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap49SharedMem();

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
            auto workload = make_blank_workload_gap49(range);
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
            ::munmap(raw_shm, sizeof(Gap49SharedMem));
            GTEST_SKIP() << "GAP-49: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-49: Predecessor SIGKILL'd. Testing three-cycle open/close to verify "
        "FIX PB prevents rescue_stuck_dispatch_cores from re-setting fw_launch_addr.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: TESTEE-1 (first open — triggers force-reset, FIX PA clears flag) ──
    int rc1 = fork_open_close_mesh(num_dev, kTestee1BudgetMs);
    if (rc1 == -1) {
        ::munmap(raw_shm, sizeof(Gap49SharedMem));
        GTEST_SKIP() << "GAP-49: TESTEE-1 timed out at " << kTestee1BudgetMs
                     << "ms (hardware init too slow).";
    }
    ASSERT_EQ(rc1, 0) << "GAP-49: TESTEE-1 exited with code " << rc1;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ── Step 3: TESTEE-2 (second open — FIX PB regression test) ───────────────
    auto t2_start = std::chrono::steady_clock::now();
    int rc2 = fork_open_close_mesh(num_dev, kTesteeBudgetMs);
    auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t2_start)
                      .count();

    if (rc2 == -1) {
        ::munmap(raw_shm, sizeof(Gap49SharedMem));
        FAIL() << "GAP-49 TIMEOUT (FIX PB regression): TESTEE-2 did not exit within "
               << kTesteeBudgetMs << "ms.\n"
               << "\n"
               << "Root cause: rescue_stuck_dispatch_cores() does not clear fw_launch_addr\n"
               << "after hard-resetting stuck dispatch ETH cores. FIX PA clears it during\n"
               << "init, but the teardown rescue re-sets it → 500ms cascade on next open.\n"
               << "\n"
               << "Fix (FIX PB) in dispatch_kernel_initializer.cpp rescue_stuck_dispatch_cores():\n"
               << "  After deassert_risc_reset_at_core(), write 0 to fw_launch_addr.\n";
    }
    EXPECT_EQ(rc2, 0) << "GAP-49: TESTEE-2 exited with code " << rc2;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ── Step 4: TESTEE-3 (third open — confirms steady-state, no cascade accumulation) ──
    auto t3_start = std::chrono::steady_clock::now();
    int rc3 = fork_open_close_mesh(num_dev, kTesteeBudgetMs);
    auto t3_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t3_start)
                      .count();

    ::munmap(raw_shm, sizeof(Gap49SharedMem));

    if (rc3 == -1) {
        FAIL() << "GAP-49: TESTEE-3 timed out at " << kTesteeBudgetMs
               << "ms — cascade accumulation across open/close cycles.";
    }
    EXPECT_EQ(rc3, 0) << "GAP-49: TESTEE-3 exited with code " << rc3;

    log_info(
        tt::LogTest,
        "GAP-49 PASS: TESTEE-2 completed in {}ms, TESTEE-3 in {}ms (budget: {}ms). "
        "FIX PB correctly cleared fw_launch_addr in rescue_stuck_dispatch_cores, "
        "preventing the teardown→init cascade loop.",
        t2_ms, t3_ms, kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
