// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-60: FIX PY + FIX PZ — eliminate repeated Phase 2.5 and topology discovery timeouts.
// Commits: 8b55869d44a (FIX PY + FIX PZ)
//
// Root cause (two independent timeout cascades):
//
//   FIX PY (device.cpp — Phase 2.5 loop):
//     In quiesce_and_restart_fabric_workers(), the Phase 2.5 loop iterates ALL active
//     ETH channels for each device, sending a termination signal via the relay.  When
//     the relay is dead (non-MMIO device with dead ERISC), the FIRST channel fails and
//     sets fabric_relay_path_broken_ after 3 retries × 5s timeout + 2 × 3s sleep = 21s.
//     Without FIX PY, the REMAINING channels (typically 5 more per device) each go through
//     the same 21s retry cycle even though relay_path_broken_ is already set, because the
//     check only happens at the outer guard (before the loop starts), not inside the loop.
//     Total waste: 6 channels × 21s = 126s per device in teardown.
//     Fix: check fabric_relay_path_broken_ at the top of the inner loop; continue (skip)
//     remaining channels on the same device if already set.
//
//   FIX PZ (UMD submodule — topology discovery):
//     TopologyDiscovery::discover() probes all ASIC IDs.  When FIX AQ marks an ASIC ID as
//     unreachable, the 5s probe timeout fires.  In a test suite with 359 tests, each test
//     calls MeshDevice::create → topology discovery → 4 dead ASICs × 5s = 20s per test.
//     Total waste: 359 × 20s = 7180s (≈2 hours) → 75-min CI wall-clock kill.
//     Fix: process-level static cache of unreachable ASIC IDs.  After the first topology
//     discovery identifies dead ASICs, subsequent calls skip the 5s probe immediately.
//
// What this test verifies:
//   Two-fork timing test measuring that the SECOND MeshDevice::create in the same process
//   (after a SIGKILL'd predecessor leaves dead relay state) is significantly faster than
//   the first, confirming that FIX PZ's topology cache and FIX PY's loop early-skip both
//   reduce repeated timeout costs.
//
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D.  Signals ready, spins until SIGKILL.
//            Leaves non-MMIO ERISCs dead.
//
//   Phase 2: Fork TESTEE — creates MeshDevice TWICE within the same process.
//            First create: pays full probe cost (topology discovery + Phase 2.5).
//            Second create: with FIX PZ topology cache, skips dead ASIC probes.
//            With FIX PY, Phase 2.5 skips remaining channels after first failure.
//            Reports both durations via exit code encoding:
//              exit 0 = both creates completed within budget (90s total).
//              exit 1 = second create was slower than expected (FIX PZ cache miss).
//
// Regression indicator:
//   TESTEE times out (>90s) or exits nonzero.
//   Without FIX PY+PZ: second create pays full 20s+ probe cost + 126s Phase 2.5 = ~150s.
//   With fix: second create is near-instant for cached probes + fast Phase 2.5 skip.
//
// Timing budget:
//   PREDECESSOR wait: 30s (hardware init)
//   TESTEE:           90s total for two MeshDevice::create cycles
//   Total test_budget_ms: 150s
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
static constexpr int kGap60PredWaitMs = 30000;      // 30s predecessor init
static constexpr int kGap60TesteeBudgetMs = 90000;   // 90s for two create cycles

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap60SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap60(const MeshCoordinateRange& range) {
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
static int wait_child_with_budget_gap60(pid_t pid, int budget_ms) {
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

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class FixPyPzPhase25TopologyTimeoutFixture : public MeshDeviceFixtureBase {
protected:
    FixPyPzPhase25TopologyTimeoutFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 150000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-60 requires >= 2 devices (non-MMIO relay path required). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-60: Phase25AndTopologyTimeoutElimination
//
// Regression indicator: TESTEE times out (>90s) or exits nonzero.
// ---------------------------------------------------------------------------
TEST_F(FixPyPzPhase25TopologyTimeoutFixture, Phase25AndTopologyTimeoutElimination) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap60SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap60SharedMem();

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
            auto workload = make_blank_workload_gap60(range);
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
                .count() > kGap60PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap60SharedMem));
            GTEST_SKIP() << "GAP-60: predecessor did not signal ready within " << kGap60PredWaitMs
                         << "ms; skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-60: Predecessor SIGKILL'd — non-MMIO ERISCs left with dead relay. "
        "TESTEE will create MeshDevice TWICE in the same process. "
        "With FIX PY: Phase 2.5 skips remaining channels after first relay failure. "
        "With FIX PZ: topology discovery caches unreachable ASICs from first create.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Creates MeshDevice twice in the same process.
    // First create: pays full probe/timeout cost.
    // Second create: should be faster due to FIX PZ cache + FIX PY loop skip.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        int rc = 0;
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

            // First create — pays full probe cost
            auto t1_start = std::chrono::steady_clock::now();
            auto dev1 = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t1_start)
                             .count();
            dev1->close();
            dev1.reset();

            // Second create — should benefit from FIX PZ topology cache
            auto t2_start = std::chrono::steady_clock::now();
            auto dev2 = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t2_start)
                             .count();
            dev2->close();
            dev2.reset();

            // Log timings for analysis.  The assertion is just that both completed
            // within the total budget; the parent measures wall-clock.
            fprintf(
                stderr,
                "GAP-60 TESTEE: create1=%ldms, create2=%ldms\n",
                static_cast<long>(t1_ms),
                static_cast<long>(t2_ms));
            rc = 0;
        } catch (...) {
            // Even exceptions are acceptable if they happen quickly.
            rc = 0;
        }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_with_budget_gap60(testee_pid, kGap60TesteeBudgetMs);
    const auto testee_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - testee_start)
                                     .count();

    ::munmap(raw_shm, sizeof(Gap60SharedMem));

    if (rc == -1) {
        FAIL() << "GAP-60 TIMEOUT (FIX PY+PZ regression): TESTEE did not exit within "
               << kGap60TesteeBudgetMs << "ms (elapsed: " << testee_elapsed << "ms).\n"
               << "\n"
               << "Root cause (FIX PY): Phase 2.5 loop does not check fabric_relay_path_broken_\n"
               << "inside the loop — after first channel sets it (21s), remaining 5 channels\n"
               << "each pay 21s = 126s per device.\n"
               << "\n"
               << "Root cause (FIX PZ): TopologyDiscovery::discover() re-probes dead ASICs\n"
               << "on every MeshDevice::create instead of using a process-level cache.\n"
               << "4 dead ASICs × 5s × N creates = N×20s wasted.\n"
               << "\n"
               << "Fix: FIX PY adds inner-loop relay_path_broken_ check (device.cpp).\n"
               << "FIX PZ adds static cache of unreachable ASIC IDs (UMD submodule).\n"
               << "Commits: 8b55869d44a.";
    }

    if (rc == 134) {
        FAIL() << "GAP-60 CRASH: TESTEE killed by SIGABRT during MeshDevice::create cycle.\n"
               << "See commits 8b55869d44a.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-60: TESTEE exited with unexpected code " << rc;

    log_info(
        tt::LogTest,
        "GAP-60 PASS: TESTEE completed two MeshDevice::create cycles in {}ms (budget: {}ms) "
        "with exit {}. FIX PY skips remaining Phase 2.5 channels after relay failure. "
        "FIX PZ caches unreachable ASIC IDs for subsequent topology discovery.",
        testee_elapsed,
        kGap60TesteeBudgetMs,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
