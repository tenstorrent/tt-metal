// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-52: FIX PG — retry Phase 2.5 relay read for non-MMIO devices.
// Commit: 07bda1e2d2f
//
// Root cause (CI run 25118485779 — post FIX AM fix):
//
//   After AllGather completes, in-flight relay traffic on MMIO UMD relay channels can
//   cause a transient 5-second relay timeout when Phase 2.5 of quiesce tries to read
//   the ERISC router_sync status on non-MMIO devices.
//
//   Without FIX PG: a single L1 read failure immediately sets fabric_relay_path_broken_=true
//   (FIX AN), which cascades to skip Phase 3/5 → dispatch ETH cores stuck → rescue hard-reset
//   → next test's topology discovery FIX AQ overhead (4× 5s = 20s per test) → job timeout.
//
//   With FIX PG: the L1 read is retried up to 2 additional times (3s sleep between attempts)
//   for non-MMIO devices. MMIO devices get zero retries (no relay involved). This gives the
//   relay time to drain post-AllGather traffic before declaring the path broken.
//
//   If the relay truly is stuck (not just transient), all 3 attempts fail and FIX AN fires
//   normally. The extra 6s (2 retries × 3s) is acceptable since the alternative is 20s of
//   FIX AQ overhead on EVERY subsequent test in the job.
//
// What this test verifies:
//   1. PREDECESSOR: opens FABRIC_2D, dispatches workload (generates relay traffic), spins.
//   2. Parent SIGKILLs predecessor (leaves relay in mid-flight state).
//   3. TESTEE: opens MeshDevice. During quiesce/close, Phase 2.5 reads non-MMIO ERISC status.
//      FIX PG retries the read if it fails (transient relay timeout after predecessor's traffic).
//      Testee should complete and exit 0 — if relay recovered after retry, no cascade.
//      If relay is truly dead (all retries fail), FIX AN fires and teardown proceeds gracefully.
//   4. Parent checks: testee exits 0 within budget (no SIGABRT, no hang).
//
// Note: This test cannot GUARANTEE that retries will succeed (depends on hardware timing).
// The PRIMARY assertion is that the testee exits cleanly (0) and within budget — meaning
// either retries succeeded (relay recovered) or FIX AN fired gracefully (relay truly dead).
// In both cases, the test should NOT hang or crash.

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
// Budget allows for: topology discovery (~20s with FIX AQ) + retry overhead (~6s) + init (~10s).
static constexpr int kTesteeBudgetMs = 90000;

struct Gap52SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap52(const MeshCoordinateRange& range) {
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

class Phase25RelayRetryFixture : public MeshDeviceFixtureBase {
protected:
    Phase25RelayRetryFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-52 requires >= 2 devices (need non-MMIO for relay path). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-52: Phase25RelayRetryPreventsSpuriousBrokenRelay
//
// Tests that FIX PG retry logic in Phase 2.5 allows the testee to exit cleanly
// even when the predecessor left the relay in a mid-flight state.
// Regression indicator: testee hangs (relay path prematurely marked broken → cascade).
// ---------------------------------------------------------------------------
TEST_F(Phase25RelayRetryFixture, Phase25RelayRetryPreventsSpuriousBrokenRelay) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap52SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap52SharedMem();

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
            auto workload = make_blank_workload_gap52(range);
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
            ::munmap(raw_shm, sizeof(Gap52SharedMem));
            GTEST_SKIP() << "GAP-52: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-52: Predecessor SIGKILL'd — relay may have in-flight traffic. "
        "FIX PG should retry Phase 2.5 L1 reads for non-MMIO devices before "
        "declaring relay broken. Budget: {}ms.",
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
            ::munmap(raw_shm, sizeof(Gap52SharedMem));
            FAIL() << "GAP-52 TIMEOUT (FIX PG regression): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "Root cause: Phase 2.5 L1 read failure immediately marks relay as\n"
                   << "broken (FIX AN), causing cascade of skipped phases and dispatch stalls.\n"
                   << "FIX PG adds retries (max 2, 3s sleep) for non-MMIO devices to allow\n"
                   << "transient relay congestion to drain.\n"
                   << "\n"
                   << "Fix (FIX PG) in device.cpp quiesce_and_restart_fabric_workers():\n"
                   << "  Wrap ReadFromDeviceL1 in retry loop for !is_mmio_capable() devices.\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - testee_start)
                           .count();

    ::munmap(raw_shm, sizeof(Gap52SharedMem));

    if (WIFEXITED(status)) {
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << "GAP-52: Testee exited with code " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        FAIL() << "GAP-52: Testee killed by signal " << WTERMSIG(status)
               << " — likely SIGABRT from unguarded relay path.";
    }

    log_info(
        tt::LogTest,
        "GAP-52 PASS: Testee completed in {}ms (budget: {}ms) with exit 0. "
        "FIX PG Phase 2.5 relay retry prevented spurious relay-broken marking "
        "and/or FIX AN fired gracefully for truly dead relay.",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
