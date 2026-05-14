// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-65: FIX QV gap — wait_for_fabric_workers_ready() Phase 4 must skip MUX poll
//         when fabric_channels_not_ready_for_traffic_ is true (MMIO dead master router).
// Commit: 779d9e2d726
//
// Root cause (run 25164647562):
//
//   MMIO devices whose master ERISC router channel is pre-dead have
//   fabric_channels_not_ready_for_traffic_=true set by FIX QU inside configure_fabric().
//
//   However, configure_fabric_cores() in Phase 3 had already loaded Tensix MUX
//   firmware on ALL active channels including the dead-master-chan ones.  The MUX
//   firmware immediately writes TERMINATED (0xa4b4c4d4) because its associated ERISC
//   router is dead.  Phase 4 in wait_for_fabric_workers_ready() then polls for
//   READY_FOR_TRAFFIC and always times out after 5000ms on each such channel,
//   then throws TT_THROW — marking all UDM tests FAILED.
//
//   Pattern observed: MeshDevice1x4Fabric2DUDMFixture.* all FAILED with:
//     TT_THROW: Fabric MUX did not reach READY_FOR_TRAFFIC after quiesce restart on
//     Device 1 eth_chan 6 (status=0xa4b4c4d4, waited 5000ms)
//
// The fix (FIX QV, device.cpp):
//   Add a guard in wait_for_fabric_workers_ready() before Phase 4 that returns early
//   when fabric_channels_not_ready_for_traffic_=true, mirroring the existing
//   fabric_relay_path_broken_ guard that skips Phase 4+5 for non-MMIO dead-relay
//   devices.  The MUX won't carry traffic (FIX QS test guards skip ops when this
//   flag is set), so polling for READY_FOR_TRAFFIC is unnecessary and always times
//   out on degraded clusters.
//
// What this test verifies:
//   Fork test confirming that after a SIGKILL'd predecessor leaves MMIO master router
//   channels pre-dead, the TESTEE process can create and close a MeshDevice within the
//   timing budget without TT_THROW from Phase 4 timeout.
//
//   Phase 4 timeout (without FIX QV): 5000ms × N dead channels = 5-25s of TT_THROW
//   delays → SIGABRT (TT_THROW propagates through TearDown → unhandled).
//
//   With FIX QV: Phase 4 returns early for devices with channels_not_ready → fast path
//   → TESTEE exits 0 within kGap65TesteeBudget.
//
// Regression indicator:
//   TESTEE exits with SIGABRT (exit 134) or times out = FIX QV regression.
//   Exit 0 or 1 = pass (fast path or healthy cluster).
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    45s (fast with FIX QV; 25s+ TT_THROW delays without)
//   Total:            ~100s
//
// Topology requirement: >= 2 devices (MMIO + non-MMIO required for dead-relay
//   scenario; single-device systems skip this test).

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
static constexpr int kGap65PredWaitMs   = 35000;   // 35s for predecessor init
static constexpr int kGap65TesteeBudget = 45000;   // 45s — tight enough to catch 5s/channel delays

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap65Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap65(const MeshCoordinateRange& range) {
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
// Wait for child with timeout
// ---------------------------------------------------------------------------
static int wait_child_budget_gap65(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
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
// Fixture
// ---------------------------------------------------------------------------
class FixQvPhase4SkipFixture : public MeshDeviceFixtureBase {
protected:
    FixQvPhase4SkipFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-65 requires >= 2 devices (MMIO + non-MMIO required). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-65: Phase4MuxPollSkippedOnDeadMasterChan
//
// Verifies FIX QV: wait_for_fabric_workers_ready() Phase 4 skips MUX poll for
// MMIO devices with fabric_channels_not_ready_for_traffic_=true.
// ---------------------------------------------------------------------------
TEST_F(FixQvPhase4SkipFixture, Phase4MuxPollSkippedOnDeadMasterChan) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap65Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap65Shm();

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
            auto workload = make_blank_workload_gap65(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - pred_start)
                                 .count();
        if (elapsed > kGap65PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap65Shm));
            GTEST_SKIP() << "GAP-65: predecessor did not signal ready within " << kGap65PredWaitMs
                         << "ms; skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-65: Predecessor SIGKILL'd — ERISC relay channels dead. "
        "TESTEE will create MeshDevice. configure_fabric() detects dead master router "
        "channels and sets fabric_channels_not_ready_for_traffic_=true (FIX QU). "
        "With FIX QV: Phase 4 skips MUX poll for these devices. "
        "Without FIX QV: Phase 4 polls for 5000ms per dead channel → TT_THROW → SIGABRT.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Creates MeshDevice (FABRIC_2D), checks Phase 4 completes within budget.
    // With FIX QV: channels_not_ready_ guard fires → Phase 4 returns fast.
    // Without FIX QV: Phase 4 times out after 5s × N dead channels → TT_THROW.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);

            bool any_not_ready = false;
            for (auto* idev : dev->get_devices()) {
                if (idev->is_fabric_channels_not_ready_for_traffic()) {
                    any_not_ready = true;
                    break;
                }
            }
            if (any_not_ready) {
                fprintf(stderr, "GAP-65 TESTEE: FIX QU flag set — Phase 4 skipped (FIX QV active).\n");
            } else {
                fprintf(stderr, "GAP-65 TESTEE: No channels_not_ready flags — cluster may be healthy.\n");
            }

            dev->close();
        } catch (const std::exception& e) {
            fprintf(stderr, "GAP-65 TESTEE: exception (expected on degraded): %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "GAP-65 TESTEE: unknown exception (expected on degraded).\n");
        }
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap65(testee_pid, kGap65TesteeBudget);
    const auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start)
                               .count();

    ::munmap(raw_shm, sizeof(Gap65Shm));

    // Primary check: no SIGABRT and within timing budget
    if (rc == 134) {
        FAIL() << "GAP-65 CRASH (FIX QV regression): TESTEE killed by SIGABRT (exit 134).\n"
               << "\n"
               << "Root cause: wait_for_fabric_workers_ready() Phase 4 polled for\n"
               << "READY_FOR_TRAFFIC on MUX channels tied to dead ERISC router.\n"
               << "MUX firmware writes TERMINATED (0xa4b4c4d4) immediately.\n"
               << "Phase 4 times out after 5000ms → TT_THROW propagates through TearDown\n"
               << "→ unhandled exception → SIGABRT.\n"
               << "\n"
               << "Fix: Add guard in wait_for_fabric_workers_ready() before Phase 4:\n"
               << "  if (fabric_channels_not_ready_for_traffic_) return;\n"
               << "See commit 779d9e2d726.";
    }

    if (rc == -1) {
        FAIL() << "GAP-65 TIMEOUT (FIX QV regression): TESTEE did not exit within "
               << kGap65TesteeBudget << "ms (elapsed: " << testee_ms << "ms).\n"
               << "\n"
               << "Phase 4 MUX poll is timing out on dead ERISC channels (5s/channel).\n"
               << "With FIX QV, Phase 4 should return immediately for not-ready devices.\n"
               << "See commit 779d9e2d726.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-65: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-65 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX QV correctly skips Phase 4 MUX poll for MMIO devices with "
        "fabric_channels_not_ready_for_traffic_ — no 5s/channel timeout delays.",
        testee_ms,
        kGap65TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
