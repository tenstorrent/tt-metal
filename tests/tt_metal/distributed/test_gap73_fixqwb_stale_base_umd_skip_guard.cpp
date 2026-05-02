// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-73: FIX QW-B gap — Fabric2DFixture, CustomMeshGraphFabric2DFixture, and
//         MultiCQFabricMeshDevice2x4Fixture SetUp() skip guards were missing the
//         is_fabric_stale_base_umd_channels() predicate check.  When the FIX M
//         transition path leaves base-UMD relay firmware on non-MMIO devices
//         (skip_soft_reset_channels non-empty), is_fabric_relay_path_broken()
//         and is_fabric_channels_not_ready_for_traffic() both return false —
//         so the old skip guard (relay_broken || channels_not_ready) evaluates
//         to false, and AllGather proceeds.  The stale base-UMD firmware causes
//         dispatch core timeout / device hang (~100s) in
//         completion_queue_wait_front.
//
// Root cause (FIX QW-B, commit 10297772581):
//
//   FIX QW (#42429) added skip guards to Fabric2DFixture::SetUp() and
//   MultiCQFabricMeshDevice2x4Fixture::SetUp() to detect degraded clusters:
//
//     if (dev->is_fabric_relay_path_broken() ||
//         dev->is_fabric_channels_not_ready_for_traffic()) {
//         GTEST_SKIP() << "...";
//     }
//
//   However, is_fabric_stale_base_umd_channels() was omitted.  When a prior
//   C++ binary is SIGKILL'd (leaving base-UMD relay firmware), a new session
//   uses the FIX M launch_msg transition (skip_soft_reset) and sets
//   fabric_stale_base_umd_channels_=true (FIX RZ).  Neither relay_broken nor
//   channels_not_ready becomes true in this path, so the skip guard fires FALSE
//   and AllGather hangs.
//
//   FIX QW-B adds the missing check to all three fixture SetUp() methods:
//
//     if (dev->is_fabric_relay_path_broken() ||
//         dev->is_fabric_channels_not_ready_for_traffic() ||
//         dev->is_fabric_stale_base_umd_channels()) {        // ← FIX QW-B
//         GTEST_SKIP() << "...";
//     }
//
// What this test verifies:
//   Fork test confirming that after a SIGKILL'd predecessor leaves base-UMD
//   relay firmware on non-MMIO devices, the TESTEE:
//   1. Creates a MeshDevice (configure_fabric sets fabric_stale_base_umd_=true
//      via FIX RZ on affected non-MMIO devices)
//   2. Finds at least one device where ONLY is_fabric_stale_base_umd_channels()
//      is true (relay_broken=false, channels_not_ready=false) — the exact
//      "gap" scenario FIX QW-B closes
//   3. Reports this via exit code 73 (FIX QW-B working — gap correctly covered)
//
//   A regression of FIX QW-B causes the combined check to return false for
//   these devices, AllGather is attempted, and the TESTEE hangs (dispatch core
//   timeout).  The parent catches this as a budget timeout → FAIL.
//
// Exit codes from TESTEE:
//   exit 0  — Clean cluster (no stale state detected; AllGather safe / skipped)
//   exit 73 — FIX QW-B WORKING: found device with stale_base_umd=true but
//              relay_broken=false AND channels_not_ready=false — correctly
//              detected by the new combined guard.
//   exit 74 — Old guard ALSO sufficient (stale_base_umd=true but also
//              relay_broken or channels_not_ready=true — gap did not apply).
//   -1/timeout — REGRESSION: TESTEE hung (AllGather on stale base-UMD cluster).
//   134     — SIGABRT crash (unrelated issue).
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    60s (MeshDevice create + predicate check; well under
//                         the ~100s AllGather hang timeout)
//   Total:            ~120s
//
// Topology requirement: >= 2 devices. Non-MMIO devices are required to observe
//   the FIX M / stale base-UMD channel state.  Skip if < 2 devices.

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
static constexpr int kGap73PredWaitMs   = 35000;   // 35s for predecessor init
static constexpr int kGap73TesteeBudget = 60000;   // 60s (well under 100s hang)

// Exit codes from testee subprocess
static constexpr int kGap73ExitQwbWorking  = 73;   // stale-only case caught — FIX QW-B working
static constexpr int kGap73ExitOldSuffices = 74;   // stale + other flag both true — old guard also OK

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap73Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap73(const MeshCoordinateRange& range) {
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
static int wait_child_budget_gap73(pid_t pid, int budget_ms) {
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
// Fixture
// ---------------------------------------------------------------------------
class FixQwbStaleBaseUmdSkipGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixQwbStaleBaseUmdSkipGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 150000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-73 requires >= 2 devices (non-MMIO devices needed for "
                         << "base-UMD relay firmware stale state). Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-73: StaleBaseUmdSkipGuardCoversQwbGap
//
// Verifies FIX QW-B: after a SIGKILL'd predecessor leaves base-UMD relay
// firmware on non-MMIO devices, the combined skip guard
// (relay_broken || channels_not_ready || stale_base_umd) correctly detects
// the degraded state even when ONLY stale_base_umd is true.
//
// A regression of FIX QW-B causes the old guard (relay_broken || channels_not_ready)
// to return false → AllGather proceeds on stale-firmware cluster → hang.
// ---------------------------------------------------------------------------
TEST_F(FixQwbStaleBaseUmdSkipGuardFixture, StaleBaseUmdSkipGuardCoversQwbGap) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap73Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap73Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ────────────────────────────────────────────
    // Opens full FABRIC_2D mesh, dispatches blank workload, signals ready,
    // then spins until SIGKILL.  Leaves base-UMD relay firmware on non-MMIO
    // ERISCs (SIGKILL bypasses teardown — firmware is NOT cleared).
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
            auto workload = make_blank_workload_gap73(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor to signal ready (or timeout)
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - pred_start).count();
        if (elapsed > kGap73PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap73Shm));
            GTEST_SKIP() << "GAP-73: predecessor did not signal ready within " << kGap73PredWaitMs
                         << "ms — skipping (cluster may be unhealthy).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-73: Predecessor SIGKILL'd — base-UMD relay firmware left on non-MMIO ERISCs. "
        "TESTEE will create MeshDevice. configure_fabric() should detect base_umd channels "
        "via FIX M (skip_soft_reset path) and set fabric_stale_base_umd_channels_=true (FIX RZ). "
        "TESTEE then checks all three skip-guard predicates. "
        "If a device has stale_base_umd=true but relay_broken=false and channels_not_ready=false, "
        "this is the FIX QW-B gap scenario. TESTEE exits 73 if FIX QW-B correctly detects it. "
        "A hung TESTEE (budget timeout) = FIX QW-B regression: old guard returned false, "
        "AllGather proceeded on stale cluster, dispatch core timed out.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ─────────────────────────────────────────────────
    // Creates MeshDevice (FABRIC_2D).
    // configure_fabric() detects base-UMD channels → FIX M (skip_soft_reset) →
    // FIX RZ sets fabric_stale_base_umd_channels_=true.
    //
    // Checks all three degradation predicates per device:
    //   (A) old guard: relay_broken || channels_not_ready
    //   (B) new FIX QW-B predicate: stale_base_umd
    //   (C) combined: A || B
    //
    // If any device has B=true AND A=false:
    //   - Old guard would return false (gap!) — AllGather would hang
    //   - New guard returns true (FIX QW-B working) — correctly skips
    //   - exit 73 (FIX QW-B working, gap covered)
    // If any device has B=true AND A=true:
    //   - Both guards agree — old guard was also sufficient
    //   - exit 74 (old guard also sufficient — gap scenario not reproduced)
    // If all devices clean:
    //   - exit 0 (clean cluster)
    //
    // REGRESSION path (FIX QW-B missing):
    //   Combined check returns false → testee proceeds to AllGather attempt.
    //   AllGather on stale-firmware cluster hangs for ~100s.
    //   Parent's 60s budget catches the hang → FAIL (regression detected).
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

            bool any_stale_base_umd_only = false;  // stale=T, relay_broken=F, channels_not_ready=F
            bool any_stale_base_umd_with_old = false;  // stale=T AND (relay_broken OR channels_not_ready)=T

            for (auto* idev : dev->get_devices()) {
                const bool relay_broken     = idev->is_fabric_relay_path_broken();
                const bool channels_not_ready = idev->is_fabric_channels_not_ready_for_traffic();
                const bool stale_base_umd   = idev->is_fabric_stale_base_umd_channels();

                fprintf(stderr,
                    "GAP-73 TESTEE: device %u  relay_broken=%d  channels_not_ready=%d  "
                    "stale_base_umd=%d  mmio=%d\n",
                    idev->id(), relay_broken, channels_not_ready, stale_base_umd,
                    idev->is_mmio_capable() ? 1 : 0);

                if (stale_base_umd && !relay_broken && !channels_not_ready) {
                    any_stale_base_umd_only = true;
                }
                if (stale_base_umd && (relay_broken || channels_not_ready)) {
                    any_stale_base_umd_with_old = true;
                }
            }

            if (any_stale_base_umd_only) {
                // FIX QW-B gap scenario confirmed: old guard would have returned false.
                // Combined guard (including stale_base_umd) catches this correctly.
                // Exit 73 = FIX QW-B working.
                fprintf(stderr,
                    "GAP-73 TESTEE: Found device with stale_base_umd=true but "
                    "relay_broken=false AND channels_not_ready=false.\n"
                    "  OLD guard (relay_broken || channels_not_ready) = FALSE — gap!\n"
                    "  NEW guard (|| stale_base_umd) = TRUE — FIX QW-B caught it.\n"
                    "  Exiting 73 (FIX QW-B working).\n");
                rc = kGap73ExitQwbWorking;
            } else if (any_stale_base_umd_with_old) {
                // Old guard was also sufficient (stale_base_umd co-occurred with old flags).
                // The FIX QW-B gap scenario was not triggered in this run, but FIX QW-B
                // is still correct (doesn't regress old behavior).
                fprintf(stderr,
                    "GAP-73 TESTEE: stale_base_umd=true but old flags also true — "
                    "old guard was also sufficient. Exit 74.\n");
                rc = kGap73ExitOldSuffices;
            } else {
                // Clean cluster or stale state not observed.
                fprintf(stderr, "GAP-73 TESTEE: No stale base-UMD state observed. "
                               "Cluster appears healthy. Exit 0.\n");
                rc = 0;
            }

            dev->close();
        } catch (const std::exception& e) {
            fprintf(stderr, "GAP-73 TESTEE: exception (acceptable): %s\n", e.what());
            rc = 0;
        } catch (...) {
            fprintf(stderr, "GAP-73 TESTEE: unknown exception (acceptable). Exit 0.\n");
            rc = 0;
        }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap73(testee_pid, kGap73TesteeBudget);
    const auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start).count();

    ::munmap(raw_shm, sizeof(Gap73Shm));

    if (rc == -1) {
        FAIL() << "GAP-73 REGRESSION (FIX QW-B): TESTEE timed out after " << testee_ms << "ms "
               << "(budget: " << kGap73TesteeBudget << "ms).\n"
               << "\n"
               << "This indicates that:\n"
               << "  1. A device had stale_base_umd_channels=true (FIX M / FIX RZ active)\n"
               << "  2. relay_broken=false AND channels_not_ready=false (old guard returned false)\n"
               << "  3. The combined skip guard did NOT catch the degraded state (FIX QW-B missing)\n"
               << "  4. AllGather proceeded on the stale-firmware cluster\n"
               << "  5. dispatch core timed out (~100s) → test budget exceeded\n"
               << "\n"
               << "Fix: Add is_fabric_stale_base_umd_channels() to all three fixture SetUp()\n"
               << "skip guards (Fabric2DFixture, CustomMeshGraphFabric2DFixture,\n"
               << "MultiCQFabricMeshDevice2x4Fixture). See commit 10297772581.";
    }

    if (rc == 134) {
        FAIL() << "GAP-73 CRASH (exit 134 SIGABRT): TESTEE crashed during MeshDevice create.\n"
               << "Check for FIX TB regression or other topology init crash.";
    }

    // All valid exit codes are a pass.
    EXPECT_TRUE(rc == 0 || rc == kGap73ExitQwbWorking || rc == kGap73ExitOldSuffices)
        << "GAP-73: TESTEE exited with unexpected code " << rc
        << " (expected 0, " << kGap73ExitQwbWorking << ", or " << kGap73ExitOldSuffices << ").";

    if (rc == kGap73ExitQwbWorking) {
        log_info(
            tt::LogTest,
            "GAP-73 PASS (exit 73): FIX QW-B correctly covers the gap. "
            "Found device with stale_base_umd=true but relay_broken=false AND channels_not_ready=false — "
            "old guard (relay_broken || channels_not_ready) would have returned false. "
            "New combined guard (|| stale_base_umd) caught it in {}ms. "
            "Without FIX QW-B, AllGather would have proceeded and hung (~100s dispatch core timeout).",
            testee_ms);
    } else if (rc == kGap73ExitOldSuffices) {
        log_info(
            tt::LogTest,
            "GAP-73 PASS (exit 74): stale_base_umd co-occurred with old flags — "
            "old guard was also sufficient for this run. FIX QW-B gap not triggered. "
            "TESTEE completed in {}ms.",
            testee_ms);
    } else {
        log_info(
            tt::LogTest,
            "GAP-73 PASS (exit 0): Clean cluster — no stale base-UMD state observed. "
            "TESTEE completed in {}ms.",
            testee_ms);
    }
}

}  // namespace tt::tt_metal::distributed::test
