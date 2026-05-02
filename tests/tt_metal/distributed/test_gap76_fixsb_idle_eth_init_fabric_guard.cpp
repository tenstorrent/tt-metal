// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-76: FIX SB gap — initialize_and_launch_firmware() for IDLE_ETH cores
//         unconditionally inserted virtual_core into not_done_cores, even when
//         initialize_firmware() returned early (because INIT_FABRIC was absent
//         from FabricManagerMode).  This caused a TT_FATAL in wait_until_cores_done.
//
// Root cause (FIX SB, commit 92e3edfa4cc):
//
//   In RiscFirmwareInitializer::initialize_and_launch_firmware(), the IDLE_ETH
//   branch calls initialize_firmware() with HalProgrammableCoreType::IDLE_ETH.
//   Inside initialize_firmware(), the IDLE_ETH / ACTIVE_ETH case is guarded by:
//
//     if (!has_flag(descriptor_->fabric_manager(), FabricManagerMode::INIT_FABRIC))
//         break;   // ← returns early: no go_msg written, no risc reset asserted
//
//   But the calling loop in initialize_and_launch_firmware() then did:
//
//     initialize_firmware(device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, ...);
//     not_done_cores.insert(virtual_core);  // ← UNCONDITIONAL — BUG
//
//   When INIT_FABRIC is absent, initialize_firmware() wrote nothing to the core.
//   deassert_risc_reset_at_core() is still called for every core in not_done_cores.
//   The IDLE_ETH core then starts executing stale L1 firmware that writes 0x55 to
//   run_mailbox.  wait_until_cores_done() TT_FATALs because it never sees the
//   expected completion sentinel.
//
//   FIX SB adds the guard:
//     if (has_flag(descriptor_->fabric_manager(), FabricManagerMode::INIT_FABRIC))
//         not_done_cores.insert(virtual_core);
//
// What this test verifies:
//
//   1. FIX SB: Opening a MeshDevice with FabricManagerMode that does NOT include
//      INIT_FABRIC (i.e., TERMINATE_FABRIC only) completes without SIGABRT or hang.
//      Without FIX SB, the IDLE_ETH core would be added to not_done_cores even
//      though it was never initialized → deassert_risc_reset_at_core fires → stale
//      L1 firmware → wait_until_cores_done TT_FATAL → exit 134 (SIGABRT).
//
//   2. FIX PG: A subsequent open with FabricManagerMode::DEFAULT succeeds even
//      after the TERMINATE_FABRIC-only open.  FIX PG ensures that when ALL MMIO
//      ETH heartbeats time out during teardown, FIX AY (deferred ERISC reset) is
//      skipped rather than wasting N × 5s of poll-for-relay timeouts.
//      Without FIX PG: teardown with dead-heartbeat MMIO devices would enter FIX AY
//      for every non-MMIO device, each polling 5s × num_channels times.  On T3K
//      (4 non-MMIO × up to 4 channels) that is up to 80s extra before next session
//      can open — hidden hang that looked like the next test's freeze.
//
//   TESTEE: Subprocess opens an 8-device MeshDevice (or minimum 2) configured via
//     SetFabricConfig(FABRIC_1D, ..., FabricManagerMode::TERMINATE_FABRIC, ...)
//   This exposes IDLE_ETH cores to initialize_and_launch_firmware() without INIT_FABRIC.
//   After open+close: a second open/close with FabricManagerMode::DEFAULT verifies
//   the cluster is still healthy.
//
// Exit codes from TESTEE:
//   exit 0   — Both opens succeeded: FIX SB + FIX PG working
//   exit 76  — First open crashed or hung (FIX SB regression): IDLE_ETH
//              inserted into not_done_cores without INIT_FABRIC guard
//   exit 77  — Second open (DEFAULT mode) failed after TERMINATE_FABRIC pass
//              (FIX PG regression or other teardown regression)
//   exit 134 — SIGABRT: TT_FATAL in firmware initializer → FIX SB missing
//   -1/timeout — hang in open or close → either FIX SB or FIX PG regression
//
// Timing budgets:
//   TESTEE open (TERMINATE_FABRIC only):  30s
//   TESTEE close (TERMINATE_FABRIC only): 20s
//   TESTEE open (DEFAULT mode):           30s
//   TESTEE close (DEFAULT mode):          20s
//   Total with margin:                    ~120s (generous for cold cluster)
//
// Topology requirement: >= 2 devices (IDLE_ETH cores exist on Wormhole multi-chip
//   systems).  Skip if < 2 devices.
//
// Note: This test calls SetFabricConfig before MeshDevice::create() — this matches
//   how MeshDeviceFixtureBase::SetUp() and the allgather harness call it.
//   It does NOT depend on any GTest fixture for device lifecycle.
//
// ─────────────────────────────────────────────────────────────────────────────

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
#include <experimental/fabric/fabric.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::distributed::test {

// ─── Shared-memory result struct ────────────────────────────────────────────

struct TestResult {
    std::atomic<int> phase{0};    // 0=init, 1=open_terminate, 2=close_terminate,
                                  // 3=open_default, 4=close_default, 99=done
    std::atomic<int> exit_code{0};
};

static_assert(sizeof(TestResult) <= 256, "TestResult too large for mmap page");

// ─── Helper: wait for child with timeout ─────────────────────────────────────

static int wait_child_timeout(pid_t pid, int timeout_s) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
            if (WIFEXITED(status)) return WEXITSTATUS(status);
            if (WIFSIGNALED(status)) return -1;  // signal/SIGABRT
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return -2;  // timeout
}

// ─── Test ─────────────────────────────────────────────────────────────────────

TEST(Gap76FixSbIdleEthInitFabricGuard, TerminateFabricModeDoesNotHangWaitUntilCoresDone) {
    if (tt::tt_metal::GetNumAvailableDevices() < 2) {
        GTEST_SKIP() << "GAP-76: requires >= 2 devices (IDLE_ETH cores needed for regression scenario)";
    }

    // ── Shared memory so testee can report which phase it reached ────────────
    void* mem = mmap(
        nullptr, sizeof(TestResult), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mem, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* result = new (mem) TestResult{};
    auto cleanup_mem = [&] { munmap(mem, sizeof(TestResult)); };

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork failed: " << strerror(errno);

    if (pid == 0) {
        // ──────────────────── TESTEE ──────────────────────────────────────────
        // Phase 1: Open with FabricManagerMode::TERMINATE_FABRIC only.
        // This exercises initialize_and_launch_firmware() with IDLE_ETH cores
        // but WITHOUT INIT_FABRIC in FabricManagerMode.
        // Without FIX SB: not_done_cores.insert() fires for an uninitialized
        // IDLE_ETH core → deassert_risc_reset_at_core → stale L1 firmware →
        // wait_until_cores_done TT_FATAL → SIGABRT → exit 134.
        result->phase.store(1, std::memory_order_release);

        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_1D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                std::nullopt,          // num_routing_planes default
                tt_fabric::FabricTensixConfig::DISABLED,
                tt_fabric::FabricUDMMode::DISABLED,
                tt_fabric::FabricManagerMode::TERMINATE_FABRIC);  // ← NO INIT_FABRIC

            auto mesh = MeshDevice::create(
                MeshDeviceConfig{MeshShape{1, tt::tt_metal::GetNumAvailableDevices()}});
            result->phase.store(2, std::memory_order_release);

            mesh->close();
            result->phase.store(3, std::memory_order_release);

        } catch (const std::exception& e) {
            log_warning(tt::LogAlways, "GAP-76 TESTEE phase 1/2 threw: {}", e.what());
            // Catch to get a clean exit, but mark failure
            result->exit_code.store(76, std::memory_order_release);
            result->phase.store(99, std::memory_order_release);
            _exit(76);
        }

        // Phase 2: Re-open with DEFAULT mode — confirms cluster still openable
        // after the TERMINATE_FABRIC-only session.
        // FIX PG ensures that when MMIO heartbeats fail during the prior teardown,
        // FIX AY was skipped (not wasting N×5s of dead-relay polls), so the
        // cluster recovers promptly.
        result->phase.store(3, std::memory_order_release);
        try {
            tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::FABRIC_1D);  // DEFAULT mode

            auto mesh = MeshDevice::create(
                MeshDeviceConfig{MeshShape{1, tt::tt_metal::GetNumAvailableDevices()}});
            result->phase.store(4, std::memory_order_release);
            mesh->close();
            result->phase.store(99, std::memory_order_release);
        } catch (const std::exception& e) {
            log_warning(tt::LogAlways, "GAP-76 TESTEE phase 2 threw: {}", e.what());
            result->exit_code.store(77, std::memory_order_release);
            result->phase.store(99, std::memory_order_release);
            _exit(77);
        }

        result->phase.store(99, std::memory_order_release);
        _exit(0);
    }

    // ──────────────────── PARENT ──────────────────────────────────────────────
    int child_exit = wait_child_timeout(pid, 120);

    int phase = result->phase.load(std::memory_order_acquire);
    cleanup_mem();

    if (child_exit == -2) {
        FAIL() << "GAP-76 TESTEE timed out (120s) in phase " << phase << ".\n"
               << "  Phase 1=open(TERMINATE_FABRIC), 2=close(TERMINATE_FABRIC),\n"
               << "  3=open(DEFAULT), 4=close(DEFAULT), 99=done.\n"
               << "  Likely regression: FIX SB or FIX PG missing — hang in\n"
               << "  wait_until_cores_done or FIX AY dead-relay poll loop.";
    }

    if (child_exit == -1) {
        FAIL() << "GAP-76 TESTEE exited by signal (SIGABRT?) in phase " << phase << ".\n"
               << "  FIX SB REGRESSION: initialize_and_launch_firmware() added IDLE_ETH core\n"
               << "  to not_done_cores without INIT_FABRIC guard.\n"
               << "  Core started from stale L1 firmware → wait_until_cores_done TT_FATAL.";
    }

    if (child_exit == 76) {
        FAIL() << "GAP-76 TESTEE exited 76: exception in TERMINATE_FABRIC open/close (phase " << phase << ").\n"
               << "  FIX SB REGRESSION or new firmware-initializer error path.";
    }

    if (child_exit == 77) {
        FAIL() << "GAP-76 TESTEE exited 77: DEFAULT-mode open failed after TERMINATE_FABRIC session.\n"
               << "  FIX PG REGRESSION or teardown left cluster in non-recoverable state.\n"
               << "  Check FIX AY timeout logs: if relay not restored, FIX AY should be skipped.";
    }

    // exit 0: both opens/closes succeeded
    EXPECT_EQ(child_exit, 0)
        << "GAP-76: unexpected exit code " << child_exit << " (phase " << phase << ")";
}

}  // namespace tt::tt_metal::distributed::test
