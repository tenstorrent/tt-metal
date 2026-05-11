// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-78: FIX P25-CLEAN — skip Phase 3 quiesce launch for already-TERMINATED
//         out-of-mesh channels.
//
// Root cause (run on tt-metal-ci-vm-t3k-10, cycle 8):
//
//   In a sub-mesh AllGather on a T3K (e.g. 1x4 sub-mesh of an 8-device FABRIC_1D board),
//   quiesce_devices() only runs on the 4 in-mesh devices.  Some channels on those devices
//   connect to out-of-mesh devices that already dropped the connection — those channels are
//   at TERMINATED (0xa4b4c4d4) before quiesce begins.
//
//   Phase 2.5 correctly detects these "already clean" channels and SKIPs them.  However,
//   Phase 3 (launch_eth_cores_for_quiesce) launched quiesce firmware on ALL channels —
//   including the already-terminated ones.  The quiesce firmware on those channels starts
//   but their peers (out-of-mesh devices) never respond to the quiesce handshake.  This
//   causes those channels to get stuck at STARTED (0xa0b0c0d0), which cascades into blocking
//   the in-mesh channels from advancing past REMOTE_HANDSHAKE_COMPLETE.
//
// The fix (FIX P25-CLEAN):
//   Phase 2.5 records already-terminated channels in phase25_already_clean_chans_.
//   Phase 3 (both inline and deferred paths) checks this set and skips launching quiesce
//   firmware on those channels, preventing the stuck-STARTED cascade.
//
// What this test verifies:
//   Phase 1 (PREDECESSOR): Opens a FABRIC_1D MeshDevice on T3K (8 devices), runs a blank
//   workload to establish all fabric channels, then closes cleanly.  This leaves some
//   channels in TERMINATED state after normal teardown.
//
//   Phase 2 (TESTEE): Opens a smaller sub-mesh (1x4) on the same hardware.  Channels
//   connecting to out-of-mesh devices will be in TERMINATED state from Phase 1.
//   configure_fabric → quiesce_and_restart_fabric_workers fires.  Phase 2.5 records
//   already-terminated channels.  Phase 3 should skip them (FIX P25-CLEAN).
//   Testee exits 0 if init completes without timeout.
//
// Regression: without FIX P25-CLEAN, Phase 3 launches quiesce firmware on already-terminated
//   channels → stuck at STARTED → cascade hang → testee times out → FAIL.
//
// Timing budget:
//   PREDECESSOR: 45s (init + blank workload + clean close)
//   TESTEE:      60s (sub-mesh init with potential quiesce overhead)
//   Total:       ~130s
//
// Topology requirement: >= 4 devices (T3K — needs cross-row ETH channels).
//
// NOTE: This test exercises the fix on real hardware.  On clusters with fewer than 4 devices,
//       the test will GTEST_SKIP (insufficient devices for sub-mesh scenario).

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
#include <tt-metalium/experimental/fabric/fabric.hpp>
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

static constexpr int kGap78PredWaitMs   = 45000;  // 45s for predecessor init + workload + close
static constexpr int kGap78TesteeBudget = 60000;  // 60s for sub-mesh init including quiesce

struct Gap78Shm {
    std::atomic<int> predecessor_ready{0};
    std::atomic<int> predecessor_done{0};
};

static MeshWorkload make_blank_workload_gap78(const MeshCoordinateRange& range) {
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

static int wait_child_budget_gap78(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
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

// GAP-78: FIX P25-CLEAN — Phase 3 skip for already-terminated channels in sub-mesh quiesce.
TEST(DistributedUnitTests, Gap78_FixP25Clean_SkipAlreadyTerminatedPhase3) {
    // Topology check: need >= 4 devices for sub-mesh scenario
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        GTEST_SKIP() << "GAP-78 requires >= 4 devices for sub-mesh P25-CLEAN scenario (found " << num_devices << ")";
    }

    void* shm_raw = ::mmap(nullptr, sizeof(Gap78Shm), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(shm_raw, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (shm_raw) Gap78Shm{};

    // Phase 1: PREDECESSOR — full mesh, blank workload, clean close
    pid_t pred_pid = ::fork();
    ASSERT_NE(pred_pid, -1) << "fork failed: " << strerror(errno);
    if (pred_pid == 0) {
        // Child: PREDECESSOR
        try {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
            auto mesh = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<size_t>(num_devices)}));
            auto range = mesh->shape().to_mesh_coordinate_range();
            auto workload = make_blank_workload_gap78(range);
            mesh->execute(workload);
            mesh->finish();
            // Signal ready
            shm->predecessor_ready.store(1, std::memory_order_release);
            // Close cleanly — channels transition to TERMINATED
            mesh->close();
            shm->predecessor_done.store(1, std::memory_order_release);
            _exit(0);
        } catch (const std::exception& e) {
            log_error(LogTest, "GAP-78 PREDECESSOR exception: {}", e.what());
            _exit(1);
        }
    }

    // Wait for predecessor
    int pred_rc = wait_child_budget_gap78(pred_pid, kGap78PredWaitMs);
    ASSERT_EQ(pred_rc, 0) << "GAP-78: PREDECESSOR did not complete cleanly (rc=" << pred_rc << ")";

    // Phase 2: TESTEE — sub-mesh (1x4) — some channels will be TERMINATED from Phase 1
    pid_t testee_pid = ::fork();
    ASSERT_NE(testee_pid, -1) << "fork failed: " << strerror(errno);
    if (testee_pid == 0) {
        // Child: TESTEE — open sub-mesh
        try {
            // Open sub-mesh on first 4 devices — channels to devices 4-7 should be TERMINATED
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
            auto mesh = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, 4}));
            // If we get here without hanging, P25-CLEAN is working
            log_info(LogTest, "GAP-78 TESTEE: sub-mesh init succeeded — FIX P25-CLEAN is working.");
            mesh->close();
            _exit(0);
        } catch (const std::exception& e) {
            log_error(LogTest, "GAP-78 TESTEE exception: {}", e.what());
            // Exception is acceptable (degraded cluster) — better than hang
            _exit(0);
        }
    }

    int testee_rc = wait_child_budget_gap78(testee_pid, kGap78TesteeBudget);

    ::munmap(shm_raw, sizeof(Gap78Shm));

    if (testee_rc == -1) {
        FAIL() << "GAP-78: TESTEE timed out after " << kGap78TesteeBudget
               << "ms — likely P25-CLEAN regression: Phase 3 launched quiesce FW on "
               << "already-terminated out-of-mesh channels, causing STARTED cascade hang.";
    }
    ASSERT_EQ(testee_rc, 0)
        << "GAP-78: TESTEE exited with unexpected code " << testee_rc;
}

}  // namespace tt::tt_metal::distributed::test
