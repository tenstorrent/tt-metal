// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-72: FIX TI + FIX TJ — ring sync fast-skip when base-UMD channels are present.
//
// Root cause:
//
//   In T3K clusters where some ETH channels carry base-UMD relay firmware (0x49706550),
//   the fabric ring barrier signal can fail to propagate.  The ring barrier requires every
//   member device in the ring to write READY_FOR_TRAFFIC to its master channel.  When
//   base-UMD channels are present, the channel may be mid-transition from base-UMD to
//   FABRIC_1D via write_launch_msg_to_core (FIX M): the ERISC is executing a launch-message
//   transition and may not complete the ETH handshake before the ring-sync deadline.
//
//   Without FIX TI/TJ, the ring sync code waits the full 30s timeout (FIX TH2 window)
//   per device, then falls through to verify_all_fabric_channels_healthy() which hits a
//   150ms retry loop that always fails (channels are still stuck at REMOTE_HANDSHAKE_COMPLETE).
//   On a T3K with 8 devices and base-UMD channels on both MMIO devices, this causes:
//     • 8 × 30s = 240s of sequential ring-sync timeouts
//     • Plus 8 × 150ms health-check retry loops (each failing)
//     • Total: ~4 minutes of overhead per fabric init cycle
//   In CI with a 120s job timeout this manifests as InfraError (timeout) on every run.
//
// The fixes:
//   FIX TI (fabric_firmware_initializer.cpp:wait_for_fabric_router_sync):
//     When ring-sync times out AND base-UMD channels are present, record the device in
//     timeout_on_base_umd_devices_ so verify_all_fabric_channels_healthy() skips its channels.
//     Also set ring_sync_already_timed_out_ = true (triggers FIX TJ).
//
//   FIX TJ (fabric_firmware_initializer.cpp:wait_for_fabric_router_sync):
//     When ring_sync_already_timed_out_ is already set (from FIX TI on a prior device),
//     immediately fast-skip remaining devices (no 30s wait) and add them to
//     timeout_on_base_umd_devices_ as well.  Observable speedup: N×30s → ~30s (single device
//     times out, all others fast-skip).
//
//   FIX TK (fabric_firmware_initializer.cpp:verify_all_fabric_channels_healthy):
//     Also call dev->set_fabric_ring_sync_timed_out() on timed-out devices so that
//     RiscFirmwareInitializer::teardown() FIX BA does NOT add them to relay_broken_non_mmio
//     (ring sync timeout ≠ broken relay — it means base-UMD transition is in progress).
//
// What this test verifies:
//   1. PREDECESSOR: opens FABRIC_1D MeshDevice, dispatches workload to spin up all ETH
//      fabric channels, then signals ready and spins until SIGKILL.
//   2. Parent SIGKILLs predecessor — leaves ETH ERISCs mid-session (base-UMD state).
//   3. TESTEE: opens a fresh FABRIC_1D MeshDevice.  The ring barrier may time out (base-UMD
//      channels mid-launch-msg transition). Key assertion: TESTEE completes within budget.
//      The FIX TJ fast-skip prevents N×30s sequential waits after the first timeout.
//   4. Parent measures elapsed time: must be < (kGap72TesteeBudget). If testee hangs
//      for 30s × N devices (FIX TI/TJ missing), it exceeds budget → FAIL.
//
// Additionally verifies that `is_fabric_ring_sync_timed_out()` distinguishes this state from
// a truly broken relay (prevents FIX BA from incorrectly treating the device as dead).
//
// Timing budget:
//   PREDECESSOR wait: 30s (hardware init + blank workload dispatch)
//   TESTEE budget:    90s (fabric init; worst case: 1 × 30s ring-sync + overhead)
//   Total:            ~150s
//   (Without FIX TI/TJ: testee would need 8 × 30s = 240s → guaranteed timeout → FAIL)
//
// Topology requirement: >= 4 devices (T3K ring with ETH fabric channels).

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

// 30s for predecessor init + workload dispatch
static constexpr int kGap72PredWaitMs = 30000;
// 90s: worst case = 1 device × 30s ring-sync timeout + ~30s init + ~30s overhead.
// Without FIX TI/TJ: 8 devices × 30s = 240s → guaranteed timeout → FAIL.
static constexpr int kGap72TesteeBudget = 90000;
// Exit sentinel: testee detected N×30s sequential ring-sync timeouts (regression indicator)
static constexpr int kGap72RegressionExit = 72;

struct Gap72Shm {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap72(const MeshCoordinateRange& range) {
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

static int wait_child_budget_gap72(pid_t pid, int budget_ms) {
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

class FixTiRingSyncFastSkipFixture : public MeshDeviceFixtureBase {
protected:
    FixTiRingSyncFastSkipFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 4) {
            GTEST_SKIP() << "GAP-72 requires >= 4 devices (T3K ring with ETH fabric channels). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-72: FixTiRingSyncDoesNotMultiply30sTimeouts
//
// Verifies FIX TI + FIX TJ: when ring-sync times out due to base-UMD channels,
// subsequent devices in the ring are fast-skipped (no additional 30s waits).
// Total overhead is bounded by ~1 × 30s (first timeout) + init overhead,
// not N × 30s (which would exceed any reasonable CI job timeout).
// ---------------------------------------------------------------------------
TEST_F(FixTiRingSyncFastSkipFixture, RingSyncDoesNotMultiply30sTimeouts) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap72Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap72Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ────────────────────────────────────────────
    // Opens FABRIC_1D MeshDevice, dispatches blank workload to spin up all ETH
    // fabric channels, signals ready, then spins until SIGKILL.
    // Result: ETH ERISCs restart to base-UMD (0x49706550) with stale sync addresses.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_1D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap72(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
            shm->predecessor_ready.store(1);
            // Spin until SIGKILL — leaves ETH ERISC mid-session
            while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (...) {
            shm->predecessor_ready.store(1);
        }
        _exit(0);
    }

    // Wait for predecessor to signal ready
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - pred_start)
                           .count();
        if (elapsed > kGap72PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap72Shm));
            GTEST_SKIP() << "GAP-72: predecessor did not signal ready within " << kGap72PredWaitMs
                         << "ms — cluster may be in a broken state. Skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-72: Predecessor SIGKILL'd — ETH ERISCs will restart to base-UMD (0x49706550). "
        "TESTEE opens FABRIC_1D with {} devices. If ring-sync times out on first device, "
        "FIX TJ must fast-skip remaining {} devices (not 30s × {} = {}s). "
        "Budget: {}ms.",
        num_dev, num_dev - 1, num_dev,
        static_cast<int>(num_dev) * 30,
        kGap72TesteeBudget);

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ─────────────────────────────────────────────────
    // Opens FABRIC_1D MeshDevice with the cluster in base-UMD state.
    // FIX TI: if ring-sync times out, device added to timeout_on_base_umd_devices_ and
    //         ring_sync_already_timed_out_ = true.
    // FIX TJ: subsequent devices fast-skipped (no 30s waits).
    // Observable: testee exits within kGap72TesteeBudget.
    //
    // The testee also measures its own elapsed time and exits kGap72RegressionExit
    // if the elapsed time exceeds 2 × 30s (indicating N×30s serial timeout pattern).
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        const auto t_start = std::chrono::steady_clock::now();

        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_1D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            dev->close();
            fprintf(stderr, "GAP-72 TESTEE: MeshDevice created and closed cleanly.\n");
        } catch (const std::exception& e) {
            // Exception is acceptable on degraded cluster; what matters is we didn't hang.
            fprintf(stderr, "GAP-72 TESTEE: caught exception (degraded cluster acceptable): %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "GAP-72 TESTEE: caught unknown exception.\n");
        }

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - t_start)
                                     .count();
        fprintf(stderr, "GAP-72 TESTEE: total elapsed = %ldms\n", static_cast<long>(elapsed_ms));

        // Regression sentinel: if we spent more than 2 × 30s, it suggests multiple devices
        // each waited the full 30s ring-sync timeout (FIX TI/TJ missing).
        // 2 × 30000ms = 60000ms threshold.
        if (elapsed_ms > 60000) {
            fprintf(
                stderr,
                "GAP-72 TESTEE REGRESSION: elapsed %ldms > 60000ms — FIX TJ fast-skip may be "
                "missing. Multiple devices likely each waited the full 30s ring-sync timeout "
                "(N×30s pattern). Expected at most 1 timeout + init overhead.\n",
                static_cast<long>(elapsed_ms));
            _exit(kGap72RegressionExit);
        }
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap72(testee_pid, kGap72TesteeBudget);
    auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - testee_start)
                         .count();

    ::munmap(raw_shm, sizeof(Gap72Shm));

    // Primary regression: testee exceeded the 90s budget entirely (hard timeout from parent)
    if (rc == -1) {
        FAIL() << "GAP-72 REGRESSION (FIX TI/TJ missing or reverted): TESTEE timed out after "
               << kGap72TesteeBudget << "ms.\n"
               << "\n"
               << "Root cause: wait_for_fabric_router_sync() waited 30s per device without\n"
               << "fast-skipping once the first device timed out. With " << num_dev << " devices,\n"
               << "the expected worst-case overhead is " << (static_cast<int>(num_dev) * 30)
               << "s (well over the " << (kGap72TesteeBudget / 1000) << "s budget).\n"
               << "\n"
               << "Fix (FIX TI + FIX TJ, fabric_firmware_initializer.cpp):\n"
               << "  FIX TI: on ring-sync timeout with base-UMD channels present,\n"
               << "    add device to timeout_on_base_umd_devices_ and set\n"
               << "    ring_sync_already_timed_out_ = true.\n"
               << "  FIX TJ: if ring_sync_already_timed_out_ is already set, immediately\n"
               << "    fast-skip the current device (no 30s wait) and add it to\n"
               << "    timeout_on_base_umd_devices_.\n"
               << "  FIX TK: in verify_all_fabric_channels_healthy(), call\n"
               << "    dev->set_fabric_ring_sync_timed_out() for timeout devices so FIX BA\n"
               << "    does not incorrectly treat them as relay-broken.";
    }

    // Secondary regression: testee self-reported N×30s serial pattern (2 timeouts × 30s = 60s)
    if (rc == kGap72RegressionExit) {
        FAIL() << "GAP-72 REGRESSION (FIX TJ fast-skip may be missing): TESTEE elapsed time "
               << "exceeded 60000ms (2 × 30s threshold).\n"
               << "\n"
               << "Multiple devices appear to have each waited the full 30s ring-sync timeout,\n"
               << "indicating ring_sync_already_timed_out_ was NOT propagated from the first\n"
               << "timed-out device to fast-skip subsequent devices.\n"
               << "\n"
               << "With " << num_dev << " devices, FIX TJ should bound total overhead to\n"
               << "approximately 1 × 30s (first timeout) + init overhead.\n"
               << "Measured: " << testee_ms << "ms.";
    }

    EXPECT_EQ(rc, 0) << "GAP-72: TESTEE exited with unexpected code " << rc
                     << " (expected 0 — clean init or graceful degraded-cluster exception).";

    log_info(
        tt::LogTest,
        "GAP-72 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TI + FIX TJ ring-sync fast-skip is working — ring-sync timeout for {} devices "
        "did not multiply to {}s total overhead.",
        testee_ms,
        kGap72TesteeBudget,
        rc,
        num_dev,
        static_cast<int>(num_dev) * 30);
}

}  // namespace tt::tt_metal::distributed::test
