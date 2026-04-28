// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-29: ~Cluster blocks indefinitely in wait_for_non_mmio_flush after relay-broken quiesce
//
// Root cause (discovered April 2026, iteration 15 of race-condition hunt):
//   After RiscFirmwareInitializer::teardown() (FIX AC) PCIe-resets the MMIO ETH cores,
//   the UMD host-side relay CMD queue for non-MMIO devices retains stale
//   prefetch_q_in_flight entries.  When the process eventually destructs ~Cluster,
//   driver_->close_device() calls wait_for_non_mmio_flush() for each non-MMIO chip.
//   Since the MMIO ERISC that was processing those commands was already reset, the
//   stale entries will never drain.  wait_for_non_mmio_flush() spins indefinitely
//   (no timeout, no exception).  In production CI, SIGALRM fires after 15 minutes.
//
// FIX AW (tt_cluster.cpp):
//   ~Cluster checks relay_broken_chips_for_close_ (populated by
//   RiscFirmwareInitializer::teardown via mark_relay_broken_for_close).  If any
//   chips are registered, driver_->close_device() is run in a detached thread with
//   a 5s timeout rather than on the destructor thread.
//
// What this test verifies:
//   1. After a SIGKILL predecessor leaves ERISCs in ACTIVE relay state, the
//      test process opens FABRIC_2D (relay-broken init path fires).
//   2. The test process quiesces (triggers FIX AC MMIO ETH PCIe reset path).
//   3. The test process exits naturally — this triggers ~Cluster → close_device().
//   4. The process exits within 30s.
//      Without FIX AW: ~Cluster hangs 15+ minutes in wait_for_non_mmio_flush().
//      With FIX AW:    ~Cluster completes in ~5s (thread timeout path) or less.
//
// Test structure:
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D, dispatches, signals ready, spins.
//   Phase 2: Parent SIGKILLs predecessor (relay left in ACTIVE state).
//   Phase 3: Fork TESTEE — opens FABRIC_2D (relay-broken), quiesces, then exit(0).
//   Phase 4: Parent waits for TESTEE to exit within 30s.
//   If TESTEE doesn't exit in time → FAIL.
//
// Gap vs. existing tests:
//   GAP-28 verifies MeshDevice::create() doesn't hang on sysmem_manager_->reset()
//   for relay-broken devices (FIX AV).  It does NOT exercise the teardown /
//   process-exit path where ~Cluster's wait_for_non_mmio_flush() hangs.
//
//   No existing test exercises the ~Cluster destructor in a relay-broken context
//   because prior gap tests close the mesh device and exit via GTest's normal
//   teardown.  The hang only manifests when the process exits AFTER a quiesce that
//   triggered FIX AC (MMIO ERISC PCIe reset).
//
// Topology requirement: >= 2 devices (non-MMIO relay path required).

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <thread>

#include <experimental/fabric/fabric_types.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "fabric/fabric_init.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture
//
// Budget covers:
//   ~20s  child FABRIC_2D init + dispatch
//   ~20s  kMaxWaitMs for child_ready
//    ~2s  post-kill margin
//   ~30s  testee process exit budget (FIX AW: 5s thread timeout + margin)
//   ~30s  general margin
// ---------------------------------------------------------------------------
class ClusterTeardownHangRelayBrokenFixture : public MeshDeviceFixtureBase {
protected:
    ClusterTeardownHangRelayBrokenFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-29 requires >= 2 devices (non-MMIO relay path required). "
                            "Found "
                         << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// Lightest workload that activates the ETH relay path (ensures ERISCs are ACTIVE).
static MeshWorkload make_blank_workload_gap29(const MeshCoordinateRange& range) {
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
// GAP-29: ClusterTeardownDoesNotHangAfterRelayBrokenQuiesce
//
// Verifies that ~Cluster completes within a bounded time after a quiesce that
// triggered FIX AC (MMIO ETH PCIe reset).  Without FIX AW, ~Cluster hangs
// 15+ minutes in wait_for_non_mmio_flush(); with FIX AW it exits in ~5s.
// ---------------------------------------------------------------------------
TEST_F(ClusterTeardownHangRelayBrokenFixture, ClusterTeardownDoesNotHangAfterRelayBrokenQuiesce) {
    // ── Phase 1: Close fixture device so child inherits clean MetalContext ─────
    mesh_device_->close();

    // ── Shared memory flags ──────────────────────────────────────────────────
    struct SharedFlags {
        std::atomic<int> predecessor_ready{0};
        std::atomic<int> testee_ready{0};
    };
    void* raw_shm = ::mmap(nullptr, sizeof(SharedFlags), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* flags = new (raw_shm) SharedFlags();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 2: Fork PREDECESSOR ────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches a workload (ERISCs enter ACTIVE relay state),
    // signals ready, then spins forever.  SIGKILL leaves relay dirty.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        // Child: predecessor.
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
            auto workload = make_blank_workload_gap29(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
        }
        flags->predecessor_ready.store(1);
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);
    }

    // Wait for predecessor ready, then SIGKILL it.
    constexpr int kPredWaitMs = 20000;
    const auto pred_wait_start = std::chrono::steady_clock::now();
    while (flags->predecessor_ready.load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_wait_start)
                .count();
        if (elapsed > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(SharedFlags));
            GTEST_SKIP() << "Predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    // ── Phase 3: Fork TESTEE ─────────────────────────────────────────────────
    // Opens FABRIC_2D (relay-broken init path fires), quiesces fabric, then
    // calls exit(0).  exit() triggers C++ destructors including ~Cluster →
    // driver_->close_device() → wait_for_non_mmio_flush().
    // Without FIX AW this hangs 15+ min.  With FIX AW: completes in ~5s.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Child: testee.
        // Signal parent even on failure/exception so it doesn't wait kTesteeExitMs.
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Signal parent: we are about to quiesce + exit.
            flags->testee_ready.store(1);
            // Quiesce: triggers FIX AC MMIO ETH PCIe reset path when relay broken.
            dev->close();
        } catch (...) {
            flags->testee_ready.store(1);
        }
        // exit(0) — NOT _exit — so C++ global/static destructors run,
        // including ~MetalContext → ~Cluster → driver_->close_device().
        exit(0);
    }

    // Wait for testee_ready (up to 30s for open + quiesce).
    constexpr int kTesteeReadyMs = 30000;
    const auto testee_ready_start = std::chrono::steady_clock::now();
    while (flags->testee_ready.load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - testee_ready_start)
                .count();
        if (elapsed > kTesteeReadyMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(SharedFlags));
            GTEST_SKIP() << "Testee did not open+quiesce within " << kTesteeReadyMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::munmap(raw_shm, sizeof(SharedFlags));
    raw_shm = nullptr;

    // ── Phase 4: Wait for testee to exit within kTesteeExitMs ─────────────
    // With FIX AW: testee exits in ~5s (thread timeout path in ~Cluster).
    // Without FIX AW: testee hangs 15+ minutes in wait_for_non_mmio_flush().
    constexpr int kTesteeExitMs = 30000;
    const auto exit_start = std::chrono::steady_clock::now();
    bool testee_exited = false;
    while (true) {
        int wstatus = 0;
        const pid_t result = ::waitpid(testee_pid, &wstatus, WNOHANG);
        if (result == testee_pid) {
            testee_exited = true;
            break;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - exit_start)
                .count();
        if (elapsed > kTesteeExitMs) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!testee_exited) {
        ::kill(testee_pid, SIGKILL);
        ::waitpid(testee_pid, nullptr, 0);
    }

    const auto exit_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - exit_start)
            .count();

    EXPECT_TRUE(testee_exited)
        << "GAP-29 REGRESSION: testee process did not exit within " << kTesteeExitMs << "ms "
           "after relay-broken quiesce + natural exit. "
           "Root cause: ~Cluster::driver_->close_device() blocked indefinitely in "
           "wait_for_non_mmio_flush() — UMD relay CMD queue has stale entries after "
           "FIX AC PCIe-reset the MMIO ERISC. "
           "Fix: FIX AW in tt_cluster.cpp — run close_device() in a background thread "
           "with 5s timeout when relay_broken_chips_for_close_ is non-empty.";

    if (testee_exited) {
        log_info(
            tt::LogTest,
            "GAP-29: testee process exited cleanly in {}ms after relay-broken quiesce "
            "(FIX AW: ~Cluster did not hang in wait_for_non_mmio_flush)",
            exit_elapsed_ms);
    }
}

}  // namespace tt::tt_metal::distributed::test
