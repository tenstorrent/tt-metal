// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-43: FIX NV + FIX NW — Skip relay-dependent calls for non-MMIO chips in
//         RiscFirmwareInitializer::run_async_build_phase
//
// Root cause (CI run 25083588469, t3k-06 — TestMeshWidthShardedCopy2D_Large):
//
//   run_async_build_phase spawns one std::async task per device.  Each task called
//   two relay-dependent functions for NON-MMIO (remote) chips:
//
//   1. get_device_aiclk(device_id)  [FIX NV]
//      → RemoteChip::get_clock() → WormholeArcMessenger::send_message()
//      → wait_for_non_mmio_flush()  [5s timeout]
//      When the MMIO relay is dead (prior session was SIGKILL'd, stale FABRIC firmware),
//      this throws "Timeout waiting for Ethernet core service remote IO request."
//      The exception propagates through the async future, through fut.get(), and
//      up to MetalContext::initialize() → test SetUp() → test FAILS with exception.
//      With N non-MMIO chips: N×5s of relay timeouts before the throw.
//
//   2. clear_launch_messages_on_eth_cores(device_id)  [FIX NW]
//      → cluster_.write_core()   (FIX AE catches this, marks relay broken, returns OK)
//      → cluster_.l1_barrier()   → driver_->l1_membar() → wait_for_non_mmio_flush()
//                                  FIX AE does NOT wrap l1_barrier() → throws!
//      Exception propagates same path as above.
//
//   FIX AE was already in place for write_core() but NOT for l1_barrier().
//   Therefore clear_launch_messages_on_eth_cores on a non-MMIO chip with dead relay:
//     - write_core()    → caught by FIX AE → relay marked broken → returns
//     - l1_barrier()    → NOT caught → throws → async task throws → test FAILS
//
// FIX NV (#42429) — risc_firmware_initializer.cpp:
//   Gate get_device_aiclk(device_id) on mmio_ids_set.count(device_id).
//   The aiclk value is [[maybe_unused]] debug-only — no functional impact on skipping.
//
// FIX NW (#42429) — risc_firmware_initializer.cpp:
//   Gate clear_launch_messages_on_eth_cores(device_id) on mmio_ids_set.count(device_id).
//   Safe to skip: run_launch_phase calls terminate_active_ethernet_cores_on_all_chips()
//   unconditionally (MMIO + non-MMIO), clearing all stale launch messages via full ETH reset.
//
// What this test verifies:
//   1. Parent opens a healthy cluster (FABRIC_2D), dispatches a blank workload so
//      non-MMIO ERISCs are in ACTIVE/FABRIC state.
//   2. PREDECESSOR is fork/exec'd: opens FABRIC_2D, dispatches blank workload, signals
//      ready, spins.
//   3. Parent SIGKILLs predecessor — non-MMIO ERISCs remain in FABRIC firmware state.
//   4. 2s settle.
//   5. TESTEE is fork/exec'd: opens a new MeshDevice (triggers MetalContext::initialize
//      → run_async_build_phase for ALL device_ids).
//      a. Without FIX NV: get_device_aiclk on non-MMIO chips → throws → testee exits
//         non-zero (exception unhandled in main, or caught by GTest and reported).
//      b. Without FIX NW: clear_launch_messages_on_eth_cores l1_barrier on non-MMIO
//         chips → throws → testee exits non-zero.
//      c. With both fixes: non-MMIO chips are skipped → no relay calls in async tasks
//         → no exceptions → testee exits 0.
//   6. Parent asserts:
//      (a) testee exits within kTesteeBudgetMs (timing guard — catches per-chip delays).
//      (b) testee exits with code 0 (no exception from run_async_build_phase).
//   7. Timing budget:
//      Normal init:                ~10-15s
//      Without FIX NV (N=3 T3K): + 3×5 = 15s extra → SetUp hangs before throw
//      Without FIX NW (N=3 T3K): + 3×1×5 = 15s extra (write_core FIX AE absorbs, l1_barrier throws)
//      Combined regression:        ~40-45s → exceeds kTesteeBudgetMs = 45s
//      With both fixes:            ~10-15s normal init (no relay overhead)
//
// Distinction from prior GAPs:
//   GAP-39 (FIX NS): relay queue overflow from double topology discovery.
//   GAP-40 (FIX AE): flush timeout in active-session write_core/~Cluster destructor.
//   GAP-41 (FIX NT): EthCoord missing from chip_locations (instant crash, not hang).
//   GAP-43 (this):   relay-dependent calls in run_async_build_phase INIT PATH;
//                    primary failure is exception propagation from async future,
//                    secondary symptom is N×5s timing delay before the throw.

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

// Testee budget: with both fixes, open+close takes ~10-15s.
// Without FIX NV (3 non-MMIO chips on T3K): +3×5 = 15s extra.
// Without FIX NW (3 non-MMIO chips on T3K): +3×5 = 15s extra (l1_barrier throws after FIX AE absorbs write_core).
// Combined: ~40-45s → exceeds this budget, failing the test.
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 45000;

struct Gap43SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// Minimal FABRIC_2D workload to put non-MMIO ERISCs into ACTIVE relay state.
static MeshWorkload make_blank_workload_gap43(const MeshCoordinateRange& range) {
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

class AsyncBuildPhaseRelayGuardFixture : public MeshDeviceFixtureBase {
protected:
    AsyncBuildPhaseRelayGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-43 requires >= 2 devices (need at least 1 non-MMIO chip). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-43: AsyncBuildPhaseSkipsRelayCallsForNonMmioChips
//
// Verifies that run_async_build_phase does NOT call relay-dependent functions
// (get_device_aiclk, clear_launch_messages_on_eth_cores) for non-MMIO chips
// when the MMIO relay is dead from a prior SIGKILL'd session.
//
// Primary failure (FIX NV/NW missing): testee exits non-zero — exception from
// wait_for_non_mmio_flush() propagates through async future up to SetUp().
// Secondary failure: testee takes > kTesteeBudgetMs due to per-chip 5s relay hangs.
// ---------------------------------------------------------------------------
TEST_F(AsyncBuildPhaseRelayGuardFixture, AsyncBuildPhaseSkipsRelayCallsForNonMmioChips) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap43SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap43SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches blank workload (non-MMIO ERISCs → ACTIVE state),
    // signals ready, then spins.  SIGKILL leaves FABRIC firmware active on all ETH cores.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        // Predecessor child
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
            auto workload = make_blank_workload_gap43(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Even if dispatch fails, signal ready so parent can proceed.
        }
        shm->predecessor_ready.store(1);
        // Spin until SIGKILL — keeps non-MMIO ERISCs in FABRIC state.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);
    }

    // Wait for predecessor ready signal, then SIGKILL.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap43SharedMem));
            GTEST_SKIP() << "GAP-43: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);
    log_info(
        tt::LogTest,
        "GAP-43: Predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state. "
        "Next MeshDevice open will trigger run_async_build_phase with stale relay.");

    // Brief settle so UMD relay CMD queue state stabilises.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE ───────────────────────────────────────────────────
    // Opens a new full MeshDevice.  This triggers:
    //   MetalContext::initialize()
    //     → RiscFirmwareInitializer::run_async_build_phase(all_device_ids)
    //       For each device in async task:
    //         a. FIX NV: get_device_aiclk() SKIPPED for non-MMIO chips → no relay call
    //         b. FIX NW: clear_launch_messages_on_eth_cores() SKIPPED for non-MMIO chips
    //            → no write_core/l1_barrier relay calls
    //
    // Without FIX NV: get_device_aiclk(non-MMIO) → ARC messenger
    //                 → wait_for_non_mmio_flush → throws after 5s
    //                 → propagates from async task → fut.get() throws → testee exits non-zero
    //
    // Without FIX NW: clear_launch_messages_on_eth_cores(non-MMIO):
    //   - write_core()   → FIX AE catches, marks relay broken, returns
    //   - l1_barrier()   → l1_membar() → wait_for_non_mmio_flush → throws (FIX AE doesn't cover l1_barrier)
    //   → propagates from async task → fut.get() throws → testee exits non-zero
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Testee child: open a full MeshDevice, immediately close it, exit 0.
        // If run_async_build_phase throws for any non-MMIO device, the exception
        // propagates to MetalContext::initialize() and then to MeshDevice::create(),
        // which propagates to this try-catch.  We exit 1 to signal the parent.
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
            // run_async_build_phase already completed by the time create() returns.
            dev->close();
            rc = 0;
        } catch (const std::exception& e) {
            // This is the regression path: exception from run_async_build_phase
            // (wait_for_non_mmio_flush threw) propagated up.
            // Exit code 1 signals "exception caught — FIX NV or FIX NW is missing".
            rc = 1;
        } catch (...) {
            rc = 2;
        }
        _exit(rc);
    }

    // ── Step 3: Wait for testee and verify exit code + timing ─────────────────
    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    pid_t waited = 0;
    while (true) {
        waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - testee_start)
                                 .count();
        if (elapsed > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap43SharedMem));
            FAIL() << "GAP-43 TIMEOUT (FIX NV or FIX NW): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "This indicates that run_async_build_phase is making relay-dependent calls\n"
                   << "for non-MMIO chips (dead relay → 5s per wait_for_non_mmio_flush call).\n"
                   << "With " << (num_dev - 1) << " non-MMIO chips:\n"
                   << "  FIX NV missing: +" << (num_dev - 1) * 5 << "s (get_device_aiclk per chip)\n"
                   << "  FIX NW missing: +" << (num_dev - 1) * 5 << "s (l1_barrier after FIX AE absorbs write_core)\n"
                   << "\n"
                   << "Fix: gate get_device_aiclk() and clear_launch_messages_on_eth_cores()\n"
                   << "on mmio_ids_set.count(device_id) in run_async_build_phase.\n"
                   << "\n"
                   << "CI reference: run 25083588469 (t3k-06, TestMeshWidthShardedCopy2D_Large).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - testee_start)
                                .count();

    ::munmap(raw_shm, sizeof(Gap43SharedMem));

    // Check for non-zero exit code: this is the primary regression indicator.
    // Exit 1 = exception caught from run_async_build_phase (wait_for_non_mmio_flush threw).
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        const int ec = WEXITSTATUS(status);
        if (ec == 1) {
            FAIL() << "GAP-43 REGRESSION (FIX NV or FIX NW): Testee caught exception from\n"
                      "MeshDevice::create() — exception propagated from run_async_build_phase.\n"
                      "\n"
                      "Root cause: get_device_aiclk() or clear_launch_messages_on_eth_cores()\n"
                      "was called for a non-MMIO chip whose MMIO relay is dead (FABRIC firmware\n"
                      "from prior SIGKILL'd session).  wait_for_non_mmio_flush() threw after 5s.\n"
                      "\n"
                      "FIX NV: gate get_device_aiclk(device_id) on mmio_ids_set.count(device_id).\n"
                      "FIX NW: gate clear_launch_messages_on_eth_cores(device_id) on same.\n"
                      "Both are in RiscFirmwareInitializer::run_async_build_phase().\n"
                      "\n"
                      "Note: FIX AE handles write_core() exceptions but NOT l1_barrier() —\n"
                      "FIX NW's l1_barrier() path is the hole FIX AE does not close.\n"
                      "\n"
                      "CI reference: run 25083588469 (t3k-06, TestMeshWidthShardedCopy2D_Large).";
        }
        FAIL() << "GAP-43: Testee exited with unexpected code " << ec << ".";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-43: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    log_info(
        tt::LogTest,
        "GAP-43 PASS: Testee completed run_async_build_phase in {}ms (budget: {}ms) with exit 0. "
        "FIX NV + FIX NW correctly skipped relay-dependent calls for non-MMIO chips.",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
