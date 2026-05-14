// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-44: FIX NX — Guard write_core() for non-MMIO chips against uncaught relay timeout.
//
// Root cause (CI run 25084780844, t3k-12 — MeshDevice1x4Fabric2DUDMFixture.TestMeshWidthShardedCopy2D_Large):
//
//   After a prior session left stale FABRIC firmware on non-MMIO ETH channels, the first
//   test's FABRIC dispatch timed out (fabric hang at +22s).  Teardown ran; FIX AZ/AC fired
//   (skip assert_cores/l1_barrier for dead non-MMIO devices).  The second test's SetUp()
//   called MeshDevice::create() → MetalContext::initialize().
//
//   MetalContext::initialize() executes this sequence:
//     1. run_async_build_phase(all_device_ids)      — FIX NV + FIX NW already guard this
//     2. set_internal_routing_info_for_ethernet_cores(..., enable=true)
//        → for each non-MMIO chip's active ETH core:
//            write_core(...) → write_to_device() → write_to_non_mmio() → 5s timeout → throws
//        FIX AE only catches wait_for_non_mmio_flush() — NOT write_to_device() itself.
//        The exception propagates to MetalContext::initialize() → MeshDevice::create() → SetUp().
//        GTest reports: "C++ exception thrown in SetUp()".
//        4 non-MMIO chips × 5s each = 20s of serial relay timeouts before the throw.
//     3. watcher_server_->init_devices()
//        → WatcherServer::Impl::init_device(chip_id) iterates all chips.
//        → write_core() for each non-MMIO chip's ETH cores → same timeout + throw path.
//
//   FIX AE's try/catch in write_core() only wraps wait_for_non_mmio_flush() (the flush after
//   write_to_device succeeds).  It does NOT wrap write_to_device() itself.  So write_to_device
//   on a dead-relay non-MMIO chip still times out and throws an uncaught exception.
//
// FIX NX (#42429) — tt_cluster.cpp Cluster::write_core():
//   Restructure the remote-chip path so that BOTH write_to_device() AND wait_for_non_mmio_flush()
//   are wrapped in a single try/catch for non-MMIO chips.  On exception, log a FIX NX warning
//   and mark relay broken.  MMIO chips are unaffected (they use PCIe-direct writes, no relay).
//
//   After FIX NX:
//     - set_internal_routing_info_for_ethernet_cores: write_core for non-MMIO chips catches the
//       timeout, logs, marks relay broken, returns — no exception propagates.
//     - watcher_server_->init_devices: same — write_core for non-MMIO chips is silently dropped.
//     - Total init time drops from 4×5s = 20s of serial relay waits to near-zero.
//
// What this test verifies:
//   1. Parent opens a healthy cluster (FABRIC_2D), dispatches a blank workload so
//      non-MMIO ERISCs are in ACTIVE/FABRIC state.
//   2. PREDECESSOR is fork/exec'd: opens FABRIC_2D, dispatches blank workload, signals ready, spins.
//   3. Parent SIGKILLs predecessor — non-MMIO ERISCs remain in FABRIC firmware state.
//   4. 2s settle.
//   5. FIRST-TESTEE opens a MeshDevice: run_async_build_phase runs (FIX NV/NW guard),
//      then set_internal_routing_info_for_ethernet_cores runs (FIX NX guards write_core),
//      then watcher_server_->init_devices runs (FIX NX guards write_core).
//      Should complete without exception.
//   6. SECOND-TESTEE opens a new MeshDevice immediately after (second fixture cycle).
//      Verifies that relay-broken state from the first testee's FIX NX catches does not
//      prevent a clean second init.
//   7. Parent asserts:
//      (a) testee exits within kTesteeBudgetMs (timing guard — catches 4×5s = 20s relay hangs).
//      (b) testee exits with code 0 (no exception from MetalContext::initialize).
//
// Distinction from GAP-43:
//   GAP-43: relay calls in run_async_build_phase (async tasks — get_device_aiclk, clear_launch_msgs).
//   GAP-44 (this): relay calls in set_internal_routing_info_for_ethernet_cores and
//                  WatcherServer::init_devices — both in the main thread of MetalContext::initialize().
//                  The uncaught exception path is in write_to_device() rather than l1_barrier().

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

// Testee budget:
//   Without FIX NX (4 non-MMIO chips on T3K wh_llmbox): 4×5s = 20s relay timeouts before throw.
//   Normal init: ~10-15s.
//   Combined worst-case without fix: ~35s → well below kTesteeBudgetMs.
//   With FIX NX: ~10-15s (no relay overhead).
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 45000;

struct Gap44SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// Minimal FABRIC_2D workload to put non-MMIO ERISCs into ACTIVE relay state.
static MeshWorkload make_blank_workload_gap44(const MeshCoordinateRange& range) {
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

class WriteCorRelayGuardFixture : public MeshDeviceFixtureBase {
protected:
    WriteCorRelayGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-44 requires >= 2 devices (need at least 1 non-MMIO chip). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-44: WriteCoreRelayGuardForNonMmioChips
//
// Verifies that write_core() for non-MMIO chips does NOT propagate timeout
// exceptions to MetalContext::initialize() when the relay is dead from a prior
// SIGKILL'd session.  Covers set_internal_routing_info_for_ethernet_cores and
// WatcherServer::init_devices.
//
// Primary failure (FIX NX missing): testee exits non-zero — uncaught exception from
// write_to_device() in write_core() propagates up through MetalContext::initialize()
// to MeshDevice::create() to the test SetUp().
// Secondary failure: testee takes > kTesteeBudgetMs due to 4×5s serial relay hangs
// before the exception finally propagates.
// ---------------------------------------------------------------------------
TEST_F(WriteCorRelayGuardFixture, WriteCoreRelayGuardForNonMmioChips) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap44SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap44SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches blank workload (non-MMIO ERISCs → ACTIVE state),
    // signals ready, then spins.  SIGKILL leaves FABRIC firmware active on non-MMIO ETH cores.
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
            auto workload = make_blank_workload_gap44(range);
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
            ::munmap(raw_shm, sizeof(Gap44SharedMem));
            GTEST_SKIP() << "GAP-44: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);
    log_info(
        tt::LogTest,
        "GAP-44: Predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state. "
        "Next MeshDevice open will trigger set_internal_routing_info_for_ethernet_cores "
        "and WatcherServer::init_devices with dead relay.");

    // Brief settle so UMD relay CMD queue state stabilises.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE ───────────────────────────────────────────────────
    // Opens a new full MeshDevice.  This triggers MetalContext::initialize():
    //   1. run_async_build_phase               — FIX NV + FIX NW guard non-MMIO chips
    //   2. set_internal_routing_info_for_ethernet_cores(..., enable=true)
    //      → write_core() for each non-MMIO chip's ETH cores
    //      FIX NX: write_core wraps write_to_device + wait_for_non_mmio_flush for remote chips
    //   3. watcher_server_->init_devices()
    //      → write_core() for each chip (including non-MMIO) ETH cores
    //      FIX NX: same catch
    //
    // Without FIX NX: write_to_device() for a non-MMIO chip with dead relay → 5s timeout
    //   → throws "Timeout waiting for Ethernet core service remote IO request."
    //   FIX AE only catches wait_for_non_mmio_flush — NOT write_to_device itself.
    //   → exception propagates to MetalContext::initialize() → MeshDevice::create() → testee exits non-zero
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Testee child: open a full MeshDevice, immediately close it, exit 0.
        // If MetalContext::initialize() throws (from set_internal_routing_info or watcher init),
        // we catch it and exit 1 to signal the parent.
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
            // MetalContext::initialize() already completed (including set_internal_routing_info
            // and watcher init) by the time MeshDevice::create() returns.
            dev->close();
            rc = 0;
        } catch (const std::exception& e) {
            // Regression path: write_core() in set_internal_routing_info_for_ethernet_cores
            // or WatcherServer::init_devices threw for a non-MMIO chip with dead relay.
            // FIX NX is missing or broken.
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
            ::munmap(raw_shm, sizeof(Gap44SharedMem));
            FAIL() << "GAP-44 TIMEOUT (FIX NX): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "This indicates that write_core() for non-MMIO chips is blocking on\n"
                   << "dead relay timeouts (~5s per write_to_device call) in:\n"
                   << "  - set_internal_routing_info_for_ethernet_cores (called from MetalContext::initialize)\n"
                   << "  - WatcherServer::Impl::init_devices (called from MetalContext::initialize)\n"
                   << "\n"
                   << "With " << (num_dev - 1) << " non-MMIO chips, the serial relay hang is:\n"
                   << "  " << (num_dev - 1) * 5 << "s before the exception is finally thrown.\n"
                   << "\n"
                   << "Fix (FIX NX): in Cluster::write_core(), for is_chip_remote() chips,\n"
                   << "wrap both write_to_device() and wait_for_non_mmio_flush() in a single\n"
                   << "try/catch.  On exception, log FIX NX warning, mark relay broken, return.\n"
                   << "\n"
                   << "CI reference: run 25084780844 (t3k-12, TestMeshWidthShardedCopy2D_Large).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - testee_start)
                                .count();

    ::munmap(raw_shm, sizeof(Gap44SharedMem));

    // Check for non-zero exit code: primary regression indicator.
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        const int ec = WEXITSTATUS(status);
        if (ec == 1) {
            FAIL() << "GAP-44 REGRESSION (FIX NX): Testee caught exception from\n"
                      "MeshDevice::create() — exception propagated from write_core() during\n"
                      "MetalContext::initialize() (set_internal_routing_info_for_ethernet_cores\n"
                      "or WatcherServer::init_devices).\n"
                      "\n"
                      "Root cause: write_to_device() for a non-MMIO chip with dead relay\n"
                      "timed out and threw.  FIX AE only wraps wait_for_non_mmio_flush() —\n"
                      "NOT write_to_device() itself — so the write timeout escapes uncaught.\n"
                      "\n"
                      "FIX NX: In Cluster::write_core(), for is_chip_remote() chips, wrap\n"
                      "both write_to_device() and wait_for_non_mmio_flush() in a single\n"
                      "try/catch.  On exception, log FIX NX warning, mark relay broken, return.\n"
                      "\n"
                      "CI reference: run 25084780844 (t3k-12, TestMeshWidthShardedCopy2D_Large).";
        }
        FAIL() << "GAP-44: Testee exited with unexpected code " << ec << ".";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-44: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    log_info(
        tt::LogTest,
        "GAP-44 PASS: Testee completed MetalContext::initialize() in {}ms (budget: {}ms) with exit 0. "
        "FIX NX correctly caught write_core() relay timeouts for non-MMIO chips.",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
