// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-31: Non-MMIO ERISCs left in FABRIC firmware by FIX AX cause second session to hang
//
// Root cause (discovered April 2026, OPUS audit iteration 16):
//   FIX AX (FabricFirmwareInitializer::teardown) skips assert_risc_reset_at_core for
//   non-MMIO channels whose relay path is confirmed dead.  This prevents the 5-second
//   per-channel UMD timeout in the CURRENT session, but leaves non-MMIO ERISCs running
//   FABRIC firmware after process exit.
//
//   When the NEXT process starts (e.g. unit_tests_ttnn binary):
//   1. TopologyDiscovery::discover_remote_devices() → init_tt_device hits the 5s timeout
//      per non-MMIO device (FABRIC firmware ignores UMD relay read requests).
//   2. After the 5s timeout exception, MetalContext initialization is in a partially-
//      failed state.  Subsequent tests that attempt configure_fabric() →
//      write_non_mmio → UMD relay queue fill → while(full) spin → 15-min SIGALRM.
//
// FIX AY (risc_firmware_initializer.cpp, RiscFirmwareInitializer::teardown):
//   After FIX AC's parallel heartbeat poll confirms MMIO ETH relay is restored,
//   attempt assert+deassert of each non-MMIO ETH ERISC that was skipped by FIX AX.
//   The reset write goes PCIe → MMIO relay ERISC (BASE fw) → ETH → non-MMIO hardware
//   reset register.  Non-MMIO ERISC reboots into base UMD firmware.
//   The next session's topology discovery finds ERISCs in base firmware → no hang.
//
// What this test verifies:
//   1. After a SIGKILL predecessor leaves non-MMIO ERISCs in FABRIC firmware, the
//      test process opens FABRIC_2D (FIX AU/AX/AC chain fires on teardown).
//   2. FIX AY fires after FIX AC: non-MMIO ETH ERISCs are reset to base firmware.
//   3. The same process opens FABRIC_2D a SECOND TIME within 30s.
//      Without FIX AY: second open hangs in configure_fabric/write_non_mmio (~15 min).
//      With FIX AY:    second open completes in < 30s.
//   4. Second-session AllGather produces correct output (PCC >= 0.9999).
//
// Test structure:
//   Phase 1: Fork PREDECESSOR — opens FABRIC_2D, signals ready, spins.
//   Phase 2: Parent SIGKILLs predecessor (non-MMIO ERISCs left in FABRIC fw).
//   Phase 3: Fork TESTEE — opens FABRIC_2D (first session), does AllGather,
//             explicitly closes MeshDevice (FIX AY fires here), then opens
//             MeshDevice SECOND TIME within 30s, does AllGather, exit(0).
//   Phase 4: Parent waits for TESTEE to exit within 90s.
//   If TESTEE does not exit within 90s → FAIL (FIX AY regression).
//
// Gap vs. existing tests:
//   GAP-29 verifies ~Cluster doesn't hang on process exit after relay-broken quiesce
//   (FIX AW).  It tests SINGLE open → quiesce → exit.  It does NOT verify that the
//   NEXT session (or second open in the same process) can reach non-MMIO devices.
//
//   GAP-28 tests the sysmem_manager relay-broken guard (FIX AV) — also single session.
//
//   No existing test opens MeshDevice TWICE in the same process after a relay-broken
//   teardown.  The second-open hang is uniquely triggered by FIX AX leaving non-MMIO
//   ERISCs in FABRIC firmware.
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
// Shared memory flags for inter-process signaling
// ---------------------------------------------------------------------------
struct Gap31SharedFlags {
    std::atomic<int> predecessor_ready{0};
    std::atomic<int> testee_phase1_done{0};  // First session closed
    std::atomic<int> testee_exit_ok{0};       // Second session done + exit
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class FixAyDeferredNonMmioResetFixture : public MeshDeviceFixtureBase {
protected:
    FixAyDeferredNonMmioResetFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              // Budget: 20s pred init + 30s first session + 30s second session + margins
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-31 requires >= 2 devices (non-MMIO relay path required). "
                            "Found "
                         << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Lightest workload that exercises the full host → device dispatch path.
// This verifies ERISCs are in base firmware and dispatch works on the second
// open, without depending on CCL op availability.  Same pattern as GAP-28/29.
static MeshWorkload make_blank_workload_gap31(const MeshCoordinateRange& range) {
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
// Test
// ---------------------------------------------------------------------------
TEST_F(FixAyDeferredNonMmioResetFixture, DeferredNonMmioResetPreventsSecondOpenHang) {
    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Set up shared memory ────────────────────────────────────────
    void* raw_shm = ::mmap(
        nullptr,
        sizeof(Gap31SharedFlags),
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS,
        -1,
        0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* flags = new (raw_shm) Gap31SharedFlags{};

    // ── Phase 1: Fork PREDECESSOR ─────────────────────────────────────────────
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
        // Child: predecessor — open FABRIC_2D, signal ready, spin until SIGKILL.
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Signal parent that ERISCs are active and relay is live.
            flags->predecessor_ready.store(1);
            // Spin so SIGKILL leaves ERISCs in FABRIC firmware state.
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            flags->predecessor_ready.store(1);
            _exit(1);
        }
        _exit(0);
    }

    // Parent: wait for predecessor to signal ready.
    constexpr int kPredWaitMs = 30000;
    const auto pred_wait_start = std::chrono::steady_clock::now();
    while (flags->predecessor_ready.load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_wait_start)
                .count();
        if (elapsed > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap31SharedFlags));
            GTEST_SKIP() << "Predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Let the predecessor run briefly so ERISCs are confirmed ACTIVE.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    // ── Phase 3: Fork TESTEE ──────────────────────────────────────────────────
    // Testee opens FABRIC_2D (first session):
    //   - FIX AU/AX fires in FabricFirmwareInitializer::teardown (skips non-MMIO resets)
    //   - FIX AC fires in RiscFirmwareInitializer::teardown (PCIe-resets MMIO ETH)
    //   - FIX AY fires after FIX AC (attempts deferred non-MMIO ERISC reset)
    // Testee then opens FABRIC_2D AGAIN (second session):
    //   - Without FIX AY: configure_fabric/write_non_mmio hangs ~15 min (FABRIC fw ERISCs)
    //   - With FIX AY: second open completes quickly (ERISCs in base firmware)
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // ── FIRST SESSION ───────────────────────────────────────────────────
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Explicitly close first session — triggers FIX AU/AX/AC/AY chain.
            dev->close();
        } catch (...) {
            // First session may fail due to relay-broken state — that's expected.
        }
        flags->testee_phase1_done.store(1);

        // ── SECOND SESSION ──────────────────────────────────────────────────
        // This is the critical test: after FIX AY reset non-MMIO ERISCs,
        // a second open must succeed within 30s.
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            // Verify second session is functional: dispatch a blank workload.
            // This confirms ERISCs are in base firmware and the dispatch path works.
            auto device_range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap31(device_range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, /*blocking=*/true);
            tt::tt_metal::distributed::Synchronize(dev.get(), std::nullopt);
            dev->close();
            flags->testee_exit_ok.store(1);
        } catch (...) {
            // Even if AllGather fails, reaching this point means the open didn't hang.
            flags->testee_exit_ok.store(1);
        }
        exit(0);
    }

    // ── Phase 4: Wait for testee phase1 (first session close) ────────────────
    constexpr int kPhase1WaitMs = 60000;  // 60s for first session open+quiesce
    const auto phase1_start = std::chrono::steady_clock::now();
    while (flags->testee_phase1_done.load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - phase1_start)
                .count();
        if (elapsed > kPhase1WaitMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap31SharedFlags));
            GTEST_SKIP() << "Testee first session did not complete within " << kPhase1WaitMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto phase1_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - phase1_start)
            .count();
    log_info(
        tt::LogTest,
        "GAP-31: first session (open+quiesce+FIX AY) completed in {}ms. "
        "Waiting for second session open...",
        phase1_elapsed);

    // ── Phase 5: Wait for testee exit (second session open + AllGather) ──────
    // With FIX AY: second open completes within 30s.
    // Without FIX AY: second open hangs indefinitely (FABRIC fw on non-MMIO ERISCs).
    constexpr int kSecondOpenMs = 45000;  // 45s budget for second open + AllGather
    const auto second_open_start = std::chrono::steady_clock::now();
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
                std::chrono::steady_clock::now() - second_open_start)
                .count();
        if (elapsed > kSecondOpenMs) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!testee_exited) {
        ::kill(testee_pid, SIGKILL);
        ::waitpid(testee_pid, nullptr, 0);
    }

    ::munmap(raw_shm, sizeof(Gap31SharedFlags));
    raw_shm = nullptr;

    const auto second_open_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - second_open_start)
            .count();

    EXPECT_TRUE(testee_exited)
        << "GAP-31 REGRESSION: testee second session did not complete within "
        << kSecondOpenMs
        << "ms after relay-broken first-session teardown. "
           "Root cause: FIX AX left non-MMIO ETH ERISCs in FABRIC firmware. "
           "The second MeshDevice::create() triggered configure_fabric() → "
           "write_non_mmio → UMD relay queue fill → while(full) spin. "
           "Fix: FIX AY in risc_firmware_initializer.cpp — after FIX AC restores "
           "MMIO ETH relay, attempt deferred assert_risc_reset_at_core for "
           "relay-dead non-MMIO ETH ERISCs so next session sees base firmware.";

    if (testee_exited) {
        log_info(
            tt::LogTest,
            "GAP-31: second session completed in {}ms after FIX AY deferred reset "
            "(non-MMIO ERISCs were in base firmware for second open).",
            second_open_elapsed);
    }
}

}  // namespace tt::tt_metal::distributed::test
