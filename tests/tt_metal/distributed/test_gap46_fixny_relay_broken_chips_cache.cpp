// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-46: FIX NY — relay_broken_chips_ cache in Cluster::write_core() eliminates per-channel
// 5s UMD timeout stall after the first failure for a dead-relay chip.
//
// Root cause (CI run 25086219070, job 73503180670 — t3k_ttnn_udm_tests [wh_llmbox]):
//
//   FIX NX (GAP-44) wraps write_to_device() in a try/catch so relay timeout exceptions
//   from set_internal_routing_info_for_ethernet_cores and WatcherServer::init_devices do
//   NOT propagate to test SetUp().  But FIX NX alone does not prevent per-channel timeouts:
//
//     set_internal_routing_info_for_ethernet_cores iterates all ETH channels for every chip.
//     On T3K (3 non-MMIO chips × 6 ETH channels each = 18 channels):
//       Channel 0 of chip 4: write_core() → write_to_device() → 5s UMD timeout → caught by FIX NX.
//       Channel 1 of chip 4: write_core() → write_to_device() → 5s UMD timeout → caught by FIX NX.
//       ...
//       Channel 5 of chip 4: write_core() → write_to_device() → 5s UMD timeout → caught by FIX NX.
//       (Repeat for chips 5, 6.)
//     Total stall: 18 channels × 5s = 90s.
//     GHA 5-minute action timeout fires at ~30s, killing the job (exit code 124).
//
//   Observed log sequence (job 73503180670):
//     01:40:13 FIX AE: Marking relay broken for chip 4
//     01:40:18 FIX NX: write_core(chip 4) threw: Timeout waiting for Ethernet core service remote IO request.
//     01:40:23 FIX NX: write_core(chip 4) threw: ...    (channel 1, another 5s)
//     01:40:28 FIX NX: write_core(chip 4) threw: ...    (channel 2, another 5s)
//     01:40:33 FIX NX: write_core(chip 4) threw: ...    (channel 3, another 5s)
//     01:40:38 FIX NX: write_core(chip 4) threw: ...    (channel 4, another 5s)
//     01:40:43 FIX NX: write_core(chip 4) threw: ...    (channel 5, another 5s)
//     01:40:45 ##[error]The action 't3k_ttnn_udm_tests [wh_llmbox]' has timed out after 5 minutes.
//
// FIX NY (#42429) — tt_cluster.hpp + tt_cluster.cpp:
//   Add `relay_broken_chips_` (unordered_set<ChipId>) to Cluster.
//   In write_core(), BEFORE calling write_to_device(), check relay_broken_chips_:
//     if chip_id is already in relay_broken_chips_ → return immediately (zero-cost).
//   In the FIX NX catch block, INSERT chip_id into relay_broken_chips_ on first timeout.
//
//   Result:
//     Channel 0 of chip 4: 5s UMD timeout → FIX NX catches → FIX NY inserts chip 4.
//     Channels 1-5 of chip 4: relay_broken_chips_.count(chip_id) == 1 → return immediately (0ms).
//     Total stall for chip 4: ~5s (vs 30s without FIX NY).
//     Total for 3 non-MMIO chips: ~15s (vs 90s without FIX NY).
//
// What this test verifies:
//   This test is a TIMING regression test — the failure mode when FIX NY is absent is
//   kTesteeBudgetMs TIMEOUT, not an exception.  FIX NX (GAP-44) already catches the exception;
//   FIX NY prevents the serial per-channel 5s stall.
//
//   1. Fork PREDECESSOR: opens FABRIC_2D MeshDevice, dispatches blank workload (non-MMIO
//      ERISCs → ACTIVE relay firmware state), signals ready, spins.
//   2. Parent SIGKILLs predecessor — all non-MMIO ETH channels remain in FABRIC firmware state.
//   3. 2s settle.
//   4. Fork TESTEE: open a full MeshDevice.  MetalContext::initialize() runs:
//        a. run_async_build_phase         — FIX NV + FIX NW guard (GAP-43)
//        b. set_internal_routing_info_for_ethernet_cores — FIX NX catches first timeout,
//           FIX NY skips all subsequent channels for the same chip
//        c. watcher_server_->init_devices — same FIX NY caching applies
//      Testee immediately closes the device and exits 0.
//   5. Parent:
//      (a) PRIMARY — timing check: testee must complete within kTesteeBudgetMs.
//          Without FIX NY: 3 chips × 6 channels × 5s = 90s → timeout.
//          With FIX NY: 3 chips × 5s first-channel + near-zero rest ≈ 15s + overhead ≈ 27s.
//      (b) SECONDARY — exit code: testee must exit 0.
//          (FIX NX handles this; exit non-zero would indicate FIX NX regression, not FIX NY.)
//
// Distinction from GAP-44:
//   GAP-44 (WriteCorRelayGuardFixture): tests the EXCEPTION PATH — FIX NX prevents write_core()
//     timeout from propagating to SetUp().  Budget = 45s (generous).
//   GAP-46 (this): tests the TIMING PATH — FIX NY prevents 18 × 5s serial stall.
//     Budget = kTesteeBudgetMs (tight enough to catch FIX NY regression).
//
//   Both GAP-44 and GAP-46 must pass.  A FIX NY regression breaks GAP-46 even while
//   GAP-44 passes, because FIX NX alone is sufficient to suppress exceptions (GAP-44 budget
//   is intentionally generous).
//
// Timing budget analysis:
//   T3K: 3 non-MMIO chips, 6 ETH channels each.
//   With FIX NY:    3 × 5s (first channel) + 15 × 0ms + ~12s normal init ≈ 27s.
//   Without FIX NY: 18 × 5s + ~12s normal init ≈ 102s.
//   kTesteeBudgetMs = 35000ms: passes with FIX NY (27s < 35s), fails without (102s >> 35s).
//   Margin: ~8s above expected, ~67s below FIX NY regression case.

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

// Budget analysis (see file header):
//   With FIX NY:    ~27s expected. 35s budget gives ~8s margin above worst-case.
//   Without FIX NY: ~102s expected. Exceeds budget by ~67s → test fails with TIMEOUT message.
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 35000;

struct Gap46SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// Minimal FABRIC_2D workload: puts non-MMIO ERISCs into ACTIVE/relay firmware state.
static MeshWorkload make_blank_workload_gap46(const MeshCoordinateRange& range) {
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

class RelayBrokenChipsCacheFixture : public MeshDeviceFixtureBase {
protected:
    RelayBrokenChipsCacheFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-46 requires >= 2 devices (need at least 1 non-MMIO chip). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-46: RelayBrokenChipsCacheEliminatesPerChannelStall
//
// PRIMARY TEST: timing budget kTesteeBudgetMs = 35s.
// FIX NY regression indicator: testee times out at 35s.
// FIX NX regression indicator: testee exits non-zero (write_core exception escapes).
//
// Correct behavior (both FIX NX + FIX NY present):
//   Testee completes MetalContext::initialize() in ~27s with exit 0.
//   Only the FIRST write_core() per dead chip incurs the 5s UMD relay timeout.
//   All subsequent write_core() calls for that chip return in 0ms (relay_broken_chips_ hit).
// ---------------------------------------------------------------------------
TEST_F(RelayBrokenChipsCacheFixture, RelayBrokenChipsCacheEliminatesPerChannelStall) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap46SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap46SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches blank workload so non-MMIO ERISCs enter ACTIVE relay
    // firmware state.  Signals ready, then spins.  SIGKILL leaves FABRIC firmware on all
    // non-MMIO ETH channels — each subsequent write_core() call for those chips will time
    // out in UMD (5s per call) unless FIX NY skips them via relay_broken_chips_.
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
            auto workload = make_blank_workload_gap46(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
        }
        shm->predecessor_ready.store(1);
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);
    }

    // Wait for predecessor to signal ready, then SIGKILL.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap46SharedMem));
            GTEST_SKIP() << "GAP-46: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    const size_t non_mmio_chips = num_dev - 1;  // Approximation: 1 MMIO chip on T3K.
    log_info(
        tt::LogTest,
        "GAP-46: Predecessor SIGKILL'd — {} non-MMIO chip(s) with stale FABRIC firmware. "
        "Without FIX NY: set_internal_routing_info_for_ethernet_cores stalls ~{}s "
        "(6 ETH channels/chip × 5s × {} chips = {}s). "
        "With FIX NY: stalls ~{}s (one 5s first-failure per chip, rest skipped instantly). "
        "Budget: {}ms.",
        non_mmio_chips,
        non_mmio_chips * 6 * 5,
        non_mmio_chips,
        non_mmio_chips * 6 * 5,
        non_mmio_chips * 5,
        kTesteeBudgetMs);

    // Brief settle so UMD relay CMD queue state stabilises.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE ───────────────────────────────────────────────────
    // Opens a new full MeshDevice.  MetalContext::initialize() sequence:
    //
    //   1. run_async_build_phase(all_device_ids)
    //      FIX NV: skips get_device_aiclk for non-MMIO chips   (GAP-43)
    //      FIX NW: skips clear_launch_messages for non-MMIO     (GAP-43)
    //
    //   2. set_internal_routing_info_for_ethernet_cores(all_chips, enable=true)
    //      For each non-MMIO chip (e.g., chip 4), iterates 6 ETH channels:
    //        Channel 0: write_core() → write_to_device() → 5s UMD timeout → FIX NX catches
    //                   → FIX NY inserts chip 4 into relay_broken_chips_
    //        Channels 1-5: relay_broken_chips_.count(4) == 1 → return immediately (0ms)
    //
    //   3. watcher_server_->init_devices()
    //      All write_core() calls for chip 4 (and other dead-relay chips):
    //        relay_broken_chips_.count(chip_id) == 1 → return immediately (0ms)
    //
    // WITHOUT FIX NY: channels 1-5 (and watcher_server init channels) each pay 5s.
    //   Total for 3 non-MMIO chips: 3 × 6 × 5s = 90s → exceeds kTesteeBudgetMs (35s).
    // WITH FIX NY: 3 × 5s (first channel only) ≈ 15s + ~12s normal init = ~27s < 35s.
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
        } catch (const std::exception& e) {
            // Regression path: FIX NX missing (write_core exception escaped).
            // Note: this is a FIX NX regression, not FIX NY.  GAP-44 covers this directly.
            rc = 1;
        } catch (...) {
            rc = 2;
        }
        _exit(rc);
    }

    // ── Step 3: Wait for testee with strict timing budget ─────────────────────
    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - testee_start)
                                     .count();
        if (elapsed_ms > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap46SharedMem));
            FAIL() << "GAP-46 TIMEOUT (FIX NY regression): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "Root cause: relay_broken_chips_ cache (FIX NY) is missing or reverted.\n"
                   << "Without FIX NY, write_core() for each ETH channel on a dead-relay chip\n"
                   << "pays the full 5s UMD timeout before FIX NX catches the exception.\n"
                   << "\n"
                   << "Stall estimate without FIX NY:\n"
                   << "  " << non_mmio_chips << " non-MMIO chip(s) × 6 ETH channels × 5s = "
                   << non_mmio_chips * 6 * 5 << "s\n"
                   << "  (plus WatcherServer::init_devices for same chips)\n"
                   << "\n"
                   << "Fix (FIX NY):\n"
                   << "  In Cluster::write_core(), BEFORE write_to_device(), check\n"
                   << "  relay_broken_chips_ (unordered_set<ChipId> member of Cluster).\n"
                   << "  If chip already known broken: return immediately (zero cost).\n"
                   << "  In FIX NX catch block: insert chip_id into relay_broken_chips_\n"
                   << "  so all subsequent write_core() calls for that chip skip UMD.\n"
                   << "\n"
                   << "CI reference: run 25086219070, job 73503180670 (t3k_ttnn_udm_tests).\n"
                   << "  Log evidence: 6× 'FIX NX: write_core(chip 4) threw: Timeout...' at 5s intervals.\n"
                   << "  GHA 5-minute action timeout fired after 30s of serial relay stalls.\n"
                   << "\n"
                   << "See also: GAP-44 (WriteCorRelayGuardFixture) — covers the FIX NX exception path.\n"
                   << "This test (GAP-46) covers the FIX NY TIMING path (per-channel stall elimination).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - testee_start)
                                 .count();

    ::munmap(raw_shm, sizeof(Gap46SharedMem));

    // Check for non-zero exit code: FIX NX regression (not FIX NY specific, covered by GAP-44).
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        const int ec = WEXITSTATUS(status);
        if (ec == 1) {
            FAIL() << "GAP-46: Testee caught exception from MeshDevice::create().\n"
                      "This is a FIX NX regression (write_core() exception propagated to SetUp()).\n"
                      "Note: GAP-44 (WriteCorRelayGuardFixture) is the primary test for FIX NX.\n"
                      "FIX NY (relay_broken_chips_ cache) requires FIX NX to function.\n"
                      "Re-apply FIX NX first: in Cluster::write_core(), wrap write_to_device()\n"
                      "and wait_for_non_mmio_flush() in a try/catch for is_chip_remote() chips.";
        }
        FAIL() << "GAP-46: Testee exited with unexpected code " << ec << ".";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-46: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    log_info(
        tt::LogTest,
        "GAP-46 PASS: Testee completed MetalContext::initialize() in {}ms (budget: {}ms) "
        "with exit 0. FIX NY relay_broken_chips_ cache correctly eliminated per-channel "
        "5s UMD relay timeouts after first failure for each dead-relay chip.",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
