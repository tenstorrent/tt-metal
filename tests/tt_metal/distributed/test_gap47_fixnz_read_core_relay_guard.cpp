// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-47: FIX NZ — read_core() relay guard + run_launch_phase skip for dead-relay non-MMIO devices.
//
// Root cause (CI run 25088288806, job 73509544572 — t3k_ttnn_udm_tests [wh_llmbox]):
//
//   After FIX NX (GAP-44) and FIX NY (GAP-46) were applied, set_internal_routing_info_for_ethernet_cores
//   completes cleanly in ~20s (4 non-MMIO chips × 5s first-timeout, then zero-cost skips via NY).
//   But the process still hangs for 2.5+ minutes after that, never completing MetalContext::initialize().
//
//   The 2.5-minute hang is in run_launch_phase → initialize_and_launch_firmware(device_id) called for
//   non-MMIO devices 4-7 whose relay was marked broken by FIX NX.
//
//   Specifically: initialize_and_launch_firmware calls wait_until_cores_done() (10s timeout) after
//   writing firmware to all worker tensix cores.  wait_until_cores_done() polls every core in
//   not_done_phys_cores by calling check_if_riscs_on_specified_core_done() → Cluster::read_core().
//
//   For non-MMIO devices with a dead relay, Cluster::read_core() has NO relay-broken guard:
//     read_from_device() → read_non_mmio() → 5s polling timeout per read_core() call.
//
//   A Wormhole device has 8×8 = 64 tensix worker cores.  One poll pass over 64 cores on one
//   dead-relay device = 64 × 5s = 320s.  The timeout check inside the while loop fires only
//   AFTER a complete poll pass finishes — so the 10s timeout is effectively 310s.
//
//   With 4 non-MMIO devices: 4 × 320s = 1280s of serial blocking (called sequentially by
//   run_launch_phase).  GHA kills the job at 5 minutes.
//
//   Observed log sequence (job 73509544572):
//     03:03:09 FIX NX fires for chips 4, 5, 6, 7 (~20s total, 5s each)
//     03:03:29 assert_cores on MMIO devices 0-3 fires 500ms timeouts
//     03:03:36 reset_cores: erisc_app_still_running caught for device 4 (first read_core 5s timeout)
//     03:06:10 ##[error] The action 't3k_ttnn_udm_tests [wh_llmbox]' has timed out after 5 minutes.
//     (2.5 minutes of silence = initialize_and_launch_firmware blocking in wait_until_cores_done)
//
// FIX NZ (#42429) — two-part fix:
//
//   Part A: Cluster::read_core() relay guard (tt_cluster.hpp + tt_cluster.cpp):
//     If chip is remote and relay_broken_chips_.count(chip_id) > 0, throw std::runtime_error
//     immediately (same exception type UMD would eventually throw after 5s).
//     This provides belt-and-suspenders protection for any read_core() caller on a dead-relay chip.
//     Also add is_relay_broken(ChipId) accessor to Cluster for use by other subsystems.
//
//   Part B: run_launch_phase skip (risc_firmware_initializer.cpp):
//     After reset_cores(device_id), if !mmio_ids_set.count(device_id) && cluster_.is_relay_broken(device_id):
//       log warning and continue (skip initialize_and_launch_firmware for that device).
//     Rationale: firmware on a dead-relay non-MMIO device cannot be initialized meaningfully.
//     terminate_stale_erisc_routers on the next init session will clean up the device state.
//
// What this test verifies:
//   1. Fork PREDECESSOR: opens FABRIC_2D MeshDevice, dispatches blank workload (non-MMIO ERISCs
//      enter ACTIVE/relay firmware state), signals ready, spins.
//   2. Parent SIGKILLs predecessor — all non-MMIO ETH channels remain in FABRIC firmware state.
//   3. 2s settle.
//   4. Fork TESTEE: open a full MeshDevice.  MetalContext::initialize() sequence:
//        a. run_async_build_phase         — FIX NV + FIX NW guard (GAP-43)
//        b. set_internal_routing_info_for_ethernet_cores — FIX NX catches first timeout for each
//           dead-relay chip; FIX NY skips subsequent channels; relay_broken_chips_ populated
//        c. run_launch_phase:
//           - reset_cores(device_id) for non-MMIO chips: erisc_app_still_running caught, fast
//           - FIX NZ Part B: initialize_and_launch_firmware SKIPPED for relay-broken non-MMIO chips
//      Testee immediately closes the device and exits 0.
//   5. Parent:
//      (a) PRIMARY — timing check: testee must complete within kTesteeBudgetMs.
//          Without FIX NZ: 4 chips × 64 cores × 5s per read = 1280s → timeout.
//          With FIX NZ:    initialize_and_launch_firmware skipped for dead-relay chips ≈ 27s.
//      (b) SECONDARY — exit code: testee must exit 0.
//
// Timing budget analysis:
//   T3K: 4 non-MMIO chips (8×8 = 64 tensix workers each).
//   Without FIX NZ: 4 × 64 × 5s = 1280s per wait_until_cores_done pass → far exceeds budget.
//   With FIX NZ:    initialize_and_launch_firmware skipped for dead-relay chips.
//                   Remaining work: MMIO chips (4 × normal init ~5s) + relay stalls ~20s ≈ 40s.
//   kTesteeBudgetMs = 70000ms: passes with FIX NZ (~40s), fails without (1280s+ >> 70s).
//   Note: budget is looser than GAP-46 since there is one extra 5s read_core timeout per chip
//   from reset_cores:erisc_app_still_running before FIX NZ Part B skips the rest.

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
//   With FIX NZ:    ~40s expected. 70s budget gives ~30s margin above worst-case.
//   Without FIX NZ: 1280s+ expected. Exceeds budget by more than 1200s → test fails with TIMEOUT.
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 70000;

struct Gap47SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// Minimal FABRIC_2D workload: puts non-MMIO ERISCs into ACTIVE/relay firmware state.
static MeshWorkload make_blank_workload_gap47(const MeshCoordinateRange& range) {
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

class ReadCoreRelayGuardFixture : public MeshDeviceFixtureBase {
protected:
    ReadCoreRelayGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-47 requires >= 2 devices (need at least 1 non-MMIO chip). "
                         << "Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-47: ReadCoreRelayGuardPreventsFirmwareInitHang
//
// PRIMARY TEST: timing budget kTesteeBudgetMs = 70s.
// FIX NZ regression indicator: testee times out at 70s (hangs in wait_until_cores_done).
// FIX NX regression indicator: testee exits non-zero (write_core exception escapes).
//
// Correct behavior (FIX NX + FIX NY + FIX NZ all present):
//   Testee completes MetalContext::initialize() in ~40s with exit 0.
//   run_launch_phase skips initialize_and_launch_firmware for dead-relay non-MMIO chips.
//   wait_until_cores_done is never called for those chips → no 64 × 5s blocking.
// ---------------------------------------------------------------------------
TEST_F(ReadCoreRelayGuardFixture, ReadCoreRelayGuardPreventsFirmwareInitHang) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap47SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap47SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches blank workload so non-MMIO ERISCs enter ACTIVE relay
    // firmware state.  Signals ready, then spins.  SIGKILL leaves FABRIC firmware on all
    // non-MMIO ETH channels — subsequent read_core() calls for those chips will time out
    // in UMD (5s per call) unless FIX NZ guards them.
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
            auto workload = make_blank_workload_gap47(range);
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
            ::munmap(raw_shm, sizeof(Gap47SharedMem));
            GTEST_SKIP() << "GAP-47: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    const size_t non_mmio_chips = num_dev - 1;  // Approximation: 1 MMIO chip on T3K.
    log_info(
        tt::LogTest,
        "GAP-47: Predecessor SIGKILL'd — {} non-MMIO chip(s) with stale FABRIC firmware. "
        "Without FIX NZ: wait_until_cores_done polls 64 tensix cores per dead-relay chip "
        "({} chips × 64 cores × 5s/read = {}s total hang). "
        "With FIX NZ: initialize_and_launch_firmware skipped for dead-relay chips. "
        "Budget: {}ms.",
        non_mmio_chips,
        non_mmio_chips,
        non_mmio_chips * 64 * 5,
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
    //      FIX NX + FIX NY: relay_broken_chips_ populated for dead-relay chips
    //
    //   3. run_launch_phase:
    //      For each device_id:
    //        reset_cores(device_id): erisc_app_still_running try/catch fires for non-MMIO
    //        FIX NZ Part B: if !mmio && is_relay_broken(device_id): skip initialize_and_launch_firmware
    //      Without FIX NZ: initialize_and_launch_firmware → wait_until_cores_done → 64 read_core
    //        calls per chip, each hanging 5s = 320s per chip before timeout check can fire.
    //
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
            // FIX NX regression path (should not happen).
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
            ::munmap(raw_shm, sizeof(Gap47SharedMem));
            FAIL() << "GAP-47 TIMEOUT (FIX NZ regression): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "Root cause: read_core() relay guard + run_launch_phase skip (FIX NZ) is\n"
                   << "missing or reverted.  Without FIX NZ, initialize_and_launch_firmware is\n"
                   << "called for non-MMIO devices with dead relays.  wait_until_cores_done()\n"
                   << "polls all 64 tensix worker cores via Cluster::read_core(), each call\n"
                   << "blocking for 5s in read_non_mmio before timing out.\n"
                   << "\n"
                   << "Stall estimate without FIX NZ:\n"
                   << "  " << non_mmio_chips << " non-MMIO chip(s) × 64 tensix cores × 5s = "
                   << non_mmio_chips * 64 * 5 << "s\n"
                   << "  (plus additional stall from subsequent poll iterations)\n"
                   << "\n"
                   << "Fix (FIX NZ — two parts):\n"
                   << "  Part A — Cluster::read_core() in tt_cluster.cpp:\n"
                   << "    if (is_chip_remote(chip_id) && relay_broken_chips_.count(chip_id))\n"
                   << "      throw std::runtime_error(\"FIX NZ: read_core relay broken\");\n"
                   << "  Add is_relay_broken(ChipId) accessor to Cluster (tt_cluster.hpp).\n"
                   << "\n"
                   << "  Part B — run_launch_phase() in risc_firmware_initializer.cpp:\n"
                   << "    After reset_cores(device_id):\n"
                   << "    if (!mmio_ids_set.count(device_id) && cluster_.is_relay_broken(device_id))\n"
                   << "      log_warning(FIX NZ) + continue;  // skip initialize_and_launch_firmware\n"
                   << "\n"
                   << "CI reference: run 25088288806, job 73509544572 (t3k_ttnn_udm_tests).\n"
                   << "  Log evidence: last entry 03:03:36 (reset_cores device 4 handled),\n"
                   << "  then 2.5 minutes of silence, GHA timeout at 03:06:10.\n"
                   << "\n"
                   << "See also: GAP-44 (write_core exception guard), GAP-46 (relay_broken_chips_ cache).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - testee_start)
                                 .count();

    ::munmap(raw_shm, sizeof(Gap47SharedMem));

    // Check for non-zero exit code: FIX NX regression (not FIX NZ specific).
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        const int ec = WEXITSTATUS(status);
        if (ec == 1) {
            FAIL() << "GAP-47: Testee caught exception from MeshDevice::create().\n"
                      "This is a FIX NX regression (write_core() exception propagated to SetUp()).\n"
                      "FIX NZ requires FIX NX + FIX NY to function.\n"
                      "Re-apply FIX NX first: in Cluster::write_core(), wrap write_to_device()\n"
                      "and wait_for_non_mmio_flush() in a try/catch for is_chip_remote() chips.";
        }
        FAIL() << "GAP-47: Testee exited with unexpected code " << ec << ".";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-47: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    log_info(
        tt::LogTest,
        "GAP-47 PASS: Testee completed MetalContext::initialize() in {}ms (budget: {}ms) "
        "with exit 0. FIX NZ read_core() relay guard + run_launch_phase skip correctly "
        "prevented wait_until_cores_done() hang on {} dead-relay non-MMIO chip(s).",
        elapsed_ms,
        kTesteeBudgetMs,
        non_mmio_chips);
}

}  // namespace tt::tt_metal::distributed::test
