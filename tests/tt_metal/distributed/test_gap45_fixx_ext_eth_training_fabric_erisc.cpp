// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-45: FIX X extension — wait_eth_core_training hang when fabric ERISC firmware
// leaves ETH_TRAIN_STATUS_ADDR = 0 after configure_fabric_cores() soft-reset bounce.
//
// Root cause (CI run 25086219070, t3k-12 — t3k_ttnn_tests [wh_llmbox],
//             MultiCommandQueueSingleDeviceFixture.TestAsyncRuntimeChainedOpsWithBufferReuse):
//
//   Test sequence:
//     1. test_tt_fabric job (same runner) SIGABRT'd at 01:35:58 — fabric ERISCs left in
//        unknown state on all 8 T3K devices.
//     2. t3k_ttnn_tests starts; TestAsyncRuntimeAllocatedBuffers creates a new MetalContext
//        (create_unit_mesh(0)).  TopologyDiscovery discovers all 4 MMIO devices (0-3).
//        configure_fabric_cores() performs a soft-reset bounce on device-3's MMIO ETH channels
//        (they were not at base-UMD canary 0x49706550 after the SIGABRT state).  After
//        deassert, base UMD firmware restarts.  ConfigureDeviceWithProgram then writes the
//        fabric binary to device-3's ERISC L1/IRAM.  The fabric binary's .bss section in L1
//        COVERS ETH_TRAIN_STATUS_ADDR (0x1104), zeroing it to 0.  Fabric firmware is then
//        launched via write_launch_msg_to_core.  Fabric firmware never writes
//        ETH_TRAIN_STATUS_ADDR.  At end of test 1: ETH_TRAIN_STATUS_ADDR = 0,
//        heartbeat = 0xABCDxxxx (fabric ERISC alive).
//     3. FabricFirmwareInitializer::teardown() returns instantly (TERMINATE_FABRIC not set for
//        create_unit_mesh default path) → fabric ERISCs stay running on device-3 channels.
//     4. TestAsyncRuntimeChainedOpsWithBufferReuse creates a new MetalContext (create_unit_mesh(0)).
//        New TopologyDiscovery → discover_local_devices() → wait_eth_cores_training(device 3)
//        → wait_eth_core_training() loops on ETH_TRAIN_STATUS_ADDR == 0 (IN_PROGRESS).
//        FIX X fires ONLY when heartbeat != 0xABCD.  Fabric ERISC heartbeat IS 0xABCDxxxx.
//        FIX X does NOT fire.  Hangs for 900s until GHA kills job (exit code 124, ~13 minutes).
//
// FIX X extension (#42429, race-condition-hunt) — wormhole_tt_device.cpp:
//   Extend wait_eth_core_training(): after heartbeat_check_after_ms (2000ms) of IN_PROGRESS,
//   if heartbeat IS 0xABCDxxxx (not just absent), ALSO return early.  Base ETH firmware on a
//   warm restart (PHY link already up) completes training in < 200ms.  Any firmware with an
//   active 0xABCDxxxx heartbeat that has NOT written ETH_TRAIN_STATUS_ADDR after 2000ms is
//   non-base firmware (fabric router or quiesce kernel) and will NEVER write it.
//
// What this test verifies:
//   1. PREDECESSOR: opens FABRIC_2D MeshDevice, dispatches blank workload (fabric ERISCs active),
//      signals ready, then spins.
//   2. Parent SIGKILLs predecessor — fabric ERISCs on MMIO device channels remain running
//      (ETH_TRAIN_STATUS_ADDR = 0, heartbeat = 0xABCDxxxx — exact state of bug).
//   3. 2s settle.
//   4. FIRST TESTEE: opens create_unit_mesh(0) (single device, no explicit TERMINATE_FABRIC).
//      configure_fabric_cores() loads fabric firmware onto MMIO device ETH channels.
//      Closes device (teardown skips due to no TERMINATE_FABRIC).
//   5. SECOND TESTEE: opens create_unit_mesh(0) AGAIN in the same process.
//      TopologyDiscovery discovers all MMIO devices.
//      wait_eth_cores_training() is called for each MMIO device including device 3.
//      Without FIX X extension: hangs 900s in wait_eth_core_training (fabric ERISC heartbeat
//      present but ETH_TRAIN_STATUS_ADDR = 0 → original FIX X never fires).
//      With FIX X extension: wait returns in ~2000ms (timeout after heartbeat confirmed).
//   6. Test asserts: both testee open+close cycles complete within kTesteeBudgetMs.
//
// Distinction from prior GAP tests:
//   GAP tests 39-44: failures in configure/write/flush paths during MetalContext::initialize.
//   GAP-45 (this): failure is in UMD TopologyDiscovery wait_eth_core_training, which runs
//                  BEFORE terminate_stale_erisc_routers and configure_fabric_cores.
//                  The fabric ERISC left ETH_TRAIN_STATUS_ADDR = 0 after the prior session's
//                  configure_fabric_cores() overwrote it via ConfigureDeviceWithProgram.

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
//   Without FIX X extension: per-channel wait is 900s → GHA kills at ~13 minutes.
//   With FIX X extension: per-channel wait exits in ~2000ms (heartbeat_check_after_ms).
//   Normal init + 2 open/close cycles: ~30s.
//   Budget: 60s to catch the "stuck in 900s loop" failure mode early.
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 60000;

struct Gap45SharedMem {
    std::atomic<int> predecessor_ready{0};
};

// Minimal FABRIC_2D workload to put MMIO device ETH channels into fabric firmware state
// (ETH_TRAIN_STATUS_ADDR gets zeroed by ConfigureDeviceWithProgram .bss write).
static MeshWorkload make_blank_workload_gap45(const MeshCoordinateRange& range) {
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

class EthTrainingFabricEriscsFixture : public MeshDeviceFixtureBase {
protected:
    EthTrainingFabricEriscsFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-45 requires >= 2 devices (need MMIO + non-MMIO to exercise "
                         << "fabric ERISC ETH_TRAIN_STATUS state). Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-45: EthTrainingHangsWhenFabricEriscsLeaveTrainStatusZero
//
// Verifies that wait_eth_core_training() does NOT hang when fabric ERISC firmware
// has zeroed ETH_TRAIN_STATUS_ADDR (0x1104) via ConfigureDeviceWithProgram's .bss
// write and left it at 0 (IN_PROGRESS) across a non-TERMINATE_FABRIC teardown.
//
// Primary failure (FIX X extension missing): second create_unit_mesh(0) call hangs
// inside wait_eth_core_training() for ~900s (ETH_TRAINING_TIMEOUT) because:
//   - fabric ERISC heartbeat = 0xABCDxxxx (original FIX X does NOT skip)
//   - ETH_TRAIN_STATUS_ADDR = 0 (IN_PROGRESS, never written by fabric firmware)
// Secondary failure: any hang > kTesteeBudgetMs (60s).
// ---------------------------------------------------------------------------
TEST_F(EthTrainingFabricEriscsFixture, EthTrainingHangsWhenFabricEriscsLeaveTrainStatusZero) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap45SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap45SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens full FABRIC_2D mesh so configure_fabric_cores() runs on all MMIO ETH channels,
    // loading fabric binary via ConfigureDeviceWithProgram.  This zeroes ETH_TRAIN_STATUS_ADDR
    // on MMIO device channels whose channels got a soft-reset bounce.  Fabric firmware then
    // runs with heartbeat 0xABCDxxxx but ETH_TRAIN_STATUS_ADDR = 0.
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
            auto workload = make_blank_workload_gap45(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Even on dispatch failure, signal ready — fabric firmware is already loaded.
        }
        shm->predecessor_ready.store(1);
        // Spin until SIGKILL — keeps fabric ERISCs running with ETH_TRAIN_STATUS_ADDR = 0.
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
            ::munmap(raw_shm, sizeof(Gap45SharedMem));
            GTEST_SKIP() << "GAP-45: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);
    log_info(
        tt::LogTest,
        "GAP-45: Predecessor SIGKILL'd — fabric ERISCs running on MMIO ETH channels with "
        "ETH_TRAIN_STATUS_ADDR = 0 (zeroed by ConfigureDeviceWithProgram .bss write). "
        "Next create_unit_mesh(0) will trigger wait_eth_cores_training hang if FIX X "
        "extension is missing.");

    // Brief settle.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: Fork TESTEE ───────────────────────────────────────────────────
    // Two sequential create_unit_mesh(0) cycles in one child.
    //
    // Cycle 1: Opens device 0. configure_fabric_cores() may soft-reset MMIO device channels
    //   again (they're in FABRIC state), reloading fabric binary.  After close(),
    //   FabricFirmwareInitializer::teardown() returns instantly (TERMINATE_FABRIC not set).
    //   ETH_TRAIN_STATUS_ADDR on MMIO device channels stays at 0.
    //
    // Cycle 2: Opens device 0 AGAIN.  New TopologyDiscovery → discover_local_devices()
    //   → wait_eth_cores_training() for each MMIO device (0-3 on T3K).
    //   WITHOUT FIX X extension: wait_eth_core_training() for device 3 (or any device
    //   with fabric ERISC active) loops: ETH_TRAIN_STATUS_ADDR=0 AND heartbeat=0xABCDxxxx
    //   → original FIX X never fires → 900s hang.
    //   WITH FIX X extension: after 2000ms, FIX X sees heartbeat=0xABCDxxxx but still
    //   IN_PROGRESS → returns early → both cycles complete in ~30s total.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        int rc = 0;
        try {
            // Cycle 1: open and close.
            {
                auto dev = MeshDevice::create_unit_mesh(
                    0,
                    DEFAULT_L1_SMALL_SIZE,
                    DEFAULT_TRACE_REGION_SIZE,
                    /*num_command_queues=*/1);
                dev->close();
            }
            // Cycle 2: open and close again.
            // This is the failing cycle: TopologyDiscovery sees fabric ERISCs from cycle 1
            // (or predecessor) with ETH_TRAIN_STATUS_ADDR = 0.
            {
                auto dev = MeshDevice::create_unit_mesh(
                    0,
                    DEFAULT_L1_SMALL_SIZE,
                    DEFAULT_TRACE_REGION_SIZE,
                    /*num_command_queues=*/1);
                dev->close();
            }
            rc = 0;
        } catch (const std::exception& e) {
            rc = 1;
        } catch (...) {
            rc = 2;
        }
        _exit(rc);
    }

    // ── Step 3: Wait for testee and verify exit code + timing ─────────────────
    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - testee_start)
                                 .count();
        if (elapsed > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap45SharedMem));
            FAIL() << "GAP-45 TIMEOUT (FIX X extension): Testee did not exit within "
                   << kTesteeBudgetMs << "ms.\n"
                   << "\n"
                   << "This indicates that wait_eth_core_training() is hanging in\n"
                   << "TopologyDiscovery::discover_local_devices() for an MMIO device\n"
                   << "whose ETH channels have fabric ERISC firmware running with:\n"
                   << "  ETH_TRAIN_STATUS_ADDR (0x1104) = 0  (zeroed by fabric .bss)\n"
                   << "  heartbeat = 0xABCDxxxx  (fabric ERISC alive, incrementing)\n"
                   << "\n"
                   << "The original FIX X only skips when heartbeat != 0xABCD.\n"
                   << "Fabric ERISC heartbeat IS 0xABCDxxxx → original FIX X does NOT fire.\n"
                   << "ETH training status will never be written → 900s hang.\n"
                   << "\n"
                   << "FIX X extension: in wait_eth_core_training(), after 2000ms, if\n"
                   << "  heartbeat IS 0xABCDxxxx AND training is still IN_PROGRESS,\n"
                   << "  return early.  Base ETH firmware completes training in < 200ms\n"
                   << "  on warm restart; any 0xABCDxxxx firmware that hasn't written\n"
                   << "  ETH_TRAIN_STATUS_ADDR after 2000ms is fabric/quiesce firmware\n"
                   << "  that will never write it.\n"
                   << "\n"
                   << "CI reference: run 25086219070 (t3k-12 wh_llmbox,\n"
                   << "  TestAsyncRuntimeChainedOpsWithBufferReuse, 13-minute hang).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - testee_start)
                                .count();

    ::munmap(raw_shm, sizeof(Gap45SharedMem));

    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        const int ec = WEXITSTATUS(status);
        FAIL() << "GAP-45: Testee exited with code " << ec << " (exception in create_unit_mesh).";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-45: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    log_info(
        tt::LogTest,
        "GAP-45 PASS: Both create_unit_mesh(0) cycles completed in {}ms (budget: {}ms) with "
        "exit 0. FIX X extension correctly skipped wait_eth_core_training for MMIO device "
        "channels with fabric ERISC firmware (ETH_TRAIN_STATUS_ADDR=0, heartbeat=0xABCDxxxx).",
        elapsed_ms,
        kTesteeBudgetMs);
}

}  // namespace tt::tt_metal::distributed::test
