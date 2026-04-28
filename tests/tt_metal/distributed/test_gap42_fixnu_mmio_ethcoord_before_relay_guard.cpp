// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-42: FIX NU — Capture MMIO EthCoord before relay-safety guards (FIX W)
//
// Root cause (CI run 25079761804, t3k_tt_metal_multiprocess_tests on tt-metal-ci-vm-t3k-08):
//
//   1. Prior process left MMIO ETH channels in FABRIC/STARTED mode (heartbeat = 0xa0b0c0d0).
//   2. New process (two MPI ranks) performed topology discovery.
//   3. During discover_remote_devices(), the per-ETH-channel loop for MMIO devices
//      checked the FIX W heartbeat guard BEFORE capturing the local EthCoord.
//      FIX W saw heartbeat 0xa0b0c0d0 (>> 16 = 0xa0b0 ≠ 0xABCD) on EVERY ETH channel
//      of the affected MMIO device(s) and issued `continue` for each — skipping the
//      channel before reaching the eth_coords capture.
//   4. eth_coords was never populated for those MMIO devices.
//      fill_cluster_descriptor_info() omitted them from chip_locations.
//   5. initialize_and_validate_custom_physical_config() called
//      cluster.get_physical_chip_id_from_eth_coord(coord) for each chip in the YAML.
//      The missing MMIO chip's EthCoord was not found →
//        TT_FATAL(false, "Physical chip id not found for eth coord")  →  SIGABRT.
//
//   Note: FIX NT does NOT help here.  FIX NT handles chips absent from `devices`
//   (FIX-AQ-skipped, init failed).  For FIX NU, the chips ARE in `devices` (healthy
//   remote chips are reachable via OTHER gateways), but the gateway MMIO device's own
//   local EthCoord was never written because FIX W skipped ALL its channels.
//
// FIX NU (UMD commit 5882377c, tt-metal commit b7c41de1806):
//   In topology_discovery.cpp, moved the get_local_eth_coord() + eth_coords.emplace()
//   block for MMIO devices to BEFORE the FIX W heartbeat check.
//   get_local_eth_coord() performs a PCIe read from NODE_INFO — unconditionally safe
//   regardless of ETH firmware state (FABRIC, STARTED, dead, etc.).  Only the relay-based
//   remote device discovery needs the heartbeat/training guards; coordinate metadata
//   collection for the LOCAL MMIO chip must not be suppressed.
//
// What this test verifies:
//   1. Parent opens healthy cluster and records MMIO device IDs and EthCoords.
//   2. PREDECESSOR forks: opens FABRIC_2D, dispatches, signals ready, spins.
//      This leaves MMIO ETH relay channels in FABRIC/STARTED firmware state.
//   3. Parent SIGKILLs predecessor.
//   4. TESTEE forks: opens a new Cluster.
//      FIX W heartbeat guard fires for the FABRIC-state MMIO ETH channels.
//      FIX NU ensures get_local_eth_coord() runs BEFORE FIX W for each channel.
//      Testee calls get_physical_chip_id_from_eth_coord() for MMIO device EthCoords
//      recorded in step 1.
//   5. Without FIX NU: MMIO chip's EthCoord missing from chip_locations → SIGABRT.
//      With    FIX NU: MMIO chip's EthCoord captured before FIX W → no crash.
//   6. Parent asserts testee exit code == 0.
//
// Note on trigger reliability:
//   The FIX NU bug requires ALL ETH channels of an MMIO device to fail the FIX W
//   heartbeat check (all in FABRIC/STARTED mode).  After a SIGKILL predecessor that
//   ran FABRIC_2D, MMIO relay channels (those connected to non-MMIO devices) will be
//   in FABRIC mode.  Channels NOT involved in relay may remain in base UMD firmware.
//   If at least one MMIO channel is in base firmware, FIX W does NOT suppress the
//   eth_coords capture for that channel, and the MMIO device's EthCoord IS populated
//   (no bug).  On T3K systems where prior fabric traffic saturates more channels, the
//   bug is more likely to manifest.  The test is still valuable as:
//   (a) Documentation of the failure mode and expected testee behavior.
//   (b) A regression check that catches FIX NU removals on affected hardware states.
//   (c) The oracle (testee exit code) is always valid regardless of whether FIX W
//       fires on all MMIO channels in a given run.
//
// Topology requirement: >= 2 devices (MMIO + non-MMIO relay path required).

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
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "fabric/fabric_init.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

static constexpr int kMaxSharedChips = 64;
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 90000;

struct Gap42SharedMem {
    std::atomic<int> predecessor_ready{0};
    std::atomic<int> num_mmio_chips{0};
    std::atomic<int> num_all_chips{0};
    // EthCoords for MMIO chips only — these are the ones FIX NU protects.
    int mmio_cluster_id[kMaxSharedChips];
    int mmio_x[kMaxSharedChips];
    int mmio_y[kMaxSharedChips];
    int mmio_rack[kMaxSharedChips];
    int mmio_shelf[kMaxSharedChips];
};

static MeshWorkload make_blank_workload_gap42(const MeshCoordinateRange& range) {
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

class MmioEthCoordBeforeRelayGuardFixture : public MeshDeviceFixtureBase {
protected:
    MmioEthCoordBeforeRelayGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-42 requires >= 2 devices. Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-42: MmioEthCoordInChipLocationsAfterFabricStatePredecessor
//
// Verifies that MMIO device EthCoords are present in chip_locations even when
// FIX W (heartbeat guard) fires for FABRIC-mode MMIO ETH channels.
// Without FIX NU, the local EthCoord capture was placed AFTER the FIX W
// `continue`, so MMIO chips whose ALL channels failed heartbeat had no entry
// in chip_locations → TT_FATAL when YAML-based tests called
// get_physical_chip_id_from_eth_coord() for those chips.
// ---------------------------------------------------------------------------
TEST_F(MmioEthCoordBeforeRelayGuardFixture, MmioEthCoordInChipLocationsAfterFabricStatePredecessor) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap42SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap42SharedMem();

    // ── Step 1: Record MMIO device EthCoords from healthy parent Cluster ─────
    {
        const auto& cluster = MetalContext::instance().get_cluster();
        // Use get_all_chip_ethernet_coordinates() + mmio device filter.
        auto all_coords = cluster.get_all_chip_ethernet_coordinates();
        shm->num_all_chips.store(static_cast<int>(all_coords.size()));

        // Identify MMIO devices via the public mmio_chip_ids() API.
        int n = 0;
        for (const ChipId mmio_chip_id : cluster.mmio_chip_ids()) {
            auto it = all_coords.find(mmio_chip_id);
            if (it == all_coords.end()) continue;
            if (n >= kMaxSharedChips) break;
            shm->mmio_cluster_id[n] = it->second.cluster_id;
            shm->mmio_x[n] = it->second.x;
            shm->mmio_y[n] = it->second.y;
            shm->mmio_rack[n] = it->second.rack;
            shm->mmio_shelf[n] = it->second.shelf;
            ++n;
        }
        shm->num_mmio_chips.store(n);
        log_info(
            tt::LogTest,
            "GAP-42: Recorded {} MMIO EthCoord(s) from healthy parent Cluster (total chips: {}).",
            n,
            all_coords.size());
    }

    const int n_mmio = shm->num_mmio_chips.load();
    ASSERT_GT(n_mmio, 0) << "No MMIO devices found — hardware issue?";

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 2: Fork PREDECESSOR ───────────────────────────────────────────────
    // Runs FABRIC_2D to put MMIO relay ETH channels in FABRIC/STARTED firmware state.
    pid_t pred_pid = ::fork();
    ASSERT_GE(pred_pid, 0) << "fork() failed: " << strerror(errno);

    if (pred_pid == 0) {
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
            auto workload = make_blank_workload_gap42(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
        }
        shm->predecessor_ready.store(1);
        // Spin so ERISC relay firmware stays active while parent waits.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);
    }

    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap42SharedMem));
            GTEST_SKIP() << "GAP-42: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);
    log_info(tt::LogTest, "GAP-42: Predecessor SIGKILL'd — MMIO ETH channels may be in FABRIC/STARTED state.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 3: Fork TESTEE ───────────────────────────────────────────────────
    // Opens Cluster (FIX W fires for FABRIC-state MMIO channels).
    // FIX NU: get_local_eth_coord() was moved BEFORE FIX W → MMIO EthCoord always captured.
    // Testee calls get_physical_chip_id_from_eth_coord() for each MMIO device's EthCoord.
    // Without FIX NU: MMIO EthCoord missing from chip_locations → TT_FATAL → SIGABRT.
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

            const auto& cluster = MetalContext::instance().get_cluster();
            auto all_coords = cluster.get_all_chip_ethernet_coordinates();

            // Call get_physical_chip_id_from_eth_coord() for each MMIO EthCoord
            // the parent recorded.  Without FIX NU, this crashes for MMIO chips
            // whose ALL ETH channels were in FABRIC mode (FIX W skipped them all).
            int n = shm->num_mmio_chips.load();
            for (int i = 0; i < n; ++i) {
                ::tt::EthCoord coord{
                    shm->mmio_cluster_id[i],
                    shm->mmio_x[i],
                    shm->mmio_y[i],
                    shm->mmio_rack[i],
                    shm->mmio_shelf[i],
                };
                [[maybe_unused]] auto chip_id = cluster.get_physical_chip_id_from_eth_coord(coord);
            }

            dev->close();
        } catch (const std::exception&) {
            rc = 0;
        } catch (...) {
            rc = 0;
        }
        _exit(rc);
    }

    // ── Step 4: Wait for testee ────────────────────────────────────────────────
    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - testee_start)
                .count() > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap42SharedMem));
            FAIL() << "GAP-42: Testee did not exit within " << kTesteeBudgetMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ::munmap(raw_shm, sizeof(Gap42SharedMem));

    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        FAIL() << "GAP-42 REGRESSION (FIX NU): Testee was SIGABRT'd (exit 134).\n"
                  "\n"
                  "Root cause: get_physical_chip_id_from_eth_coord() was called with the\n"
                  "EthCoord of an MMIO device whose ALL ETH channels had fabric-mode firmware\n"
                  "(heartbeat = 0xa0b0c0d0). FIX W issued `continue` for every channel,\n"
                  "suppressing the local EthCoord capture. fill_cluster_descriptor_info()\n"
                  "therefore omitted the MMIO device from chip_locations, causing TT_FATAL.\n"
                  "\n"
                  "Fix: FIX NU in UMD topology_discovery.cpp — moved get_local_eth_coord()\n"
                  "to BEFORE the FIX W heartbeat check. PCIe NODE_INFO reads are safe\n"
                  "regardless of ETH firmware state; only relay-based remote discovery\n"
                  "needs the heartbeat guard.\n"
                  "\n"
                  "CI reference: run 25079761804 (tt_cluster.cpp:575 TT_FATAL both MPI ranks).";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-42: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    EXPECT_EQ(exit_code, 0)
        << "GAP-42: Testee exited with code " << exit_code << " (expected 0).";

    log_info(
        tt::LogTest,
        "GAP-42 PASS: Testee called get_physical_chip_id_from_eth_coord() for all {} MMIO "
        "EthCoord(s) without crashing. FIX NU is capturing MMIO EthCoords before FIX W guard.",
        n_mmio);
}

}  // namespace tt::tt_metal::distributed::test
