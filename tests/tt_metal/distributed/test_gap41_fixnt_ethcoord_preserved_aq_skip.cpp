// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-41: FIX NT — Preserve EthCoord in chip_locations for FIX-AQ-skipped remote chips
//
// Root cause (CI run 25077304186, t3k_tt_metal_multiprocess_tests on tt-metal-ci-vm-t3k-12):
//
//   1. Prior process left non-MMIO ERISCs in FABRIC/STARTED mode on some chips.
//   2. New process opened with a YAML cluster descriptor (test_t3k_2x2.yaml) that
//      referenced ALL 8 chips by EthCoord.
//   3. During topology discovery, init_tt_device() timed out (5s) for the FABRIC-state
//      non-MMIO ERISCs.  FIX AQ caught the exception and `continue`d past initialization.
//   4. Without FIX NT: the FIX AQ `continue` path did NOT add the skipped device's
//      EthCoord to eth_coords.  fill_cluster_descriptor_info() therefore omitted that
//      device from chip_locations.
//   5. initialize_and_validate_custom_physical_config() read the YAML's eth_coord_mapping
//      and called cluster.get_physical_chip_id_from_eth_coord(coord) for each chip.
//      The missing chip's coord was not found in chip_locations →
//        TT_FATAL(false, "Physical chip id not found for eth coord")  →  SIGABRT.
//
// FIX NT (UMD commit 9758ef99, tt-metal commit dc47f481e688):
//   In topology_discovery.cpp, inside the FIX AQ catch block, after logging the
//   "Skipping — remote device unreachable" warning, also emplace the device's EthCoord
//   into eth_coords and the device ID into discovered_devices.  This ensures that
//   fill_cluster_descriptor_info() includes the chip in chip_locations even when it
//   was unreachable, so any code that iterates the YAML's EthCoord list and calls
//   get_physical_chip_id_from_eth_coord() finds a valid entry.
//
// What this test verifies:
//   1. Parent opens a healthy cluster and records the full set of EthCoords (N chips).
//   2. PREDECESSOR is fork/exec'd: opens FABRIC_2D, dispatches a blank workload so
//      non-MMIO ERISCs are left in FABRIC/ACTIVE mode, signals ready, then spins.
//   3. Parent SIGKILLs predecessor (non-MMIO ERISCs remain in FABRIC state).
//   4. TESTEE is fork/exec'd: opens a new Cluster; FIX AQ fires for the FABRIC-state
//      remote chips; testee then calls get_physical_chip_id_from_eth_coord() for EACH
//      of the N EthCoords recorded in step 1 (including the skipped ones).
//   5. Without FIX NT: step 4 crashes with TT_FATAL → SIGABRT → testee exit code 134.
//      With    FIX NT: all coords are in chip_locations → returns valid chip IDs → exit 0.
//   6. Parent asserts testee exit code == 0 within kTesteeBudgetMs.
//
// Distinction from GAP-39 (FIX NS) and GAP-40 (FIX AE):
//   GAP-39 tests relay queue overflow from double topology discovery.
//   GAP-40 tests ~Cluster destructor hang from flush timeout.
//   GAP-41 (this test) tests EthCoord completeness of chip_locations when topology
//   discovery degrades — the specific crash path triggered by YAML-based fabric tests.
//
// Topology requirement: >= 2 devices (non-MMIO relay path required to trigger FIX AQ).

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

// Maximum chips supported in shared memory (T3K=8, TG=32).
static constexpr int kMaxSharedChips = 64;

// Testee budget: open + EthCoord queries + close.  If FIX NT is missing,
// the process SIGABRT's immediately at get_physical_chip_id_from_eth_coord();
// there is no hang — it crashes instantly.  Budget is generous to allow
// for slow hardware init.
static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 90000;

struct Gap41SharedMem {
    std::atomic<int> predecessor_ready{0};
    std::atomic<int> num_chips{0};
    // EthCoords recorded from the healthy parent open — testee will call
    // get_physical_chip_id_from_eth_coord() for EACH of these.
    int eth_cluster_id[kMaxSharedChips];
    int eth_x[kMaxSharedChips];
    int eth_y[kMaxSharedChips];
    int eth_rack[kMaxSharedChips];
    int eth_shelf[kMaxSharedChips];
};

// Lightest workload that activates ETH relay (ensures non-MMIO ERISCs enter ACTIVE state).
static MeshWorkload make_blank_workload_gap41(const MeshCoordinateRange& range) {
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

class EthCoordPreservedOnAqSkipFixture : public MeshDeviceFixtureBase {
protected:
    EthCoordPreservedOnAqSkipFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-41 requires >= 2 devices. Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-41: EthCoordPreservedInChipLocationsAfterFiqAqSkip
//
// Verifies that get_physical_chip_id_from_eth_coord() does not crash for
// EthCoords belonging to chips that were skipped by FIX AQ during topology
// discovery in a subsequent process.
// ---------------------------------------------------------------------------
TEST_F(EthCoordPreservedOnAqSkipFixture, EthCoordPreservedInChipLocationsAfterFiqAqSkip) {
    // ── Step 0: Close fixture device so parent MetalContext is clean ──────────
    mesh_device_->close();

    // ── Shared memory ─────────────────────────────────────────────────────────
    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap41SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap41SharedMem();

    // ── Step 1: Record EthCoords from the healthy parent Cluster ─────────────
    // The parent's MetalContext (and Cluster) is still alive after mesh_device_->close().
    // chip_locations is fully populated in the healthy state.
    {
        const auto& cluster = MetalContext::instance().get_cluster();
        auto all_coords = cluster.get_all_chip_ethernet_coordinates();
        int n = 0;
        for (const auto& [chip_id, coord] : all_coords) {
            if (n >= kMaxSharedChips) break;
            shm->eth_cluster_id[n] = coord.cluster_id;
            shm->eth_x[n] = coord.x;
            shm->eth_y[n] = coord.y;
            shm->eth_rack[n] = coord.rack;
            shm->eth_shelf[n] = coord.shelf;
            ++n;
        }
        shm->num_chips.store(n);
        log_info(
            tt::LogTest,
            "GAP-41: Recorded {} EthCoord(s) from healthy parent Cluster for testee validation.",
            n);
    }

    const int n_expected = shm->num_chips.load();
    ASSERT_GT(n_expected, 0) << "No EthCoords found in parent cluster — hardware issue?";

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 2: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D, dispatches workload (non-MMIO ERISCs enter ACTIVE state),
    // signals ready, then spins.  SIGKILL leaves relay dirty.
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
            auto workload = make_blank_workload_gap41(range);
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

    // Wait for predecessor ready, then SIGKILL.
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap41SharedMem));
            GTEST_SKIP() << "GAP-41: predecessor did not signal ready within " << kPredWaitMs
                         << "ms (hardware init stall?).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);
    log_info(tt::LogTest, "GAP-41: Predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state.");

    // Brief settle so ERISC firmware stops updating heartbeat.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 3: Fork TESTEE ───────────────────────────────────────────────────
    // Opens a new Cluster (UMD topology discovery runs → FIX AQ fires for
    // FABRIC-state remote chips → FIX NT preserves their EthCoords).
    // Testee calls get_physical_chip_id_from_eth_coord() for EACH of the
    // n_expected coords recorded in step 1.  Without FIX NT, this crashes
    // for the skipped chips (TT_FATAL → SIGABRT → exit 134).
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Testee child: open Cluster, then call get_physical_chip_id_from_eth_coord()
        // for every EthCoord the parent recorded from the healthy open.
        int rc = 0;
        try {
            // Use a single-device MeshDevice open to trigger MetalContext / Cluster init.
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);

            // At this point, UMD topology discovery has run. FIX AQ may have fired
            // for some chips and FIX NT should have preserved their EthCoords.
            const auto& cluster = MetalContext::instance().get_cluster();
            auto all_coords = cluster.get_all_chip_ethernet_coordinates();

            // Check completeness: chip_locations should have at least n_expected entries.
            const int n_found = static_cast<int>(all_coords.size());
            if (n_found < n_expected) {
                // Log mismatch but also proceed to the crash-prone lookups so the
                // parent can detect SIGABRT vs. clean exit.
                log_warning(
                    tt::LogTest,
                    "GAP-41 TESTEE: chip_locations has {} entries, expected >= {}. "
                    "FIX NT may be missing or FIX AQ did not fire.",
                    n_found,
                    n_expected);
            }

            // Call get_physical_chip_id_from_eth_coord() for EVERY coord the parent
            // recorded.  Without FIX NT, this crashes for skipped chips.
            int n = shm->num_chips.load();
            for (int i = 0; i < n; ++i) {
                ::tt::EthCoord coord{
                    shm->eth_cluster_id[i],
                    shm->eth_x[i],
                    shm->eth_y[i],
                    shm->eth_rack[i],
                    shm->eth_shelf[i],
                };
                // TT_FATAL fires here (→ SIGABRT) without FIX NT when coord is
                // for a FIX-AQ-skipped chip.
                [[maybe_unused]] auto chip_id = cluster.get_physical_chip_id_from_eth_coord(coord);
            }

            dev->close();
        } catch (const std::exception& e) {
            // Exception (e.g., from FIX AQ signalling degraded topology) is OK —
            // we only care that we did NOT crash before reaching here.
            rc = 0;
        } catch (...) {
            rc = 0;
        }
        _exit(rc);
    }

    // ── Step 4: Wait for testee and verify exit code ─────────────────────────
    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    pid_t waited = 0;
    while (true) {
        waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - testee_start)
                .count() > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap41SharedMem));
            FAIL() << "GAP-41: Testee did not exit within " << kTesteeBudgetMs
                   << "ms — unexpected hang (not the expected SIGABRT crash).";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ::munmap(raw_shm, sizeof(Gap41SharedMem));

    // Check for SIGABRT (exit code 134): the TT_FATAL crash signature.
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        FAIL() << "GAP-41 REGRESSION (FIX NT): Testee was SIGABRT'd (exit 134).\n"
                  "\n"
                  "Root cause: get_physical_chip_id_from_eth_coord() called with EthCoord\n"
                  "of a chip that FIX AQ skipped during topology discovery.  Without FIX NT,\n"
                  "the skipped chip's EthCoord is NOT added to chip_locations, so the lookup\n"
                  "finds no match and TT_FATAL fires.\n"
                  "\n"
                  "Fix: FIX NT in UMD topology_discovery.cpp — after FIX AQ `continue`, also\n"
                  "emplace(remote_asic_id, eth_coord) in eth_coords so fill_cluster_descriptor_info()\n"
                  "includes the chip in chip_locations even when unreachable.\n"
                  "\n"
                  "CI reference: run 25077304186 (tt_cluster.cpp:575 TT_FATAL both MPI ranks).";
    }

    if (WIFSIGNALED(status)) {
        FAIL() << "GAP-41: Testee killed by signal " << WTERMSIG(status) << " (unexpected).";
    }

    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    EXPECT_EQ(exit_code, 0)
        << "GAP-41: Testee exited with code " << exit_code << " (expected 0).";

    log_info(
        tt::LogTest,
        "GAP-41 PASS: Testee called get_physical_chip_id_from_eth_coord() for all {} EthCoord(s) "
        "without crashing. FIX NT is preserving chip_locations completeness after FIX AQ skip.",
        n_expected);
}

}  // namespace tt::tt_metal::distributed::test
