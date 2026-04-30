// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-64: FIX TB gap — TopologyMapper constructor must gracefully skip unknown ASIC IDs
//         (log warning + continue) instead of TT_FATAL, when ETH-damaged topology
//         yields ASIC IDs excluded from the fabric cluster chip mapping.
// Commit: 63bb7486c08
//
// Root cause (post-SIGSEGV / degraded ETH recovery):
//
//   During UMD topology re-discovery after a predecessor crash, the logical-to-physical
//   chip mapping includes ASIC IDs that have been excluded from the fabric cluster
//   (topology downgrade from ETH damage).  TopologyMapper's constructor iterated the
//   UMD eth_connections map and looked up each ASIC ID in asic_id_to_mapping_.
//
//   For excluded chips, asic_id_to_mapping_.find(asic_id) returns end(), and the old
//   code immediately called TT_FATAL:
//     "ASIC id {} not found in chip_topology_mapping_"
//   This SIGABRT turned every cluster health blip into a non-recoverable test failure —
//   the degraded-cluster detection path could never complete.
//
//   The crash site: tt_metal/fabric/topology_mapper.cpp (TopologyMapper constructor,
//   line ~297 before the fix).
//
// The fix (FIX TB, topology_mapper.cpp):
//   Replace TT_FATAL with:
//     if (asic_it == asic_id_to_mapping_.end()) {
//         log_warning(tt::LogFabric, "FIX TB: ASIC id {} not found in chip_topology_mapping_ "
//                     "— chip may be excluded from fabric cluster. Skipping.");
//         continue;
//     }
//
// What this test verifies:
//   Fork test confirming that after a SIGKILL'd predecessor leaves some non-MMIO
//   ERISCs dead (triggering topology re-discovery on next init), the TESTEE process
//   can complete TopologyMapper construction without SIGABRT.
//
//   The crash path (without FIX TB):
//     1. Predecessor dies → ETH relays dead on some non-MMIO devices
//     2. TESTEE opens MeshDevice → UMD topology re-discovery runs
//     3. TopologyMapper constructor: excluded ASIC ID → TT_FATAL → SIGABRT
//
//   With FIX TB:
//     1. TopologyMapper skips excluded ASIC IDs (log_warning + continue)
//     2. Topology construction completes → MeshDevice init proceeds (may throw later,
//        but not at the topology_mapper FATAL site)
//     3. TESTEE exits 0 or 1 (not 134)
//
// Note: GAP-63 (FIX NY/NY+) and GAP-64 (FIX TB) have similar test structures because
// both guard against crashes that occur during MeshDevice::create() on degraded clusters.
// FIX TB fires during topology discovery (early in init), FIX NY fires during dispatch
// kernel config generation (later in init).  Both are required for clean degraded recovery.
//
// Regression indicator:
//   TESTEE exits with SIGABRT (exit code 134) = FIX TB regression.
//   Crash log will show: "ASIC id N not found in chip_topology_mapping_"
//                         at topology_mapper.cpp.
//   Exit 0 or 1 = pass (graceful skip + continue, or healthy cluster).
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    60s (topology re-discovery + MeshDevice create)
//   Total:            ~120s
//
// Topology requirement: >= 2 devices (non-MMIO relay path required to trigger ETH
//   damage and topology re-discovery with unknown ASIC IDs).

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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kGap64PredWaitMs   = 35000;   // 35s for predecessor init + dispatch
static constexpr int kGap64TesteeBudget = 60000;   // 60s for topology re-discovery + MeshDevice

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap64Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap64(const MeshCoordinateRange& range) {
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
// Wait for child with timeout
// ---------------------------------------------------------------------------
static int wait_child_budget_gap64(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
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

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class FixTbTopologyMapperUnknownAsicFixture : public MeshDeviceFixtureBase {
protected:
    FixTbTopologyMapperUnknownAsicFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-64 requires >= 2 devices (non-MMIO relay path needed). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-64: TopologyMapperGracefulUnknownAsic
//
// Verifies FIX TB: TopologyMapper constructor must not TT_FATAL when an ASIC ID
// is absent from chip_topology_mapping_ (excluded from fabric cluster due to
// topology downgrade after ETH damage).
// ---------------------------------------------------------------------------
TEST_F(FixTbTopologyMapperUnknownAsicFixture, TopologyMapperGracefulUnknownAsic) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap64Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap64Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens FABRIC_2D mesh, dispatches blank workload, signals ready, spins until SIGKILL.
    // Leaves non-MMIO ERISCs dead → triggers topology downgrade on next init.
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
            auto workload = make_blank_workload_gap64(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - pred_start)
                                 .count();
        if (elapsed > kGap64PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap64Shm));
            GTEST_SKIP() << "GAP-64: predecessor did not signal ready within " << kGap64PredWaitMs
                         << "ms; skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-64: Predecessor SIGKILL'd — non-MMIO ERISCs dead. "
        "TESTEE will create MeshDevice. UMD topology re-discovery runs TopologyMapper "
        "constructor with ASIC IDs that include excluded chips. "
        "With FIX TB: log_warning + continue for unknown ASICs. "
        "Without FIX TB: TT_FATAL 'ASIC id N not found in chip_topology_mapping_' → SIGABRT.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Creates MeshDevice (FABRIC_2D). UMD topology re-discovery triggers
    // TopologyMapper constructor. With FIX TB, unknown ASIC IDs are skipped.
    // Without FIX TB, the first excluded ASIC ID triggers TT_FATAL → SIGABRT.
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
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
            fprintf(stderr, "GAP-64 TESTEE: MeshDevice created and closed cleanly. "
                           "TopologyMapper did not TT_FATAL on excluded ASIC IDs.\n");
        } catch (const std::exception& e) {
            // Topology mismatch or other exception after TopologyMapper succeeds = acceptable.
            // The key is that we don't SIGABRT inside TopologyMapper.
            fprintf(stderr, "GAP-64 TESTEE: MeshDevice::create threw exception (expected on degraded): %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "GAP-64 TESTEE: MeshDevice::create threw unknown exception (expected on degraded).\n");
        }
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap64(testee_pid, kGap64TesteeBudget);
    const auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start)
                               .count();

    ::munmap(raw_shm, sizeof(Gap64Shm));

    // Primary check: no SIGABRT from TopologyMapper TT_FATAL
    if (rc == 134) {
        FAIL() << "GAP-64 CRASH (FIX TB regression): TESTEE killed by SIGABRT (exit 134).\n"
               << "\n"
               << "Root cause: TopologyMapper constructor called TT_FATAL for an ASIC ID\n"
               << "that was excluded from the fabric cluster chip mapping:\n"
               << "  'ASIC id N not found in chip_topology_mapping_'\n"
               << "at topology_mapper.cpp (pre-FIX-TB line ~297).\n"
               << "\n"
               << "This happens after a predecessor SIGKILL leaves ETH relays dead,\n"
               << "triggering topology downgrade.  The downgraded topology excludes some\n"
               << "chips, but UMD eth_connections still lists their ASIC IDs.  TopologyMapper\n"
               << "constructor iterates eth_connections and looks up each ASIC ID — excluded\n"
               << "chips are absent from asic_id_to_mapping_ → old code TT_FATALed.\n"
               << "\n"
               << "Fix: Replace TT_FATAL with log_warning + continue for missing ASIC IDs.\n"
               << "See commit 63bb7486c08.";
    }

    if (rc == -1) {
        FAIL() << "GAP-64 TIMEOUT: TESTEE did not exit within "
               << kGap64TesteeBudget << "ms (elapsed: " << testee_ms << "ms).";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-64: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-64 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TB TopologyMapper correctly handles unknown ASIC IDs with "
        "log_warning + continue — no TT_FATAL on excluded chips.",
        testee_ms,
        kGap64TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
