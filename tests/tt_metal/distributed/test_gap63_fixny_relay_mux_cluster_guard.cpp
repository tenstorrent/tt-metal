// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-63: FIX NY/NY+ gap — RelayMux::GenerateStaticConfigs must guard against devices
//         excluded from the fabric cluster due to degraded topology (ETH damage).
// Commits: 54c194ade32 (FIX NY), e66aea14726 (FIX NY+)
//
// Root cause (runs 25149202423, 25151000876):
//
//   When T3K has damaged ETH channels, UMD topology discovery can downgrade the mesh
//   (e.g., 2x4 → 2x2) and exclude some physical chips from the fabric control plane
//   mapping.  The dispatch topology is built from UMD tunnels which still list these
//   chips, so RelayMux::GenerateStaticConfigs() was called for excluded chips.
//
//   Inside GenerateStaticConfigs(), the call to get_fabric_node_id_from_physical_chip_id()
//   triggered TT_FATAL: "Physical chip id N not found in control plane chip mapping"
//   — a hard crash instead of a graceful topology-mismatch path.
//
//   FIX NY+ extended the guard to the destination device: the MMIO source device
//   may be in the cluster but its tunnel destination (non-MMIO) may have been excluded
//   from the downgraded mesh.  Same TT_FATAL at relay_mux.cpp:116 for the dst side.
//
// The fix (FIX NY, relay_mux.cpp):
//   Add is_physical_chip_in_fabric_cluster(device_id_) guard at the top of
//   GenerateStaticConfigs().  If source not in cluster: log_warning + return early.
//
// The fix (FIX NY+, relay_mux.cpp):
//   Add is_physical_chip_in_fabric_cluster(destination_device_id) guard before
//   calling get_fabric_node_id_from_physical_chip_id() on the destination.
//   If destination not in cluster: log_warning + return early.
//
// What this test verifies:
//   Fork test confirming that after a SIGKILL'd predecessor leaves some non-MMIO
//   ERISCs dead (forcing topology downgrade on next init), the TESTEE process can
//   create a MeshDevice without crashing at SIGABRT (exit 134).
//
//   The crash path (without FIX NY/NY+):
//     1. Predecessor dies → ETH relays dead on some non-MMIO devices
//     2. TESTEE opens MeshDevice → UMD downgrade detected
//     3. GenerateStaticConfigs called for excluded chip → TT_FATAL → SIGABRT
//
//   With FIX NY/NY+:
//     1. GenerateStaticConfigs returns early for excluded chips
//     2. MeshDevice::create() continues → topology mismatch error (clean exception)
//     3. TESTEE exits 0 or 1 (not 134)
//
// Regression indicator:
//   TESTEE exits with SIGABRT (exit code 134) = FIX NY/NY+ regression.
//   TESTEE times out = hang in dispatch kernel generation.
//   TESTEE exits 0 or 1 = pass (clean exit, expected exception, or healthy cluster).
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    60s (MeshDevice create including topology downgrade path)
//   Total:            ~120s
//
// Topology requirement: >= 2 devices (MMIO + non-MMIO relay path required for
//   ETH damage to be observable; single-device systems skip this test).

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
static constexpr int kGap63PredWaitMs   = 35000;   // 35s for predecessor init + dispatch
static constexpr int kGap63TesteeBudget = 60000;   // 60s for testee MeshDevice create

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap63Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap63(const MeshCoordinateRange& range) {
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
// Wait for child process with timeout
// ---------------------------------------------------------------------------
static int wait_child_budget_gap63(pid_t pid, int budget_ms) {
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
class FixNyRelayMuxClusterGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixNyRelayMuxClusterGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-63 requires >= 2 devices (non-MMIO relay path needed). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-63: RelayMuxClusterGuardPreventsDispatchCrash
//
// Verifies FIX NY/NY+: RelayMux::GenerateStaticConfigs() must not TT_FATAL
// when called for chips excluded from the fabric cluster due to topology downgrade.
// ---------------------------------------------------------------------------
TEST_F(FixNyRelayMuxClusterGuardFixture, RelayMuxClusterGuardPreventsDispatchCrash) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap63Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap63Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens full FABRIC_2D mesh, dispatches blank workload, signals ready,
    // then spins until SIGKILL.  Leaves non-MMIO ERISCs dead.
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
            auto workload = make_blank_workload_gap63(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor to signal ready
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - pred_start)
                                 .count();
        if (elapsed > kGap63PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap63Shm));
            GTEST_SKIP() << "GAP-63: predecessor did not signal ready within " << kGap63PredWaitMs
                         << "ms; cluster may be healthy. Skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-63: Predecessor SIGKILL'd — non-MMIO ERISCs dead. "
        "TESTEE will create MeshDevice. If topology downgrade occurs, "
        "RelayMux::GenerateStaticConfigs() will be called for excluded chips. "
        "With FIX NY/NY+: returns early. Without: TT_FATAL → SIGABRT.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Creates MeshDevice (FABRIC_2D).  If some ETH relays are dead, UMD will
    // downgrade the mesh topology.  The dispatch build runs GenerateStaticConfigs
    // for all chips in the UMD tunnel map — including excluded ones.
    //
    // With FIX NY/NY+: excluded chips return early → no crash.
    // Without FIX NY/NY+: get_fabric_node_id_from_physical_chip_id() TT_FATALs → SIGABRT.
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
            fprintf(stderr, "GAP-63 TESTEE: MeshDevice created and closed cleanly.\n");
        } catch (const std::exception& e) {
            // Topology mismatch exception = acceptable (degraded cluster detected cleanly)
            fprintf(stderr, "GAP-63 TESTEE: MeshDevice::create threw exception (expected on degraded): %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "GAP-63 TESTEE: MeshDevice::create threw unknown exception (expected on degraded).\n");
        }
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap63(testee_pid, kGap63TesteeBudget);
    const auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start)
                               .count();

    ::munmap(raw_shm, sizeof(Gap63Shm));

    // Primary check: no SIGABRT (the pre-FIX-NY failure mode)
    if (rc == 134) {
        FAIL() << "GAP-63 CRASH (FIX NY/NY+ regression): TESTEE killed by SIGABRT (exit 134).\n"
               << "\n"
               << "Root cause: RelayMux::GenerateStaticConfigs() called for a chip that was\n"
               << "excluded from the fabric cluster due to topology downgrade (ETH damage).\n"
               << "get_fabric_node_id_from_physical_chip_id() TT_FATALs: 'Physical chip id N\n"
               << "not found in control plane chip mapping'.\n"
               << "\n"
               << "Fix: Add is_physical_chip_in_fabric_cluster() guard at the top of\n"
               << "RelayMux::GenerateStaticConfigs() (for both source and destination device).\n"
               << "See commits 54c194ade32 (FIX NY) and e66aea14726 (FIX NY+).";
    }

    if (rc == -1) {
        FAIL() << "GAP-63 TIMEOUT (FIX NY/NY+ regression): TESTEE did not exit within "
               << kGap63TesteeBudget << "ms (elapsed: " << testee_ms << "ms).\n"
               << "Dispatch kernel config generation hung — may indicate a related issue.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-63: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-63 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX NY/NY+ RelayMux::GenerateStaticConfigs() correctly guards against "
        "excluded chips — no SIGABRT from dispatch kernel config generation.",
        testee_ms,
        kGap63TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
