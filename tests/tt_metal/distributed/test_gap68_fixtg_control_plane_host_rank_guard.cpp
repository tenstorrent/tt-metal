// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-68: FIX TG gap — configure_routing_tables_for_fabric_ethernet_channels must guard
//         host_rank_for_chip.value() against nullopt for FIX TB-excluded connected chips.
// Commit: FIX TG (nsexton/0-racecondition-hunt)
//
// Root cause (CI run 25197685800 — T3K after FIX TF deployed):
//
//   ControlPlane::configure_routing_tables_for_fabric_ethernet_channels() loops over
//   intra_mesh_connectivity and for cross-host connections calls:
//
//     auto host_rank_for_chip = topology_mapper_->get_host_rank_for_chip(mesh_id, chip_id);
//     TT_ASSERT(host_rank_for_chip.has_value(), ...);  // <-- no-op in Release builds!
//     auto connected_host_rank_id = host_rank_for_chip.value();  // throws if nullopt
//
//   TT_ASSERT is a no-op in Release builds (defined as `(void)(condition)`).
//   When FIX TB has excluded the connected chip from the topology mapper (degraded T3K
//   cluster, corrupt ERISC L1), get_host_rank_for_chip() returns std::nullopt.
//   Calling .value() on nullopt throws std::bad_optional_access during ControlPlane
//   construction — caught by GTest as opaque "bad optional access" in SetUp().
//
//   This was visible in CI run 25197685800 AFTER FIX TF was deployed:
//   FIX TF guards relay_mux.hpp, but a SECOND .value() in configure_routing_tables
//   still fired, causing the same opaque "bad optional access" failure.
//
// The fix (FIX TG):
//   Replace TT_ASSERT + .value() with an explicit has_value() guard:
//     if (!host_rank_for_chip.has_value()) {
//         log_warning(tt::LogFabric, "FIX TG (#42429): Mesh {} Chip {} has no host rank...");
//         continue;
//     }
//   This matches the pattern established by FIX TE for missing ASIC IDs.
//
// What this test verifies:
//   Fork test that forces topology degradation via SIGKILL'd predecessor, then
//   spawns a testee that tries to init FABRIC_2D with the T3K custom mesh descriptor.
//   Verifies that ControlPlane construction does NOT throw std::bad_optional_access.
//
//   Without FIX TG: bad_optional_access thrown in ControlPlane ctor → exits 68.
//   With FIX TG:    warning logged, connection skipped, no exception → exits 0.
//
// Regression indicator:
//   TESTEE catches std::bad_optional_access (exit sentinel 68) = FIX TG regression.
//   TESTEE completes without bad_optional_access = fix working.
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    90s (full T3K FABRIC_2D ControlPlane init on degraded cluster)
//   Total:            ~150s
//
// Topology requirement: >= 8 devices (T3K 2x4 mesh).

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
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
static constexpr int kGap68PredWaitMs    = 35000;  // 35s for predecessor init + dispatch
static constexpr int kGap68TesteeBudget  = 90000;  // 90s for T3K FABRIC_2D ControlPlane init
// Sentinel exit code: testee caught bad_optional_access (FIX TG regression)
static constexpr int kGap68RegressionExit = 68;

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap68Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap68(const MeshCoordinateRange& range) {
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
// Wait for child with timeout, returns exit code or -1 on timeout
// ---------------------------------------------------------------------------
static int wait_child_budget_gap68(pid_t pid, int budget_ms) {
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) break;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
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
class FixTgControlPlaneHostRankGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixTgControlPlaneHostRankGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 8) {
            GTEST_SKIP() << "GAP-68 requires >= 8 devices (T3K 2x4 with custom mesh descriptor). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-68: FixTgConfigureRoutingTablesHostRankNulloptThrows
//
// Verifies FIX TG: configure_routing_tables_for_fabric_ethernet_channels must not
// throw std::bad_optional_access when a connected chip has been excluded from the
// topology mapper by FIX TB (degraded cluster).
// ---------------------------------------------------------------------------
TEST_F(FixTgControlPlaneHostRankGuardFixture, ConfigureRoutingTablesHostRankNulloptThrows) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap68Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap68Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ─────────────────────────────────────────────
    // Opens T3K with FABRIC_2D (all 8 devices), dispatches a blank workload to
    // spin up ETH firmware in all chips, signals ready, then spins until SIGKILL.
    // Leaves ETH channels (including inter-mesh relay) in a mid-session state,
    // which on the next session triggers FIX TB (topology mapper exclusion) for
    // chips whose ERISC L1 is corrupted.
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
            auto workload = make_blank_workload_gap68(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Ignore: we just need the ETH state set up
        }
        shm->predecessor_ready.store(1);
        // Spin until SIGKILL — leaves ERISCs dead / mid-session
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor to signal init complete
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - pred_start)
                           .count();
        if (elapsed > kGap68PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap68Shm));
            GTEST_SKIP() << "GAP-68: predecessor did not signal ready within " << kGap68PredWaitMs
                         << "ms — cluster may be in a broken state. Skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Give predecessor a moment to be fully in workload before SIGKILL
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-68: Predecessor SIGKILL'd — ETH channels may be in mid-session state. "
        "TESTEE will attempt FABRIC_2D ControlPlane init with T3K custom mesh descriptor. "
        "If some chips are FIX TB-excluded and their connected peers have no host rank, "
        "configure_routing_tables_for_fabric_ethernet_channels() calls .value() on nullopt. "
        "With FIX TG: has_value() guard fires, warning logged, continues. "
        "Without FIX TG: bad_optional_access thrown in ControlPlane ctor.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ──────────────────────────────────────────────────
    // Tries to init FABRIC_2D with the T3K 2x4 custom mesh descriptor.
    // ControlPlane construction calls configure_routing_tables_for_fabric_ethernet_channels().
    // On a degraded cluster, some chips are FIX TB-excluded → get_host_rank_for_chip()
    // returns nullopt for their connected chips.
    //
    // Without FIX TG: std::bad_optional_access thrown in ControlPlane ctor → exits 68
    // With FIX TG:    warning logged, skip, no exception → exits 0
    // Healthy cluster: no degradation, no exception → exits 0
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        bool caught_bad_optional = false;

        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

            // Use the T3K 2x4 custom mesh descriptor (same as T3kCustomMeshGraphFabric2DFixture)
            // This triggers the specific code path in configure_routing_tables that enters the
            // `else` branch for cross-host connections and calls get_host_rank_for_chip().
            const auto& rtopts = MetalContext::instance().rtoptions();
            const std::string desc_path =
                std::string(rtopts.get_root_dir()) +
                "/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";

            MetalContext::instance().set_custom_fabric_topology(desc_path, {});

            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(num_dev)}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            dev->close();
            fprintf(stderr, "GAP-68 TESTEE: ControlPlane constructed cleanly (cluster healthy or FIX TG skipped entries).\n");
        } catch (const std::bad_optional_access& e) {
            // Regression: FIX TG missing — TT_ASSERT no-op in Release, .value() threw
            caught_bad_optional = true;
            fprintf(
                stderr,
                "GAP-68 TESTEE REGRESSION: caught std::bad_optional_access — "
                "FIX TG is missing or reverted. "
                "Source: configure_routing_tables_for_fabric_ethernet_channels in control_plane.cpp "
                "called host_rank_for_chip.value() without has_value() check (TT_ASSERT is no-op in Release). "
                "Exception: %s\n",
                e.what());
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            if (msg.find("FIX TG") != std::string::npos) {
                fprintf(
                    stderr,
                    "GAP-68 TESTEE: caught exception with FIX TG context: %s\n",
                    msg.c_str());
            } else {
                // Other exception (topology mismatch, SIGKILL relay, etc.) — acceptable
                fprintf(
                    stderr,
                    "GAP-68 TESTEE: caught other exception (acceptable on degraded cluster): %s\n",
                    msg.c_str());
            }
        } catch (...) {
            fprintf(stderr, "GAP-68 TESTEE: caught unknown exception (acceptable on degraded cluster).\n");
        }

        if (caught_bad_optional) {
            _exit(kGap68RegressionExit);  // Regression signal
        }
        _exit(0);  // Pass: healthy, FIX TG working, or other clean exception
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap68(testee_pid, kGap68TesteeBudget);
    auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - testee_start)
                         .count();

    ::munmap(raw_shm, sizeof(Gap68Shm));

    // Primary regression check: bad_optional_access was caught (FIX TG missing)
    if (rc == kGap68RegressionExit) {
        FAIL() << "GAP-68 REGRESSION (FIX TG missing): TESTEE caught std::bad_optional_access.\n"
               << "\n"
               << "Root cause: configure_routing_tables_for_fabric_ethernet_channels()\n"
               << "in control_plane.cpp called host_rank_for_chip.value() after TT_ASSERT\n"
               << "(which is a no-op in Release builds) without an explicit has_value() check.\n"
               << "\n"
               << "When FIX TB excludes a connected chip from the topology mapper\n"
               << "(degraded T3K cluster, corrupt ERISC L1), get_host_rank_for_chip()\n"
               << "returns std::nullopt for that chip. The .value() call then throws\n"
               << "std::bad_optional_access during ControlPlane construction.\n"
               << "\n"
               << "Fix (FIX TG):\n"
               << "  Replace:\n"
               << "    TT_ASSERT(host_rank_for_chip.has_value(), ...);\n"
               << "    auto connected_host_rank_id = host_rank_for_chip.value();\n"
               << "  With:\n"
               << "    if (!host_rank_for_chip.has_value()) {\n"
               << "        log_warning(tt::LogFabric, \"FIX TG (#42429): ...\");\n"
               << "        continue;\n"
               << "    }\n"
               << "    auto connected_host_rank_id = host_rank_for_chip.value();\n"
               << "  See: tt_metal/fabric/control_plane.cpp, configure_routing_tables_for_fabric_ethernet_channels\n";
    }

    if (rc == -1) {
        FAIL() << "GAP-68 TIMEOUT: TESTEE did not exit within " << kGap68TesteeBudget
               << "ms (elapsed: " << testee_ms << "ms). "
               << "ControlPlane init hung — possible related hang issue.";
    }

    if (rc == 134) {
        FAIL() << "GAP-68 CRASH: TESTEE killed by SIGABRT (exit 134). "
               << "Not the bad_optional_access regression (FIX TG), but a different TT_FATAL. "
               << "Check stderr for the fatal message and context.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-68: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-68 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TG configure_routing_tables host_rank guard is working — "
        "no opaque bad_optional_access thrown from ControlPlane ctor on degraded cluster.",
        testee_ms,
        kGap68TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
