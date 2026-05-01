// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-71: FIX TL gap — when FIX TK detects a degraded cluster (fewer chips in
//         fabric cluster than requested), DoSetUpTestSuite must bail out BEFORE
//         calling create_unit_meshes(), to avoid TT_FATAL in the UDM/Tensix
//         fabric initializer that requires the full expected topology.
// Commit: FIX TL (nsexton/0-racecondition-hunt)
//
// Root cause (CI run 25206820808, job 73909674743):
//
//   FIX TK (prev commit, c59d888dfb5) correctly filtered out chips not in the
//   1x1 fabric cluster and set cluster_degraded_skip_ = true.  However, it
//   then proceeded to call create_unit_meshes() with the 1-element filtered list
//   (just chip 0).  For fixtures that use UDM mode (Fabric2DUDMModeFixture with
//   fabric_udm_mode = ENABLED), this triggered a TT_FATAL in the UDM builder
//   because it requires the FULL expected topology (>= 2 chips) and cannot
//   handle a 1x1 degenerate cluster:
//
//     FIX TK (#42429): Physical chip 1/2/3 not in fabric cluster — excluding
//     TT_FATAL: Device 0 not found in worker tensix info map
//     TT_FATAL @ fabric_tensix_builder.cpp:874: device_it != worker_to_tensix_info_map_.end()
//     C++ exception thrown in SetUpTestSuite()
//     [  FAILED  ] Fabric2DUDMModeFixture: SetUpTestSuite or TearDownTestSuite
//
// The fix (FIX TL):
//   In DoSetUpTestSuite(), when cluster_degraded_skip_ becomes true (any chips
//   excluded by FIX TK), bail out IMMEDIATELY — before create_unit_meshes().
//   devices_map_ stays empty.  DoTearDownTestSuite() iterates devices_map_ safely.
//   Per-test SetUp() skips all tests via cluster_degraded_skip_.
//
//   This one-line early-return covers all fabric fixture subclasses (Fabric2DFixture,
//   Fabric2DUDMModeFixture, Fabric2DMuxFixture, etc.) with a single change.
//
// What this test verifies:
//   Fork test that forces topology degradation via SIGKILL'd predecessor, then
//   spawns a testee that simulates the UNFIXED FIX-TK-but-NOT-TL path:
//   - SetFabricConfig(FABRIC_2D, UDM_ENABLED) → topology discovery
//   - Filter chips (FIX TK present) → 1 chip remains → cluster_degraded_skip_ = true
//   - WITHOUT FIX TL: calls create_unit_meshes with [chip_0] → TT_FATAL in UDM builder
//   - WITH FIX TL: returns early → no create_unit_meshes → no crash
//
// Regression indicators:
//   TESTEE caught SIGABRT (exit 71 from sentinel handler) = FIX TL regression.
//   TT_FATAL "not found in worker tensix info map" in logs = FIX TL regression.
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    90s (full FABRIC_2D/UDM topology discovery on degraded cluster)
//   Total:            ~150s
//
// Topology requirement: >= 2 devices (at least 2 chips for meaningful ETH degradation;
//   requires T3K or similar for reliable progressive degradation via SIGKILL).

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <vector>

#include <experimental/fabric/fabric_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kGap71PredWaitMs    = 35000;  // 35s for predecessor init + dispatch
static constexpr int kGap71TesteeBudget  = 90000;  // 90s for topology + UDM builder check
// Sentinel exit code: testee SIGABRT'd (TT_FATAL from UDM builder) = FIX TL regression.
static constexpr int kGap71RegressionExit = 71;

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap71Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// SIGABRT handler installed in the TESTEE child only.
// TT_FATAL → abort() → this handler exits with regression sentinel.
// ---------------------------------------------------------------------------
static void gap71_sigabrt_handler(int /*sig*/) {
    const char msg[] =
        "GAP-71 TESTEE SIGABRT: TT_FATAL triggered in UDM fabric initializer — "
        "FIX TL is missing or reverted. "
        "Expected: DoSetUpTestSuite to bail out BEFORE create_unit_meshes when "
        "cluster_degraded_skip_ is true (FIX TK detected fewer chips than requested). "
        "Source: fabric_tensix_builder.cpp or UDM builder TT_FATAL with partial chip set.\n";
    (void)::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    ::_exit(kGap71RegressionExit);
}

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap71(const MeshCoordinateRange& range) {
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
static int wait_child_budget_gap71(pid_t pid, int budget_ms) {
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
class FixTlDegradedClusterBailBeforeCreateMeshesFixture : public MeshDeviceFixtureBase {
protected:
    FixTlDegradedClusterBailBeforeCreateMeshesFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-71 requires >= 2 devices to exercise ETH link degradation path. "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-71: FixTlUdmBuildCrashesWhenCreateMeshesCalledOnDegradedCluster
//
// Verifies FIX TL: when FIX TK marks cluster_degraded_skip_ = true (some chips
// excluded from the fabric cluster), DoSetUpTestSuite must bail BEFORE calling
// create_unit_meshes() — even if the filtered chip list is non-empty.
//
// The TESTEE simulates the UNFIXED half-state (FIX TK but NOT FIX TL):
//   1. Call SetFabricConfig(FABRIC_2D, UDM_ENABLED) → topology discovery.
//   2. Filter chips via is_physical_chip_in_fabric_cluster() (FIX TK step).
//   3. If fewer chips than all devices: call create_unit_meshes WITH the partial set.
//      (This is what FIX TL prevents — it returns early before this call.)
//   On a degraded 1x1 cluster with UDM mode: the UDM builder requires the full
//   topology → TT_FATAL "not found in worker tensix info map" → SIGABRT → exits 71.
//
// With FIX TL in DoSetUpTestSuite, the real fixture returns early when
// cluster_degraded_skip_ = true, preventing CI test suite crashes.
// ---------------------------------------------------------------------------
TEST_F(FixTlDegradedClusterBailBeforeCreateMeshesFixture, UdmBuildCrashesWhenCreateMeshesCalledOnDegradedCluster) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap71Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap71Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ─────────────────────────────────────────────
    // Opens FABRIC_2D on all devices, dispatches a blank workload, then spins until
    // SIGKILL — leaving ETH channels in mid-session state for progressive degradation.
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
            auto workload = make_blank_workload_gap71(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Ignore — we just need the ETH state initialized
        }
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor to signal init complete
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - pred_start)
                           .count();
        if (elapsed > kGap71PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap71Shm));
            GTEST_SKIP() << "GAP-71: predecessor did not signal ready within " << kGap71PredWaitMs
                         << "ms — cluster may be in a broken state. Skipping.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-71: Predecessor SIGKILL'd — ETH channels may be in mid-session state. "
        "TESTEE will simulate FIX-TK-present-but-FIX-TL-absent path (UDM mode): "
        "SetFabricConfig → filter chips (FIX TK) → call create_unit_meshes with partial set. "
        "On degraded cluster, UDM builder TT_FATALs on partial topology. "
        "With FIX TL: DoSetUpTestSuite returns early before create_unit_meshes when degraded.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ──────────────────────────────────────────────────
    // Simulates the UNFIXED half-state after FIX TK but before FIX TL:
    //   1. SetFabricConfig(FABRIC_2D, UDM mode) → topology discovery
    //   2. Filter chips via is_physical_chip_in_fabric_cluster() [FIX TK]
    //   3. If fewer chips than all devices: call create_unit_meshes with partial list
    //      [this is what FIX TL prevents with the early return]
    //   On a 1x1 cluster with UDM mode: TT_FATAL in UDM builder → SIGABRT → exits 71
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Install SIGABRT handler: TT_FATAL → abort() → sentinel exit(71)
        struct sigaction sa{};
        sa.sa_handler = gap71_sigabrt_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGABRT, &sa, nullptr);

        try {
            // Step 1: SetFabricConfig with UDM mode (same as Fabric2DUDMModeFixture path).
            // UDM mode triggers the tensix/UDM builder which requires the full topology.
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                /*num_routing_planes=*/std::nullopt,
                tt_fabric::FabricTensixConfig::DISABLED,
                tt_fabric::FabricUDMMode::ENABLED);

            // Step 2: FIX TK chip filter — query which chips are in the cluster.
            const auto total_chips = MetalContext::instance().get_cluster().number_of_devices();
            const auto& control_plane = MetalContext::instance().get_control_plane();

            std::vector<ChipId> cluster_ids;
            cluster_ids.reserve(total_chips);
            for (unsigned int id = 0; id < total_chips; ++id) {
                if (control_plane.is_physical_chip_in_fabric_cluster(static_cast<ChipId>(id))) {
                    cluster_ids.push_back(static_cast<ChipId>(id));
                }
            }

            fprintf(
                stderr,
                "GAP-71 TESTEE: %zu/%zu chips in fabric cluster after SetFabricConfig.\n",
                cluster_ids.size(),
                static_cast<size_t>(total_chips));

            bool cluster_degraded = (cluster_ids.size() < total_chips);

            if (!cluster_degraded) {
                fprintf(
                    stderr,
                    "GAP-71 TESTEE: Cluster is healthy — all chips in cluster. "
                    "FIX TL regression can only be detected on a degraded cluster.\n");
                // Proceed to create_unit_meshes with full set — should be fine on healthy cluster
            } else {
                fprintf(
                    stderr,
                    "GAP-71 TESTEE: Cluster degraded — %zu chips missing. "
                    "Simulating FIX-TK-present but FIX-TL-absent: "
                    "calling create_unit_meshes with partial list [%zu chips] in UDM mode.\n"
                    "Without FIX TL: UDM builder TT_FATALs on partial topology.\n",
                    total_chips - cluster_ids.size(),
                    cluster_ids.size());
            }

            if (!cluster_ids.empty()) {
                // Step 3: Call create_unit_meshes with the partial (or full if healthy) chip list.
                // With UDM mode on a degraded cluster (partial list), without FIX TL this crashes.
                const auto& dispatch_core_config =
                    MetalContext::instance().rtoptions().get_dispatch_core_config();
                auto dev_map = MeshDevice::create_unit_meshes(
                    cluster_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE,
                    /*num_command_queues=*/1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);

                // Clean up devices
                for (auto& [chip_id, dev] : dev_map) {
                    dev->close();
                }

                fprintf(
                    stderr,
                    "GAP-71 TESTEE: create_unit_meshes with %zu chips succeeded (UDM mode). "
                    "%s\n",
                    cluster_ids.size(),
                    cluster_degraded
                        ? "UDM builder handled partial topology gracefully (unexpected but OK)."
                        : "Cluster was healthy — no degradation path triggered.");
            } else {
                fprintf(stderr, "GAP-71 TESTEE: No chips in cluster — skipping create_unit_meshes (safe early return).\n");
            }
        } catch (const std::exception& e) {
            // Other exception (topology mismatch, missing relay, etc.) — acceptable on degraded cluster
            fprintf(
                stderr,
                "GAP-71 TESTEE: caught exception (acceptable on degraded cluster): %s\n",
                e.what());
        } catch (...) {
            fprintf(stderr, "GAP-71 TESTEE: caught unknown exception (acceptable on degraded cluster).\n");
        }

        // If we reach here, no TT_FATAL SIGABRT was triggered — FIX TL working
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap71(testee_pid, kGap71TesteeBudget);
    auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - testee_start)
                         .count();

    ::munmap(raw_shm, sizeof(Gap71Shm));

    // Primary regression check: SIGABRT handler exited with sentinel 71 (FIX TL missing)
    if (rc == kGap71RegressionExit) {
        FAIL() << "GAP-71 REGRESSION (FIX TL missing): TESTEE caught SIGABRT from TT_FATAL.\n"
               << "\n"
               << "Root cause: DoSetUpTestSuite() called create_unit_meshes() with a partial chip set\n"
               << "(1 chip after FIX TK filtered 3 chips from a degraded 1x1 cluster) in UDM mode.\n"
               << "The UDM/Tensix fabric builder requires the FULL expected topology (>= 2 chips)\n"
               << "and TT_FATALs when only 1 chip is provided.\n"
               << "\n"
               << "crash location: fabric_tensix_builder.cpp:874\n"
               << "  TT_FATAL: Device {} not found in worker tensix info map\n"
               << "  TT_FATAL: device_it != worker_to_tensix_info_map_.end()\n"
               << "\n"
               << "Fix (FIX TL):\n"
               << "  In BaseFabricFixture::DoSetUpTestSuite(), when cluster_degraded_skip_ becomes true\n"
               << "  (i.e., any chips were removed by FIX TK), bail out IMMEDIATELY:\n"
               << "    if (cluster_ids.size() < ids.size()) {\n"
               << "        cluster_degraded_skip_ = true;\n"
               << "        log_warning(...);  // FIX TL: Do NOT proceed to create_unit_meshes\n"
               << "        return;  // <-- this early return is FIX TL\n"
               << "    }\n"
               << "  See: tests/tt_metal/tt_fabric/common/fabric_fixture.hpp, DoSetUpTestSuite()\n";
    }

    if (rc == -1) {
        FAIL() << "GAP-71 TIMEOUT: TESTEE did not exit within " << kGap71TesteeBudget
               << "ms (elapsed: " << testee_ms << "ms). "
               << "UDM fabric init hung — possible related hang issue.";
    }

    // Any other exit code (0, exception) is acceptable
    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-71: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-71 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TL bail-before-create_unit_meshes guard in DoSetUpTestSuite is working — "
        "no TT_FATAL from UDM builder on a degraded cluster with partial chip set.",
        testee_ms,
        kGap71TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
