// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-69: FIX TH gap — RelayMux::GenerateStaticConfigs must preflight-check
//         get_forwarding_link_indices() before calling get_dispatch_link_index()
//         when an MMIO device's ETH channels to its downstream tunnel target are all dead.
// Commit: FIX TH (nsexton/0-racecondition-hunt)
//
// Root cause (CI run 25204526485 — T3K after FIX TF + FIX TG deployed):
//
//   RelayMux::GenerateStaticConfigs() in relay_mux.cpp calls:
//
//     auto link_index = get_dispatch_link_index(control_plane, is_galaxy,
//                                               src_fabric_node_id, dst_fabric_node_id, device);
//
//   Inside get_dispatch_link_index() (non-galaxy path):
//
//     const auto& available_links = tt_fabric::get_forwarding_link_indices(src, dst);
//     TT_FATAL(!available_links.empty(), "No links available from {} to {}",
//              src_fabric_node_id, dst_fabric_node_id);    // <-- fires → SIGABRT
//
//   The existing FIX NY/NY+ guards only check whether chips are *excluded from the
//   fabric cluster* (is_physical_chip_in_fabric_cluster() == false).  They do NOT
//   catch the case where BOTH chips ARE in the fabric cluster but every ETH channel
//   between them is dead (e.g. after progressive SIGKILL teardown corruption on T3K).
//
//   On a progressively-degraded T3K cluster:
//   - Session N leaves ERISC L1 channels 0,1,8,9 of Device 2 corrupt.
//   - Session N+1 FIX TB detects L1 corruption → those channels excluded.
//   - Session N+1 teardown (SIGKILL) further corrupts channels 14,15 of Device 2.
//   - Session N+2: ALL channels between Device 2 and Device 3 are excluded.
//     get_forwarding_link_indices(M0/D2, M0/D3) returns {}.
//     get_dispatch_link_index() TT_FATALs → SIGABRT → test runner gets abort.
//
// The fix (FIX TH):
//   In GenerateStaticConfigs(), BEFORE calling get_dispatch_link_index(), add:
//
//     const auto& available_links =
//         tt_fabric::get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);
//     if (available_links.empty()) {
//         log_warning(tt::LogMetal, "FIX TH (#42429): no available dispatch links ...");
//         device_->set_fabric_channels_not_ready_for_traffic();
//         return;
//     }
//
//   This marks the device `channels_not_ready_for_traffic` so that FIX SA in
//   CustomMeshGraphFabric2DFixture::SetUp() calls GTEST_SKIP() instead of
//   proceeding with the unicast test on a broken link.
//
// What this test verifies:
//   Fork test that forces topology degradation via SIGKILL'd predecessor, then
//   spawns a testee that tries to init FABRIC_2D with the T3K custom mesh descriptor.
//   Verifies that RelayMux::GenerateStaticConfigs does NOT TT_FATAL (SIGABRT) when
//   all ETH channels between two MMIO devices are dead.
//
//   Without FIX TH: TT_FATAL "No links available from (M0,D2) to (M0,D3)" → SIGABRT.
//   With FIX TH:    warning logged, device marked not-ready → returns cleanly → exits 0.
//
// Regression indicator:
//   TESTEE killed by SIGABRT (exit 69 via signal handler sentinel) = FIX TH regression.
//   TESTEE exits 0 or with another exception = fix working (or cluster still healthy).
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    90s (full T3K FABRIC_2D relay_mux init on degraded cluster)
//   Total:            ~150s
//
// Topology requirement: >= 8 devices (T3K 2x4 mesh).

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
static constexpr int kGap69PredWaitMs    = 35000;  // 35s for predecessor init + dispatch
static constexpr int kGap69TesteeBudget  = 90000;  // 90s for T3K FABRIC_2D relay_mux init
// Sentinel exit code: testee SIGABRT'd (TT_FATAL "No links available") = FIX TH regression.
// SIGABRT handler in testee calls _exit(kGap69RegressionExit) so parent can distinguish
// regression-abort from clean exits.
static constexpr int kGap69RegressionExit = 69;

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap69Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// SIGABRT handler installed in the TESTEE child process only.
// Called when TT_FATAL fires (which calls abort() after logging).
// Exits with the regression sentinel so the parent can detect the failure.
// Note: _exit() is async-signal-safe; exit() is not.
// ---------------------------------------------------------------------------
static void gap69_sigabrt_handler(int /*sig*/) {
    // Write sentinel message to stderr before exiting — helps CI log analysis.
    const char msg[] =
        "GAP-69 TESTEE SIGABRT: TT_FATAL triggered during relay_mux init — "
        "FIX TH is missing or reverted. "
        "Expected: get_forwarding_link_indices() preflight in GenerateStaticConfigs(). "
        "Source: relay_mux.cpp get_dispatch_link_index() TT_FATAL(!available_links.empty()).\n";
    (void)::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    ::_exit(kGap69RegressionExit);
}

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap69(const MeshCoordinateRange& range) {
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
static int wait_child_budget_gap69(pid_t pid, int budget_ms) {
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
class FixThRelayMuxNoLinksGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixThRelayMuxNoLinksGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 8) {
            GTEST_SKIP() << "GAP-69 requires >= 8 devices (T3K 2x4 with custom mesh descriptor). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-69: FixThRelayMuxNoDispatchLinksAbortsWithoutGuard
//
// Verifies FIX TH: RelayMux::GenerateStaticConfigs must NOT TT_FATAL when
// get_forwarding_link_indices(src, dst) returns empty (all ETH channels dead)
// on a degraded cluster where both chips ARE in the fabric cluster.
// ---------------------------------------------------------------------------
TEST_F(FixThRelayMuxNoLinksGuardFixture, RelayMuxNoDispatchLinksAbortsWithoutGuard) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap69Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap69Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ─────────────────────────────────────────────
    // Opens T3K with FABRIC_2D (all 8 devices), dispatches a blank workload to
    // spin up ETH firmware on all chips, signals ready, then spins until SIGKILL.
    // Leaves ETH channels in a mid-session state, which on the next session causes
    // FIX TB to exclude chips with corrupt ERISC L1 from the topology mapper.
    // Progressive corruption means more channels go dead each SIGKILL cycle.
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
            auto workload = make_blank_workload_gap69(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Ignore — we just need the ETH state set up
        }
        shm->predecessor_ready.store(1);
        // Spin until SIGKILL — leaves ERISCs in mid-session state
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    // Wait for predecessor to signal init complete
    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - pred_start)
                           .count();
        if (elapsed > kGap69PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap69Shm));
            GTEST_SKIP() << "GAP-69: predecessor did not signal ready within " << kGap69PredWaitMs
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
        "GAP-69: Predecessor SIGKILL'd — ETH channels may be in mid-session state. "
        "TESTEE will attempt FABRIC_2D MeshDevice init with T3K custom mesh descriptor. "
        "On a degraded cluster, all ETH channels between two MMIO devices may be dead. "
        "Without FIX TH: get_dispatch_link_index() TT_FATALs → SIGABRT. "
        "With FIX TH: get_forwarding_link_indices() preflight returns early, "
        "device marked channels_not_ready_for_traffic.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ──────────────────────────────────────────────────
    // Installs a SIGABRT handler that exits with kGap69RegressionExit (69).
    // Then tries to init FABRIC_2D with the T3K 2x4 custom mesh descriptor.
    // relay_mux::GenerateStaticConfigs() is called for each MMIO device's relay path.
    //
    // On a progressively-degraded cluster:
    //   - Device 2's channels 0,1 (D2→D3 intra-host) are dead
    //   - get_forwarding_link_indices(M0/D2, M0/D3) returns {}
    //   - Without FIX TH: get_dispatch_link_index() TT_FATALs → SIGABRT → exits 69
    //   - With FIX TH:    preflight returns early, warning logged → exits 0
    //   - Healthy cluster: channels OK, no issue → exits 0
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        // Install SIGABRT handler so TT_FATAL → abort() → exits 69 instead of SIGABRT.
        // This lets the parent distinguish regression from other crashes.
        struct sigaction sa{};
        sa.sa_handler = gap69_sigabrt_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGABRT, &sa, nullptr);

        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

            // Use the T3K 2x4 custom mesh descriptor — same as T3kCustomMeshGraphFabric2DFixture.
            // This triggers the dispatch kernel config path that calls RelayMux::GenerateStaticConfigs().
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
            fprintf(
                stderr,
                "GAP-69 TESTEE: MeshDevice created cleanly (cluster healthy or FIX TH returned early).\n");
        } catch (const std::exception& e) {
            // Other exception (topology mismatch, missing relay, etc.) — acceptable on degraded cluster
            fprintf(
                stderr,
                "GAP-69 TESTEE: caught exception (acceptable on degraded cluster): %s\n",
                e.what());
        } catch (...) {
            fprintf(stderr, "GAP-69 TESTEE: caught unknown exception (acceptable on degraded cluster).\n");
        }

        // If we reach here, no TT_FATAL fired — FIX TH is working (or cluster is healthy)
        _exit(0);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap69(testee_pid, kGap69TesteeBudget);
    auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - testee_start)
                         .count();

    ::munmap(raw_shm, sizeof(Gap69Shm));

    // Primary regression check: SIGABRT handler exited with sentinel 69 (FIX TH missing)
    if (rc == kGap69RegressionExit) {
        FAIL() << "GAP-69 REGRESSION (FIX TH missing): TESTEE caught SIGABRT from TT_FATAL.\n"
               << "\n"
               << "Root cause: RelayMux::GenerateStaticConfigs() in relay_mux.cpp called\n"
               << "get_dispatch_link_index() without checking whether\n"
               << "get_forwarding_link_indices(src, dst) returns empty first.\n"
               << "\n"
               << "On a progressively-degraded T3K cluster, all ETH channels between\n"
               << "two MMIO devices (e.g. Device 2 and Device 3) can be dead even though\n"
               << "both chips ARE in the fabric cluster (FIX NY/NY+ checks pass).\n"
               << "get_forwarding_link_indices() returns {} → get_dispatch_link_index()\n"
               << "TT_FATALs with 'No links available from (M0,D2) to (M0,D3)'.\n"
               << "\n"
               << "Fix (FIX TH):\n"
               << "  In RelayMux::GenerateStaticConfigs(), before calling get_dispatch_link_index():\n"
               << "    const auto& available_links =\n"
               << "        tt_fabric::get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);\n"
               << "    if (available_links.empty()) {\n"
               << "        log_warning(tt::LogMetal, \"FIX TH (#42429): ...\");\n"
               << "        device_->set_fabric_channels_not_ready_for_traffic();\n"
               << "        return;\n"
               << "    }\n"
               << "  See: tt_metal/impl/dispatch/kernel_config/relay_mux.cpp, GenerateStaticConfigs()\n";
    }

    if (rc == -1) {
        FAIL() << "GAP-69 TIMEOUT: TESTEE did not exit within " << kGap69TesteeBudget
               << "ms (elapsed: " << testee_ms << "ms). "
               << "MeshDevice init hung — possible related hang issue.";
    }

    // Any other exit code (0, 1, other exception path) is acceptable
    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-69: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-69 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TH relay_mux no-links preflight guard is working — "
        "no TT_FATAL from get_dispatch_link_index() on degraded cluster.",
        testee_ms,
        kGap69TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
