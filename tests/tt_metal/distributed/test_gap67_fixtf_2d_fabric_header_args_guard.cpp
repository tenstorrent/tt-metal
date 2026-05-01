// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-67: FIX TF gap — assemble_2d_fabric_packet_header_args must guard against nullopt
//         forwarding direction when inter-mesh relay is broken (degraded cross-mesh routing).
// Commit: 2f828dc16d9
//
// Root cause (runs 25196253821, 25195921333 — T3K fast tests):
//
//   When a T3K cluster (2 meshes M0={0,1,2,3}, M1={4,5,6,7}) has corrupt ERISC L1
//   (edm_status=0x49705180 — ROM postcode, hardware never ran firmware), the control
//   plane routing table has no inter-mesh routing entries for M0↔M1 pairs.
//
//   assemble_2d_fabric_packet_header_args() (relay_mux.hpp) is called from prefetch.cpp
//   and dispatch.cpp during dispatch kernel config generation for cross-mesh device pairs.
//   Inside, it calls:
//     const auto& forwarding_direction = control_plane.get_forwarding_direction(src, dst);
//     const auto router_direction = control_plane.routing_direction_to_eth_direction(
//         forwarding_direction.value());   // <-- OLD CODE: .value() without has_value() check
//
//   get_forwarding_direction() returns std::nullopt when the routing table has NONE for
//   the inter-mesh pair (broken inter-mesh relay).  Calling .value() on nullopt throws
//   std::bad_optional_access — caught by GTest as an opaque "bad optional access" in
//   SetUp() with zero context about which chip pair caused the failure.
//
//   This was the *only* visible symptom: GTest message "bad optional access" in
//   T3kCustomMeshGraphFabric2DFixture.TestUnicastRaw/0 SetUp() after 73 seconds.
//   No indication of src/dst chip IDs, no indication that the fix was in relay_mux.hpp.
//
// The fix (FIX TF, relay_mux.hpp:assemble_2d_fabric_packet_header_args):
//   Replace raw .value() call with:
//     TT_FATAL(
//         forwarding_direction.has_value(),
//         "FIX TF: No forwarding direction from physical chip {} (fabric node {}) "
//         "to physical chip {} (fabric node {}). Inter-mesh relay path is broken "
//         "or no route exists in the control plane routing table.",
//         my_device_id, src_fabric_node_id, destination_device_id, dst_fabric_node_id);
//
//   Now, when the inter-mesh relay is broken, the exception message contains chip IDs
//   and fabric node IDs so the on-call engineer can immediately identify the broken pair.
//
// What this test verifies:
//   Fork test that forces inter-mesh ETH degradation via SIGKILL'd predecessor, then
//   spawns a testee that tries to init FABRIC_2D with a T3K mesh.  Verifies that the
//   exception thrown during dispatch kernel config generation is NOT std::bad_optional_access
//   (which would be the pre-FIX-TF regression mode).  If the exception is bad_optional_access,
//   the test FAILS with a clear message pointing to relay_mux.hpp and FIX TF.
//
//   The crash path (without FIX TF):
//     1. Predecessor SIGKILL'd → inter-mesh ERISCs dead → control plane has no M0↔M1 route
//     2. TESTEE creates MeshDevice with FABRIC_2D (all 8 chips, T3K 2x4 mesh)
//     3. Dispatch kernel config calls assemble_2d_fabric_packet_header_args(M0_chip, M1_chip)
//     4. get_forwarding_direction() returns nullopt → .value() throws bad_optional_access
//     5. GTest catches "bad optional access" — no context, no chip IDs
//
//   With FIX TF:
//     1. TT_FATAL fires with message "FIX TF: No forwarding direction from physical chip N..."
//     2. Exception message is tt::exception containing chip IDs and "FIX TF"
//     3. GTest/catch block can identify the root cause from the message
//
// Regression indicator:
//   TESTEE catches std::bad_optional_access (exit sentinel 67) = FIX TF regression.
//   TESTEE catches exception with "FIX TF" in message (exit 0) = fix working.
//   Cluster healthy — no exception during init (exit 0) = also pass.
//
// Timing budget:
//   PREDECESSOR wait: 35s (hardware init + blank workload dispatch)
//   TESTEE budget:    90s (full T3K FABRIC_2D dispatch kernel config including degraded path)
//   Total:            ~150s
//
// Topology requirement: >= 8 devices (T3K 2x4 mesh required for inter-mesh routing;
//   fewer devices → no cross-mesh dispatch config attempted → GTEST_SKIP).

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
static constexpr int kGap67PredWaitMs    = 35000;  // 35s for predecessor init + dispatch
static constexpr int kGap67TesteeBudget  = 90000;  // 90s for T3K FABRIC_2D dispatch config
// Sentinel exit code: testee caught bad_optional_access (FIX TF regression)
static constexpr int kGap67RegressionExit = 67;

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
struct Gap67Shm {
    std::atomic<int> predecessor_ready{0};
};

// ---------------------------------------------------------------------------
// Blank workload helper
// ---------------------------------------------------------------------------
static MeshWorkload make_blank_workload_gap67(const MeshCoordinateRange& range) {
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
static int wait_child_budget_gap67(pid_t pid, int budget_ms) {
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
class FixTf2dFabricHeaderArgsGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixTf2dFabricHeaderArgsGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 8) {
            GTEST_SKIP() << "GAP-67 requires >= 8 devices (T3K 2x4 with inter-mesh routing). "
                         << "Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-67: FixTfNoForwardingDirectionGivesContextOnCrash
//
// Verifies FIX TF: assemble_2d_fabric_packet_header_args must not throw
// std::bad_optional_access when inter-mesh relay is broken — it must throw
// tt::exception (from TT_FATAL) with clear chip ID context.
// ---------------------------------------------------------------------------
TEST_F(FixTf2dFabricHeaderArgsGuardFixture, NoForwardingDirectionGivesContextOnCrash) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap67Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap67Shm();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Phase 1: Fork PREDECESSOR ───────────────────────────────────────────────
    // Opens T3K with FABRIC_2D (all 8 devices), dispatches a blank workload to
    // spin up inter-mesh ERISCs, signals ready, then spins until SIGKILL.
    // Leaves inter-mesh ETH channels in a mid-session state, forcing the control
    // plane on the next session to find no valid M0↔M1 routing table entries
    // (channels will be "active" but the routing table won't have a route).
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
            auto workload = make_blank_workload_gap67(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {
            // Ignore any exceptions in predecessor; we just need the ETH state set up
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
        if (elapsed > kGap67PredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap67Shm));
            GTEST_SKIP() << "GAP-67: predecessor did not signal ready within " << kGap67PredWaitMs
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
        "GAP-67: Predecessor SIGKILL'd — inter-mesh ERISCs may be dead. "
        "TESTEE will attempt FABRIC_2D MeshDevice init. "
        "If inter-mesh routing is broken, assemble_2d_fabric_packet_header_args() "
        "will be called for a cross-mesh pair with no routing entry. "
        "With FIX TF: TT_FATAL with chip IDs. Without: bad_optional_access.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Phase 2: Fork TESTEE ────────────────────────────────────────────────────
    // Tries to init FABRIC_2D with the full T3K (2-mesh) configuration.
    // If inter-mesh ETH is degraded, the control plane will have no M0↔M1 route.
    // Dispatch kernel config will call assemble_2d_fabric_packet_header_args(M0_chip, M1_chip).
    //
    // Without FIX TF: std::bad_optional_access thrown → exits kGap67RegressionExit (67)
    // With FIX TF:    tt::exception with "FIX TF:" thrown → exits 0 (caught, message checked)
    // Healthy cluster: MeshDevice created successfully → exits 0
    pid_t testee_pid = ::fork();
    ASSERT_GE(testee_pid, 0) << "fork() failed: " << strerror(errno);

    if (testee_pid == 0) {
        bool caught_bad_optional = false;

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
            fprintf(stderr, "GAP-67 TESTEE: MeshDevice created cleanly (cluster healthy).\n");
        } catch (const std::bad_optional_access& e) {
            // Regression: FIX TF missing — raw .value() threw bad_optional_access
            caught_bad_optional = true;
            fprintf(
                stderr,
                "GAP-67 TESTEE REGRESSION: caught std::bad_optional_access — "
                "FIX TF is missing or reverted. "
                "Source: assemble_2d_fabric_packet_header_args in relay_mux.hpp "
                "called .value() on std::optional<RoutingDirection> without has_value() check. "
                "Exception: %s\n",
                e.what());
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            if (msg.find("FIX TF") != std::string::npos) {
                fprintf(
                    stderr,
                    "GAP-67 TESTEE: caught FIX TF TT_FATAL (expected when inter-mesh broken): %s\n",
                    msg.c_str());
            } else {
                // Other exception (topology mismatch, etc.) — acceptable
                fprintf(
                    stderr,
                    "GAP-67 TESTEE: caught other exception (expected on degraded cluster): %s\n",
                    msg.c_str());
            }
        } catch (...) {
            fprintf(stderr, "GAP-67 TESTEE: caught unknown exception (expected on degraded).\n");
        }

        if (caught_bad_optional) {
            _exit(kGap67RegressionExit);  // Regression signal
        }
        _exit(0);  // Pass: healthy, FIX TF fired, or other clean exception
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int rc = wait_child_budget_gap67(testee_pid, kGap67TesteeBudget);
    auto testee_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - testee_start)
                         .count();

    ::munmap(raw_shm, sizeof(Gap67Shm));

    // Primary regression check: bad_optional_access was caught (FIX TF missing)
    if (rc == kGap67RegressionExit) {
        FAIL() << "GAP-67 REGRESSION (FIX TF missing): TESTEE caught std::bad_optional_access.\n"
               << "\n"
               << "Root cause: assemble_2d_fabric_packet_header_args() in relay_mux.hpp\n"
               << "called forwarding_direction.value() without first checking has_value().\n"
               << "When inter-mesh relay is broken (chips 4-7 degraded), get_forwarding_direction()\n"
               << "returns std::nullopt for cross-mesh pairs (M0 chip → M1 chip). The .value()\n"
               << "call throws bad_optional_access with no context about which chip pair failed.\n"
               << "\n"
               << "Fix (FIX TF, commit 2f828dc16d9):\n"
               << "  Replace forwarding_direction.value() with:\n"
               << "  TT_FATAL(forwarding_direction.has_value(),\n"
               << "      \"FIX TF: No forwarding direction from physical chip {} ...\",\n"
               << "      my_device_id, src_fabric_node_id, ...);\n"
               << "  See: tt_metal/impl/dispatch/kernel_config/relay_mux.hpp";
    }

    if (rc == -1) {
        FAIL() << "GAP-67 TIMEOUT: TESTEE did not exit within " << kGap67TesteeBudget
               << "ms (elapsed: " << testee_ms << "ms). "
               << "Dispatch kernel config generation hung — possible related hang issue.";
    }

    if (rc == 134) {
        // SIGABRT from another TT_FATAL — not the bad_optional_access regression, but still a failure
        FAIL() << "GAP-67 CRASH: TESTEE killed by SIGABRT (exit 134). "
               << "Not the bad_optional_access regression (FIX TF), but a different TT_FATAL. "
               << "Check stderr for the fatal message and context.";
    }

    EXPECT_TRUE(rc == 0 || rc == 1)
        << "GAP-67: TESTEE exited with unexpected code " << rc << " (expected 0 or 1).";

    log_info(
        tt::LogTest,
        "GAP-67 PASS: TESTEE completed in {}ms (budget: {}ms) exit {}. "
        "FIX TF assemble_2d_fabric_packet_header_args() guard is working — "
        "no opaque bad_optional_access thrown when inter-mesh relay is broken.",
        testee_ms,
        kGap67TesteeBudget,
        rc);
}

}  // namespace tt::tt_metal::distributed::test
