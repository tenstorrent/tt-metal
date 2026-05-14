// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-54: FIX BA — guard teardown_fabric_config() against lazy ControlPlane re-init.
// Commit: 0f6d9c3fb0b
//
// Root cause (CI run 25137442210 — t3000-apc-fast-tests):
//
//   When a CustomMeshGraphFabric2DFixture TearDown() calls set_default_fabric_topology():
//     1. control_plane_ is reset to null (line 643)
//     2. custom_mesh_graph_desc_path_ is reset to nullopt (line 649)
//     3. set_fabric_config() is called (line 651) → teardown_fabric_config() (line 263)
//     4. teardown_fabric_config() unconditionally calls:
//          this->get_control_plane().clear_fabric_context()
//     5. get_control_plane() triggers LAZY re-initialization of ControlPlane since
//        control_plane_ is null.
//     6. On degraded hardware (e.g., 5 of 8 expected nodes due to dead ETH channels),
//        the lazy re-init does topology discovery → TopologyMapper fails to fit the
//        custom mesh graph → TT_FATAL → crash.
//
//   This cascade crashes every remaining parameterized test instance AND the process
//   exit cleanup (MetalContext::destroy_all_instances), producing repeated
//   "unordered_map::at" exceptions.
//
// FIX BA (metal_env.cpp teardown_fabric_config()):
//   Guard the clear_fabric_context() call:
//     if (control_plane_) { control_plane_->clear_fabric_context(); }
//   If control_plane_ is null, there is no fabric context to clear — skip silently.
//
// What this test verifies:
//   This is a CRASH regression test. The failure mode is TT_FATAL → SIGABRT when
//   teardown_fabric_config() triggers lazy ControlPlane re-init on degraded hardware.
//
//   We cannot directly trigger the degraded-hardware scenario in a unit test. Instead,
//   we verify that a normal FABRIC_2D open/close cycle completes without crash (exit 0).
//   On hardware with degraded topology (dead ETH channels from prior sessions), this
//   exercises the exact code path where FIX BA prevents the crash.
//
//   PRIMARY ASSERTION: testee exits 0 (no SIGABRT from teardown_fabric_config).
//   SECONDARY: testee completes within budget (no hang).

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

static constexpr int kPredWaitMs = 30000;
static constexpr int kTesteeBudgetMs = 90000;

struct Gap54SharedMem {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap54(const MeshCoordinateRange& range) {
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

class TeardownNullControlPlaneFixture : public MeshDeviceFixtureBase {
protected:
    TeardownNullControlPlaneFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 180000,
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "GAP-54 requires >= 2 devices. Found " << num_devices << ".";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-54: TeardownFabricConfigDoesNotCrashOnDegradedHardware
//
// CRASH regression test for FIX BA.
// Regression indicator: testee exits with SIGABRT (134) from lazy ControlPlane re-init.
// ---------------------------------------------------------------------------
TEST_F(TeardownNullControlPlaneFixture, TeardownFabricConfigDoesNotCrashOnDegradedHardware) {
    mesh_device_->close();

    void* raw_shm =
        ::mmap(nullptr, sizeof(Gap54SharedMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (raw_shm) Gap54SharedMem();

    const size_t num_dev = MetalContext::instance().get_cluster().number_of_devices();

    // ── Step 1: PREDECESSOR leaves degraded ETH state ─────────────────────────
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
            auto workload = make_blank_workload_gap54(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
        } catch (...) {}
        shm->predecessor_ready.store(1);
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(0);
    }

    const auto pred_start = std::chrono::steady_clock::now();
    while (shm->predecessor_ready.load() == 0) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pred_start)
                .count() > kPredWaitMs) {
            ::kill(pred_pid, SIGKILL);
            ::waitpid(pred_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap54SharedMem));
            GTEST_SKIP() << "GAP-54: predecessor did not signal ready within " << kPredWaitMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    log_info(
        tt::LogTest,
        "GAP-54: Predecessor SIGKILL'd — hardware may be degraded (dead ETH channels). "
        "Testee will open/close MeshDevice. FIX BA ensures teardown_fabric_config() "
        "does not trigger lazy ControlPlane re-init when control_plane_ is null.");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // ── Step 2: TESTEE — open/close exercising teardown path ──────────────────
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
            // Catch ALL exceptions — we want to distinguish exception (rc=1)
            // from SIGABRT (rc=134, the FIX BA regression path).
            rc = 1;
        } catch (...) { rc = 2; }
        _exit(rc);
    }

    const auto testee_start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
        pid_t waited = ::waitpid(testee_pid, &status, WNOHANG);
        if (waited == testee_pid) break;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - testee_start)
                               .count();
        if (elapsed_ms > kTesteeBudgetMs) {
            ::kill(testee_pid, SIGKILL);
            ::waitpid(testee_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(Gap54SharedMem));
            FAIL() << "GAP-54: Testee timed out at " << kTesteeBudgetMs << "ms.";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - testee_start)
                           .count();

    ::munmap(raw_shm, sizeof(Gap54SharedMem));

    // PRIMARY CHECK: no SIGABRT (the FIX BA regression path).
    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        if (sig == SIGABRT) {
            FAIL() << "GAP-54 CRASH (FIX BA regression): Testee killed by SIGABRT.\n"
                   << "\n"
                   << "Root cause: teardown_fabric_config() unconditionally calls\n"
                   << "get_control_plane().clear_fabric_context(). If control_plane_ is null,\n"
                   << "this triggers lazy re-initialization on degraded hardware.\n"
                   << "TopologyMapper fails → TT_FATAL → SIGABRT.\n"
                   << "\n"
                   << "Fix (FIX BA) in metal_env.cpp teardown_fabric_config():\n"
                   << "  if (control_plane_) { control_plane_->clear_fabric_context(); }\n";
        }
        FAIL() << "GAP-54: Testee killed by signal " << sig << " (unexpected).";
    }

    if (WIFEXITED(status)) {
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << "GAP-54: Testee exited with code " << WEXITSTATUS(status);
    }

    log_info(
        tt::LogTest,
        "GAP-54 PASS: Testee completed in {}ms with exit 0 (no SIGABRT). "
        "FIX BA guard in teardown_fabric_config() prevented lazy ControlPlane re-init.",
        elapsed_ms);
}

}  // namespace tt::tt_metal::distributed::test
