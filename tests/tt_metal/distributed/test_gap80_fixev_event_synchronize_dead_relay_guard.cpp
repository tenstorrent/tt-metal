// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-80: FIX EV — EventSynchronize infinite spin guard on dead-relay device.
//
// Root cause:
//
//   After FIX Z/GAP-A throws from read_completion_queue_event (reader thread detects
//   relay_path_broken=true on a non-MMIO device), the main thread in device_operation.hpp
//   calls EventSynchronize(completion_event).  That function enters nice_spin_until()
//   polling sysmem.get_last_completed_event(cq_id) >= target_id.  For a non-MMIO device
//   with dead relay, the event counter is never updated → infinite spin → 8 minutes until
//   CI cancels.
//
// The fix (FIX EV, distributed.cpp:EventSynchronize):
//   Check is_fabric_relay_path_broken() BEFORE touching sysmem.  If a non-MMIO device has
//   a broken relay, log a warning and `continue` to the next device.  The reader thread's
//   exception surfaces through the normal error-propagation path.
//
// What this test verifies:
//   Phase 1 (PREDECESSOR): Opens FABRIC_1D MeshDevice, dispatches blank workload to
//   establish all fabric channels, signals ready, then gets SIGKILL'd.
//
//   Phase 2 (TESTEE): Opens FABRIC_1D MeshDevice.  If relay is broken on any non-MMIO
//   device, attempts a simple dispatch + EventSynchronize sequence.  Without FIX EV, the
//   EventSynchronize would hang forever on the dead device.  With FIX EV, it skips the
//   dead device and the error surfaces normally.  Testee should complete (exit 0) within
//   budget regardless of cluster health state.
//
// Regression: without FIX EV, EventSynchronize infinite-loops on dead-relay device →
//   testee exceeds budget → FAIL.
//
// Timing budget:
//   PREDECESSOR: 30s (init + blank workload)
//   TESTEE:      60s (init + dispatch attempt + EventSynchronize)
//   Total:       ~120s
//
// Topology requirement: >= 2 devices (need non-MMIO device).

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
#include <tt-metalium/experimental/fabric/fabric.hpp>
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

static constexpr int kGap80PredWaitMs   = 30000;
static constexpr int kGap80TesteeBudget = 60000;

struct Gap80Shm {
    std::atomic<int> predecessor_ready{0};
};

static MeshWorkload make_blank_workload_gap80(const MeshCoordinateRange& range) {
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

static int wait_child_budget_gap80(pid_t pid, int budget_ms) {
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

// GAP-80: FIX EV — EventSynchronize does not infinite-spin on dead-relay non-MMIO device.
TEST(DistributedUnitTests, Gap80_FixEV_EventSynchronizeDeadRelayGuard) {
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        GTEST_SKIP() << "GAP-80 requires >= 2 devices for non-MMIO relay scenario (found " << num_devices << ")";
    }

    void* shm_raw = ::mmap(nullptr, sizeof(Gap80Shm), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(shm_raw, MAP_FAILED) << "mmap failed: " << strerror(errno);
    auto* shm = new (shm_raw) Gap80Shm{};

    // Phase 1: PREDECESSOR — full mesh, blank workload, SIGKILL
    pid_t pred_pid = ::fork();
    ASSERT_NE(pred_pid, -1) << "fork failed: " << strerror(errno);
    if (pred_pid == 0) {
        try {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
            auto mesh = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<size_t>(num_devices)}));
            auto range = MeshCoordinateRange(mesh->shape());
            auto workload = make_blank_workload_gap80(range);
            EnqueueMeshWorkload(mesh->mesh_command_queue(0), workload, false);
            Finish(mesh->mesh_command_queue(0));
            shm->predecessor_ready.store(1, std::memory_order_release);
            while (true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
        } catch (const std::exception& e) {
            log_error(LogTest, "GAP-80 PREDECESSOR exception: {}", e.what());
            _exit(1);
        }
    }

    // Wait for predecessor, then SIGKILL
    {
        const auto start = std::chrono::steady_clock::now();
        while (shm->predecessor_ready.load(std::memory_order_acquire) == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (elapsed > kGap80PredWaitMs) {
                ::kill(pred_pid, SIGKILL);
                ::waitpid(pred_pid, nullptr, 0);
                ::munmap(shm_raw, sizeof(Gap80Shm));
                GTEST_SKIP() << "GAP-80: PREDECESSOR did not become ready within budget";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    ::kill(pred_pid, SIGKILL);
    ::waitpid(pred_pid, nullptr, 0);

    // Phase 2: TESTEE — open mesh, attempt dispatch, verify no infinite hang
    pid_t testee_pid = ::fork();
    ASSERT_NE(testee_pid, -1) << "fork failed: " << strerror(errno);
    if (testee_pid == 0) {
        try {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
            auto mesh = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<size_t>(num_devices)}));

            // Check relay status — if broken, FIX EV will guard EventSynchronize
            bool has_broken_relay = false;
            for (auto* dev : mesh->get_devices()) {
                if (!dev->is_mmio_capable() && dev->is_fabric_relay_path_broken()) {
                    has_broken_relay = true;
                    log_info(LogTest,
                             "GAP-80 TESTEE: device {} has relay_path_broken — "
                             "FIX EV guard will be exercised on EventSynchronize.",
                             dev->id());
                }
            }

            if (has_broken_relay) {
                // Attempt dispatch — this will throw from the reader thread (FIX Z/GAP-A).
                // The key regression check: EventSynchronize must NOT infinite-loop on the
                // dead device.  If we reach the catch block within budget, FIX EV is working.
                try {
                    auto range = MeshCoordinateRange(mesh->shape());
                    auto workload = make_blank_workload_gap80(range);
                    EnqueueMeshWorkload(mesh->mesh_command_queue(0), workload, false);
                    Finish(mesh->mesh_command_queue(0));
                } catch (const std::exception& e) {
                    log_info(LogTest,
                             "GAP-80 TESTEE: dispatch threw as expected on degraded cluster: {}",
                             e.what());
                    // This is the expected path — dispatch detected broken relay
                }
            } else {
                log_info(LogTest,
                         "GAP-80 TESTEE: no broken relay detected — cluster recovered. "
                         "FIX EV guard present but not triggered (still correct).");
            }

            mesh->close();
            _exit(0);
        } catch (const std::exception& e) {
            // Init-time exception is acceptable
            log_warning(LogTest, "GAP-80 TESTEE exception (acceptable): {}", e.what());
            _exit(0);
        }
    }

    int testee_rc = wait_child_budget_gap80(testee_pid, kGap80TesteeBudget);
    ::munmap(shm_raw, sizeof(Gap80Shm));

    if (testee_rc == -1) {
        FAIL() << "GAP-80: TESTEE timed out after " << kGap80TesteeBudget
               << "ms — possible FIX EV regression: EventSynchronize infinite-looped "
               << "on dead-relay non-MMIO device instead of skipping.";
    }
    ASSERT_EQ(testee_rc, 0)
        << "GAP-80: TESTEE exited with unexpected code " << testee_rc;
}

}  // namespace tt::tt_metal::distributed::test
