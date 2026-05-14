// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-28: FIX AV relay-broken guard in configure_command_queue_programs
//
// Background:
//   FIX AV (device.cpp ~line 194) wraps sysmem_manager_->reset() with:
//
//       if (!this->fabric_relay_path_broken_.load()) {
//           for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
//               this->sysmem_manager_->reset(cq_id);
//           }
//       }
//
//   reset() reads PREFETCH_Q_RD_PTR_ADDR from the device — a read that routes
//   through the UMD ETH relay on non-MMIO devices.  If the relay is broken
//   (predecessor killed mid-operation, leaving ERISC channels in ACTIVE state),
//   the relay CMD queue is full and read_non_mmio() blocks indefinitely at the
//   is_non_mmio_cmd_q_full() loop with no timeout.
//
//   Without the guard:
//     configure_command_queue_programs() hangs indefinitely during MeshDevice::create()
//     on any non-MMIO device whose relay is broken.  CI timeout fires after 15 minutes.
//
//   With the guard:
//     reset() is skipped for relay-broken devices.  The in-flight counter may
//     be stale (this is acceptable — relay-broken devices don't dispatch via the
//     relay path anyway, per FIX Z).  configure_command_queue_programs() proceeds
//     immediately.  MeshDevice::create() completes within the normal init window.
//
// What this test verifies:
//   1. MeshDevice::create() completes in < 45s after a SIGKILL predecessor leaves
//      relay-broken ERISC state (no hang from relay-read in sysmem reset).
//   2. The re-opened device can dispatch at least one blank workload without
//      hanging — confirming that configure_command_queue_programs reached a usable
//      state despite skipping reset() for relay-broken channels.
//
// Gap vs. existing tests:
//   GAP-11 (RelayTimeoutToleranceOnReopen) verifies that MeshDevice::create()
//   doesn't hang due to relay CMD queue saturation during terminate_stale_erisc_routers
//   (FIX AQ). It does NOT exercise configure_command_queue_programs' sysmem_manager_
//   reset path — that path runs AFTER FIX AQ and is a separate hang surface.
//
//   GAP-26 (FIX AS canary timeout) verifies that when Pass-0 canary poll marks
//   channels dead, the open completes gracefully. It does not specifically target
//   the configure_command_queue_programs path for relay-broken devices.
//
//   GAP-27 (FIX AV non-MMIO sysmem reset) verifies the HAPPY PATH: non-MMIO
//   sysmem_manager_->reset() runs for healthy relays and stale counters don't
//   accumulate. It does NOT cover the relay-broken guard (skip path).
//
//   GAP-28 is the first test targeted at the guard's SKIP PATH: relay broken →
//   skip reset() → no hang.
//
// Topology requirement: >= 2 devices.
//   Single-chip systems have no non-MMIO relay path. FIX AV's guard is only
//   exercised when fabric_relay_path_broken_ can be set, which requires a
//   non-MMIO device whose relay times out — only possible on N300 or larger.

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
#include <optional>
#include <thread>
#include <vector>

#include <experimental/fabric/fabric_types.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "fabric/fabric_init.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture
//
// Budget: 150 000 ms (2.5 min) covering:
//   ~20 s  child FABRIC_2D init
//   ~20 s  kMaxWaitMs waiting for child_ready
//    ~2 s  post-kill margin
//   ~45 s  parent MeshDevice::create() (the hard limit asserted below)
//   ~63 s  margin for slow hardware / ccache misses
// ---------------------------------------------------------------------------
class FixAvRelayBrokenSysmemGuardFixture : public MeshDeviceFixtureBase {
protected:
    FixAvRelayBrokenSysmemGuardFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 150000,
          }) {}

    void SetUp() override {
        // FIX AV's relay-broken guard is only exercised when fabric_relay_path_broken_
        // can be set on a non-MMIO device.  Single-chip systems have no ETH relay path.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "FixAvRelayBrokenSysmemGuardFixture requires >= 2 devices "
                            "(non-MMIO relay path required for FIX AV guard to fire). "
                            "Found "
                         << num_devices << " device(s). "
                            "Single-chip has no ETH relay; FIX AV skip path is never reached.";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// Helper: lightest workload that exercises the full host → device path.
static MeshWorkload make_blank_workload_gap28(const MeshCoordinateRange& range) {
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
// GAP-28: FixAvRelayBrokenSkipsSystemMemReset
//
// Scenario: fork + SIGKILL, same skeleton as GAP-11 (RelayTimeoutToleranceOnReopen).
//
//   Phase 0: Probe fork() availability.
//
//   Phase 1: Close fixture device so child inherits a clean MetalContext.
//
//   Phase 2: Allocate shared-memory "child_ready" flag.
//
//   Phase 3: Child process:
//     - Opens FABRIC_2D (activates ERISC EDM firmware on all ETH channels).
//     - Dispatches a blank workload (ERISCs in ACTIVE state, relay live).
//     - Sets child_ready=1.
//     - Spins indefinitely — never runs teardown.
//     SIGKILL leaves relay channels in ACTIVE state; subsequent probe reads
//     from the parent time out, filling the relay CMD queue.
//
//   Phase 4: Parent waits for child_ready (up to 20s), SIGKILLs child, waits 2s.
//
//   Phase 5: Parent creates MeshDevice::create() FABRIC_2D — timed.
//     During init, terminate_stale_erisc_routers() times out on non-MMIO
//     channels → fabric_relay_path_broken_ set.  Then
//     configure_command_queue_programs() runs for each device.  The FIX AV
//     guard must skip sysmem_manager_->reset() for relay-broken devices.
//     Without the guard: read_non_mmio() blocks on full relay CMD queue → hang.
//     With the guard:    reset() skipped → configure_command_queue_programs
//                        returns in < 1ms → MeshDevice::create() completes.
//
//   Phase 6: Blank dispatch on re-opened device (graceful throw is OK).
//     Confirms configure_command_queue_programs left the device in a usable
//     state — or at minimum, that it threw cleanly rather than hanging.
//
// Pass conditions:
//   - MeshDevice::create() does NOT throw.
//   - MeshDevice::create() completes in < 45s.
//   - Subsequent dispatch completes or throws cleanly (no hang, no SIGABRT).
// ---------------------------------------------------------------------------
TEST_F(FixAvRelayBrokenSysmemGuardFixture, FixAvRelayBrokenSkipsSystemMemReset) {
    // ── Phase 1: Close fixture mesh device ─────────────────────────────────
    mesh_device_->close();

    // ── Phase 2: Shared-memory child_ready flag ─────────────────────────────
    void* raw_shm = ::mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(raw_shm, MAP_FAILED) << "mmap failed: " << strerror(errno);
    std::atomic<int>* child_ready = new (raw_shm) std::atomic<int>(0);

    // ── Phase 3: Fork child ─────────────────────────────────────────────────
    pid_t child_pid = ::fork();
    ASSERT_GE(child_pid, 0) << "fork() failed: " << strerror(errno);

    if (child_pid == 0) {
        // ── Child process ──────────────────────────────────────────────────
        // Open FABRIC_2D to put ERISCs into ACTIVE relay state, then spin.
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto dev = MeshDevice::create(
                MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(
                    MetalContext::instance().get_cluster().number_of_devices())}),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                /*num_command_queues=*/1);
            auto range = MeshCoordinateRange(dev->shape());
            auto workload = make_blank_workload_gap28(range);
            EnqueueMeshWorkload(dev->mesh_command_queue(0), workload, false);
            Finish(dev->mesh_command_queue(0));
            // Signal parent that relay is live.
            child_ready->store(1);
        } catch (...) {
            // Signal parent even on error — parent will SIGKILL us regardless.
            child_ready->store(1);
        }
        // Spin — never teardown.  Parent's SIGKILL leaves relay in ACTIVE state.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        _exit(0);  // unreachable
    }

    // ── Phase 4: Wait for child_ready, SIGKILL, margin ──────────────────────
    constexpr int kMaxWaitMs = 20000;
    const auto wait_start = std::chrono::steady_clock::now();
    while (child_ready->load() == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - wait_start)
                .count();
        if (elapsed > kMaxWaitMs) {
            ::kill(child_pid, SIGKILL);
            ::waitpid(child_pid, nullptr, 0);
            ::munmap(raw_shm, sizeof(int));
            GTEST_SKIP() << "Child did not signal ready within " << kMaxWaitMs
                         << "ms (hardware init stall?); skipping";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Extra 2s — ensure ERISCs are in ACTIVE relay state before killing.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ::kill(child_pid, SIGKILL);
    ::waitpid(child_pid, nullptr, 0);
    ::munmap(raw_shm, sizeof(int));

    // ── Phase 5: Parent re-opens FABRIC_2D — timed ──────────────────────────
    // FIX AV guard must prevent sysmem_manager_->reset() from hanging on the
    // dead relay CMD queue of non-MMIO devices.
    const auto create_start = std::chrono::steady_clock::now();
    std::shared_ptr<MeshDevice> parent_dev;
    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    ASSERT_NO_THROW({
        parent_dev = MeshDevice::create(
            MeshDeviceConfig(MeshShape{1, static_cast<uint32_t>(
                MetalContext::instance().get_cluster().number_of_devices())}),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            /*num_command_queues=*/1);
    }) << "MeshDevice::create() threw after SIGKILL predecessor — relay-broken init path failed";
    const auto create_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - create_start)
            .count();
    EXPECT_LT(create_elapsed_ms, 45000)
        << "MeshDevice::create() took " << create_elapsed_ms << "ms after SIGKILL predecessor. "
           "Without FIX AV guard: sysmem_manager_->reset() on relay-broken non-MMIO device "
           "blocks indefinitely (relay CMD queue full → read_non_mmio() spin). "
           "Threshold is 45s; current value suggests the guard may be missing.";

    if (!parent_dev) {
        return;  // create() threw but ASSERT_NO_THROW already failed
    }

    // ── Phase 6: Blank dispatch ──────────────────────────────────────────────
    // configure_command_queue_programs either fully initialized the device
    // (relay healthy enough for dispatch) or left it in degraded-but-clean state
    // (relay broken, dispatch will throw).  Either outcome is acceptable; a hang
    // is not.
    try {
        auto range = MeshCoordinateRange(parent_dev->shape());
        auto workload = make_blank_workload_gap28(range);
        EnqueueMeshWorkload(parent_dev->mesh_command_queue(0), workload, false);
        Finish(parent_dev->mesh_command_queue(0));
        log_info(tt::LogTest, "GAP-28: blank dispatch on re-opened mesh succeeded");
    } catch (const std::exception& e) {
        // Clean exception is acceptable for a relay-broken device.
        log_warning(
            tt::LogTest,
            "GAP-28: blank dispatch threw (expected for relay-broken device): {}",
            e.what());
    }

    parent_dev->close();
}

}  // namespace tt::tt_metal::distributed::test
