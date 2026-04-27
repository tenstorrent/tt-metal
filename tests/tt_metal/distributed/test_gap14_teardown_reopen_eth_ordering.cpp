// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-14: Covers FIX AE + FIX AF + FIX AC/AD — 3-pass ETH launch ordering
//         across teardown + re-open boundary
//
// Background:
//   mesh_device.cpp quiesce_internal() uses a 3-pass ETH launch ordering:
//
//   Pass 1a: All devices — setup fabric cores, runtime args, L1 barrier,
//            WORKER write_launch_msg.  ETH write_launch_msg deferred.
//   Pass 1b: MMIO devices only — ETH write_launch_msg (fast, direct PCIe).
//            MMIO ERISCs reach STARTED before non-MMIO peers start.
//   Pass 1c: Non-MMIO devices only — ETH write_launch_msg sequentially.
//            After each device: poll ETH channels until EDMStatus::STARTED
//            (FIX AF) before launching the next non-MMIO device.
//
//   FIX AE (#42429): defers ETH launch via defer_eth_launch=true in
//   quiesce_and_restart_fabric_workers(), separating MMIO from non-MMIO start.
//   FIX AF (#42429): wait_for_eth_cores_launched() polling loop guarantees
//   sender ERISC is past Object Setup before peer RECEIVER starts.
//   FIX AC/AD (#42429): Related fixes addressing ERISC state management and
//   channel initialization ordering that interact with the 3-pass sequencing.
//
// What GAP-1 already tests:
//   GAP-1 verifies the 3-pass ordering within quiesce cycles (2 consecutive
//   quiesce calls), confirming FIX AE + FIX AF prevent STARTED deadlock.
//
// What this test additionally verifies (the GAP-14 boundary):
//   The 3-pass ordering is NOT tested across the teardown / re-open boundary.
//   A full mesh_device.close() + MeshDevice::create() tears down and re-creates
//   all ERISC state from scratch.  After re-open, the first quiesce_internal()
//   call must apply the 3-pass ordering correctly to the freshly loaded ERISCs.
//   A regression in FIX AE/AF/AC/AD that only manifests on a cold ERISC state
//   (not a warm re-start) would be missed by GAP-1 but caught here.
//
// Strategy:
//   Perform 5 full close + re-open cycles.  Each cycle:
//     1. Dispatch a blank workload (async) to activate ERISC channels.
//     2. Time quiesce_devices() — 3-pass ordering on fresh ERISC state.
//        A STARTED deadlock would cause an indefinite stall >> 30s.
//     3. Post-quiesce blocking dispatch — confirms fabric is operational.
//     4. Time mesh_device_.close() + MeshDevice::create() — teardown + re-open.
//        Must complete in < 60s (generous budget for FABRIC_2D init cycle).
//   Final: blocking dispatch after all 5 cycles + buffer round-trip.
//
// Pass = all 5 cycles complete within time bounds, all dispatches succeed,
//        buffer round-trip data matches.
// Fail = any quiesce > 30s (STARTED deadlock), any re-open > 60s (init hang),
//        hang (watchdog kills at 600s), crash, throw, or data corruption.
//
// Topology requirement: >= 2 devices (FABRIC_2D requires ETH routing).
//   The test skips gracefully on single-chip systems where FABRIC_2D is not
//   meaningful.  Reasons at the level of device characteristics (MMIO vs
//   non-MMIO) rather than hardcoded device numbers.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <vector>

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
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Helper: create a Program with 3 blank kernels (BRISC, NCRISC, compute) on a
// single core.  This is the lightest workload that exercises the full dispatch
// path and activates ERISC channels — identical pattern to GAP-1/GAP-5.
// ---------------------------------------------------------------------------
static Program create_blank_program_gap14(const CoreRange& cores) {
    Program program;

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cores, ComputeConfig{});

    return program;
}

// ---------------------------------------------------------------------------
// Fixture: TeardownReopenEthOrderingFixture
//
// Opens a full-system FABRIC_2D mesh with a 600-second watchdog.
//
// Budget rationale: 5 full teardown + re-open cycles, each up to 60s for
// FABRIC_2D init (generous bound; T3K init is ~15s per cycle in practice).
// 5 × 60s = 300s for re-opens + 5 × 30s max quiesce = 150s + dispatch overhead
// = ~500s worst case.  600s gives a comfortable margin before SIGKILL.
//
// Skips gracefully on single-chip systems: FABRIC_2D needs ETH routing between
// at least 2 chips.  The key regression path (STARTED deadlock between peer
// ERISCs) only manifests in a multi-chip topology where non-MMIO devices are
// present.  We reason at the level of device count (>= 2), not device IDs.
// ---------------------------------------------------------------------------
class TeardownReopenEthOrderingFixture : public MeshDeviceFixtureBase {
protected:
    TeardownReopenEthOrderingFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 600000,  // 10-minute budget: 5 close+reopen cycles
          }) {}

    void SetUp() override {
        // Require >= 2 devices: FABRIC_2D and the 3-pass ETH ordering only apply
        // in topologies where ERISC channels connect chips.  A single-chip system
        // has no non-MMIO peers for the handshake ordering to matter.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "TeardownReopenEthOrderingFixture requires >= 2 devices (FABRIC_2D needs ETH routing). "
                         << "Found " << num_devices << " device(s). "
                         << "3-pass ETH launch ordering (FIX AE/AF/AC/AD) only manifests on multi-chip topologies.";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-14: TeardownReopenPreservesEthLaunchOrdering
//
// Verifies that the 3-pass ETH launch ordering (FIX AE + FIX AF + FIX AC/AD)
// works correctly across repeated full teardown + re-open boundaries.
//
// Each of the 5 cycles:
//   Step A. Dispatch a blank workload (blocking=false) — ERISC channels live.
//   Step B. quiesce_devices() — timed; assert < 30000ms.
//           On fresh (cold) ERISC state after re-open, FIX AE must defer ETH
//           launch and FIX AF must poll STARTED before the next non-MMIO
//           device launches.  A STARTED deadlock stalls indefinitely.
//   Step C. Blocking dispatch after quiesce — confirms fabric is healthy.
//   Step D. mesh_device_.close() + SetFabricConfig + MeshDevice::create().
//           Timed; assert < 60000ms.
//
// Final:
//   Step E. Blocking dispatch on the device open after cycle 5.
//   Step F. Buffer round-trip — detects DRAM corruption from stale ERISC NOC
//           writes that can be introduced by mis-ordered 3-pass restart on
//           cold ERISC state.
// ---------------------------------------------------------------------------
TEST_F(TeardownReopenEthOrderingFixture, TeardownReopenPreservesEthLaunchOrdering) {
    // Time limits:
    //   30s for quiesce: STARTED deadlock would stall indefinitely;
    //                    30s catches it before the 600s watchdog with a clear error.
    //   60s for re-open: FABRIC_2D init is ~15s on T3K; 60s gives 4× margin.
    constexpr int64_t kMaxQuiesceMs = 30000;
    constexpr int64_t kMaxReopenMs  = 60000;
    constexpr int kNumCycles        = 5;

    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};

    // Snapshot the mesh shape and config fields we need for re-open.
    // We capture these before any close() so the fixture's config_ is still valid.
    const auto mesh_shape = mesh_device_->shape();

    for (int cycle = 1; cycle <= kNumCycles; cycle++) {
        log_info(tt::LogTest,
            "[GAP-14] === Cycle {}/{} START ===", cycle, kNumCycles);

        auto device_range = MeshCoordinateRange(mesh_device_->shape());

        // ------------------------------------------------------------------
        // Step A: async dispatch — puts ERISC channels into active state.
        // We leave the workload in-flight so that quiesce_internal() encounters
        // live ERISC channels when it begins the 3-pass ETH launch restart.
        // This is the scenario most likely to expose a cold-ERISC ordering bug
        // on the first cycle and a warm-re-open bug on subsequent cycles.
        // ------------------------------------------------------------------
        log_info(tt::LogTest,
            "[GAP-14] Cycle {}: Step A — async dispatch (ERISC channels live)", cycle);
        {
            auto program = create_blank_program_gap14(cores);
            auto workload = MeshWorkload();
            workload.add_program(device_range, std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            ASSERT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/false))
                << "[GAP-14] Cycle " << cycle << " Step A: async dispatch threw";
        }

        // ------------------------------------------------------------------
        // Step B: timed quiesce_devices() — the critical 3-pass ETH ordering
        // check.  A STARTED deadlock would cause the test to stall here and
        // eventually be killed by the 600s watchdog.  The 30s bound provides
        // an early, actionable signal.
        //
        // On cycle 1 this exercises FIX AE/AF on a fresh ERISC state (cold
        // after MeshDevice::create() in fixture SetUp).  On cycles 2-5 this
        // exercises FIX AE/AF on ERISC state that was torn down and re-created
        // by the previous cycle's re-open — the novel GAP-14 boundary.
        // ------------------------------------------------------------------
        log_info(tt::LogTest,
            "[GAP-14] Cycle {}: Step B — calling quiesce_devices() (FIX AE + FIX AF)", cycle);

        const auto t_quiesce_start = std::chrono::steady_clock::now();

        EXPECT_NO_THROW(mesh_device_->quiesce_devices())
            << "[GAP-14] Cycle " << cycle << " Step B: quiesce_devices() threw";

        const auto elapsed_quiesce_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_quiesce_start)
                .count();

        log_info(tt::LogTest,
            "[GAP-14] Cycle {}: Step B — quiesce_devices() in {}ms (limit {}ms)",
            cycle, elapsed_quiesce_ms, kMaxQuiesceMs);

        EXPECT_LT(elapsed_quiesce_ms, kMaxQuiesceMs)
            << "[GAP-14] Cycle " << cycle << " Step B: quiesce_devices() exceeded " << kMaxQuiesceMs
            << "ms (" << elapsed_quiesce_ms << "ms). "
            << "STARTED deadlock (FIX AE/AF regression) stalls indefinitely; "
            << "30s bound catches it on cold ERISC state after teardown + re-open. "
            << "FIX AC/AD interaction may also contribute if ERISC state is not "
            << "fully re-initialized before the 3-pass ETH launch begins.";

        // ------------------------------------------------------------------
        // Step C: blocking dispatch after quiesce — verifies 3-pass restart
        // left fabric in an operational state.  If Pass 1b/1c ordering is
        // wrong, ERISC state is corrupted and this blocking dispatch hangs in
        // completion_queue_wait_front.
        // ------------------------------------------------------------------
        log_info(tt::LogTest,
            "[GAP-14] Cycle {}: Step C — post-quiesce blocking dispatch (fabric health check)", cycle);
        {
            auto program = create_blank_program_gap14(cores);
            auto workload = MeshWorkload();
            workload.add_program(device_range, std::move(program));
            auto& cq = mesh_device_->mesh_command_queue();
            EXPECT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
                << "[GAP-14] Cycle " << cycle
                << " Step C: post-quiesce blocking dispatch failed — "
                << "3-pass ETH restart left fabric unhealthy after teardown + re-open boundary";
        }

        // ------------------------------------------------------------------
        // Step D: timed full teardown + re-open.
        //
        // close() + reset() tears down all ERISC firmware, DMA channels, and
        // fabric control-plane state.  SetFabricConfig re-arms the fabric
        // configuration (close()/post_teardown() resets it to DISABLED).
        // MeshDevice::create() re-initializes everything from scratch.
        //
        // This is the boundary that GAP-1 does NOT test: after create(), the
        // first call to quiesce_internal() (in cycle N+1 Step B) will see
        // fresh ERISC state and must apply the 3-pass ordering from scratch.
        //
        // Skip the teardown + re-open on the last cycle to leave mesh_device_
        // open for the final Steps E and F.
        // ------------------------------------------------------------------
        if (cycle < kNumCycles) {
            log_info(tt::LogTest,
                "[GAP-14] Cycle {}: Step D — timing teardown + re-open (FIX AP interaction)", cycle);

            const auto t_reopen_start = std::chrono::steady_clock::now();

            mesh_device_->close();
            mesh_device_.reset();

            // SetFabricConfig must be called before MeshDevice::create because
            // the previous close() / post_teardown() already reset the fabric
            // config to DISABLED.
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

            ASSERT_NO_THROW(
                mesh_device_ = MeshDevice::create(
                    MeshDeviceConfig(mesh_shape),
                    config_.l1_small_size,
                    config_.trace_region_size,
                    config_.num_cqs,
                    DispatchCoreConfig{},
                    {},
                    config_.worker_l1_size))
                << "[GAP-14] Cycle " << cycle
                << " Step D: MeshDevice::create() threw — "
                << "terminate_stale_erisc_routers() may be attempting relay operations on "
                << "non-MMIO devices with broken relay state (FIX AP regression).";

            const auto elapsed_reopen_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_reopen_start)
                    .count();

            log_info(tt::LogTest,
                "[GAP-14] Cycle {}: Step D — teardown + re-open in {}ms (limit {}ms)",
                cycle, elapsed_reopen_ms, kMaxReopenMs);

            EXPECT_LT(elapsed_reopen_ms, kMaxReopenMs)
                << "[GAP-14] Cycle " << cycle << " Step D: teardown + re-open exceeded "
                << kMaxReopenMs << "ms (" << elapsed_reopen_ms << "ms). "
                << "This may indicate FABRIC_2D init hang or stale ERISC state not "
                << "cleaned up correctly (FIX AC/AD/AP interaction).";

            log_info(tt::LogTest,
                "[GAP-14] === Cycle {}/{} COMPLETE (teardown + re-open boundary crossed) ===",
                cycle, kNumCycles);
        } else {
            log_info(tt::LogTest,
                "[GAP-14] === Cycle {}/{} COMPLETE (last cycle — device stays open for final checks) ===",
                cycle, kNumCycles);
        }
    }

    // ------------------------------------------------------------------
    // Step E: final blocking dispatch on the device open after all 5 cycles.
    // This dispatch uses the device state left by cycle 5 (after its Step B/C
    // quiesce and post-quiesce dispatch, no teardown).  Confirms the fabric
    // remains operational across the full test run.
    // ------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-14] Final Step E: blocking dispatch after {} teardown + re-open cycles",
        kNumCycles - 1);
    {
        auto final_range = MeshCoordinateRange(mesh_device_->shape());
        auto program = create_blank_program_gap14(cores);
        auto workload = MeshWorkload();
        workload.add_program(final_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EXPECT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
            << "[GAP-14] Final Step E: blocking dispatch failed after "
            << (kNumCycles - 1) << " teardown + re-open cycles";
    }

    // ------------------------------------------------------------------
    // Step F: buffer round-trip — detects DRAM corruption from stale ERISC
    // NOC writes that could be introduced by mis-ordered 3-pass restart on
    // cold ERISC state across the teardown + re-open boundary.
    //
    // Non-MMIO devices connect via ETH relay; we reason at the level of mesh
    // shape (rows × cols) rather than hardcoding device IDs.
    // ------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-14] Final Step F: buffer round-trip (DRAM corruption check after {} cycles)",
        kNumCycles - 1);
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};

        // Use the current mesh shape — may differ from fixture shape if a smaller
        // shape was explicitly configured.  Reasoning at the level of shape
        // characteristics avoids hardcoding device counts.
        const auto final_shape = mesh_device_->shape();
        const uint32_t shape_rows = final_shape[0];
        const uint32_t shape_cols = final_shape[1];
        auto global_shape = Shape2D{shape_rows, shape_cols};
        auto shard_shape = Shape2D{1, 1};
        auto dist_config = ShardedBufferConfig{
            .global_size = static_cast<size_t>(shape_rows) * shape_cols * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = shard_shape};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());

        const size_t n_words =
            page_size / sizeof(uint32_t) * shape_rows * shape_cols;
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            // Pattern encodes GAP-14 tag (AE0AF) and index for easy debugging.
            src[i] = static_cast<uint32_t>(0xAE0AF000 | (i & 0xFFFF));
        }

        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);

        ASSERT_EQ(dst.size(), src.size())
            << "[GAP-14] Step F: buffer size mismatch after "
            << (kNumCycles - 1) << " teardown + re-open cycles";

        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[GAP-14] Step F: data corruption at index " << i
                << " after " << (kNumCycles - 1) << " teardown + re-open cycles. "
                << "Stale ERISC NOC write from mis-ordered 3-pass ETH launch on "
                << "cold ERISC state across the teardown + re-open boundary? "
                << "(FIX AE/AF/AC/AD regression)";
        }

        log_info(tt::LogTest,
            "[GAP-14] Step F: buffer round-trip clean — "
            "FIX AE + FIX AF + FIX AC/AD 3-pass ETH launch ordering "
            "verified across {} teardown + re-open boundaries",
            kNumCycles - 1);
    }

    log_info(tt::LogTest,
        "[GAP-14] TeardownReopenPreservesEthLaunchOrdering PASSED — "
        "3-pass ETH launch ordering (FIX AE + FIX AF + FIX AC/AD) works correctly "
        "across {} full teardown + re-open cycles. "
        "STARTED deadlocks would have caused hangs >> 30s per quiesce call.",
        kNumCycles - 1);
}

}  // namespace tt::tt_metal::distributed::test
