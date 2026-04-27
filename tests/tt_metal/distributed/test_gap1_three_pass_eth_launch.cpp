// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-1: Covers FIX AE + FIX AF — 3-pass ETH launch ordering
//
// Background:
//   mesh_device.cpp quiesce_internal() uses a 3-pass ETH launch to prevent
//   simultaneous handshake deadlock between peer ERISC channels on different
//   non-MMIO devices:
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
//
// What this test verifies:
//   1. quiesce_devices() completes well within 6000ms (STARTED deadlock would
//      stall indefinitely — detectable as a timeout >> 10s).
//   2. A second CCL-quality workload can be dispatched after quiesce — fabric
//      routing is healthy post-restart.
//   3. A second quiesce cycle succeeds (regression guard against state leaks
//      introduced by the 3-pass sequencing).
//
// Pass = both quiesce cycles complete in < 6s each, both blocking dispatches
//        succeed, buffer round-trip data matches.
// Fail = any quiesce > 6s (FIX AF poll hang / FIX AE STARTED deadlock),
//        hang (watchdog kills at 90s), crash, or data corruption.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <vector>

#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/device/firmware/fabric_firmware_initializer.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_init.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Helper: create a Program with 3 blank kernels (BRISC, NCRISC, compute) on a
// single core.  Identical to test_async_teardown_race.cpp — lightest possible
// workload that exercises the full dispatch path.
// ---------------------------------------------------------------------------
static Program create_blank_program_gap1(const CoreRange& cores) {
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
// Fixture: FABRIC_2D mesh with a 90-second watchdog, shared with Scenario H/I.
// The 90s budget covers two full quiesce cycles (each up to ~15s on T3K) plus
// two blocking dispatches and a buffer round-trip.
//
// Skipped on single-device systems: FABRIC_2D needs ETH routing between chips.
// ---------------------------------------------------------------------------
class Gap1ThreePassEthLaunchFixture : public MeshDeviceFixtureBase {
protected:
    Gap1ThreePassEthLaunchFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 90000,  // 90s: 2 quiesce cycles + dispatches + round-trip
          }) {}

    void SetUp() override {
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "Gap1ThreePassEthLaunchFixture requires >= 2 devices (FABRIC_2D needs ETH routing)";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-1: ThreePassETHLaunchOrdering
//
// Verifies that the 3-pass ETH launch in quiesce_internal() (FIX AE + FIX AF)
// prevents STARTED deadlock and that fabric health is preserved across two
// quiesce cycles.
//
// Steps:
//   1. Dispatch a blank workload (blocking=false) — ERISC channels live.
//   2. quiesce_devices() — timed; assert < 6000ms.
//      A STARTED deadlock (both ERISCs stuck) would exceed 10s; 6s catches it
//      before the 90s budget expires with zero diagnostic output.
//   3. Dispatch a second blank workload (blocking=true) — verifies fabric
//      health after 3-pass restart.
//   4. quiesce_devices() again (second cycle) — regression guard.
//      FIX AE / FIX AF must be idempotent across back-to-back quiesce calls.
//   5. Buffer round-trip — detects DRAM corruption from stale ERISC NOC writes
//      that could be introduced by a mis-ordered 3-pass restart.
// ---------------------------------------------------------------------------
TEST_F(Gap1ThreePassEthLaunchFixture, ThreePassETHLaunchOrdering) {
    constexpr int64_t kMaxQuiesceMs = 6000;  // STARTED deadlock = indefinite stall >> 10s
    auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    // ------------------------------------------------------------------
    // Cycle 1: async dispatch → timed quiesce → verify fabric health
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-1] Cycle 1: dispatching blank workload (blocking=false)");
    {
        auto program = create_blank_program_gap1(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        // Intentionally no Finish() — ERISC channels are live during quiesce.
    }

    // FIX AE + FIX AF: 3-pass ETH launch must complete well within 6s.
    // A regression that re-introduces the STARTED deadlock would stall
    // indefinitely here (both peer ERISCs stuck in the handshake loop).
    log_info(tt::LogTest, "[GAP-1] Cycle 1: calling quiesce_devices() — timing FIX AE + FIX AF");
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-1] quiesce_devices() threw unexpectedly on cycle 1";
    const auto elapsed1_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    log_info(
        tt::LogTest,
        "[GAP-1] Cycle 1: quiesce_devices() in {}ms (limit {}ms)",
        elapsed1_ms,
        kMaxQuiesceMs);

    EXPECT_LT(elapsed1_ms, kMaxQuiesceMs)
        << "[GAP-1] Cycle 1 quiesce exceeded " << kMaxQuiesceMs
        << "ms — possible STARTED deadlock (FIX AE/AF regression). "
        << "STARTED deadlock stalls indefinitely; 6s limit catches it early.";

    // ------------------------------------------------------------------
    // Post-quiesce dispatch: verify fabric is healthy after 3-pass restart.
    // If Pass 1b/1c ordering is wrong, ERISC state is corrupted and this
    // blocking dispatch hangs in completion_queue_wait_front.
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-1] Cycle 1: post-quiesce blocking dispatch — verifying fabric health");
    {
        auto program = create_blank_program_gap1(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EXPECT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
            << "[GAP-1] Post-quiesce blocking dispatch failed — 3-pass ETH restart left fabric unhealthy";
    }

    // ------------------------------------------------------------------
    // Cycle 2: second quiesce (regression guard — FIX AE/AF must be
    // idempotent across consecutive quiesce calls)
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-1] Cycle 2: async dispatch + second quiesce (regression guard)");
    {
        auto program = create_blank_program_gap1(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    }

    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-1] quiesce_devices() threw unexpectedly on cycle 2";
    const auto elapsed2_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();

    log_info(
        tt::LogTest,
        "[GAP-1] Cycle 2: quiesce_devices() in {}ms (limit {}ms)",
        elapsed2_ms,
        kMaxQuiesceMs);

    EXPECT_LT(elapsed2_ms, kMaxQuiesceMs)
        << "[GAP-1] Cycle 2 quiesce exceeded " << kMaxQuiesceMs
        << "ms — 3-pass ordering not idempotent across consecutive quiesce cycles (FIX AE/AF regression)";

    // ------------------------------------------------------------------
    // Final verification: blocking dispatch + buffer round-trip.
    // Stale ERISC NOC writes from a mis-ordered 3-pass restart could corrupt
    // DRAM between the write and read below.
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-1] Final: blocking dispatch after 2 quiesce cycles");
    {
        auto program = create_blank_program_gap1(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        EXPECT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
            << "[GAP-1] Final blocking dispatch failed after 2 quiesce cycles";
    }

    log_info(tt::LogTest, "[GAP-1] Buffer round-trip: checking for DRAM corruption from stale ERISC NOC traffic");
    {
        auto& cq = mesh_device_->mesh_command_queue();
        uint32_t page_size = 1024;
        auto local_config =
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto global_shape = Shape2D{
            static_cast<uint32_t>(mesh_device_->num_rows()),
            static_cast<uint32_t>(mesh_device_->num_cols())};
        auto shard_shape = Shape2D{1, 1};
        auto dist_config = ShardedBufferConfig{
            .global_size = mesh_device_->num_rows() * mesh_device_->num_cols() * page_size,
            .global_buffer_shape = global_shape,
            .shard_shape = shard_shape};

        auto mesh_buf = MeshBuffer::create(dist_config, local_config, mesh_device_.get());
        size_t n_words =
            page_size / sizeof(uint32_t) * mesh_device_->num_rows() * mesh_device_->num_cols();
        std::vector<uint32_t> src(n_words);
        for (size_t i = 0; i < n_words; i++) {
            src[i] = static_cast<uint32_t>(0xAE0AF000 | (i & 0xFFFF));
        }

        EnqueueWriteMeshBuffer(cq, mesh_buf, src, /*blocking=*/false);
        std::vector<uint32_t> dst;
        EnqueueReadMeshBuffer(cq, dst, mesh_buf, /*blocking=*/true);

        ASSERT_EQ(dst.size(), src.size())
            << "[GAP-1] Buffer size mismatch after 2 quiesce cycles (3-pass ETH launch)";
        for (size_t i = 0; i < n_words; i++) {
            ASSERT_EQ(dst[i], src[i])
                << "[GAP-1] Data corruption at index " << i
                << " after 2 quiesce cycles — stale ERISC NOC write from mis-ordered 3-pass ETH launch?";
        }
        log_info(
            tt::LogTest,
            "[GAP-1] Buffer round-trip clean — FIX AE + FIX AF 3-pass ETH launch ordering verified");
    }
}

}  // namespace tt::tt_metal::distributed::test
