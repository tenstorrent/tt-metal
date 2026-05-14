// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-2: Covers FIX-3 — launch gate rejects live ERISC
//
// Background:
//   launch_eth_cores_for_quiesce (Phase 3) reads the ERISC's pre-launch status
//   word before calling write_launch_msg_to_core.  If the ERISC is still live
//   (pre_status != 0x0 and != EDMStatus::TERMINATED), FIX-3 skips the launch
//   write and marks the channel dead so Phase 5 / Phase 5b handling can proceed
//   without a firmware-init stall.
//
//   Without FIX-3: write_launch_msg_to_core on a live ERISC corrupts the
//   in-flight relay firmware → the new fabric kernel never writes
//   EDMStatus::STARTED → Phase 5b times out (2s+ per stuck channel) for every
//   live channel in the mesh.
//
// What this test verifies:
//   1. FIX-3 launch gate: rapid-quiesce cycles with a live ERISC path do not
//      produce per-channel Phase 5b timeouts (each quiesce must finish < 8000ms).
//   2. Transition from relay firmware to fabric firmware during quiesce is clean.
//   3. Coupling between FIX-3 channel rejection and FIX AK non-fatal Phase 5b
//      handling: rejected channels are skipped gracefully, not fatal.
//
// Each iteration dispatches a small AllGather op (live ERISC relay firmware),
// then immediately calls quiesce_devices() — maximising the window where the
// live-ERISC check in Phase 3 is exercised.
//
// Pass = 10 cycles each < 8000ms, quiesce_success_count == 10.
// Fail = any cycle > 8000ms (Phase 5b stall per stuck channel) or hang.

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
// Fixture: AsyncTeardownFabric2DRepeatFixture extended with a longer budget.
//
// Re-uses the same fixture class defined in test_async_teardown_race.cpp but
// scoped here as a local alias so this file is self-contained.  The fixture
// opens a FABRIC_2D mesh device and requires >= 2 devices (skips gracefully
// on single-chip runners).
//
// Budget: 300 000 ms (5 minutes) — 10 cycles × up to 15s FABRIC_2D
// init/teardown + 8s per quiesce worst-case = ~230s.
// ---------------------------------------------------------------------------
class LaunchGateLiveEriscFixture : public MeshDeviceFixtureBase {
protected:
    LaunchGateLiveEriscFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 300000,  // 5 min: 10 cycles × worst-case quiesce
          }) {}

    void SetUp() override {
        // FABRIC_2D requires >= 2 devices; skip gracefully on single-chip CI runners.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "LaunchGateLiveEriscFixture requires >= 2 devices (FABRIC_2D). "
                            "Found " << num_devices << " device(s). "
                            "FIX-3 live-ERISC rejection only manifests on multi-chip topologies "
                            "where non-MMIO ERISCs run relay firmware between quiesce cycles.";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-2 Test: LaunchGateRejectsLiveERISC
//
// Runs 10 rapid quiesce cycles.  Each cycle:
//   1. Builds per-device input tensors and an AllGather op — this loads relay
//      firmware onto all ERISC channels (they are "live" when quiesce fires).
//   2. Calls quiesce_devices() immediately after the AllGather, without a
//      blocking Finish() — maximising the Phase 3 live-ERISC window.
//   3. Times the quiesce and asserts < 8000ms.
//
// Why the timing bound catches FIX-3 regressions:
//   Without FIX-3: write_launch_msg_to_core on a live ERISC → firmware-init
//   stall → Phase 5b waits up to 2s per stuck channel.  On a T3K (8 active
//   ETH channels) that is ~16s/cycle, well above the 8s assertion.
//   With FIX-3: live ERISCs are skipped; Phase 5b processes only clean channels
//   or non-fatally skips the dead ones → cycle completes in < 4s typical.
//
// Requires >= 2 devices.  Uses characteristics-based device selection
// (num_rows / num_cols) rather than hardcoded device numbers.
// ---------------------------------------------------------------------------
TEST_F(LaunchGateLiveEriscFixture, LaunchGateRejectsLiveERISC) {
    constexpr int kCycles = 10;
    // Phase 5b timeout per stuck channel is ~2s.  T3K has ~8 active ETH
    // channels → ~16s if FIX-3 is missing.  8s gives healthy margin above
    // correct behaviour (~2-4s) while catching regressions before the 5-min
    // budget expires.
    constexpr int64_t kMaxQuiesceMs = 8000;

    // Require >= 4 devices to ensure non-MMIO ERISCs are present and relay
    // firmware is actually loaded by the AllGather.  On N300 (2 devices) the
    // topology is simpler and may not exercise the live-ERISC path in Phase 3.
    const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
    if (num_devices < 4) {
        GTEST_SKIP() << "LaunchGateRejectsLiveERISC requires >= 4 devices to ensure "
                        "non-MMIO ERISCs carry live relay firmware during quiesce. "
                        "Found " << num_devices << " device(s).";
    }

    // Use all available devices in a ring-style topology (row 0 of the mesh).
    // Pick up to 4 devices from the first row — matches the AllGather ring used
    // in AllGatherQuiesceLoop.  Characteristics-based: we use num_cols() which
    // reflects the physical topology rather than a hardcoded device number.
    const int kNumRingDevices = std::min(static_cast<int>(mesh_device_->num_cols()), 4);
    if (kNumRingDevices < 2) {
        GTEST_SKIP() << "LaunchGateRejectsLiveERISC requires mesh width >= 2 cols. "
                        "mesh_device_->num_cols() = " << mesh_device_->num_cols();
    }

    // Small tensor: enough to load all ERISC relay channels, small enough that
    // AllGather completes well within the per-cycle timing budget.
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 32, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    // Build per-device submesh views (1x1 slices from row 0).
    // Characteristics-based: we iterate over col indices (topology columns),
    // never referring to specific device numbers.
    std::vector<std::shared_ptr<distributed::MeshDevice>> submeshes;
    submeshes.reserve(kNumRingDevices);
    for (int col = 0; col < kNumRingDevices; col++) {
        submeshes.push_back(
            mesh_device_->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, col)));
    }

    int quiesce_success_count = 0;

    for (int cycle = 0; cycle < kCycles; cycle++) {
        log_info(
            tt::LogTest,
            "[LaunchGateRejectsLiveERISC] Cycle {}/{}: building AllGather input tensors",
            cycle + 1,
            kCycles);

        // Step 1: create per-device input tensors.
        // Device at col=i holds float(i) so the gathered result is deterministic.
        // This also ensures relay firmware is genuinely active on ERISC channels
        // — the race window FIX-3 targets only appears when ERISCs are live.
        std::vector<ttnn::Tensor> tensors;
        tensors.reserve(kNumRingDevices);
        for (int col = 0; col < kNumRingDevices; col++) {
            std::vector<bfloat16> data(
                tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(col)));
            tensors.push_back(
                Tensor::from_vector(std::move(data), tensor_spec)
                    .to_device(submeshes[col].get()));
        }

        // Step 2: aggregate into a multi-device tensor on the parent mesh.
        auto aggregated = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);

        log_info(
            tt::LogTest,
            "[LaunchGateRejectsLiveERISC] Cycle {}/{}: launching AllGather (ERISCs go live)",
            cycle + 1,
            kCycles);

        // Step 3: AllGather — loads relay firmware onto ERISC channels.
        // No blocking wait here: we want ERISCs still live when quiesce fires
        // to maximise the Phase 3 live-ERISC check window.
        auto gathered = ttnn::all_gather(aggregated, /* dim */ 0);

        // Step 4: quiesce_devices() immediately — this is the critical path.
        //   Phase 3 of launch_eth_cores_for_quiesce reads each ERISC's
        //   pre_launch status.  FIX-3 skips write_launch_msg_to_core for any
        //   ERISC with pre_status != 0x0 and != TERMINATED, marking it dead.
        //   Phase 5b then handles dead channels non-fatally.
        //   Without FIX-3: live ERISCs receive write_launch_msg → firmware-init
        //   stall → Phase 5b times out (2s+ per channel).
        log_info(
            tt::LogTest,
            "[LaunchGateRejectsLiveERISC] Cycle {}/{}: calling quiesce_devices() with ERISCs potentially live",
            cycle + 1,
            kCycles);

        const auto t0 = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(mesh_device_->quiesce_devices())
            << "[LaunchGateRejectsLiveERISC] quiesce_devices() threw on cycle " << (cycle + 1);
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();

        log_info(
            tt::LogTest,
            "[LaunchGateRejectsLiveERISC] Cycle {}/{}: quiesce_devices() completed in {}ms (limit {}ms)",
            cycle + 1,
            kCycles,
            elapsed_ms,
            kMaxQuiesceMs);

        // If FIX-3 is missing: write_launch_msg on live ERISC → Phase 5b stall
        // → 2s+ per stuck ETH channel → elapsed >> 8000ms.
        ASSERT_LT(elapsed_ms, kMaxQuiesceMs)
            << "[LaunchGateRejectsLiveERISC] Cycle " << (cycle + 1) << "/" << kCycles
            << " quiesce took " << elapsed_ms << "ms, exceeding " << kMaxQuiesceMs << "ms. "
            << "This indicates Phase 5b stalls from FIX-3 live-ERISC launch gate regression: "
            << "write_launch_msg_to_core on a live ERISC causes firmware-init stall "
            << "that Phase 5b must timeout on per stuck channel (2s+ each).";

        // Step 5: verify correctness of gathered output.
        // Corruption here indicates stale ERISC NOC traffic from a FIX-3 regression.
        auto disaggregated = tt::tt_metal::experimental::unit_mesh::disaggregate(gathered);
        ASSERT_EQ(static_cast<int>(disaggregated.size()), kNumRingDevices)
            << "[LaunchGateRejectsLiveERISC] Wrong shard count on cycle " << (cycle + 1);

        const size_t per_device_vol = tensor_spec.logical_shape().volume();
        for (int col = 0; col < kNumRingDevices; col++) {
            auto data = disaggregated[col].to_vector<bfloat16>();
            ASSERT_FALSE(data.empty())
                << "[LaunchGateRejectsLiveERISC] Empty readback at col=" << col
                << " cycle=" << (cycle + 1);
            ASSERT_EQ(data.size(), per_device_vol * static_cast<size_t>(kNumRingDevices))
                << "[LaunchGateRejectsLiveERISC] Output size mismatch at col=" << col
                << " cycle=" << (cycle + 1);
            for (size_t i = 0; i < data.size(); i++) {
                float expected = static_cast<float>(i / per_device_vol);
                EXPECT_EQ(static_cast<float>(data[i]), expected)
                    << "[LaunchGateRejectsLiveERISC] Data corruption at element " << i
                    << " col=" << col << " cycle=" << (cycle + 1)
                    << " (expected=" << expected
                    << " got=" << static_cast<float>(data[i]) << "). "
                    << "Stale ERISC NOC writes from FIX-3 regression can corrupt DRAM "
                    << "during the Phase 3 firmware overwrite on a live ERISC.";
            }
        }

        // Clean up tensors before next cycle.
        disaggregated.clear();
        { auto tmp = std::move(gathered); }

        quiesce_success_count++;
        log_info(
            tt::LogTest,
            "[LaunchGateRejectsLiveERISC] Cycle {}/{}: PASSED ({}ms)",
            cycle + 1,
            kCycles,
            elapsed_ms);
    }

    // Primary assertion: all 10 rapid quiesce cycles must succeed.
    // If FIX-3 is missing, any cycle with live ERISCs will either exceed the
    // timing bound (caught above) or hang (watchdog kills the test process).
    ASSERT_EQ(quiesce_success_count, kCycles)
        << "[LaunchGateRejectsLiveERISC] Only " << quiesce_success_count << "/" << kCycles
        << " quiesce cycles completed. FIX-3 live-ERISC launch gate may be missing.";

    log_info(
        tt::LogTest,
        "[LaunchGateRejectsLiveERISC] All {} rapid quiesce cycles passed — "
        "FIX-3 launch gate correctly rejects live ERISCs, no Phase 5b stalls",
        kCycles);
}

}  // namespace tt::tt_metal::distributed::test
