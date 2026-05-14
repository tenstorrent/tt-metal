// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-10: Covers FIX AM — fabric_channels_not_ready_for_traffic_ lifecycle
//
// Background:
//   FIX AM (#42429): when Phase 5b in wait_for_fabric_workers_ready() detects
//   ETH channels stuck below READY_FOR_TRAFFIC (partial-mesh scenario), it sets
//   fabric_channels_not_ready_for_traffic_=true on the affected device so that
//   callers can distinguish "partial-mesh peer didn't respond" from "relay path
//   broken".
//
//   FIX AM also adds a clear of the flag at the top of configure_fabric()
//   (device.cpp ~line 418) so that a subsequent successful re-initialization
//   resets the flag to false.  Without this clear, a quiesce that sets the flag
//   followed by a full-mesh re-open would leave the flag stale: every subsequent
//   AllGather would spuriously GTEST_SKIP even though the fabric is healthy.
//
// What this test verifies:
//   1. After a partial-mesh quiesce (which sets the flag on mesh-edge devices),
//      the flag IS set on at least one device (FIX AM set-path still active).
//   2. After closing the partial mesh and re-opening a full mesh, the flag is
//      CLEARED on every device (FIX AM clear-path, the novel invariant).
//   3. AllGather on the full mesh succeeds — no spurious GTEST_SKIP from a
//      stale channels_not_ready flag.
//
// Gap vs. GAP-4:
//   GAP-4 opens one partial mesh and checks the flag IS set.
//   GAP-10 additionally re-opens a full mesh and checks the flag IS CLEARED,
//   verifying the clear at the top of configure_fabric() is not regressed.
//
// Topology requirement: >= 4 devices (T3K or larger).
//   - Phase 1 uses a 1x4 partial mesh (same as GAP-4) to reliably trigger
//     out-of-mesh ETH channel stalls and set the flag.
//   - Phase 2 re-opens the full SystemMesh to exercise configure_fabric()
//     clearing the flag.

#include <gtest/gtest.h>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture: ChannelsNotReadyLifecycleFixture
//
// Opens a 1x4 partial mesh on T3K (or larger) with FABRIC_2D active.
// Skips on systems with fewer than 4 devices.
//
// Budget: 120s — two FABRIC_2D init cycles (~15s each) + AllGather overhead.
// ---------------------------------------------------------------------------
class ChannelsNotReadyLifecycleFixture : public MeshDeviceFixtureBase {
protected:
    ChannelsNotReadyLifecycleFixture()
        : MeshDeviceFixtureBase(Config{
              .mesh_shape = MeshShape{1, 4},  // partial mesh — same topology as GAP-4
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,  // 2-minute budget: 2x FABRIC_2D init + AllGather
          }) {}

    void SetUp() override {
        // Require >= 4 devices so we can open a 1x4 partial mesh and have
        // out-of-mesh ETH peers that never complete the handshake.  On a
        // 4-device T3K this opens all 4 devices; on an 8-device system it
        // opens the first 4, leaving 4 as out-of-mesh peers.  Either way
        // Phase 5b will encounter out-of-mesh channels and FIX AM will set
        // fabric_channels_not_ready_for_traffic_ on the edge devices.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 4) {
            GTEST_SKIP() << "ChannelsNotReadyLifecycleFixture requires >= 4 devices (T3K topology). "
                         << "Found " << num_devices << " device(s). "
                         << "Partial-mesh out-of-mesh ETH peer behaviour only manifests on 4+ device systems.";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Test: FlagIsSetAfterPartialMeshQuiesceThenClearedOnFullMeshReopen
//
// Steps:
//   Phase 1. AllGather within the 1x4 partial mesh — exercises in-mesh ETH
//            channels so ERISC firmware is live and the handshake completes
//            for in-mesh channel pairs.
//   Phase 2. quiesce_devices() — mesh-edge ETH channels face out-of-mesh peers
//            whose ERISC is stuck below READY_FOR_TRAFFIC.  FIX AK returns
//            non-fatally; FIX AM sets fabric_channels_not_ready_for_traffic_
//            on the affected devices.
//   Phase 3. Assert flag IS set on >= 1 device (FIX AM set-path is active).
//   Phase 4. Close the partial mesh device entirely (close + reset).
//   Phase 5. Re-open a FULL-mesh MeshDevice (all available devices) with
//            FABRIC_2D.  configure_fabric() runs on every device, and the
//            FIX AM clear at the top of configure_fabric() must reset the flag.
//   Phase 6. Assert flag is CLEARED on every device (FIX AM clear-path).
//            Failure here means the clear at configure_fabric() regressed and
//            stale flag will cause spurious GTEST_SKIP on all subsequent tests.
//   Phase 7. AllGather on the full mesh — must succeed (no GTEST_SKIP from a
//            stale channels_not_ready flag, and no data corruption).
//
// Pass = flag set after partial quiesce AND flag cleared after full-mesh
//        reopen AND AllGather completes without error.
// Fail = flag never set (FIX AM set-path regressed), flag still set after
//        reopen (FIX AM clear-path regressed), or AllGather skipped/crashed.
// ---------------------------------------------------------------------------
TEST_F(ChannelsNotReadyLifecycleFixture, FlagIsSetAfterPartialMeshQuiesceThenClearedOnFullMeshReopen) {
    // Small tensor: exercises all in-mesh ERISC channels while completing
    // quickly within the test budget.
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 32, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    // Build per-device submesh views for the 1x4 partial mesh.
    // Reasoning at the level of characteristics, not device numbers:
    // each column in the 1x4 mesh gets a 1x1 submesh view at its coordinate.
    constexpr int kPartialMeshWidth = 4;
    std::vector<std::shared_ptr<distributed::MeshDevice>> partial_submeshes;
    for (int col = 0; col < kPartialMeshWidth; col++) {
        partial_submeshes.push_back(
            mesh_device_->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, col)));
    }

    // -----------------------------------------------------------------------
    // Phase 1: AllGather on the 1x4 partial mesh — exercise in-mesh channels.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-10] Phase 1: building input tensors for AllGather on 1x4 partial mesh");

    // Device i holds a tensor filled with float(i) so the gathered result
    // along dim=0 is [0, 1, 2, 3] per element — deterministic, detects corruption.
    std::vector<ttnn::Tensor> partial_tensors;
    for (int dev_idx = 0; dev_idx < kPartialMeshWidth; dev_idx++) {
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        partial_tensors.push_back(
            Tensor::from_vector(std::move(data), tensor_spec)
                .to_device(partial_submeshes[dev_idx].get()));
    }

    auto partial_aggregated = tt::tt_metal::experimental::unit_mesh::aggregate(partial_tensors);

    log_info(tt::LogTest, "[GAP-10] Phase 1: launching AllGather on 1x4 partial mesh");
    auto partial_gathered = ttnn::all_gather(partial_aggregated, /* dim */ 0);
    log_info(tt::LogTest, "[GAP-10] Phase 1: AllGather on partial mesh complete");

    // -----------------------------------------------------------------------
    // Phase 2: quiesce_devices() — triggers FIX AK + FIX AM flag set.
    //
    // Mesh-edge ETH channels connect to out-of-mesh peers running base-UMD
    // firmware.  Those peers never advance the EDM handshake; their local
    // ERISC is left stuck below READY_FOR_TRAFFIC.  FIX AK returns non-fatally
    // (no throw); FIX AM sets fabric_channels_not_ready_for_traffic_=true on
    // the devices whose edge channels were stuck.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-10] Phase 2: calling quiesce_devices() on partial mesh — "
        "FIX AK must handle out-of-mesh ETH peers non-fatally; "
        "FIX AM must set fabric_channels_not_ready_for_traffic_");

    ASSERT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-10] Phase 2: quiesce_devices() threw — FIX AK (non-fatal partial-mesh Phase 5b) may be absent";

    log_info(tt::LogTest, "[GAP-10] Phase 2: quiesce_devices() returned cleanly — FIX AK confirmed");

    // Verify AllGather output correctness before moving on.
    {
        auto disaggregated = tt::tt_metal::experimental::unit_mesh::disaggregate(partial_gathered);
        ASSERT_EQ(static_cast<int>(disaggregated.size()), kPartialMeshWidth)
            << "[GAP-10] Phase 1 AllGather: wrong number of output shards";
        const size_t per_device_vol = tensor_spec.logical_shape().volume();
        for (int dev_idx = 0; dev_idx < kPartialMeshWidth; dev_idx++) {
            auto data = disaggregated[dev_idx].to_vector<bfloat16>();
            ASSERT_EQ(data.size(), per_device_vol * kPartialMeshWidth)
                << "[GAP-10] Phase 1 AllGather: output size mismatch at dev_idx=" << dev_idx;
            for (size_t i = 0; i < data.size(); i++) {
                float expected = static_cast<float>(i / per_device_vol);
                EXPECT_EQ(static_cast<float>(data[i]), expected)
                    << "[GAP-10] Phase 1 AllGather data corruption at element " << i
                    << " dev_idx=" << dev_idx
                    << " (expected=" << expected << " got=" << static_cast<float>(data[i]) << ")";
            }
        }
        log_info(tt::LogTest, "[GAP-10] Phase 1 AllGather output verified — no data corruption");
    }
    { auto tmp = std::move(partial_gathered); }

    // -----------------------------------------------------------------------
    // Phase 3: Assert flag IS set on >= 1 device (FIX AM set-path).
    //
    // Mesh-edge devices have ETH channels that faced out-of-mesh peers; FIX AM
    // must have set fabric_channels_not_ready_for_traffic_ on at least one of
    // them.  We reason at the level of device membership (iterating all devices
    // in the submesh) rather than hardcoding specific device IDs.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-10] Phase 3: checking fabric_channels_not_ready_for_traffic_ is set "
        "on >= 1 device after partial-mesh quiesce");

    bool flag_was_set = false;
    for (auto* dev : mesh_device_->get_devices()) {
        if (dev->is_fabric_channels_not_ready_for_traffic()) {
            flag_was_set = true;
            log_info(tt::LogTest,
                "[GAP-10] Phase 3: flag IS set on device {} (expected — out-of-mesh edge channel)",
                dev->id());
        }
    }
    ASSERT_TRUE(flag_was_set)
        << "[GAP-10] Phase 3: fabric_channels_not_ready_for_traffic_ was NOT set on any device "
        << "after partial-mesh quiesce — FIX AM (flag set-path) may have regressed.";

    log_info(tt::LogTest,
        "[GAP-10] Phase 3: FIX AM set-path confirmed — flag is set on >= 1 device");

    // -----------------------------------------------------------------------
    // Phase 4: Close the partial mesh — full teardown so the next open starts
    // from a clean state.  This is the same pattern used in GAP-5 Phase 4.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-10] Phase 4: closing partial mesh device");

    mesh_device_->close();
    mesh_device_.reset();

    log_info(tt::LogTest, "[GAP-10] Phase 4: partial mesh device closed");

    // -----------------------------------------------------------------------
    // Phase 5: Re-open a full-mesh MeshDevice (all available devices) with
    // FABRIC_2D.  configure_fabric() runs on every device; the FIX AM clear
    // at the top of configure_fabric() must reset the flag to false.
    //
    // SetFabricConfig must be called before MeshDevice::create because the
    // previous close() / post_teardown() already reset it to DISABLED.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-10] Phase 5: re-opening FULL mesh device with FABRIC_2D — "
        "configure_fabric() must clear fabric_channels_not_ready_for_traffic_");

    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    const auto system_mesh_shape =
        tt::tt_metal::MetalContext::instance().get_system_mesh().shape();

    ASSERT_NO_THROW(
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(system_mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            DispatchCoreConfig{},
            {},
            config_.worker_l1_size))
        << "[GAP-10] Phase 5: MeshDevice::create() threw on full-mesh re-open.";

    log_info(tt::LogTest,
        "[GAP-10] Phase 5: full mesh MeshDevice::create() succeeded — "
        "now verifying configure_fabric() cleared the flag");

    // -----------------------------------------------------------------------
    // Phase 6: Assert flag is CLEARED on every device (FIX AM clear-path).
    //
    // configure_fabric() runs on all devices during MeshDevice::create().
    // The clear at the top of configure_fabric() must have reset the flag to
    // false on every device — including those that had it set in Phase 3.
    // A stale flag here would cause every subsequent AllGather to spuriously
    // GTEST_SKIP even though the fabric is healthy.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-10] Phase 6: checking fabric_channels_not_ready_for_traffic_ is cleared "
        "on ALL devices after full-mesh re-open");

    bool any_stale = false;
    for (auto* dev : mesh_device_->get_devices()) {
        if (dev->is_fabric_channels_not_ready_for_traffic()) {
            any_stale = true;
            log_info(tt::LogTest,
                "[GAP-10] Phase 6: STALE flag found on device {} — configure_fabric() clear path broken",
                dev->id());
        }
    }
    ASSERT_FALSE(any_stale)
        << "[GAP-10] Phase 6: fabric_channels_not_ready_for_traffic_ is STILL SET on >= 1 device "
        << "after full-mesh configure_fabric(). "
        << "FIX AM clear (configure_fabric() top-of-function flag reset) has regressed — "
        << "stale flag will cause every subsequent AllGather to spuriously GTEST_SKIP.";

    log_info(tt::LogTest,
        "[GAP-10] Phase 6: FIX AM clear-path confirmed — flag cleared on ALL devices after configure_fabric()");

    // -----------------------------------------------------------------------
    // Phase 7: AllGather on the full mesh — confirms the re-initialized fabric
    // is operationally healthy and no spurious GTEST_SKIP is triggered.
    //
    // Build per-device submesh views for the full mesh.  We use a 1x1 submesh
    // for each device column, iterating over the total number of devices as
    // reported by the full mesh rather than hardcoding a device count.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-10] Phase 7: AllGather on full mesh — verifying no spurious GTEST_SKIP");

    const int num_full_devices = static_cast<int>(mesh_device_->get_devices().size());
    // system_mesh_shape may be NxM; the total device count equals mesh_size().
    // For the AllGather we use a 1xN view of the flattened device list, placing
    // each device in its own 1x1 submesh by linear index within the full mesh.
    //
    // Simpler approach: use a 1x(total) mesh topology for the AllGather.
    // The full mesh device was created with the system mesh shape, so we can
    // reshape the submeshes to match the flattened device layout.
    // We iterate all mesh coordinates in row-major order.
    auto full_mesh_range = MeshCoordinateRange(mesh_device_->shape());
    std::vector<MeshCoordinate> all_coords(full_mesh_range.begin(), full_mesh_range.end());
    ASSERT_EQ(static_cast<int>(all_coords.size()), num_full_devices)
        << "[GAP-10] Phase 7: coordinate count mismatch with device count";

    TensorSpec full_tensor_spec(
        ttnn::Shape({1, 1, 32, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    std::vector<std::shared_ptr<distributed::MeshDevice>> full_submeshes;
    full_submeshes.reserve(num_full_devices);
    for (const auto& coord : all_coords) {
        full_submeshes.push_back(mesh_device_->create_submesh(MeshShape(1, 1), coord));
    }

    std::vector<ttnn::Tensor> full_tensors;
    full_tensors.reserve(num_full_devices);
    for (int dev_idx = 0; dev_idx < num_full_devices; dev_idx++) {
        std::vector<bfloat16> data(
            full_tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        full_tensors.push_back(
            Tensor::from_vector(std::move(data), full_tensor_spec)
                .to_device(full_submeshes[dev_idx].get()));
    }

    auto full_aggregated = tt::tt_metal::experimental::unit_mesh::aggregate(full_tensors);

    log_info(tt::LogTest, "[GAP-10] Phase 7: launching AllGather on full mesh ({} devices)", num_full_devices);
    auto full_gathered = ttnn::all_gather(full_aggregated, /* dim */ 0);
    log_info(tt::LogTest, "[GAP-10] Phase 7: AllGather on full mesh complete — no spurious GTEST_SKIP");

    // Verify AllGather output on the full mesh.
    {
        auto disaggregated = tt::tt_metal::experimental::unit_mesh::disaggregate(full_gathered);
        ASSERT_EQ(static_cast<int>(disaggregated.size()), num_full_devices)
            << "[GAP-10] Phase 7 AllGather: wrong number of output shards";
        const size_t per_device_vol = full_tensor_spec.logical_shape().volume();
        for (int dev_idx = 0; dev_idx < num_full_devices; dev_idx++) {
            auto data = disaggregated[dev_idx].to_vector<bfloat16>();
            ASSERT_EQ(data.size(), per_device_vol * static_cast<size_t>(num_full_devices))
                << "[GAP-10] Phase 7 AllGather: output size mismatch at dev_idx=" << dev_idx;
            for (size_t i = 0; i < data.size(); i++) {
                float expected = static_cast<float>(i / per_device_vol);
                EXPECT_EQ(static_cast<float>(data[i]), expected)
                    << "[GAP-10] Phase 7 AllGather data corruption at element " << i
                    << " dev_idx=" << dev_idx
                    << " (expected=" << expected << " got=" << static_cast<float>(data[i]) << ")";
            }
        }
        log_info(tt::LogTest, "[GAP-10] Phase 7 AllGather output verified — no data corruption");
    }
    { auto tmp = std::move(full_gathered); }

    log_info(tt::LogTest,
        "[GAP-10] FlagIsSetAfterPartialMeshQuiesceThenClearedOnFullMeshReopen PASSED — "
        "FIX AM set-path (flag set after partial quiesce) and clear-path "
        "(flag cleared by configure_fabric() on full-mesh reopen) both confirmed");
}

}  // namespace tt::tt_metal::distributed::test
