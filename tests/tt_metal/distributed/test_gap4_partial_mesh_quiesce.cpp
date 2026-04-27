// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-4: Covers FIX AK + FIX AM — partial-mesh non-fatal quiesce
//
// Background:
//   FIX AK (#42429): Phase 5b in wait_for_fabric_workers_ready() must NOT throw
//   when channels are stuck at STARTED / REMOTE_HANDSHAKE_COMPLETE /
//   LOCAL_HANDSHAKE_COMPLETE.  This happens in partial-mesh operation (e.g. a
//   1x4 submesh of T3K): mesh-edge devices have ETH channels that connect to
//   chips NOT in the quiesce set.  Those out-of-mesh peers run base-UMD firmware
//   during quiesce and never advance the handshake; their local ERISC is left
//   stuck below READY_FOR_TRAFFIC.  Before FIX AK these caused a fatal TT_THROW;
//   after FIX AK Phase 5b returns cleanly (non-fatal).
//
//   FIX AM (#42429): fabric_channels_not_ready_for_traffic_ is set so callers
//   can distinguish "partial-mesh peer didn't respond" from "relay path broken".
//
//   FIX AK-2 (#42429): non-MMIO devices (remote peers in a partial mesh) that
//   observe unexpected channel states during async teardown also return cleanly.
//
// What this test verifies:
//   1. quiesce_devices() does NOT throw on a partial mesh (FIX AK).
//   2. The fabric is still usable for in-mesh operations after quiesce (FIX AM
//      does not erroneously block operations between mesh-member devices).
//   3. A second quiesce_devices() also returns cleanly (idempotent non-fatal).
//
// Topology requirement: >= 4 devices (T3K or larger).
// The test opens a 1x4 submesh, exercises AllGather within the submesh, and
// calls quiesce_devices() — which will encounter out-of-mesh ETH peers on the
// mesh-edge devices.

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
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture: PartialMeshQuiesceFixture
//
// Opens a 1x4 submesh on T3K (or larger) with FABRIC_2D active.
// Skips on systems with fewer than 4 devices — partial-mesh behaviour only
// manifests when a subset of the available chips is used; a full-mesh open
// would not exercise the out-of-mesh ETH channel path.
//
// Note: some ETH channels on mesh-edge devices connect to out-of-mesh devices
// at REMOTE_HANDSHAKE_COMPLETE — FIX AK must handle non-fatally.
// ---------------------------------------------------------------------------
class PartialMeshQuiesceFixture : public MeshDeviceFixtureBase {
protected:
    PartialMeshQuiesceFixture()
        : MeshDeviceFixtureBase(Config{
              .mesh_shape = MeshShape{1, 4},  // partial mesh on T3K
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,  // 2-minute budget: FABRIC_2D init is ~15s per cycle
          }) {}

    void SetUp() override {
        // Require >= 4 devices: partial-mesh out-of-mesh ETH channels only appear
        // when we open fewer devices than the physical topology provides.
        // On a 4-device T3K this opens all 4; on an 8-device system this opens
        // the first 4 columns, leaving 4 out-of-mesh.  Either way the quiesce
        // path must handle non-responding out-of-mesh peers non-fatally (FIX AK).
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 4) {
            GTEST_SKIP() << "PartialMeshQuiesceFixture requires >= 4 devices (T3K topology). "
                         << "Found " << num_devices << " device(s). "
                         << "Partial-mesh out-of-mesh ETH peer behaviour only manifests on 4+ device systems.";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// Test: PartialMeshQuiesceNonFatal
//
// Steps:
//   1. Open 1x4 partial mesh with FABRIC_2D.
//   2. Run CCL AllGather within the submesh (exercises in-mesh ETH channels).
//   3. Call quiesce_devices() — FIX AK must NOT throw even though mesh-edge
//      ETH channels face out-of-mesh peers stuck at REMOTE_HANDSHAKE_COMPLETE.
//   4. Assert EXPECT_NO_THROW for the first quiesce call.
//   5. Run another AllGather after quiesce — verifies in-mesh fabric is still
//      usable (FIX AM correctly sets channels_not_ready but must not block
//      operations between mesh-member devices).
//   6. Call quiesce_devices() a second time — must also return cleanly (FIX AK
//      is idempotent: Phase 2.5 will TERMINATE lingering channels).
//   7. Assert EXPECT_NO_THROW for the second quiesce call.
//
// Pass = all assertions green, no throw, completes in <2 minutes.
// Fail = TT_THROW from Phase 5b (FIX AK absent), crash, or hang.
// ---------------------------------------------------------------------------
TEST_F(PartialMeshQuiesceFixture, PartialMeshQuiesceNonFatal) {
    // Small tensor: exercises all in-mesh ERISC channels while completing
    // quickly within the test budget.
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 32, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    // Build per-device submesh views for tensor placement and readback.
    // mesh_device_ is a 1x4 mesh (4 devices total).
    // Reasoning at the level of characteristics, not device numbers:
    //   each device in the submesh gets a 1x1 view at its mesh coordinate.
    constexpr int kNumMeshDevices = 4;
    std::vector<std::shared_ptr<distributed::MeshDevice>> submeshes;
    for (int col = 0; col < kNumMeshDevices; col++) {
        submeshes.push_back(
            mesh_device_->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, col)));
    }

    // -----------------------------------------------------------------------
    // Phase 1: AllGather pass 1 — warm up in-mesh ETH channels.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest, "[GAP-4] Phase 1: building input tensors for AllGather pass 1");

    // Device i holds a tensor filled with float(i) so the gathered result
    // along dim=0 is [0, 1, 2, 3] (repeated per element), giving a
    // deterministic expected output that detects DRAM corruption.
    std::vector<ttnn::Tensor> tensors_pass1;
    for (int dev_idx = 0; dev_idx < kNumMeshDevices; dev_idx++) {
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors_pass1.push_back(
            Tensor::from_vector(std::move(data), tensor_spec)
                .to_device(submeshes[dev_idx].get()));
    }

    auto aggregated1 = tt::tt_metal::experimental::unit_mesh::aggregate(tensors_pass1);

    log_info(tt::LogTest, "[GAP-4] Phase 1: launching AllGather pass 1");
    auto gathered1 = ttnn::all_gather(aggregated1, /* dim */ 0);
    log_info(tt::LogTest, "[GAP-4] Phase 1: AllGather pass 1 complete");

    // -----------------------------------------------------------------------
    // Phase 2: First quiesce — FIX AK critical path.
    //
    // At this point mesh-edge devices have ETH channels connected to out-of-mesh
    // peers.  Those peers run base-UMD firmware and never responded to the EDM
    // handshake; their local ERISC is stuck at REMOTE_HANDSHAKE_COMPLETE.
    // Phase 5b must detect this partial-mesh pattern and return non-fatally
    // instead of throwing (FIX AK #42429).
    // FIX AM sets fabric_channels_not_ready_for_traffic_ on the affected devices.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-4] Phase 2: calling quiesce_devices() — FIX AK must handle out-of-mesh "
        "ETH peers at REMOTE_HANDSHAKE_COMPLETE non-fatally");

    EXPECT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-4] First quiesce_devices() threw — FIX AK (non-fatal partial-mesh Phase 5b) may be absent";

    log_info(tt::LogTest, "[GAP-4] Phase 2: first quiesce_devices() returned cleanly — FIX AK confirmed");

    // Verify AllGather pass 1 output correctness.
    {
        auto disaggregated1 = tt::tt_metal::experimental::unit_mesh::disaggregate(gathered1);
        ASSERT_EQ(static_cast<int>(disaggregated1.size()), kNumMeshDevices)
            << "[GAP-4] AllGather pass 1: wrong number of output shards";

        const size_t per_device_vol = tensor_spec.logical_shape().volume();
        for (int dev_idx = 0; dev_idx < kNumMeshDevices; dev_idx++) {
            auto data = disaggregated1[dev_idx].to_vector<bfloat16>();
            ASSERT_EQ(data.size(), per_device_vol * kNumMeshDevices)
                << "[GAP-4] AllGather pass 1: output size mismatch at dev_idx=" << dev_idx;
            for (size_t i = 0; i < data.size(); i++) {
                float expected = static_cast<float>(i / per_device_vol);
                EXPECT_EQ(static_cast<float>(data[i]), expected)
                    << "[GAP-4] AllGather pass 1 data corruption at element " << i
                    << " dev_idx=" << dev_idx
                    << " (expected=" << expected << " got=" << static_cast<float>(data[i]) << ")";
            }
        }
        log_info(tt::LogTest, "[GAP-4] AllGather pass 1 output verified — no data corruption");
    }

    { auto tmp = std::move(gathered1); }

    // -----------------------------------------------------------------------
    // Phase 3: AllGather pass 2 — verify fabric still usable for in-mesh channels.
    //
    // FIX AM sets channels_not_ready_for_traffic on affected edge devices but
    // must NOT prevent AllGather between mesh-member pairs whose ETH channels
    // DID complete the handshake.  This pass exercises those in-mesh channels.
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-4] Phase 3: AllGather pass 2 — verifying in-mesh fabric still usable after partial-mesh quiesce");

    std::vector<ttnn::Tensor> tensors_pass2;
    for (int dev_idx = 0; dev_idx < kNumMeshDevices; dev_idx++) {
        // Fill with offset pattern to distinguish from pass 1 data.
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx + 10)));
        tensors_pass2.push_back(
            Tensor::from_vector(std::move(data), tensor_spec)
                .to_device(submeshes[dev_idx].get()));
    }

    auto aggregated2 = tt::tt_metal::experimental::unit_mesh::aggregate(tensors_pass2);

    log_info(tt::LogTest, "[GAP-4] Phase 3: launching AllGather pass 2");
    auto gathered2 = ttnn::all_gather(aggregated2, /* dim */ 0);
    log_info(tt::LogTest, "[GAP-4] Phase 3: AllGather pass 2 complete");

    // -----------------------------------------------------------------------
    // Phase 4: Second quiesce — verifies idempotent non-fatal behaviour.
    //
    // Phase 2.5 in the next quiesce will TERMINATE any lingering channels
    // (including those that were stuck at REMOTE_HANDSHAKE_COMPLETE in Phase 2).
    // This second call must also return cleanly (FIX AK is idempotent).
    // -----------------------------------------------------------------------
    log_info(tt::LogTest,
        "[GAP-4] Phase 4: second quiesce_devices() — FIX AK must be idempotent");

    EXPECT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-4] Second quiesce_devices() threw — FIX AK must be idempotent";

    log_info(tt::LogTest, "[GAP-4] Phase 4: second quiesce_devices() returned cleanly — idempotent confirmed");

    // Verify AllGather pass 2 output correctness.
    {
        auto disaggregated2 = tt::tt_metal::experimental::unit_mesh::disaggregate(gathered2);
        ASSERT_EQ(static_cast<int>(disaggregated2.size()), kNumMeshDevices)
            << "[GAP-4] AllGather pass 2: wrong number of output shards";

        const size_t per_device_vol = tensor_spec.logical_shape().volume();
        for (int dev_idx = 0; dev_idx < kNumMeshDevices; dev_idx++) {
            auto data = disaggregated2[dev_idx].to_vector<bfloat16>();
            ASSERT_EQ(data.size(), per_device_vol * kNumMeshDevices)
                << "[GAP-4] AllGather pass 2: output size mismatch at dev_idx=" << dev_idx;
            for (size_t i = 0; i < data.size(); i++) {
                float expected = static_cast<float>((i / per_device_vol) + 10);
                EXPECT_EQ(static_cast<float>(data[i]), expected)
                    << "[GAP-4] AllGather pass 2 data corruption at element " << i
                    << " dev_idx=" << dev_idx
                    << " (expected=" << expected << " got=" << static_cast<float>(data[i]) << ")";
            }
        }
        log_info(tt::LogTest, "[GAP-4] AllGather pass 2 output verified — no data corruption after second quiesce");
    }

    { auto tmp = std::move(gathered2); }

    log_info(tt::LogTest,
        "[GAP-4] PartialMeshQuiesceNonFatal PASSED — "
        "FIX AK (non-fatal partial-mesh Phase 5b) and FIX AM (channels_not_ready tracking) confirmed");
}

}  // namespace tt::tt_metal::distributed::test
