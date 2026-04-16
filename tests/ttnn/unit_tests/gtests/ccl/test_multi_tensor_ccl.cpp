// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "test_fabric_edm_common.hpp"

#include <vector>
namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::vector<std::shared_ptr<distributed::MeshDevice>> get_line_devices(distributed::MeshDevice* mesh_device) {
    return {
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 0)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 1)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 2)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 3)),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

class MeshDevice1x4Fixture : public MeshDeviceFixtureBase {
protected:
    // Use Config::fabric_config to scope fabric init/teardown to the active devices only,
    // rather than calling SetFabricConfig() globally in the constructor.  The base class
    // SetUp() calls SetFabricConfig() right before MeshDevice::create() and TearDown()
    // calls SetFabricConfig(DISABLED) after mesh close — preventing fabric firmware on
    // un-owned devices from being left in a dirty state across test iterations.
    MeshDevice1x4Fixture() :
        MeshDeviceFixtureBase(
            Config{.mesh_shape = MeshShape{1, 4}, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}) {}
};

class MultiCQFabricMeshDevice2x4Fixture : public MultiCQMeshDevice2x4Fixture {
protected:
    MultiCQFabricMeshDevice2x4Fixture() { tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D); }
    void TearDown() override {
        MultiCQMeshDevice2x4Fixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

TEST_F(MeshDevice1x4Fixture, AllGatherReturnedTensor) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }

    auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);

    // Quiesce parent mesh before all gather
    mesh_device_->quiesce_devices();

    auto all_gathered_tensor = ttnn::all_gather(
        aggregated_tensor,
        /* dim */ 0);

    // Quiesce parent mesh after all gather
    mesh_device_->quiesce_devices();
    log_info(tt::LogMetal, "[test_body] second quiesce_devices() returned");

    log_info(tt::LogMetal, "[test_body] calling disaggregate()");
    auto disaggregated_output_tensors = tt::tt_metal::experimental::unit_mesh::disaggregate(all_gathered_tensor);
    log_info(tt::LogMetal, "[test_body] disaggregate() returned, {} tensors", disaggregated_output_tensors.size());
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        log_info(tt::LogMetal, "[test_body] calling to_vector() for dev_idx={}", dev_idx);
        auto data = disaggregated_output_tensors[dev_idx].to_vector<bfloat16>();
        log_info(tt::LogMetal, "[test_body] to_vector() returned for dev_idx={}, {} elements", dev_idx, data.size());
        for (int i = 0; i < data.size(); i++) {
            // NOLINTNEXTLINE(bugprone-integer-division)
            auto expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
        log_info(tt::LogMetal, "[test_body] EXPECT_EQ loop done for dev_idx={}", dev_idx);
    }
    log_info(tt::LogMetal, "[test_body] all loops done, destroying disaggregated_output_tensors");
    disaggregated_output_tensors.clear();
    log_info(tt::LogMetal, "[test_body] disaggregated_output_tensors destroyed, destroying all_gathered_tensor");
    { auto tmp = std::move(all_gathered_tensor); }
    log_info(tt::LogMetal, "[test_body] all_gathered_tensor destroyed, test body finishing");
}

TEST_F(MeshDevice1x4Fixture, AllGatherPersistentOutput) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors, output_tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    TensorSpec output_tensor_spec(
        ttnn::Shape({4, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
        std::vector<bfloat16> output_data(output_tensor_spec.logical_shape().volume(), bfloat16(0));
        output_tensors.push_back(
            Tensor::from_vector(std::move(output_data), output_tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }

    auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);
    auto aggregated_output_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(output_tensors);

    // Quiesce parent mesh before all gather
    mesh_device_->quiesce_devices();

    auto all_gathered_tensor = ttnn::all_gather(
        aggregated_tensor,
        /* dim */ 0,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        aggregated_output_tensor);

    // Quiesce parent mesh after all gather
    mesh_device_->quiesce_devices();

    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = output_tensors[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            // NOLINTNEXTLINE(bugprone-integer-division)
            auto expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
}

TEST_F(MeshDevice1x4Fixture, ReduceScatter) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors, output_tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    TensorSpec output_tensor_spec(
        ttnn::Shape({1, 8, 1024, 192}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (auto& mesh_device : mesh_devices) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_device.get()));
        std::vector<bfloat16> output_data(output_tensor_spec.logical_shape().volume(), bfloat16(0));
        output_tensors.push_back(
            Tensor::from_vector(std::move(output_data), output_tensor_spec).to_device(mesh_device.get()));
    }
    auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);
    auto aggregated_output_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(output_tensors);

    // Quiesce parent mesh before reduce scatter
    mesh_device_->quiesce_devices();
    auto reduced = ttnn::reduce_scatter(
        aggregated_tensor,
        /* dim */ 3,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        aggregated_output_tensor);
    // Quiesce parent mesh after reduce scatter
    mesh_device_->quiesce_devices();

    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = output_tensors[dev_idx].to_vector<bfloat16>();
        for (auto val : data) {
            float expected = static_cast<float>(mesh_devices.size());
            EXPECT_EQ(static_cast<float>(val), expected);
        }
    }
}

TEST_F(MeshDevice1x4Fixture, AllReduce) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (auto& mesh_device : mesh_devices) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_device.get()));
    }

    auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);

    // Quiesce parent mesh before all reduce
    mesh_device_->quiesce_devices();
    auto all_reduced_tensor = ttnn::all_reduce(
        aggregated_tensor,
        /* cluster_axis */ 1);
    // Quiesce parent mesh after all reduce
    mesh_device_->quiesce_devices();

    auto disaggregated_output_tensors = tt::tt_metal::experimental::unit_mesh::disaggregate(all_reduced_tensor);
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = disaggregated_output_tensors[dev_idx].to_vector<bfloat16>();
        for (auto val : data) {
            float expected = static_cast<float>(mesh_devices.size());
            EXPECT_EQ(static_cast<float>(val), expected);
        }
    }
}

TEST_F(MeshDevice1x4Fixture, AllGatherReturnedTensorNoHang) {
    // REGRESSION TEST: multiple bugs fixed on branch nsexton/0-racecondition-hunt.
    // This test exercises the full quiesce→all_gather→readback→destroy cycle across N iterations
    // to stress all three race conditions simultaneously.
    //
    // ----- Bug 1: EventSynchronize hang in MeshBuffer::wait_for_pending_events() -----
    //
    // Sequence that triggers the hang (without the fix):
    //   1. all_gather() on parent mesh CQ records event E, stored in MeshBuffer
    //   2. quiesce_devices() → finish_and_reset_in_use(): sets in_use_=false, resets event counters
    //   3. to_vector() on submesh CQ → mark_in_use() on shared physical devices clears quiesced flag
    //   4. Submesh events complete → last_completed_event = 1, 2, ... (new post-reset sequence)
    //   5. all_gathered_tensor destructor → wait_for_pending_events() → EventSynchronize(E):
    //      in_use() returns false incorrectly (quiesced was cleared), last_completed < E → infinite spin
    //
    // Fix: check mesh_command_queue.in_use() before calling EventSynchronize; if the parent
    // mesh CQ was quiesced, all work was already drained — skip the stale event.
    //
    // ----- Bug 2: AllGather writer exits before tt_fabric_mux terminates -----
    //   (commits 3804c11fc3, 430292f6c6)
    //
    // The AllGather writer kernel was completing and returning to the dispatch layer
    // while the fabric mux kernel was still running on an adjacent core, leading to
    // the mux being reset mid-operation on the next iteration.
    //
    // Fix: writer now calls wait_for_fabric_endpoint_terminated() before exiting,
    // polling the mux's L1 status address until FabricMuxStatus::TERMINATED is written.
    //
    // ----- Bug 3 (A/B): tt_fabric_mux TERMINATED write gated behind ETH teardown ACK -----
    //   (commit ff78c87e44 — see https://github.com/tenstorrent/tt-metal/issues/42429)
    //
    // The mux wrote FabricMuxStatus::TERMINATED only *after* fabric_connection.close()
    // completed in full, including close_finish() which spin-polls for an ETH router ACK.
    // If the ETH ACK was delayed or never arrived (e.g. due to ARC flagging the NOC access
    // to the ETH core as unsafe), TERMINATED was never written and Bug 2's fix would
    // itself hang forever — a classic A/B deadlock.
    //
    // Fix: split close() into close_start() + TERMINATED write + close_finish() so the
    // writer can unblock as soon as the close request is sent, while the mux still completes
    // the full ETH handshake before signalling dispatch completion.
    //
    // Without all three fixes, this test hangs on iteration 1 or 2.

    constexpr int kIterations = 3;
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 32, 128}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    for (int iter = 0; iter < kIterations; iter++) {
        log_info(tt::LogMetal, "[AllGatherReturnedTensorNoHang] iteration {}/{}", iter + 1, kIterations);

        // Step 1: Create tensors on submesh devices
        std::vector<ttnn::Tensor> tensors;
        for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
            std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
            tensors.push_back(
                Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
        }

        // Step 2: Aggregate and all_gather on parent mesh → records event E
        auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);
        mesh_device_->quiesce_devices();

        auto all_gathered_tensor = ttnn::all_gather(aggregated_tensor, /* dim */ 0);

        // Step 3: Quiesce parent mesh → resets event counters, sets in_use_=false
        mesh_device_->quiesce_devices();

        // Step 4: Readback via submesh → mark_in_use on shared physical devices
        auto disaggregated = tt::tt_metal::experimental::unit_mesh::disaggregate(all_gathered_tensor);
        for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
            auto data = disaggregated[dev_idx].to_vector<bfloat16>();
            ASSERT_FALSE(data.empty()) << "Empty readback at dev_idx=" << dev_idx << " iter=" << iter;
        }

        // Step 5: Destroy all_gathered_tensor → triggers wait_for_pending_events() with stale event E
        // Without the fix, this hangs forever.
        disaggregated.clear();
        { auto tmp = std::move(all_gathered_tensor); }

        log_info(tt::LogMetal, "[AllGatherReturnedTensorNoHang] iteration {}/{} completed", iter + 1, kIterations);
    }
}

}  // namespace tt::tt_metal
