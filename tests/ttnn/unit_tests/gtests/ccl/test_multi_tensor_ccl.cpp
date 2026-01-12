// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    MeshDevice1x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 4}}) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void TearDown() override {
        MeshDeviceFixtureBase::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
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

    auto disaggregated_output_tensors = tt::tt_metal::experimental::unit_mesh::disaggregate(all_gathered_tensor);
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = disaggregated_output_tensors[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            // NOLINTNEXTLINE(bugprone-integer-division)
            auto expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
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

}  // namespace tt::tt_metal
