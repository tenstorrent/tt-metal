// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "test_fabric_edm_common.hpp"

#include <vector>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

ttnn::global_semaphore::MultiDeviceGlobalSemaphore create_global_semaphore(const std::vector<IDevice*>& devices) {
    return ttnn::global_semaphore::create_global_semaphore_with_same_address(
        devices,
        devices.at(0)->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,
        tt::tt_metal::BufferType::L1,
        10);
}

std::vector<IDevice*> get_line_devices(distributed::MeshDevice* mesh_device) {
    const auto& view = mesh_device->get_view();
    return {
        view.get_device(distributed::MeshCoordinate(0, 0)),
        view.get_device(distributed::MeshCoordinate(0, 1)),
        view.get_device(distributed::MeshCoordinate(0, 2)),
        view.get_device(distributed::MeshCoordinate(0, 3))};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

class MultiCQFabricMeshDevice2x4Fixture : public MultiCQMeshDevice2x4Fixture {
protected:
    MultiCQFabricMeshDevice2x4Fixture() { tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D); }
    void TearDown() override {
        MultiCQMeshDevice2x4Fixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

TEST_F(MultiCQMeshDevice2x4Fixture, AllGather) {
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        auto unit_mesh = MeshDevice::create_unit_mesh(devices[dev_idx]->id());
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(unit_mesh.get()));
    }
    auto all_gathered =
        ttnn::all_gather(tensors, 0, 1, std::nullopt, std::nullopt, std::nullopt, ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiCQFabricMeshDevice2x4Fixture, AllGatherAsync) {
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        auto unit_mesh = MeshDevice::create_unit_mesh(devices[dev_idx]->id());
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(unit_mesh.get()));
    }
    auto semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {semaphore};
    auto all_gathered = ttnn::experimental::all_gather_async(
        tensors, 0, multi_dev_semaphore, 1, std::nullopt, ttnn::ccl::Topology::Linear, SubDeviceId(0));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiCQMeshDevice2x4Fixture, ReduceScatter) {
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        auto unit_mesh = MeshDevice::create_unit_mesh(devices[dev_idx]->id());
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(unit_mesh.get()));
    }
    auto reduced = ttnn::reduce_scatter(
        tensors, 3, ttnn::operations::reduction::ReduceType::Sum, 1, std::nullopt, ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devices.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiCQMeshDevice2x4Fixture, AllReduce) {
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        auto unit_mesh = MeshDevice::create_unit_mesh(devices[dev_idx]->id());
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(unit_mesh.get()));
    }
    auto reduced = ttnn::experimental::all_reduce(
        tensors, ttnn::operations::reduction::ReduceType::Sum, 1, std::nullopt, ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devices.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

}  // namespace tt::tt_metal
