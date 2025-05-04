// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
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
    auto view = mesh_device->get_view();
    return {
        view.get_device(distributed::MeshCoordinate(0, 0)),
        view.get_device(distributed::MeshCoordinate(0, 1)),
        view.get_device(distributed::MeshCoordinate(0, 2)),
        view.get_device(distributed::MeshCoordinate(0, 3))};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// Only testing top row of T3000.
constexpr int kNumDevices = 4;

}  // namespace

class T3000MultiCQFabricMeshDeviceFixture : public T3000MultiCQMeshDeviceFixture {
protected:
    T3000MultiCQFabricMeshDeviceFixture() {
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::FABRIC_1D);
    }
    void TearDown() override {
        T3000MultiCQMeshDeviceFixture::TearDown();
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

TEST_F(T3000MultiCQMeshDeviceFixture, AllGather) {
    mesh_device_->enable_program_cache();
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    TensorSpec tensor_spec(
        ttnn::Shape({kNumDevices, 8, 1024, 768}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<bfloat16> host_data(tensor_spec.logical_shape().volume());
    for (int dev_idx = 0; dev_idx < kNumDevices; dev_idx++) {
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume() / kNumDevices, bfloat16(static_cast<float>(dev_idx)));
        std::move(data.begin(), data.end(), std::back_inserter(host_data));
    }

    Tensor tensor = ttnn::distributed::distribute_tensor(
        Tensor::from_vector(std::move(host_data), tensor_spec),
        *ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/0),
        *mesh_device_);

    auto all_gathered = ttnn::all_gather(
        ttnn::distributed::get_device_tensors(tensor),
        0,
        1,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(T3000MultiCQFabricMeshDeviceFixture, AllGatherAsync) {
    mesh_device_->enable_program_cache();
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    TensorSpec tensor_spec(
        ttnn::Shape({kNumDevices, 8, 1024, 768}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<bfloat16> host_data;
    host_data.reserve(tensor_spec.logical_shape().volume());
    for (int dev_idx = 0; dev_idx < kNumDevices; dev_idx++) {
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume() / kNumDevices, bfloat16(static_cast<float>(dev_idx)));
        std::move(data.begin(), data.end(), std::back_inserter(host_data));
    }

    Tensor tensor = ttnn::distributed::distribute_tensor(
        Tensor::from_vector(std::move(host_data), tensor_spec),
        *ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/0),
        *mesh_device_);

    auto semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    auto all_gathered = ttnn::experimental::all_gather_async(
        ttnn::distributed::get_device_tensors(tensor),
        0,
        semaphore,
        1,
        std::nullopt,
        ttnn::ccl::Topology::Linear,
        SubDeviceId(0));
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(T3000MultiCQMeshDeviceFixture, ReduceScatter) {
    mesh_device_->enable_program_cache();
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    TensorSpec tensor_spec(
        ttnn::Shape({kNumDevices, 8, 1024, 768}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<bfloat16> host_data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));

    Tensor tensor = ttnn::distributed::distribute_tensor(
        Tensor::from_vector(std::move(host_data), tensor_spec),
        *ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/0),
        *mesh_device_);

    auto reduced = ttnn::reduce_scatter(
        ttnn::distributed::get_device_tensors(tensor),
        3,
        ttnn::operations::reduction::ReduceType::Sum,
        1,
        std::nullopt,
        ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devices.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(T3000MultiCQMeshDeviceFixture, AllReduce) {
    mesh_device_->enable_program_cache();
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());

    TensorSpec tensor_spec(
        ttnn::Shape({kNumDevices, 8, 1024, 768}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<bfloat16> host_data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));

    Tensor tensor = ttnn::distributed::distribute_tensor(
        Tensor::from_vector(std::move(host_data), tensor_spec),
        *ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/0),
        *mesh_device_);

    auto reduced = ttnn::experimental::all_reduce(
        ttnn::distributed::get_device_tensors(tensor),
        ttnn::operations::reduction::ReduceType::Sum,
        1,
        std::nullopt,
        ttnn::ccl::Topology::Linear);
    for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devices.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

}  // namespace tt::tt_metal
