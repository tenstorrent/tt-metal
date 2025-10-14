// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/all_gather_command_processor_async/all_gather_command_processor_async.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

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

std::vector<std::shared_ptr<distributed::MeshDevice>> get_line_devices(distributed::MeshDevice* mesh_device) {
    return {
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 0)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 1)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 2)),
        mesh_device->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, 3)),
    };
}

std::vector<IDevice*> get_line_devices_as_idevice(
    const std::vector<std::shared_ptr<distributed::MeshDevice>>& mesh_devices) {
    std::vector<IDevice*> devices;
    devices.reserve(mesh_devices.size());
    for (auto& mesh_device : mesh_devices) {
        devices.push_back(mesh_device.get());
    }
    return devices;
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

TEST_F(MeshDevice1x4Fixture, AllGatherCommandProcessorAsync) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices_as_idevice(mesh_devices);

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        // the values in the tensors seem correct
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }

    auto aggregated_tensor = ttnn::experimental::unit_mesh::aggregate(tensors);
    auto disaggregated_output_tensors = ttnn::experimental::unit_mesh::disaggregate(aggregated_tensor);
    // aggregated_tensor.print();

    // auto all_gathered_tensor = ttnn::all_gather(
    //     aggregated_tensor,
    //     /* dim */ 0);

    // // all_gathered_tensor.print(); // missing 1 values on coord  (0, 1)

    // auto disaggregated_output_tensors = ttnn::experimental::unit_mesh::disaggregate(all_gathered_tensor);
    // for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
    //     auto data = disaggregated_output_tensors[dev_idx].to_vector<bfloat16>();
    //     for (int i = 0; i < data.size(); i++) {
    //         float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
    //         EXPECT_EQ(static_cast<float>(data[i]), expected);
    //     }
    // }
    for (const auto& tensor : disaggregated_output_tensors) {
        tensor.print();
    }
}

TEST_F(MeshDevice1x4Fixture, AllGatherPreallocatedOutputTensor) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices_as_idevice(mesh_devices);

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

    auto aggregated_tensor = ttnn::experimental::unit_mesh::aggregate(tensors);
    auto aggregated_output_tensor = ttnn::experimental::unit_mesh::aggregate(output_tensors);

    auto all_gathered_tensor = ttnn::all_gather(
        aggregated_tensor,
        /* dim */ 0,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        aggregated_output_tensor);

    // all_gathered_tensor.print(); // missing 1 values on coord  (0, 1)

    // auto disaggregated_output_tensors = ttnn::experimental::unit_mesh::disaggregate(all_gathered_tensor);
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = output_tensors[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
    for (const auto& tensor : output_tensors) {
        tensor.print();
    }
}

// TODO: uncomment this once the composite implementation is completed (#28556)
TEST_F(MultiCQFabricMeshDevice2x4Fixture, DISABLED_AllGatherMinimalAsyncComposite) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices_as_idevice(mesh_devices);

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }
    auto forward_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    auto backward_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
        forward_semaphore, backward_semaphore};
    tt::tt_metal::distributed::Synchronize(mesh_device_.get(), std::nullopt, std::vector<SubDeviceId>());

    auto all_gathered = ttnn::experimental::all_gather_async(
        /* input_tensors */ tensors,
        /* persistent_output_buffer */ std::nullopt,
        /* dim */ 0,
        /* multi_device_global_semaphore */ multi_dev_semaphore,
        /* num_links */ 1,
        /* memory_config */ std::nullopt,
        /* topology */ ttnn::ccl::Topology::Linear,
        /* subdevice_id */ SubDeviceId(0),
        /* cluster_axis */ std::nullopt,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ std::nullopt,
        /* chunks_per_sync */ std::nullopt,
        /* num_workers_per_link */ std::nullopt,
        /* num_buffers_per_channel */ std::nullopt);
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
}

// same as above but with a different tensor shape which triggers the native implementation
TEST_F(MultiCQFabricMeshDevice2x4Fixture, AllGatherMinimalAsyncNative) {
    // Issue #29828: This has some assertion error, issue assigned to CCL team
    GTEST_SKIP();
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices_as_idevice(mesh_devices);

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }
    auto forward_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    auto backward_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
        forward_semaphore, backward_semaphore};
    tt::tt_metal::distributed::Synchronize(mesh_device_.get(), std::nullopt, std::vector<SubDeviceId>());

    auto all_gathered = ttnn::experimental::all_gather_async(
        /* input_tensors */ tensors,
        /* persistent_output_buffer */ std::nullopt,
        /* dim */ 0,
        /* multi_device_global_semaphore */ multi_dev_semaphore,
        /* num_links */ 1,
        /* memory_config */ std::nullopt,
        /* topology */ ttnn::ccl::Topology::Linear,
        /* subdevice_id */ SubDeviceId(0),
        /* cluster_axis */ std::nullopt,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ std::nullopt,
        /* chunks_per_sync */ std::nullopt,
        /* num_workers_per_link */ std::nullopt,
        /* num_buffers_per_channel */ std::nullopt);
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
}

// TODO: This test is failing in the CI pipeline "(T3K) T3000 unit tests" but cannot be reproduced locally.
// Temporarily disabling this test as we plan to deprecate this API soon (#29340).
TEST_F(MultiCQFabricMeshDevice2x4Fixture, DISABLED_ReduceScatterAsync) {
    auto mesh_devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices(mesh_device_.get());
    auto devices = CMAKE_UNIQUE_NAMESPACE::get_line_devices_as_idevice(mesh_devices);

    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec).to_device(mesh_devices[dev_idx].get()));
    }
    auto from_remote_multi_device_global_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);
    auto to_remote_multi_device_global_semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devices);

    tt::tt_metal::distributed::Synchronize(mesh_device_.get(), std::nullopt, std::vector<SubDeviceId>());
    auto reduced = ttnn::experimental::reduce_scatter_async(
        tensors,
        3,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType::Sum,
        std::nullopt,
        ttnn::ccl::Topology::Linear,
        1,
        SubDeviceId(0));

    for (int dev_idx = 0; dev_idx < mesh_devices.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(mesh_devices.size());
            EXPECT_EQ(static_cast<float>(data[i]), expected);
        }
    }
}

}  // namespace tt::tt_metal
