// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {
namespace {

using ::testing::FloatEq;
using ::testing::Pointwise;

using MeshTensorTest = T3kMultiDeviceFixture;

TEST_F(MeshTensorTest, Lifecycle) {
    const TensorSpec tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    Tensor input_tensor = allocate_tensor_on_mesh(tensor_spec, mesh_device_.get());

    EXPECT_EQ(input_tensor.workers.size(), mesh_device_->num_devices());
    EXPECT_TRUE(input_tensor.is_allocated());

    const auto& storage = input_tensor.get_storage();
    auto* multi_device_storage = std::get_if<tt::tt_metal::MultiDeviceStorage>(&storage);

    ASSERT_NE(multi_device_storage, nullptr);
    EXPECT_NE(multi_device_storage->mesh_buffer, nullptr);

    // Buffer address is the same across all device buffers.
    const auto buffer_address = multi_device_storage->mesh_buffer->address();
    for (auto* device : mesh_device_->get_devices()) {
        auto buffer = multi_device_storage->get_buffer_for_device(device);
        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_allocated());
        EXPECT_EQ(buffer->address(), buffer_address);
    }

    input_tensor.deallocate();
    EXPECT_FALSE(input_tensor.is_allocated());
}

using MeshTensorDeviceTest = T3kMultiDeviceFixture;

TEST_F(MeshTensorDeviceTest, ToHostNonMeshTensor) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_host_tensor = Tensor::from_vector(std::vector<float>(shape.volume()), tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::OWNED);

    EXPECT_ANY_THROW(tensor_impl::to_host_mesh_tensor_wrapper(input_host_tensor));
}

TEST_F(MeshTensorDeviceTest, ReplicateHostTensor) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    // Prepare host tensor to offload on device.
    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::OWNED);
    EXPECT_EQ(input_host_tensor.get_tensor_spec().logical_shape(), shape);

    // Write host tensor to device.
    Tensor device_tensor =
        tensor_impl::to_device_mesh_tensor_wrapper(input_host_tensor, mesh_device_.get(), MemoryConfig{});
    EXPECT_TRUE(distributed::is_mesh_buffer_tensor(device_tensor));
    EXPECT_EQ(device_tensor.get_tensor_spec().logical_shape(), shape);

    auto* multi_device_storage = std::get_if<tt::tt_metal::MultiDeviceStorage>(&device_tensor.get_storage());
    ASSERT_NE(multi_device_storage, nullptr);
    for (const auto& [_, shard_spec] : multi_device_storage->specs) {
        EXPECT_EQ(shard_spec.logical_shape(), shape);
    }
    EXPECT_TRUE(std::holds_alternative<tt::tt_metal::ReplicateTensor>(multi_device_storage->strategy));

    // Read the tensor back, and compare it with input data.
    Tensor output_host_tensor = tensor_impl::to_host_mesh_tensor_wrapper(device_tensor);
    EXPECT_TRUE(output_host_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST);
    EXPECT_EQ(output_host_tensor.get_tensor_spec().logical_shape(), shape);

    for (const auto& tensor : get_tensors_from_multi_device_storage(output_host_tensor)) {
        EXPECT_EQ(tensor.get_tensor_spec().logical_shape(), shape);
        EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatEq(), host_data));
    }
}

TEST_F(MeshTensorDeviceTest, WriteMultiDeviceHostTensor) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_EQ(num_devices, 8);
    // Test uneven shard shapes.
    const ttnn::Shape shape{1, 9, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    // Prepare multi-device host tensor to offload on device.
    Tensor input_host_tensor_sharded = distribute_tensor(
        Tensor::from_vector(host_data, tensor_spec), *shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/1));
    EXPECT_TRUE(input_host_tensor_sharded.storage_type() == StorageType::MULTI_DEVICE_HOST);

    auto* multi_device_host_storage =
        std::get_if<tt::tt_metal::MultiDeviceHostStorage>(&input_host_tensor_sharded.get_storage());
    ASSERT_NE(multi_device_host_storage, nullptr);
    const auto* strategy = std::get_if<tt::tt_metal::ShardTensor>(&multi_device_host_storage->strategy);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->shard_dimension, 1);

    // Write host tensor to device.
    Tensor device_tensor =
        tensor_impl::to_device_mesh_tensor_wrapper(input_host_tensor_sharded, mesh_device_.get(), MemoryConfig{});
    EXPECT_TRUE(distributed::is_mesh_buffer_tensor(device_tensor));

    auto* multi_device_storage = std::get_if<tt::tt_metal::MultiDeviceStorage>(&device_tensor.get_storage());
    ASSERT_NE(multi_device_storage, nullptr);
    const auto* device_tensor_strategy = std::get_if<tt::tt_metal::ShardTensor>(&multi_device_storage->strategy);
    ASSERT_NE(device_tensor_strategy, nullptr);
    EXPECT_EQ(device_tensor_strategy->shard_dimension, 1);

    // Read the tensor back, and compare it with input data.
    Tensor output_host_tensor = aggregate_tensor(
        tensor_impl::to_host_mesh_tensor_wrapper(device_tensor), *concat_mesh_to_tensor_composer(/*dim=*/1));
    EXPECT_TRUE(output_host_tensor.storage_type() == StorageType::OWNED);
    EXPECT_EQ(output_host_tensor.get_tensor_spec().logical_shape(), shape);

    EXPECT_THAT(output_host_tensor.to_vector<float>(), Pointwise(FloatEq(), host_data));
}

}  // namespace
}  // namespace ttnn::distributed::test
