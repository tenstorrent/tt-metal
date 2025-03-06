// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "gmock/gmock.h"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

using MeshTensorTest = T3kMultiDeviceFixture;

TEST_F(MeshTensorTest, Lifecycle) {
    const TensorSpec tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    Tensor input_tensor = allocate_tensor_on_mesh(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(input_tensor.is_allocated());

    const auto& storage = input_tensor.get_storage();
    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&storage);

    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);

    // Buffer address is the same across all device buffers.
    const auto& view = mesh_device_->get_view();
    const auto buffer_address = device_storage->mesh_buffer->address();

    for (auto* device : view.get_devices()) {
        auto coordinate = view.find_device(device->id());
        auto buffer = device_storage->mesh_buffer->get_device_buffer(coordinate);

        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_allocated());
        EXPECT_EQ(buffer->address(), buffer_address);
    }

    input_tensor.deallocate();
    EXPECT_FALSE(input_tensor.is_allocated());
}

TEST_F(MeshTensorTest, ToHostNonMeshTensor) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_host_tensor = Tensor::from_vector(std::vector<float>(shape.volume()), tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::OWNED);

    EXPECT_ANY_THROW(tensor_impl::to_host_mesh_tensor_wrapper(input_host_tensor));
}

TEST_F(MeshTensorTest, ReplicateOwnedTensor) {
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

    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.get_storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);
    for (const auto& [coord, spec] : device_storage->specs) {
        EXPECT_THAT(spec.logical_shape(), Eq(ttnn::Shape{1, 1, 32, 32}));
    }

    // Read the tensor back, and compare it with input data.
    Tensor output_host_tensor = tensor_impl::to_host_mesh_tensor_wrapper(device_tensor);
    EXPECT_TRUE(output_host_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST);
    EXPECT_EQ(output_host_tensor.get_tensor_spec().logical_shape(), shape);

    for (const auto& tensor : get_tensors_from_multi_device_storage(output_host_tensor)) {
        EXPECT_EQ(tensor.get_tensor_spec().logical_shape(), shape);
        EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatEq(), host_data));
    }
}

struct MeshTensorWriteTestParams {
    ttnn::Shape shape;
    bool use_pre_allocated_tensor = false;
    std::vector<ttnn::Shape> expected_shapes;
    std::vector<distributed::MeshCoordinate> expected_coords;
    std::function<std::unique_ptr<ttnn::distributed::TensorToMesh>(MeshDevice*)> get_mapper;
};

class MeshTensorWriteTest : public T3kMultiDeviceFixture,
                            public ::testing::WithParamInterface<MeshTensorWriteTestParams> {};

TEST_P(MeshTensorWriteTest, WriteMultiDeviceHostTensor) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_EQ(num_devices, 8);

    const ttnn::Shape shape = GetParam().shape;

    std::vector<::testing::Matcher<ttnn::Shape>> shape_matchers;
    for (const auto& expected_shape : GetParam().expected_shapes) {
        shape_matchers.push_back(Eq(expected_shape));
    }

    std::vector<::testing::Matcher<distributed::MeshCoordinate>> coord_matchers;
    for (const auto& expected_coord : GetParam().expected_coords) {
        coord_matchers.push_back(Eq(expected_coord));
    }

    const auto mapper = GetParam().get_mapper(mesh_device_.get());

    // Prepare multi-device host tensor to offload on device.
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);
    Tensor input_host_tensor_sharded = distribute_tensor(Tensor::from_vector(host_data, tensor_spec), *mapper);
    EXPECT_TRUE(input_host_tensor_sharded.storage_type() == StorageType::MULTI_DEVICE_HOST);
    std::vector<Tensor> input_host_shards = get_device_tensors(input_host_tensor_sharded);

    auto* multi_device_host_storage =
        std::get_if<tt::tt_metal::MultiDeviceHostStorage>(&input_host_tensor_sharded.get_storage());
    ASSERT_NE(multi_device_host_storage, nullptr);
    EXPECT_EQ(multi_device_host_storage->strategy, mapper->config());

    auto device_tensor = [&]() {
        if (GetParam().use_pre_allocated_tensor) {
            Tensor device_tensor =
                allocate_tensor_on_mesh(input_host_shards.at(0).get_tensor_spec(), mesh_device_.get());
            write_tensor(input_host_tensor_sharded, device_tensor);
            return device_tensor;
        } else {
            return tensor_impl::to_device_mesh_tensor_wrapper(
                input_host_tensor_sharded, mesh_device_.get(), MemoryConfig{});
        }
    }();

    EXPECT_TRUE(distributed::is_mesh_buffer_tensor(device_tensor));

    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.get_storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_EQ(device_storage->strategy, mapper->config());

    std::vector<distributed::MeshCoordinate> device_shard_coords;
    std::vector<ttnn::Shape> device_shard_shapes;
    for (const auto& [coord, spec] : device_storage->specs) {
        device_shard_coords.push_back(coord);
        device_shard_shapes.push_back(spec.logical_shape());
    }
    EXPECT_THAT(device_shard_shapes, ElementsAreArray(shape_matchers));
    EXPECT_THAT(device_shard_coords, ElementsAreArray(coord_matchers));

    // Read the tensor back, and compare it with input data.
    auto output_host_tensor = tensor_impl::to_host_mesh_tensor_wrapper(device_tensor);
    auto* output_multi_device_host_storage =
        std::get_if<tt::tt_metal::MultiDeviceHostStorage>(&output_host_tensor.get_storage());
    ASSERT_NE(output_multi_device_host_storage, nullptr);
    EXPECT_EQ(output_multi_device_host_storage->strategy, mapper->config());
    std::vector<ttnn::Shape> output_host_shapes;
    for (const auto& spec : output_multi_device_host_storage->specs) {
        output_host_shapes.push_back(spec.logical_shape());
    }
    EXPECT_THAT(output_host_shapes, ElementsAreArray(shape_matchers));

    std::vector<Tensor> output_host_shards = get_device_tensors(output_host_tensor);
    ASSERT_EQ(output_host_shards.size(), input_host_shards.size());
    for (int i = 0; i < output_host_shards.size(); i++) {
        EXPECT_THAT(
            output_host_shards[i].to_vector<float>(), Pointwise(FloatEq(), input_host_shards[i].to_vector<float>()));
    }
}

// Returns a vector of `MeshTensorWriteTestParams`, with and without `use_pre_allocated_tensor` set to true.
auto get_mesh_tensor_write_test_params() {
    std::vector<MeshTensorWriteTestParams> base_params = {
        MeshTensorWriteTestParams{
            .shape = ttnn::Shape{1, 8, 32, 32},
            .expected_shapes =
                {ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32}},
            .expected_coords =
                {distributed::MeshCoordinate{0, 0},
                 distributed::MeshCoordinate{0, 1},
                 distributed::MeshCoordinate{0, 2},
                 distributed::MeshCoordinate{0, 3},
                 distributed::MeshCoordinate{1, 0},
                 distributed::MeshCoordinate{1, 1},
                 distributed::MeshCoordinate{1, 2},
                 distributed::MeshCoordinate{1, 3}},
            .get_mapper = [](MeshDevice* device) { return shard_tensor_to_mesh_mapper(*device, 1); },
        },
        MeshTensorWriteTestParams{
            .shape = ttnn::Shape{1, 9, 32, 32},
            .expected_shapes =
                {ttnn::Shape{1, 2, 32, 32},
                 ttnn::Shape{1, 2, 32, 32},
                 ttnn::Shape{1, 2, 32, 32},
                 ttnn::Shape{1, 2, 32, 32},
                 ttnn::Shape{1, 1, 32, 32}},
            .expected_coords =
                {distributed::MeshCoordinate{0, 0},
                 distributed::MeshCoordinate{0, 1},
                 distributed::MeshCoordinate{0, 2},
                 distributed::MeshCoordinate{0, 3},
                 distributed::MeshCoordinate{1, 0}},
            .get_mapper = [](MeshDevice* device) { return shard_tensor_to_mesh_mapper(*device, 1); },
        },
        MeshTensorWriteTestParams{
            .shape = ttnn::Shape{1, 1, 32, 32},
            .expected_shapes =
                {ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32},
                 ttnn::Shape{1, 1, 32, 32}},
            .expected_coords =
                {distributed::MeshCoordinate{0, 0},
                 distributed::MeshCoordinate{0, 1},
                 distributed::MeshCoordinate{0, 2},
                 distributed::MeshCoordinate{0, 3},
                 distributed::MeshCoordinate{1, 0},
                 distributed::MeshCoordinate{1, 1},
                 distributed::MeshCoordinate{1, 2},
                 distributed::MeshCoordinate{1, 3}},
            .get_mapper = [](MeshDevice* device) { return replicate_tensor_to_mesh_mapper(*device); },
        },
        MeshTensorWriteTestParams{
            .shape = ttnn::Shape{7, 3, 32, 32},
            .expected_shapes =
                {ttnn::Shape{7, 1, 32, 32},
                 ttnn::Shape{7, 1, 32, 32},
                 ttnn::Shape{7, 1, 32, 32},
                 ttnn::Shape{7, 1, 32, 32},
                 ttnn::Shape{7, 1, 32, 32},
                 ttnn::Shape{7, 1, 32, 32}},
            .expected_coords =
                {distributed::MeshCoordinate{0, 0},
                 distributed::MeshCoordinate{0, 1},
                 distributed::MeshCoordinate{0, 2},
                 distributed::MeshCoordinate{1, 0},
                 distributed::MeshCoordinate{1, 1},
                 distributed::MeshCoordinate{1, 2}},
            .get_mapper =
                [](MeshDevice* device) {
                    // Replicate to a submesh 2x3
                    // Replicate within each row, then split by second dimension.
                    return shard_tensor_to_2d_mesh_mapper(*device, MeshShape{2, 3}, Shard2dConfig{std::nullopt, 1});
                },
        },
    };

    std::vector<MeshTensorWriteTestParams> params;
    for (auto param : base_params) {
        param.use_pre_allocated_tensor = false;
        params.push_back(param);
        param.use_pre_allocated_tensor = true;
        params.push_back(param);
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MeshTensorWriteTest, MeshTensorWriteTest, ::testing::ValuesIn(get_mesh_tensor_write_test_params()));

}  // namespace
}  // namespace ttnn::distributed::test
