// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "ttnn/tensor/tensor_ops.hpp"
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {
namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::ThrowsMessage;

using MeshTensorTest = GenericMeshDeviceFixture;
using MeshTensorTest2x4 = MeshDevice2x4Fixture;

TEST(MeshTensorHostTest, ToHostAlreadyOnHost) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_host_tensor = Tensor::from_vector(std::vector<float>(shape.volume()), tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::HOST);

    EXPECT_ANY_THROW(tensor_impl::to_host(input_host_tensor));
}

TEST(MeshTensorHostTest, FromHostShardsDifferentSpecs) {
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards = {
                Tensor::from_vector(
                    std::vector<float>(7),
                    TensorSpec(ttnn::Shape{7}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
                Tensor::from_vector(
                    std::vector<float>(10),
                    TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
            };
            from_host_shards(shards, MeshShape(1, 2));
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All tensor shards must have the same tensor spec")));
}

TEST_F(MeshTensorTest, FromHostShardsDeviceStorage) {
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards = {
                Tensor::from_vector(
                    std::vector<float>(10),
                    TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})),
                    mesh_device_.get()),
                Tensor::from_vector(
                    std::vector<float>(10),
                    TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})),
                    mesh_device_.get()),
            };
            from_host_shards(shards, MeshShape(1, 2));
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All tensor shards must be on host")));
}

TEST(MeshTensorHostTest, FromHostShardsMeshShapeMismatch) {
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards = {
                Tensor::from_vector(
                    std::vector<float>(7),
                    TensorSpec(ttnn::Shape{7}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
                Tensor::from_vector(
                    std::vector<float>(10),
                    TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
            };
            from_host_shards(shards, MeshShape(1, 3));
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Number of tensor shards must match mesh size")));
}

TEST(MeshTensorHostTest, FromHostShards) {
    std::vector<float> host_data1(10);
    std::iota(host_data1.begin(), host_data1.end(), 0);

    std::vector<float> host_data2(10);
    std::iota(host_data2.begin(), host_data2.end(), 10);

    auto tensor = from_host_shards(
        std::vector<Tensor>{
            Tensor::from_vector(
                host_data1,
                TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
            Tensor::from_vector(
                host_data2,
                TensorSpec(ttnn::Shape{10}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}))),
        },
        MeshShape(1, 2));

    EXPECT_EQ(tensor.tensor_spec().logical_shape(), ttnn::Shape{10});
    EXPECT_EQ(tensor.storage_type(), StorageType::HOST);
    EXPECT_EQ(tensor.tensor_topology(), TensorTopology::create_sharded_tensor_topology(MeshShape(1, 2)));

    auto tensors = get_device_tensors(tensor);
    ASSERT_THAT(tensors, SizeIs(2));
    EXPECT_THAT(tensors[0].to_vector<float>(), Pointwise(FloatEq(), host_data1));
    EXPECT_THAT(tensors[1].to_vector<float>(), Pointwise(FloatEq(), host_data2));
}

TEST_F(MeshTensorTest, Lifecycle) {
    const TensorSpec tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    Tensor input_tensor = create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(input_tensor.is_allocated());

    const auto& storage = input_tensor.storage();
    const auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&storage);

    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);

    // Buffer address is the same across all device buffers.
    const auto& view = mesh_device_->get_view();
    const auto buffer_address = device_storage->mesh_buffer->address();

    for (auto* device : view.get_devices()) {
        auto coordinate = view.find_device(device->id());
        auto* buffer = device_storage->mesh_buffer->get_device_buffer(coordinate);

        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_allocated());
        EXPECT_EQ(buffer->address(), buffer_address);
    }

    input_tensor.deallocate();
    EXPECT_FALSE(input_tensor.is_allocated());
}

TEST_F(MeshTensorTest, ToDeviceMemoryConfigOverride) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{BufferType::L1}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::HOST);
    EXPECT_EQ(input_host_tensor.tensor_spec().memory_config().buffer_type(), BufferType::L1);

    Tensor device_tensor_default = tensor_impl::to_device(input_host_tensor, mesh_device_.get());
    EXPECT_EQ(device_tensor_default.tensor_spec().memory_config().buffer_type(), BufferType::L1);

    Tensor device_tensor_dram =
        tensor_impl::to_device(input_host_tensor, mesh_device_.get(), MemoryConfig{BufferType::DRAM});
    EXPECT_EQ(device_tensor_dram.tensor_spec().memory_config().buffer_type(), BufferType::DRAM);
}

TEST_F(MeshTensorTest, ReplicateHostStorageTensor) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    // Prepare host tensor to offload on device.
    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::HOST);
    EXPECT_EQ(input_host_tensor.tensor_spec().logical_shape(), shape);

    // Write host tensor to device.
    Tensor device_tensor = tensor_impl::to_device(input_host_tensor, mesh_device_.get(), MemoryConfig{});
    EXPECT_EQ(device_tensor.tensor_spec().logical_shape(), shape);
    EXPECT_EQ(
        device_tensor.tensor_topology(),
        TensorTopology::create_fully_replicated_tensor_topology(mesh_device_->shape()));

    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);
    EXPECT_THAT(device_storage->coords, SizeIs(mesh_device_->num_devices()));

    // Read the tensor back, and compare it with input data.
    Tensor output_host_tensor = tensor_impl::to_host(device_tensor);
    EXPECT_TRUE(output_host_tensor.storage_type() == StorageType::HOST);
    EXPECT_EQ(output_host_tensor.tensor_spec().logical_shape(), shape);

    for (const auto& tensor : get_device_tensors(output_host_tensor)) {
        EXPECT_EQ(tensor.tensor_spec().logical_shape(), shape);
        EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatEq(), host_data));
    }
}

TEST_F(MeshTensorTest, GetDeviceTensors) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);

    Tensor device_tensor = tensor_impl::to_device(input_host_tensor, mesh_device_.get());
    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);
    EXPECT_THAT(device_storage->coords, SizeIs(mesh_device_->num_devices()));

    // Validate each tensor shard.
    std::vector<Tensor> device_tensors = get_device_tensors(device_tensor);
    std::vector<distributed::MeshCoordinate> device_shard_coords;
    EXPECT_THAT(device_tensors, SizeIs(mesh_device_->num_devices()));
    for (const auto& tensor_shard : device_tensors) {
        const auto* shard_storage = std::get_if<tt::tt_metal::DeviceStorage>(&tensor_shard.storage());
        ASSERT_NE(shard_storage, nullptr);
        EXPECT_NE(shard_storage->mesh_buffer, nullptr);
        EXPECT_THAT(shard_storage->coords, SizeIs(1));
        device_shard_coords.push_back(shard_storage->coords.front());
        EXPECT_THAT(tensor_shard.to_vector<float>(), Pointwise(FloatEq(), host_data));
    }

    // Expect coordiantes to cover the entire mesh.
    std::vector<::testing::Matcher<distributed::MeshCoordinate>> coord_matchers;
    for (const auto& expected_coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
        coord_matchers.push_back(Eq(expected_coord));
    }
    EXPECT_THAT(device_shard_coords, ElementsAreArray(coord_matchers));
}

TEST_F(MeshTensorTest2x4, CombineDeviceTensors) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);

    Tensor device_tensor1 = tensor_impl::to_device(input_host_tensor, mesh_device_.get());
    Tensor device_tensor2 = tensor_impl::to_device(input_host_tensor, mesh_device_.get());

    auto device_tensors1 = get_device_tensors(device_tensor1);
    auto device_tensors2 = get_device_tensors(device_tensor2);

    EXPECT_THAT(device_tensors1, SizeIs(mesh_device_->num_devices()));
    EXPECT_THAT(device_tensors2, SizeIs(mesh_device_->num_devices()));

    // Try to aggregate shards from different mesh buffers.
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards_to_aggregate = {device_tensors1[0], device_tensors2[1]};
            combine_device_tensors(shards_to_aggregate);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("tensor shards must be allocated on the same mesh buffer.")));

    // Try to aggregate the same shard twice
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards_to_aggregate = {device_tensors1[0], device_tensors1[0]};
            combine_device_tensors(shards_to_aggregate);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Found a tensor shard at duplicate coordinate")));

    // Aggregate every second shard into a new mesh tensor.
    int shard_dim = 2;
    auto partial_tensor = combine_device_tensors(
        std::vector<Tensor>{device_tensors1[6], device_tensors1[4], device_tensors1[2], device_tensors1[0]}, shard_dim);

    auto* partial_device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&partial_tensor.storage());
    ASSERT_NE(partial_device_storage, nullptr);
    EXPECT_NE(partial_device_storage->mesh_buffer, nullptr);

    EXPECT_EQ(partial_tensor.tensor_topology().distribution_shape(), MeshShape(4));
    EXPECT_EQ(
        std::get<distributed::MeshMapperConfig::Shard>(partial_tensor.tensor_topology().placements()[0]).dim,
        shard_dim);

    // Validate the shards are sorted, and are as expected.
    ASSERT_THAT(partial_device_storage->coords, SizeIs(4));
    EXPECT_EQ(partial_device_storage->coords[0], (distributed::MeshCoordinate{0, 0}));
    EXPECT_EQ(partial_device_storage->coords[1], (distributed::MeshCoordinate{0, 2}));
    EXPECT_EQ(partial_device_storage->coords[2], (distributed::MeshCoordinate{1, 0}));
    EXPECT_EQ(partial_device_storage->coords[3], (distributed::MeshCoordinate{1, 2}));
}

struct MeshTensorWriteTestParams {
    ttnn::Shape shape;

    // If true, uses pre-allocated tensor APIs (create_device_tensor/device + copy_to_device/host).
    bool use_pre_allocated_tensor_api = false;

    // Shape of the resulting shards.
    ttnn::Shape sharded_shape;
    std::vector<distributed::MeshCoordinate> expected_coords;
    std::function<std::unique_ptr<ttnn::distributed::TensorToMesh>(MeshDevice*)> get_mapper;
};

class MeshTensorWriteTest : public MeshTensorTest2x4,
                            public ::testing::WithParamInterface<MeshTensorWriteTestParams> {};

TEST_P(MeshTensorWriteTest, WriteMultiDeviceHostTensor) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_EQ(num_devices, 8);

    const ttnn::Shape shape = GetParam().shape;
    const ttnn::Shape sharded_shape = GetParam().sharded_shape;

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
    EXPECT_TRUE(input_host_tensor_sharded.storage_type() == StorageType::HOST);
    EXPECT_EQ(input_host_tensor_sharded.tensor_spec().logical_shape(), sharded_shape);

    std::vector<Tensor> input_host_shards = get_device_tensors(input_host_tensor_sharded);

    auto device_tensor = [&]() {
        if (GetParam().use_pre_allocated_tensor_api) {
            Tensor device_tensor = create_device_tensor(input_host_shards.at(0).tensor_spec(), mesh_device_.get());
            tensor_impl::copy_to_device(input_host_tensor_sharded, device_tensor);
            return device_tensor;
        }
        return tensor_impl::to_device(input_host_tensor_sharded, mesh_device_.get());
    }();

    EXPECT_EQ(device_tensor.tensor_topology(), input_host_tensor_sharded.tensor_topology());

    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_THAT(device_storage->coords, ElementsAreArray(coord_matchers));

    auto output_host_tensor = [&]() {
        if (GetParam().use_pre_allocated_tensor_api) {
            Tensor host_tensor = allocate_tensor_on_host(device_tensor.tensor_spec(), mesh_device_.get());
            tensor_impl::copy_to_host(device_tensor, host_tensor, /*blocking=*/true);
            return host_tensor;
        }
        return tensor_impl::to_host(device_tensor);
    }();

    EXPECT_EQ(output_host_tensor.tensor_topology(), input_host_tensor_sharded.tensor_topology());

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
            .sharded_shape = ttnn::Shape{1, 1, 32, 32},
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
            .shape = ttnn::Shape{1, 1, 32, 32},
            .sharded_shape = ttnn::Shape{1, 1, 32, 32},
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
            .sharded_shape = ttnn::Shape{7, 1, 32, 32},
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
                    return create_mesh_mapper(
                        *device,
                        MeshMapperConfig{
                            .placements =
                                {
                                    MeshMapperConfig::Replicate(),
                                    MeshMapperConfig::Shard(1),
                                },
                            .mesh_shape_override = MeshShape(2, 3),
                        });
                },
        },
    };

    std::vector<MeshTensorWriteTestParams> params;
    for (auto param : base_params) {
        param.use_pre_allocated_tensor_api = false;
        params.push_back(param);
        param.use_pre_allocated_tensor_api = true;
        params.push_back(param);
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MeshTensorWriteTest, MeshTensorWriteTest, ::testing::ValuesIn(get_mesh_tensor_write_test_params()));

}  // namespace
}  // namespace ttnn::distributed::test
