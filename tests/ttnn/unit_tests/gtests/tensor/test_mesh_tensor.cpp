// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>

#include "ttnn/tensor/tensor_ops.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace ttnn::distributed::test {
namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::ThrowsMessage;

using MeshTensorTest = GenericMeshDeviceFixture;
using MeshTensorTest1x2 = MeshDevice1x2Fixture;
using MeshTensorTest2x4 = MeshDevice2x4Fixture;

TEST(MeshTensorHostTest, ToHostAlreadyOnHost) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_host_tensor = Tensor::from_vector(std::vector<float>(shape.volume()), tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == StorageType::HOST);

    EXPECT_NO_THROW(cpu(input_host_tensor));
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

    EXPECT_NO_THROW({ input_tensor.mesh_buffer(); });

    // Buffer address is the same across all device buffers.
    const auto& view = mesh_device_->get_view();
    const auto buffer_address = input_tensor.mesh_buffer().address();

    for (auto* device : view.get_devices()) {
        auto coordinate = view.find_device(device->id());
        auto* buffer = input_tensor.mesh_buffer().get_device_buffer(coordinate);

        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_allocated());
        EXPECT_EQ(buffer->address(), buffer_address);
    }

    input_tensor.deallocate();
    EXPECT_FALSE(input_tensor.is_allocated());
    EXPECT_THROW({ input_tensor.mesh_buffer(); }, std::runtime_error);
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

    Tensor device_tensor_default = to_device(input_host_tensor, mesh_device_.get());
    EXPECT_EQ(device_tensor_default.tensor_spec().memory_config().buffer_type(), BufferType::L1);

    Tensor device_tensor_dram = to_device(input_host_tensor, mesh_device_.get(), MemoryConfig{BufferType::DRAM});
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
    Tensor device_tensor = to_device(input_host_tensor, mesh_device_.get(), MemoryConfig{});
    EXPECT_EQ(device_tensor.tensor_spec().logical_shape(), shape);
    EXPECT_EQ(
        device_tensor.tensor_topology(),
        TensorTopology::create_fully_replicated_tensor_topology(mesh_device_->shape()));

    const auto& device_storage = device_tensor.device_storage();
    EXPECT_NO_THROW({ device_storage.get_mesh_buffer(); });
    EXPECT_THAT(device_storage.get_coords(), SizeIs(mesh_device_->num_devices()));

    // Read the tensor back, and compare it with input data.
    Tensor output_host_tensor = cpu(device_tensor);
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

    Tensor device_tensor = to_device(input_host_tensor, mesh_device_.get());
    const auto& device_storage = device_tensor.device_storage();
    EXPECT_NO_THROW({ device_storage.get_mesh_buffer(); });
    EXPECT_TRUE(device_storage.is_allocated());
    EXPECT_THAT(device_storage.get_coords(), SizeIs(mesh_device_->num_devices()));

    // Validate each tensor shard.
    std::vector<Tensor> device_tensors = get_device_tensors(device_tensor);
    std::vector<distributed::MeshCoordinate> device_shard_coords;
    EXPECT_THAT(device_tensors, SizeIs(mesh_device_->num_devices()));
    for (const auto& tensor_shard : device_tensors) {
        const auto& shard_storage = tensor_shard.device_storage();
        EXPECT_NO_THROW({ shard_storage.get_mesh_buffer(); });
        EXPECT_TRUE(shard_storage.is_allocated());
        EXPECT_THAT(shard_storage.get_coords(), SizeIs(1));
        device_shard_coords.push_back(shard_storage.get_coords().front());
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

    Tensor device_tensor1 = to_device(input_host_tensor, mesh_device_.get());
    Tensor device_tensor2 = to_device(input_host_tensor, mesh_device_.get());

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

    const auto& partial_device_storage = partial_tensor.device_storage();
    EXPECT_NO_THROW({ partial_device_storage.get_mesh_buffer(); });

    EXPECT_EQ(partial_tensor.tensor_topology().distribution_shape(), MeshShape(4));
    EXPECT_EQ(
        std::get<distributed::MeshMapperConfig::Shard>(partial_tensor.tensor_topology().placements()[0]).dim,
        shard_dim);

    // Validate the shards are sorted, and are as expected.
    ASSERT_THAT(partial_device_storage.get_coords(), SizeIs(4));
    EXPECT_EQ(partial_device_storage.get_coords()[0], (distributed::MeshCoordinate{0, 0}));
    EXPECT_EQ(partial_device_storage.get_coords()[1], (distributed::MeshCoordinate{0, 2}));
    EXPECT_EQ(partial_device_storage.get_coords()[2], (distributed::MeshCoordinate{1, 0}));
    EXPECT_EQ(partial_device_storage.get_coords()[3], (distributed::MeshCoordinate{1, 2}));
}

// This is the mini version of MeshTensorTest2x4.CombineDeviceTensors above.
TEST_F(MeshTensorTest1x2, CombineDeviceTensorsMini) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);

    Tensor device_tensor1 = to_device(input_host_tensor, mesh_device_.get());
    Tensor device_tensor2 = to_device(input_host_tensor, mesh_device_.get());

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

    // Try to aggregate the same shard twice.
    EXPECT_THAT(
        ([&]() {
            std::vector<Tensor> shards_to_aggregate = {device_tensors1[0], device_tensors1[0]};
            combine_device_tensors(shards_to_aggregate);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Found a tensor shard at duplicate coordinate")));

    // Aggregate both shards in reverse order; verify coords come out sorted.
    auto partial_tensor = combine_device_tensors(std::vector<Tensor>{device_tensors1[1], device_tensors1[0]});

    const auto& partial_device_storage = partial_tensor.device_storage();
    EXPECT_NO_THROW({ partial_device_storage.get_mesh_buffer(); });

    ASSERT_THAT(partial_device_storage.get_coords(), SizeIs(2));
    EXPECT_EQ(partial_device_storage.get_coords()[0], (distributed::MeshCoordinate{0, 0}));
    EXPECT_EQ(partial_device_storage.get_coords()[1], (distributed::MeshCoordinate{0, 1}));
}

TEST_F(MeshTensorTest2x4, CombineDeviceTensorsWithDifferentShardDims) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    Tensor input_host_tensor = Tensor::from_vector(host_data, tensor_spec);
    Tensor device_tensor = to_device(input_host_tensor, mesh_device_.get());
    auto device_tensors = get_device_tensors(device_tensor);
    ASSERT_THAT(device_tensors, SizeIs(mesh_device_->num_devices()));

    const int num_shards = static_cast<int>(device_tensors.size());
    const MeshShape expected_distribution_shape(num_shards);

    for (int shard_dim : {0, 3}) {
        auto combined = combine_device_tensors(device_tensors, shard_dim);
        EXPECT_EQ(combined.tensor_topology().distribution_shape(), expected_distribution_shape);
        EXPECT_EQ(
            std::get<distributed::MeshMapperConfig::Shard>(combined.tensor_topology().placements()[0]).dim, shard_dim);
    }
}

TEST_F(MeshTensorTest, DefaultConstructedDeviceStorageGetters) {
    tt::tt_metal::DeviceStorage storage;

    EXPECT_THAT(([&]() { storage.get_buffer(); }), ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(([&]() { storage.get_mesh_buffer(); }), ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(([&]() { storage.get_mesh_tensor(); }), ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(([&]() { storage.get_tensor_spec(); }), ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(
        ([&]() { storage.get_tensor_topology(); }), ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(
        ([&]() { storage.get_mesh_buffer_leak_ownership(); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));
    EXPECT_THAT(
        ([&]() { storage.get_device_bypass_deallocate_check(); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("not allocated")));

    EXPECT_FALSE(storage.is_allocated());
    EXPECT_TRUE(storage.is_uniform_storage());
}

TEST_F(MeshTensorTest2x4, CombineDeviceTensorsShardDimValidation) {
    const ttnn::Shape shape{1, 1, 32, 32};
    const TensorSpec tensor_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    Tensor input_host_tensor = Tensor::from_vector(std::vector<float>(shape.volume()), tensor_spec);
    Tensor device_tensor = to_device(input_host_tensor, mesh_device_.get());
    auto device_tensors = get_device_tensors(device_tensor);
    ASSERT_THAT(device_tensors, SizeIs(mesh_device_->num_devices()));

    EXPECT_THAT(
        ([&]() {
            const int invalid_shard_dim = static_cast<int>(shape.rank());
            combine_device_tensors(std::vector<Tensor>{device_tensors[0]}, invalid_shard_dim);
        }),
        ThrowsMessage<std::runtime_error>(HasSubstr("shard_dim")));
    EXPECT_THAT(
        ([&]() { combine_device_tensors(std::vector<Tensor>{device_tensors[0]}, /*shard_dim=*/-1); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("shard_dim")));
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
            copy_to_device(input_host_tensor_sharded, device_tensor);
            return device_tensor;
        }
        return to_device(input_host_tensor_sharded, mesh_device_.get());
    }();

    EXPECT_EQ(device_tensor.tensor_topology(), input_host_tensor_sharded.tensor_topology());

    const auto& device_storage = device_tensor.device_storage();
    EXPECT_THAT(device_storage.get_coords(), ElementsAreArray(coord_matchers));

    auto output_host_tensor = [&]() {
        if (GetParam().use_pre_allocated_tensor_api) {
            Tensor host_tensor = allocate_tensor_on_host(device_tensor.tensor_spec(), mesh_device_.get());
            copy_to_host(device_tensor, host_tensor, /*blocking=*/true);
            return host_tensor;
        }
        return cpu(device_tensor);
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

// ======================================================================================
//                    Data Movement Tests (2x2 mesh, Runtime Tensor)
// ======================================================================================

class MeshDevice2x2Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x2Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 2}}) {}
};

using MeshTensorDataMovementTest = MeshDevice2x2Fixture;

using tt::tt_metal::DistributedHostBuffer;
using tt::tt_metal::HostBuffer;
using tt::tt_metal::HostTensor;
using tt::tt_metal::MeshTensor;
using tt::tt_metal::TensorTopology;
namespace tensor_impl = tt::tt_metal::tensor_impl;

// Helper: create a HostTensor with a single shard at [0,0].
HostTensor make_single_shard_host_tensor(const ttnn::Shape& shape, uint32_t fill) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    return HostTensor(HostBuffer(std::vector<uint32_t>(shape.volume(), fill)), spec, TensorTopology{});
}

// Helper: create a HostTensor with one shard per coordinate in a mesh.
// Each shard is filled with `volume` copies of `shard_fills[i]`.
HostTensor make_full_coverage_host_tensor(
    const ttnn::Shape& shape, const distributed::MeshShape& mesh_shape, const std::vector<uint32_t>& shard_fills) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    distributed::MeshCoordinateRange range(mesh_shape);
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        return HostBuffer(std::vector<uint32_t>(shape.volume(), shard_fills.at(idx++)));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_shape);
    return HostTensor(std::move(dhb), spec, topology);
}

// Helper: create a HostTensor with shards at only a subset of mesh coordinates.
// Each shard is filled with `volume` copies of `shard_fills[i]`.
HostTensor make_partial_coverage_host_tensor(
    const ttnn::Shape& shape,
    const distributed::MeshShape& mesh_shape,
    const std::vector<distributed::MeshCoordinate>& coords,
    const std::vector<uint32_t>& shard_fills) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        return HostBuffer(std::vector<uint32_t>(shape.volume(), shard_fills.at(idx++)));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(distributed::MeshShape(coords.size()));
    return HostTensor(std::move(dhb), spec, topology);
}

// Helper: assert that two HostTensors have the same populated coords and identical shard contents.
void expect_host_tensors_eq(const HostTensor& expected, const HostTensor& actual) {
    const auto& exp_coords = expected.buffer().shard_coords();
    const auto& act_coords = actual.buffer().shard_coords();
    ASSERT_EQ(exp_coords, act_coords);
    for (const auto& coord : exp_coords) {
        auto exp_shard = expected.buffer().get_shard(coord);
        auto act_shard = actual.buffer().get_shard(coord);
        ASSERT_TRUE(exp_shard.has_value());
        ASSERT_TRUE(act_shard.has_value());
        auto exp_span = exp_shard->view_as<uint32_t>();
        auto act_span = act_shard->view_as<uint32_t>();
        EXPECT_THAT(
            std::vector<uint32_t>(act_span.begin(), act_span.end()),
            Pointwise(Eq(), std::vector<uint32_t>(exp_span.begin(), exp_span.end())));
    }
}

// ---------------------------------------------------------------------------
//  is_uniform_write
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_FullCoverage) {
    const ttnn::Shape shape{1, 1, 32, 32};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {0, 0, 0, 0});
    EXPECT_TRUE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_NO_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_SingleShard) {
    const ttnn::Shape shape{1, 1, 32, 32};
    auto host_tensor = make_single_shard_host_tensor(shape, 1);
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_PartialCoverage) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {0, 1}};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, {0, 0});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_SmallerMeshShape) {
    const ttnn::Shape shape{1, 1, 32, 32};
    distributed::MeshShape smaller_mesh{1, 2};
    auto host_tensor = make_full_coverage_host_tensor(shape, smaller_mesh, {0, 0});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_EmptyDistributedHostBuffer) {
    const ttnn::Shape shape{1, 1, 32, 32};
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_device_->shape());
    HostTensor host_tensor(std::move(dhb), spec, TensorTopology{});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_));
}

// ---------------------------------------------------------------------------
//  Uniform to_device / to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformToDevice_ToHost_Roundtrip) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {10, 20, 30, 40};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result = tensor_impl::enqueue_read_mesh_tensor(cq, device_tensor);

    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Uniform copy_to_device / copy_to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformCopyToDevice_CopyToHost_Roundtrip) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {100, 200, 300, 400};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = tensor_impl::allocate_mesh_tensor(spec, *mesh_device_, host_tensor.tensor_topology());
    tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, device_tensor);

    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});
    tensor_impl::enqueue_read_mesh_tensor(cq, device_tensor, result);

    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Uniform copy_to_device rejects non-uniform host tensor
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformCopyToDevice_RejectsPartialCoverage) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {1, 1}};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, {0, 0});

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = tensor_impl::allocate_mesh_tensor(spec, *mesh_device_, TensorTopology{});
    EXPECT_ANY_THROW(tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, device_tensor));
}

// ---------------------------------------------------------------------------
//  Non-uniform to_device / to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToDevice_ToHost_Roundtrip) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {1, 0}};
    std::vector<uint32_t> shard_fills = {7, 13};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto [device_tensor, written_coords] =
        tensor_impl::non_uniform_data_movement::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_);
    ASSERT_EQ(written_coords.size(), coords.size());

    HostTensor result =
        tensor_impl::non_uniform_data_movement::enqueue_read_mesh_tensor(cq, device_tensor, written_coords);
    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform copy_to_device / copy_to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformCopyToDevice_CopyToHost_Roundtrip) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 1}, {1, 1}};
    std::vector<uint32_t> shard_fills = {42, 99};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = tensor_impl::allocate_mesh_tensor(spec, *mesh_device_, TensorTopology{});

    auto written_coords =
        tensor_impl::non_uniform_data_movement::enqueue_write_mesh_tensor(cq, host_tensor, device_tensor);
    ASSERT_EQ(written_coords.size(), coords.size());

    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : written_coords) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});
    tensor_impl::non_uniform_data_movement::enqueue_read_mesh_tensor(cq, device_tensor, result, written_coords);
    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform to_device with single shard (replicate)
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToDevice_SingleShard_Roundtrip) {
    const ttnn::Shape shape{1, 1, 32, 32};
    auto host_tensor = make_single_shard_host_tensor(shape, 55);

    auto& cq = mesh_device_->mesh_command_queue();
    auto [device_tensor, written_coords] =
        tensor_impl::non_uniform_data_movement::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result =
        tensor_impl::non_uniform_data_movement::enqueue_read_mesh_tensor(cq, device_tensor, written_coords);
    auto expected = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {55, 55, 55, 55});
    expect_host_tensors_eq(expected, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform D2H sheds extra shards from the host tensor
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToHost_ShedsExtraShards) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {10, 20, 30, 40};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_);

    std::vector<distributed::MeshCoordinate> subset = {{0, 1}, {1, 0}};
    HostTensor result = tensor_impl::non_uniform_data_movement::enqueue_read_mesh_tensor(cq, device_tensor, subset);

    ASSERT_EQ(result.buffer().shard_coords().size(), subset.size());
    auto expected = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), subset, {20, 30});
    expect_host_tensors_eq(expected, result);
}

TEST_F(MeshTensorDataMovementTest, NonUniformCopyToHost_ShedsExtraShards) {
    const ttnn::Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {5, 15, 25, 35};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = tensor_impl::enqueue_write_mesh_tensor(cq, host_tensor, *mesh_device_);

    // Pass a full-coverage host tensor as the destination, but only read back a subset of coords.
    std::vector<distributed::MeshCoordinate> subset = {{1, 1}};
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dest = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {0, 0, 0, 0});
    tensor_impl::non_uniform_data_movement::enqueue_read_mesh_tensor(cq, device_tensor, dest, subset);

    ASSERT_EQ(dest.buffer().shard_coords().size(), subset.size());
    auto expected = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), subset, {35});
    expect_host_tensors_eq(expected, dest);
}

}  // namespace
}  // namespace ttnn::distributed::test
