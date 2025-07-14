// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/distributed/utils.hpp"

#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include <filesystem>
#include <vector>

namespace ttnn {

namespace CMAKE_UNIQUE_NAMESPACE {

TensorSpec get_tensor_spec(const ttnn::Shape& shape, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, Layout::ROW_MAJOR, MemoryConfig{}));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_metal::distributed::test::utils::TemporaryFile;

using namespace tt::tt_metal;

using TensorSerializationFlatbufferTest = GenericMeshDeviceFixture;

TEST_F(TensorSerializationFlatbufferTest, ReplicatedTensorRoundtrip) {
    TemporaryFile test_file("flatbuffer.bin");
    std::vector<float> test_data{1.0f, 2.5f, -3.7f, 42.0f, -0.5f, 100.0f};

    Tensor original_tensor = Tensor::from_vector(
        test_data, CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, 2, 3, 1}, DataType::FLOAT32));

    ASSERT_TRUE(original_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), original_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), original_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), original_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), original_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    EXPECT_THAT(loaded_tensor.to_vector<float>(), Pointwise(FloatEq(), test_data));
}

TEST_F(TensorSerializationFlatbufferTest, ReplicatedTensorDifferentDataTypes) {
    {
        TemporaryFile test_file("uint32.bin");
        std::vector<uint32_t> test_data{1, 2, 3, 4, 5, 6};
        Tensor original_tensor = Tensor::from_vector(
            test_data, CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{2, 3}, DataType::UINT32));

        dump_tensor_flatbuffer(test_file.string(), original_tensor);
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        ASSERT_EQ(loaded_tensor.dtype(), DataType::UINT32);
        EXPECT_THAT(loaded_tensor.to_vector<uint32_t>(), Pointwise(testing::Eq(), test_data));
    }

    {
        TemporaryFile test_file("bfloat16.bin");
        std::vector<bfloat16> test_data{bfloat16(1.5f), bfloat16(2.5f), bfloat16(-3.5f), bfloat16(4.5f)};
        Tensor original_tensor = Tensor::from_vector(
            test_data, CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, 4}, DataType::BFLOAT16));

        dump_tensor_flatbuffer(test_file.string(), original_tensor);
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        ASSERT_EQ(loaded_tensor.dtype(), DataType::BFLOAT16);
        auto loaded_data = loaded_tensor.to_vector<bfloat16>();
        ASSERT_THAT(loaded_data, SizeIs(test_data.size()));
        for (size_t i = 0; i < test_data.size(); i++) {
            EXPECT_FLOAT_EQ(test_data[i].to_float(), loaded_data[i].to_float());
        }
    }
}

using TensorSerializationFlatbufferT3000Test = T3000MeshDeviceFixture;

TEST_F(TensorSerializationFlatbufferT3000Test, Shard1DTensorRoundtrip) {
    TemporaryFile test_file("shard1d_flatbuffer.bin");
    const int num_devices = mesh_device_->num_devices();
    constexpr int kNumElements = 1024;
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data,
        CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, num_devices, kNumElements, 1}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    ASSERT_TRUE(sharded_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
    }
}

TEST_F(TensorSerializationFlatbufferT3000Test, Shard2DTensorRoundtrip) {
    TemporaryFile test_file("shard2d_flatbuffer.bin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 4;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data,
        CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}},
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    ASSERT_TRUE(sharded_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
        EXPECT_EQ(loaded_device_tensors[i].logical_shape(), original_device_tensors[i].logical_shape());
    }
}

TEST_F(TensorSerializationFlatbufferT3000Test, Shard1DFewerShardsThanDevicesRoundtrip) {
    TemporaryFile test_file("shard1d_fewer_flatbuffer.bin");
    const int num_devices = mesh_device_->num_devices();
    constexpr int kNumElements = 1024;
    std::vector<float> test_data;
    for (int i = 0; i < num_devices - 1; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data,
        CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, num_devices - 1, kNumElements, 1}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    ASSERT_TRUE(sharded_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));
    ASSERT_THAT(loaded_device_tensors, SizeIs(num_devices - 1));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
    }
}

TEST_F(TensorSerializationFlatbufferT3000Test, Shard2x3SubmeshRoundtrip) {
    TemporaryFile test_file("shard2x3_flatbuffer.bin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 3;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data,
        CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}},
            .mesh_shape_override = ttnn::distributed::MeshShape(kNumRows, kNumCols),
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    ASSERT_TRUE(sharded_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
        EXPECT_EQ(loaded_device_tensors[i].logical_shape(), original_device_tensors[i].logical_shape());
    }
}

TEST_F(TensorSerializationFlatbufferT3000Test, PartiallyReplicatedRoundtrip) {
    TemporaryFile test_file("partially_replicated_flatbuffer.bin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 4;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data,
        CMAKE_UNIQUE_NAMESPACE::get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Replicate{}},
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    ASSERT_TRUE(sharded_tensor.storage_type() == StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    ASSERT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    ASSERT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    ASSERT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    ASSERT_TRUE(loaded_tensor.storage_type() == StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
        EXPECT_EQ(loaded_device_tensors[i].logical_shape(), original_device_tensors[i].logical_shape());
    }
}

}  // namespace
}  // namespace ttnn
