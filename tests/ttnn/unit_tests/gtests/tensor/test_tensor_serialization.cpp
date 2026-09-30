// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt_stl/cleanup.hpp>
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/distributed/utils.hpp"

#include "ttnn/config.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ttnn {
namespace {

using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_metal::distributed::test::utils::TemporaryFile;

using namespace tt::tt_metal;

tt::tt_metal::TensorSpec get_tensor_spec(const ttnn::Shape& shape, DataType dtype) {
    return tt::tt_metal::TensorSpec(shape, TensorLayout(dtype, Layout::ROW_MAJOR, MemoryConfig{}));
}

// Alignment that the data region and every shard buffer within it are expected to satisfy, so that a caller can
// use the mapped file as a pinned DMA source without copying it first. Spelled out rather than taken from the
// serializer's own `kTensorDataAlignment`, so that changing the on-disk format has to change this golden too.
constexpr uintptr_t kExpectedDataAlignment = 64;

// Reads the byte offset at which the tensor data region begins in a serialized tensor file.
uint64_t read_data_region_offset(const std::string& file_name) {
    std::ifstream file(file_name, std::ios::binary);
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    EXPECT_TRUE(file.good());
    return sizeof(header_size) + header_size;
}

// Collects the address of every shard buffer of a host tensor. `load_tensor_flatbuffer` maps the file at offset 0,
// so these addresses have the same alignment as the shards' offsets within the file.
std::vector<uintptr_t> shard_addresses(const Tensor& tensor) {
    std::vector<uintptr_t> addresses;
    tensor.host_storage().buffer().apply(
        [&](const HostBuffer& shard) { addresses.push_back(reinterpret_cast<uintptr_t>(shard.view_bytes().data())); });
    return addresses;
}

using TensorSerializationFlatbufferTest = GenericMeshDeviceFixture;

TEST_F(TensorSerializationFlatbufferTest, ReplicatedTensorRoundtrip) {
    TemporaryFile test_file("flatbuffer.tensorbin");
    std::vector<float> test_data{1.0f, 2.5f, -3.7f, 42.0f, -0.5f, 100.0f};

    Tensor original_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, 2, 3, 1}, DataType::FLOAT32));

    EXPECT_TRUE(original_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), original_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), original_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), original_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), original_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

    EXPECT_THAT(loaded_tensor.to_vector<float>(), Pointwise(FloatEq(), test_data));
}

TEST_F(TensorSerializationFlatbufferTest, ReplicatedTensorDifferentDataTypes) {
    {
        TemporaryFile test_file("uint32.tensorbin");
        std::vector<uint32_t> test_data{1, 2, 3, 4, 5, 6};
        Tensor original_tensor = Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{2, 3}, DataType::UINT32));

        dump_tensor_flatbuffer(test_file.string(), original_tensor);
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        EXPECT_EQ(loaded_tensor.dtype(), DataType::UINT32);
        EXPECT_THAT(loaded_tensor.to_vector<uint32_t>(), Pointwise(testing::Eq(), test_data));
    }

    {
        TemporaryFile test_file("int8.tensorbin");
        std::vector<int8_t> test_data{-128, -1, 0, 1, 42, 127};
        Tensor original_tensor = Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{2, 3}, DataType::INT8));

        dump_tensor_flatbuffer(test_file.string(), original_tensor);
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        EXPECT_EQ(loaded_tensor.dtype(), DataType::INT8);
        EXPECT_THAT(loaded_tensor.to_vector<int8_t>(), Pointwise(testing::Eq(), test_data));
    }

    {
        TemporaryFile test_file("bfloat16.tensorbin");
        std::vector<bfloat16> test_data{bfloat16(1.5f), bfloat16(2.5f), bfloat16(-3.5f), bfloat16(4.5f)};
        Tensor original_tensor = Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, 4}, DataType::BFLOAT16));

        dump_tensor_flatbuffer(test_file.string(), original_tensor);
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        EXPECT_EQ(loaded_tensor.dtype(), DataType::BFLOAT16);

        auto loaded_data = loaded_tensor.to_vector<bfloat16>();
        ASSERT_THAT(loaded_data, SizeIs(test_data.size()));

        for (size_t i = 0; i < test_data.size(); i++) {
            EXPECT_FLOAT_EQ(static_cast<float>(test_data[i]), static_cast<float>(loaded_data[i]));
        }
    }
}

// Range lockstep lives on the MemoryConfig, so it has to survive serialization: any path that
// rebuilds a spec from its stored form -- a warm tensor cache, for instance -- would otherwise
// silently drop it and the buffer would revert to a chip-wide allocation scan.
TEST_F(TensorSerializationFlatbufferTest, WithRangeLockstepAllocation) {
    TemporaryFile test_file("flatbuffer_range_lockstep.tensorbin");
    std::vector<float> test_data{1.0f, 2.0f, 3.0f, 4.0f};

    auto memory_config = MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(CoreRangeSet(CoreCoord(0, 0)), {32, 32}, ShardOrientation::ROW_MAJOR));
    tt::tt_metal::experimental::range_lockstep_allocation::set_range_lockstep_allocation(memory_config, true);

    Tensor original_tensor = Tensor::from_vector(
        test_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, 2, 2}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, memory_config)));

    dump_tensor_flatbuffer(test_file.string(), original_tensor);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_TRUE(tt::tt_metal::experimental::range_lockstep_allocation::is_range_lockstep_allocation(
        loaded_tensor.memory_config()))
        << "range lockstep was dropped by the flatbuffer round-trip";
    EXPECT_TRUE(loaded_tensor.memory_config() == original_tensor.memory_config());
}

// A config that never opted in must not come back opted in.
TEST_F(TensorSerializationFlatbufferTest, WithoutRangeLockstepAllocation) {
    TemporaryFile test_file("flatbuffer_no_range_lockstep.tensorbin");
    std::vector<float> test_data{1.0f, 2.0f};

    Tensor original_tensor = Tensor::from_vector(
        test_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, 1, 2},
            TensorLayout(
                DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1})));

    dump_tensor_flatbuffer(test_file.string(), original_tensor);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_FALSE(tt::tt_metal::experimental::range_lockstep_allocation::is_range_lockstep_allocation(
        loaded_tensor.memory_config()));
}

TEST_F(TensorSerializationFlatbufferTest, WithMemoryConfig) {
    TemporaryFile test_file("flatbuffer.tensorbin");
    std::vector<float> test_data{1.0f, 2.5f, -3.7f, 42.0f, -0.5f, 100.0f};

    Tensor original_tensor = Tensor::from_vector(
        test_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 2, 3, 1},
            TensorLayout(
                DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1})));

    EXPECT_TRUE(original_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), original_tensor);

    // Load as host tensor.
    {
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

        EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), original_tensor.tensor_spec().logical_shape());
        EXPECT_EQ(loaded_tensor.dtype(), original_tensor.dtype());
        EXPECT_EQ(loaded_tensor.layout(), original_tensor.layout());
        EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);
        EXPECT_TRUE(loaded_tensor.memory_config() == original_tensor.memory_config());
    }

    // Load as device tensor.
    {
        Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string(), mesh_device_.get());

        EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), original_tensor.tensor_spec().logical_shape());
        EXPECT_EQ(loaded_tensor.dtype(), original_tensor.dtype());
        EXPECT_EQ(loaded_tensor.layout(), original_tensor.layout());
        EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::DEVICE);
        EXPECT_TRUE(loaded_tensor.memory_config() == original_tensor.memory_config());
    }
}

using TensorSerializationFlatbuffer2x4Test = MeshDevice2x4Fixture;

TEST_F(TensorSerializationFlatbuffer2x4Test, Shard1DTensorRoundtrip) {
    TemporaryFile test_file("shard1d_flatbuffer.tensorbin");
    const int num_devices = mesh_device_->num_devices();
    constexpr int kNumElements = 1024;
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{1, num_devices, kNumElements, 1}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    EXPECT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

    std::vector<Tensor> original_device_tensors = ttnn::distributed::get_device_tensors(sharded_tensor);
    std::vector<Tensor> loaded_device_tensors = ttnn::distributed::get_device_tensors(loaded_tensor);

    ASSERT_THAT(loaded_device_tensors, SizeIs(original_device_tensors.size()));

    for (size_t i = 0; i < original_device_tensors.size(); i++) {
        EXPECT_THAT(
            loaded_device_tensors[i].to_vector<float>(),
            Pointwise(FloatEq(), original_device_tensors[i].to_vector<float>()));
    }
}

TEST_F(TensorSerializationFlatbuffer2x4Test, Shard2DTensorRoundtrip) {
    TemporaryFile test_file("shard2d_flatbuffer.tensorbin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 4;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}},
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    EXPECT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

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

TEST_F(TensorSerializationFlatbuffer2x4Test, Shard1DFewerShardsThanDevicesRoundtrip) {
    TemporaryFile test_file("shard1d_fewer_flatbuffer.tensorbin");
    const int num_devices = mesh_device_->num_devices();
    constexpr int kNumElements = 1024;
    std::vector<float> test_data;
    for (int i = 0; i < num_devices - 1; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{1, num_devices - 1, kNumElements, 1}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    EXPECT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

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

TEST_F(TensorSerializationFlatbuffer2x4Test, Shard2x3SubmeshRoundtrip) {
    TemporaryFile test_file("shard2x3_flatbuffer.tensorbin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 3;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}},
            .mesh_shape_override = ttnn::distributed::MeshShape(kNumRows, kNumCols),
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    EXPECT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

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

TEST_F(TensorSerializationFlatbuffer2x4Test, PartiallyReplicatedRoundtrip) {
    TemporaryFile test_file("partially_replicated_flatbuffer.tensorbin");
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 4;
    constexpr int kNumElements = 1024;
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return i * 1.0f; });
    }

    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, kNumElements}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Replicate{}},
        });

    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    EXPECT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::HOST);

    dump_tensor_flatbuffer(test_file.string(), sharded_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), sharded_tensor.tensor_spec().logical_shape());
    EXPECT_EQ(loaded_tensor.dtype(), sharded_tensor.dtype());
    EXPECT_EQ(loaded_tensor.layout(), sharded_tensor.layout());
    EXPECT_TRUE(loaded_tensor.storage_type() == ttnn::StorageType::HOST);

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

TEST_F(TensorSerializationFlatbuffer2x4Test, FullyReplicatedRoundtrip) {
    TemporaryFile test_file("fully_replicated_flatbuffer.tensorbin");
    constexpr int kHeight = 32;
    constexpr int kWidth = 40;
    const int total_elements = 1 * 1 * kHeight * kWidth;

    std::vector<float> test_data(total_elements, 0.0f);

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, 1, kHeight, kWidth}, DataType::FLOAT32));

    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper);

    MeshShape expected_shape = MeshShape(mesh_device_->num_devices());
    ttsl::SmallVector<ttnn::distributed::MeshMapperConfig::Placement> expected_placements;
    expected_placements.emplace_back(ttnn::distributed::MeshMapperConfig::Replicate{});

    EXPECT_EQ(replicated_tensor.tensor_topology().distribution_shape(), expected_shape);
    EXPECT_EQ(replicated_tensor.tensor_topology().placements(), expected_placements);

    dump_tensor_flatbuffer(test_file.string(), replicated_tensor);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_topology().distribution_shape(), expected_shape);
    EXPECT_EQ(loaded_tensor.tensor_topology().placements(), expected_placements);
}

// Shards are 12 bytes each, so they only land on `kExpectedDataAlignment` if the serializer pads between them.
TEST(TensorSerializationFlatbufferAlignmentTest, ShardsAreAlignedForPinnedMemory) {
    TemporaryFile test_file("alignment.tensorbin");
    constexpr size_t kNumShards = 3;

    std::vector<Tensor> shards;
    shards.reserve(kNumShards);
    for (size_t i = 0; i < kNumShards; i++) {
        shards.push_back(Tensor::from_vector(
            std::vector<float>{1.0f * i, 2.0f * i, 3.0f * i}, get_tensor_spec(ttnn::Shape{1, 3}, DataType::FLOAT32)));
    }
    Tensor original_tensor = ttnn::distributed::from_host_shards(shards, MeshShape(1, kNumShards));

    dump_tensor_flatbuffer(test_file.string(), original_tensor, DumpTensorMode::LOCAL);

    EXPECT_EQ(read_data_region_offset(test_file.string()) % kExpectedDataAlignment, 0);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    const std::vector<uintptr_t> addresses = shard_addresses(loaded_tensor);
    EXPECT_THAT(addresses, SizeIs(kNumShards));
    for (uintptr_t address : addresses) {
        EXPECT_EQ(address % kExpectedDataAlignment, 0);
    }

    std::vector<Tensor> loaded_shards = ttnn::distributed::get_device_tensors(loaded_tensor);
    ASSERT_THAT(loaded_shards, SizeIs(kNumShards));
    for (size_t i = 0; i < kNumShards; i++) {
        EXPECT_THAT(loaded_shards[i].to_vector<float>(), Pointwise(FloatEq(), shards[i].to_vector<float>()));
    }
}

// Write-side guards: `to_flatbuffer` writes shards that the topology labels as replicas once, so it has to check
// the label against the data first. The tests below build host tensors straight from a `DistributedHostBuffer`
// instead of going through `from_host_shards` or a mesh mapper, which both open the cluster to get a distributed
// context, so they need no device.

using Placement = ttnn::distributed::MeshMapperConfig::Placement;
using Placements = ttsl::SmallVector<Placement>;
constexpr auto kReplicate = ttnn::distributed::MeshMapperConfig::Replicate{};

// Builds a float32 host tensor with `shard_data[i]` placed at `shard_coords[i]` in a `buffer_shape` host buffer,
// labelled with `topology`. Each shard gets its own HostBuffer, so equal values never alias.
Tensor make_host_tensor(
    const MeshShape& buffer_shape,
    const std::vector<MeshCoordinate>& shard_coords,
    const std::vector<std::vector<float>>& shard_data,
    const TensorTopology& topology) {
    ASSERT_EQ(shard_coords.size(), shard_data.size()) << "test bug: one data vector per coordinate";
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    for (size_t i = 0; i < shard_coords.size(); ++i) {
        buffer.emplace_shard(
            shard_coords[i], [&data = shard_data[i]]() { return HostBuffer(std::vector<float>(data)); });
    }
    const auto shard_shape = ttnn::Shape{1, static_cast<uint32_t>(shard_data.front().size())};
    return Tensor(host_tensor_from_buffer_with_topology(
        std::move(buffer), get_tensor_spec(shard_shape, DataType::FLOAT32), topology));
}

std::vector<float> shard_values(const Tensor& tensor, const MeshCoordinate& coord) {
    const auto shard = tensor.host_storage().buffer().get_shard(coord);
    if (!shard.has_value()) {
        ADD_FAILURE() << "no shard at " << coord;
        return {};
    }
    const auto values = shard->view_as<float>();
    return std::vector<float>(values.begin(), values.end());
}

// Runs `dump_tensor_flatbuffer` in LOCAL mode expecting it to be rejected, and checks that the rejection names
// `diagnostic` and that no file was created.
void expect_dump_rejected(const TemporaryFile& file, const Tensor& tensor, const std::string& diagnostic) {
    try {
        dump_tensor_flatbuffer(file.string(), tensor, DumpTensorMode::LOCAL);
        FAIL() << "dump was accepted; expected a rejection mentioning: " << diagnostic;
    } catch (const std::runtime_error& e) {
        EXPECT_THAT(e.what(), HasSubstr(diagnostic));
    }
    EXPECT_FALSE(std::filesystem::exists(file.path())) << "a rejected dump must not leave a file behind";
}

// Two distinct shards under the collapsed 1-D Replicate label the default mappers produce. Before the byte compare
// this dumped successfully and wrote the (0,0) shard for both coordinates, so loading the file lost (0,1).
TEST(TensorSerializationFlatbufferGuardTest, MislabelledReplicateRejected) {
    TemporaryFile test_file("mislabelled_replicate.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    Tensor tensor = make_host_tensor(
        MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, TensorTopology(MeshShape(2), Placements{kReplicate}, coords));

    expect_dump_rejected(test_file, tensor, "contents differ");
}

// A 2x2 tensor sharded along both axes and relabelled as if its columns were replicas. The dedup groups are the
// rows, so (0,0) and (0,1) are compared and differ.
TEST(TensorSerializationFlatbufferGuardTest, MislabelledPartialReplicateRejected) {
    TemporaryFile test_file("mislabelled_partial_replicate.tensorbin");
    const std::vector<MeshCoordinate> coords{
        MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(1, 0), MeshCoordinate(1, 1)};
    const std::vector<std::vector<float>> data{{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
    const Placements true_label{ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}};
    const Placements wrong_label{ttnn::distributed::MeshMapperConfig::Shard{1}, kReplicate};

    // Control: with the label that describes the data the dump is accepted.
    Tensor tensor = make_host_tensor(MeshShape(2, 2), coords, data, TensorTopology(MeshShape(2, 2), true_label, coords));
    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    EXPECT_TRUE(std::filesystem::exists(test_file.path()));
    std::filesystem::remove(test_file.path());

    tensor.update_tensor_topology(TensorTopology(MeshShape(2, 2), wrong_label, coords));
    expect_dump_rejected(test_file, tensor, "contents differ");
}

// Replicas that really are identical still share one copy on disk, and the label survives the round-trip.
TEST(TensorSerializationFlatbufferGuardTest, IdenticalReplicasStillDeduplicated) {
    TemporaryFile test_file("identical_replicas.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const TensorTopology topology(MeshShape(2), Placements{kReplicate}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {values, values}, topology);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t one_shard_bytes = values.size() * sizeof(float);
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), read_data_region_offset(test_file.string()) + one_shard_bytes);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), topology);
    for (const auto& coord : coords) {
        EXPECT_THAT(shard_values(loaded_tensor, coord), Pointwise(FloatEq(), values));
    }
}

// With the byte compare turned off, the mislabelled tensor is written the way it was before the check existed:
// one copy, taken from the first replica, for the whole group.
TEST(TensorSerializationFlatbufferGuardTest, ReplicaCheckOptOutWritesFirstReplica) {
    const bool previous = ttnn::CONFIG.get<"verify_replicated_shards_on_dump">();
    ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(false);
    auto restore = ttsl::make_cleanup([previous]() { ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(previous); });

    TemporaryFile test_file("replica_check_opt_out.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    const std::vector<float> first{1.0f, 2.0f, 3.0f};
    Tensor tensor = make_host_tensor(
        MeshShape(1, 2), coords, {first, {4.0f, 5.0f, 6.0f}}, TensorTopology(MeshShape(2), Placements{kReplicate}, coords));

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    EXPECT_EQ(std::filesystem::file_size(test_file.path()), read_data_region_offset(test_file.string()) + first.size() * sizeof(float));
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    for (const auto& coord : coords) {
        EXPECT_THAT(shard_values(loaded_tensor, coord), Pointwise(FloatEq(), first));
    }
}

// The size check is not gated by the opt-out: replicas of different sizes cannot share a record.
TEST(TensorSerializationFlatbufferGuardTest, ReplicaSizeMismatchRejectedEvenWhenOptedOut) {
    const bool previous = ttnn::CONFIG.get<"verify_replicated_shards_on_dump">();
    ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(false);
    auto restore = ttsl::make_cleanup([previous]() { ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(previous); });

    TemporaryFile test_file("replica_size_mismatch.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    Tensor tensor = make_host_tensor(
        MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f}}, TensorTopology(MeshShape(2), Placements{kReplicate}, coords));

    expect_dump_rejected(test_file, tensor, "sizes differ");
}

// The label lists (0,1) but only (0,0) holds a shard, as for a single shard taken out of a distributed tensor.
// Before the guard this dumped one shard under a two-coordinate label.
TEST(TensorSerializationFlatbufferGuardTest, LabelCoordWithoutShardRejected) {
    TemporaryFile test_file("label_coord_without_shard.tensorbin");
    const std::vector<MeshCoordinate> label_coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    Tensor tensor = make_host_tensor(
        MeshShape(1, 2),
        {MeshCoordinate(0, 0)},
        {{1.0f, 2.0f, 3.0f}},
        TensorTopology(MeshShape(2), Placements{ttnn::distributed::MeshMapperConfig::Shard{1}}, label_coords));

    expect_dump_rejected(test_file, tensor, "has no shard there");
}

// Both coordinates hold a shard but the label only covers (0,0). Before the guard the (0,1) shard was silently
// left out of the file.
TEST(TensorSerializationFlatbufferGuardTest, ShardOutsideLabelRejected) {
    TemporaryFile test_file("shard_outside_label.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1)};
    Tensor tensor = make_host_tensor(
        MeshShape(1, 2),
        coords,
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
        TensorTopology(MeshShape(1), Placements{kReplicate}, {MeshCoordinate(0, 0)}));

    expect_dump_rejected(test_file, tensor, "does not cover");
}

TEST(TensorSerializationFlatbufferGuardTest, ShardedRoundtripPreservesTopology1D) {
    TemporaryFile test_file("sharded_1d_topology.tensorbin");
    const std::vector<MeshCoordinate> coords{MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(0, 2)};
    const std::vector<std::vector<float>> data{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    const TensorTopology topology(MeshShape(3), Placements{ttnn::distributed::MeshMapperConfig::Shard{1}}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 3), coords, data, topology);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_topology(), topology);
    for (size_t i = 0; i < coords.size(); ++i) {
        EXPECT_THAT(shard_values(loaded_tensor, coords[i]), Pointwise(FloatEq(), data[i]));
    }
}

TEST(TensorSerializationFlatbufferGuardTest, ShardedRoundtripPreservesTopology2D) {
    TemporaryFile test_file("sharded_2d_topology.tensorbin");
    const std::vector<MeshCoordinate> coords{
        MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(1, 0), MeshCoordinate(1, 1)};
    const std::vector<std::vector<float>> data{{0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}};
    const TensorTopology topology(
        MeshShape(2, 2),
        Placements{ttnn::distributed::MeshMapperConfig::Shard{1}, ttnn::distributed::MeshMapperConfig::Shard{2}},
        coords);
    Tensor tensor = make_host_tensor(MeshShape(2, 2), coords, data, topology);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_topology(), topology);
    for (size_t i = 0; i < coords.size(); ++i) {
        EXPECT_THAT(shard_values(loaded_tensor, coords[i]), Pointwise(FloatEq(), data[i]));
    }
}

}  // namespace
}  // namespace ttnn
