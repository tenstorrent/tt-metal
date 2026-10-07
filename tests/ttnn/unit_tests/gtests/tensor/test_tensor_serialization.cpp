// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt_stl/assert.hpp>
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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ttnn {
namespace {

using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::ThrowsMessage;
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
// instead of going through `from_host_shards`, which opens the cluster to get a distributed context, so they need
// no device. Every "Rejected" test is a negative control: the same dump succeeded before the guard it exercises.

using Placements = ttsl::SmallVector<ttnn::distributed::MeshMapperConfig::Placement>;
constexpr ttnn::distributed::MeshMapperConfig::Replicate kReplicate{};
ttnn::distributed::MeshMapperConfig::Shard shard_on(int dim) { return {.dim = dim}; }

std::vector<MeshCoordinate> all_coords(const MeshShape& shape) {
    const MeshCoordinateRange range(shape);
    return std::vector<MeshCoordinate>(range.begin(), range.end());
}

// Wraps an already populated `buffer` of float32 shards holding `elements_per_shard` values each into a tensor
// labelled with `topology`.
Tensor wrap_host_buffer(DistributedHostBuffer buffer, size_t elements_per_shard, const TensorTopology& topology) {
    auto spec = get_tensor_spec(ttnn::Shape{1, static_cast<uint32_t>(elements_per_shard)}, DataType::FLOAT32);
    return Tensor(host_tensor_from_buffer_with_topology(std::move(buffer), std::move(spec), topology));
}

// Builds a float32 host tensor with `shard_data[i]` placed at `shard_coords[i]` in a `buffer_shape` host buffer,
// labelled with `topology`. Each shard gets its own HostBuffer, so equal values never alias.
Tensor make_host_tensor(
    const MeshShape& buffer_shape,
    const std::vector<MeshCoordinate>& shard_coords,
    const std::vector<std::vector<float>>& shard_data,
    const TensorTopology& topology) {
    TT_FATAL(shard_coords.size() == shard_data.size(), "test bug: one data vector per coordinate");
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    for (size_t i = 0; i < shard_coords.size(); ++i) {
        buffer.emplace_shard(
            shard_coords[i], [&data = shard_data[i]]() { return HostBuffer(std::vector<float>(data)); });
    }
    return wrap_host_buffer(std::move(buffer), shard_data.front().size(), topology);
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
    EXPECT_THAT(
        [&]() { dump_tensor_flatbuffer(file.string(), tensor, DumpTensorMode::LOCAL); },
        ThrowsMessage<std::runtime_error>(HasSubstr(diagnostic)));
    EXPECT_FALSE(std::filesystem::exists(file.path())) << "a rejected dump must not leave a file behind";
}

// Sets the replica byte compare for the rest of the calling test and restores the previous setting after it.
[[nodiscard]] auto scoped_replica_check(bool enabled) {
    const bool previous = ttnn::CONFIG.get<"verify_replicated_shards_on_dump">();
    ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(enabled);
    return ttsl::make_cleanup([previous]() { ttnn::CONFIG.set<"verify_replicated_shards_on_dump">(previous); });
}

// Two distinct shards under the collapsed 1-D Replicate label the default mappers produce. Before the byte compare
// this dump succeeded and wrote the (0,0) shard for both coordinates, so loading the file lost (0,1).
TEST(TensorSerializationFlatbufferGuardTest, MislabelledReplicateRejected) {
    TemporaryFile test_file("mislabelled_replicate.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, label);

    expect_dump_rejected(test_file, tensor, "contents differ");
}

// A 2x2 tensor sharded along both mesh axes and relabelled as if its columns were replicas. The dedup groups are
// the rows, so (0,0) and (0,1) are compared and differ. The correctly labelled tensor dumps first, as a control.
TEST(TensorSerializationFlatbufferGuardTest, MislabelledPartialReplicateRejected) {
    TemporaryFile test_file("mislabelled_partial_replicate.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(2, 2));
    const std::vector<std::vector<float>> data{{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
    const TensorTopology true_label(MeshShape(2, 2), Placements{shard_on(0), shard_on(1)}, coords);
    const TensorTopology wrong_label(MeshShape(2, 2), Placements{shard_on(0), kReplicate}, coords);

    Tensor tensor = make_host_tensor(MeshShape(2, 2), coords, data, true_label);
    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    EXPECT_TRUE(std::filesystem::exists(test_file.path()));
    std::filesystem::remove(test_file.path());

    tensor.update_tensor_topology(wrong_label);
    expect_dump_rejected(test_file, tensor, "contents differ");
}

// Replicas that really are identical still share one copy on disk, and the label survives the round-trip.
TEST(TensorSerializationFlatbufferGuardTest, IdenticalReplicasStillDeduplicated) {
    TemporaryFile test_file("identical_replicas.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {values, values}, label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t one_shard_bytes = values.size() * sizeof(float);
    const uint64_t data_offset = read_data_region_offset(test_file.string());
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), data_offset + one_shard_bytes);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    for (const auto& coord : coords) {
        EXPECT_THAT(shard_values(loaded_tensor, coord), Pointwise(FloatEq(), values));
    }
}

// With the byte compare turned off, the mislabelled tensor is written the way it was before the check existed:
// one copy, taken from the first replica, for the whole group.
TEST(TensorSerializationFlatbufferGuardTest, ReplicaCheckOptOutWritesFirstReplica) {
    auto restore_config = scoped_replica_check(false);
    TemporaryFile test_file("replica_check_opt_out.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const std::vector<float> first{1.0f, 2.0f, 3.0f};
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {first, {4.0f, 5.0f, 6.0f}}, label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t data_offset = read_data_region_offset(test_file.string());
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), data_offset + first.size() * sizeof(float));
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    for (const auto& coord : coords) {
        EXPECT_THAT(shard_values(loaded_tensor, coord), Pointwise(FloatEq(), first));
    }
}

// The size check is not gated by the opt-out: replicas of different sizes cannot share a record.
TEST(TensorSerializationFlatbufferGuardTest, ReplicaSizeMismatchRejectedEvenWhenOptedOut) {
    auto restore_config = scoped_replica_check(false);
    TemporaryFile test_file("replica_size_mismatch.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f}}, label);

    expect_dump_rejected(test_file, tensor, "sizes differ");
}

// The label lists (0,1) but only (0,0) holds a shard, as for a single shard taken out of a distributed tensor.
// Before the guard this dumped one shard under a two-coordinate label.
TEST(TensorSerializationFlatbufferGuardTest, LabelCoordWithoutShardRejected) {
    TemporaryFile test_file("label_coord_without_shard.tensorbin");
    const TensorTopology label(MeshShape(2), Placements{shard_on(1)}, all_coords(MeshShape(1, 2)));
    Tensor tensor = make_host_tensor(MeshShape(1, 2), {MeshCoordinate(0, 0)}, {{1.0f, 2.0f, 3.0f}}, label);

    expect_dump_rejected(test_file, tensor, "has no shard there");
}

// Both coordinates hold a shard but the label only covers (0,0). Before the guard the (0,1) shard was silently
// left out of the file.
TEST(TensorSerializationFlatbufferGuardTest, ShardOutsideLabelRejected) {
    TemporaryFile test_file("shard_outside_label.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology label(MeshShape(1), Placements{kReplicate}, {MeshCoordinate(0, 0)});
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, label);

    expect_dump_rejected(test_file, tensor, "does not cover");
}

// One HostBuffer emplaced at both coordinates shares one copy on disk whatever the label says: the pointer lookup
// runs before the label-driven group lookup, so under a Shard label (two different replica groups, nothing to
// compare) the file still holds a single copy. This is the fully replicated mapper path, which aliases one buffer.
TEST(TensorSerializationFlatbufferGuardTest, AliasedShardsDeduplicatedByPointer) {
    TemporaryFile test_file("aliased_shards.tensorbin");
    const MeshShape buffer_shape(1, 2);
    const std::vector<MeshCoordinate> coords = all_coords(buffer_shape);
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const TensorTopology label(MeshShape(2), Placements{shard_on(1)}, coords);

    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    HostBuffer shared_buffer{std::vector<float>(values)};
    for (const auto& coord : coords) {
        buffer.emplace_shard(coord, [&shared_buffer]() { return shared_buffer; });
    }
    Tensor tensor = wrap_host_buffer(std::move(buffer), values.size(), label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t data_offset = read_data_region_offset(test_file.string());
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), data_offset + values.size() * sizeof(float));
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    for (const auto& coord : coords) {
        EXPECT_THAT(shard_values(loaded_tensor, coord), Pointwise(FloatEq(), values));
    }
}

// Two borrowed-span HostBuffers over one allocation of three floats: they start at the same address, so the
// address lookup in `to_flatbuffer` sees them, but one views two floats and the other all three. The pins keep the
// allocation alive.
struct SameAddressViews {
    HostBuffer short_view;  // The first two floats.
    HostBuffer long_view;   // All three floats.
};
SameAddressViews make_same_address_views() {
    auto storage = std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f});
    return {
        .short_view = HostBuffer(ttsl::Span<float>(storage->data(), 2), MemoryPin(storage)),
        .long_view = HostBuffer(ttsl::Span<float>(storage->data(), 3), MemoryPin(storage))};
}

// Builds a 1x2 host tensor with `views.short_view` at (0,0) and `views.long_view` at (0,1) under `label`.
Tensor make_same_address_tensor(SameAddressViews& views, const TensorTopology& label) {
    const MeshShape buffer_shape(1, 2);
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    buffer.emplace_shard(MeshCoordinate(0, 0), [&views]() { return views.short_view; });
    buffer.emplace_shard(MeshCoordinate(0, 1), [&views]() { return views.long_view; });
    return wrap_host_buffer(std::move(buffer), /*elements_per_shard=*/3, label);
}

// The address shortcut has to compare lengths too. Under a Replicate label the two views land in one group; without
// the length check the second one reused the first one's record and the file claimed 12 bytes where 8 were written.
TEST(TensorSerializationFlatbufferGuardTest, SameAddressDifferentLengthReplicasRejected) {
    TemporaryFile test_file("same_address_replicas.tensorbin");
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, all_coords(MeshShape(1, 2)));
    SameAddressViews views = make_same_address_views();
    Tensor tensor = make_same_address_tensor(views, label);

    expect_dump_rejected(test_file, tensor, "sizes differ");
}

// Under a Shard label the two views are different groups and not an alias, so each gets its own copy: the second
// buffer starts on the next 64-byte boundary, and each coordinate loads with its own length.
TEST(TensorSerializationFlatbufferGuardTest, SameAddressDifferentLengthShardsWrittenSeparately) {
    TemporaryFile test_file("same_address_shards.tensorbin");
    const TensorTopology label(MeshShape(2), Placements{shard_on(1)}, all_coords(MeshShape(1, 2)));
    SameAddressViews views = make_same_address_views();
    Tensor tensor = make_same_address_tensor(views, label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t data_offset = read_data_region_offset(test_file.string());
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), data_offset + kExpectedDataAlignment + 3 * sizeof(float));
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    EXPECT_THAT(
        shard_values(loaded_tensor, MeshCoordinate(0, 0)), Pointwise(FloatEq(), std::vector<float>{1.0f, 2.0f}));
    EXPECT_THAT(
        shard_values(loaded_tensor, MeshCoordinate(0, 1)), Pointwise(FloatEq(), std::vector<float>{1.0f, 2.0f, 3.0f}));
}

// A labelled coordinate that is remote to this host (a multi-host LOCAL dump) legitimately has no shard here. The
// buffer's local window is the 1x1 at (0,0) of a 1x2 global shape, so (0,1) is remote: the dump writes the one local
// shard under the two-coordinate label instead of rejecting the missing one.
TEST(TensorSerializationFlatbufferGuardTest, RemoteLabelCoordWithoutShardAccepted) {
    TemporaryFile test_file("remote_coord_without_shard.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const TensorTopology label(MeshShape(2), Placements{shard_on(1)}, coords);

    auto buffer =
        DistributedHostBuffer::create(MeshShape(1, 2), MeshShape(1, 1), MeshCoordinate(0, 0), /*context=*/nullptr);
    ASSERT_TRUE(buffer.is_local(MeshCoordinate(0, 0)));
    ASSERT_FALSE(buffer.is_local(MeshCoordinate(0, 1)));
    buffer.emplace_shard(MeshCoordinate(0, 0), [&values]() { return HostBuffer(std::vector<float>(values)); });
    Tensor tensor = wrap_host_buffer(std::move(buffer), values.size(), label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    const uint64_t data_offset = read_data_region_offset(test_file.string());
    EXPECT_EQ(std::filesystem::file_size(test_file.path()), data_offset + values.size() * sizeof(float));
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    EXPECT_THAT(shard_values(loaded_tensor, MeshCoordinate(0, 0)), Pointwise(FloatEq(), values));
    EXPECT_FALSE(loaded_tensor.host_storage().buffer().get_shard(MeshCoordinate(0, 1)).has_value());
}

TEST(TensorSerializationFlatbufferGuardTest, ShardedRoundtripPreservesTopology1D) {
    TemporaryFile test_file("sharded_1d_topology.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 3));
    const std::vector<std::vector<float>> data{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    const TensorTopology label(MeshShape(3), Placements{shard_on(1)}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 3), coords, data, label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    for (size_t i = 0; i < coords.size(); ++i) {
        EXPECT_THAT(shard_values(loaded_tensor, coords[i]), Pointwise(FloatEq(), data[i]));
    }
}

TEST(TensorSerializationFlatbufferGuardTest, ShardedRoundtripPreservesTopology2D) {
    TemporaryFile test_file("sharded_2d_topology.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(2, 2));
    const std::vector<std::vector<float>> data{{0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}};
    const TensorTopology label(MeshShape(2, 2), Placements{shard_on(0), shard_on(1)}, coords);
    Tensor tensor = make_host_tensor(MeshShape(2, 2), coords, data, label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(loaded_tensor.tensor_topology(), label);
    for (size_t i = 0; i < coords.size(); ++i) {
        EXPECT_THAT(shard_values(loaded_tensor, coords[i]), Pointwise(FloatEq(), data[i]));
    }
}

// Schema identifier, version and read-side validation. The files are patched in place through the root table's
// vtable, which is how a reader resolves a field, so a patch changes exactly what the reader sees. Every "Rejected"
// test is a negative control: the same file loaded without complaint before the check it exercises.

constexpr size_t kFlatbufferStart = sizeof(uint64_t);  // The uint64 header size precedes the flatbuffer.
// Field indices in tensor.fbs (root `Tensor` table) and tensor_topology.fbs (`TensorTopology` table).
constexpr size_t kTensorTopologyField = 3;
constexpr size_t kSchemaVersionField = 4;
constexpr size_t kTopologyPlacementsField = 1;
constexpr size_t kTopologyMeshCoordsField = 2;
// Spelled out rather than taken from the serializer's `kTensorFileSchemaVersion`, so that bumping the on-disk
// version has to change this golden too.
constexpr uint32_t kExpectedSchemaVersion = 1;

std::vector<std::byte> read_file_bytes(const std::filesystem::path& path) {
    std::vector<std::byte> data(std::filesystem::file_size(path));
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    EXPECT_TRUE(file.good());
    return data;
}

void write_file_bytes(const std::filesystem::path& path, const std::vector<std::byte>& data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    EXPECT_TRUE(file.good());
}

template <typename T>
T read_le(const std::vector<std::byte>& data, size_t pos) {
    T value{};
    std::memcpy(&value, data.data() + pos, sizeof(T));
    return value;
}

template <typename T>
void write_le(std::vector<std::byte>& data, size_t pos, T value) {
    std::memcpy(data.data() + pos, &value, sizeof(T));
}

// Position that the uoffset stored at `pos` points to.
size_t follow(const std::vector<std::byte>& data, size_t pos) { return pos + read_le<uint32_t>(data, pos); }

size_t root_table(const std::vector<std::byte>& data) { return follow(data, kFlatbufferStart); }

// Position of the vtable slot holding the offset of field `index` of the table at `table`. Writing 0 there makes
// the field absent, so a reader gets its default (nullptr or 0).
size_t vtable_slot(const std::vector<std::byte>& data, size_t table, size_t index) {
    const size_t vtable = table - static_cast<size_t>(read_le<int32_t>(data, table));
    return vtable + 2 * sizeof(uint16_t) + index * sizeof(uint16_t);
}

// Position of field `index` of the table at `table`, which must be present.
size_t field_position(const std::vector<std::byte>& data, size_t table, size_t index) {
    const uint16_t offset = read_le<uint16_t>(data, vtable_slot(data, table, index));
    EXPECT_NE(offset, 0) << "field " << index << " is absent";
    return table + offset;
}

// Position of element `i` of the vector of tables whose uoffset is stored at `vector_field`.
size_t table_element(const std::vector<std::byte>& data, size_t vector_field, size_t i) {
    const size_t vector = follow(data, vector_field);
    return follow(data, vector + sizeof(uint32_t) + i * sizeof(uint32_t));
}

// Per-coordinate values of the file the schema tests patch: distinct shards at (0,0) and (0,1).
const std::vector<std::vector<float>>& two_shard_data() {
    static const std::vector<std::vector<float>> data{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    return data;
}

// Dumps a 1x2 tensor holding `two_shard_data()` under a `{1,2},[Replicate,Shard{1}]` label and returns its bytes.
std::vector<std::byte> dump_two_shard_file(const TemporaryFile& file) {
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology label(MeshShape(1, 2), Placements{kReplicate, shard_on(1)}, coords);
    dump_tensor_flatbuffer(
        file.string(), make_host_tensor(MeshShape(1, 2), coords, two_shard_data(), label), DumpTensorMode::LOCAL);
    return read_file_bytes(file.path());
}

void expect_load_rejected(const TemporaryFile& file, const std::string& diagnostic) {
    EXPECT_THAT(
        [&]() { load_tensor_flatbuffer(file.string()); }, ThrowsMessage<std::runtime_error>(HasSubstr(diagnostic)));
}

// Before this change the flatbuffer had no identifier (bytes [12, 16) of the file were table data) and no version
// field.
TEST(TensorSerializationFlatbufferSchemaTest, FileCarriesIdentifierAndSchemaVersion) {
    TemporaryFile test_file("schema_identifier.tensorbin");
    const auto data = dump_two_shard_file(test_file);

    ASSERT_GE(data.size(), kFlatbufferStart + 2 * sizeof(uint32_t));
    const std::string identifier(
        reinterpret_cast<const char*>(data.data()) + kFlatbufferStart + sizeof(uint32_t), sizeof(uint32_t));
    EXPECT_EQ(identifier, "TTNB");
    EXPECT_EQ(
        read_le<uint32_t>(data, field_position(data, root_table(data), kSchemaVersionField)), kExpectedSchemaVersion);
}

// A file written by a newer tt-metal is refused rather than misread; before this change there was no version to
// refuse on. Restoring the version loads the file again, as a control.
TEST(TensorSerializationFlatbufferSchemaTest, NewerSchemaVersionRejected) {
    TemporaryFile test_file("newer_schema_version.tensorbin");
    auto data = dump_two_shard_file(test_file);
    const size_t version_pos = field_position(data, root_table(data), kSchemaVersionField);

    write_le<uint32_t>(data, version_pos, kExpectedSchemaVersion + 1);
    write_file_bytes(test_file.path(), data);
    expect_load_rejected(test_file, "schema version 2");

    write_le<uint32_t>(data, version_pos, kExpectedSchemaVersion);
    write_file_bytes(test_file.path(), data);
    EXPECT_NO_THROW(load_tensor_flatbuffer(test_file.string()));
}

// A versioned writer always records the topology, so a versioned file without one is corrupt. Before this change
// the same file loaded silently as fully replicated.
TEST(TensorSerializationFlatbufferSchemaTest, VersionedFileWithoutTopologyRejected) {
    TemporaryFile test_file("versioned_without_topology.tensorbin");
    auto data = dump_two_shard_file(test_file);
    write_le<uint16_t>(data, vtable_slot(data, root_table(data), kTensorTopologyField), 0);
    write_file_bytes(test_file.path(), data);

    expect_load_rejected(test_file, "no tensor topology");
}

// A file from before the topology field existed (neither topology nor version) keeps loading as fully replicated
// with both shards intact, and the loader logs that the label is unknown. This pins the pre-change behaviour (the
// warning is the only addition); the refusal the plan first proposed would have thrown here.
TEST(TensorSerializationFlatbufferSchemaTest, LegacyFileWithoutTopologyLoadsAsReplicated) {
    TemporaryFile test_file("legacy_without_topology.tensorbin");
    auto data = dump_two_shard_file(test_file);
    const size_t root = root_table(data);
    write_le<uint16_t>(data, vtable_slot(data, root, kTensorTopologyField), 0);
    write_le<uint16_t>(data, vtable_slot(data, root, kSchemaVersionField), 0);
    write_file_bytes(test_file.path(), data);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());

    EXPECT_EQ(
        loaded_tensor.tensor_topology(), TensorTopology::create_fully_replicated_tensor_topology(MeshShape(1, 2)));
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    for (size_t i = 0; i < coords.size(); ++i) {
        EXPECT_THAT(shard_values(loaded_tensor, coords[i]), Pointwise(FloatEq(), two_shard_data()[i]));
    }
}

// The label's second coordinate is changed to (0,0), so the record at (0,1) is no longer covered. Before this change
// the file loaded with a label through which that shard was unreachable.
TEST(TensorSerializationFlatbufferSchemaTest, RecordOutsideLabelRejected) {
    TemporaryFile test_file("record_outside_label.tensorbin");
    auto data = dump_two_shard_file(test_file);
    const size_t topology = follow(data, field_position(data, root_table(data), kTensorTopologyField));
    const size_t second_coord = table_element(data, field_position(data, topology, kTopologyMeshCoordsField), 1);
    const size_t values = follow(data, field_position(data, second_coord, 0));
    ASSERT_EQ(read_le<uint32_t>(data, values), 2u) << "a 1x2 coordinate has two values";
    const size_t column = values + sizeof(uint32_t) + 1 * sizeof(uint32_t);
    ASSERT_EQ(read_le<uint32_t>(data, column), 1u);
    write_le<uint32_t>(data, column, 0);
    write_file_bytes(test_file.path(), data);

    expect_load_rejected(test_file, "does not cover");
}

// The Shard{1} placement is flipped to Replicate, so the label calls (0,0) and (0,1) replicas while their records
// point at different data. Before this change the file loaded under that label; re-dumping it then failed the
// replica byte check, and before that check existed dropped the (0,1) shard.
TEST(TensorSerializationFlatbufferSchemaTest, InconsistentReplicaGroupRejected) {
    TemporaryFile test_file("inconsistent_replica_group.tensorbin");
    auto data = dump_two_shard_file(test_file);
    const size_t topology = follow(data, field_position(data, root_table(data), kTensorTopologyField));
    const size_t shard_placement = table_element(data, field_position(data, topology, kTopologyPlacementsField), 1);
    const size_t type = field_position(data, shard_placement, 0);
    ASSERT_EQ(std::to_integer<uint8_t>(data[type]), 1) << "MeshMapperPlacementType::Shard";
    data[type] = std::byte{0};  // MeshMapperPlacementType::Replicate
    write_file_bytes(test_file.path(), data);

    expect_load_rejected(test_file, "different data");
}

// The label's coordinate list is cut to one entry for a two-position distribution shape. `TensorTopology` accepts
// that, so before this change the file loaded and the first label-driven lookup indexed past the list.
TEST(TensorSerializationFlatbufferSchemaTest, MeshCoordsLengthMismatchRejected) {
    TemporaryFile test_file("mesh_coords_mismatch.tensorbin");
    auto data = dump_two_shard_file(test_file);
    const size_t topology = follow(data, field_position(data, root_table(data), kTensorTopologyField));
    const size_t coords_vector = follow(data, field_position(data, topology, kTopologyMeshCoordsField));
    ASSERT_EQ(read_le<uint32_t>(data, coords_vector), 2u);
    write_le<uint32_t>(data, coords_vector, 1);
    write_file_bytes(test_file.path(), data);

    expect_load_rejected(test_file, "lists 1 mesh coordinate(s)");
}

// A label with one placement for a two-dimensional distribution shape, over two shards holding the same bytes. Before
// this change the dump succeeded (the single placement was applied to axis 0 and both shards fell into one replica
// group that compared equal) and the file was then refused on load as corrupt; with distinct bytes the dedup guard
// fired instead, with a message about replicas. Now the writer rejects the label itself and writes no file.
TEST(TensorSerializationFlatbufferSchemaTest, PlacementCountMismatchRejectedOnDump) {
    TemporaryFile test_file("placement_count_mismatch.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const TensorTopology label(MeshShape(1, 2), Placements{shard_on(1)}, coords);
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {values, values}, label);

    expect_dump_rejected(test_file, tensor, "1 placement(s) for the 2-dimensional distribution shape");
}

}  // namespace
}  // namespace ttnn
