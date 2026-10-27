// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/cleanup.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <fmt/format.h>
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/distributed/utils.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/tensor/pinned_upload.hpp"

#include "ttnn/config.hpp"
#include "ttnn/tensor/overlapped_tensor.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/topology_check.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <random>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace ttnn {
namespace {

using ::testing::Each;
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

TEST_F(TensorSerializationFlatbufferTest, DeviceShardIsLabelledAsOneDeviceTensorAndDumps) {
    const int num_devices = mesh_device_->num_devices();
    if (num_devices < 2) {
        GTEST_SKIP() << "A shard taken out of a distributed tensor needs a mesh with more than one device";
    }
    TemporaryFile test_file("device_shard_flatbuffer.tensorbin");
    constexpr int kNumElements = 64;
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        std::generate_n(std::back_inserter(test_data), kNumElements, [i]() { return static_cast<float>(i); });
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_devices, kNumElements}, DataType::FLOAT32));
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device_, /*dim=*/0);
    Tensor sharded_tensor = ttnn::distributed::distribute_tensor(input_tensor, *mapper, *mesh_device_);
    ASSERT_TRUE(sharded_tensor.storage_type() == ttnn::StorageType::DEVICE);
    // The mapper's label: dim 0 split over every device, in row-major mesh order.
    const auto coords = sharded_tensor.device_storage().get_coords();
    const TensorTopology sharded_label(
        MeshShape(num_devices),
        {ttnn::distributed::MeshMapperConfig::Shard{0}},
        std::vector<MeshCoordinate>(coords.begin(), coords.end()));
    ASSERT_EQ(sharded_tensor.tensor_topology(), sharded_label);

    // Each shard is labelled as the one-device tensor it is, on the device and on the host, and the tensor it was
    // taken from keeps its label.
    std::vector<Tensor> shards = ttnn::distributed::get_device_tensors(sharded_tensor);
    ASSERT_THAT(shards, SizeIs(num_devices));
    for (int i = 0; i < num_devices; i++) {
        const TensorTopology expected_label(
            MeshShape(1), {ttnn::distributed::MeshMapperConfig::Replicate{}}, std::vector<MeshCoordinate>{coords[i]});
        EXPECT_EQ(shards[i].tensor_topology(), expected_label) << "shard " << i;
        EXPECT_EQ(shards[i].cpu().tensor_topology(), expected_label) << "shard " << i;
    }
    EXPECT_EQ(sharded_tensor.tensor_topology(), sharded_label);

    // A shard dumps on its own and loads back as that shard.
    dump_tensor_flatbuffer(test_file.string(), shards[1]);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_spec().logical_shape(), (ttnn::Shape{1, kNumElements}));
    std::vector<Tensor> loaded_shards = ttnn::distributed::get_device_tensors(loaded_tensor);
    ASSERT_THAT(loaded_shards, SizeIs(1));
    EXPECT_THAT(loaded_shards[0].to_vector<float>(), Each(FloatEq(1.0F)));

    // The shards still combine back into the tensor they were taken from.
    Tensor combined = ttnn::distributed::combine_device_tensors(shards, /*shard_dim=*/0);
    auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
    EXPECT_THAT(
        ttnn::distributed::aggregate_tensor(combined, *composer).to_vector<float>(), Pointwise(FloatEq(), test_data));
}

TEST_F(TensorSerializationFlatbufferTest, HostTensorAllocatedOverMeshIsLabelledOverEveryDeviceAndDumps) {
    TemporaryFile test_file("allocated_on_host_flatbuffer.tensorbin");
    constexpr int kNumElements = 64;
    const auto spec = get_tensor_spec(ttnn::Shape{1, kNumElements}, DataType::FLOAT32);
    const MeshShape& mesh_shape = mesh_device_->shape();

    // Before anything is read into it, the tensor is labelled over every device of the mesh, like the device tensor
    // create_device_tensor allocates, and the label matches the shards it holds.
    Tensor host_tensor = ttnn::allocate_tensor_on_host(spec, mesh_device_.get());
    const auto& label = host_tensor.tensor_topology();
    EXPECT_EQ(label.distribution_shape(), mesh_shape);
    EXPECT_EQ(
        label.placements(),
        ttsl::SmallVector<ttnn::distributed::MeshMapperConfig::Placement>(
            mesh_shape.dims(), ttnn::distributed::MeshMapperConfig::Replicate{}));
    std::vector<MeshCoordinate> every_coord;
    for (const auto& coord : MeshCoordinateRange(mesh_shape)) {
        every_coord.push_back(coord);
    }
    EXPECT_EQ(label.mesh_coords(), every_coord);
    EXPECT_EQ(label, ttnn::create_device_tensor(spec, mesh_device_.get()).tensor_topology());
    verify_shards_match_topology(host_tensor, /*compare_replica_contents=*/false);

    // Filled from a replicated device tensor, it dumps and loads as that tensor.
    std::vector<float> values(kNumElements);
    std::generate(values.begin(), values.end(), [n = 0.0F]() mutable { return n++; });
    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor =
        ttnn::distributed::distribute_tensor(Tensor::from_vector(values, spec), *mapper, *mesh_device_);
    ttnn::copy_to_host(replicated_tensor, host_tensor);
    dump_tensor_flatbuffer(test_file.string(), host_tensor);
    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    std::vector<Tensor> loaded_shards = ttnn::distributed::get_device_tensors(loaded_tensor);
    ASSERT_THAT(loaded_shards, SizeIs(mesh_device_->num_devices()));
    for (const auto& shard : loaded_shards) {
        EXPECT_THAT(shard.to_vector<float>(), Pointwise(FloatEq(), values));
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
// no device. Every "Rejected" test exercises one guard: the dump throws with that guard's diagnostic and leaves no
// file.

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
// `diagnostic` and that no file was created. The path is a fixed name under the temp directory, so a file left by an
// earlier interrupted run is removed first.
void expect_dump_rejected(const TemporaryFile& file, const Tensor& tensor, const std::string& diagnostic) {
    std::filesystem::remove(file.path());
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

// The check the writer runs is callable on its own, for callers that gather a tensor by its label: the mislabelled
// tensor is refused, a size-only check passes it, and the correctly labelled tensor passes.
TEST(TopologyCheckTest, VerifyShardsMatchTopology) {
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology replicate_label(MeshShape(2), Placements{kReplicate}, coords);
    const TensorTopology shard_label(MeshShape(2), Placements{shard_on(1)}, coords);
    Tensor tensor =
        make_host_tensor(MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, replicate_label);

    EXPECT_THAT(
        [&]() { verify_shards_match_topology(tensor); },
        ThrowsMessage<std::runtime_error>(HasSubstr("contents differ")));
    EXPECT_NO_THROW(verify_shards_match_topology(tensor, /*compare_replica_contents=*/false));
    tensor.update_tensor_topology(shard_label);
    EXPECT_NO_THROW(verify_shards_match_topology(tensor));
}

// Two distinct shards under the collapsed 1-D Replicate label the default mappers produce: the byte compare rejects
// the dump, since one copy standing for both coordinates would lose (0,1).
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

// With the byte compare turned off, the label is trusted: one copy, taken from the first replica, is written for
// the whole group.
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

// The label lists (0,1) but only (0,0) holds a shard: a two-coordinate label over one shard is rejected.
TEST(TensorSerializationFlatbufferGuardTest, LabelCoordWithoutShardRejected) {
    TemporaryFile test_file("label_coord_without_shard.tensorbin");
    const TensorTopology label(MeshShape(2), Placements{shard_on(1)}, all_coords(MeshShape(1, 2)));
    Tensor tensor = make_host_tensor(MeshShape(1, 2), {MeshCoordinate(0, 0)}, {{1.0f, 2.0f, 3.0f}}, label);

    expect_dump_rejected(test_file, tensor, "has no shard there");
}

// Both coordinates hold a shard but the label only covers (0,0): a shard the label does not cover is rejected
// rather than left out of the file.
TEST(TensorSerializationFlatbufferGuardTest, ShardOutsideLabelRejected) {
    TemporaryFile test_file("shard_outside_label.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(1, 2));
    const TensorTopology label(MeshShape(1), Placements{kReplicate}, {MeshCoordinate(0, 0)});
    Tensor tensor = make_host_tensor(MeshShape(1, 2), coords, {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, label);

    expect_dump_rejected(test_file, tensor, "does not cover");
}

// A label with one placement for a 2-D distribution shape says nothing about the second axis, so it cannot group the
// shards; it is rejected before any grouping, with or without the byte compare.
TEST(TensorSerializationFlatbufferGuardTest, PlacementCountMismatchRejected) {
    auto restore_config = scoped_replica_check(false);
    TemporaryFile test_file("placement_count_mismatch.tensorbin");
    const std::vector<MeshCoordinate> coords = all_coords(MeshShape(2, 2));
    const TensorTopology label(MeshShape(2, 2), Placements{shard_on(0)}, coords);
    Tensor tensor = make_host_tensor(MeshShape(2, 2), coords, {{0.0f}, {1.0f}, {2.0f}, {3.0f}}, label);

    expect_dump_rejected(test_file, tensor, "one placement per distribution dimension");
}

// A 30 x 30 float32 tile-layout tensor is one 32 x 32 tile with 124 padding elements. `from_vector` encodes the
// logical ones with zero padding; the result is the physical buffer of replica A, and replica B is a copy with every
// padding element overwritten by a distinct non-zero value, so the two differ in bytes but not in values.
struct PaddedReplicas {
    TensorSpec spec{ttnn::Shape{1, 1, 30, 30}, TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{})};
    std::vector<float> a;  // zero padding
    std::vector<float> b;  // random padding
};
PaddedReplicas make_padded_replicas() {
    PaddedReplicas r;
    const Tensor encoded = Tensor::from_vector(std::vector<float>(30 * 30, 1.0f), r.spec);
    const auto& buffer = encoded.host_storage().buffer();
    const auto shard = buffer.get_shard(*buffer.shard_coords().begin());
    TT_FATAL(shard.has_value(), "test bug: the encoded tensor has no shard");
    const auto view = shard->view_as<float>();
    r.a.assign(view.begin(), view.end());
    TT_FATAL(r.a.size() == 32 * 32, "test bug: expected one 32 x 32 tile");
    r.b = r.a;
    float fill = 2.0f;
    for (float& value : r.b) {
        if (value == 0.0f) {  // every logical value is one, so a zero is a padding element
            value = fill;
            fill += 0.5f;
        }
    }
    TT_FATAL(fill > 2.0f, "test bug: no padding element found");
    return r;
}

// Builds the 1x2 host tensor with `a` at (0,0) and `b` at (0,1) under a Replicate label.
Tensor make_padded_replica_tensor(const PaddedReplicas& r, const std::vector<float>& b) {
    const MeshShape buffer_shape(1, 2);
    const std::vector<MeshCoordinate> coords = all_coords(buffer_shape);
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    buffer.emplace_shard(coords[0], [&r]() { return HostBuffer(std::vector<float>(r.a)); });
    buffer.emplace_shard(coords[1], [&b]() { return HostBuffer(std::vector<float>(b)); });
    return Tensor(host_tensor_from_buffer_with_topology(std::move(buffer), r.spec, label));
}

// Replicas whose logical values match while every tile padding element differs. Padding holds whatever the producing
// op left there, so the label is honest: the dump writes one copy and the logical values load back intact at both
// coordinates. The bytes differ, so the values decide (the in-place walk over the logical region).
TEST(TensorSerializationFlatbufferGuardTest, ReplicasDifferingOnlyInTilePaddingAccepted) {
    TemporaryFile test_file("replicas_padding_differs.tensorbin");
    const PaddedReplicas r = make_padded_replicas();
    Tensor tensor = make_padded_replica_tensor(r, r.b);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_EQ(loaded_tensor.tensor_topology(), tensor.tensor_topology());
    for (const auto& coord : all_coords(MeshShape(1, 2))) {
        const auto shard = loaded_tensor.host_storage().buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value()) << "no shard at " << coord;
        const auto values = tt::tt_metal::HostTensor::from_buffer(*shard, r.spec).to_vector<float>();
        EXPECT_THAT(values, SizeIs(30 * 30));
        EXPECT_THAT(values, Each(FloatEq(1.0f)));
    }
}

// The same padded replicas, but one logical value of the second differs as well: the value comparison rejects the
// dump, so differing padding does not hide a real difference.
TEST(TensorSerializationFlatbufferGuardTest, PaddedReplicasWithDifferentValuesRejected) {
    TemporaryFile test_file("padded_replicas_values_differ.tensorbin");
    const PaddedReplicas r = make_padded_replicas();
    std::vector<float> b = r.b;
    const auto logical = std::find(b.begin(), b.end(), 1.0f);
    ASSERT_NE(logical, b.end());
    *logical = 3.0f;
    Tensor tensor = make_padded_replica_tensor(r, b);

    expect_dump_rejected(test_file, tensor, "contents differ");
}

// The value comparison of padded replicas across layouts: partial tiles in both dimensions, padding rows inside
// every batch element, a single column, a single row, rank 3, rank 1 (a partial and a whole tile width) and rank 0;
// bf16, fp32, uint8 and uint32. Replica A is the encoding of random values with zero padding, B carries the same values
// under random padding and is accepted, C is B with one logical value changed and is rejected.
template <typename T>
T padded_value(float f) {
    if constexpr (std::is_same_v<T, bfloat16>) {
        return bfloat16(f);
    } else {
        return static_cast<T>(f);
    }
}
template <typename T>
bool padded_is_zero(T v) {
    if constexpr (std::is_same_v<T, bfloat16>) {
        return static_cast<float>(v) == 0.0f;
    } else {
        return v == T(0);
    }
}
template <typename T>
DataType padded_dtype() {
    if constexpr (std::is_same_v<T, bfloat16>) {
        return DataType::BFLOAT16;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else {
        return DataType::UINT32;
    }
}
// The physical buffer of `spec` encoded from random non-zero values (so a zero marks a padding element), with the
// padding positions.
template <typename T>
std::pair<std::vector<T>, std::vector<size_t>> encode_random_padded(const TensorSpec& spec, std::mt19937& rng) {
    std::vector<T> logical(spec.logical_shape().volume());
    std::uniform_int_distribution<int> value(1, 100);
    for (auto& v : logical) {
        v = padded_value<T>(static_cast<float>(value(rng)));
    }
    const Tensor encoded = Tensor::from_vector(logical, spec);
    const auto& buffer = encoded.host_storage().buffer();
    const auto shard = buffer.get_shard(*buffer.shard_coords().begin());
    TT_FATAL(shard.has_value(), "test bug: the encoded tensor has no shard");
    const auto view = shard->view_as<T>();
    std::vector<T> physical(view.begin(), view.end());
    std::vector<size_t> padding;
    for (size_t i = 0; i < physical.size(); ++i) {
        if (padded_is_zero(physical[i])) {
            padding.push_back(i);
        }
    }
    return {physical, padding};
}
template <typename T>
Tensor two_replicas(const TensorSpec& spec, const std::vector<T>& a, const std::vector<T>& b) {
    const MeshShape buffer_shape(1, 2);
    const std::vector<MeshCoordinate> coords = all_coords(buffer_shape);
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    buffer.emplace_shard(coords[0], [&a]() { return HostBuffer(std::vector<T>(a)); });
    buffer.emplace_shard(coords[1], [&b]() { return HostBuffer(std::vector<T>(b)); });
    return Tensor(host_tensor_from_buffer_with_topology(std::move(buffer), spec, label));
}
template <typename T>
void check_padded_replicas(const ttnn::Shape& shape, std::mt19937& rng) {
    const TensorSpec spec(shape, TensorLayout(padded_dtype<T>(), Layout::TILE, MemoryConfig{}));
    auto [a, padding] = encode_random_padded<T>(spec, rng);
    ASSERT_FALSE(padding.empty()) << shape;
    std::vector<size_t> logical_positions;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!padded_is_zero(a[i])) {
            logical_positions.push_back(i);
        }
    }
    std::uniform_int_distribution<int> fill(1, 250);
    std::uniform_int_distribution<size_t> pick(0, logical_positions.size() - 1);
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<T> b = a;
        for (size_t idx : padding) {
            b[idx] = padded_value<T>(static_cast<float>(fill(rng)));
        }
        std::vector<T> c = b;
        const size_t pos = logical_positions[pick(rng)];
        if constexpr (std::is_integral_v<T>) {
            c[pos] = static_cast<T>(c[pos] + T(1));  // wrapping still differs from the original
        } else {
            c[pos] = padded_value<T>(static_cast<float>(fill(rng)) + 300.0f);
        }
        ASSERT_FALSE(padded_is_zero(c[pos])) << "test bug: the changed value reads as padding";
        EXPECT_NO_THROW(verify_shards_match_topology(two_replicas<T>(spec, a, b))) << shape << " trial " << trial;
        EXPECT_THAT(
            [&]() { verify_shards_match_topology(two_replicas<T>(spec, a, c)); },
            ThrowsMessage<std::runtime_error>(HasSubstr("contents differ")))
            << shape << " trial " << trial;
    }
}

TEST(TensorSerializationFlatbufferGuardTest, PaddedReplicasAcrossLayoutsAcceptedAndRejected) {
    std::mt19937 rng(7);
    for (const auto& shape :
         {ttnn::Shape{1, 1, 30, 30},
          ttnn::Shape{1, 1, 100, 70},
          ttnn::Shape{2, 3, 45, 33},
          ttnn::Shape{1, 1, 33, 1},
          ttnn::Shape{3, 17, 40},
          ttnn::Shape{1, 1, 1, 50},
          ttnn::Shape{50},
          ttnn::Shape{64},
          ttnn::Shape{}}) {
        check_padded_replicas<bfloat16>(shape, rng);
        check_padded_replicas<float>(shape, rng);
        check_padded_replicas<uint8_t>(shape, rng);
        check_padded_replicas<uint32_t>(shape, rng);
    }
}

// Padded replicas on the layouts the in-place walk does not describe (block floats, a transposed tile, FP8_E4M3 under
// an alignment that pads a leading dimension), on every tile shape the host supports, and on row-major tensors with
// alignment padding. Each replica is encoded from the same values with a different pad value, so the bytes differ in
// the padding alone; a third replica changes one logical value. Values 2..100 keep a block float's shared exponents
// unchanged by a pad value of 1.
// The shard buffer `from_vector` produces, kept as is so it carries the element type the dtype's readers expect
// (uint32 for packed block floats), as a shard read back from a device does.
HostBuffer encoded_shard(const TensorSpec& spec, const std::vector<float>& logical, float pad_value) {
    const Tensor encoded = Tensor::from_vector(logical, spec, /*device=*/nullptr, /*cq_id=*/std::nullopt, pad_value);
    const auto& buffer = encoded.host_storage().buffer();
    const auto shard = buffer.get_shard(*buffer.shard_coords().begin());
    TT_FATAL(shard.has_value(), "test bug: the encoded tensor has no shard");
    return *shard;
}
std::vector<std::byte> bytes_of(const HostBuffer& shard) {
    const auto view = shard.view_bytes();
    return std::vector<std::byte>(view.begin(), view.end());
}
Tensor two_encoded_replicas(const TensorSpec& spec, const HostBuffer& a, const HostBuffer& b) {
    const MeshShape buffer_shape(1, 2);
    const std::vector<MeshCoordinate> coords = all_coords(buffer_shape);
    const TensorTopology label(MeshShape(2), Placements{kReplicate}, coords);
    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    buffer.emplace_shard(coords[0], [&a]() { return a; });
    buffer.emplace_shard(coords[1], [&b]() { return b; });
    return Tensor(host_tensor_from_buffer_with_topology(std::move(buffer), spec, label));
}
void check_encoded_padded_replicas(const std::string& what, const TensorSpec& spec, std::mt19937& rng) {
    ASSERT_NE(spec.logical_2d_shape(), spec.physical_shape()) << what << ": the spec has no padding";
    std::vector<float> logical(spec.logical_shape().volume());
    std::uniform_int_distribution<int> value(2, 100);
    for (auto& v : logical) {
        v = static_cast<float>(value(rng));
    }
    std::vector<float> changed = logical;
    std::uniform_int_distribution<size_t> pick(0, logical.size() - 1);
    const size_t pos = pick(rng);
    changed[pos] = logical[pos] + 300.0f;

    const HostBuffer a = encoded_shard(spec, logical, 0.0f);
    const HostBuffer b = encoded_shard(spec, logical, 1.0f);
    const HostBuffer c = encoded_shard(spec, changed, 1.0f);
    ASSERT_NE(bytes_of(a), bytes_of(b)) << what << ": the pad value left the bytes unchanged";

    EXPECT_NO_THROW(verify_shards_match_topology(two_encoded_replicas(spec, a, b))) << what;
    EXPECT_THAT(
        [&]() { verify_shards_match_topology(two_encoded_replicas(spec, a, c)); },
        ThrowsMessage<std::runtime_error>(HasSubstr("contents differ")))
        << what;
}

TEST(TensorSerializationFlatbufferGuardTest, PaddedReplicasOnEveryTileShapeAcceptedAndRejected) {
    std::mt19937 rng(13);
    const ttnn::Shape shape{1, 1, 30, 30};
    for (const auto& tile_shape : std::vector<std::array<uint32_t, 2>>{
             {32, 32},
             {16, 32},
             {32, 16},
             {16, 16},
             {8, 32},
             {4, 32},
             {2, 32},
             {1, 32},
             {8, 16},
             {4, 16},
             {2, 16},
             {1, 16}}) {
        for (const DataType dtype : {DataType::BFLOAT16, DataType::FLOAT32}) {
            const TensorSpec spec(
                shape, TensorLayout(dtype, PageConfig(Layout::TILE, Tile(tile_shape)), MemoryConfig{}));
            check_encoded_padded_replicas(fmt::format("tile {}x{} {}", tile_shape[0], tile_shape[1], dtype), spec, rng);
        }
    }
}

TEST(TensorSerializationFlatbufferGuardTest, PaddedReplicasBeyondTheInPlaceWalkAcceptedAndRejected) {
    std::mt19937 rng(17);
    // Block floats: packed rows share an exponent, so the comparison is on unpacked values.
    check_encoded_padded_replicas(
        "bfloat8_b 30x30",
        TensorSpec(ttnn::Shape{1, 1, 30, 30}, TensorLayout(DataType::BFLOAT8_B, Layout::TILE, MemoryConfig{})),
        rng);
    check_encoded_padded_replicas(
        "bfloat4_b 30x30",
        TensorSpec(ttnn::Shape{1, 1, 30, 30}, TensorLayout(DataType::BFLOAT4_B, Layout::TILE, MemoryConfig{})),
        rng);
    // FP8_E4M3, which is row-major only, under an alignment that pads the leading dimension from 3 to 4.
    check_encoded_padded_replicas(
        "fp8_e4m3 3x2x30 aligned to 4x1x32",
        TensorSpec(
            ttnn::Shape{3, 2, 30},
            TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, Alignment({4, 1, 32}))),
        rng);
    // A transposed tile.
    check_encoded_padded_replicas(
        "transposed tile bf16 30x30",
        TensorSpec(
            ttnn::Shape{1, 1, 30, 30},
            TensorLayout(
                DataType::BFLOAT16, PageConfig(Layout::TILE, Tile({32, 32}, /*transpose_tile=*/true)), MemoryConfig{})),
        rng);
}

// Row-major replicas with alignment padding on the width, compared in place: rank 4 and rank 1, bf16 and FP8_E4M3.
TEST(TensorSerializationFlatbufferGuardTest, PaddedRowMajorReplicasAcceptedAndRejected) {
    std::mt19937 rng(19);
    const std::vector<std::pair<std::string, ttnn::Shape>> shapes{
        {"1x1x30x30", ttnn::Shape{1, 1, 30, 30}}, {"rank-1 50", ttnn::Shape{50}}};
    for (const DataType dtype : {DataType::BFLOAT16, DataType::FP8_E4M3}) {
        for (const auto& [name, shape] : shapes) {
            check_encoded_padded_replicas(
                fmt::format("row-major {} {} aligned to 64", dtype, name),
                TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, Alignment({64}))),
                rng);
        }
    }
}

// A rank-1 tile-layout tensor is one row padded to a whole tile, and its replicas are compared in place. Held as raw
// bytes, which the in-place walk reads and the decoded comparison does not, they are still accepted when only the
// padding differs and rejected when a value does.
TEST(TensorSerializationFlatbufferGuardTest, RankOneReplicasComparedInPlace) {
    const TensorSpec spec(ttnn::Shape{50}, TensorLayout(DataType::BFLOAT16, Layout::TILE, MemoryConfig{}));
    std::vector<float> logical(50);
    std::iota(logical.begin(), logical.end(), 2.0f);
    std::vector<float> changed = logical;
    changed[7] += 300.0f;
    const auto raw = [&spec](const std::vector<float>& values, float pad_value) {
        return HostBuffer(bytes_of(encoded_shard(spec, values, pad_value)));
    };

    EXPECT_NO_THROW(verify_shards_match_topology(two_encoded_replicas(spec, raw(logical, 0.0f), raw(logical, 1.0f))));
    EXPECT_THAT(
        [&]() { verify_shards_match_topology(two_encoded_replicas(spec, raw(logical, 0.0f), raw(changed, 1.0f))); },
        ThrowsMessage<std::runtime_error>(HasSubstr("contents differ")));
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

// With the byte compare off, the first replica stands for its group, and a shard folded into that copy must not lend
// it to another group through the address lookup: (0,1) and (1,0) share one HostBuffer but sit in different groups
// (the rows), so row 1 has to be written from its own bytes.
TEST(TensorSerializationFlatbufferGuardTest, AliasAcrossGroupsKeepsItsOwnBytesWhenOptedOut) {
    auto restore_config = scoped_replica_check(false);
    TemporaryFile test_file("alias_across_groups_opt_out.tensorbin");
    const MeshShape buffer_shape(2, 2);
    const std::vector<float> x{1.0f, 2.0f};
    const std::vector<float> y{3.0f, 4.0f};
    const std::vector<float> z{5.0f, 6.0f};
    const TensorTopology label(buffer_shape, Placements{shard_on(0), kReplicate}, all_coords(buffer_shape));

    auto buffer = DistributedHostBuffer::create(
        buffer_shape, buffer_shape, MeshCoordinate::zero_coordinate(buffer_shape.dims()), /*context=*/nullptr);
    HostBuffer shared_y{std::vector<float>(y)};
    buffer.emplace_shard(MeshCoordinate(0, 0), [&x]() { return HostBuffer(std::vector<float>(x)); });
    buffer.emplace_shard(MeshCoordinate(0, 1), [&shared_y]() { return shared_y; });
    buffer.emplace_shard(MeshCoordinate(1, 0), [&shared_y]() { return shared_y; });
    buffer.emplace_shard(MeshCoordinate(1, 1), [&z]() { return HostBuffer(std::vector<float>(z)); });
    Tensor tensor = wrap_host_buffer(std::move(buffer), x.size(), label);

    dump_tensor_flatbuffer(test_file.string(), tensor, DumpTensorMode::LOCAL);

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    EXPECT_THAT(shard_values(loaded_tensor, MeshCoordinate(0, 0)), Pointwise(FloatEq(), x));
    EXPECT_THAT(shard_values(loaded_tensor, MeshCoordinate(0, 1)), Pointwise(FloatEq(), x));
    EXPECT_THAT(shard_values(loaded_tensor, MeshCoordinate(1, 0)), Pointwise(FloatEq(), y));
    EXPECT_THAT(shard_values(loaded_tensor, MeshCoordinate(1, 1)), Pointwise(FloatEq(), y));
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

// The address shortcut compares lengths too. Under a Replicate label the two views land in one group, where the
// size check rejects them: one record cannot stand for both 8 and 12 bytes.
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

// A loaded tensor keeps its file mapped, so dump_overlapped_tensors has to replace an existing file rather than
// rewrite it: views loaded from the old file keep reading the old contents.
TEST(TensorSerializationOverwriteTest, OverwritingOverlappedTensorsFileKeepsLoadedViewContents) {
    TemporaryFile test_file("overwrite.overlappedtensorbin");
    const ttnn::Shape shape{1, 1, 32, 32};
    const std::vector<uint32_t> original_data(shape.volume(), 1);
    const std::vector<uint32_t> replacement_data(shape.volume(), 2);
    const auto single_view = [&](const std::vector<uint32_t>& data) {
        return std::vector<OverlappedTensorView>{{
            .name = "view",
            .fused_tensor = Tensor::from_vector(data, get_tensor_spec(shape, DataType::UINT32)),
            .tensor_shape = {32, 32},
            .shard_shape = {32, 32},
            .core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
            .dtype = DataType::UINT32,
            .tile_shape = {32, 32},
            .byte_offset = 0,
            .total_size = data.size() * sizeof(uint32_t),
        }};
    };

    dump_overlapped_tensors(test_file.string(), single_view(original_data));
    std::vector<OverlappedTensorView> loaded_views = load_overlapped_tensors(test_file.string());
    dump_overlapped_tensors(test_file.string(), single_view(replacement_data));

    ASSERT_THAT(loaded_views, SizeIs(1));
    EXPECT_EQ(loaded_views[0].fused_tensor.to_vector<uint32_t>(), original_data);
    std::vector<OverlappedTensorView> reloaded_views = load_overlapped_tensors(test_file.string());
    ASSERT_THAT(reloaded_views, SizeIs(1));
    EXPECT_EQ(reloaded_views[0].fused_tensor.to_vector<uint32_t>(), replacement_data);
}

// Returns the permission column of the /proc/self/maps entry that contains `address` -- e.g. "r--s", where the
// fourth character is 's' for a shared mapping and 'p' for a private one -- or nullopt if no entry contains it.
std::optional<std::string> mapping_permissions(uintptr_t address) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        uintptr_t start = 0;
        uintptr_t end = 0;
        std::array<char, 5> permissions{};
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " %4s", &start, &end, permissions.data()) == 3 &&
            start <= address && address < end) {
            return std::string(permissions.data());
        }
    }
    return std::nullopt;
}

using TensorSerializationPinnedUploadTest = MeshDevice1x1Fixture;

// load_tensor_flatbuffer maps the file MAP_SHARED so that a pinned upload pins the file's page-cache pages in place,
// instead of making the kernel copy every page of a private mapping first. This checks both halves: the loader's
// mapping is shared, and an upload from it creates a device-read-only pin.
TEST_F(TensorSerializationPinnedUploadTest, LargeLoadedTensorUploadPinsSharedFileMapping) {
    const ttnn::Shape shape{1, 1, 1024, 9216};
    const size_t size_bytes = static_cast<size_t>(shape.volume()) * sizeof(uint32_t);
    ASSERT_GT(size_bytes, pinned_upload::k_pin_write_threshold_bytes);

    distributed::MeshDevice& mesh_device = *mesh_device_;
    const auto pinning_params = experimental::GetMemoryPinningParameters(mesh_device);
    if (!pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }
    const size_t cache_limit = mesh_device.impl().metal_env().get_rtoptions().get_pinned_memory_cache_limit_bytes();
    if (cache_limit < size_bytes || pinning_params.max_total_pin_size < size_bytes || pinning_params.max_pins < 1) {
        GTEST_SKIP() << "Requires a pin budget large enough for a " << size_bytes << " byte mapping";
    }

    // Position-dependent contents, so the round trip below catches a wrong offset or a repeated page.
    std::vector<uint32_t> data(shape.volume());
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint32_t>(i) * 2654435761u + 0x9e3779b9u;
    }
    TemporaryFile test_file("large_read_only.tensorbin");
    dump_tensor_flatbuffer(test_file.string(), Tensor::from_vector(data, get_tensor_spec(shape, DataType::UINT32)));

    // load_tensor_flatbuffer falls back to MAP_PRIVATE on a filesystem that refuses MAP_SHARED, so a private mapping
    // there is correct behavior, not a regression. Probe the filesystem directly to tell the two apart.
    // LargeReadOnlyPrivateFileBackedWriteUsesReadOnlyPinnedMemory in unit_tests_tensor covers pinning the fallback.
    {
        const int fd = open(test_file.string().c_str(), O_RDONLY | O_CLOEXEC);
        ASSERT_NE(fd, -1) << strerror(errno);
        void* probe = mmap(nullptr, size_bytes, PROT_READ, MAP_SHARED, fd, 0);
        const int probe_errno = errno;
        close(fd);
        if (probe == MAP_FAILED) {
            GTEST_SKIP() << "The filesystem holding " << test_file.string()
                         << " refuses MAP_SHARED: " << strerror(probe_errno);
        }
        munmap(probe, size_bytes);
    }

    Tensor loaded_tensor = load_tensor_flatbuffer(test_file.string());
    const std::vector<uintptr_t> addresses = shard_addresses(loaded_tensor);
    ASSERT_THAT(addresses, SizeIs(1));
    EXPECT_EQ(mapping_permissions(addresses[0]), std::string("r--s"));

    auto& cache = experimental::PinnedMemoryCache::instance();
    const size_t entries_before = cache.num_entries();
    Tensor device_tensor = loaded_tensor.to_device(&mesh_device);

    // The upload itself must have created the pin, and the pin must be device-read-only: the mapping is PROT_READ of
    // an O_RDONLY descriptor, so a read/write pin of it would fail. The try_pin below finds the upload's entry rather
    // than creating a second one, so the entry count does not move.
    ASSERT_EQ(cache.num_entries(), entries_before + 1);
    const auto coord = *distributed::MeshCoordinateRange(mesh_device.shape()).begin();
    std::optional<HostBuffer> shard = loaded_tensor.host_storage().buffer().get_shard(coord);
    ASSERT_TRUE(shard.has_value());
    auto pinned = cache.try_pin(
        mesh_device,
        distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord)),
        *shard,
        /*map_to_noc=*/true,
        experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned);
    EXPECT_EQ(cache.num_entries(), entries_before + 1);
    EXPECT_EQ(pinned->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadOnly);

    EXPECT_EQ(device_tensor.cpu().to_vector<uint32_t>(), data);
}

}  // namespace
}  // namespace ttnn
