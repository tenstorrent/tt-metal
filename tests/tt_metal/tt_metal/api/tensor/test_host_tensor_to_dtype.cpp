// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {


// Deterministic ramp for test data
template <typename T>
std::vector<T> make_ramp(size_t count) {
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(static_cast<float>(i % 251));
    }
    return data;
}

bool exact_spec_match(const TensorSpec& a, const TensorSpec& b) {
    return a == b && experimental::per_core_allocation::is_per_core_allocation(a.memory_config()) ==
                         experimental::per_core_allocation::is_per_core_allocation(b.memory_config());
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

TEST(HostTensorToDtype, NonBfpPreservesMetadata) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
}

TEST(HostTensorToDtype, RowMajorToBfpChangesLayoutToTileAndPreservesTile) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    // Create a ROW_MAJOR tensor, but specify a tile in the page config (embedded tile)
    auto source_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT8_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    // Value check on the aligned supported path
    // We can unpack the BFP8_B data to float and check if it matches the original data
    auto result_packed_data = host_buffer::get_as<uint32_t>(result);
    auto unpacked_data =
        unpack_bfp8_tiles_into_float_vec(result_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto rm_data =
        tensor_impl::to_row_major_layout(expected_spec.physical_shape(), tile, ttsl::make_const_span(unpacked_data));

    EXPECT_EQ(rm_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(rm_data[i], data[i], 1.0f);
    }
}

TEST(HostTensorToDtype, TileBfp8ToFloat32ValueCheck) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile(
        {16,
         16});  // Custom tile size could be used here if supported, but let's stick to 16x16 or 32x32. Let's use 16x16.

    // Create boundary-sensitive float data
    std::vector<float> float_data(shape.volume());
    for (size_t i = 0; i < float_data.size(); ++i) {
        // Values that test quantization and faces
        float_data[i] = static_cast<float>(i % 16) + (i / 16) * 0.5f;
    }

    // Independently pack to BFP8
    auto packed_bfp8_data =
        pack_as_bfp8_tiles(ttsl::make_const_span(float_data), /*row_major_input=*/true, /*is_exp_a=*/false, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<uint32_t>(packed_bfp8_data)); });
    auto source = HostTensor::from_buffer(std::move(dist_buffer), source_spec, TensorTopology{});

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    // Independently unpack golden
    auto expected_unpacked_data =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto result_data = host_buffer::get_as<float>(result);
    EXPECT_EQ(result_data.size(), expected_unpacked_data.size());
    for (size_t i = 0; i < expected_unpacked_data.size(); ++i) {
        EXPECT_EQ(result_data[i], expected_unpacked_data[i]);
    }
}

TEST(HostTensorToDtype, Float32ToTileBfp8ValueCheck) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    // Create boundary-sensitive float data in TILE layout
    std::vector<float> float_data(shape.volume());
    for (size_t i = 0; i < float_data.size(); ++i) {
        float_data[i] = static_cast<float>(i % 16) + (i / 16) * 0.5f;
    }

    // Independently pack to BFP8 golden
    auto expected_packed_data =
        pack_as_bfp8_tiles(ttsl::make_const_span(float_data), /*row_major_input=*/false, /*is_exp_a=*/false, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(float_data)); });
    auto source = HostTensor::from_buffer(std::move(dist_buffer), source_spec, TensorTopology{});

    auto result = to_dtype(source, DataType::BFLOAT8_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    auto result_data = host_buffer::get_as<uint32_t>(result);
    EXPECT_EQ(result_data.size(), expected_packed_data.size());
    for (size_t i = 0; i < expected_packed_data.size(); ++i) {
        EXPECT_EQ(result_data[i], expected_packed_data[i]);
    }
}

TEST(HostTensorToDtype, TileBfp4ToFloat32ValueCheck) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    std::vector<float> float_data(shape.volume());
    for (size_t i = 0; i < float_data.size(); ++i) {
        float_data[i] = static_cast<float>(i % 16) + (i / 16) * 0.5f;
    }

    // Independently pack to BFP4
    auto packed_bfp4_data =
        pack_as_bfp4_tiles(ttsl::make_const_span(float_data), /*row_major_input=*/true, /*is_exp_a=*/false, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<uint32_t>(packed_bfp4_data)); });
    auto source = HostTensor::from_buffer(std::move(dist_buffer), source_spec, TensorTopology{});

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    // Independently unpack golden
    auto expected_unpacked_data =
        unpack_bfp4_tiles_into_float_vec(packed_bfp4_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto result_data = host_buffer::get_as<float>(result);
    EXPECT_EQ(result_data.size(), expected_unpacked_data.size());
    for (size_t i = 0; i < expected_unpacked_data.size(); ++i) {
        EXPECT_EQ(result_data[i], expected_unpacked_data[i]);
    }
}

TEST(HostTensorToDtype, Float32ToTileBfp4ValueCheck) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    std::vector<float> float_data(shape.volume());
    for (size_t i = 0; i < float_data.size(); ++i) {
        float_data[i] = static_cast<float>(i % 16) + (i / 16) * 0.5f;
    }

    // Independently pack to BFP4 golden
    auto expected_packed_data =
        pack_as_bfp4_tiles(ttsl::make_const_span(float_data), /*row_major_input=*/false, /*is_exp_a=*/false, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(float_data)); });
    auto source = HostTensor::from_buffer(std::move(dist_buffer), source_spec, TensorTopology{});

    auto result = to_dtype(source, DataType::BFLOAT4_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT4_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    auto result_data = host_buffer::get_as<uint32_t>(result);
    EXPECT_EQ(result_data.size(), expected_packed_data.size());
    for (size_t i = 0; i < expected_packed_data.size(); ++i) {
        EXPECT_EQ(result_data[i], expected_packed_data[i]);
    }
}

TEST(HostTensorToDtype, Float32ToBfloat16RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<bfloat16>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(data[i])));
    }
}

TEST(HostTensorToDtype, Bfloat16ToFloat32RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<bfloat16>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<bfloat16>(data, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<float>(data[i]));
    }
}

TEST(HostTensorToDtype, PerCoreAllocationPreserved) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());

    // Sharded MemoryConfig
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto alignment = tt::tt_metal::Alignment({32, 64});
    auto tile = Tile({32, 32});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.tensor_spec().memory_config()));
}

TEST(HostTensorToDtype, OversizedBufferToDtype) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    // Create an oversized buffer
    auto oversized_data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume() + 100);

    // Create tensor from buffer (from_buffer accepts oversized buffers if they are large enough)
    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(oversized_data)); });
    auto host_tensor = HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});

    // Currently to_dtype asserts on exact packed size, so it will fail if it's oversized.
    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::BFLOAT16));
}

TEST(HostTensorToDtype, UndersizedBufferToDtypeThrows) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Use BFLOAT8_B so we test the pre-unpack size check
    auto spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    // Create an undersized buffer
    auto undersized_data = std::vector<uint32_t>(10, 0);  // Much smaller than required

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<uint32_t>(undersized_data)); });

    // from_buffer doesn't check size, so this succeeds
    auto host_tensor = HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});

    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::FLOAT32));
}

TEST(HostTensorToDtype, MalformedBfpBufferToDtypeThrows) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    // Create a buffer that is slightly off in size (e.g. missing one word)
    size_t expected_size_bytes = spec.compute_packed_buffer_size_bytes();
    auto malformed_data = std::vector<uint32_t>((expected_size_bytes / sizeof(uint32_t)) - 1, 0);

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<uint32_t>(malformed_data)); });

    auto host_tensor = HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});

    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::FLOAT32));
}
TEST(HostTensorToDtype, MultiShardTopologyPreservation) {
    const Shape shape{32, 64};
    const size_t volume = shape.volume();

    // Create sharded MemoryConfig
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};

    auto alignment = tt::tt_metal::Alignment({32, 64});
    auto tile = Tile({16, 16});
    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    // Create fully populated multi-shard host tensor on 1x2 mesh
    auto topology = TensorTopology::create_fully_replicated_tensor_topology(distributed::MeshShape(1, 2));

    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 2));

    auto data_0 = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(volume);
    auto data_1 = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(volume);
    // Make data distinct
    for (auto& v : data_1) {
        v += 10.0f;
    }

    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(data_0)); });
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 1), [&]() { return HostBuffer(std::vector<float>(data_1)); });

    auto source = HostTensor::from_buffer(std::move(distributed_buffer), source_spec, topology);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.tensor_topology(), topology);

    const size_t expected_shard_size = expected_spec.compute_packed_buffer_size_bytes();

    // Check coordinate set
    auto coords = result.buffer().shard_coords();
    EXPECT_EQ(coords.size(), 2);
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 0)));
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 1)));

    for (const auto& coord : coords) {
        auto shard = result.buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value());
        EXPECT_EQ(shard->view_bytes().size(), expected_shard_size);

        auto result_data = shard->view_as<bfloat16>();
        EXPECT_EQ(result_data.size(), volume);

        const auto& expected_data = (coord == distributed::MeshCoordinate(0, 0)) ? data_0 : data_1;
        for (size_t i = 0; i < volume; ++i) {
            EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(expected_data[i])));
        }
    }
}

TEST(HostTensorToDtype, RowMajorToBfpPhysicalMismatchThrows) {
    const Shape shape{32, 24};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Alignment that is NOT a multiple of 16 (tile size)
    auto alignment = tt::tt_metal::Alignment({32, 24});
    auto tile = Tile({16, 16});

    auto source_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    // This should throw because BFLOAT8_B forces TILE layout, which forces alignment to be multiple of 16
    // So output physical shape width will be 32 instead of 24
    EXPECT_ANY_THROW(to_dtype(source, DataType::BFLOAT8_B));
}

}  // namespace
}  // namespace tt::tt_metal
