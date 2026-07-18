// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

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

void expect_packed_sizes(const HostTensor& tensor) {
    const size_t expected = tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& coord : tensor.buffer().shard_coords()) {
        auto shard = tensor.buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value());
        EXPECT_EQ(shard->view_bytes().size(), expected);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

TEST(HostTensorSpecPreservation, ToLayoutInterleavedRmTileRoundTrip) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    // Interleaved cannot set per_core (requires sharded); exact_spec_match still checks the flag.
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Alignment coherent with TILE after initialize_alignment (multiples of tile H/W).
    auto alignment = Alignment({32, 32});
    auto tile = Tile({16, 16});

    auto tile_source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto tile_source = HostTensor::from_vector<float>(data, tile_source_spec);

    auto rm = to_layout(tile_source, Layout::ROW_MAJOR);
    auto expected_rm_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(rm.tensor_spec(), expected_rm_spec));
    EXPECT_EQ(rm.memory_config().buffer_type(), BufferType::DRAM);
    EXPECT_EQ(rm.tensor_topology(), tile_source.tensor_topology());
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(rm);

    // RM PageConfig does not carry a non-default tile; restore TILE with explicit tile.
    auto tiled_back = to_tile_layout(rm, tile);
    auto expected_tile_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(tiled_back.tensor_spec(), expected_tile_spec));
    EXPECT_EQ(tiled_back.memory_config().buffer_type(), BufferType::DRAM);
    EXPECT_EQ(tiled_back.tensor_topology(), tile_source.tensor_topology());
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(tiled_back);

    auto round_trip = tiled_back.to_vector<float>();
    ASSERT_EQ(round_trip.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(round_trip[i], data[i]);
    }
}

TEST(HostTensorSpecPreservation, ToLayoutShardedPreservesShardSpecAndPackedSizes) {
    const Shape shape{32, 64};
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);
    auto alignment = Alignment({32, 64});
    auto tile = Tile({16, 16});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto topology = TensorTopology::create_fully_replicated_tensor_topology(distributed::MeshShape(1, 2));
    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 2));
    auto data_0 = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto data_1 = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    for (auto& v : data_1) {
        v += 10.0f;
    }
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(data_0)); });
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 1), [&]() { return HostBuffer(std::vector<float>(data_1)); });
    auto source = HostTensor::from_buffer(std::move(distributed_buffer), source_spec, topology);

    auto result = to_layout(source, Layout::ROW_MAJOR);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_TRUE(result.memory_config().shard_spec().has_value());
    EXPECT_EQ(result.memory_config().shard_spec(), memory_config.shard_spec());
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.memory_config()));
    EXPECT_EQ(result.tensor_topology(), topology);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto coords = result.buffer().shard_coords();
    EXPECT_EQ(coords.size(), 2);
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 0)));
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 1)));
}

TEST(HostTensorSpecPreservation, ToLayoutPhysicalMismatchThrows) {
    const Shape shape{32, 24};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Width alignment is not a multiple of tile width 16 → TILE initialize_alignment rounds up.
    auto alignment = Alignment({32, 24});
    auto tile = Tile({16, 16});

    auto source_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = to_tile_layout(source, tile));
}

TEST(HostTensorSpecPreservation, ToTileLayoutTileMismatchThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({32, 32})), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = to_tile_layout(source, Tile({16, 16})));
}

TEST(HostTensorSpecPreservation, ToTileLayoutBfpTileMismatchThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({32, 32})), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = to_tile_layout(source, Tile({16, 16})));
}

TEST(HostTensorSpecPreservation, PadUnpadInterleavedPreservesMemoryAndGeometry) {
    const Shape logical{2, 3};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(logical.volume());
    // pad/unpad reject sharded hosts, so per_core cannot be set true here; exact_spec_match still covers the flag.
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto source_spec =
        TensorSpec(logical, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    const Shape padded{4, 6};
    auto padded_tensor = pad(source, padded, Shape{0, 0}, 0.0f);

    auto expected_pad_spec = TensorSpec(
        logical,
        TensorLayout::fromPaddedShape(
            DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, logical, padded));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(padded_tensor.tensor_spec(), expected_pad_spec));
    EXPECT_EQ(padded_tensor.memory_config().buffer_type(), BufferType::DRAM);
    EXPECT_EQ(padded_tensor.padded_shape(), padded);
    EXPECT_EQ(padded_tensor.logical_shape(), logical);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(padded_tensor);

    auto unpadded = unpad(padded_tensor, Shape{0, 0}, Shape{2, 3});
    auto expected_unpad_spec = TensorSpec(
        logical,
        TensorLayout::fromPaddedShape(
            DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, logical, logical));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(unpadded.tensor_spec(), expected_unpad_spec));
    EXPECT_EQ(unpadded.memory_config().buffer_type(), BufferType::DRAM);
    EXPECT_EQ(unpadded.logical_shape(), logical);
    EXPECT_EQ(unpadded.padded_shape(), logical);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(unpadded);

    auto values = unpadded.to_vector<float>();
    ASSERT_EQ(values.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(values[i], data[i]);
    }
}

TEST(HostTensorSpecPreservation, PadShardedLegacyThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 32}, ShardOrientation::ROW_MAJOR}};
    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = pad(source, Shape{64, 32}, Shape{0, 0}, 0.0f));
}

TEST(HostTensorSpecPreservation, UnpadShardedLegacyThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 32}, ShardOrientation::ROW_MAJOR}};
    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = unpad(source, Shape{0, 0}, Shape{16, 32}));
}

TEST(HostTensorSpecPreservation, PadShardedConvertibleNdThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    // Convertible ND: 2D height-sharded via NdShardSpec (auto-fills legacy shard_spec).
    NdShardSpec nd_shard_spec{Shape{32, 32}, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    ASSERT_TRUE(source_spec.memory_config().is_sharded());
    ASSERT_TRUE(source_spec.memory_config().shard_spec().has_value());
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = pad(source, Shape{64, 32}, Shape{0, 0}, 0.0f));
}

TEST(HostTensorSpecPreservation, UnpadShardedConvertibleNdThrows) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    NdShardSpec nd_shard_spec{Shape{32, 32}, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    EXPECT_ANY_THROW(std::ignore = unpad(source, Shape{0, 0}, Shape{16, 32}));
}

}  // namespace
}  // namespace tt::tt_metal
