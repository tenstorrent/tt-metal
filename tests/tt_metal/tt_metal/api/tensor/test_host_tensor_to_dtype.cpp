// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <cstdint>
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

std::vector<float> make_bfp_data(const Shape& shape, const Tile& tile) {
    // Boundary-sensitive to faces: cycles face indices and steps magnitudes for BFP checks.
    const size_t count = shape.volume();
    const size_t face_width = tile.get_face_shape()[1];
    std::vector<float> data(count);
    size_t i = 0;
    for (size_t face_row = 0; i < count; ++face_row) {
        for (size_t face_index = 0; face_index < face_width && i < count; ++face_index) {
            data[i++] = static_cast<float>(face_index) + static_cast<float>(face_row) * 0.5f;
        }
    }
    return data;
}

using PackedBfp = std::vector<uint32_t>;
using UnpackedBfp = std::vector<float>;

// BFP -> float golden: packed bytes plus their reference unpack (not the pre-quant floats).
std::pair<PackedBfp, UnpackedBfp> generate_bfp8_dataset(const Shape& shape, const Tile& tile) {
    const auto src = make_bfp_data(shape, tile);
    auto packed = pack_as_bfp8_tiles(ttsl::make_const_span(src), /*row_major_input=*/true, /*is_exp_a=*/false, tile);
    auto unpacked = unpack_bfp8_tiles_into_float_vec(packed, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    return {std::move(packed), std::move(unpacked)};
}

std::pair<PackedBfp, UnpackedBfp> generate_bfp4_dataset(const Shape& shape, const Tile& tile) {
    const auto src = make_bfp_data(shape, tile);
    auto packed = pack_as_bfp4_tiles(ttsl::make_const_span(src), /*row_major_input=*/true, /*is_exp_a=*/false, tile);
    auto unpacked = unpack_bfp4_tiles_into_float_vec(packed, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    return {std::move(packed), std::move(unpacked)};
}

// Float(TILE) -> BFP golden: tile-layout floats plus their reference pack.
std::pair<UnpackedBfp, PackedBfp> generate_float_to_bfp8_dataset(const Shape& shape, const Tile& tile) {
    auto floats = make_bfp_data(shape, tile);
    auto packed =
        pack_as_bfp8_tiles(ttsl::make_const_span(floats), /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    return {std::move(floats), std::move(packed)};
}

std::pair<UnpackedBfp, PackedBfp> generate_float_to_bfp4_dataset(const Shape& shape, const Tile& tile) {
    auto floats = make_bfp_data(shape, tile);
    auto packed =
        pack_as_bfp4_tiles(ttsl::make_const_span(floats), /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    return {std::move(floats), std::move(packed)};
}

template <typename T>
HostTensor make_host_tensor(std::vector<T> data, const TensorSpec& spec) {
    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [data = std::move(data)]() mutable { return HostBuffer(std::move(data)); });
    return HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});
}

bool exact_spec_match(const TensorSpec& a, const TensorSpec& b) {
    return a == b && experimental::per_core_allocation::is_per_core_allocation(a.memory_config()) ==
                         experimental::per_core_allocation::is_per_core_allocation(b.memory_config());
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using ::testing::Eq;
using ::testing::Pointwise;

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
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp8_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<float>(result), Pointwise(Eq(), unpacked_golden));
}

TEST(HostTensorToDtype, Float32ToTileBfp8ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp8_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT8_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
}

TEST(HostTensorToDtype, TileBfp4ToFloat32ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp4_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<float>(result), Pointwise(Eq(), unpacked_golden));
}

TEST(HostTensorToDtype, Float32ToTileBfp4ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp4_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT4_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT4_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
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
