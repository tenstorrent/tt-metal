// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/float8.hpp>
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

void expect_packed_sizes(const HostTensor& tensor) {
    const size_t expected = tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& coord : tensor.buffer().shard_coords()) {
        auto shard = tensor.buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value());
        EXPECT_EQ(shard->view_bytes().size(), expected);
    }
}

TensorSpec make_rm_spec(const Shape& shape, DataType dtype, const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config));
}

TensorSpec make_tile_spec(
    const Shape& shape,
    DataType dtype,
    const Tile& tile = Tile({32, 32}),
    const MemoryConfig& memory_config = MemoryConfig{},
    const Alignment& alignment = {}) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE, tile), memory_config, alignment));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using ::testing::Eq;
using ::testing::Pointwise;

TEST(HostTensorToTensorSpec, EarlyOutExactSpecMatch) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = Alignment({32, 32});
    auto tile = Tile({16, 16});
    auto spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, spec);

    auto result = to_tensor_spec<float>(source, spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    EXPECT_THAT(result.to_vector<float>(), Pointwise(Eq(), data));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);
}

TEST(HostTensorToTensorSpec, PerCoreOnlyMismatchFullRewrite) {
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());

    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 32}, ShardOrientation::ROW_MAJOR}};

    auto src_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, Alignment({32, 32})));
    auto source = HostTensor::from_vector<float>(data, src_spec, /*pad_value=*/99.f);
    EXPECT_FALSE(experimental::per_core_allocation::is_per_core_allocation(source.tensor_spec().memory_config()));

    auto dest_memory = memory_config;
    experimental::per_core_allocation::set_per_core_allocation(dest_memory, true);
    auto dest_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), dest_memory, Alignment({32, 32})));

    // Spec equality ignores per_core, but exact-spec predicate must not early-out.
    EXPECT_TRUE(source.tensor_spec() == dest_spec);
    EXPECT_FALSE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(source.tensor_spec(), dest_spec));

    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/77.f);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.tensor_spec().memory_config()));
    EXPECT_THAT(result.to_vector<float>(), Pointwise(Eq(), data));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto result_physical = host_buffer::get_as<float>(result);
    // RM layout, shape 20x20, physical 32x32. Index (31, 31) is 31 * 32 + 31.
    EXPECT_EQ(result_physical[31 * 32 + 31], 77.f);
}

TEST(HostTensorToTensorSpec, LogicalRoundTripRmTileRm) {
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tile = Tile({16, 16});
    auto rm_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::FLOAT32, memory_config);
    auto tile_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::FLOAT32, tile, memory_config);

    auto source = HostTensor::from_vector<float>(data, rm_spec);
    auto tiled = to_tensor_spec<float>(source, tile_spec, /*pad_value=*/0.f);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(tiled.tensor_spec(), tile_spec));
    EXPECT_EQ(tiled.layout(), Layout::TILE);
    EXPECT_EQ(tiled.tensor_spec().tile(), tile);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(tiled);

    auto back = to_tensor_spec<float>(tiled, rm_spec);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_EQ(back.layout(), Layout::ROW_MAJOR);
    EXPECT_THAT(back.to_vector<float>(), Pointwise(Eq(), data));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(back);
}

TEST(HostTensorToTensorSpec, RmToTileAndTileToRm) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<uint16_t>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto rm_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::UINT16, memory_config);
    auto tile_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::UINT16, Tile({32, 32}), memory_config);

    auto source = HostTensor::from_vector<uint16_t>(data, rm_spec);
    auto tiled = to_tensor_spec<uint16_t>(source, tile_spec);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(tiled.tensor_spec(), tile_spec));
    EXPECT_EQ(tiled.layout(), Layout::TILE);
    EXPECT_EQ(tiled.dtype(), DataType::UINT16);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(tiled);

    auto back = to_tensor_spec<uint16_t>(tiled, rm_spec);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_THAT(back.to_vector<uint16_t>(), Pointwise(Eq(), data));
}

TEST(HostTensorToTensorSpec, CustomTileAndAlignment) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto tile = Tile({16, 16});
    auto alignment = Alignment({16, 16});
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    auto src_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto source = HostTensor::from_vector<float>(data, src_spec);
    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    EXPECT_EQ(result.tensor_spec().tensor_layout().get_alignment(), dest_spec.tensor_layout().get_alignment());
    EXPECT_THAT(result.to_vector<float>(), Pointwise(Eq(), data));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);
}

TEST(HostTensorToTensorSpec, Float32ToBfloat16PreservesMetadata) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = Alignment({32, 32});
    auto tile = Tile({16, 16});

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto result_data = result.to_vector<bfloat16>();
    ASSERT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(data[i])));
    }
}

TEST(HostTensorToTensorSpec, Float32ToBfloat16RowMajorShardedValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto src_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto dest_spec = TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.tensor_spec().memory_config()));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto result_data = result.to_vector<bfloat16>();
    ASSERT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(data[i])));
    }
}

TEST(HostTensorToTensorSpec, Float32TileToBfp8ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp8_dataset(shape, tile);

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/0.f);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);
}

TEST(HostTensorToTensorSpec, Float32TileToBfp4ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp4_dataset(shape, tile);

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/0.f);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT4_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);
}

TEST(HostTensorToTensorSpec, Bfp8TileToFloat32RowMajorValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp8_dataset(shape, tile);

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    // unpacked_golden is tile-major; convert to RM logical for comparison.
    auto rm_golden =
        tensor_impl::to_row_major_layout(src_spec.physical_shape(), tile, ttsl::make_const_span(unpacked_golden));
    EXPECT_THAT(result.to_vector<float>(), Pointwise(Eq(), rm_golden));
}

TEST(HostTensorToTensorSpec, Bfp4TileToFloat32RowMajorValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp4_dataset(shape, tile);

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto rm_golden =
        tensor_impl::to_row_major_layout(src_spec.physical_shape(), tile, ttsl::make_const_span(unpacked_golden));
    EXPECT_THAT(result.to_vector<float>(), Pointwise(Eq(), rm_golden));
}

TEST(HostTensorToTensorSpec, RowMajorToBfp8ChangesLayoutAndPreservesTile) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = Alignment({32, 32});
    auto tile = Tile({16, 16});

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, alignment));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, src_spec);

    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/0.f);

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(result);

    auto result_packed_data = host_buffer::get_as<uint32_t>(result);
    auto unpacked_data =
        unpack_bfp8_tiles_into_float_vec(result_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto rm_data =
        tensor_impl::to_row_major_layout(dest_spec.physical_shape(), tile, ttsl::make_const_span(unpacked_data));

    ASSERT_EQ(rm_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(rm_data[i], data[i], 1.0f);
    }
}

TEST(HostTensorToTensorSpec, BfpStagingRequiresFloatPad) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto src_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::FLOAT32, Tile({16, 16}));
    auto dest_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::BFLOAT8_B, Tile({16, 16}));

    auto source = HostTensor::from_vector<float>(data, src_spec);
    EXPECT_ANY_THROW(to_tensor_spec<bfloat16>(source, dest_spec, bfloat16(0.f)));
    EXPECT_NO_THROW(to_tensor_spec<float>(source, dest_spec, 0.f));
}

TEST(HostTensorToTensorSpec, Fp8SrcOrDestFatal) {
    const Shape shape{16, 16};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto fp8_spec = TensorSpec(shape, TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto f32_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::FLOAT32, memory_config);

    auto f32_data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto f32_tensor = HostTensor::from_vector<float>(f32_data, f32_spec);
    EXPECT_ANY_THROW(to_tensor_spec<float>(f32_tensor, fp8_spec));

    std::vector<float8_e4m3> fp8_data(shape.volume());
    for (size_t i = 0; i < fp8_data.size(); ++i) {
        fp8_data[i] = float8_e4m3(static_cast<float>(i % 16));
    }
    auto fp8_tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::move(fp8_data), fp8_spec);
    EXPECT_ANY_THROW(to_tensor_spec<float>(fp8_tensor, f32_spec));
}

TEST(HostTensorToTensorSpec, ExactMatchFp8Fatal) {
    const Shape shape{16, 16};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto fp8_spec = TensorSpec(shape, TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), memory_config));

    std::vector<float8_e4m3> fp8_data(shape.volume());
    for (size_t i = 0; i < fp8_data.size(); ++i) {
        fp8_data[i] = float8_e4m3(static_cast<float>(i % 16));
    }
    auto fp8_tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::move(fp8_data), fp8_spec);

    // Exact match FP8 should still fatal.
    EXPECT_ANY_THROW(to_tensor_spec<float>(fp8_tensor, fp8_spec));
}

TEST(HostTensorToTensorSpec, ShapeMismatchFatal) {
    auto src = HostTensor::from_vector<float>(
        CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(32 * 32),
        CMAKE_UNIQUE_NAMESPACE::make_rm_spec(Shape{32, 32}, DataType::FLOAT32));
    auto dest = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(Shape{16, 16}, DataType::FLOAT32);
    EXPECT_ANY_THROW(to_tensor_spec<float>(src, dest));
}

TEST(HostTensorToTensorSpec, TypedPadUint8AndWrongTFatal) {
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<uint8_t>(shape.volume());
    auto src_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::UINT8);
    auto dest_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::UINT8, Tile({16, 16}));

    auto source = HostTensor::from_vector<uint8_t>(data, src_spec);
    auto result = to_tensor_spec<uint8_t>(source, dest_spec, /*pad_value=*/uint8_t{7});
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_THAT(result.to_vector<uint8_t>(), Pointwise(Eq(), data));
    EXPECT_EQ(host_buffer::get_as<uint8_t>(result).back(), 7);

    // Wrong T vs working encode dtype (source UINT8).
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, 0.f));
}

TEST(HostTensorToTensorSpec, FloatPadToIntegralDestRejectsNanInfOor) {
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto src_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::FLOAT32);
    auto dest_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::UINT8, Tile({16, 16}));
    auto source = HostTensor::from_vector<float>(data, src_spec);

    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, std::numeric_limits<float>::quiet_NaN()));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, std::numeric_limits<float>::infinity()));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, -1.f));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, 256.f));

    auto ok = to_tensor_spec<float>(source, dest_spec, 7.f);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(ok.tensor_spec(), dest_spec));
    EXPECT_EQ(ok.dtype(), DataType::UINT8);
    EXPECT_EQ(ok.layout(), Layout::TILE);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(ok);

    auto logical = ok.to_vector<uint8_t>();
    ASSERT_EQ(logical.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(logical[i], static_cast<uint8_t>(data[i]));
    }

    // Physical shape is 32x32 TILE; trailing pad element should be 7.
    EXPECT_EQ(host_buffer::get_as<uint8_t>(ok).back(), 7);
}

TEST(HostTensorToTensorSpec, EqualPaddedDifferentPackingRoundTrip) {
    // Same logical shape and equal physical 2D size, but different packing (padded RM vs TILE).
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto rm_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, Alignment({16, 16})));
    auto tile_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    EXPECT_EQ(rm_spec.physical_shape(), tile_spec.physical_shape());
    EXPECT_NE(rm_spec.page_config(), tile_spec.page_config());
    EXPECT_TRUE(rm_spec.logical_2d_shape() != rm_spec.physical_shape());
    EXPECT_TRUE(tile_spec.logical_2d_shape() != tile_spec.physical_shape());

    auto source = HostTensor::from_vector<float>(data, rm_spec, /*pad_value=*/3.f);
    auto tiled = to_tensor_spec<float>(source, tile_spec, /*pad_value=*/5.f);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(tiled.tensor_spec(), tile_spec));
    EXPECT_THAT(tiled.to_vector<float>(), Pointwise(Eq(), data));
    EXPECT_EQ(host_buffer::get_as<float>(tiled).back(), 5.f);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(tiled);

    auto back = to_tensor_spec<float>(tiled, rm_spec, /*pad_value=*/9.f);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_THAT(back.to_vector<float>(), Pointwise(Eq(), data));
    // RM layout, shape 20x20, physical 32x32. Index (31, 31) is 31 * 32 + 31.
    EXPECT_EQ(host_buffer::get_as<float>(back)[31 * 32 + 31], 9.f);
    CMAKE_UNIQUE_NAMESPACE::expect_packed_sizes(back);
}

TEST(HostTensorToTensorSpec, TypedIntegralPadBoundaries) {
    const Shape shape{20, 20};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<uint32_t>(shape.volume());
    auto src_spec = CMAKE_UNIQUE_NAMESPACE::make_rm_spec(shape, DataType::UINT32);
    auto dest_spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::UINT32, Tile({16, 16}));
    auto source = HostTensor::from_vector<uint32_t>(data, src_spec);

    auto min_pad = std::numeric_limits<uint32_t>::lowest();
    auto result_min = to_tensor_spec<uint32_t>(source, dest_spec, min_pad);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result_min.tensor_spec(), dest_spec));
    EXPECT_EQ(host_buffer::get_as<uint32_t>(result_min).back(), min_pad);
    EXPECT_THAT(result_min.to_vector<uint32_t>(), Pointwise(Eq(), data));

    auto max_pad = std::numeric_limits<uint32_t>::max();
    auto result_max = to_tensor_spec<uint32_t>(source, dest_spec, max_pad);
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result_max.tensor_spec(), dest_spec));
    EXPECT_EQ(host_buffer::get_as<uint32_t>(result_max).back(), max_pad);
    EXPECT_THAT(result_max.to_vector<uint32_t>(), Pointwise(Eq(), data));
}

TEST(HostTensorToTensorSpec, Rank0Unsupported) {
    Shape rank0_shape(ttsl::SmallVector<uint32_t>{});
    EXPECT_EQ(rank0_shape.rank(), 0);

    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::vector<float>{}, spec);

    // Dest with different packing metadata forces a rewrite attempt (not early-out).
    auto dest = TensorSpec(
        rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({32, 32})), memory_config));
    EXPECT_ANY_THROW(to_tensor_spec<float>(tensor, dest));
}

TEST(HostTensorToTensorSpec, ExactMatchRank0Fatal) {
    Shape rank0_shape(ttsl::SmallVector<uint32_t>{});
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::vector<float>{}, spec);

    // Exact match rank-0 should still fatal.
    EXPECT_ANY_THROW(to_tensor_spec<float>(tensor, spec));
}

TEST(HostTensorToTensorSpec, ExactMatchBfpWrongTFatal) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto spec = CMAKE_UNIQUE_NAMESPACE::make_tile_spec(shape, DataType::BFLOAT8_B, Tile({16, 16}));
    auto source = HostTensor::from_vector<float>(data, spec);

    // Exact match but wrong T (not float) should still fatal.
    EXPECT_ANY_THROW(to_tensor_spec<bfloat16>(source, spec, bfloat16(0.f)));
}

TEST(HostTensorToTensorSpec, OversizedBufferThrows) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto src_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    auto oversized_data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume() + 100);
    auto host_tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::move(oversized_data), src_spec);

    // decode_tensor_data asserts physical size matches the spec.
    EXPECT_ANY_THROW(to_tensor_spec<float>(host_tensor, dest_spec));
}

TEST(HostTensorToTensorSpec, UndersizedBfpBufferThrows) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tile = Tile({16, 16});
    auto src_spec = TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config));
    auto dest_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    auto undersized_data = std::vector<uint32_t>(10, 0);
    auto host_tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::move(undersized_data), src_spec);

    // Staging to_dtype asserts packed input size before unpack.
    EXPECT_ANY_THROW(to_tensor_spec<float>(host_tensor, dest_spec));
}

TEST(HostTensorToTensorSpec, MalformedBfpBufferThrows) {
    const Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tile = Tile({16, 16});
    auto src_spec = TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config));
    auto dest_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    size_t expected_size_bytes = src_spec.compute_packed_buffer_size_bytes();
    auto malformed_data = std::vector<uint32_t>((expected_size_bytes / sizeof(uint32_t)) - 1, 0);
    auto host_tensor = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(std::move(malformed_data), src_spec);

    EXPECT_ANY_THROW(to_tensor_spec<float>(host_tensor, dest_spec));
}

}  // namespace
}  // namespace tt::tt_metal
