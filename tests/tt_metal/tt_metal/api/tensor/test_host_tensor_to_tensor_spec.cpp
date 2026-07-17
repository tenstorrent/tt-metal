// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
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

TensorSpec make_rm_spec(const Shape& shape, DataType dtype, const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config));
}

TensorSpec make_tile_spec(
    const Shape& shape,
    DataType dtype,
    const Tile& tile = Tile({32, 32}),
    const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE, tile), memory_config));
}

TEST(HostTensorToTensorSpec, EarlyOutExactSpecMatch) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto spec = make_tile_spec(shape, DataType::FLOAT32);
    auto source = HostTensor::from_vector(data, spec);

    auto result = to_tensor_spec<float>(source, spec);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), spec));
    EXPECT_EQ(result.to_vector<float>(), data);
}

TEST(HostTensorToTensorSpec, PerCoreOnlyMismatchFullRewrite) {
    const Shape shape{20, 20};
    auto data = make_ramp<float>(shape.volume());

    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 32}, ShardOrientation::ROW_MAJOR}};

    auto src_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, Alignment({32, 32})));
    auto source = HostTensor::from_vector(data, src_spec, /*pad_value=*/99.f);
    EXPECT_FALSE(experimental::per_core_allocation::is_per_core_allocation(source.tensor_spec().memory_config()));

    auto dest_memory = memory_config;
    experimental::per_core_allocation::set_per_core_allocation(dest_memory, true);
    auto dest_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), dest_memory, Alignment({32, 32})));

    // Spec equality ignores per_core, but exact-spec predicate must not early-out.
    EXPECT_TRUE(source.tensor_spec() == dest_spec);
    EXPECT_FALSE(exact_spec_match(source.tensor_spec(), dest_spec));

    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/77.f);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.tensor_spec().memory_config()));
    EXPECT_EQ(result.to_vector<float>(), data);

    auto result_physical = host_buffer::get_as<float>(result);
    // RM layout, shape 20x20, physical 32x32.
    // Index (31, 31) is 31 * 32 + 31.
    EXPECT_EQ(result_physical[31 * 32 + 31], 77.f);
}

TEST(HostTensorToTensorSpec, LogicalRoundTripRmTileRm) {
    const Shape shape{20, 20};
    auto data = make_ramp<float>(shape.volume());
    auto rm_spec = make_rm_spec(shape, DataType::FLOAT32);
    auto tile_spec = make_tile_spec(shape, DataType::FLOAT32, Tile({16, 16}));

    auto source = HostTensor::from_vector(data, rm_spec);
    auto tiled = to_tensor_spec<float>(source, tile_spec, /*pad_value=*/0.f);
    EXPECT_TRUE(exact_spec_match(tiled.tensor_spec(), tile_spec));

    auto back = to_tensor_spec<float>(tiled, rm_spec);
    EXPECT_TRUE(exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_EQ(back.to_vector<float>(), data);
}

TEST(HostTensorToTensorSpec, RmToTileAndTileToRm) {
    const Shape shape{32, 32};
    auto data = make_ramp<uint16_t>(shape.volume());
    auto rm_spec = make_rm_spec(shape, DataType::UINT16);
    auto tile_spec = make_tile_spec(shape, DataType::UINT16);

    auto source = HostTensor::from_vector(data, rm_spec);
    auto tiled = to_tensor_spec<uint16_t>(source, tile_spec);
    EXPECT_TRUE(exact_spec_match(tiled.tensor_spec(), tile_spec));
    EXPECT_EQ(tiled.layout(), Layout::TILE);

    auto back = to_tensor_spec<uint16_t>(tiled, rm_spec);
    EXPECT_TRUE(exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_EQ(back.to_vector<uint16_t>(), data);
}

TEST(HostTensorToTensorSpec, CustomTileAndAlignment) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto tile = Tile({16, 16});
    auto alignment = Alignment({16, 16});
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    auto src_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto source = HostTensor::from_vector(data, src_spec);
    auto result = to_tensor_spec<float>(source, dest_spec);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.tensor_spec().tile(), tile);
    EXPECT_EQ(result.tensor_spec().tensor_layout().get_alignment(), dest_spec.tensor_layout().get_alignment());
    EXPECT_EQ(result.to_vector<float>(), data);
}

TEST(HostTensorToTensorSpec, CrossDtypeFloat32ToBfloat16) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto src_spec = make_tile_spec(shape, DataType::FLOAT32);
    auto dest_spec = make_tile_spec(shape, DataType::BFLOAT16);

    auto source = HostTensor::from_vector(data, src_spec);
    auto result = to_tensor_spec<float>(source, dest_spec);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));

    auto result_data = result.to_vector<bfloat16>();
    ASSERT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(data[i])));
    }
}

TEST(HostTensorToTensorSpec, Fp8SrcOrDestFatal) {
    const Shape shape{16, 16};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto fp8_spec = TensorSpec(shape, TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto f32_spec = make_rm_spec(shape, DataType::FLOAT32);

    auto f32_data = make_ramp<float>(shape.volume());
    auto f32_tensor = HostTensor::from_vector(f32_data, f32_spec);
    EXPECT_ANY_THROW(to_tensor_spec<float>(f32_tensor, fp8_spec));

    std::vector<float8_e4m3> fp8_data(shape.volume());
    for (size_t i = 0; i < fp8_data.size(); ++i) {
        fp8_data[i] = float8_e4m3(static_cast<float>(i % 16));
    }
    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float8_e4m3>(fp8_data)); });
    auto fp8_tensor = HostTensor::from_buffer(std::move(dist_buffer), fp8_spec, TensorTopology{});
    EXPECT_ANY_THROW(to_tensor_spec<float>(fp8_tensor, f32_spec));
}

TEST(HostTensorToTensorSpec, MultiShardTopologyPreservation) {
    const Shape shape{32, 64};
    const size_t volume = shape.volume();

    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    auto tile = Tile({16, 16});
    auto alignment = Alignment({32, 64});

    auto src_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto dest_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    auto topology = TensorTopology::create_fully_replicated_tensor_topology(distributed::MeshShape(1, 2));
    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 2));

    auto data_0 = make_ramp<float>(volume);
    auto data_1 = make_ramp<float>(volume);
    for (auto& v : data_1) {
        v += 10.0f;
    }

    // Encode each shard into the TILE FLOAT32 physical layout expected by from_buffer.
    auto encoded_0 = tensor_impl::encode_tensor_data(ttsl::make_const_span(data_0), src_spec, 0.f);
    auto encoded_1 = tensor_impl::encode_tensor_data(ttsl::make_const_span(data_1), src_spec, 0.f);
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(encoded_0)); });
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 1), [&]() { return HostBuffer(std::vector<float>(encoded_1)); });

    auto source = HostTensor::from_buffer(std::move(distributed_buffer), src_spec, topology);
    auto result = to_tensor_spec<float>(source, dest_spec);

    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.tensor_topology(), topology);

    const size_t expected_shard_size = dest_spec.compute_packed_buffer_size_bytes();
    auto coords = result.buffer().shard_coords();
    EXPECT_EQ(coords.size(), 2);
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 0)));
    EXPECT_TRUE(coords.contains(distributed::MeshCoordinate(0, 1)));

    for (const auto& coord : coords) {
        auto shard = result.buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value());
        EXPECT_EQ(shard->view_bytes().size(), expected_shard_size);

        auto logical = tensor_impl::decode_tensor_data(shard->view_as<const bfloat16>(), dest_spec);
        const auto& expected_data = (coord == distributed::MeshCoordinate(0, 0)) ? data_0 : data_1;
        ASSERT_EQ(logical.size(), expected_data.size());
        for (size_t i = 0; i < logical.size(); ++i) {
            EXPECT_EQ(static_cast<float>(logical[i]), static_cast<float>(bfloat16(expected_data[i])));
        }
    }
}

TEST(HostTensorToTensorSpec, ShapeMismatchFatal) {
    auto src = HostTensor::from_vector(make_ramp<float>(32 * 32), make_rm_spec(Shape{32, 32}, DataType::FLOAT32));
    auto dest = make_rm_spec(Shape{16, 16}, DataType::FLOAT32);
    EXPECT_ANY_THROW(to_tensor_spec<float>(src, dest));
}

TEST(HostTensorToTensorSpec, TypedPadUint8AndWrongTFatal) {
    const Shape shape{20, 20};
    auto data = make_ramp<uint8_t>(shape.volume());
    auto src_spec = make_rm_spec(shape, DataType::UINT8);
    auto dest_spec = make_tile_spec(shape, DataType::UINT8, Tile({16, 16}));

    auto source = HostTensor::from_vector(data, src_spec);
    auto result = to_tensor_spec<uint8_t>(source, dest_spec, /*pad_value=*/uint8_t{7});
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.to_vector<uint8_t>(), data);

    // Wrong T vs working encode dtype (source UINT8).
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, 0.f));
}

TEST(HostTensorToTensorSpec, BfpStagingRequiresFloatPad) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto src_spec = make_tile_spec(shape, DataType::FLOAT32, Tile({16, 16}));
    auto dest_spec = make_tile_spec(shape, DataType::BFLOAT8_B, Tile({16, 16}));

    auto source = HostTensor::from_vector(data, src_spec);
    EXPECT_ANY_THROW(to_tensor_spec<bfloat16>(source, dest_spec, bfloat16(0.f)));
    EXPECT_NO_THROW(to_tensor_spec<float>(source, dest_spec, 0.f));
}

TEST(HostTensorToTensorSpec, BfpDestinationStaging) {
    const Shape shape{20, 20};
    auto data = make_ramp<float>(shape.volume());
    auto tile = Tile({16, 16});
    auto src_spec = make_rm_spec(shape, DataType::FLOAT32);
    auto dest_spec = make_tile_spec(shape, DataType::BFLOAT8_B, tile);

    auto source = HostTensor::from_vector(data, src_spec);
    auto result = to_tensor_spec<float>(source, dest_spec, /*pad_value=*/0.f);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);

    // Round-trip logical values through FLOAT32 (BFP quantization tolerant).
    auto as_float = to_dtype(result, DataType::FLOAT32);
    auto logical = as_float.to_vector<float>();
    ASSERT_EQ(logical.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(logical[i], data[i], 1.0f);
    }
}

TEST(HostTensorToTensorSpec, BfpSourceStaging) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto tile = Tile({16, 16});
    auto bfp_spec = make_tile_spec(shape, DataType::BFLOAT8_B, tile);
    auto dest_spec = make_rm_spec(shape, DataType::FLOAT32);

    auto source = HostTensor::from_vector(data, bfp_spec);
    auto result = to_tensor_spec<float>(source, dest_spec);
    EXPECT_TRUE(exact_spec_match(result.tensor_spec(), dest_spec));

    auto logical = result.to_vector<float>();
    ASSERT_EQ(logical.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(logical[i], data[i], 1.0f);
    }
}

TEST(HostTensorToTensorSpec, FloatPadToIntegralDestRejectsNanInfOor) {
    const Shape shape{20, 20};
    auto data = make_ramp<float>(shape.volume());
    auto src_spec = make_rm_spec(shape, DataType::FLOAT32);
    auto dest_spec = make_tile_spec(shape, DataType::UINT8, Tile({16, 16}));
    auto source = HostTensor::from_vector(data, src_spec);

    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, std::numeric_limits<float>::quiet_NaN()));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, std::numeric_limits<float>::infinity()));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, -1.f));
    EXPECT_ANY_THROW(to_tensor_spec<float>(source, dest_spec, 256.f));

    auto ok = to_tensor_spec<float>(source, dest_spec, 7.f);
    EXPECT_TRUE(exact_spec_match(ok.tensor_spec(), dest_spec));
    auto logical = ok.to_vector<uint8_t>();
    ASSERT_EQ(logical.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(logical[i], static_cast<uint8_t>(data[i]));
    }

    // Verify physical padding
    auto physical = host_buffer::get_as<uint8_t>(ok);
    // Physical shape is 32x32.
    // Index (31, 31) in tile layout is at the end.
    // The padding should be filled with 7.
    EXPECT_EQ(physical.back(), 7);
}

TEST(HostTensorToTensorSpec, EqualPaddedDifferentPackingRoundTrip) {
    // Same logical shape and equal physical 2D size, but different packing (padded RM vs TILE).
    const Shape shape{20, 20};
    auto data = make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto rm_spec = TensorSpec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, Alignment({16, 16})));
    auto tile_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    EXPECT_EQ(rm_spec.physical_shape(), tile_spec.physical_shape());
    EXPECT_NE(rm_spec.page_config(), tile_spec.page_config());
    EXPECT_TRUE(rm_spec.logical_2d_shape() != rm_spec.physical_shape());
    EXPECT_TRUE(tile_spec.logical_2d_shape() != tile_spec.physical_shape());

    auto source = HostTensor::from_vector(data, rm_spec, /*pad_value=*/3.f);
    auto tiled = to_tensor_spec<float>(source, tile_spec, /*pad_value=*/5.f);
    EXPECT_TRUE(exact_spec_match(tiled.tensor_spec(), tile_spec));
    EXPECT_EQ(tiled.to_vector<float>(), data);

    // Check physical pad of tiled
    auto tiled_physical = host_buffer::get_as<float>(tiled);
    EXPECT_EQ(tiled_physical.back(), 5.f);

    auto back = to_tensor_spec<float>(tiled, rm_spec, /*pad_value=*/9.f);
    EXPECT_TRUE(exact_spec_match(back.tensor_spec(), rm_spec));
    EXPECT_EQ(back.to_vector<float>(), data);

    // Check physical pad of back
    auto back_physical = host_buffer::get_as<float>(back);
    // RM layout, shape 20x20, physical 32x32.
    // Index (31, 31) is 31 * 32 + 31.
    EXPECT_EQ(back_physical[31 * 32 + 31], 9.f);
}

TEST(HostTensorToTensorSpec, Rank0Unsupported) {
    // Rank-0 shape: empty dimensions vector.
    Shape rank0_shape(ttsl::SmallVector<uint32_t>{});
    EXPECT_EQ(rank0_shape.rank(), 0);

    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>{}); });
    auto tensor = HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});

    // Dest with different packing metadata forces a rewrite attempt (not early-out).
    auto dest = TensorSpec(
        rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, Tile({32, 32})), memory_config));
    EXPECT_ANY_THROW(to_tensor_spec<float>(tensor, dest));
}

TEST(HostTensorToTensorSpec, ExactMatchFp8Fatal) {
    const Shape shape{16, 16};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto fp8_spec = TensorSpec(shape, TensorLayout(DataType::FP8_E4M3, PageConfig(Layout::ROW_MAJOR), memory_config));

    std::vector<float8_e4m3> fp8_data(shape.volume());
    for (size_t i = 0; i < fp8_data.size(); ++i) {
        fp8_data[i] = float8_e4m3(static_cast<float>(i % 16));
    }
    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float8_e4m3>(fp8_data)); });
    auto fp8_tensor = HostTensor::from_buffer(std::move(dist_buffer), fp8_spec, TensorTopology{});

    // Exact match FP8 should still fatal
    EXPECT_ANY_THROW(to_tensor_spec<float>(fp8_tensor, fp8_spec));
}

TEST(HostTensorToTensorSpec, ExactMatchRank0Fatal) {
    Shape rank0_shape(ttsl::SmallVector<uint32_t>{});
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(rank0_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    auto dist_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    dist_buffer.emplace_shard(distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>{}); });
    auto tensor = HostTensor::from_buffer(std::move(dist_buffer), spec, TensorTopology{});

    // Exact match rank-0 should still fatal
    EXPECT_ANY_THROW(to_tensor_spec<float>(tensor, spec));
}

TEST(HostTensorToTensorSpec, ExactMatchBfpWrongTFatal) {
    const Shape shape{32, 32};
    auto data = make_ramp<float>(shape.volume());
    auto spec = make_tile_spec(shape, DataType::BFLOAT8_B, Tile({16, 16}));

    auto source = HostTensor::from_vector(data, spec);

    // Exact match but wrong T (not float) should still fatal
    EXPECT_ANY_THROW(to_tensor_spec<bfloat16>(source, spec, bfloat16(0.f)));
}

TEST(HostTensorToTensorSpec, TypedIntegralPadBoundaries) {
    const Shape shape{20, 20};
    auto data = make_ramp<uint32_t>(shape.volume());
    auto src_spec = make_rm_spec(shape, DataType::UINT32);
    auto dest_spec = make_tile_spec(shape, DataType::UINT32, Tile({16, 16}));

    auto source = HostTensor::from_vector(data, src_spec);

    // Min boundary
    auto min_pad = std::numeric_limits<uint32_t>::lowest();
    auto result_min = to_tensor_spec<uint32_t>(source, dest_spec, min_pad);
    EXPECT_TRUE(exact_spec_match(result_min.tensor_spec(), dest_spec));
    auto physical_min = host_buffer::get_as<uint32_t>(result_min);
    EXPECT_EQ(physical_min.back(), min_pad);

    // Max boundary
    auto max_pad = std::numeric_limits<uint32_t>::max();
    auto result_max = to_tensor_spec<uint32_t>(source, dest_spec, max_pad);
    EXPECT_TRUE(exact_spec_match(result_max.tensor_spec(), dest_spec));
    auto physical_max = host_buffer::get_as<uint32_t>(result_max);
    EXPECT_EQ(physical_max.back(), max_pad);
}

}  // namespace
}  // namespace tt::tt_metal
