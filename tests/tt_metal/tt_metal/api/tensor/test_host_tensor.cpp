// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sanity tests for HostTensor class.
//
// These are minimal tests to verify HostTensor's basic type properties and
// functionality including:
// - Type traits (copyable, movable, etc.)
// - Construction with DistributedHostBuffer, TensorSpec, and TensorTopology
// - Copy and move semantics
// - Getter methods for tensor properties

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal {
namespace {

TensorSpec create_simple_spec(const Shape& shape, DataType dtype = DataType::BFLOAT16) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(dtype, page_config, memory_config);
    return TensorSpec(shape, tensor_layout);
}

HostTensor create_simple_host_tensor(const Shape& shape, DataType dtype = DataType::BFLOAT16) {
    auto spec = create_simple_spec(shape, dtype);
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape{1});
    auto topology = TensorTopology();
    return HostTensor(std::move(buffer), std::move(spec), std::move(topology));
}

// Type trait tests verifying HostTensor's semantic constraints

TEST(HostTensorTypeTraitsTest, IsDefaultConstructible) { EXPECT_FALSE(std::is_default_constructible_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsDestructible) { EXPECT_TRUE(std::is_destructible_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsCopyConstructible) { EXPECT_TRUE(std::is_copy_constructible_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsCopyAssignable) { EXPECT_TRUE(std::is_copy_assignable_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsMoveConstructible) { EXPECT_TRUE(std::is_move_constructible_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsMoveAssignable) { EXPECT_TRUE(std::is_move_assignable_v<HostTensor>); }

TEST(HostTensorTypeTraitsTest, IsNothrowMoveConstructible) {
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<HostTensor>);
}

TEST(HostTensorTypeTraitsTest, IsNothrowMoveAssignable) { EXPECT_TRUE(std::is_nothrow_move_assignable_v<HostTensor>); }

// Runtime tests

TEST(HostTensorTest, ConstructionWithSpec) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);

    EXPECT_EQ(tensor.logical_shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
}

TEST(HostTensorTest, ConstructionWithHostBuffer) {
    Shape shape{2, 32};
    auto spec = create_simple_spec(shape, DataType::FLOAT32);
    auto topology = TensorTopology();

    std::vector<float> data(shape.volume(), 1.0f);
    HostBuffer host_buffer(std::move(data));

    HostTensor tensor(std::move(host_buffer), std::move(spec), std::move(topology));

    EXPECT_EQ(tensor.logical_shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);

    const auto& buffer = tensor.buffer();
    EXPECT_TRUE(buffer.get_shard(distributed::MeshCoordinate(0, 0)).has_value());
}

TEST(HostTensorTest, LogicalVolume) {
    Shape shape{4, 8, 16};
    auto tensor = create_simple_host_tensor(shape);

    EXPECT_EQ(tensor.logical_volume(), 4 * 8 * 16);
}

TEST(HostTensorTest, ElementSizeBFloat16) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::BFLOAT16);
    EXPECT_EQ(tensor.element_size(), sizeof(bfloat16));
}

TEST(HostTensorTest, ElementSizeFloat32) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::FLOAT32);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
}

TEST(HostTensorTest, ElementSizeUInt32) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::UINT32);
    EXPECT_EQ(tensor.element_size(), sizeof(uint32_t));
}

TEST(HostTensorTest, ElementSizeInt32) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::INT32);
    EXPECT_EQ(tensor.element_size(), sizeof(int32_t));
}

TEST(HostTensorTest, ElementSizeUInt16) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::UINT16);
    EXPECT_EQ(tensor.element_size(), sizeof(uint16_t));
}

TEST(HostTensorTest, ElementSizeUInt8) {
    auto tensor = create_simple_host_tensor(Shape{1, 32}, DataType::UINT8);
    EXPECT_EQ(tensor.element_size(), sizeof(uint8_t));
}

TEST(HostTensorTest, MoveConstruction) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);
    HostTensor moved_tensor(std::move(tensor));

    EXPECT_EQ(moved_tensor.logical_shape(), shape);
    EXPECT_EQ(moved_tensor.dtype(), DataType::BFLOAT16);
}

TEST(HostTensorTest, MoveAssignment) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);
    auto other = create_simple_host_tensor(Shape{1, 32});
    other = std::move(tensor);

    EXPECT_EQ(other.logical_shape(), shape);
    EXPECT_EQ(other.dtype(), DataType::BFLOAT16);
}

TEST(HostTensorTest, CopyConstruction) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);
    const HostTensor copied_tensor(tensor);  // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(copied_tensor.logical_shape(), shape);
    EXPECT_EQ(copied_tensor.dtype(), DataType::BFLOAT16);
    // Original should still be valid
    EXPECT_EQ(tensor.logical_shape(), shape);
}

TEST(HostTensorTest, CopyAssignment) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);
    auto other = create_simple_host_tensor(Shape{1, 32});
    other = tensor;

    EXPECT_EQ(other.logical_shape(), shape);
    EXPECT_EQ(other.dtype(), DataType::BFLOAT16);
    // Original should still be valid
    EXPECT_EQ(tensor.logical_shape(), shape);
}

TEST(HostTensorTest, TensorSpecAccess) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape, DataType::FLOAT32);

    const auto& spec = tensor.tensor_spec();
    EXPECT_EQ(spec.logical_shape(), shape);
    EXPECT_EQ(spec.data_type(), DataType::FLOAT32);
    EXPECT_EQ(spec.layout(), Layout::ROW_MAJOR);
}

TEST(HostTensorTest, TensorTopologyAccess) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    const auto& topology = tensor.tensor_topology();
    // Default topology should have distribution shape of {1}
    EXPECT_EQ(topology.distribution_shape().dims(), 1);
}

TEST(HostTensorTest, MemoryConfigAccess) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    const auto& memory_config = tensor.memory_config();
    EXPECT_EQ(memory_config.memory_layout(), TensorMemoryLayout::INTERLEAVED);
    EXPECT_EQ(memory_config.buffer_type(), BufferType::DRAM);
}

TEST(HostTensorTest, IsSharded) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    // Default memory config is INTERLEAVED, not sharded
    EXPECT_FALSE(tensor.is_sharded());
}

TEST(HostTensorTest, Strides) {
    Shape shape{2, 4, 8};
    auto tensor = create_simple_host_tensor(shape);

    auto strides = tensor.strides();
    // For row-major layout, strides should be [4*8, 8, 1] = [32, 8, 1]
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 32);
    EXPECT_EQ(strides[1], 8);
    EXPECT_EQ(strides[2], 1);
}

TEST(HostTensorTest, BufferAccess) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    const auto& buffer = tensor.buffer();
    // Buffer should have shape {1} (single device)
    EXPECT_EQ(buffer.shape().dims(), 1);
}

TEST(HostTensorTest, PaddedShape) {
    Shape shape{2, 64};
    auto tensor = create_simple_host_tensor(shape);

    // For row-major layout without explicit padding, padded shape equals logical shape
    const auto& padded = tensor.padded_shape();
    EXPECT_EQ(padded, shape);
}

TEST(HostTensorTest, PhysicalVolume) {
    Shape shape{4, 8, 16};
    auto tensor = create_simple_host_tensor(shape);

    // For non-padded row-major layout, physical volume equals logical volume
    EXPECT_EQ(tensor.physical_volume(), tensor.logical_volume());
}

TEST(HostTensorTest, LegacyShardSpecNotSet) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    // INTERLEAVED memory config should not have shard spec
    EXPECT_FALSE(tensor.legacy_shard_spec().has_value());
}

TEST(HostTensorTest, NdShardSpecNotSet) {
    Shape shape{4, 32};
    auto tensor = create_simple_host_tensor(shape);

    // INTERLEAVED memory config should not have nd shard spec
    EXPECT_FALSE(tensor.nd_shard_spec().has_value());
}

TEST(HostTensorTest, IsValuelessAfterMoveReturnsFalse) {
    // A freshly constructed tensor is not valueless.
    auto tensor = create_simple_host_tensor(Shape{4, 32});
    EXPECT_FALSE(tensor.is_valueless_after_move());

    // Copy construction leaves neither source nor destination valueless.
    HostTensor copy(tensor);  // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT_FALSE(tensor.is_valueless_after_move());
    EXPECT_FALSE(copy.is_valueless_after_move());

    // Copy assignment leaves neither source nor destination valueless.
    auto target = create_simple_host_tensor(Shape{1, 8});
    target = tensor;
    EXPECT_FALSE(tensor.is_valueless_after_move());
    EXPECT_FALSE(target.is_valueless_after_move());
}

TEST(HostTensorTest, IsValuelessAfterMoveReturnsTrueAfterMoveConstruction) {
    auto tensor = create_simple_host_tensor(Shape{4, 32});
    HostTensor moved(std::move(tensor));

    EXPECT_TRUE(tensor.is_valueless_after_move());  // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(moved.is_valueless_after_move());
}

TEST(HostTensorTest, IsValuelessAfterMoveReturnsTrueAfterMoveAssignment) {
    auto source = create_simple_host_tensor(Shape{4, 32});
    auto target = create_simple_host_tensor(Shape{1, 8});

    target = std::move(source);

    EXPECT_TRUE(source.is_valueless_after_move());  // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(target.is_valueless_after_move());
}

// FP8_E4M3 host-side tensor operations.
//
// FP8 support in tt-metal tensor infra is intentionally limited (only what the DeepSeek V3
// Prefill combine op needs today). These tests pin that contract on the host side:
//   - to_layout: ROW_MAJOR is the only supported layout; any other target throws.
//   - to_dtype: only FP8 <-> FLOAT32 is wired up; other cross-type conversions throw.
//   - pad / unpad: not wired up, both throw.

HostTensor make_fp8_host_tensor(const Shape& shape, std::vector<float8_e4m3> data) {
    auto spec = create_simple_spec(shape, DataType::FP8_E4M3);
    HostBuffer host_buffer(std::move(data));
    return HostTensor(std::move(host_buffer), std::move(spec), TensorTopology());
}

TEST(HostTensorFp8Test, ToLayoutRowMajorIsNoOp) {
    Shape shape{2, 32};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    auto out = to_layout(tensor, Layout::ROW_MAJOR);

    EXPECT_EQ(out.dtype(), DataType::FP8_E4M3);
    EXPECT_EQ(out.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(out.logical_shape(), shape);
}

TEST(HostTensorFp8Test, ToLayoutToTileThrows) {
    Shape shape{2, 32};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    EXPECT_ANY_THROW(to_layout(tensor, Layout::TILE));
}

TEST(HostTensorFp8Test, ToDtypeFp8ToFp8IsNoOp) {
    Shape shape{2, 32};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    auto out = to_dtype(tensor, DataType::FP8_E4M3);

    EXPECT_EQ(out.dtype(), DataType::FP8_E4M3);
    EXPECT_EQ(out.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(out.logical_shape(), shape);
}

TEST(HostTensorFp8Test, ToDtypeFp8ToFloat32Succeeds) {
    Shape shape{1, 8};
    // All values below are exactly representable in FP8 E4M3 so the FP8 round-trip is lossless.
    std::vector<float> float_data{0.0f, 0.5f, 1.0f, 2.0f, -1.0f, -2.0f, 4.0f, -0.5f};
    std::vector<float8_e4m3> fp8_data(float_data.size());
    std::transform(float_data.begin(), float_data.end(), fp8_data.begin(), [](float v) { return float8_e4m3(v); });

    auto tensor = make_fp8_host_tensor(shape, std::move(fp8_data));
    auto out = to_dtype(tensor, DataType::FLOAT32);

    EXPECT_EQ(out.dtype(), DataType::FLOAT32);
    EXPECT_EQ(out.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(out.logical_shape(), shape);

    auto out_data = host_buffer::get_as<const float>(out);
    ASSERT_EQ(out_data.size(), float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
        EXPECT_FLOAT_EQ(out_data[i], float_data[i]);
    }
}

TEST(HostTensorFp8Test, ToDtypeFloat32ToFp8RoundTripsCleanly) {
    Shape shape{1, 8};
    std::vector<float> data{0.0f, 0.5f, 1.0f, 2.0f, -1.0f, -2.0f, 4.0f, -0.5f};
    auto spec = create_simple_spec(shape, DataType::FLOAT32);
    HostBuffer host_buffer(data);
    HostTensor tensor(std::move(host_buffer), std::move(spec), TensorTopology());

    auto fp8_tensor = to_dtype(tensor, DataType::FP8_E4M3);
    EXPECT_EQ(fp8_tensor.dtype(), DataType::FP8_E4M3);
    EXPECT_EQ(fp8_tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(fp8_tensor.logical_shape(), shape);

    auto round_trip = to_dtype(fp8_tensor, DataType::FLOAT32);
    auto round_trip_data = host_buffer::get_as<const float>(round_trip);
    ASSERT_EQ(round_trip_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(round_trip_data[i], data[i]);
    }
}

TEST(HostTensorFp8Test, ToDtypeFp8ToBfloat16Throws) {
    Shape shape{1, 8};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    EXPECT_ANY_THROW(to_dtype(tensor, DataType::BFLOAT16));
}

TEST(HostTensorFp8Test, ToDtypeFp8ToUInt32Throws) {
    Shape shape{1, 8};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    EXPECT_ANY_THROW(to_dtype(tensor, DataType::UINT32));
}

TEST(HostTensorFp8Test, ToDtypeBfloat16ToFp8Throws) {
    Shape shape{1, 8};
    auto spec = create_simple_spec(shape, DataType::BFLOAT16);
    HostBuffer host_buffer(std::vector<bfloat16>(shape.volume(), bfloat16(0.5f)));
    HostTensor tensor(std::move(host_buffer), std::move(spec), TensorTopology());

    EXPECT_ANY_THROW(to_dtype(tensor, DataType::FP8_E4M3));
}

TEST(HostTensorFp8Test, PadThrows) {
    Shape shape{1, 8};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    EXPECT_ANY_THROW(pad(tensor, Shape{1, 16}, Shape{0, 0}, /*pad_value=*/0.0f));
}

TEST(HostTensorFp8Test, UnpadThrows) {
    Shape shape{1, 16};
    auto tensor = make_fp8_host_tensor(shape, std::vector<float8_e4m3>(shape.volume(), float8_e4m3(0.5f)));
    EXPECT_ANY_THROW(unpad(tensor, Shape{0, 0}, Shape{1, 8}));
}

}  // namespace
}  // namespace tt::tt_metal
