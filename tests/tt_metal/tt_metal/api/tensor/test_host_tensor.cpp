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
// - pad() and pad_to_tile() with typed pad_value

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <type_traits>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
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
    return HostTensor::from_buffer(std::move(buffer), std::move(spec), std::move(topology));
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

    HostTensor tensor = HostTensor::from_buffer(std::move(host_buffer), std::move(spec), std::move(topology));

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

// ==================================================================================
// pad() and pad_to_tile() — typed pad_value
// ==================================================================================

// Helper: build a 1-shard HostTensor with actual data from a vector<T>.
template <typename T>
HostTensor make_tensor(const std::vector<T>& data, const Shape& shape, DataType dtype) {
    return HostTensor::from_vector(data, create_simple_spec(shape, dtype));
}

TEST(HostTensorPadTest, PadFloat32FillsCorrectValue) {
    // 1×2 tensor padded to 1×4 with pad_value = 9.0f
    auto t = make_tensor<float>({1.0f, 2.0f}, Shape{1, 2}, DataType::FLOAT32);
    auto padded = pad<float>(t, Shape{1, 4}, Shape{0, 0}, 9.0f);

    EXPECT_EQ(padded.logical_shape(), (Shape{1, 2}));
    EXPECT_EQ(padded.padded_shape(), (Shape{1, 4}));

    auto data = host_buffer::get_as<float>(padded);
    ASSERT_EQ(data.size(), 4u);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 9.0f);
    EXPECT_FLOAT_EQ(data[3], 9.0f);
}

TEST(HostTensorPadTest, PadBfloat16FillsCorrectValue) {
    auto t = make_tensor<bfloat16>({bfloat16(3.0f), bfloat16(4.0f)}, Shape{1, 2}, DataType::BFLOAT16);
    auto padded = pad<bfloat16>(t, Shape{1, 4}, Shape{0, 0}, bfloat16(0.0f));

    auto data = host_buffer::get_as<bfloat16>(padded);
    ASSERT_EQ(data.size(), 4u);
    EXPECT_FLOAT_EQ(float(data[0]), 3.0f);
    EXPECT_FLOAT_EQ(float(data[1]), 4.0f);
    EXPECT_FLOAT_EQ(float(data[2]), 0.0f);
    EXPECT_FLOAT_EQ(float(data[3]), 0.0f);
}

TEST(HostTensorPadTest, PadUint32FillsCorrectValue) {
    auto t = make_tensor<uint32_t>({7u, 8u}, Shape{1, 2}, DataType::UINT32);
    auto padded = pad<uint32_t>(t, Shape{1, 4}, Shape{0, 0}, 42u);

    auto data = host_buffer::get_as<uint32_t>(padded);
    ASSERT_EQ(data.size(), 4u);
    EXPECT_EQ(data[0], 7u);
    EXPECT_EQ(data[1], 8u);
    EXPECT_EQ(data[2], 42u);
    EXPECT_EQ(data[3], 42u);
}

TEST(HostTensorPadTest, PadZeroFill) {
    auto t = make_tensor<float>({5.0f}, Shape{1, 1}, DataType::FLOAT32);
    auto padded = pad<float>(t, Shape{1, 4}, Shape{0, 0}, 0.0f);

    auto data = host_buffer::get_as<float>(padded);
    ASSERT_EQ(data.size(), 4u);
    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 0.0f);
}

TEST(HostTensorPadTest, PadDtypeMismatchFatals) {
    // Passing bfloat16 pad_value to a FLOAT32 tensor must fatal.
    auto t = make_tensor<float>({1.0f, 2.0f}, Shape{1, 2}, DataType::FLOAT32);
    EXPECT_ANY_THROW((pad<bfloat16>(t, Shape{1, 4}, Shape{0, 0}, bfloat16(0.0f))));
}

TEST(HostTensorPadToTileTest, PadToTileFloat32ShapeAndFill) {
    // pad_to_tile() rounds the last TWO dims up to TILE_HEIGHT×TILE_WIDTH, so a 1×20 tensor pads
    // to 32×32: row 0 holds the 20 logical values then 12 width-pads, rows 1..31 are all pad.
    std::vector<float> data(20, 1.0f);
    auto t = make_tensor<float>(data, Shape{1, 20}, DataType::FLOAT32);
    auto padded = pad_to_tile<float>(t, 0.0f);

    EXPECT_EQ(padded.padded_shape(), (Shape{32, 32}));

    auto out = host_buffer::get_as<float>(padded);
    ASSERT_EQ(out.size(), 32u * 32u);
    for (int col = 0; col < 32; ++col) {
        EXPECT_FLOAT_EQ(out[col], col < 20 ? 1.0f : 0.0f);
    }
    for (size_t i = 32; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}

TEST(HostTensorPadToTileTest, PadToTileZeroFill) {
    std::vector<float> data(4, 7.0f);
    auto t = make_tensor<float>(data, Shape{1, 4}, DataType::FLOAT32);
    auto padded = pad_to_tile<float>(t, 0.0f);

    EXPECT_EQ(padded.padded_shape(), (Shape{32, 32}));

    auto out = host_buffer::get_as<float>(padded);
    ASSERT_EQ(out.size(), 32u * 32u);
    EXPECT_FLOAT_EQ(out[0], 7.0f);   // logical value
    EXPECT_FLOAT_EQ(out[4], 0.0f);   // width pad on row 0
    EXPECT_FLOAT_EQ(out[32], 0.0f);  // height pad (row 1)
}

TEST(HostTensorPadToTileTest, PadToTileAlreadyAlignedIsNoop) {
    // A 32×32 tensor is already tile-aligned in both dims, so pad_to_tile is a no-op:
    // padded_shape stays 32×32 and every element keeps its original value.
    std::vector<float> data(32 * 32, 5.0f);
    auto t = make_tensor<float>(data, Shape{32, 32}, DataType::FLOAT32);
    auto padded = pad_to_tile<float>(t, -1.0f);

    EXPECT_EQ(padded.padded_shape(), (Shape{32, 32}));

    auto out = host_buffer::get_as<float>(padded);
    ASSERT_EQ(out.size(), 32u * 32u);
    for (auto v : out) {
        EXPECT_FLOAT_EQ(v, 5.0f);
    }
}

}  // namespace
}  // namespace tt::tt_metal
