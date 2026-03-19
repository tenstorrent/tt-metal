// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
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

#include <type_traits>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
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
    return HostTensor(std::move(buffer), std::move(spec), std::move(topology));
}

// Type trait tests verifying HostTensor's semantic constraints

TEST(HostTensorTypeTraitsTest, IsDefaultConstructible) { EXPECT_TRUE(std::is_default_constructible_v<HostTensor>); }

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

TEST(HostTensorTest, CopyConstructionFromDefaultConstructed) {
    HostTensor default_tensor;
    const HostTensor& copied(default_tensor);
    (void)copied;
    // Both should be in default-constructed state (no assertions, just shouldn't crash)
}

TEST(HostTensorTest, CopyAssignmentFromDefaultConstructed) {
    HostTensor default_tensor;
    [[maybe_unused]] auto tensor = create_simple_host_tensor(Shape{2, 64});
    tensor = default_tensor;
    // tensor should now be in default-constructed state (no assertions, just shouldn't crash)
}

TEST(HostTensorTest, MoveConstructionWithNewSpecFromDefaultConstructedFails) {
    HostTensor default_tensor;
    auto new_spec = create_simple_spec(Shape{4, 32}, DataType::FLOAT32);
    auto new_topology = TensorTopology();

    EXPECT_ANY_THROW(HostTensor(std::move(default_tensor), std::move(new_spec), std::move(new_topology)));
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

TEST(HostTensorTest, MoveConstructionWithNewSpecAndTopology) {
    Shape original_shape{2, 64};
    auto tensor = create_simple_host_tensor(original_shape);

    Shape new_shape{4, 32};
    auto new_spec = create_simple_spec(new_shape, DataType::FLOAT32);
    auto new_topology = TensorTopology();

    HostTensor new_tensor(std::move(tensor), std::move(new_spec), std::move(new_topology));

    EXPECT_EQ(new_tensor.logical_shape(), new_shape);
    EXPECT_EQ(new_tensor.dtype(), DataType::FLOAT32);
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

}  // namespace
}  // namespace tt::tt_metal
