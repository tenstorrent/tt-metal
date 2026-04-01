// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Sanity tests for MeshTensor class.
//
// These are minimal compile-time and runtime checks to verify MeshTensor's
// basic type properties and construction semantics. MeshTensor enforces unique
// ownership of device memory: movable but non-copyable.
//
// Device-based tests use GenericMeshDeviceFixture and require hardware access.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <type_traits>

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/math.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Type trait tests verifying MeshTensor's semantic constraints

TEST(MeshTensorTypeTraitsTest, IsDefaultConstructible) { EXPECT_TRUE(std::is_default_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsDestructible) { EXPECT_TRUE(std::is_destructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNotCopyConstructible) { EXPECT_FALSE(std::is_copy_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNotCopyAssignable) { EXPECT_FALSE(std::is_copy_assignable_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsMoveConstructible) { EXPECT_TRUE(std::is_move_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsMoveAssignable) { EXPECT_TRUE(std::is_move_assignable_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNothrowMoveConstructible) {
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<MeshTensor>);
}

TEST(MeshTensorTypeTraitsTest, IsNothrowMoveAssignable) { EXPECT_TRUE(std::is_nothrow_move_assignable_v<MeshTensor>); }

// Runtime tests for default construction and move semantics

TEST(MeshTensorTest, DefaultConstruction) {
    MeshTensor tensor;
    // Default constructed tensor is in valueless state
    // Accessing members would trigger TT_ASSERT in debug builds
    (void)tensor;
}

TEST(MeshTensorTest, MoveConstruction) {
    MeshTensor tensor;
    MeshTensor moved(std::move(tensor));
    (void)moved;
}

TEST(MeshTensorTest, MoveAssignment) {
    MeshTensor tensor;
    MeshTensor other;
    other = std::move(tensor);
    (void)other;
}

TEST(MeshTensorTest, ConstructionWithNullMeshBufferFails) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);
    auto topology = TensorTopology();

    EXPECT_ANY_THROW(MeshTensor(nullptr, std::move(spec), std::move(topology)));
}

TEST(MeshTensorTest, MoveConstructionWithNewSpecFromDefaultConstructedFails) {
    MeshTensor default_tensor;

    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto new_spec = TensorSpec(Shape{4, 32}, tensor_layout);
    auto new_topology = TensorTopology();

    EXPECT_ANY_THROW(MeshTensor(std::move(default_tensor), std::move(new_spec), std::move(new_topology)));
}

// Device-based tests using GenericMeshDeviceFixture
using MeshTensorDeviceTest = GenericMeshDeviceFixture;

auto create_mesh_buffer(
    distributed::MeshDevice& mesh_device, const TensorSpec& spec, float buffer_size_multiplier = 1.0f) {
    auto page_size = spec.compute_page_size_bytes();

    distributed::DeviceLocalBufferConfig local_config{
        .page_size = static_cast<uint32_t>(page_size),
        .buffer_type = BufferType::DRAM,
    };

    auto buffer_size_needed = spec.compute_packed_buffer_size_bytes();
    auto scaled_size = static_cast<DeviceAddr>(buffer_size_needed * buffer_size_multiplier);
    // Round down to nearest multiple of page_size (buffer_size % page_size must == 0).
    // Ensure minimum of one page.
    auto buffer_size = std::max(round_down(scaled_size, page_size), page_size);

    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    return distributed::MeshBuffer::create(buffer_config, local_config, &mesh_device);
}

TEST_F(MeshTensorDeviceTest, ConstructionWithMeshBuffer) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    ASSERT_NE(mesh_buffer, nullptr);
    ASSERT_TRUE(mesh_buffer->is_allocated());

    auto topology = TensorTopology();

    MeshTensor tensor(mesh_buffer, std::move(spec), std::move(topology));

    EXPECT_EQ(&tensor.mesh_buffer(), mesh_buffer.get());
    EXPECT_EQ(&tensor.device(), mesh_device_.get());
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({1, 32}));
}

TEST_F(MeshTensorDeviceTest, MoveConstructionTransfersOwnership) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{2, 64}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    auto topology = TensorTopology();

    MeshTensor original(mesh_buffer, std::move(spec), std::move(topology));
    auto* buffer_ptr = &original.mesh_buffer();

    MeshTensor moved(std::move(original));

    EXPECT_EQ(&moved.mesh_buffer(), buffer_ptr);
    EXPECT_EQ(moved.logical_shape(), Shape({2, 64}));
}

TEST_F(MeshTensorDeviceTest, MoveAssignmentTransfersOwnership) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);

    auto spec1 = TensorSpec(Shape{1, 32}, tensor_layout);
    auto spec2 = TensorSpec(Shape{2, 64}, tensor_layout);

    auto mesh_buffer1 = create_mesh_buffer(*mesh_device_, spec1);
    auto mesh_buffer2 = create_mesh_buffer(*mesh_device_, spec2);

    MeshTensor tensor1(mesh_buffer1, std::move(spec1), TensorTopology());
    MeshTensor tensor2(mesh_buffer2, std::move(spec2), TensorTopology());

    auto* buffer1_ptr = &tensor1.mesh_buffer();

    tensor2 = std::move(tensor1);

    EXPECT_EQ(&tensor2.mesh_buffer(), buffer1_ptr);
    EXPECT_EQ(tensor2.logical_shape(), Shape({1, 32}));
}

TEST_F(MeshTensorDeviceTest, MoveConstructionWithNewSpecAndTopology) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{2, 64}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    MeshTensor original(mesh_buffer, std::move(spec), TensorTopology());
    auto* buffer_ptr = &original.mesh_buffer();

    auto new_tensor_layout = TensorLayout(DataType::FLOAT32, page_config, memory_config);
    auto new_spec = TensorSpec(Shape{4, 32}, new_tensor_layout);
    auto new_topology = TensorTopology();

    MeshTensor moved(std::move(original), std::move(new_spec), std::move(new_topology));

    EXPECT_EQ(&moved.mesh_buffer(), buffer_ptr);
    EXPECT_EQ(moved.logical_shape(), Shape({4, 32}));
    EXPECT_EQ(moved.dtype(), DataType::FLOAT32);
}

TEST_F(MeshTensorDeviceTest, TensorProperties) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::FLOAT32, page_config, memory_config);
    auto spec = TensorSpec(Shape{4, 8, 16}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    auto topology = TensorTopology();

    MeshTensor tensor(mesh_buffer, std::move(spec), std::move(topology));

    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({4, 8, 16}));
    EXPECT_EQ(tensor.logical_volume(), 4 * 8 * 16);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
    EXPECT_FALSE(tensor.is_sharded());
    EXPECT_EQ(tensor.memory_config().buffer_type(), BufferType::DRAM);
}

TEST_F(MeshTensorDeviceTest, ConstructionWithTooSmallBufferFails) {
    // Create a TensorSpec that requires multiple pages, then create a buffer
    // that's too small to hold the tensor but still valid (multiple of page_size).
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    // Use a larger shape to ensure multiple pages are required.
    auto spec = TensorSpec(Shape{4, 512}, tensor_layout);

    const size_t required_size = spec.compute_packed_buffer_size_bytes();
    const size_t page_size = spec.compute_page_size_bytes();
    ASSERT_GT(required_size, page_size);  // Ensure we need multiple pages

    // Create a buffer at half the required size (rounded to page boundary).
    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec, 0.5f);
    ASSERT_NE(mesh_buffer, nullptr);
    ASSERT_TRUE(mesh_buffer->is_allocated());
    ASSERT_LT(mesh_buffer->size(), required_size);

    auto topology = TensorTopology();

    EXPECT_ANY_THROW(MeshTensor(mesh_buffer, std::move(spec), std::move(topology)));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace tt::tt_metal
