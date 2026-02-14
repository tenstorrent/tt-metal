// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "ttnn/tensor/tensor_ops.hpp"
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::experimental::unit_mesh {
namespace {

using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::ThrowsMessage;
using ::tt::tt_metal::distributed::MeshShape;

using UnitMeshUtils2x4Test = ::tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(UnitMeshUtils2x4Test, AggregateAndDisaggregate) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_THAT(unit_meshes, SizeIs(mesh_device_->shape().mesh_size()));

    // Allocates and deallocates a buffer, returning the allocation address.
    // Used to probe where new buffers are being allocated on the parent mesh, as a proxy for the parent mesh allocator
    // state.
    auto get_parent_allocation_address = [&]() {
        auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
            tt::tt_metal::distributed::ReplicatedBufferConfig{.size = 16 << 10},
            tt::tt_metal::distributed::DeviceLocalBufferConfig{
                .page_size = 1024, .buffer_type = tt::tt_metal::BufferType::DRAM},
            mesh_device_.get());
        EXPECT_TRUE(buffer->is_allocated());
        return buffer->address();
    };
    const auto initial_parent_address = get_parent_allocation_address();
    EXPECT_EQ(get_parent_allocation_address(), initial_parent_address);

    // Create a tensor spec for testing
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto dtype = tt::tt_metal::DataType::BFLOAT16;
    auto layout = tt::tt_metal::Layout::TILE;

    // Create tensors on each unit mesh at the same address, assuming deterministic lock-step allocation.
    std::vector<Tensor> unit_tensors;
    unit_tensors.reserve(unit_meshes.size());

    for (const auto& unit_mesh : unit_meshes) {
        auto tensor = create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig())),
            unit_mesh.get());
        unit_tensors.push_back(tensor);
    }

    // Verify all tensors are at the same address
    auto reference_address = unit_tensors[0].mesh_buffer()->address();
    for (size_t i = 1; i < unit_tensors.size(); i++) {
        EXPECT_EQ(unit_tensors[i].mesh_buffer()->address(), reference_address);
    }

    // Test aggregate
    auto aggregated_tensor = aggregate(unit_tensors);
    EXPECT_NE(get_parent_allocation_address(), initial_parent_address);

    EXPECT_EQ(aggregated_tensor.device(), mesh_device_.get());
    EXPECT_EQ(aggregated_tensor.logical_shape(), shape);
    EXPECT_EQ(aggregated_tensor.dtype(), dtype);
    EXPECT_EQ(aggregated_tensor.layout(), layout);
    EXPECT_EQ(aggregated_tensor.mesh_buffer()->address(), reference_address);

    // Test disaggregate
    auto disaggregated_tensors = disaggregate(aggregated_tensor);

    ASSERT_THAT(disaggregated_tensors, SizeIs(unit_meshes.size()));

    for (const auto& tensor : disaggregated_tensors) {
        EXPECT_NE(tensor.device(), nullptr);
        EXPECT_EQ(tensor.device()->shape().mesh_size(), 1);
        EXPECT_EQ(tensor.logical_shape(), shape);
        EXPECT_EQ(tensor.dtype(), dtype);
        EXPECT_EQ(tensor.layout(), layout);
        EXPECT_EQ(tensor.mesh_buffer()->address(), reference_address);
    }
}

TEST_F(UnitMeshUtils2x4Test, AggregateEmptyVector) {
    std::vector<Tensor> empty_tensors;
    EXPECT_THAT(
        ([&]() { aggregate(empty_tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot aggregate empty tensor vector")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateNonUnitMeshes) {
    auto non_unit_meshes = mesh_device_->create_submeshes(MeshShape(2, 2));
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto dtype = tt::tt_metal::DataType::BFLOAT16;
    auto layout = tt::tt_metal::Layout::TILE;

    std::vector<Tensor> tensors;
    tensors.reserve(non_unit_meshes.size());
    for (const auto& non_unit_mesh : non_unit_meshes) {
        tensors.push_back(create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig())),
            non_unit_mesh.get()));
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }), ThrowsMessage<std::runtime_error>(HasSubstr("Expected unit mesh (1x1)")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateMismatchedTensorSpecs) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));

    auto shape1 = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto shape2 = ttnn::Shape(std::array<uint32_t, 2>{64, 64});

    std::vector<Tensor> tensors;
    tensors.reserve(unit_meshes.size());
    for (int i = 0; i < unit_meshes.size(); i++) {
        if (i % 2 == 0) {
            tensors.push_back(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    shape1,
                    tt::tt_metal::TensorLayout(
                        tt::tt_metal::DataType::BFLOAT16,
                        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                        tt::tt_metal::MemoryConfig())),
                unit_meshes[i].get()));
        } else {
            tensors.push_back(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    shape2,
                    tt::tt_metal::TensorLayout(
                        tt::tt_metal::DataType::BFLOAT16,
                        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                        tt::tt_metal::MemoryConfig())),
                unit_meshes[i].get()));
        }
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All tensors must have the same TensorSpec")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateMismatchedAddresses) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));

    auto shape = ttnn::Shape(std::array<uint32_t, 2>{64, 64});

    // Make an additional allocation on the first unit mesh to make the addresses mismatch.
    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[0].get());

    std::vector<Tensor> tensors;
    tensors.reserve(unit_meshes.size());
    for (const auto& unit_mesh : unit_meshes) {
        tensors.push_back(create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig())),
            unit_mesh.get()));
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All mesh buffers must be at the same address")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateWrongNumberOfTensors) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_GE(unit_meshes.size(), 2);

    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});

    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[0].get());

    std::vector<Tensor> tensors = {tensor};
    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Input tensors must span the entire parent mesh")));
}

TEST_F(UnitMeshUtils2x4Test, DisaggregateWithoutSubmeshes) {
    // Create a tensor on the parent mesh directly (no submeshes created yet)
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});

    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        mesh_device_.get());

    // Should throw because no submeshes exist
    EXPECT_THAT(
        ([&]() { disaggregate(tensor); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Number of submeshes (0) must match mesh size")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::unit_mesh
