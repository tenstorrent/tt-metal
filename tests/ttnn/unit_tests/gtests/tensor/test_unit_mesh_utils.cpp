// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn::experimental::unit_mesh {
namespace {

using ::testing::SizeIs;
using ::tt::tt_metal::distributed::MeshShape;

using UnitMeshUtilsTest = ::tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(UnitMeshUtilsTest, AggregateAndDisaggregate) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_THAT(unit_meshes, SizeIs(mesh_device_->shape().mesh_size()));

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

    EXPECT_EQ(aggregated_tensor.device(), mesh_device_.get());
    EXPECT_EQ(aggregated_tensor.logical_shape(), shape);
    EXPECT_EQ(aggregated_tensor.dtype(), dtype);
    EXPECT_EQ(aggregated_tensor.layout(), layout);
    EXPECT_EQ(aggregated_tensor.mesh_buffer()->address(), reference_address);

    // Test disaggregate
    auto disaggregated_tensors = disaggregate(aggregated_tensor);

    ASSERT_THAT(disaggregated_tensors, SizeIs(unit_meshes.size()));

    for (size_t i = 0; i < disaggregated_tensors.size(); i++) {
        const auto& tensor = disaggregated_tensors[i];

        EXPECT_NE(tensor.device(), nullptr);
        EXPECT_EQ(tensor.device()->shape().mesh_size(), 1);
        EXPECT_EQ(tensor.logical_shape(), shape);
        EXPECT_EQ(tensor.dtype(), dtype);
        EXPECT_EQ(tensor.layout(), layout);
        EXPECT_EQ(tensor.mesh_buffer()->address(), reference_address);
    }
}

TEST_F(UnitMeshUtilsTest, AggregateEmptyVector) {
    std::vector<Tensor> empty_tensors;
    EXPECT_ANY_THROW(aggregate(empty_tensors));
}

TEST_F(UnitMeshUtilsTest, AggregateMismatchedTensorSpecs) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_GE(unit_meshes.size(), 2);

    auto shape1 = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto shape2 = ttnn::Shape(std::array<uint32_t, 2>{64, 64});

    auto tensor1 = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape1,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[0].get());
    auto tensor2 = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape2,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[1].get());

    std::vector<Tensor> tensors = {tensor1, tensor2};
    EXPECT_ANY_THROW(aggregate(tensors));
}

TEST_F(UnitMeshUtilsTest, AggregateWrongNumberOfTensors) {
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
    EXPECT_ANY_THROW(aggregate(tensors));
}

TEST_F(UnitMeshUtilsTest, DisaggregateWithoutSubmeshes) {
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
    EXPECT_ANY_THROW(disaggregate(tensor));
}

}  // namespace
}  // namespace ttnn::experimental::unit_mesh
