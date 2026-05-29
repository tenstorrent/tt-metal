// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Device tests
// ---------------------------------------------------------------------------

class UncheckedReinterpretLayoutDeviceTest
    : public ttnn::TTNNFixtureWithSuiteDevice<UncheckedReinterpretLayoutDeviceTest> {};

TEST_F(UncheckedReinterpretLayoutDeviceTest, TileToRowMajorPreservesShapeAndDtype) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    Tensor tile_tensor = create_device_tensor(spec, device_);
    ASSERT_EQ(tile_tensor.layout(), Layout::TILE);

    Tensor rm_tensor = unchecked_reinterpret_layout(tile_tensor, Layout::ROW_MAJOR);

    EXPECT_EQ(rm_tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(rm_tensor.logical_shape(), tile_tensor.logical_shape());
    EXPECT_EQ(rm_tensor.dtype(), tile_tensor.dtype());
    EXPECT_EQ(rm_tensor.tensor_spec().tile(), tile_tensor.tensor_spec().tile());
}

TEST_F(UncheckedReinterpretLayoutDeviceTest, RowMajorToTilePreservesShapeAndDtype) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg));

    Tensor rm_tensor = create_device_tensor(spec, device_);
    ASSERT_EQ(rm_tensor.layout(), Layout::ROW_MAJOR);

    Tensor tile_tensor = unchecked_reinterpret_layout(rm_tensor, Layout::TILE);

    EXPECT_EQ(tile_tensor.layout(), Layout::TILE);
    EXPECT_EQ(tile_tensor.logical_shape(), rm_tensor.logical_shape());
    EXPECT_EQ(tile_tensor.dtype(), rm_tensor.dtype());
    EXPECT_EQ(tile_tensor.tensor_spec().tile(), rm_tensor.tensor_spec().tile());
}

TEST_F(UncheckedReinterpretLayoutDeviceTest, PreservesNonDefaultTile) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    Tile custom_tile({16, 32});
    ttnn::Shape shape({1, 1, 16, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, custom_tile), mem_cfg));

    Tensor original = create_device_tensor(spec, device_);
    ASSERT_EQ(original.tensor_spec().tile(), custom_tile);

    Tensor reinterpreted = unchecked_reinterpret_layout(original, Layout::ROW_MAJOR);

    EXPECT_EQ(reinterpreted.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(reinterpreted.tensor_spec().tile(), custom_tile);
}

TEST_F(UncheckedReinterpretLayoutDeviceTest, AliasesTheSameDeviceAddress) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    Tensor original = create_device_tensor(spec, device_);
    Tensor reinterpreted = unchecked_reinterpret_layout(original, Layout::ROW_MAJOR);

    EXPECT_EQ(
        original.device_storage().get_mesh_buffer().address(),
        reinterpreted.device_storage().get_mesh_buffer().address());
}

TEST_F(UncheckedReinterpretLayoutDeviceTest, SameLayoutIsIdentity) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    Tensor original = create_device_tensor(spec, device_);
    Tensor reinterpreted = unchecked_reinterpret_layout(original, Layout::TILE);

    EXPECT_EQ(reinterpreted.layout(), Layout::TILE);
    EXPECT_EQ(reinterpreted.tensor_spec(), original.tensor_spec());
    EXPECT_EQ(
        original.device_storage().get_mesh_buffer().address(),
        reinterpreted.device_storage().get_mesh_buffer().address());
}

TEST_F(UncheckedReinterpretLayoutDeviceTest, OriginalTensorStaysAliveAfterReinterpretedDeallocated) {
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    Tensor original = create_device_tensor(spec, device_);
    {
        Tensor reinterpreted = unchecked_reinterpret_layout(original, Layout::ROW_MAJOR);
        ASSERT_TRUE(reinterpreted.is_allocated());
    }
    EXPECT_TRUE(original.is_allocated());
}

// ---------------------------------------------------------------------------
// Host tests
// ---------------------------------------------------------------------------

class UncheckedReinterpretLayoutHostTest : public ::testing::Test {};

TEST_F(UncheckedReinterpretLayoutHostTest, TileToRowMajorOnHost) {
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    auto num_elements = shape.volume();
    std::vector<bfloat16> data(num_elements, bfloat16(1.0f));
    Tensor host_tensor(HostBuffer(data), spec);
    ASSERT_EQ(host_tensor.layout(), Layout::TILE);

    Tensor reinterpreted = unchecked_reinterpret_layout(host_tensor, Layout::ROW_MAJOR);

    EXPECT_EQ(reinterpreted.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(reinterpreted.logical_shape(), host_tensor.logical_shape());
    EXPECT_EQ(reinterpreted.dtype(), host_tensor.dtype());
    EXPECT_EQ(reinterpreted.tensor_spec().tile(), host_tensor.tensor_spec().tile());
}

TEST_F(UncheckedReinterpretLayoutHostTest, PreservesNonDefaultTileOnHost) {
    Tile custom_tile({16, 32});
    ttnn::Shape shape({1, 1, 16, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, custom_tile), MemoryConfig{}));

    auto num_elements = shape.volume();
    std::vector<bfloat16> data(num_elements, bfloat16(1.0f));
    Tensor host_tensor(HostBuffer(data), spec);
    ASSERT_EQ(host_tensor.tensor_spec().tile(), custom_tile);

    Tensor reinterpreted = unchecked_reinterpret_layout(host_tensor, Layout::ROW_MAJOR);

    EXPECT_EQ(reinterpreted.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(reinterpreted.tensor_spec().tile(), custom_tile);
}

TEST_F(UncheckedReinterpretLayoutHostTest, RowMajorToTileOnHost) {
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    auto num_elements = shape.volume();
    std::vector<bfloat16> data(num_elements, bfloat16(2.0f));
    Tensor host_tensor(HostBuffer(data), spec);
    ASSERT_EQ(host_tensor.layout(), Layout::ROW_MAJOR);

    Tensor reinterpreted = unchecked_reinterpret_layout(host_tensor, Layout::TILE);

    EXPECT_EQ(reinterpreted.layout(), Layout::TILE);
    EXPECT_EQ(reinterpreted.logical_shape(), host_tensor.logical_shape());
    EXPECT_EQ(reinterpreted.dtype(), host_tensor.dtype());
}
