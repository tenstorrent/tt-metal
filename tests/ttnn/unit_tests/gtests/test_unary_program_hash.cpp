// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cache-key tests for UnaryDeviceOperation::compute_program_hash.
//
// These stop at the hash generation intentionally. Both CBs are sized from tile_size(DataFormat)
// (unary_program_factory.cpp:356-358, 371-372), which assumes a fixed 32x32 tile. Dispatching a
// different tile size will write past the output buffer and returns wrong data.

#include <gtest/gtest.h>

#include <array>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {
namespace {

using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::Tile;

const ttnn::Shape kShape{1, 1, 64, 64};

Tensor make_tensor(tt::tt_metal::distributed::MeshDevice* device, const std::array<uint32_t, 2>& tile) {
    return ttnn::create_device_tensor(
        TensorSpec(
            kShape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, Tile(tile)), ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

UnaryDeviceOperation::operation_attributes_t make_attributes(tt::tt_metal::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    return UnaryDeviceOperation::operation_attributes_t{
        .op_chain = {EltwiseUnaryWithParam{UnaryOpType::RELU}},
        .output_dtype = DataType::BFLOAT16,
        .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        .worker_grid = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
    };
}

class UnaryProgramHashFixture : public TTNNFixtureWithSuiteDevice<UnaryProgramHashFixture> {};

TEST_F(UnaryProgramHashFixture, SameTileSharesKey) {
    const auto attributes = make_attributes(device_);
    const auto first = make_tensor(device_, {32, 32});
    const auto second = make_tensor(device_, {32, 32});

    EXPECT_EQ(
        UnaryDeviceOperation::compute_program_hash(attributes, {first, std::nullopt}),
        UnaryDeviceOperation::compute_program_hash(attributes, {second, std::nullopt}));
}

TEST_F(UnaryProgramHashFixture, InputTileSeparatesKeys) {
    const auto attributes = make_attributes(device_);
    const auto tile_32x32 = make_tensor(device_, {32, 32});
    const auto tile_16x32 = make_tensor(device_, {16, 32});

    ASSERT_EQ(tile_32x32.padded_shape(), tile_16x32.padded_shape());
    EXPECT_NE(
        UnaryDeviceOperation::compute_program_hash(attributes, {tile_32x32, std::nullopt}),
        UnaryDeviceOperation::compute_program_hash(attributes, {tile_16x32, std::nullopt}));
}

TEST_F(UnaryProgramHashFixture, OutputTileSeparatesKeys) {
    const auto attributes = make_attributes(device_);
    const auto input = make_tensor(device_, {32, 32});
    const auto output_32x32 = make_tensor(device_, {32, 32});
    const auto output_16x32 = make_tensor(device_, {16, 32});

    EXPECT_NE(
        UnaryDeviceOperation::compute_program_hash(attributes, {input, output_32x32}),
        UnaryDeviceOperation::compute_program_hash(attributes, {input, output_16x32}));
}

}  // namespace
}  // namespace ttnn::operations::unary::test
