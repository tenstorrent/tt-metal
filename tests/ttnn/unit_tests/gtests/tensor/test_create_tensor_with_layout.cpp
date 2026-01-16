// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <memory>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {
struct Inputs {
    ttnn::Shape shape;
    TensorLayout layout;
};

struct Expected {
    ttnn::Shape padded_shape;
};

struct CreateTensorParams {
    Inputs inputs;
    Expected expected;
};

}  // namespace

class CreateTensorWithLayoutTest : public ttnn::TTNNFixtureWithSuiteDevice<CreateTensorWithLayoutTest>,
                                   public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorWithLayoutTest, Tile) {
    const CreateTensorParams& params = GetParam();

    auto tensor = tt::tt_metal::create_device_tensor(TensorSpec(params.inputs.shape, params.inputs.layout), device_);
    EXPECT_EQ(tensor.padded_shape(), params.expected.padded_shape);
    EXPECT_EQ(tensor.logical_shape(), params.inputs.shape);
}

namespace {
const tt::tt_metal::MemoryConfig DefaultMemoryConfig{
    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt};
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorWithLayoutTestWithShape,
    CreateTensorWithLayoutTest,
    ::testing::Values(
        CreateTensorParams{
            Inputs{
                .shape = ttnn::Shape({1, 1, 32, 32}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)},
            Expected{.padded_shape = ttnn::Shape({1, 1, 32, 32})}},

        CreateTensorParams{
            Inputs{
                .shape = ttnn::Shape({1, 1, 16, 10}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)},
            Expected{.padded_shape = ttnn::Shape({1, 1, 32, 32})}},

        CreateTensorParams{
            Inputs{
                .shape = ttnn::Shape({1, 1, 16, 10}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig)},
            Expected{.padded_shape = ttnn::Shape({1, 1, 16, 10})}}));
