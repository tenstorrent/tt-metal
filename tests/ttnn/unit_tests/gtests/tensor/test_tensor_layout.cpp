// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/tensor/tensor_layout.hpp"
#include "tt_metal/common/logger.hpp"

namespace {
const tt::tt_metal::MemoryConfig DefaultMemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt};
}

struct Inputs {
    ttnn::SimpleShape shape;
    tt::tt_metal::DataType data_type;
    tt::tt_metal::Layout layout;
};

struct Expected {
    tt::tt_metal::Size physical_size;
    tt::tt_metal::Size tile_alignment_padding;
};

struct TensorLayoutTestParams {
    Inputs inputs;
    Expected expected;
};

class TensorLayoutTests : public ::testing::TestWithParam<TensorLayoutTestParams> {};

TEST_P(TensorLayoutTests, Tensor_PhysicalSize) {
    using namespace tt::tt_metal;

    const auto& params = GetParam();
    TensorLayout layout(params.inputs.data_type, params.inputs.layout, DefaultMemoryConfig);
    Size physical_size = layout.get_physical_size(params.inputs.shape);
    Size tile_alignment_padding = layout.get_tile_alignment_padding(params.inputs.shape);

    EXPECT_EQ(physical_size, params.expected.physical_size);
    EXPECT_EQ(tile_alignment_padding, params.expected.tile_alignment_padding);
}

INSTANTIATE_TEST_SUITE_P(
    TiledTensorTests,
    TensorLayoutTests,
    ::testing::Values(
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::TILE
            },
            Expected{
                .physical_size = {32, 32},
                .tile_alignment_padding = {31, 31}
            }
        },

        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 1},
                .tile_alignment_padding = {0, 0}
            }
        }
    )
);
