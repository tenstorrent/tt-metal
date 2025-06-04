// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <tt-metalium/shape2d.hpp>
#include <initializer_list>
#include <memory>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/alignment.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace ttnn;

namespace {
const MemoryConfig DefaultMemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt};

struct Inputs {
    Shape shape;
    DataType data_type;
    Layout layout;
};

struct Expected {
    tt::tt_metal::Shape2D physical_size;
    tt::tt_metal::Alignment alignment;
    tt::tt_metal::Strides strides;
    bool tensor_creation_works = true;
};

struct TensorLayoutTestParams {
    Inputs inputs;
    Expected expected;
};
}  // namespace

class TensorLayoutComputeTests : public ::testing::TestWithParam<TensorLayoutTestParams> {};

TEST_P(TensorLayoutComputeTests, TensorLayout_Generic) {
    const auto& params = GetParam();
    TensorLayout layout(params.inputs.data_type, PageConfig(params.inputs.layout), DefaultMemoryConfig);

    EXPECT_EQ(layout.get_alignment(), params.expected.alignment);
    EXPECT_EQ(layout.compute_physical_shape(params.inputs.shape), params.expected.physical_size);
    EXPECT_EQ(layout.compute_strides(params.inputs.shape), params.expected.strides);

    if (params.expected.tensor_creation_works) {
        test_utils::test_tensor_on_device(params.inputs.shape, layout);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutComputeTests,
    ::testing::Values(
        // Tiled
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{5, 4, 3, 2}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {5 * 4 * 32, 32},
                .alignment = tt::tt_metal::Alignment({32, 32}),
                .strides = tt::tt_metal::Strides({32 * 3 * 4, 32 * 3, 32, 1})}},

        // Row Major, bfloat16, requires padding to 2
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{6, 5, 4, 3}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 3},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({5 * 4 * 3, 4 * 3, 3, 1})}},

        // Row Major, uint32
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{6, 5, 4, 3}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 3},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({5 * 4 * 3, 4 * 3, 3, 1})}},

        // Row Major, bfloat16, requires padding to 2, aligned
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{6, 5, 4, 8}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 8},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({5 * 4 * 8, 4 * 8, 8, 1})}},

        // Tile, 1 element
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1, 1, 1, 1}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {32, 32},
                .alignment = tt::tt_metal::Alignment({32, 32}),
                .strides = tt::tt_metal::Strides({32, 32, 32, 1})}},

        // Row Major, 1 element
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1, 1, 1, 1}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({1, 1, 1, 1})}},

        // Row Major, uint32_t 1 element
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1, 1, 1, 1}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({1, 1, 1, 1})}},

        // Rank 0, RM, in bfloat16 needs additional padding to 4 bytes
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({}),
                .tensor_creation_works = false}},

        // Rank 0, RM, in uint32_t needs no additional padding
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({}),
                .tensor_creation_works = false}},

        // Rank 0, Tile
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {32, 32},
                .alignment = tt::tt_metal::Alignment({32, 32}),
                .strides = tt::tt_metal::Strides({}),
                .tensor_creation_works = false}},

        // Rank 1, RM, bfloat16
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({1}),
                .tensor_creation_works = false}},

        // Rank 1, RM, uint32
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = tt::tt_metal::Alignment({1}),
                .strides = tt::tt_metal::Strides({1}),
                .tensor_creation_works = false}},

        // Rank 1, Tile
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::Shape{1}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {32, 32},
                .alignment = tt::tt_metal::Alignment({32, 32}),
                .strides = tt::tt_metal::Strides({1})}}));

struct LegacyPaddingRoundtripTestParams {
    Shape shape;
    Shape padded_shape;
};

class TensorLayoutLegacyPaddingRoundtipTests : public ::testing::TestWithParam<LegacyPaddingRoundtripTestParams> {};

TEST_P(TensorLayoutLegacyPaddingRoundtipTests, Tensor_LagacyPaddingRoundtrip) {
    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromPaddedShape(
        DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig, params.shape, params.padded_shape);
    EXPECT_EQ(layout.compute_padded_shape(params.shape), params.padded_shape);

    test_utils::test_tensor_on_device(params.shape, layout);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutLegacyPaddingRoundtipTests,
    ::testing::Values(
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{10},
            .padded_shape = Shape{32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{10},
            .padded_shape = Shape{20},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = Shape{40, 30},
            .padded_shape = Shape{64, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{40, 30},
            .padded_shape = Shape{40, 32},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = Shape{30, 20, 10},
            .padded_shape = Shape{32, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{30, 20, 10},
            .padded_shape = Shape{30, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{30, 20, 10},
            .padded_shape = Shape{30, 20, 12},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{30, 20, 10},
            .padded_shape = Shape{30, 20, 10},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{2, 16, 16, 16},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{2, 3, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{2, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{2, 3, 16, 16},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = Shape{5, 4, 3, 16, 16},
            .padded_shape = Shape{16, 16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{5, 4, 3, 16, 16},
            .padded_shape = Shape{5, 4, 4, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{5, 4, 3, 16, 16},
            .padded_shape = Shape{5, 4, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = Shape{5, 4, 3, 16, 16},
            .padded_shape = Shape{5, 4, 3, 16, 16},
        }));
