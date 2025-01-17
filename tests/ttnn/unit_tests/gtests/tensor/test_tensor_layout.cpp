// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "gtest/gtest.h"
#include <tt-metalium/logger.hpp>
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"

#include "common_tensor_test_utils.hpp"

using namespace ttnn;
using namespace tt::tt_metal;

namespace {
const MemoryConfig DefaultMemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt};

struct Inputs {
    SimpleShape shape;
    DataType data_type;
    Layout layout;
};

struct Expected {
    Size physical_size;
    Alignment alignment;
    Strides strides;
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
            Inputs{.shape = ttnn::SimpleShape{5, 4, 3, 2}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {5 * 4 * 32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({32 * 3 * 4, 32 * 3, 32, 1})}},

        // Row Major, bfloat16, requires padding to 2
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{6, 5, 4, 3}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 3},
                .alignment = Alignment({1}),
                .strides = Strides({5 * 4 * 3, 4 * 3, 3, 1})}},

        // Row Major, uint32
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{6, 5, 4, 3}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 3},
                .alignment = Alignment({1}),
                .strides = Strides({5 * 4 * 3, 4 * 3, 3, 1})}},

        // Row Major, bfloat16, requires padding to 2, aligned
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{6, 5, 4, 8}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {6 * 5 * 4, 8},
                .alignment = Alignment({1}),
                .strides = Strides({5 * 4 * 8, 4 * 8, 8, 1})}},

        // Tile, 1 element
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{1, 1, 1, 1}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{.physical_size = {32, 32}, .alignment = Alignment({32, 32}), .strides = Strides({32, 32, 32, 1})}},

        // Row Major, 1 element
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{.physical_size = {1, 1}, .alignment = Alignment({1}), .strides = Strides({1, 1, 1, 1})}},

        // Row Major, uint32_t 1 element
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{1, 1, 1, 1}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{.physical_size = {1, 1}, .alignment = Alignment({1}), .strides = Strides({1, 1, 1, 1})}},

        // Rank 0, RM, in bfloat16 needs additional padding to 4 bytes
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({}),
                .tensor_creation_works = false}},

        // Rank 0, RM, in uint32_t needs no additional padding
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({}),
                .tensor_creation_works = false}},

        // Rank 0, Tile
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{
                .physical_size = {32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({}),
                .tensor_creation_works = false}},

        // Rank 1, RM, bfloat16
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{1}, .data_type = DataType::BFLOAT16, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({1}),
                .tensor_creation_works = false}},

        // Rank 1, RM, uint32
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{1}, .data_type = DataType::UINT32, .layout = Layout::ROW_MAJOR},
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({1}),
                .tensor_creation_works = false}},

        // Rank 1, Tile
        TensorLayoutTestParams{
            Inputs{.shape = ttnn::SimpleShape{1}, .data_type = DataType::BFLOAT16, .layout = Layout::TILE},
            Expected{.physical_size = {32, 32}, .alignment = Alignment({32, 32}), .strides = Strides({1})}}));

struct LegacyPaddingRoundtripTestParams {
    SimpleShape shape;
    SimpleShape padded_shape;
};

class TensorLayoutLegacyPaddingRoundtipTests : public ::testing::TestWithParam<LegacyPaddingRoundtripTestParams> {};

TEST_P(TensorLayoutLegacyPaddingRoundtipTests, Tensor_LagacyPaddingRoundtrip) {
    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromLegacyPaddedShape(
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        DefaultMemoryConfig,
        Shape(params.shape.view(), params.padded_shape.view()));
    EXPECT_EQ(layout.compute_padded_shape(params.shape), params.padded_shape);

    test_utils::test_tensor_on_device(params.shape, layout);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutLegacyPaddingRoundtipTests,
    ::testing::Values(
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{10},
            .padded_shape = SimpleShape{32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{10},
            .padded_shape = SimpleShape{20},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{40, 30},
            .padded_shape = SimpleShape{64, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{40, 30},
            .padded_shape = SimpleShape{40, 32},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{30, 20, 10},
            .padded_shape = SimpleShape{32, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{30, 20, 10},
            .padded_shape = SimpleShape{30, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{30, 20, 10},
            .padded_shape = SimpleShape{30, 20, 12},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{30, 20, 10},
            .padded_shape = SimpleShape{30, 20, 10},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{2, 3, 16, 16},
            .padded_shape = SimpleShape{16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{2, 3, 16, 16},
            .padded_shape = SimpleShape{2, 16, 16, 16},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{2, 3, 16, 16},
            .padded_shape = SimpleShape{2, 3, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{2, 3, 16, 16},
            .padded_shape = SimpleShape{2, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{2, 3, 16, 16},
            .padded_shape = SimpleShape{2, 3, 16, 16},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = SimpleShape{16, 16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = SimpleShape{5, 4, 4, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = SimpleShape{5, 4, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = SimpleShape{5, 4, 3, 16, 16},
        }));
