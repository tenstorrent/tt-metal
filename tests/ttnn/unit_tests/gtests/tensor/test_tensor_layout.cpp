// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "gtest/gtest.h"
#include "tt_metal/common/logger.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"

#include "common_tensor_test_utils.hpp"

namespace {
const tt::tt_metal::MemoryConfig DefaultMemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt};

struct Inputs {
    ttnn::SimpleShape shape;
    tt::tt_metal::DataType data_type;
    tt::tt_metal::Layout layout;
};

struct Expected {
    tt::tt_metal::Size physical_size;
    Alignment alignment;
    Strides strides;
    bool tensor_creation_works = true;
};

struct TensorLayoutTestParams {
    Inputs inputs;
    Expected expected;
};
}

class TensorLayoutComputeTests : public ::testing::TestWithParam<TensorLayoutTestParams> {};

TEST_P(TensorLayoutComputeTests, TensorLayout_Generic) {
    using namespace tt::tt_metal;

    const auto& params = GetParam();
    TensorLayout layout(params.inputs.data_type, PageConfig(params.inputs.layout), DefaultMemoryConfig);

    EXPECT_EQ(layout.get_alignment(), params.expected.alignment);
    EXPECT_EQ(layout.get_physical_shape(params.inputs.shape), params.expected.physical_size);
    EXPECT_EQ(layout.get_strides(params.inputs.shape), params.expected.strides);

    if(params.expected.tensor_creation_works) {
        test_utils::test_tensor_on_device(params.inputs.shape, layout);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutComputeTests,
    ::testing::Values(
        // Tiled
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{5, 4, 3, 2},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::TILE
            },
            Expected{
                .physical_size = {5*4*32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({32*3*4, 32*3, 32, 1})
            }
        },

        // Row Major, bfloat16, requires padding to 2
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{6, 5, 4, 3},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {6*5*4, 4},
                .alignment = Alignment({2}),
                .strides = Strides({5*4*4, 4*4, 4, 1})
            }
        },

        // Row Major, uint32
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{6, 5, 4, 3},
                .data_type = DataType::UINT32,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {6*5*4, 3},
                .alignment = Alignment({1}),
                .strides = Strides({5*4*3, 4*3, 3, 1})
            }
        },

        // Row Major, bfloat16, requires padding to 2, aligned
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{6, 5, 4, 8},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {6*5*4, 8},
                .alignment = Alignment({2}),
                .strides = Strides({5*4*8, 4*8, 8, 1})
            }
        },

        // Tile, 1 element
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::TILE
            },
            Expected{
                .physical_size = {32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({32, 32, 32, 1})
            }
        },

        // Row Major, 1 element
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 2},
                .alignment = Alignment({2}),
                .strides = Strides({2, 2, 2, 1})
            }
        },

        // Row Major, uint32_t 1 element
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1, 1, 1, 1},
                .data_type = DataType::UINT32,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({1, 1, 1, 1})
            }
        },

        // Rank 0, RM, in bfloat16 needs additional padding to 4 bytes
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 2},
                .alignment = Alignment({2}),
                .strides = Strides({}),
                .tensor_creation_works = false
            }
        },

        // Rank 0, RM, in uint32_t needs no additional padding
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{},
                .data_type = DataType::UINT32,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({}),
                .tensor_creation_works = false
            }
        },

        // Rank 0, Tile
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::TILE
            },
            Expected{
                .physical_size = {32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({}),
                .tensor_creation_works = false
            }
        },

        // Rank 1, RM, bfloat16
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 2},
                .alignment = Alignment({2}),
                .strides = Strides({1}),
                .tensor_creation_works = false
            }
        },

        // Rank 1, RM, uint32
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1},
                .data_type = DataType::UINT32,
                .layout = Layout::ROW_MAJOR
            },
            Expected{
                .physical_size = {1, 1},
                .alignment = Alignment({1}),
                .strides = Strides({1}),
                .tensor_creation_works = false
            }
        },

        // Rank 1, Tile
        TensorLayoutTestParams{
            Inputs{
                .shape = ttnn::SimpleShape{1},
                .data_type = DataType::BFLOAT16,
                .layout = Layout::TILE
            },
            Expected{
                .physical_size = {32, 32},
                .alignment = Alignment({32, 32}),
                .strides = Strides({1})
            }
        }
    )
);


struct LegacyPaddingRoundtripTestParams {
    ttnn::SimpleShape shape;
    ttnn::SimpleShape padded_shape;
};

class TensorLayoutLegacyPaddingRoundtipTests : public ::testing::TestWithParam<LegacyPaddingRoundtripTestParams> {};

TEST_P(TensorLayoutLegacyPaddingRoundtipTests, Tensor_LagacyPaddingRoundtrip) {
    using namespace tt::tt_metal;

    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromLegacyPaddedShape(DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig, params.padded_shape);
    EXPECT_EQ(layout.get_padded_shape(params.shape), params.padded_shape);

    test_utils::test_tensor_on_device(params.shape, layout);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutLegacyPaddingRoundtipTests,
    ::testing::Values(
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{10},
            .padded_shape = ttnn::SimpleShape{32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{10},
            .padded_shape = ttnn::SimpleShape{20},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{40, 30},
            .padded_shape = ttnn::SimpleShape{64, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{40, 30},
            .padded_shape = ttnn::SimpleShape{40, 32},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{30, 20, 10},
            .padded_shape = ttnn::SimpleShape{32, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{30, 20, 10},
            .padded_shape = ttnn::SimpleShape{30, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{30, 20, 10},
            .padded_shape = ttnn::SimpleShape{30, 20, 12},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{30, 20, 10},
            .padded_shape = ttnn::SimpleShape{30, 20, 10},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{2, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{2, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{2, 16, 16, 16},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{2, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{2, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{2, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{2, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{2, 3, 16, 16},
        },

        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{16, 16, 16, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{5, 4, 4, 32, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{5, 4, 3, 16, 32},
        },
        LegacyPaddingRoundtripTestParams{
            .shape = ttnn::SimpleShape{5, 4, 3, 16, 16},
            .padded_shape = ttnn::SimpleShape{5, 4, 3, 16, 16},
        }
    )
);

struct LegacyPaddingFromAlignmentTestParams {
    tt::tt_metal::Alignment alignment;
    ttnn::SimpleShape shape;
    ttnn::SimpleShape padded_shape;
};

class TensorLayoutLegacyPaddingFromAlignmentTests : public ::testing::TestWithParam<LegacyPaddingFromAlignmentTestParams> {};

TEST_P(TensorLayoutLegacyPaddingFromAlignmentTests, Tensor_LagacyPaddingFromAlignment) {
    using namespace tt::tt_metal;

    const auto& params = GetParam();
    TensorLayout layout = TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig, params.alignment);
    EXPECT_EQ(layout.get_padded_shape(params.shape), params.padded_shape);

    test_utils::test_tensor_on_device(params.shape, layout);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutLegacyPaddingFromAlignmentTests,
    ::testing::Values(
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{64, 64},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 64, 64}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },

        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{2 * 32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 4, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{3 * 32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{6 * 32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 6, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{8 * 32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 8, 32, 32}
        },

        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{2, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{4, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{16, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },

        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{1, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{2, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{4, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{32, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{3, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{12, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{32 * 3, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{2, 3, 32, 32}
        },

        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{4 * 3 * 32, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{4, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{3 * 3 * 32, 1, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{3, 3, 32, 32}
        },
        LegacyPaddingFromAlignmentTestParams{
            .alignment = tt::tt_metal::Alignment{6 * 3 * 32, 3 * 32, 32, 32},
            .shape = ttnn::SimpleShape{2, 3, 12, 13},
            .padded_shape = ttnn::SimpleShape{6, 3, 32, 32}
        }
    )
);


TEST(TensorLayoutTests, TensorLayout_TinyTiles) {
    using namespace tt::tt_metal;

    const tt::tt_metal::MemoryConfig DefaultMemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt};
    {
        TensorLayout layout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({16, 32})), DefaultMemoryConfig);
        ttnn::SimpleShape shape({2, 3, 10, 16});
        EXPECT_EQ(layout.get_packed_buffer_size_bytes(shape), 3264); // 32 * 16 * 3 * 2 + (32 * 16 * 3 * 2)/16
    }

    {
        ttnn::SimpleShape shape({8, 4, 256, 256});
        Tile tile({16, 16});
        TensorLayout layout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), DefaultMemoryConfig);
        EXPECT_EQ(layout.get_packed_buffer_size_bytes(shape), 2228224); // 256 * 256 * 4 * 8 + (256 * 256 * 4 * 8)/16
    }
}
