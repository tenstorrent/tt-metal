// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
#include "ttnn_test_fixtures.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/layout/alignment.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace ttnn;

namespace {
const MemoryConfig DefaultMemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt};
const MemoryConfig L1MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1, std::nullopt};

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

struct RowMajorPaddedShapeAlignmentTestParams {
    Shape shape;
    Shape padded_shape;
    tt::tt_metal::Alignment expected_alignment;
    tt::tt_metal::Shape2D expected_physical_shape;
    tt::tt_metal::Strides expected_strides;
};

class TensorLayoutRowMajorPaddedShapeAlignmentTests
    : public ::testing::TestWithParam<RowMajorPaddedShapeAlignmentTestParams> {};

TEST_P(TensorLayoutRowMajorPaddedShapeAlignmentTests, FromPaddedShape_ComputesFullAlignment) {
    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromPaddedShape(
        DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig, params.shape, params.padded_shape);

    EXPECT_EQ(layout.get_alignment(), params.expected_alignment);
    EXPECT_EQ(layout.compute_physical_shape(params.shape), params.expected_physical_shape);
    EXPECT_EQ(layout.compute_strides(params.shape), params.expected_strides);
    EXPECT_EQ(layout.compute_padded_shape(params.shape), params.padded_shape);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutRowMajorPaddedShapeAlignmentTests,
    ::testing::Values(
        RowMajorPaddedShapeAlignmentTestParams{
            .shape = Shape{30, 20, 10},
            .padded_shape = Shape{32, 32, 32},
            .expected_alignment = tt::tt_metal::Alignment({1024, 32, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{1024, 32},
            .expected_strides = tt::tt_metal::Strides({640, 32, 1})},
        RowMajorPaddedShapeAlignmentTestParams{
            .shape = Shape{2, 3, 16, 16},
            .padded_shape = Shape{2, 16, 32, 32},
            .expected_alignment = tt::tt_metal::Alignment({1024, 512, 32, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{1024, 32},
            .expected_strides = tt::tt_metal::Strides({1536, 512, 32, 1})}));

// Regression tests for `legacyShapeToAlignment` on TILE-layout tensors that
// take the "INTERLEAVED with (deprecated) non-height/width padding" branch
// (i.e. logical and padded shapes differ on outer dimensions).
//
// Before the fix in tensor_layout.cpp, the outer-dimension alignment cascade
// was seeded with the substituted tile dimensions for TILE layout, causing
// `compute_physical_shape` to return a size smaller than the legacy padded
// volume whenever `padded[-2] > tile_height`. That smaller size was used to
// allocate device buffers, while host-side buffers were still packed for the
// original legacy padded volume, leading to a buffer overflow during
// `enqueue_write_tensor` (TT_FATAL in buffer.cpp: region.offset + region.size
// <= size()).
//
// These cases pick parameters where `alignment_can_be_2D == false` and
// `padded[-2]` exceeds the 32x32 tile dims, so the buggy cascade yields a
// strictly smaller physical shape than the fixed cascade.
struct TilePaddedAlignmentTestParams {
    Shape shape;
    Shape padded_shape;
    DataType dtype;
    tt::tt_metal::Alignment expected_alignment;
    tt::tt_metal::Shape2D expected_physical_shape;
};

class TensorLayoutTilePaddedAlignmentTests : public ::testing::TestWithParam<TilePaddedAlignmentTestParams> {};

TEST_P(TensorLayoutTilePaddedAlignmentTests, Tensor_TilePaddedAlignmentRegression) {
    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromPaddedShape(
        params.dtype, Layout::TILE, DefaultMemoryConfig, params.shape, params.padded_shape);

    EXPECT_EQ(layout.get_alignment(), params.expected_alignment);

    // The physical shape must match the expected value: it reflects the outer
    // dimension cumulative alignment correctly carrying the original padded
    // dims, not the substituted tile dims. A too-small physical shape is the
    // direct signature of the bug.
    EXPECT_EQ(layout.compute_physical_shape(params.shape), params.expected_physical_shape);

    // The physical volume must cover the legacy padded volume,
    // otherwise a host buffer packed for the legacy padded shape would not
    // fit into the device allocation.
    const size_t physical_volume =
        static_cast<size_t>(params.expected_physical_shape.height()) * params.expected_physical_shape.width();
    EXPECT_EQ(physical_volume, params.padded_shape.volume());

    // Roundtrip: compute_padded_shape on the logical shape must recover the
    // originally supplied legacy padded shape.
    EXPECT_EQ(layout.compute_padded_shape(params.shape), params.padded_shape);

    // End-to-end: allocating a device tensor with this spec and writing a
    // host buffer of `compute_packed_buffer_size_bytes` into it must succeed.
    // (With the buggy alignment the device allocation would be smaller than
    // the host buffer, triggering the buffer.cpp TT_FATAL.)
    test_utils::test_tensor_on_device(params.shape, layout);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutTilePaddedAlignmentTests,
    ::testing::Values(
        // Minimal reproducer: outer dim N padded from 1 -> 2, padded[-2]=64 > tile_h=32.
        TilePaddedAlignmentTestParams{
            .shape = Shape{1, 2, 60, 30},
            .padded_shape = Shape{2, 2, 64, 32},
            .dtype = DataType::BFLOAT16,
            .expected_alignment = tt::tt_metal::Alignment({256, 128, 32, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{256, 32}},
        // Overpadding on H/W.
        TilePaddedAlignmentTestParams{
            .shape = Shape{1, 2, 18, 15},
            .padded_shape = Shape{2, 2, 64, 32},
            .dtype = DataType::BFLOAT16,
            .expected_alignment = tt::tt_metal::Alignment({256, 128, 64, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{256, 32}},
        // Padding on C (dim -3) while N stays equal; padded[-2]=64 > 32.
        TilePaddedAlignmentTestParams{
            .shape = Shape{3, 1, 60, 30},
            .padded_shape = Shape{3, 2, 64, 32},
            .dtype = DataType::BFLOAT16,
            .expected_alignment = tt::tt_metal::Alignment({384, 128, 32, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{384, 32}},

        // padded[-2]=96 > 32 and outer-dim padding on N.
        TilePaddedAlignmentTestParams{
            .shape = Shape{1, 3, 90, 30},
            .padded_shape = Shape{2, 3, 96, 32},
            .dtype = DataType::BFLOAT16,
            .expected_alignment = tt::tt_metal::Alignment({576, 288, 32, 32}),
            .expected_physical_shape = tt::tt_metal::Shape2D{576, 32}}));

// `alignment_can_be_2D` branch for TILE: when legacy H/W padding is no larger than the minimum required for
// tile layout on each of the last two dimensions, legacyShapeToAlignment uses tile height/width (not the
// padded H/W) so the layout matches default TILE alignment. If a dimension is overpadded, that edge is kept.
struct Tile2DInterleavedAlignmentTestParams {
    Shape shape;
    Shape padded_shape;
    Tile tile;
    tt::tt_metal::Alignment expected_alignment;
};

class TensorLayoutTile2DInterleavedAlignmentTests
    : public ::testing::TestWithParam<Tile2DInterleavedAlignmentTestParams> {};

TEST_P(TensorLayoutTile2DInterleavedAlignmentTests, FromPaddedShape_TileAlignmentUsesTileUnlessOverpadded) {
    const auto& params = GetParam();
    TensorLayout layout = TensorLayout::fromPaddedShape(
        DataType::BFLOAT16,
        PageConfig(Layout::TILE, params.tile),
        DefaultMemoryConfig,
        params.shape,
        params.padded_shape);

    EXPECT_EQ(layout.get_alignment(), params.expected_alignment);
    // Padded shape roundtrip: alignment must still recover the supplied legacy padded shape.
    EXPECT_EQ(layout.compute_padded_shape(params.shape), params.padded_shape);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    TensorLayoutTile2DInterleavedAlignmentTests,
    ::testing::Values(
        // 2D: minimum tile padding only -> alignment is tile 32x32, not 32x64 from padded H/W.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{30, 60},
            .padded_shape = Shape{32, 64},
            .expected_alignment = tt::tt_metal::Alignment({32, 32}),
        },
        // Rank-3: only H/W differ from logical; same rule.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{1, 30, 60},
            .padded_shape = Shape{1, 32, 64},
            .expected_alignment = tt::tt_metal::Alignment({32, 32}),
        },
        // Width overpadded beyond minimum for logical width -> keep legacy padded width in alignment.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{30, 60},
            .padded_shape = Shape{32, 96},
            .expected_alignment = tt::tt_metal::Alignment({32, 96}),
        },
        // Height overpadded (e.g. 30 -> 64) -> keep that edge; width still uses tile when not overpadded.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{30, 60},
            .padded_shape = Shape{64, 64},
            .expected_alignment = tt::tt_metal::Alignment({64, 32}),
        },
        // Both H/W overpadded beyond minimum tile padding -> keep both legacy padded edges.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{30, 60},
            .padded_shape = Shape{64, 96},
            .expected_alignment = tt::tt_metal::Alignment({64, 96}),
        },
        // Custom tile: minimum tile padding should use the custom tile H/W, not the legacy padded H/W.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{15, 60},
            .padded_shape = Shape{16, 64},
            .tile = Tile({16, 32}),
            .expected_alignment = tt::tt_metal::Alignment({16, 32}),
        },
        // Custom tile with both H/W overpadded -> preserve overpadded dimensions.
        Tile2DInterleavedAlignmentTestParams{
            .shape = Shape{15, 60},
            .padded_shape = Shape{32, 96},
            .tile = Tile({16, 32}),
            .expected_alignment = tt::tt_metal::Alignment({32, 96}),
        }));

struct ConsumedMemoryBytesPerBankTestParams {
    Shape shape;
    TensorLayout tensor_layout;
    size_t expected_consumed_memory_bytes_per_bank = 0;
};

class ConsumedMemoryBytesPerBankTests : public ::testing::TestWithParam<ConsumedMemoryBytesPerBankTestParams> {};

TEST_P(ConsumedMemoryBytesPerBankTests, TestConsumedMemoryBytesPerBank) {
    const auto& params = GetParam();

    size_t page_alignment = 0;
    size_t num_banks = 0;
    if (params.tensor_layout.get_memory_config().buffer_type() == BufferType::L1) {
        num_banks = 64;
        page_alignment = 16;
    } else {
        num_banks = 12;
        page_alignment = 32;
    }

    EXPECT_EQ(
        params.tensor_layout.compute_consumed_memory_bytes_per_bank(params.shape, page_alignment, num_banks),
        params.expected_consumed_memory_bytes_per_bank);
}

INSTANTIATE_TEST_SUITE_P(
    TensorLayoutTests,
    ConsumedMemoryBytesPerBankTests,
    ::testing::Values(
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{1, 1},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{12, 1},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{12, 32},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{13, 32},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 64,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{13, 33},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 128,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{1, 1},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{12, 1},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{12, 16},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{13, 16},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 64,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{13, 17},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), DefaultMemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 128,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{1, 1},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{64, 1},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{64, 16},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{65, 16},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{65, 17},
            .tensor_layout = TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 64,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{1, 1},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{64, 1},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{64, 8},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 16,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{65, 8},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{65, 9},
            .tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), L1MemoryConfig),
            .expected_consumed_memory_bytes_per_bank = 64,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{7 * 32, 7 * 32},
            .tensor_layout = TensorLayout(
                DataType::UINT8,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::DRAM,
                    ShardSpec(CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{12, 0}}), Shape2D{32, 7 * 32}))),
            .expected_consumed_memory_bytes_per_bank = 32 * 7 * 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{7 * 32, 7 * 32},
            .tensor_layout = TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::DRAM,
                    ShardSpec(CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{12, 0}}), Shape2D{32, 7 * 32}))),
            .expected_consumed_memory_bytes_per_bank = 32 * 7 * 32 * 2,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{125 * 32, 125 * 32},
            .tensor_layout = TensorLayout(
                DataType::UINT8,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    BufferType::L1,
                    NdShardSpec{
                        .shard_shape = Shape{2 * 32, 2 * 32},
                        .grid = CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}),
                    })),
            .expected_consumed_memory_bytes_per_bank = 63 * 2 * 32 * 2 * 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{125 * 32, 125 * 32},
            .tensor_layout = TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    BufferType::L1,
                    NdShardSpec{
                        .shard_shape = Shape{2 * 32, 2 * 32},
                        .grid = CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}),
                    })),
            .expected_consumed_memory_bytes_per_bank = 63 * 2 * 32 * 2 * 32 * 2,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{5, 17, 4 * 32, 4 * 32},
            .tensor_layout = TensorLayout(
                DataType::UINT8,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    BufferType::L1,
                    NdShardSpec{
                        .shard_shape = Shape{5, 1, 32, 32},
                        .grid = CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}),
                    })),
            .expected_consumed_memory_bytes_per_bank = 5 * 5 * 32 * 32,
        },
        ConsumedMemoryBytesPerBankTestParams{
            .shape = Shape{5, 17, 4 * 32, 4 * 32},
            .tensor_layout = TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::TILE),
                MemoryConfig(
                    BufferType::L1,
                    NdShardSpec{
                        .shard_shape = Shape{5, 1, 32, 32},
                        .grid = CoreRangeSet(CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}),
                    })),
            .expected_consumed_memory_bytes_per_bank = 5 * 5 * 32 * 32 * 2,
        }));
