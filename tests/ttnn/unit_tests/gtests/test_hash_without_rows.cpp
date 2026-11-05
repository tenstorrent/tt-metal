// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Contract of the row-free program-cache key helpers (ttnn/operations/common/hash_without_rows.hpp): two specs
// that differ only in shape[-2] share a key; any other difference that changes the compiled program separates
// them. Host-only: the helpers read TensorSpecs and host tensors, no device is opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/common/hash_without_rows.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::common {
namespace {

using ::tt::tt_metal::Alignment;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;

TensorSpec make_spec(
    const ttnn::Shape& shape,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const Alignment& alignment = {}) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config, alignment));
}

TEST(HashWithoutRowsTest, RowOnlyChangeSharesKey) {
    const auto a = make_spec(ttnn::Shape({1, 1, 4736, 5376}));
    const auto b = make_spec(ttnn::Shape({1, 1, 13664, 5376}));
    const auto c = make_spec(ttnn::Shape({1, 1, 33, 5376}));  // padded rows change too
    EXPECT_EQ(hash_spec_without_rows(a), hash_spec_without_rows(b));
    EXPECT_EQ(hash_spec_without_rows(a), hash_spec_without_rows(c));
}

TEST(HashWithoutRowsTest, WidthSeparatesKeys) {
    const auto a = make_spec(ttnn::Shape({1, 1, 4736, 5376}));
    const auto b = make_spec(ttnn::Shape({1, 1, 4736, 5376 + 32}));
    EXPECT_NE(hash_spec_without_rows(a), hash_spec_without_rows(b));
}

TEST(HashWithoutRowsTest, UpperDimsAndRankSeparateKeys) {
    const auto rank4 = make_spec(ttnn::Shape({1, 1, 64, 128}));
    const auto batch2 = make_spec(ttnn::Shape({2, 1, 64, 128}));
    const auto heads2 = make_spec(ttnn::Shape({1, 2, 64, 128}));
    const auto rank2 = make_spec(ttnn::Shape({64, 128}));
    const auto rank3 = make_spec(ttnn::Shape({1, 64, 128}));
    EXPECT_NE(hash_spec_without_rows(rank4), hash_spec_without_rows(batch2));
    EXPECT_NE(hash_spec_without_rows(rank4), hash_spec_without_rows(heads2));
    EXPECT_NE(hash_spec_without_rows(rank4), hash_spec_without_rows(rank2));
    EXPECT_NE(hash_spec_without_rows(rank4), hash_spec_without_rows(rank3));
    EXPECT_NE(hash_spec_without_rows(rank2), hash_spec_without_rows(rank3));
}

TEST(HashWithoutRowsTest, DtypeLayoutAndMemoryConfigSeparateKeys) {
    const auto shape = ttnn::Shape({1, 1, 64, 128});
    const auto base = make_spec(shape);
    EXPECT_NE(hash_spec_without_rows(base), hash_spec_without_rows(make_spec(shape, DataType::BFLOAT8_B)));
    EXPECT_NE(
        hash_spec_without_rows(base), hash_spec_without_rows(make_spec(shape, DataType::BFLOAT16, Layout::ROW_MAJOR)));
    EXPECT_NE(
        hash_spec_without_rows(base),
        hash_spec_without_rows(make_spec(shape, DataType::BFLOAT16, Layout::TILE, ttnn::L1_MEMORY_CONFIG)));
}

TEST(HashWithoutRowsTest, AlignmentSeparatesKeysExceptOnRows) {
    const auto shape = ttnn::Shape({1, 1, 64, 128});
    const auto base = make_spec(shape);  // TILE: aligned to the 32x32 tile
    // A wider alignment changes the padded width the accessor compiles in.
    const auto wide = make_spec(shape, DataType::BFLOAT16, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, Alignment({32, 64}));
    EXPECT_NE(hash_spec_without_rows(base), hash_spec_without_rows(wide));
    // The row alignment only changes the row count and stays out of the key.
    const auto tall = make_spec(shape, DataType::BFLOAT16, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, Alignment({64, 32}));
    EXPECT_EQ(hash_spec_without_rows(base), hash_spec_without_rows(tall));
}

TEST(HashWithoutRowsTest, OptionalPresenceSeparatesKeys) {
    const auto spec = make_spec(ttnn::Shape({1, 1, 64, 128}));
    const std::optional<TensorSpec> none = std::nullopt;
    const std::optional<TensorSpec> some = spec;
    EXPECT_NE(hash_spec_without_rows(none), hash_spec_without_rows(some));
    EXPECT_NE(hash_spec_without_rows(none), hash_spec_without_rows(spec));
    EXPECT_EQ(hash_spec_without_rows(some), hash_spec_without_rows(some));
}

TEST(HashWithoutRowsTest, RowsDimPicksTheSkippedDim) {
    // A transposed [B, H, D, S] tensor keeps its sequence in the last dim: rows_dim = -1 skips that one instead.
    const auto a = make_spec(ttnn::Shape({1, 2, 128, 256}));
    const auto longer = make_spec(ttnn::Shape({1, 2, 128, 512}));
    const auto taller = make_spec(ttnn::Shape({1, 2, 256, 256}));
    EXPECT_EQ(hash_spec_without_rows(a, -1), hash_spec_without_rows(longer, -1));
    EXPECT_NE(hash_spec_without_rows(a, -1), hash_spec_without_rows(taller, -1));
    EXPECT_NE(hash_spec_without_rows(a, -1), hash_spec_without_rows(a));
    // A rank-1 spec has no dim -2: nothing is skipped and the length stays in the key.
    EXPECT_NE(
        hash_spec_without_rows(make_spec(ttnn::Shape({64}))), hash_spec_without_rows(make_spec(ttnn::Shape({96}))));
}

TEST(HashWithoutRowsTest, FoldedRowsFoldsUpperDimsAndPadsToTiles) {
    const auto make = [](const ttnn::Shape& shape) {
        const auto spec = make_spec(shape);
        return Tensor::from_vector(std::vector<bfloat16>(spec.logical_shape().volume(), bfloat16(0.0f)), spec);
    };
    EXPECT_EQ(folded_rows(make(ttnn::Shape({1, 1, 64, 128}))), 64u);
    EXPECT_EQ(folded_rows(make(ttnn::Shape({2, 1, 64, 128}))), 128u);
    EXPECT_EQ(folded_rows(make(ttnn::Shape({1, 3, 64, 128}))), 192u);
    EXPECT_EQ(folded_rows(make(ttnn::Shape({64, 128}))), 64u);
    // 33 logical rows pad to 64 per upper index, and padded widths do not leak into the row count.
    EXPECT_EQ(folded_rows(make(ttnn::Shape({2, 1, 33, 100}))), 128u);
    // The spec overload: a ROW_MAJOR layout folds the upper dims without tile padding.
    const auto shape = ttnn::Shape({2, 1, 70, 64});
    EXPECT_EQ(folded_rows(make_spec(shape, DataType::BFLOAT16, Layout::ROW_MAJOR)), 140u);
    EXPECT_EQ(folded_rows(make_spec(shape, DataType::BFLOAT16, Layout::TILE)), 192u);
    EXPECT_EQ(folded_rows(make_spec(ttnn::Shape({64}), DataType::BFLOAT16, Layout::TILE)), 32u);
}

TEST(HashWithoutRowsTest, PositiveRowsDimIsRefused) {
    const auto spec = make_spec(ttnn::Shape({1, 1, 64, 128}));
    EXPECT_ANY_THROW(hash_spec_without_rows(spec, 2));
    EXPECT_ANY_THROW(hash_spec_without_rows(spec, 0));
}

}  // namespace
}  // namespace ttnn::operations::common
