// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Does a unary comparison write every datum it is given? No special values involved.

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {
namespace {

using tt::tt_metal::distributed::MeshDevice;

class UnaryComparisonCoverage : public TTNNFixtureWithSuiteDevice<UnaryComparisonCoverage>,
                                public testing::WithParamInterface<uint32_t> {};

TEST_P(UnaryComparisonCoverage, EveryDatumIsWritten) {
    const uint32_t tiles = GetParam();
    const uint32_t rows = tiles * tt::constants::TILE_HEIGHT;
    const size_t n = static_cast<size_t>(rows) * tt::constants::TILE_WIDTH;

    // every element 1.0, the scalar 2.0: every answer is 0.0, no special value anywhere
    std::vector<::bfloat16> host(n, ::bfloat16(1.0f));
    tt::tt_metal::TensorSpec spec(
        tt::tt_metal::Shape({1, 1, rows, tt::constants::TILE_WIDTH}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));
    const auto input = ttnn::Tensor::from_vector(host, spec).to_device(device_);

    const auto got = ttnn::from_device(ttnn::gt_unary(input, 2.0f)).to_vector<::bfloat16>();

    size_t untouched = 0;
    uint32_t first = 0;
    for (size_t i = 0; i < n; i++) {
        if (static_cast<float>(got[i]) != 0.0f) {
            if (untouched == 0) {
                first = static_cast<uint32_t>(i);
            }
            untouched++;
        }
    }
    EXPECT_EQ(untouched, 0u) << tiles << " tiles: " << untouched << " of " << n
                             << " datums are not 0.0, first at index " << first << " (tile " << (first / 1024)
                             << ", offset " << (first % 1024) << ")";
}

INSTANTIATE_TEST_SUITE_P(
    TileCounts, UnaryComparisonCoverage, testing::Values(1u, 2u, 4u, 7u, 8u, 9u, 16u, 64u), [](const auto& info) {
        return std::to_string(info.param) + "_tiles";
    });

}  // namespace
}  // namespace ttnn::operations::unary::test
