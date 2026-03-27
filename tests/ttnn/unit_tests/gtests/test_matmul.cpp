// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <array>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::matmul::test {

// ============================================================
// Test: Single-Tile Matmul (1x1 tiles = 32x32)
// ones(32,32) * full(32,32, 0.5) = full(32,32, 16.0)
// ============================================================

class SingleTileMatmulFixture : public TTNNFixtureWithSuiteDevice<SingleTileMatmulFixture> {};

TEST_F(SingleTileMatmulFixture, SingleTileMatmul) {
    auto& device = *device_;

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    // A = all ones, B = all 0.5
    // C = A * B => each element = sum of 32 * (1.0 * 0.5) = 16.0
    const auto a_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_tensor = ttnn::full(shape, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    const auto c_tensor = ttnn::operations::matmul::matmul(a_tensor, b_tensor);

    const auto expected = ttnn::full(shape, 16.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    TT_FATAL(
        ttnn::allclose<::bfloat16>(ttnn::from_device(expected), ttnn::from_device(c_tensor), 0.5f, 0.05f),
        "Matmul result does not match expected");
}

// ============================================================
// Test: Multi-Tile Matmul (2x2 tiles = 64x64)
// ones(64,64) * full(64,64, 0.25) = full(64,64, 16.0)
// ============================================================

class MultiTileMatmulFixture : public TTNNFixtureWithSuiteDevice<MultiTileMatmulFixture> {};

TEST_F(MultiTileMatmulFixture, MultiTileMatmul) {
    auto& device = *device_;

    std::array<uint32_t, 4> dims = {1, 1, 64, 64};
    ttnn::Shape shape(dims);

    // A = all ones, B = all 0.25
    // C = A * B => each element = sum of 64 * (1.0 * 0.25) = 16.0
    const auto a_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_tensor = ttnn::full(shape, 0.25f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    const auto c_tensor = ttnn::operations::matmul::matmul(a_tensor, b_tensor);

    const auto expected = ttnn::full(shape, 16.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    TT_FATAL(
        ttnn::allclose<::bfloat16>(ttnn::from_device(expected), ttnn::from_device(c_tensor), 0.5f, 0.05f),
        "Multi-tile matmul result does not match expected");
}

}  // namespace ttnn::operations::matmul::test
