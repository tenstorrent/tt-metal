// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <tt-metalium/tilize_untilize.hpp>

template <bool tilize_first, typename T>
void tilize_untilize_helper(
    uint max_num_batches, uint max_num_row_tiles, uint max_num_col_tiles, uint TILE_HEIGHT, uint TILE_WIDTH) {
    for (uint i = 1; i <= max_num_batches; i++) {
        for (uint nrows = TILE_HEIGHT; nrows <= max_num_row_tiles * TILE_HEIGHT; nrows += TILE_HEIGHT) {
            for (uint ncols = TILE_WIDTH; ncols <= max_num_col_tiles * TILE_WIDTH; ncols += TILE_WIDTH) {
                // Create bfloat16 arange
                std::vector<T> data;
                for (float datum = 0; datum < i * nrows * ncols; datum++) {
                    data.push_back(datum);
                }

                std::vector<T> target = data;
                if constexpr (tilize_first) {
                    tilize(data, nrows, ncols);
                    ASSERT_FALSE(data == target);
                    untilize(data, nrows, ncols);
                } else {
                    untilize(data, nrows, ncols);
                    ASSERT_FALSE(data == target);
                    tilize(data, nrows, ncols);
                }
                ASSERT_TRUE(data == target);
            }
        }
    }
}

// The following run the tilize/untilize APIs and their inverses
TEST(Host, TestTilizeAndThenUntilizeBfloat16) {
    uint max_num_batches = 8;
    uint max_num_row_tiles = 8;
    uint max_num_col_tiles = 8;
    uint TILE_HEIGHT = 32;
    uint TILE_WIDTH = 32;

    tilize_untilize_helper<true, bfloat16>(
        max_num_batches, max_num_row_tiles, max_num_col_tiles, TILE_HEIGHT, TILE_WIDTH);
}

TEST(Host, TestTilizeThrowErrorForNonBfloat16DataType) {
    std::vector<float> vec(1024, 0);
    EXPECT_ANY_THROW(tilize(vec, 32, 32));
}

TEST(Host, TestTilizeThrowErrorForInvalidTileMandN) {
    // m and n are not divisible by tile size
    std::vector<bfloat16> vec(16, 0);
    EXPECT_ANY_THROW(tilize(vec, 4, 4));  // m and n not divisible by 32
    EXPECT_ANY_THROW(tilize(vec, 0, 4));  // Cannot have 0 shapes
    EXPECT_ANY_THROW(tilize(vec, 4, 0));
    EXPECT_ANY_THROW(tilize(vec, 0, 0));
}

TEST(Host, TestTilizeThrowErrorForInvalidVectorShape) {
    std::vector<bfloat16> vec(16, 0);       // Size not divisible by 1024
    EXPECT_ANY_THROW(tilize(vec, 32, 32));  // m and n not divisible by 32
    vec = {};                               // Cannot have a zero vector either
    EXPECT_ANY_THROW(tilize(vec, 32, 32));  // m and n not divisible by 32
}

TEST(Host, TestUntilizeThrowErrorForNonBfloat16DataType) {
    std::vector<float> vec(1024, 0);
    EXPECT_ANY_THROW(untilize(vec, 32, 32));
}

TEST(Host, TestUntilizeThrowErrorForInvalidTileMandN) {
    // m and n are not divisible by tile side lengths
    std::vector<bfloat16> vec(16, 0);
    EXPECT_ANY_THROW(untilize(vec, 4, 4));
    EXPECT_ANY_THROW(untilize(vec, 0, 4));
    EXPECT_ANY_THROW(untilize(vec, 4, 0));
    EXPECT_ANY_THROW(untilize(vec, 0, 0));
}

TEST(Host, TestUntilizeThrowErrorForInvalidVectorShape) {
    std::vector<bfloat16> vec(16, 0);         // Size not divisible by 1024
    EXPECT_ANY_THROW(untilize(vec, 32, 32));  // m and n not divisible by 32
    vec = {};                                 // Cannot have a zero vector either
    EXPECT_ANY_THROW(untilize(vec, 32, 32));  // m and n not divisible by 32
}

TEST(Host, TestUntilizeAndThenTilizeBfloat16) {
    uint max_num_batches = 8;
    uint max_num_row_tiles = 8;
    uint max_num_col_tiles = 8;
    uint TILE_HEIGHT = 32;
    uint TILE_WIDTH = 32;

    tilize_untilize_helper<false, bfloat16>(
        max_num_batches, max_num_row_tiles, max_num_col_tiles, TILE_HEIGHT, TILE_WIDTH);
}
