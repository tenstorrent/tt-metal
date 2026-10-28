// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::random::test {

template <typename T>
void expect_zeros_and_ones(tt::tt_metal::DataType dtype) {
    seed(0);
    const std::vector<T> values = random(ttnn::Shape({32, 32}), dtype).to_vector<T>();
    EXPECT_TRUE(std::ranges::all_of(values, [](T v) { return v <= 1; }));
    // A uniform draw from {0, 1}: each value should take roughly half the elements.
    const auto min_count = static_cast<std::ptrdiff_t>(values.size()) * 2 / 5;
    EXPECT_GT(std::ranges::count(values, T{0}), min_count);
    EXPECT_GT(std::ranges::count(values, T{1}), min_count);
}

// Regression: random() returned all zeros for uint8/uint16.
TEST(RandomTest, Uint8DrawsValuesInRange) { expect_zeros_and_ones<uint8_t>(tt::tt_metal::DataType::UINT8); }

TEST(RandomTest, Uint16DrawsValuesInRange) { expect_zeros_and_ones<uint16_t>(tt::tt_metal::DataType::UINT16); }

}  // namespace ttnn::random::test
