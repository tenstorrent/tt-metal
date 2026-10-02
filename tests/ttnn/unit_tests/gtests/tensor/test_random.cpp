// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::random::test {

template <typename T>
void expect_zeros_and_ones(tt::tt_metal::DataType dtype) {
    seed(0);
    const std::vector<T> values = random(ttnn::Shape({32, 32}), dtype).to_vector<T>();
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](T v) { return v <= 1; }));
    EXPECT_TRUE(std::any_of(values.begin(), values.end(), [](T v) { return v == 1; }));
    EXPECT_TRUE(std::any_of(values.begin(), values.end(), [](T v) { return v == 0; }));
}

// Regression: random() returned all zeros for uint8/uint16.
TEST(RandomTest, Uint8DrawsValuesInRange) { expect_zeros_and_ones<uint8_t>(tt::tt_metal::DataType::UINT8); }

TEST(RandomTest, Uint16DrawsValuesInRange) { expect_zeros_and_ones<uint16_t>(tt::tt_metal::DataType::UINT16); }

}  // namespace ttnn::random::test
