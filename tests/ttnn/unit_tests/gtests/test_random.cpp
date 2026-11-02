// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "ttnn/operations/functions.hpp"

namespace {

// Two draws of 32 values must continue one sequence, i.e. match a single draw of 64.
template <typename T>
void expect_draws_continue_sequence(T low, T high) {
    ttnn::random::seed(0);
    auto draws = ttnn::random::uniform(low, high, ttnn::Shape({1, 32})).template to_vector<T>();
    const auto next = ttnn::random::uniform(low, high, ttnn::Shape({1, 32})).template to_vector<T>();
    draws.insert(draws.end(), next.begin(), next.end());

    ttnn::random::seed(0);
    const auto single = ttnn::random::uniform(low, high, ttnn::Shape({1, 64})).template to_vector<T>();
    EXPECT_EQ(draws, single);
}

TEST(RandomUniform, SuccessiveCallsAdvanceTheGenerator) {
    expect_draws_continue_sequence(0.0f, 1.0f);
    expect_draws_continue_sequence(0u, 100000u);
    expect_draws_continue_sequence(::bfloat16(0.0f), ::bfloat16(1.0f));
    ttnn::random::seed(0);
}

}  // namespace
