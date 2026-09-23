// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/functions.hpp"

namespace {

TEST(RandomUniform, SuccessiveCallsAdvanceTheGenerator) {
    const ttnn::Shape shape({1, 32});
    ttnn::random::seed(0);
    const auto first = ttnn::random::uniform(0.0f, 1.0f, shape).to_vector<float>();
    const auto second = ttnn::random::uniform(0.0f, 1.0f, shape).to_vector<float>();
    EXPECT_NE(first, second);

    ttnn::random::seed(0);
    EXPECT_EQ(ttnn::random::uniform(0.0f, 1.0f, shape).to_vector<float>(), first);

    const auto first_int = ttnn::random::uniform(0u, 100000u, shape).to_vector<uint32_t>();
    const auto second_int = ttnn::random::uniform(0u, 100000u, shape).to_vector<uint32_t>();
    EXPECT_NE(first_int, second_int);

    const auto first_bf16 = ttnn::random::uniform(::bfloat16(0.0f), ::bfloat16(1.0f), shape).to_vector<::bfloat16>();
    const auto second_bf16 = ttnn::random::uniform(::bfloat16(0.0f), ::bfloat16(1.0f), shape).to_vector<::bfloat16>();
    EXPECT_NE(first_bf16, second_bf16);

    ttnn::random::seed(0);
}

}  // namespace
