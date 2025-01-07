// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/scoped.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

TEST(ScopedTest, Scoped) {
    int variable = 0;

    {
        EXPECT_EQ(variable, 0);
        auto scoped = ttml::core::Scoped([&variable]() { variable = 1; }, [&variable]() { variable = 2; });
        EXPECT_EQ(variable, 1);
    }

    EXPECT_EQ(variable, 2);
};
