// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "harness/session/session.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

TEST(X280PllMhzFromEnv, DefaultsTo1750AndRefusesAnUnsupportedValue) {
    using x280::harness::session::x280_pll_mhz_from_env;
    {
        x280::testing::ScopedEnv env("X280_PLL_MHZ", nullptr);
        EXPECT_EQ(x280_pll_mhz_from_env(), 1750u);
    }
    {
        x280::testing::ScopedEnv env("X280_PLL_MHZ", "800");
        EXPECT_EQ(x280_pll_mhz_from_env(), 800u);
    }
    {
        x280::testing::ScopedEnv env("X280_PLL_MHZ", "17");
        EXPECT_THROW((void)x280_pll_mhz_from_env(), std::invalid_argument);
    }
}

}  // namespace
