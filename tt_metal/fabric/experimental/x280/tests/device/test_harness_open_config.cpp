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

TEST(X280RuntimeOpenConfig, NeverPrimesAndPinsThePreparedProfileExactly) {
    using x280::harness::session::OpenConfig;
    using x280::harness::session::runtime_open_config;
    const x280::LimShape prepared{.lim_ways = x280::kPreparedLimWays};

    const auto plain = runtime_open_config(OpenConfig{});
    EXPECT_EQ(plain.priming_policy, x280::LimEccPrimingPolicy::RequirePrepared);
    EXPECT_EQ(plain.lim_shape, x280::LimShape{});
    EXPECT_EQ(plain.shape_change_policy, x280::LimShapeChangePolicy::ApplyWhileReset);
    EXPECT_EQ(plain.lim_layout, x280::LimLayout::Legacy);

    const auto suite = runtime_open_config(OpenConfig{.prepared_suite_requested = true});
    EXPECT_EQ(suite.priming_policy, x280::LimEccPrimingPolicy::RequirePrepared);
    EXPECT_EQ(suite.lim_shape, prepared);
    EXPECT_EQ(suite.shape_change_policy, x280::LimShapeChangePolicy::RequireExact);

    const auto injection = runtime_open_config(OpenConfig{.lim_layout = x280::LimLayout::PacketInjection});
    EXPECT_EQ(injection.priming_policy, x280::LimEccPrimingPolicy::RequirePrepared);
    EXPECT_EQ(injection.lim_shape, prepared);
    EXPECT_EQ(injection.shape_change_policy, x280::LimShapeChangePolicy::RequireExact);
    EXPECT_EQ(injection.lim_layout, x280::LimLayout::PacketInjection);
}

}  // namespace
