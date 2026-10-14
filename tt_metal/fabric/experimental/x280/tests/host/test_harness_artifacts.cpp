// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for the strict numeric environment knobs of harness/artifacts.

#include "harness/artifacts/artifacts.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

namespace {

namespace hart = x280::harness::artifacts;
using x280::testing::ScopedEnv;

constexpr const char* kKnob = "X280_HARNESS_TEST_KNOB";

}  // namespace

TEST(HarnessArtifacts, EnvU32ReadsADecimalAndFallsBackOnlyWhenUnsetOrEmpty) {
    {
        ScopedEnv env(kKnob, nullptr);
        EXPECT_FALSE(hart::env_u32(kKnob).has_value());
        EXPECT_EQ(hart::env_u32(kKnob, 7u), 7u);
    }
    {
        ScopedEnv env(kKnob, "");
        EXPECT_EQ(hart::env_u32(kKnob, 7u), 7u);
    }
    {
        ScopedEnv env(kKnob, "4294967295");
        EXPECT_EQ(hart::env_u32(kKnob, 7u), UINT32_MAX);
    }
}

TEST(HarnessArtifacts, EnvU32RefusesMalformedOrOutOfRangeValues) {
    for (const char* bad : {"12x", "x12", "0x10", "-1", "+1", " 1", "1 ", "4294967296"}) {
        SCOPED_TRACE(bad);
        ScopedEnv env(kKnob, bad);
        EXPECT_THROW((void)hart::env_u32(kKnob), std::runtime_error);
        EXPECT_THROW((void)hart::env_u32(kKnob, 7u), std::runtime_error);
    }
}
