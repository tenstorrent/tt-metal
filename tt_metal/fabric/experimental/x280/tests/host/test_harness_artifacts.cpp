// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for the image lookup and the strict numeric environment knobs of
// harness/artifacts.

#include "harness/artifacts/artifacts.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

TEST(HarnessArtifacts, ImagePathTakesTheVariableOverTheBuildImage) {
    constexpr const char* kImage = "X280_HARNESS_TEST_FW_IMAGE";
    const std::string build_image = std::string(X280_IMAGE_DIR) + "/firmware-test.bin";
    {
        ScopedEnv env(kImage, nullptr);
        EXPECT_EQ(hart::image_path(kImage, "firmware-test.bin"), build_image);
    }
    {
        ScopedEnv env(kImage, "");
        EXPECT_EQ(hart::image_path(kImage, "firmware-test.bin"), build_image);
    }
    {
        ScopedEnv env(kImage, "/elsewhere/image.bin");
        EXPECT_EQ(hart::image_path(kImage, "firmware-test.bin"), "/elsewhere/image.bin");
    }
}
