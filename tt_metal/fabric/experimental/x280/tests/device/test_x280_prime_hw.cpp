// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Automatic open-time LIM ECC priming, against real silicon.
//
// DESTRUCTIVE, and deliberately in its own binary so it cannot run by
// accident. Priming commits CCACHE0_WAYENABLE, which is increase-only: the
// chip's L2CPU LIM is unavailable until that device is ASIC-reset. The blast
// radius is the L2CPU tiles' own L3 — Tensix, DRAM and Ethernet are untouched.
//
// Refuses to run without an explicit opt-in:
//
//   X280_ALLOW_LIM_ECC_PRIME=1 TT_METAL_RUNTIME_ROOT=<tt-metal>
//   TT_VISIBLE_DEVICES=N x280_prime_test
//
// The reserved device must start genuinely unprimed. Afterwards: tt-smi -r N.
// The chip self-heals — priming re-seeds the marker into the way that reappears
// at LIM base once WayEnable returns to 0.

#include "x280/device/x280_runtime.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <cstdlib>
#include <string>

namespace {

using namespace x280;

bool opted_in() {
    const char* opt = std::getenv("X280_ALLOW_LIM_ECC_PRIME");
    return opt != nullptr && std::string(opt) == "1";
}

}  // namespace

TEST(X280Prime, FirstOpenAutomaticallyPrimesAndDemandsReset) {
    if (!opted_in()) {
        GTEST_SKIP() << "destructive: commits WayEnable on this chip. Set X280_ALLOW_LIM_ECC_PRIME=1 to run.";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(mesh);

    ASSERT_FALSE(opened.runtime.has_value()) << "the first open of an unprimed chip must not return a usable runtime";
    ASSERT_TRUE(opened.requires_reset()) << opened.diagnostic;
    EXPECT_EQ(opened.outcome, RuntimeOpenOutcome::ResetRequired);
    ASSERT_EQ(opened.chips.size(), 1u);
    const auto& chip = opened.chips.front();
    EXPECT_EQ(chip.outcome, RuntimeOpenOutcome::ResetRequired) << chip.diagnostic;
    EXPECT_NE(chip.diagnostic.find("LIM ECC primed"), std::string::npos)
        << "the reset-required result must identify automatic LIM ECC priming";
    printf("runtime open: %s\n", opened.diagnostic.c_str());

    printf("\nThis chip now needs an ASIC reset: tt-smi -r <device>\n");
}
