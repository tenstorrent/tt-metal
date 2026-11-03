// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One-way cache/LIM repartitioning against real silicon.
//
// Refuses to run without an explicit opt-in:
//
//   X280_ALLOW_LIM_SHAPE_CHANGE=1 TT_METAL_RUNTIME_ROOT=<tt-metal>
//   TT_VISIBLE_DEVICES=N x280_shape_test
//
// The device must have primed LIM and every L2CPU held in reset. This test
// leaves eleven LIM ways and five cache ways. Restore the default fifteen-way LIM
// shape with an ASIC reset after the run.

#include "x280/device/x280_runtime.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <cstdlib>
#include <string>

namespace {

using namespace x280;

bool opted_in() {
    const char* opt = std::getenv("X280_ALLOW_LIM_SHAPE_CHANGE");
    return opt != nullptr && std::string(opt) == "1";
}

}  // namespace

TEST(X280Shape, PreparedOpenAppliesRequestedShapeWhileResetHeld) {
    if (!opted_in()) {
        GTEST_SKIP() << "one-way: increases WayEnable on this chip. Set X280_ALLOW_LIM_SHAPE_CHANGE=1 to run.";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    constexpr LimShape requested{.lim_ways = 11};
    auto opened = X280Runtime::open(
        mesh,
        RuntimeOpenConfig{
            .priming_policy = LimEccPrimingPolicy::RequirePrepared,
            .lim_shape = requested,
        });

    if (!opened.runtime && opened.requires_preparation()) {
        GTEST_SKIP() << opened.diagnostic << "; prime and ASIC-reset the device before running this test";
    }
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    EXPECT_EQ(opened.runtime->lim_shape(), requested);

    for (const auto& id : opened.runtime->tile_ids()) {
        const auto state = opened.runtime->tile(id).inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        EXPECT_EQ(state.raw_wayenable & 0xFu, 4u);
        EXPECT_EQ(state.lim_backed_bytes, 11u * kLimWayBytes);
        EXPECT_EQ(state.ecc, LimEccState::Primed);
    }

    printf("\nRestore the default LIM shape before unrelated work: tt-smi -r\n");
}
