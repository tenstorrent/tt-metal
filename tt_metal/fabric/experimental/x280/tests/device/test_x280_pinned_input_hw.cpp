// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One-way pinned-input preparation against real silicon.
//
// Refuses to run without an explicit opt-in:
//
//   X280_ALLOW_PINNED_INPUT_PREPARE=1 TT_METAL_RUNTIME_ROOT=<tt-metal>
//   TT_VISIBLE_DEVICES=N x280_pinned_input_test
//
// The device must have primed LIM. The first run after an ASIC reset applies 14
// LIM ways if every L2CPU is held in reset, boots resident idle on the first
// tile and prepares it; a later run without a reset finds it already prepared.
// Only an ASIC reset (tt-smi -r <device>) undoes either.

#include "x280/device/x280_runtime.hpp"

#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <chrono>
#include <cstdlib>
#include <string>

namespace {

using namespace x280;

bool opted_in() {
    const char* opt = std::getenv("X280_ALLOW_PINNED_INPUT_PREPARE");
    return opt != nullptr && std::string(opt) == "1";
}

}  // namespace

TEST(X280PinnedInputPrepare, RefusesWhileBusyThenPreparesOnce) {
    if (!opted_in()) {
        GTEST_SKIP() << "one-way: pins a cache way until an ASIC reset. Set X280_ALLOW_PINNED_INPUT_PREPARE=1 to run.";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(
        mesh,
        RuntimeOpenConfig{
            .priming_policy = LimEccPrimingPolicy::RequirePrepared,
            .lim_shape = LimShape{.lim_ways = kPreparedLimWays},
        });
    if (!opened.runtime && (opened.requires_preparation() || opened.requires_reset())) {
        GTEST_SKIP() << x280::test::unprepared(opened);
    }
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    auto runtime = std::move(*opened.runtime);
    const auto ids = runtime.tile_ids();
    ASSERT_FALSE(ids.empty());
    x280::test::boot_resident_idle(runtime, ids.front());
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    auto& tile = runtime.tile(ids.front());
    const auto before = tile.pinned_input_state();
    PinnedInputPrepareRequest request;
    request.image_path =
        x280::test::test_image("X280_PINNED_INPUT_PREPARE_FW_IMAGE", "firmware-pinned-input-prepare.bin");

    ActiveRunRequest sentinel;
    sentinel.image_path = x280::test::test_image("X280_ACTIVE_FW_IMAGE", "firmware.bin");
    sentinel.run_value = 0x50494E0B00000001ull;
    const auto enqueued = runtime.enqueue(tile, sentinel);
    ASSERT_TRUE(enqueued.session) << describe(enqueued);
    const auto busy = tile.prepare_pinned_input(request);
    EXPECT_EQ(busy.outcome, PinnedInputPrepareOutcome::Busy) << describe(busy);
    const auto cleaned = runtime.cleanup(enqueued.session, std::chrono::seconds(3));
    ASSERT_TRUE(cleaned.cleaned_up()) << describe(cleaned);
    EXPECT_EQ(tile.pinned_input_state().words, before.words);

    const auto first = tile.prepare_pinned_input(request);
    switch (before.status) {
        case PinnedInputStatus::None: EXPECT_EQ(first.outcome, PinnedInputPrepareOutcome::Prepared); break;
        case PinnedInputStatus::Prepared: EXPECT_EQ(first.outcome, PinnedInputPrepareOutcome::AlreadyPrepared); break;
        default:
            EXPECT_EQ(first.outcome, PinnedInputPrepareOutcome::ResetRequired) << describe(first);
            GTEST_SKIP() << "the record needs an ASIC reset (tt-smi -r <device>): " << describe(before);
    }
    ASSERT_EQ(first.state.status, PinnedInputStatus::Prepared) << describe(first);

    const auto again = tile.prepare_pinned_input(request);
    EXPECT_EQ(again.outcome, PinnedInputPrepareOutcome::AlreadyPrepared) << describe(again);
    EXPECT_FALSE(again.enqueue.has_value());
    EXPECT_EQ(tile.pinned_input_state().status, PinnedInputStatus::Prepared);
    EXPECT_TRUE(tile.inspect().resident_idle_live);
}
