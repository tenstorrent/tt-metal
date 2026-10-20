// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The pinned input the pipeline_stage members run over: each case has the
// runtime prepare it on the session tile, which happens once per ASIC-reset
// lifetime (docs/runtime.md, "Prepared pipeline inputs").

#ifndef X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP
#define X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/runner.hpp"
#include "harness/session/session.hpp"
#include "x280/device/x280_tile.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

namespace x280::suites::pipeline_stage::pinned_input {

/**
 * Has the runtime prepare the pinned input on `f`'s session tile, with the
 * build's prepare image or X280_PINNED_INPUT_PREPARE_FW_IMAGE. Returns with the
 * region prepared, or records against the running test: a skip when the tile
 * is at the wrong shape or needs an ASIC reset first, a fatal failure for every
 * other outcome. Blocks up to two 3 s timeouts when it prepares.
 */
inline void prepare_or_skip(const x280::harness::session::Fixture& f) {
    const auto result = f.tile->prepare_pinned_input(x280::PinnedInputPrepareRequest{
        .image_path = x280::harness::artifacts::image_path(
            "X280_PINNED_INPUT_PREPARE_FW_IMAGE", "firmware-pinned-input-prepare.bin"),
    });
    const std::string why = x280::describe(result);
    switch (result.outcome) {
        case x280::PinnedInputPrepareOutcome::Prepared:
        case x280::PinnedInputPrepareOutcome::AlreadyPrepared:
            std::printf("chip %d L2CPU %d pinned input: %s\n", f.tile_id.chip_id, f.tile_id.hw_index, why.c_str());
            return;
        case x280::PinnedInputPrepareOutcome::WrongShape:
        case x280::PinnedInputPrepareOutcome::ResetRequired: GTEST_SKIP() << "pinned input unavailable: " << why;
        case x280::PinnedInputPrepareOutcome::Busy:
        case x280::PinnedInputPrepareOutcome::InvalidObservation:
        case x280::PinnedInputPrepareOutcome::InvalidRequest:
        case x280::PinnedInputPrepareOutcome::Failed: FAIL() << "pinned-input preparation: " << why;
    }
    FAIL() << "pinned-input preparation returned an unknown outcome " << static_cast<int>(result.outcome);
}

/** Runner whose cases run over the prepared pinned input. */
template <typename Axis>
class PreparedInputRunner : public x280::harness::Runner<Axis> {
protected:
    void SetUp() override {
        x280::harness::Runner<Axis>::SetUp();
        if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
            return;
        }
        prepare_or_skip(this->fixture());
    }
};

}  // namespace x280::suites::pipeline_stage::pinned_input

#endif  // X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP
