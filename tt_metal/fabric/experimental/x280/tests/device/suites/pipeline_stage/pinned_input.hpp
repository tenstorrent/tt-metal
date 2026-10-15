// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The pinned-input epoch the pipeline_stage members run in: the suite prepares
// the cached Zero Device region once per process, runs its cases, then tears it
// down (docs/runtime.md, "Prepared pipeline inputs").

#ifndef X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP
#define X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/runner.hpp"
#include "harness/session/session.hpp"
#include "x280/device/x280_runtime.hpp"

#include "rt/x280_pinned_input.h"

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <string>

namespace x280::suites::pipeline_stage::pinned_input {

enum class Operation : uint64_t {
    Prepare = 1,
    Teardown = 2,
};

inline const char* operation_name(Operation op) { return op == Operation::Prepare ? "prepare" : "teardown"; }

/** The build's image for `op`, or its X280_PINNED_INPUT_{PREPARE,TEARDOWN}_FW_IMAGE override. */
inline std::string operation_image(Operation op) {
    return op == Operation::Prepare ? x280::harness::artifacts::image_path(
                                          "X280_PINNED_INPUT_PREPARE_FW_IMAGE", "firmware-pinned-input-prepare.bin")
                                    : x280::harness::artifacts::image_path(
                                          "X280_PINNED_INPUT_TEARDOWN_FW_IMAGE", "firmware-pinned-input-teardown.bin");
}

/**
 * Success when `runtime` owns no active session and `id` reads healthy and in
 * live resident idle, the precondition of either operation; otherwise a
 * failure carrying the tile's state.
 */
inline ::testing::AssertionResult tile_idle(x280::X280Runtime& runtime, const x280::TileId& id) {
    if (!runtime.sessions().empty()) {
        return ::testing::AssertionFailure()
               << "pinned-input operation refused: the runtime still owns active sessions";
    }
    auto& tile = runtime.tile(id);
    const auto state = tile.inspect();
    if (!state.observation_valid || !state.resident_idle_live || !tile.healthy()) {
        return ::testing::AssertionFailure() << "pinned-input operation refused: chip " << id.chip_id << " L2CPU "
                                             << id.hw_index << " is not confirmed healthy resident idle\n"
                                             << x280::describe(state);
    }
    return ::testing::AssertionSuccess();
}

/**
 * Runs `op`'s image on `id` as an active session and cleans it up. Success
 * needs hart 0 to report PREPARED or TORN_DOWN, harts 1-3 NOOP, a confirmed
 * cleanup and the tile back in live resident idle. A failure carries the
 * wait, cleanup and final-state diagnostics; it leaves the region in an
 * unknown state, so the epoch is over. Blocks up to three 3 s timeouts.
 */
inline ::testing::AssertionResult run_operation(x280::X280Runtime& runtime, const x280::TileId& id, Operation op) {
    auto& tile = runtime.tile(id);
    x280::ActiveRunRequest request;
    request.image_path = operation_image(op);
    request.run_value =
        UINT64_C(0x50494E0000000000) | (static_cast<uint64_t>(op) << 32) | static_cast<uint64_t>(id.hw_index + 1);
    request.cleanup_timeout = std::chrono::milliseconds(3000);
    const char* name = operation_name(op);

    const auto enqueued = runtime.enqueue(tile, request);
    if (!enqueued.enqueued()) {
        std::ostringstream why;
        why << "chip " << id.chip_id << " L2CPU " << id.hw_index << " pinned-input " << name
            << " enqueue failed: " << x280::describe(enqueued);
        if (enqueued.session) {
            why << "\ncleanup after uncertain enqueue: "
                << x280::describe(runtime.cleanup(enqueued.session, request.cleanup_timeout));
        }
        return ::testing::AssertionFailure() << why.str();
    }

    const auto wait = runtime.session(enqueued.session).wait_until_result_ready(request.cleanup_timeout);
    bool firmware_ok = wait.outcome == x280::WaitOutcome::ResultReady;
    const uint64_t expected_hart0 =
        op == Operation::Prepare ? X280_PINNED_INPUT_RESULT_PREPARED : X280_PINNED_INPUT_RESULT_TORN_DOWN;
    for (uint32_t hart = 0; hart < x280::kHartsPerTile; ++hart) {
        const uint64_t expected = hart == 0 ? expected_hart0 : X280_PINNED_INPUT_RESULT_NOOP_HART;
        firmware_ok =
            firmware_ok && wait.hartstates[hart].observation_valid && wait.hartstates[hart].state.result == expected;
    }

    const auto cleanup = runtime.cleanup(enqueued.session, request.cleanup_timeout);
    const auto final_state = tile.inspect();
    const bool stopped = cleanup.cleaned_up() && final_state.observation_valid && final_state.resident_idle_live;
    if (!firmware_ok || !stopped) {
        return ::testing::AssertionFailure()
               << "chip " << id.chip_id << " L2CPU " << id.hw_index << " pinned-input " << name
               << " failed\nwait: " << x280::describe(wait) << "\ncleanup: " << x280::describe(cleanup) << "\nfinal:\n"
               << x280::describe(final_state);
    }
    std::printf(
        "chip %d L2CPU %d pinned-input %s completed and all harts returned to resident idle\n",
        id.chip_id,
        id.hw_index,
        name);
    return ::testing::AssertionSuccess();
}

/**
 * The process's pinned-input epoch on the session tile. prepare() runs from
 * each case's SetUp and acts once; teardown() runs once after the suite's
 * last case, whatever the cases' outcomes. Not thread-safe; gtest runs cases
 * serially.
 *
 * Preparation needs fresh cache placement, which nothing can read back, so it
 * runs only on a tile this process booted from reset-held: each pipeline
 * executable needs an ASIC reset before it runs. A failed prepare or teardown
 * aborts the epoch; the region then stays in an unknown state until a reset.
 */
class Epoch {
public:
    /** Prepares on the first call, then skips or fails every case as the first call did. */
    static void prepare(const x280::harness::session::Fixture& f) {
        switch (state()) {
            case State::Prepared: return;
            case State::Unavailable: GTEST_SKIP() << reason();
            case State::Aborted: FAIL() << "pinned-input epoch aborted earlier in this process: " << reason();
            case State::Unopened: break;
        }
        const auto booted = std::find(f.booted.begin(), f.booted.end(), f.tile_id);
        if (booted == f.booted.end()) {
            state() = State::Unavailable;
            std::ostringstream why;
            why << "the pinned input needs fresh cache placement, and chip " << f.tile_id.chip_id << " L2CPU "
                << f.tile_id.hw_index
                << " was already out of reset when this process opened it; reset the device (tt-smi -r <device>) "
                   "before running this executable";
            reason() = why.str();
            GTEST_SKIP() << reason();
        }
        tile() = f.tile_id;
        auto result = tile_idle(*f.runtime, f.tile_id);
        if (result) {
            result = run_operation(*f.runtime, f.tile_id, Operation::Prepare);
        }
        if (!result) {
            abort_epoch(result.message());
            FAIL() << reason();
        }
        state() = State::Prepared;
    }

    /** Tears down a prepared epoch through its own unit mesh; records a failure on the suite if it cannot. */
    static void teardown() {
        if (state() != State::Prepared) {
            return;
        }
        ::testing::AssertionResult result = ::testing::AssertionSuccess();
        try {
            namespace ttmd = tt::tt_metal::distributed;
            auto mesh = ttmd::MeshDevice::create_unit_mesh(0);
            auto opened = x280::X280Runtime::open(
                mesh,
                x280::RuntimeOpenConfig{
                    .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared,
                    .lim_shape = x280::LimShape{.lim_ways = x280::kPreparedLimWays},
                    .shape_change_policy = x280::LimShapeChangePolicy::RequireExact,
                });
            if (!opened.runtime) {
                result = ::testing::AssertionFailure() << "could not reopen the runtime: " << opened.diagnostic;
            } else {
                result = tile_idle(*opened.runtime, tile());
                if (result) {
                    result = run_operation(*opened.runtime, tile(), Operation::Teardown);
                }
            }
        } catch (const std::exception& error) {
            result = ::testing::AssertionFailure() << "pinned-input teardown threw: " << error.what();
        }
        if (!result) {
            abort_epoch(result.message());
            ADD_FAILURE() << reason();
            return;
        }
        state() = State::Unopened;
    }

private:
    enum class State { Unopened, Prepared, Unavailable, Aborted };

    static State& state() {
        static State s = State::Unopened;
        return s;
    }
    static std::string& reason() {
        static std::string r;
        return r;
    }
    static x280::TileId& tile() {
        static x280::TileId t{};
        return t;
    }
    static void abort_epoch(const std::string& why) {
        state() = State::Aborted;
        reason() = why +
                   "\nthe prepared epoch is aborted: the region is in an unknown state until an ASIC reset "
                   "(tt-smi -r <device>)";
    }
};

/** Runner whose cases run inside the process's pinned-input epoch. */
template <typename Axis>
class EpochRunner : public x280::harness::Runner<Axis> {
public:
    static void TearDownTestSuite() { Epoch::teardown(); }

protected:
    void SetUp() override {
        x280::harness::Runner<Axis>::SetUp();
        if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
            return;
        }
        Epoch::prepare(this->fixture());
    }
};

}  // namespace x280::suites::pipeline_stage::pinned_input

#endif  // X280_SUITES_PIPELINE_STAGE_PINNED_INPUT_HPP
