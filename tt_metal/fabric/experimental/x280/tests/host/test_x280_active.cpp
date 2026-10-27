// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_active.hpp"

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>

namespace {

using namespace x280;

std::array<HartState, kHartsPerTile> successful(uint64_t value) {
    std::array<HartState, kHartsPerTile> hartstates{};
    for (uint32_t hart_id = 0; hart_id < kHartsPerTile; ++hart_id) {
        auto& hart = hartstates[hart_id];
        hart.hart_id = hart_id;
        hart.observation_valid = true;
        hart.control.run_value = value;
        hart.control.entry = X280_ACTIVE_FW_LOAD_ADDR;
        hart.control.command = X280_HART_COMMAND_JUMP;
        hart.state.protocol_version = X280_HART_PROTOCOL_VERSION;
        hart.state.run_value = value;
        hart.state.phase = X280_HART_PHASE_RUNNING_ACTIVE_FW;
        hart.state.result = value;
        hart.state.heartbeat = X280_HART_ACTIVE_HEARTBEAT(value, hart_id);
        hart.state.terminal_status = X280_HART_STATUS_SUCCESS;
    }
    return hartstates;
}

bool mentions(const std::string& text, const char* value) { return text.find(value) != std::string::npos; }

}  // namespace

// The runtime owns every session and hands out references to it.
static_assert(!std::is_copy_constructible_v<X280ActiveSession>);
static_assert(!std::is_copy_assignable_v<X280ActiveSession>);
static_assert(!std::is_move_constructible_v<X280ActiveSession>);
static_assert(!std::is_move_assignable_v<X280ActiveSession>);

// A SessionId is a value a caller stores and compares.
static_assert(std::is_trivially_copyable_v<SessionId>);
static_assert(!std::is_convertible_v<SessionId, bool>);

static_assert(sizeof(x280_hart_protocol_t) == X280_HART_PROTOCOL_SIZE, "hart protocol type and region size must agree");

TEST(ActiveMailbox, PerHartRecordsHaveTheWireLayout) {
    EXPECT_EQ(sizeof(x280_hart_control_t), X280_HART_RECORD_STRIDE);
    EXPECT_EQ(sizeof(x280_hart_state_t), X280_HART_RECORD_STRIDE);
    EXPECT_EQ(sizeof(x280_hart_trap_t), X280_HART_RECORD_STRIDE);
    EXPECT_EQ(alignof(x280_hart_control_t), X280_HART_RECORD_STRIDE);
    EXPECT_EQ(offsetof(x280_hart_control_t, command), X280_HART_CONTROL_COMMAND_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_control_t, args_bytes), X280_HART_CONTROL_ARGS_BYTES_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_state_t, heartbeat), X280_HART_STATE_HEARTBEAT_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_state_t, stage), X280_HART_STATE_STAGE_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_trap_t, trap_valid), X280_HART_TRAP_VALID_OFFSET);
    EXPECT_EQ(sizeof(x280_hart_protocol_t), X280_HART_PROTOCOL_SIZE);
    EXPECT_EQ(alignof(x280_hart_protocol_t), X280_HART_RECORD_STRIDE);
    EXPECT_EQ(offsetof(x280_hart_protocol_t, control), X280_HART_CONTROL_BASE_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_protocol_t, state), X280_HART_STATE_BASE_OFFSET);
    EXPECT_EQ(offsetof(x280_hart_protocol_t, trap), X280_HART_TRAP_BASE_OFFSET);
}

TEST(ActiveHartStates, MatchesOnlyAValidCorrelatedTerminalResult) {
    constexpr uint64_t value = 0xA5A5000000000042ull;
    auto hartstates = successful(value);

    EXPECT_TRUE(active_result_matches(hartstates, value));
    EXPECT_TRUE(active_all_results_ready(hartstates, value));
    EXPECT_FALSE(active_all_results_ready(hartstates, value + 1));

    hartstates[2].state.terminal_status = 0;
    EXPECT_TRUE(active_result_matches(hartstates, value));
    EXPECT_FALSE(active_all_results_ready(hartstates, value));

    hartstates[2].observation_valid = false;
    EXPECT_FALSE(active_result_matches(hartstates, value));
}

// The result word is the application's payload, so a successful run that
// computed something other than the run value is ready. Nothing else the
// correlation rests on is relaxed.
TEST(ActiveHartStates, ReadinessAcceptsAPayloadThatIsNotTheRunValue) {
    constexpr uint64_t value = 0xA5A5000000000042ull;
    auto ready = successful(value);
    for (auto& hart : ready) {
        hart.state.result = 0x0BADC0DE00000000ull + hart.hart_id;
    }

    EXPECT_TRUE(active_all_results_ready(ready, value));
    EXPECT_TRUE(hart_result_ready(ready[0], value));
    EXPECT_FALSE(active_result_matches(ready, value));
}

// An image that finishes on its own commits its terminal record and advances
// straight to RETURNED_TO_IDLE, so the host can miss RUNNING entirely. The
// record is the same one; only the phase moved on.
TEST(ActiveHartStates, ReadinessSurvivesTheReturnToIdlePhase) {
    constexpr uint64_t value = 0x33CC;
    auto returned = successful(value);
    for (auto& hart : returned) {
        hart.state.phase = X280_HART_PHASE_RETURNED_TO_IDLE;
    }

    EXPECT_TRUE(active_all_results_ready(returned, value));
    EXPECT_FALSE(active_all_running(returned, value));

    // Correlation is not relaxed with the phase.
    EXPECT_FALSE(active_all_results_ready(returned, value + 1));
    returned[2].state.terminal_status = 0;
    EXPECT_FALSE(active_all_results_ready(returned, value));

    // Nor does the resident idle phase carry a readable result.
    auto idle = successful(value);
    for (auto& hart : idle) {
        hart.state.phase = X280_HART_PHASE_IDLE;
        hart.state.heartbeat = X280_HART_IDLE_HEARTBEAT(hart.hart_id);
    }
    EXPECT_FALSE(active_all_results_ready(idle, value));
}

TEST(ActiveHartStates, RenderingCarriesProtocolAndTrapState) {
    auto hartstates = successful(0x1234);
    hartstates[2].trap.run_value = 0x1234;
    hartstates[2].trap.mcause = 3;
    hartstates[2].trap.mepc = X280_ACTIVE_FW_LOAD_ADDR + 0x80;
    hartstates[2].trap.trap_valid = 1;

    const auto text = describe(hartstates);
    EXPECT_TRUE(mentions(text, "hart 2"));
    EXPECT_TRUE(mentions(text, "phase=0x"));
    EXPECT_TRUE(mentions(text, "result=0x1234"));
    EXPECT_TRUE(mentions(text, "status=0x10000003"));
    EXPECT_TRUE(mentions(text, "control_run=0x1234"));
    EXPECT_TRUE(mentions(text, "entry=0x8001000"));
    EXPECT_TRUE(mentions(text, "mcause=0x3"));

    std::array<HartState, kHartsPerTile> invalid{};
    invalid[0].hart_id = 0;
    invalid[0].invalid_reason = "transport timeout";
    EXPECT_TRUE(mentions(describe(invalid), "INVALID"));
    EXPECT_TRUE(mentions(describe(invalid), "transport timeout"));
}

TEST(ActiveHartStates, OneStaleHartBlocksEveryAggregate) {
    constexpr uint64_t value = 0x9911;
    auto hartstates = successful(value);
    hartstates[3].state.run_value = value - 1;

    EXPECT_FALSE(active_all_running(hartstates, value));
    EXPECT_FALSE(active_all_results_ready(hartstates, value));
}

TEST(ActiveHartStates, OneCommittedTrapBlocksAggregateSuccess) {
    constexpr uint64_t value = 0x4422;
    auto hartstates = successful(value);
    hartstates[1].trap.run_value = value;
    hartstates[1].trap.trap_valid = 1;
    hartstates[1].trap.mcause = 3;

    EXPECT_FALSE(active_all_running(hartstates, value));
    EXPECT_FALSE(active_all_results_ready(hartstates, value));
}

TEST(ActiveHartStates, OneInvalidHartBlocksAggregateSuccess) {
    constexpr uint64_t value = 0x5522;
    auto hartstates = successful(value);
    hartstates[1].observation_valid = false;
    hartstates[1].invalid_reason = "all ones";
    EXPECT_FALSE(active_all_running(hartstates, value));
}

TEST(ActiveHartStates, UncommittedStatusDoesNotExposeResult) {
    constexpr uint64_t value = 0x8877;
    auto hartstates = successful(value);
    hartstates[0].state.terminal_status = 0;
    EXPECT_FALSE(active_all_results_ready(hartstates, value));
}

TEST(ActiveHartStates, UncommittedTrapPayloadIsDiagnosticOnly) {
    constexpr uint64_t value = 0x7788;
    auto hartstates = successful(value);
    hartstates[0].trap.run_value = value;
    hartstates[0].trap.mcause = 3;
    hartstates[0].trap.trap_valid = 0;
    EXPECT_TRUE(active_all_results_ready(hartstates, value));
    EXPECT_TRUE(mentions(describe(hartstates), "uncommitted"));
}

TEST(ActiveHartStates, ReturnedAndIdleAreGenerationCorrelated) {
    constexpr uint64_t value = 0x123456;
    auto hartstates = successful(value);
    for (auto& hart : hartstates) {
        hart.state.phase = X280_HART_PHASE_RETURNED_TO_IDLE;
    }
    EXPECT_TRUE(active_all_returned(hartstates, value));

    for (auto& hart : hartstates) {
        hart.state.phase = X280_HART_PHASE_IDLE;
        hart.state.heartbeat = X280_HART_IDLE_HEARTBEAT(hart.hart_id);
    }
    EXPECT_TRUE(active_all_idle(hartstates, value));
    hartstates[0].state.protocol_version = 0;
    EXPECT_FALSE(active_all_idle(hartstates, value));
}

// The stage is a free-running postcode, not a commit word: enqueue's record
// clear leaves it at 0, so 0 means the harness never ran on that hart, and no
// predicate may read it.
TEST(ActiveHartStates, TheStageIsDiagnosticAndAClearedRecordReportsZero) {
    constexpr uint64_t value = 0x3141;
    const HartState cleared;
    EXPECT_EQ(cleared.state.stage, 0u);

    auto hartstates = successful(value);
    for (const auto& hart : hartstates) {
        EXPECT_EQ(hart.state.stage, 0u);
    }
    EXPECT_TRUE(active_all_results_ready(hartstates, value));

    // Advancing it changes no predicate, and it renders for diagnosis.
    hartstates[2].state.stage = X280_HART_STAGE_STOPPING;
    EXPECT_TRUE(active_all_results_ready(hartstates, value));
    EXPECT_TRUE(active_all_running(hartstates, value));
    EXPECT_TRUE(mentions(describe(hartstates), "stage=4"));
}

TEST(ActiveCommitState, AClearedRecordHasNothingCommittedAndNothingPending) {
    const HartState cleared;
    const auto commit = commit_state(cleared);
    EXPECT_FALSE(commit.phase);
    EXPECT_FALSE(commit.phase_payload_pending);
    EXPECT_FALSE(commit.terminal);
    EXPECT_FALSE(commit.terminal_payload_pending);
    EXPECT_FALSE(commit.trap);
    EXPECT_FALSE(commit.trap_payload_pending);
}

TEST(ActiveCommitState, ACommittedWordLeavesItsPayloadNotPending) {
    const auto hartstates = successful(0x2468);
    const auto commit = commit_state(hartstates[1]);
    EXPECT_TRUE(commit.phase);
    EXPECT_FALSE(commit.phase_payload_pending);
    EXPECT_TRUE(commit.terminal);
    EXPECT_FALSE(commit.terminal_payload_pending);
    EXPECT_FALSE(commit.trap);
    EXPECT_FALSE(commit.trap_payload_pending);
}

TEST(ActiveCommitState, PayloadBehindAnUncommittedPhaseIsPending) {
    auto hartstates = successful(0x1357);
    auto& hart = hartstates[0];
    hart.state.phase = 0;
    ASSERT_NE(hart.state.heartbeat, 0u);
    ASSERT_NE(hart.state.run_value, 0u);

    const auto commit = commit_state(hart);
    EXPECT_FALSE(commit.phase);
    EXPECT_TRUE(commit.phase_payload_pending);
}

TEST(ActiveCommitState, AnUncommittedTerminalStatusLeavesItsResultPending) {
    auto hartstates = successful(0x8877);
    auto& hart = hartstates[3];
    hart.state.terminal_status = 0;
    ASSERT_NE(hart.state.result, 0u);

    const auto commit = commit_state(hart);
    EXPECT_TRUE(commit.phase);
    EXPECT_FALSE(commit.terminal);
    EXPECT_TRUE(commit.terminal_payload_pending);
}

TEST(ActiveCommitState, ATrapPayloadIsPendingUntilTrapValidLands) {
    auto hartstates = successful(0x7788);
    auto& hart = hartstates[2];
    hart.trap.run_value = 0x7788;
    hart.trap.mcause = 3;
    hart.trap.mepc = X280_ACTIVE_FW_LOAD_ADDR + 0x40;

    const auto pending = commit_state(hart);
    EXPECT_FALSE(pending.trap);
    EXPECT_TRUE(pending.trap_payload_pending);

    hart.trap.trap_valid = 1;
    const auto landed = commit_state(hart);
    EXPECT_TRUE(landed.trap);
    EXPECT_FALSE(landed.trap_payload_pending);
}

TEST(ActiveSessionId, ADefaultIdNamesNoExecution) {
    constexpr SessionId none;
    EXPECT_EQ(none.value, 0u);
    EXPECT_FALSE(static_cast<bool>(none));

    constexpr SessionId first{1};
    EXPECT_TRUE(static_cast<bool>(first));
}

TEST(ActiveSessionId, IdsOrderByIssueOrder) {
    constexpr SessionId none;
    constexpr SessionId first{1};
    constexpr SessionId second{2};

    EXPECT_LT(none, first);
    EXPECT_LT(first, second);
    EXPECT_GT(second, first);
    EXPECT_EQ(first, SessionId{1});
    EXPECT_NE(first, second);
}

TEST(ActiveSessionId, ADefaultIdNamesNoTile) {
    constexpr SessionId none;
    EXPECT_EQ(none.tile, TileKey{});
    EXPECT_EQ(none.tile.chip_id, -1);
    EXPECT_EQ(none.tile.tile_index, -1);
}

TEST(ActiveSessionId, AnIdNamesTheTileItRunsOn) {
    constexpr SessionId id{7, TileKey{1, 2}};

    EXPECT_EQ(id.value, 7u);
    EXPECT_EQ(id.tile.chip_id, 1);
    EXPECT_EQ(id.tile.tile_index, 2);
}

// The registry is keyed by tile and the generation is what separates successive
// executions on one of them, so both halves have to participate in equality.
TEST(ActiveSessionId, TileAndGenerationBothDiscriminate) {
    constexpr SessionId first{1, TileKey{0, 2}};
    constexpr SessionId same_tile_later{2, TileKey{0, 2}};
    constexpr SessionId other_tile{1, TileKey{0, 3}};

    EXPECT_NE(first, same_tile_later);
    EXPECT_NE(first, other_tile);
    EXPECT_EQ(first, (SessionId{1, TileKey{0, 2}}));
}

TEST(TileKeyIdentity, DerivedIdentityFieldsAreNotPartOfTheKey) {
    TileIdentity id;
    id.chip_id = 1;
    id.hw_index = 3;
    id.noc_x = 8;
    id.noc_y = 7;

    TileIdentity same_tile_unpopulated;
    same_tile_unpopulated.chip_id = 1;
    same_tile_unpopulated.hw_index = 3;

    // TileIdentity compares all five fields; the key compares only the two that
    // identify, so a caller holding a partly-filled identity still keys alike.
    EXPECT_NE(id, same_tile_unpopulated);
    EXPECT_EQ(tile_key(id), tile_key(same_tile_unpopulated));
    EXPECT_EQ(tile_key(id).chip_id, 1);
    EXPECT_EQ(tile_key(id).tile_index, 3);
}

TEST(EnqueueOutcome, EveryOutcomeRendersDistinctly) {
    constexpr EnqueueOutcome outcomes[] = {
        EnqueueOutcome::Enqueued,
        EnqueueOutcome::Busy,
        EnqueueOutcome::WrongRuntime,
        EnqueueOutcome::Unhealthy,
        EnqueueOutcome::NotResidentIdle,
        EnqueueOutcome::ImageRejected,
        EnqueueOutcome::ArgumentsRejected,
        EnqueueOutcome::InvalidObservation,
        EnqueueOutcome::JumpPublicationUncertain,
        EnqueueOutcome::Failed,
    };

    for (const EnqueueOutcome outcome : outcomes) {
        const char* text = to_string(outcome);
        ASSERT_NE(text, nullptr);
        EXPECT_STRNE(text, "");
        // An enumerator added without extending the name table renders as
        // "unknown", which is neither empty nor a duplicate of any other name,
        // so neither of the other two checks would catch it.
        EXPECT_STRNE(text, "unknown");
    }

    // A rejected argument block and a rejected image are separate caller
    // mistakes, so neither may render as the other.
    for (size_t left = 0; left < std::size(outcomes); ++left) {
        for (size_t right = left + 1; right < std::size(outcomes); ++right) {
            EXPECT_STRNE(to_string(outcomes[left]), to_string(outcomes[right]))
                << "outcomes " << left << " and " << right;
        }
    }
}

TEST(CleanupOutcome, EveryOutcomeRendersDistinctly) {
    constexpr CleanupOutcome outcomes[] = {
        CleanupOutcome::ReturnedToIdle,
        CleanupOutcome::StillRunningActiveFw,
        CleanupOutcome::ReturnAcknowledgementFailed,
        CleanupOutcome::IdleNotReArmed,
        CleanupOutcome::InvalidObservation,
        CleanupOutcome::UnknownSession,
        CleanupOutcome::AlreadyRestoring,
        CleanupOutcome::Failed,
    };

    for (const CleanupOutcome outcome : outcomes) {
        const char* text = to_string(outcome);
        ASSERT_NE(text, nullptr);
        EXPECT_STRNE(text, "");
    }

    // A stale id and a failed restoration stay distinguishable, so no case may
    // share another's rendering.
    for (size_t left = 0; left < std::size(outcomes); ++left) {
        for (size_t right = left + 1; right < std::size(outcomes); ++right) {
            EXPECT_STRNE(to_string(outcomes[left]), to_string(outcomes[right]))
                << "outcomes " << left << " and " << right;
        }
    }
}
