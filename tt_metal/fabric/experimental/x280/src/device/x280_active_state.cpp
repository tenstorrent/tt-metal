// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_active.hpp"

#include "x280_format.hpp"

#include "x280_hart_protocol.h"

#include <sstream>

namespace x280 {

namespace {

constexpr std::array<const char*, 6> kWaitOutcomeNames = {
    "running",
    "result_ready",
    "timed_out",
    "result_mismatch",
    "invalid_observation",
    "failed",
};

constexpr std::array<const char*, 9> kCleanupOutcomeNames = {
    "returned_to_idle",
    "still_running_active_fw",
    "return_acknowledgement_failed",
    "idle_not_rearmed",
    "invalid_observation",
    "unknown_session",
    "already_restoring",
    "failed",
    "revived",
};

constexpr std::array<const char*, 10> kReviveOutcomeNames = {
    "revived",
    "not_responding",
    "not_rearmed",
    "not_installed",
    "not_released",
    "busy",
    "wrong_runtime",
    "invalid_request",
    "invalid_observation",
    "failed",
};

constexpr std::array<const char*, 10> kEnqueueOutcomeNames = {
    "enqueued",
    "busy",
    "wrong_runtime",
    "unhealthy",
    "not_resident_idle",
    "image_rejected",
    "arguments_rejected",
    "invalid_observation",
    "jump_publication_uncertain",
    "failed",
};

}  // namespace

static bool trapped_for_run(const HartState& hart, uint64_t run_value) {
    return hart.trap.trap_valid != 0 && hart.trap.run_value == run_value;
}

// Everything an active-image observation must satisfy before its phase means
// anything: a valid read, no trap committed against this generation, the
// expected protocol version, and the run value and heartbeat the image
// publishes. Both RUNNING and RETURNED_TO_IDLE carry the active heartbeat --
// firmware rewrites it alongside the run value ahead of each phase store.
static bool correlated_active(const HartState& hart, uint64_t run_value) {
    return hart.observation_valid && !trapped_for_run(hart, run_value) &&
           hart.state.protocol_version == X280_HART_PROTOCOL_VERSION && hart.state.run_value == run_value &&
           hart.state.heartbeat == X280_HART_ACTIVE_HEARTBEAT(run_value, hart.hart_id);
}

bool hart_running(const HartState& hart, uint64_t run_value) {
    return correlated_active(hart, run_value) && hart.state.phase == X280_HART_PHASE_RUNNING_ACTIVE_FW;
}

// Either active phase, because an image commits its terminal record and then
// advances to RETURNED_TO_IDLE without waiting for the host: requiring RUNNING
// here would make readiness unobservable for any image that finishes on its
// own, and the wait would time out on a successful run. RETURNED_TO_IDLE still
// holds the committed record -- the hart only leaves it, and the record only
// becomes the idle FW's, once the host acknowledges the return.
bool hart_result_ready(const HartState& hart, uint64_t run_value) {
    return correlated_active(hart, run_value) &&
           (hart.state.phase == X280_HART_PHASE_RUNNING_ACTIVE_FW ||
            hart.state.phase == X280_HART_PHASE_RETURNED_TO_IDLE) &&
           hart.state.terminal_status == X280_HART_STATUS_SUCCESS;
}

bool hart_returned(const HartState& hart, uint64_t run_value) {
    return correlated_active(hart, run_value) && hart.state.phase == X280_HART_PHASE_RETURNED_TO_IDLE;
}

bool hart_resident_idle(const HartState& hart) {
    return hart.observation_valid && hart.trap.trap_valid == 0 &&
           hart.state.protocol_version == X280_HART_PROTOCOL_VERSION &&
           hart.state.heartbeat == X280_HART_IDLE_HEARTBEAT(hart.hart_id) && hart.state.phase == X280_HART_PHASE_IDLE;
}

bool hart_idle(const HartState& hart, uint64_t run_value) {
    return hart_resident_idle(hart) && hart.state.run_value == run_value;
}

template <typename Predicate>
bool all_harts(const std::array<HartState, kHartsPerTile>& hartstates, Predicate predicate) {
    for (const auto& hart : hartstates) {
        if (!predicate(hart)) {
            return false;
        }
    }
    return true;
}

bool active_all_running(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value) {
    return all_harts(hartstates, [&](const HartState& hart) { return hart_running(hart, run_value); });
}

bool active_all_results_ready(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value) {
    return all_harts(hartstates, [&](const HartState& hart) { return hart_result_ready(hart, run_value); });
}

bool active_all_returned(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value) {
    return all_harts(hartstates, [&](const HartState& hart) { return hart_returned(hart, run_value); });
}

bool active_all_idle(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value) {
    return all_harts(hartstates, [&](const HartState& hart) { return hart_idle(hart, run_value); });
}

bool active_result_matches(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value) {
    return all_harts(hartstates, [&](const HartState& hart) {
        return hart.observation_valid && hart.state.run_value == run_value && hart.state.result == run_value;
    });
}

HartCommit commit_state(const HartState& hart) {
    HartCommit commit;
    commit.phase = hart.state.phase != 0;
    commit.phase_payload_pending = !commit.phase && (hart.state.run_value != 0 || hart.state.heartbeat != 0);
    commit.terminal = hart.state.terminal_status != 0;
    commit.terminal_payload_pending = !commit.terminal && hart.state.result != 0;
    commit.trap = hart.trap.trap_valid != 0;
    commit.trap_payload_pending =
        !commit.trap && (hart.trap.run_value != 0 || hart.trap.mcause != 0 || hart.trap.mepc != 0 ||
                         hart.trap.mtval != 0 || hart.trap.probe_state != 0);
    return commit;
}

std::string describe(const std::array<HartState, kHartsPerTile>& hartstates) {
    std::ostringstream out;
    out << "active";
    for (const auto& hart : hartstates) {
        out << " hart " << hart.hart_id;
        if (!hart.observation_valid) {
            out << '=' << describe(hart);
            continue;
        }
        out << '(' << describe(hart) << ')';
    }
    return out.str();
}

const char* to_string(WaitOutcome outcome) { return enum_name(outcome, kWaitOutcomeNames); }

const char* to_string(CleanupOutcome outcome) { return enum_name(outcome, kCleanupOutcomeNames); }

const char* to_string(EnqueueOutcome outcome) { return enum_name(outcome, kEnqueueOutcomeNames); }

const char* to_string(ReviveOutcome outcome) { return enum_name(outcome, kReviveOutcomeNames); }

ReviveRequestCheck validate_revive_request(const ReviveRequest& request, bool session_live) {
    ReviveRequestCheck check;
    if (request.hart_mask == 0) {
        check.reason = "revive hart mask is zero";
        return check;
    }
    if ((request.hart_mask & ~kAllHartsMask) != 0) {
        check.reason = "revive hart mask " + hex(request.hart_mask) + " names a hart outside the tile's " +
                       std::to_string(kHartsPerTile);
        return check;
    }
    if (request.timeout.count() <= 0) {
        check.reason = "revive timeout is " + std::to_string(request.timeout.count()) + " ms; it must be positive";
        return check;
    }
    if (session_live && request.hart_mask != kAllHartsMask) {
        check.reason = "a live session is a whole-tile execution: reviving harts " + hex(request.hart_mask) +
                       " would leave the others in its image with nothing owning them";
        return check;
    }
    check.ok = true;
    return check;
}

bool hart_revive_acknowledged(const HartState& hart, uint32_t count_before) {
    return hart.observation_valid && hart.state.revive_count != count_before;
}

bool hart_revived(const HartState& hart, uint32_t count_before) {
    return hart_revive_acknowledged(hart, count_before) && hart_resident_idle(hart);
}

uint32_t rnmi_trigger_set(uint32_t current, uint32_t mask) { return current | mask; }

uint32_t rnmi_trigger_clear(uint32_t current, uint32_t mask) { return current & ~mask; }

bool cleanup_escalates_to_revive(CleanupOutcome outcome) {
    return outcome == CleanupOutcome::StillRunningActiveFw || outcome == CleanupOutcome::IdleNotReArmed;
}

std::string describe(const ReviveResult& result) {
    std::ostringstream out;
    out << to_string(result.outcome) << " timeout=" << result.timeout.count() << "ms"
        << " ack=" << result.trigger_to_ack.count() << "us idle=" << result.trigger_to_idle.count() << "us"
        << " run_status " << hex(result.run_status_before) << "->" << hex(result.run_status_after);
    if (!result.error.empty()) {
        out << " error=" << result.error;
    }
    for (const auto& hart : result.harts) {
        if (!hart.requested) {
            continue;
        }
        out << " hart " << hart.hart_id << (hart.acknowledged ? " revived" : " no-ack")
            << " epc=" << hex(hart.revive_epc) << " cause=" << hex(hart.revive_cause);
    }
    return out.str();
}

// A result that failed before any record was read carries a default-constructed
// array, whose four zeroed entries would render as four unreadable hart 0s.
// Every path that reads the tile numbers the harts it read.
static bool records_were_read(const std::array<HartState, kHartsPerTile>& hartstates) {
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        if (hartstates[hart].hart_id != hart) {
            return false;
        }
    }
    return true;
}

std::string describe(const WaitResult& result) {
    std::ostringstream out;
    out << to_string(result.outcome) << " timeout=" << result.timeout.count() << "ms";
    if (!result.error.empty()) {
        out << " error=" << result.error;
    }
    if (records_were_read(result.hartstates)) {
        out << ' ' << describe(result.hartstates);
    }
    return out.str();
}

std::string describe(const CleanupResult& result) {
    std::ostringstream out;
    out << to_string(result.outcome) << " timeout=" << result.timeout.count() << "ms";
    if (!result.error.empty()) {
        out << " error=" << result.error;
    }
    if (result.revive) {
        out << " revive(" << describe(*result.revive) << ')';
    }
    if (records_were_read(result.hartstates)) {
        out << ' ' << describe(result.hartstates);
    }
    return out.str();
}

std::string describe(const EnqueueResult& result) {
    std::ostringstream out;
    out << to_string(result.outcome);
    if (result.session) {
        out << " session=" << result.session.value << " on "
            << label_chip_l2cpu(result.session.tile.chip_id, result.session.tile.tile_index);
    }
    if (!result.error.empty()) {
        out << " error=" << result.error;
    }
    if (records_were_read(result.hartstates)) {
        out << ' ' << describe(result.hartstates);
    }
    return out.str();
}

}  // namespace x280
