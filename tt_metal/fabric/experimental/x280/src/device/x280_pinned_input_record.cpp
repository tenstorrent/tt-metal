// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_pinned_input_record.hpp"

#include "x280_format.hpp"

#include "rt/x280_pinned_input.h"

#include <algorithm>
#include <sstream>

namespace x280 {

namespace {

constexpr std::array<const char*, 5> kPinnedInputStatusNames = {
    "none",
    "prepared",
    "aborted",
    "corrupt",
    "invalid_observation",
};

constexpr std::array<const char*, 8> kPinnedInputPrepareOutcomeNames = {
    "prepared",
    "already_prepared",
    "wrong_shape",
    "busy",
    "reset_required",
    "invalid_observation",
    "invalid_request",
    "failed",
};

uint64_t join_words(uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(hi) << 32) | lo; }

bool records_fixed_region(const PinnedInputState& state) {
    return state.wayenable == X280_PINNED_INPUT_WAYENABLE && state.region_base == X280_PINNED_INPUT_BASE &&
           state.region_bytes == X280_PINNED_INPUT_SPAN;
}

}  // namespace

const char* to_string(PinnedInputStatus status) { return enum_name(status, kPinnedInputStatusNames); }

const char* to_string(PinnedInputPrepareOutcome outcome) { return enum_name(outcome, kPinnedInputPrepareOutcomeNames); }

PinnedInputRecord encode_pinned_input_record(PinnedInputStatus status) {
    PinnedInputRecord words{};
    switch (status) {
        case PinnedInputStatus::Aborted: words[kPinnedInputWordStatus] = kPinnedInputRecordAborted; break;
        case PinnedInputStatus::Prepared: words[kPinnedInputWordStatus] = kPinnedInputRecordPrepared; break;
        default: fail(std::string("no pinned-input record encodes status ") + to_string(status));
    }
    words[kPinnedInputWordMagic] = kPinnedInputRecordMagic;
    words[kPinnedInputWordWayEnable] = X280_PINNED_INPUT_WAYENABLE;
    words[kPinnedInputWordBaseLo] = static_cast<uint32_t>(X280_PINNED_INPUT_BASE);
    words[kPinnedInputWordBaseHi] = static_cast<uint32_t>(X280_PINNED_INPUT_BASE >> 32);
    words[kPinnedInputWordBytesLo] = static_cast<uint32_t>(X280_PINNED_INPUT_SPAN);
    words[kPinnedInputWordBytesHi] = static_cast<uint32_t>(X280_PINNED_INPUT_SPAN >> 32);
    return words;
}

PinnedInputState decode_pinned_input_record(const PinnedInputRecord& words) {
    PinnedInputState state;
    state.words = words;
    if (std::all_of(words.begin(), words.end(), [](uint32_t word) { return is_all_ones(word); })) {
        state.status = PinnedInputStatus::InvalidObservation;
        return state;
    }
    if (std::all_of(words.begin(), words.end(), [](uint32_t word) { return word == 0; })) {
        state.status = PinnedInputStatus::None;
        return state;
    }
    // No record this runtime writes holds an all-ones word, so one is never
    // decoded into a field.
    const bool any_all_ones = std::any_of(words.begin(), words.end(), [](uint32_t word) { return is_all_ones(word); });
    const bool reserved_zero = std::all_of(
        words.begin() + kPinnedInputWordFirstReserved, words.end(), [](uint32_t word) { return word == 0; });
    if (words[kPinnedInputWordMagic] != kPinnedInputRecordMagic || !reserved_zero || any_all_ones) {
        state.status = PinnedInputStatus::Corrupt;
        return state;
    }
    switch (words[kPinnedInputWordStatus]) {
        case kPinnedInputRecordAborted: state.status = PinnedInputStatus::Aborted; break;
        case kPinnedInputRecordPrepared: state.status = PinnedInputStatus::Prepared; break;
        default: state.status = PinnedInputStatus::Corrupt; return state;
    }
    state.wayenable = words[kPinnedInputWordWayEnable];
    state.region_base = join_words(words[kPinnedInputWordBaseLo], words[kPinnedInputWordBaseHi]);
    state.region_bytes = join_words(words[kPinnedInputWordBytesLo], words[kPinnedInputWordBytesHi]);
    return state;
}

std::optional<PinnedInputPrepareOutcome> refuse_pinned_input_prepare(
    uint8_t runtime_lim_ways, uint32_t raw_wayenable, bool session_live, const PinnedInputState& state) {
    if (is_all_ones(raw_wayenable)) {
        return PinnedInputPrepareOutcome::InvalidObservation;
    }
    if (runtime_lim_ways != kPreparedLimWays || (raw_wayenable & 0xFu) != X280_PINNED_INPUT_WAYENABLE) {
        return PinnedInputPrepareOutcome::WrongShape;
    }
    if (session_live) {
        return PinnedInputPrepareOutcome::Busy;
    }
    switch (state.status) {
        case PinnedInputStatus::None: return std::nullopt;
        case PinnedInputStatus::Prepared:
            return records_fixed_region(state) ? PinnedInputPrepareOutcome::AlreadyPrepared
                                               : PinnedInputPrepareOutcome::ResetRequired;
        case PinnedInputStatus::Aborted:
        case PinnedInputStatus::Corrupt:
        case PinnedInputStatus::InvalidObservation: return PinnedInputPrepareOutcome::ResetRequired;
    }
    fail("unhandled pinned-input status " + std::to_string(static_cast<int>(state.status)));
}

PinnedInputStatus pinned_input_status_after_run(const PinnedInputPrepareResult& result) {
    if (!result.enqueue) {
        return PinnedInputStatus::Aborted;
    }
    if (!result.enqueue->session) {
        return PinnedInputStatus::None;
    }
    if (!result.enqueue->enqueued() || !result.wait || result.wait->outcome != WaitOutcome::ResultReady) {
        return PinnedInputStatus::Aborted;
    }
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        const auto& observed = result.wait->hartstates[hart];
        const uint64_t expected = hart == 0 ? X280_PINNED_INPUT_RESULT_PREPARED : X280_PINNED_INPUT_RESULT_NOOP_HART;
        if (!observed.observation_valid || observed.state.result != expected) {
            return PinnedInputStatus::Aborted;
        }
    }
    const bool restored = result.cleanup && result.cleanup->cleaned_up() && result.final_state &&
                          result.final_state->observation_valid && result.final_state->resident_idle_live;
    return restored ? PinnedInputStatus::Prepared : PinnedInputStatus::Aborted;
}

std::string describe(const PinnedInputState& state) {
    std::ostringstream out;
    out << to_string(state.status);
    switch (state.status) {
        case PinnedInputStatus::Prepared:
        case PinnedInputStatus::Aborted:
            out << " wayenable=" << state.wayenable << " region=[" << hex(state.region_base) << ", +"
                << hex(state.region_bytes) << ")";
            break;
        case PinnedInputStatus::Corrupt:
            out << " words";
            for (const uint32_t word : state.words) {
                out << ' ' << hex(word);
            }
            break;
        case PinnedInputStatus::None:
        case PinnedInputStatus::InvalidObservation: break;
    }
    return out.str();
}

std::string describe(const PinnedInputPrepareResult& result) {
    std::ostringstream out;
    out << to_string(result.outcome) << "; record " << describe(result.state) << "; runtime "
        << static_cast<unsigned>(result.runtime_lim_ways) << " LIM ways, WayEnable " << hex(result.raw_wayenable);
    if (result.outcome == PinnedInputPrepareOutcome::InvalidRequest) {
        out << "; wait_timeout " << result.wait_timeout.count() << " ms and cleanup_timeout "
            << result.cleanup_timeout.count() << " ms must both be positive";
    }
    if (result.outcome == PinnedInputPrepareOutcome::ResetRequired) {
        out << "; reset the device (tt-smi -r <device>)";
    }
    if (!result.error.empty()) {
        out << "\nerror: " << result.error;
    }
    if (result.enqueue) {
        out << "\nenqueue: " << describe(*result.enqueue);
    }
    if (result.wait) {
        out << "\nwait: " << describe(*result.wait);
    }
    if (result.cleanup) {
        out << "\ncleanup: " << describe(*result.cleanup);
    }
    if (result.final_state) {
        out << "\nfinal:\n" << describe(*result.final_state);
    }
    return out.str();
}

}  // namespace x280
