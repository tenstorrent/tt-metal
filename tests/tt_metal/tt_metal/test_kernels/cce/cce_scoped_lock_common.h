// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the scoped-lock producer and consumer kernels share: their argument checks and their status line.

#pragma once

#include <stdint.h>

#include "cce_scoped_lock_abi.h"
#include "cce_test_common.h"

namespace cce_scoped_lock {

namespace abi = cce_scoped_lock_abi;

inline bool lock_lines_valid(uint32_t lock_first_line, uint32_t lock_num_lines) {
    return lock_num_lines != 0 && lock_first_line + lock_num_lines <= abi::kNumLines;
}

inline bool ring_wrap_valid(uint32_t ring_slots, uint32_t slot_lines, uint32_t rounds) {
    return ring_slots != 0 && slot_lines != 0 && ring_slots * slot_lines <= abi::kNumLines && rounds != 0 &&
           rounds <= 0xFF;
}

struct Status {
    uint32_t code = abi::kCodeDone;
    uint32_t flags = 0;
    uint32_t wait_cycles = 0;
    uint32_t poll_cycles = 0;
    uint32_t last_poll = abi::kPollNone;
    bool last_poll_ready = false;
    uint32_t flag_observed = 0;
    uint32_t rounds_done = 0;
    cce_test::DataCheck check;
};

inline void write_status(uint32_t status_offset, const Status& status, uint32_t done_magic) {
    volatile uint32_t* status_line = cce_test::own_tl1_words(status_offset);
    status_line[abi::kStatusCode] = status.code;
    status_line[abi::kStatusFlags] = status.flags;
    status_line[abi::kStatusWaitCycles] = status.wait_cycles;
    status_line[abi::kStatusPollCycles] = status.poll_cycles;
    status_line[abi::kStatusLastPoll] = status.last_poll;
    status_line[abi::kStatusLastPollReady] = status.last_poll_ready ? 1 : 0;
    status_line[abi::kStatusFlagObserved] = status.flag_observed;
    status_line[abi::kStatusMismatchedWords] = status.check.mismatched_words;
    status_line[abi::kStatusFirstMismatchWord] = status.check.first_mismatch_word;
    status_line[abi::kStatusObserved] = status.check.observed;
    status_line[abi::kStatusExpected] = status.check.expected;
    status_line[abi::kStatusRounds] = status.rounds_done;
    cce_test::publish_done_magic(status_line, done_magic);
}

}  // namespace cce_scoped_lock
