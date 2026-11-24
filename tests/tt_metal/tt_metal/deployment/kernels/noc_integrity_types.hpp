// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NOC_INTEGRITY_TYPES_HPP
#define NOC_INTEGRITY_TYPES_HPP

#include <stdint.h>

enum : uint32_t {
    NOC_PATTERN_WALKING_ONES,
    NOC_PATTERN_WALKING_ZEROES,
    NOC_PATTERN_CHECKERBOARD,
    NOC_PATTERN_CHECKERBOARD_INV,
    NOC_PATTERN_ADDRESS,
    NOC_PATTERN_PRBS,
    NOC_PATTERN_COUNT,
};

enum : uint32_t {
    NOC_PROGRESS_HEARTBEAT,
    NOC_PROGRESS_STATE,
    NOC_PROGRESS_DEST,
    NOC_PROGRESS_PATTERN,
    NOC_PROGRESS_ITERATION,
    NOC_PROGRESS_WORDS,
};

enum : uint32_t {
    NOC_STATE_RUNNING = 0x52554e00,
    NOC_STATE_DONE = 0x444f4e45,
    NOC_STATE_ABORTED = 0x41425254,
};

enum : uint32_t {
    NOC_RESULT_STATUS,
    NOC_RESULT_MISMATCHES,
    NOC_RESULT_FIRST_OFFSET,
    NOC_RESULT_EXPECTED,
    NOC_RESULT_ACTUAL,
    NOC_RESULT_PATTERN,
    NOC_RESULT_ITERATION,
    NOC_RESULT_TRANSFERS,
    NOC_RESULT_WORDS,
};

enum : uint32_t {
    NOC_STATUS_NOT_RUN,
    NOC_STATUS_DONE,
    NOC_STATUS_WRITE_TIMEOUT,
    NOC_STATUS_READ_TIMEOUT,
    NOC_STATUS_HANG,  // set by the host when a source stops making progress
};

[[maybe_unused]]
static const char* noc_pattern_name(uint32_t pattern) {
    switch (pattern) {
        case NOC_PATTERN_WALKING_ONES: return "WalkingOnes";
        case NOC_PATTERN_WALKING_ZEROES: return "WalkingZeroes";
        case NOC_PATTERN_CHECKERBOARD: return "Checkerboard";
        case NOC_PATTERN_CHECKERBOARD_INV: return "CheckerboardInv";
        case NOC_PATTERN_ADDRESS: return "Address";
        case NOC_PATTERN_PRBS: return "PRBS";
        default: return "Unknown";
    }
}

#endif /* NOC_INTEGRITY_TYPES_HPP */
