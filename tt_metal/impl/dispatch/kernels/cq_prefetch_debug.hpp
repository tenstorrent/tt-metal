// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

// Host-readable L1 counters for prefetch FetchQ / read_from_pcie debugging (no DEVICE_PRINT).
// Layout is fixed; host hang probe reads PREFETCH_DEBUG_COUNTERS_L1 + index * 4.
// Requires PREFETCH_DEBUG_COUNTERS_L1 kernel define (SD_PREFETCH_DEBUG_COUNTERS_BYTES at
// CMDDAT_Q_BASE + CMDDAT_Q_SIZE; CMDDAT_Q_SIZE is page-aligned and CMDDAT_Q_PAGES matches).
//
// Bump PREFETCH_DEBUG_BUILD_ID when changing prefetch debug instrumentation so hang probes can
// confirm the loaded kernel binary matches the host probe (SD_PREFETCH_DEBUG_BUILD_ID in common.h).

#ifndef PREFETCH_DEBUG_BUILD_ID
#define PREFETCH_DEBUG_BUILD_ID 14u
#endif

// Published POST_WHILE_LAST_MILESTONE ids for fetch_q_get_cmds epilogue (after issue-while exit).
enum class PrefetchPostWhileMilestone : uint32_t {
    NONE = 0,
    EPILOGUE_ENTER = 1,
    AFTER_WHILE_EXIT_MIRROR = 2,
    POST_ISSUE_LOOP_EXIT = 3,
    STALL_REEVAL = 4,
    CMD_READY_REEVAL = 5,
    POST_ISSUE_RECORD = 6,
    RETIRE_PRE_BRANCH = 7,
    RETIRE_PATH_ENTRY = 8,
    RETIRE_INFLIGHT_ENTER = 9,
    RETIRE_START = 10,
    RETIRE_TR_ACK_SNAPSHOT = 11,
    BARRIER_ENTER = 12,
    BARRIER_RETURNED = 13,
    CACHE_INVALIDATE_START = 14,
    CACHE_INVALIDATE_DONE = 15,
    RETIRE_DONE = 16,
    RETIRE_STALL_RETURN = 17,
    RETIRE_LOOP_CONTINUE = 18,
    HQW_ENTER = 19,
    FETCHQ_RETURN_PATH = 20,
    FETCHQ_RETURN = 21,
};

constexpr uint32_t PREFETCH_POST_WHILE_MILESTONE_MAX = 21u;

// Published PRE_ISSUE_LAST_MILESTONE ids for fetch_q_get_cmds issue-while entry (before epilogue).
enum class PrefetchPreIssueMilestone : uint32_t {
    NONE = 0,
    BLOCK_ENTER = 1,
    STALL_SKIP = 2,
    WHILE_ENTER = 3,
    COND_FAIL = 4,
    COND_OK = 5,
    LOOP_ENTER = 6,
    READ_FROM_PCIE_CALL = 7,
};

constexpr uint32_t PREFETCH_PRE_ISSUE_MILESTONE_MAX = 7u;

constexpr uint32_t PREFETCH_DEBUG_RETIRE_BARRIER_PHASE_NONE = 0u;
constexpr uint32_t PREFETCH_DEBUG_RETIRE_BARRIER_PHASE_ENTERED = 1u;
constexpr uint32_t PREFETCH_DEBUG_RETIRE_BARRIER_PHASE_RETURNED = 2u;

// Low-slot ISSUE_WHILE_TAIL_MIRROR (0xF8) arg0 packing: [31:24]=waypoint [16]=stall_flag [15:0]=inflight.
// Low-slot ISSUE_WHILE_COND_MIRROR (0xF6) arg0 packing: [31:24]=waypoint [17]=would_continue [16]=stall_flag
// [15:0]=inflight.
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_WAYPOINT_COND_CHECK = 5u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_WAYPOINT_AFTER_STALL_REFRESH = 1u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_WAYPOINT_STALL_BREAK = 2u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_WAYPOINT_ISSUE_FAIL_BREAK = 3u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_WAYPOINT_AFTER_WHILE_EXIT = 4u;

enum class PrefetchPcieEarlyExitCode : uint32_t {
    NONE = 0,
    RING_FULL = 1,
    INSUFF_SPACE_CMD_GT_FENCE = 2,
    INSUFF_SPACE_AT_BEGINNING = 3,
    // Low-slot mirrors (slots 2/3/4); not read_from_pcie early exits.
    FLUSH_ENTER_MIRROR = 0xF4,
    RETIRE_BARRIER_ENTER_MIRROR = 0xF5,
    ISSUE_WHILE_COND_MIRROR = 0xF6,
    ISSUE_OK_INFLIGHT_MIRROR = 0xF7,
    ISSUE_WHILE_TAIL_MIRROR = 0xF8,
    ISSUE_LOOP_EXIT_MIRROR = 0xF9,
    STALL_REFRESH_MIRROR = 0xFA,
    FETCHQ_RETURN_MIRROR = 0xFB,
    BRANCH_MIRROR = 0xFC,
    ISSUE_DONE_MIRROR = 0xFD,
    RETIRE_MIRROR = 0xFE,
};

constexpr uint32_t PREFETCH_DEBUG_FLUSH_ENTER_MIRROR_CODE = 0xF4u;
constexpr uint32_t PREFETCH_DEBUG_RETIRE_BARRIER_ENTER_MIRROR_CODE = 0xF5u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_COND_MIRROR_CODE = 0xF6u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_OK_INFLIGHT_MIRROR_CODE = 0xF7u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_WHILE_TAIL_MIRROR_CODE = 0xF8u;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_LOOP_EXIT_MIRROR_CODE = 0xF9u;
constexpr uint32_t PREFETCH_DEBUG_STALL_REFRESH_MIRROR_CODE = 0xFAu;
constexpr uint32_t PREFETCH_DEBUG_ISSUE_DONE_MIRROR_CODE = 0xFDu;
constexpr uint32_t PREFETCH_DEBUG_BRANCH_MIRROR_CODE = 0xFCu;
constexpr uint32_t PREFETCH_DEBUG_FETCHQ_RETURN_MIRROR_CODE = 0xFBu;
constexpr uint32_t PREFETCH_DEBUG_RETIRE_MIRROR_CODE = 0xFEu;

enum class PrefetchDebugCounter : uint32_t {
    ISSUE_OK = 0,
    ISSUE_FAILED = 1,
    PCIE_EARLY_EXIT_CODE = 2,
    PCIE_EARLY_EXIT_ARG0 = 3,
    PCIE_EARLY_EXIT_ARG1 = 4,
    LAST_FETCHQ_ENTRY = 5,
    LAST_FETCH_SIZE = 6,
    HQW_SPINS = 7,
    RETIRE_START = 8,
    RETIRE_DONE = 9,
    FETCHQ_RETURN = 10,
    INFLIGHT_AT_RETURN = 11,
    CMD_PTR_AT_RETURN = 12,
    FENCE_AT_RETURN = 13,
    PROCESS_CMD = 14,
    LAST_CMD_TAG = 15,  // (cmd_id << 16) | (stride & 0xffff)
    PROCESS_CMD_ENTER = 16,
    RELAY_INLINE_ENTER = 17,
    CB_ACQUIRE_ENTER = 18,
    CB_ACQUIRE_DONE = 19,
    CB_ACQUIRE_SPINS = 20,
    RELAY_WRITE_DONE = 21,
    FIRST_FETCHQ_POLL_RAW = 22,
    FETCHQ_POLL_COUNT = 23,
    LAST_FETCHQ_POLL_RAW = 24,
    PCIE_ISSUE_ENTER = 25,
    MAIN_HD_LOOP = 26,
    ISSUE_LOOP_ENTER = 27,
    READ_FROM_PCIE_ENTER = 28,
    READ_FROM_PCIE_PAST_CMDDAT = 29,
    PCIE_NOC_XY_READY = 30,
    PCIE_READ_ISSUED = 31,
    PCIE_EARLY_EXIT = 32,
    LAST_DECODED_FETCH_SIZE = 33,
    FETCHQ_GET_CMDS_ITER = 34,
    RETIRE_TRID = 35,
    RETIRE_INFLIGHT_COUNT = 36,
    RETIRE_SCMD_TR_ACK = 37,
    RETIRE_TR_ACK_TR2 = 38,
    RETIRE_TR_ACK_TR3 = 39,
    RETIRE_TR_ACK_TR4 = 40,
    RETIRE_TR_ACK_TR5 = 41,
    RETIRE_POST_BARRIER = 42,
    RETIRE_PRE_BARRIER = 43,
    DEBUG_BUILD_ID = 44,
    ISSUE_LOOP_DONE = 45,
    POST_STALL_REFRESH_DONE = 46,
    RETIRE_PATH_ENTER = 47,
    RETIRE_INFLIGHT_ENTER = 48,
    CMD_READY_AT_ISSUE_DONE = 49,
    CMD_PTR_AT_ISSUE_DONE = 50,
    FENCE_AT_ISSUE_DONE = 51,
    INFLIGHT_AT_ISSUE_DONE = 52,
    FETCHQ_RETURN_ENTER = 53,
    CMD_READY_REEVAL = 54,
    HAS_PENDING_STALL_AFTER = 55,
    RETIRE_PRE_ENTER = 56,
    RETIRE_PATH_WRITTEN = 57,
    FETCHQ_RETURN_WRITTEN = 58,
    RETIRE_BARRIER_ENTER_WRITTEN = 59,
    RETIRE_BARRIER_RETURNED_WRITTEN = 60,
    POST_ISSUE_LOOP_EXIT = 61,
    POST_ISSUE_STALL_REEVAL = 62,
    POST_ISSUE_CMD_READY_REEVAL = 63,
    POST_ISSUE_RECORD_ENTER = 64,
    POST_STALL_REFRESH_WRITTEN = 65,
    // Published after each ++inflight_count (immediately at ISSUE_OK); last value survives hang.
    INFLIGHT_AT_EACH_ISSUE_OK = 66,
    ISSUE_WHILE_COND_CHECK_COUNT = 67,
    ISSUE_WHILE_COND_LAST_WOULD_CONTINUE = 68,
    ISSUE_WHILE_COND_LAST_INFLIGHT = 69,
    // Published milestones on the post-while / retire / barrier path (host hang_site classification).
    POST_WHILE_EPILOGUE_ENTER = 70,
    RETIRE_BARRIER_TRID_AT_ENTER = 71,
    RETIRE_BARRIER_PHASE = 72,
    RETIRE_BARRIER_INFLIGHT_AT_ENTER = 73,
    POST_ISSUE_EPILOGUE_DONE = 74,
    RETIRE_BRANCH_REACHED = 75,
    POST_WHILE_LAST_MILESTONE = 76,
    POST_WHILE_MILESTONE_COUNT = 77,
    PRE_ISSUE_LAST_MILESTONE = 78,
    PRE_ISSUE_MILESTONE_COUNT = 79,
    // Recorded immediately before tl1_publish_flush (no publish on these slots).
    FLUSH_TARGET_ADDR = 80,
    FLUSH_LINE_ADDR = 81,
    FLUSH_TARGET_SLOT = 82,
    FLUSH_ENTER_COUNT = 83,
    FLUSH_RETURN_COUNT = 84,
};

constexpr uint32_t PREFETCH_DEBUG_NUM_COUNTERS = 85u;

#if defined(PREFETCH_DEBUG_COUNTERS_L1)

FORCE_INLINE volatile uint32_t* prefetch_debug_slot(PrefetchDebugCounter idx) {
    return reinterpret_cast<volatile uint32_t*>(PREFETCH_DEBUG_COUNTERS_L1 +
                                                static_cast<uint32_t>(idx) * sizeof(uint32_t));
}

FORCE_INLINE void prefetch_debug_inc_no_publish(PrefetchDebugCounter idx) {
    (*prefetch_debug_slot(idx))++;
}

FORCE_INLINE void prefetch_debug_set_no_publish(PrefetchDebugCounter idx, uint32_t value) {
    *prefetch_debug_slot(idx) = value;
}

FORCE_INLINE void prefetch_debug_mirror_low_no_publish(uint32_t code, uint32_t arg0, uint32_t arg1) {
    prefetch_debug_set_no_publish(PrefetchDebugCounter::PCIE_EARLY_EXIT_CODE, code);
    prefetch_debug_set_no_publish(PrefetchDebugCounter::PCIE_EARLY_EXIT_ARG0, arg0);
    prefetch_debug_set_no_publish(PrefetchDebugCounter::PCIE_EARLY_EXIT_ARG1, arg1);
}

FORCE_INLINE void prefetch_debug_publish(PrefetchDebugCounter idx) {
    const uintptr_t cached_tl1_addr =
        PREFETCH_DEBUG_COUNTERS_L1 + static_cast<uint32_t>(idx) * sizeof(uint32_t);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    const uint32_t flush_line_addr = static_cast<uint32_t>(cached_tl1_addr & ~uintptr_t(63));
    const uint32_t slot_idx = static_cast<uint32_t>(idx);
    prefetch_debug_set_no_publish(PrefetchDebugCounter::FLUSH_TARGET_ADDR, static_cast<uint32_t>(cached_tl1_addr));
    prefetch_debug_set_no_publish(PrefetchDebugCounter::FLUSH_LINE_ADDR, flush_line_addr);
    prefetch_debug_set_no_publish(PrefetchDebugCounter::FLUSH_TARGET_SLOT, slot_idx);
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::FLUSH_ENTER_COUNT);
    prefetch_debug_mirror_low_no_publish(
        PREFETCH_DEBUG_FLUSH_ENTER_MIRROR_CODE, static_cast<uint32_t>(cached_tl1_addr), slot_idx);
#endif
    tl1_publish_flush(cached_tl1_addr);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::FLUSH_RETURN_COUNT);
#endif
}

FORCE_INLINE void prefetch_debug_inc(PrefetchDebugCounter idx) {
    (*prefetch_debug_slot(idx))++;
    prefetch_debug_publish(idx);
}

FORCE_INLINE void prefetch_debug_set(PrefetchDebugCounter idx, uint32_t value) {
    *prefetch_debug_slot(idx) = value;
    prefetch_debug_publish(idx);
}

FORCE_INLINE void prefetch_debug_record_pcie_early_exit(
    PrefetchPcieEarlyExitCode code, uint32_t arg0, uint32_t arg1) {
    prefetch_debug_inc(PrefetchDebugCounter::PCIE_EARLY_EXIT);
    prefetch_debug_set(PrefetchDebugCounter::PCIE_EARLY_EXIT_CODE, static_cast<uint32_t>(code));
    prefetch_debug_set(PrefetchDebugCounter::PCIE_EARLY_EXIT_ARG0, arg0);
    prefetch_debug_set(PrefetchDebugCounter::PCIE_EARLY_EXIT_ARG1, arg1);
}

FORCE_INLINE void prefetch_debug_record_fetchq_return(
    uint32_t inflight_count, uintptr_t cmd_ptr, uintptr_t fence) {
    prefetch_debug_inc(PrefetchDebugCounter::FETCHQ_RETURN);
    prefetch_debug_set(PrefetchDebugCounter::INFLIGHT_AT_RETURN, inflight_count);
    prefetch_debug_set(PrefetchDebugCounter::CMD_PTR_AT_RETURN, static_cast<uint32_t>(cmd_ptr));
    prefetch_debug_set(PrefetchDebugCounter::FENCE_AT_RETURN, static_cast<uint32_t>(fence));
}

FORCE_INLINE void prefetch_debug_record_process_cmd(uint32_t cmd_id, uint32_t stride) {
    prefetch_debug_inc(PrefetchDebugCounter::PROCESS_CMD);
    prefetch_debug_set(
        PrefetchDebugCounter::LAST_CMD_TAG, (cmd_id << 16U) | (stride & 0xffffU));
}

FORCE_INLINE void prefetch_debug_record_fetchq_poll(uint32_t entry_raw) {
    const uint32_t poll_count = *prefetch_debug_slot(PrefetchDebugCounter::FETCHQ_POLL_COUNT);
    prefetch_debug_inc(PrefetchDebugCounter::FETCHQ_POLL_COUNT);
    if (poll_count == 0U) {
        prefetch_debug_set(PrefetchDebugCounter::FIRST_FETCHQ_POLL_RAW, entry_raw);
    }
    prefetch_debug_set(PrefetchDebugCounter::LAST_FETCHQ_POLL_RAW, entry_raw);
}

FORCE_INLINE void prefetch_debug_post_while_milestone(uint32_t milestone_id) {
    prefetch_debug_set(PrefetchDebugCounter::POST_WHILE_LAST_MILESTONE, milestone_id);
    prefetch_debug_inc(PrefetchDebugCounter::POST_WHILE_MILESTONE_COUNT);
}

FORCE_INLINE void prefetch_debug_pre_issue_milestone(uint32_t milestone_id) {
    prefetch_debug_set(PrefetchDebugCounter::PRE_ISSUE_LAST_MILESTONE, milestone_id);
    prefetch_debug_inc(PrefetchDebugCounter::PRE_ISSUE_MILESTONE_COUNT);
}

FORCE_INLINE void prefetch_debug_record_retire_start(uint32_t trid, uint32_t inflight_count) {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::RETIRE_START));
    prefetch_debug_inc(PrefetchDebugCounter::RETIRE_START);
    // Mirror into low slots (2/3/4) first so host probes see retire state even if high slots fail.
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::RETIRE_MIRROR), trid, inflight_count);
    prefetch_debug_set(PrefetchDebugCounter::RETIRE_TRID, trid);
    prefetch_debug_set(PrefetchDebugCounter::RETIRE_INFLIGHT_COUNT, inflight_count);
}

FORCE_INLINE void prefetch_debug_record_post_while_epilogue_enter() {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::EPILOGUE_ENTER));
    prefetch_debug_inc(PrefetchDebugCounter::POST_WHILE_EPILOGUE_ENTER);
}

FORCE_INLINE void prefetch_debug_record_post_issue_loop_exit() {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::POST_ISSUE_LOOP_EXIT));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::POST_ISSUE_LOOP_EXIT);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::ISSUE_LOOP_EXIT_MIRROR),
        *prefetch_debug_slot(PrefetchDebugCounter::POST_ISSUE_LOOP_EXIT),
        0U);
    prefetch_debug_inc(PrefetchDebugCounter::POST_ISSUE_EPILOGUE_DONE);
}

FORCE_INLINE void prefetch_debug_record_post_issue_state(
    bool cmd_ready_loop_top,
    bool cmd_ready_reeval,
    bool has_pending_stall_after,
    uint32_t inflight_count,
    uintptr_t cmd_ptr,
    uintptr_t fence) {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::POST_ISSUE_RECORD));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::POST_ISSUE_RECORD_ENTER);

    // Low-slot mirror first (no publish) so host sees post-issue state even if high-slot flush wedges.
    const uint32_t flags =
        (cmd_ready_loop_top ? 1U : 0U) | (cmd_ready_reeval ? 2U : 0U) | (has_pending_stall_after ? 4U : 0U);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::ISSUE_DONE_MIRROR), flags, inflight_count);

    prefetch_debug_inc(PrefetchDebugCounter::ISSUE_LOOP_DONE);
    prefetch_debug_set(PrefetchDebugCounter::CMD_READY_AT_ISSUE_DONE, cmd_ready_loop_top ? 1U : 0U);
    prefetch_debug_set(PrefetchDebugCounter::CMD_READY_REEVAL, cmd_ready_reeval ? 1U : 0U);
    prefetch_debug_set(PrefetchDebugCounter::HAS_PENDING_STALL_AFTER, has_pending_stall_after ? 1U : 0U);
    prefetch_debug_set(PrefetchDebugCounter::INFLIGHT_AT_ISSUE_DONE, inflight_count);
    prefetch_debug_set(PrefetchDebugCounter::CMD_PTR_AT_ISSUE_DONE, static_cast<uint32_t>(cmd_ptr));
    prefetch_debug_set(PrefetchDebugCounter::FENCE_AT_ISSUE_DONE, static_cast<uint32_t>(fence));
}

FORCE_INLINE void prefetch_debug_record_branch_mirror(uint32_t arg0, uint32_t arg1) {
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::BRANCH_MIRROR), arg0, arg1);
}

FORCE_INLINE void prefetch_debug_record_fetchq_return_mirror(uint32_t arg0, uint32_t arg1) {
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::FETCHQ_RETURN_MIRROR), arg0, arg1);
}

FORCE_INLINE void prefetch_debug_mirror_issue_while_cond(
    uint32_t waypoint, uint32_t inflight_count, uint32_t fetch_size, bool stall_flag, bool would_continue) {
    const uint32_t arg0 = (waypoint << 24U) | (would_continue ? (1U << 17U) : 0U) |
                          (stall_flag ? (1U << 16U) : 0U) | (inflight_count & 0xFFFFU);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::ISSUE_WHILE_COND_MIRROR), arg0, fetch_size);
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::ISSUE_WHILE_COND_CHECK_COUNT);
    prefetch_debug_set(PrefetchDebugCounter::ISSUE_WHILE_COND_LAST_WOULD_CONTINUE, would_continue ? 1U : 0U);
    prefetch_debug_set(PrefetchDebugCounter::ISSUE_WHILE_COND_LAST_INFLIGHT, inflight_count);
}

FORCE_INLINE void prefetch_debug_record_issue_ok(uint32_t inflight_count) {
    prefetch_debug_inc(PrefetchDebugCounter::ISSUE_OK);
    const uint32_t issue_ok_count = *prefetch_debug_slot(PrefetchDebugCounter::ISSUE_OK);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::ISSUE_OK_INFLIGHT_MIRROR),
        inflight_count,
        issue_ok_count);
    prefetch_debug_set(PrefetchDebugCounter::INFLIGHT_AT_EACH_ISSUE_OK, inflight_count);
}

FORCE_INLINE void prefetch_debug_mirror_issue_while_tail(
    uint32_t waypoint, uint32_t inflight_count, uint32_t fetch_size, bool stall_flag) {
    const uint32_t arg0 =
        (waypoint << 24U) | (stall_flag ? (1U << 16U) : 0U) | (inflight_count & 0xFFFFU);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::ISSUE_WHILE_TAIL_MIRROR), arg0, fetch_size);
}

FORCE_INLINE void prefetch_debug_record_post_stall_refresh_done() {
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::POST_STALL_REFRESH_WRITTEN);
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::STALL_REFRESH_MIRROR),
        *prefetch_debug_slot(PrefetchDebugCounter::POST_STALL_REFRESH_WRITTEN),
        *prefetch_debug_slot(PrefetchDebugCounter::POST_STALL_REFRESH_DONE));
    prefetch_debug_inc(PrefetchDebugCounter::POST_STALL_REFRESH_DONE);
}

FORCE_INLINE void prefetch_debug_record_retire_pre_branch(bool cmd_ready, bool cmd_ready_reeval) {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::RETIRE_PRE_BRANCH));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::RETIRE_PRE_ENTER);
    prefetch_debug_record_branch_mirror(
        *prefetch_debug_slot(PrefetchDebugCounter::RETIRE_PRE_ENTER),
        (cmd_ready ? 1U : 0U) | (cmd_ready_reeval ? 2U : 0U));
    prefetch_debug_inc(PrefetchDebugCounter::RETIRE_BRANCH_REACHED);
}

FORCE_INLINE void prefetch_debug_record_retire_path_entry() {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::RETIRE_PATH_ENTRY));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::RETIRE_PATH_WRITTEN);
    prefetch_debug_record_branch_mirror(
        *prefetch_debug_slot(PrefetchDebugCounter::RETIRE_PATH_WRITTEN),
        *prefetch_debug_slot(PrefetchDebugCounter::RETIRE_PRE_ENTER));
    prefetch_debug_inc(PrefetchDebugCounter::RETIRE_PATH_ENTER);
}

FORCE_INLINE void prefetch_debug_record_fetchq_return_path(bool cmd_ready, bool cmd_ready_reeval) {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::FETCHQ_RETURN_PATH));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::FETCHQ_RETURN_WRITTEN);
    prefetch_debug_record_fetchq_return_mirror(
        *prefetch_debug_slot(PrefetchDebugCounter::FETCHQ_RETURN_WRITTEN),
        (cmd_ready ? 1U : 0U) | (cmd_ready_reeval ? 2U : 0U));
    prefetch_debug_inc(PrefetchDebugCounter::FETCHQ_RETURN_ENTER);
}

FORCE_INLINE void prefetch_debug_record_retire_barrier_enter(uint32_t trid, uint32_t inflight_count) {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::BARRIER_ENTER));
    prefetch_debug_mirror_low_no_publish(
        static_cast<uint32_t>(PrefetchPcieEarlyExitCode::RETIRE_BARRIER_ENTER_MIRROR), trid, inflight_count);
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::RETIRE_BARRIER_ENTER_WRITTEN);
    prefetch_debug_set(PrefetchDebugCounter::RETIRE_BARRIER_TRID_AT_ENTER, trid);
    prefetch_debug_set(PrefetchDebugCounter::RETIRE_BARRIER_INFLIGHT_AT_ENTER, inflight_count);
    prefetch_debug_set(
        PrefetchDebugCounter::RETIRE_BARRIER_PHASE, PREFETCH_DEBUG_RETIRE_BARRIER_PHASE_ENTERED);
    prefetch_debug_inc(PrefetchDebugCounter::RETIRE_PRE_BARRIER);
}

FORCE_INLINE void prefetch_debug_record_retire_barrier_returned() {
    prefetch_debug_post_while_milestone(static_cast<uint32_t>(PrefetchPostWhileMilestone::BARRIER_RETURNED));
    prefetch_debug_inc_no_publish(PrefetchDebugCounter::RETIRE_BARRIER_RETURNED_WRITTEN);
    prefetch_debug_set(
        PrefetchDebugCounter::RETIRE_BARRIER_PHASE, PREFETCH_DEBUG_RETIRE_BARRIER_PHASE_RETURNED);
    prefetch_debug_inc(PrefetchDebugCounter::RETIRE_POST_BARRIER);
}

FORCE_INLINE void prefetch_debug_init() {
    prefetch_debug_set(PrefetchDebugCounter::DEBUG_BUILD_ID, PREFETCH_DEBUG_BUILD_ID);
}

#else

FORCE_INLINE void prefetch_debug_inc(PrefetchDebugCounter) {}
FORCE_INLINE void prefetch_debug_inc_no_publish(PrefetchDebugCounter) {}
FORCE_INLINE void prefetch_debug_set(PrefetchDebugCounter, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_pcie_early_exit(PrefetchPcieEarlyExitCode, uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_fetchq_return(uint32_t, uintptr_t, uintptr_t) {}
FORCE_INLINE void prefetch_debug_record_process_cmd(uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_fetchq_poll(uint32_t) {}
FORCE_INLINE void prefetch_debug_record_retire_start(uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_post_issue_state(bool, bool, bool, uint32_t, uintptr_t, uintptr_t) {}
FORCE_INLINE void prefetch_debug_record_branch_mirror(uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_fetchq_return_mirror(uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_issue_ok(uint32_t) {}
FORCE_INLINE void prefetch_debug_mirror_issue_while_cond(uint32_t, uint32_t, uint32_t, bool, bool) {}
FORCE_INLINE void prefetch_debug_mirror_issue_while_tail(uint32_t, uint32_t, uint32_t, bool) {}
FORCE_INLINE void prefetch_debug_post_while_milestone(uint32_t) {}
FORCE_INLINE void prefetch_debug_pre_issue_milestone(uint32_t) {}
FORCE_INLINE void prefetch_debug_record_post_stall_refresh_done() {}
FORCE_INLINE void prefetch_debug_record_post_while_epilogue_enter() {}
FORCE_INLINE void prefetch_debug_record_post_issue_loop_exit() {}
FORCE_INLINE void prefetch_debug_record_retire_pre_branch(bool, bool) {}
FORCE_INLINE void prefetch_debug_record_retire_path_entry() {}
FORCE_INLINE void prefetch_debug_record_fetchq_return_path(bool, bool) {}
FORCE_INLINE void prefetch_debug_record_retire_barrier_enter(uint32_t, uint32_t) {}
FORCE_INLINE void prefetch_debug_record_retire_barrier_returned() {}
FORCE_INLINE void prefetch_debug_init() {}

#endif
