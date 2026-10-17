// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/transport_bandwidth/transport_bandwidth_common.h -- suite-owned
 * types shared by every transport_bandwidth member.
 *
 * The suite measures the achieved rate of ONE transport leg. The frontend
 * owns the phase order (INIT -> VERIFY_PATH -> WARMUP -> TIMED -> PUBLISH ->
 * DONE, with a TIMED_INJECTION -> DRAIN branch when the INJECTION boundary
 * is selected). A member-specific backend supplies issue(), drain(), and
 * measurement_now().
 *
 * This header carries only what every member needs. Member-specific arguments
 * and identity constants live beside their kernel.
 *
 * See docs/benchmarks/design.md § Transport bandwidth for the
 * semantic contract these types encode.
 */
#ifndef X280_TRANSPORT_BANDWIDTH_COMMON_H
#define X280_TRANSPORT_BANDWIDTH_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "../shared/ubench_backend_ids.h"
#include "../shared/ubench_cookies.h"
#include "../shared/ubench_section_load.h"
#include "../shared/ubench_outcome.h"

/* Suite magic for X280-side section cookies; not a layout version. The
 * Tensix->X280 member uses Metal runtime arguments instead. */
#define TRANSPORT_BW_COOKIE_BASE 0x54584257u /* 'T' 'X' 'B' 'W' */

#if defined(__cplusplus)
constexpr uint32_t transport_bw_cookie_for(ubench_section_t s) {
    return ubench_cookie_for(TRANSPORT_BW_COOKIE_BASE, s);
}
#else
static inline uint32_t transport_bw_cookie_for(ubench_section_t s) {
    return ubench_cookie_for(TRANSPORT_BW_COOKIE_BASE, s);
}
#endif

/* --------------------------------------------------------------------------
 * Measurement boundary
 * --------------------------------------------------------------------------
 * DRAINED is the sustained-throughput baseline: t1 taken after drain(). For
 * tensix_to_x280 that follows every issued transfer's acknowledgement; for
 * x280_to_tensix it follows the stores retiring and one read through the
 * posted window, which does not prove delivery. INJECTION measures issue only and drains outside the
 * bracket, so it describes finite-batch issue rate from a warm, empty
 * transport rather than sustained throughput.
 */
typedef enum { TRANSPORT_BW_BOUNDARY_DRAINED = 0u, TRANSPORT_BW_BOUNDARY_INJECTION = 1u } transport_bw_boundary_t;

/* --------------------------------------------------------------------------
 * State machine
 * --------------------------------------------------------------------------
 * Free-running, published outside the measurement bracket. The host reads it
 * once when it observes a terminal state; a non-DONE terminal names the
 * broken step without adding instrumentation to the transport.
 */
typedef enum {
    TRANSPORT_BW_STATE_INIT = 0u,
    TRANSPORT_BW_STATE_VERIFY_PATH = 1u,
    TRANSPORT_BW_STATE_WARMUP = 2u,
    TRANSPORT_BW_STATE_TIMED_DRAINED = 3u,
    TRANSPORT_BW_STATE_TIMED_INJECTION = 4u,
    TRANSPORT_BW_STATE_DRAIN = 5u, /* only under INJECTION */
    TRANSPORT_BW_STATE_PUBLISH = 6u,
    TRANSPORT_BW_STATE_DONE = 7u,
    TRANSPORT_BW_STATE_PATH_MISMATCH = 8u,
    TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT = 9u, /* another participant never reached the rendezvous */
    /* 10 is unused: no member traps. */
    TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH = 11u,
    TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED = 12u /* the TLB clear did not read back zero */
} transport_bw_state_t;

/* Map every phase explicitly so -Wswitch catches additions. */
static inline ubench_outcome_t transport_bw_outcome(transport_bw_state_t st) {
    switch (st) {
        case TRANSPORT_BW_STATE_DONE: return UBENCH_OUTCOME_OK;
        case TRANSPORT_BW_STATE_PATH_MISMATCH: return UBENCH_OUTCOME_PATH_MISMATCH;
        case TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT: return UBENCH_OUTCOME_TIMEOUT;
        case TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED: return UBENCH_OUTCOME_TRAP;
        case TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
        /* Non-terminal; DRAIN means an injection run still owes cleanup. */
        case TRANSPORT_BW_STATE_INIT:
        case TRANSPORT_BW_STATE_VERIFY_PATH:
        case TRANSPORT_BW_STATE_WARMUP:
        case TRANSPORT_BW_STATE_TIMED_DRAINED:
        case TRANSPORT_BW_STATE_TIMED_INJECTION:
        case TRANSPORT_BW_STATE_DRAIN:
        case TRANSPORT_BW_STATE_PUBLISH: return UBENCH_OUTCOME_IN_PROGRESS;
    }
    return UBENCH_OUTCOME_IN_PROGRESS;
}

/* --------------------------------------------------------------------------
 * Protocol arguments (common section)
 * --------------------------------------------------------------------------
 * transfer_size is application bytes per issue. transfer_count is the number
 * of timed issues. The host validates both are positive and computes
 * logical_bytes = uint64_t(N) * uint64_t(SIZE) in 64 bits before comparing
 * against member-specific limits.
 *
 * run_nonce carries into the member's path-verification sentinel so a
 * stale word from an earlier session cannot answer this run's verify.
 */
typedef struct {
    uint32_t transfer_size;
    uint32_t warmup_count;
    uint32_t transfer_count;
    uint32_t boundary;  /* transport_bw_boundary_t */
    uint32_t run_nonce; /* nonzero; used in path sentinel */
    uint32_t reserved;
} transport_bw_protocol_args_t;

/* --------------------------------------------------------------------------
 * Result region (device-written, host-read)
 * --------------------------------------------------------------------------
 * The device stores raw counts and up to two raw batch deltas in its own
 * counter ticks.
 *
 * Two-counter contract:
 *   - raw_batch_delta is the initiator's native wall-counter delta.
 *   - secondary_batch_delta is its raw architectural CPU-cycle delta over
 *     the same operation; the reads are ordered, not simultaneous.
 *   - CPU cycles are not converted to time.
 *
 * A result is invalid when terminal_state is not DONE, when the byte
 * definition is ambiguous (transfer_size == 0), or when raw_batch_delta
 * is zero. UINT64_MAX in either delta means a counter read failed or
 * measurement_delta refused the interval.
 */
typedef struct {
    uint32_t backend_id;
    uint32_t boundary;              /* as issued */
    uint32_t terminal_state;        /* transport_bw_state_t */
    uint32_t transfer_size;         /* echoed for the host reduction */
    uint32_t transfer_count;        /* echoed */
    uint32_t warmup_completed;      /* full W = expected */
    uint64_t logical_bytes;         /* uint64_t(N) * uint64_t(SIZE) */
    uint64_t raw_batch_delta;       /* primary counter (wall clock where available) */
    uint64_t secondary_batch_delta; /* raw native architectural CPU cycles */
    uint32_t path_sentinel;         /* what the verify step read back */
    uint32_t reserved;              /* zero */
} transport_bw_result_t;

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */
UBENCH_LAYOUT_ASSERT(sizeof(transport_bw_protocol_args_t) == 24u, "protocol args are 24 bytes: 6 x u32");
UBENCH_LAYOUT_ASSERT(offsetof(transport_bw_protocol_args_t, transfer_size) == 0u, "transfer_size is the first arg");
UBENCH_LAYOUT_ASSERT(offsetof(transport_bw_protocol_args_t, boundary) == 12u, "boundary is the fourth u32");

UBENCH_LAYOUT_ASSERT(
    sizeof(transport_bw_result_t) == 56u, "result is 56 bytes: 6 x u32 + 3 x u64 + 2 x u32 (path_sentinel + reserved)");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_result_t, logical_bytes) == 24u,
    "u64 fields begin at natural alignment after the u32 header");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_result_t, raw_batch_delta) == 32u, "raw_batch_delta immediately follows logical_bytes");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_result_t, secondary_batch_delta) == 40u,
    "secondary_batch_delta immediately follows raw_batch_delta");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_result_t, path_sentinel) == 48u, "path_sentinel follows the three u64 counters");
UBENCH_LAYOUT_ASSERT(offsetof(transport_bw_result_t, reserved) == 52u, "reserved is the trailing u32 slot");

#endif /* X280_TRANSPORT_BANDWIDTH_COMMON_H */
