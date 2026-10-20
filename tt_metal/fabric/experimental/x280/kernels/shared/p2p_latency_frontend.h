// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/shared/p2p_latency_frontend.h -- suite-owned frontend types shared
 * by every participant of the p2p_latency benchmark.
 *
 * Every participant, on every component class, is compiled against this one
 * header: X280 images, Metal peer kernels, and host cases
 * that dispatches them. The design's kernel frontend/backend split lives on
 * top of the types declared here (see p2p_latency_frontend_cxx.hpp for the
 * templated loop; this header carries only the layout).
 *
 * See docs/benchmarks/design.md, §Peer-to-peer latency for the
 * semantic contract these types encode.
 *
 * Layout invariants pinned by the assertions at the foot of this file:
 *
 *   - p2p_latency_common_args_t is a 32-byte POD. A 4-byte cookie precedes it
 *     and a 4-byte cookie separates it from the backend section, so a device
 *     that reads it lands each field at a known offset without alignment
 *     surprises.
 *
 *   - warmup_max_poll_cycles is a whole 64-bit field, not two halves. Reading
 *     a partial value here would cap warmup polling short of what the host
 *     packed and mask a slow rendezvous as a timeout. Host packs it as one
 *     u64 store into cache-line aligned LIM; firmware reads it as one
 *     u64 load.
 *
 * Every device-side status field is a 32-bit word whose low 16 bits carry the
 * state and whose next 8 bits carry the current direction. The host reads it
 * exactly once when it observes a terminal state, so it is free-running
 * (published outside measurement brackets) rather than fenced.
 */
#ifndef X280_P2P_LATENCY_FRONTEND_H
#define X280_P2P_LATENCY_FRONTEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ubench_cookies.h"
#include "ubench_outcome.h"
#include "ubench_section_load.h"

/* Suite magic for section cookies; not a layout version. */
#define P2P_LATENCY_COOKIE_BASE 0x50324C42u /* 'P' '2' 'L' 'B' */

#if defined(__cplusplus)
constexpr uint32_t p2p_latency_cookie_for(ubench_section_t s) { return ubench_cookie_for(P2P_LATENCY_COOKIE_BASE, s); }
#else
static inline uint32_t p2p_latency_cookie_for(ubench_section_t s) {
    return ubench_cookie_for(P2P_LATENCY_COOKIE_BASE, s);
}
#endif

/* --------------------------------------------------------------------------
 * Participant identity
 * --------------------------------------------------------------------------
 * A p2p_latency invocation has exactly two participants. x280_same_tile pairs two
 * X280 harts on one tile; cross-class members pair an X280 image with a
 * Tensix, idle-ERISC or DRISC kernel.
 * "self" is who this device thinks it is, not who initiates first -- direction
 * is the runtime axis.
 */
typedef enum { P2P_LATENCY_PARTICIPANT_A = 0u, P2P_LATENCY_PARTICIPANT_B = 1u } p2p_latency_participant_t;

typedef enum { P2P_LATENCY_DIRECTION_A_TO_B = 0u, P2P_LATENCY_DIRECTION_B_TO_A = 1u } p2p_latency_direction_t;

typedef enum { P2P_LATENCY_ROLE_INITIATOR = 0u, P2P_LATENCY_ROLE_RESPONDER = 1u } p2p_latency_role_t;

typedef enum {
    /* payload_size == 4: the token IS the payload, no branch on width. */
    P2P_LATENCY_TRANSFER_TOKEN_ONLY = 0u,
    /* payload_size > 4: buffer plus token in its final 4 bytes. */
    P2P_LATENCY_TRANSFER_PAYLOAD_WITH_TOKEN = 1u
} p2p_latency_transfer_shape_t;

typedef enum { P2P_LATENCY_POLL_TIGHT = 0u, P2P_LATENCY_POLL_BACKOFF = 1u } p2p_latency_poll_mode_t;

/* Backend-specific: X280 has no established departure-only primitive and
 * rejects WAIT_FOR_DEPARTURE. Tensix maps all three to its NOC issue, flush,
 * and barrier operations. x280_same_tile self-view maps FIRE_AND_FORGET and
 * WAIT_FOR_COMPLETION to a local store with and without a visibility fence;
 * WAIT_FOR_DEPARTURE has no meaning locally. */
typedef enum {
    P2P_LATENCY_TRANSMIT_FIRE_AND_FORGET = 0u,
    P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE = 1u,
    P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION = 2u
} p2p_latency_transmit_mode_t;

/* Which of a phase record's per-sample deltas this record carries. Initiator
 * publishes RTT (transmit -> observed reply); responder publishes TURN
 * (observed request -> transmit issued). The suite does not halve an RTT to
 * infer one-way. */
typedef enum { P2P_LATENCY_SAMPLE_KIND_RTT = 0u, P2P_LATENCY_SAMPLE_KIND_TURN = 1u } p2p_latency_sample_kind_t;

/* --------------------------------------------------------------------------
 * State machine (published in the result's and each phase record's
 * terminal_state)
 * --------------------------------------------------------------------------
 * Progressing forward only. A phase record carries the direction it ran, so
 * the host log names a stuck direction without adding instrumentation to the
 * transport.
 */
typedef enum {
    P2P_LATENCY_STATE_INIT = 0u,
    P2P_LATENCY_STATE_RENDEZVOUS = 1u,
    P2P_LATENCY_STATE_WARMUP = 2u,
    P2P_LATENCY_STATE_TIMED = 3u,
    /* 4 is unused: a phase publishes as it ends, with no state of its own. */
    P2P_LATENCY_STATE_DONE = 5u,
    P2P_LATENCY_STATE_TIMEOUT = 6u,
    P2P_LATENCY_STATE_TRAP = 7u,
    P2P_LATENCY_STATE_ARG_LAYOUT_MISMATCH = 8u
} p2p_latency_state_t;

/* Map every phase explicitly so -Wswitch catches additions. */
static inline ubench_outcome_t p2p_latency_outcome(p2p_latency_state_t st) {
    switch (st) {
        case P2P_LATENCY_STATE_DONE: return UBENCH_OUTCOME_OK;
        case P2P_LATENCY_STATE_TIMEOUT: return UBENCH_OUTCOME_TIMEOUT;
        case P2P_LATENCY_STATE_TRAP: return UBENCH_OUTCOME_TRAP;
        case P2P_LATENCY_STATE_ARG_LAYOUT_MISMATCH: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
        /* Non-terminal; rendezvous provides this suite's path check. */
        case P2P_LATENCY_STATE_INIT:
        case P2P_LATENCY_STATE_RENDEZVOUS:
        case P2P_LATENCY_STATE_WARMUP:
        case P2P_LATENCY_STATE_TIMED: return UBENCH_OUTCOME_IN_PROGRESS;
    }
    return UBENCH_OUTCOME_IN_PROGRESS;
}

/* --------------------------------------------------------------------------
 * Common fields shared by every flat member section.
 * --------------------------------------------------------------------------
 * The host validates upstream that
 *
 *     run_nonce + 2 * (warmup_count + sample_count) - 1 <= UINT32_MAX
 *
 * so the kernel can increment a 32-bit token in its hot loop without a wrap
 * branch (see §Roles and mailbox model). warmup_max_poll_cycles is
 * per-participant because two counter domains may disagree on tick length.
 * Native unsigned subtraction handles counter wrap inside a poll_for.
 */
typedef struct {
    uint32_t self;            /* p2p_latency_participant_t */
    uint32_t first_direction; /* p2p_latency_direction_t */
    uint32_t payload_size;    /* bytes; 4 or >4 */
    uint32_t warmup_count;    /* W >= 1 */
    uint32_t sample_count;    /* N > 0, <= P2P_LATENCY_MAX_SAMPLES */
    uint32_t run_nonce;       /* non-zero; token base */
    uint64_t warmup_max_poll_cycles;
} p2p_latency_common_args_t;

/* Suite-level bound on the raw-deltas array carried in each phase record.
 * A case may run fewer samples; the array is zero-padded beyond
 * samples_completed. Chosen at 4096 to keep two-participant, two-phase
 * result storage under 128 KiB and comfortably inside a 1920 KiB LIM. */
#define P2P_LATENCY_MAX_SAMPLES 4096u

/**
 * Whether every participant can run the common section `c`: nonzero warm-up
 * and sample counts with sample_count <= P2P_LATENCY_MAX_SAMPLES, a nonzero
 * run_nonce and warmup_max_poll_cycles, a token-only payload (payload_size 4,
 * the only one the participants implement), a known first_direction, and a
 * final token run_nonce + 2 * (warmup_count + sample_count) - 1 that fits in
 * 32 bits. `self` is not checked: each participant checks it against the role
 * it plays.
 */
static inline bool p2p_latency_common_valid(const p2p_latency_common_args_t* c) {
    if (c->warmup_count == 0u || c->sample_count == 0u || c->sample_count > P2P_LATENCY_MAX_SAMPLES) {
        return false;
    }
    if (c->run_nonce == 0u || c->warmup_max_poll_cycles == 0u || c->payload_size != 4u) {
        return false;
    }
    if (c->first_direction != (uint32_t)P2P_LATENCY_DIRECTION_A_TO_B &&
        c->first_direction != (uint32_t)P2P_LATENCY_DIRECTION_B_TO_A) {
        return false;
    }
    const uint64_t final_token = (uint64_t)c->run_nonce + 2ull * ((uint64_t)c->warmup_count + c->sample_count) - 1ull;
    return final_token <= UINT32_MAX;
}

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */

UBENCH_LAYOUT_ASSERT(sizeof(p2p_latency_common_args_t) == 32, "p2p_latency_common_args_t is a 32-byte POD");
UBENCH_LAYOUT_ASSERT(offsetof(p2p_latency_common_args_t, self) == 0, "self is the first field");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_common_args_t, warmup_max_poll_cycles) == 24, "warmup_max_poll_cycles is the trailing u64");
UBENCH_LAYOUT_ASSERT(P2P_LATENCY_MAX_SAMPLES > 0u, "sample count bound is positive");

#endif /* X280_P2P_LATENCY_FRONTEND_H */
