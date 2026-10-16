// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/shared/p2p_latency_layout.h -- p2p_latency mailbox and result region
 * layout.
 *
 * Every participant owns a mailbox and result region in its own local memory;
 * memory-sourced backends also own staging:
 *
 *   inbound mailbox      peer writes; I poll and echo
 *   local staging        host seeds; source when I initiate, if needed
 *   outbound result      I write; host reads
 *
 * See docs/benchmarks/design.md, §Roles and mailbox model.
 *
 * Layout is suite-owned rather than case-owned because every member of
 * p2p_latency uses the same region layouts. A case still
 * chooses the concrete addresses each region lives at -- the host resolves and
 * tells them via the argument block.
 *
 * Invariants pinned below:
 *
 *   - mailbox and staging are P2P_LATENCY_PAYLOAD_MAX bytes each, cache-line
 *     aligned, so a host-side whole-line write reaches the ECC bytes and a
 *     peer's later 4-byte token store is not the first thing to touch them.
 *
 *   - the token slot is the FINAL 4 bytes of a transfer, regardless of
 *     payload size. A 4-byte transfer degenerates to a token-only exchange
 *     with poll offset zero. A larger transfer echoes the whole received
 *     buffer, and the token is still where the responder wrote it last.
 *
 *   - the raw-deltas array is a fixed P2P_LATENCY_MAX_SAMPLES slots, zero-
 *     padded beyond the samples_completed count in the enclosing phase record.
 *     The host reads only up to samples_completed.
 */
#ifndef X280_P2P_LATENCY_LAYOUT_H
#define X280_P2P_LATENCY_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#include "p2p_latency_frontend.h"
#include "ubench_backend_ids.h"
#include "ubench_section_load.h"
#include "ubench_units.h"

/* --------------------------------------------------------------------------
 * Payload envelope
 * --------------------------------------------------------------------------
 * The largest payload this suite carries is fixed at the layout level so the
 * per-participant mailbox and any required staging have a determinate size. A case
 * requesting a smaller payload writes only the low bytes and the token in the
 * final 4 slots; memory-sourced responders echo the whole mailbox. The
 * x280_peer_tile, tensix_peer_tile, erisc_peer_tile and drisc_peer_tile X280 backends store the token directly
 * for both roles without staging.
 *
 * Chosen at 256 bytes: enough for the design's anchor sizes (4, 64,
 * "current production", "configured maximum") without wasting LIM.
 */
#define P2P_LATENCY_PAYLOAD_MAX 256u

/* Region sizes, whole cache lines. */
#define P2P_LATENCY_MAILBOX_BYTES P2P_LATENCY_PAYLOAD_MAX
#define P2P_LATENCY_STAGING_BYTES P2P_LATENCY_PAYLOAD_MAX

/* Token offset within a mailbox or staging region: the FINAL 4 bytes of the
 * payload, not of the whole region. A responder reading it does so with the
 * payload_size the host told it, so the pointer is
 *     region_base + payload_size - 4
 * A degenerate 4-byte case reads at region_base + 0. */
#if defined(__cplusplus)
constexpr uint32_t p2p_latency_token_offset(uint32_t payload_size) { return payload_size - 4u; }
#else
static inline uint32_t p2p_latency_token_offset(uint32_t payload_size) { return payload_size - 4u; }
#endif

/* --------------------------------------------------------------------------
 * Per-phase result record
 * --------------------------------------------------------------------------
 * Two phase records per participant: phase 0 runs first_direction, phase 1
 * runs its opposite. Both directions share one allocation and one warm-up
 * discipline; a phase record captures only what that phase produced.
 *
 * raw_deltas[i] holds one participant-local counter delta per sample --
 * cycles for X280 (rdcycle), Metal timestamp ticks for Tensix, or raw ERISC or
 * DRISC wall-clock ticks. The host reads
 * paired arrays without converting CPU cycles to time. X280's secondary
 * array supplies fixed-rate mtime wall deltas; Metal's supplies CPU cycles.
 * It does not subtract deltas across component classes.
 *
 * samples_completed differs from p2p_latency_common_args_t::sample_count only
 * on early exit (TIMEOUT, TRAP): host uses it to know how much of raw_deltas
 * is valid. warmup_progress carries the last observed warm-up iteration if
 * the phase left in WARMUP; otherwise its value is undefined.
 */
typedef struct {
    uint32_t phase_ordinal;     /* 0 or 1 */
    uint32_t direction;         /* p2p_latency_direction_t for this phase */
    uint32_t role;              /* p2p_latency_role_t under this direction */
    uint32_t sample_kind;       /* p2p_latency_sample_kind_t (RTT/TURN) */
    uint32_t samples_completed; /* <= sample_count; may be 0 on TIMEOUT */
    uint32_t warmup_progress;   /* last warm-up i, if left in WARMUP */
    uint32_t terminal_state;    /* p2p_latency_state_t on phase exit */
    uint32_t reserved;
    uint32_t raw_deltas[P2P_LATENCY_MAX_SAMPLES];
    /* Native 64-bit reads; deltas exceeding UINT32_MAX fail before storage.
     * X280: mtime; Metal: architectural cycle. Same sample as raw_deltas. */
    uint32_t secondary_deltas[P2P_LATENCY_MAX_SAMPLES];
} p2p_latency_phase_record_t;

/* --------------------------------------------------------------------------
 * Per-participant result region
 * --------------------------------------------------------------------------
 * Two phases, plus a small header. Cache-line aligned in aggregate.
 * backend_id lets the host verify the participant compiled the backend it
 * was placed for.
 *
 * path_sentinel is the rendezvous word; zero means the cleared mailbox was
 * never updated.
 */
typedef struct {
    uint32_t backend_id;
    uint32_t reserved;       /* zero */
    uint32_t terminal_state; /* p2p_latency_state_t after both phases */
    uint32_t path_sentinel;  /* word observed at the rendezvous; 0 = none */
    p2p_latency_phase_record_t phases[2];
} p2p_latency_result_t;

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */

UBENCH_LAYOUT_ASSERT(P2P_LATENCY_PAYLOAD_MAX >= 4u, "the token-only case must fit");
UBENCH_LAYOUT_ASSERT(P2P_LATENCY_PAYLOAD_MAX % 4u == 0u, "payload sizes and token offsets are 4-byte aligned");
UBENCH_LAYOUT_ASSERT(P2P_LATENCY_MAILBOX_BYTES % UBENCH_LIM_LINE_BYTES == 0u, "mailbox is whole cache lines");
UBENCH_LAYOUT_ASSERT(P2P_LATENCY_STAGING_BYTES % UBENCH_LIM_LINE_BYTES == 0u, "staging is whole cache lines");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_phase_record_t) == 32u + 8u * P2P_LATENCY_MAX_SAMPLES,
    "phase record size is header + paired deltas");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_result_t) == 16u + 2u * sizeof(p2p_latency_phase_record_t),
    "result region is header + two phase records");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_result_t) < (1u * 1024u * 1024u), "result region fits well inside a 1920 KiB LIM");

#endif /* X280_P2P_LATENCY_LAYOUT_H */
