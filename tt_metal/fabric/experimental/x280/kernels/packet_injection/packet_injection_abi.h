// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/packet_injection/packet_injection_abi.h
 *
 * Wire ABI, constants, enumerations, POD layouts, offset assertions, and
 * checked sizing formulas for the Tensix -> X280 packet injection characterization.
 * Compatible with C11 and C++17. Contains no Metal or host includes.
 */
#ifndef X280_PACKET_INJECTION_ABI_H
#define X280_PACKET_INJECTION_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "../shared/core_sync_layout.h"
#include "../shared/credit_counter.h"
#include "../shared/ubench_outcome.h"
#include "../shared/ubench_units.h"
#include "../../include/x280_noc_coord.h"

#if defined(__cplusplus)
#define PI_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define PI_ALIGNAS(n) alignas(n)
#define PI_ALIGNOF(t) alignof(t)
#define PI_CONSTEXPR constexpr
#else
#define PI_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#define PI_ALIGNAS(n) _Alignas(n)
#define PI_ALIGNOF(t) _Alignof(t)
#define PI_CONSTEXPR
#endif

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */
#define PI_ABI_MAGIC 0x5049u /* 'P' 'I' */
#define PI_SLOT_BYTES 128u
#define PI_MAX_TILES 140u
#define PI_MAX_WORKERS 280u
#define PI_LIM_LAYOUT 1u
#define PI_DEFAULT_POLL_CADENCE 1024u
#define PI_TIMER_MAX_ATTEMPTS 16u
#define PI_ARENA_BYTES 917376u
#define PI_CACHE_LIMIT_BYTES 131072u /* one reserved 128-KiB way */

/* Byte offsets of the fixed records inside one worker's L1 slice.
 * Each is a full cache line because X280 and the owning RISC touch them
 * concurrently. */
#define PI_L1_CONSUMED_OFFSET 0u       /* X280-written consumed mirror, 64 B */
#define PI_L1_CONTROL_OFFSET 64u       /* X280-written pi_control_t line, 64 B */
#define PI_L1_PRODUCED_SRC_OFFSET 128u /* stable produced-progress source, 64 B */
#define PI_L1_READY_SRC_OFFSET 192u    /* stable pi_ready_t source, 64 B */
#define PI_L1_RESULT_OFFSET 256u       /* pi_producer_result_t, 128 B */
#define PI_L1_STAGING_OFFSET 384u      /* pi_slot_t staging, 128 B * D */

/* Base arena size that pi_compute_layout() must reproduce for
 * P=280, T=140, S=16. Proved by the host layout tests, not by this header. */
#define PI_ARENA_BASE_GOLDEN_P280_T140_S16 619712u

/* --------------------------------------------------------------------------
 * Modes and flags
 * -------------------------------------------------------------------------- */
typedef enum { PI_MODE_LATENCY = 1u, PI_MODE_THROUGHPUT = 2u, PI_MODE_BACKPRESSURE = 3u } pi_mode_t;

/* --------------------------------------------------------------------------
 * Control (X280 -> Tensix producer L1)
 * -------------------------------------------------------------------------- */
typedef enum { PI_CONTROL_RUN = 0u, PI_CONTROL_STOP = 1u } pi_control_t;

/* --------------------------------------------------------------------------
 * Ready state (Tensix producer -> X280 LIM mirror)
 * -------------------------------------------------------------------------- */
typedef enum { PI_READY_NONE = 0u, PI_READY_DONE = 1u, PI_READY_FAILED = 2u } pi_ready_t;

/* --------------------------------------------------------------------------
 * Errors
 * -------------------------------------------------------------------------- */
typedef enum {
    PI_ERROR_NONE = 0u,
    PI_ERROR_BAD_ARGS = 1u,
    PI_ERROR_MODE_MISMATCH = 2u,
    PI_ERROR_BAD_PLACEMENT = 3u,
    PI_ERROR_CAPACITY = 4u,
    PI_ERROR_COUNTER_DISTANCE = 5u,
    PI_ERROR_PHASE_OVERRUN = 6u,
    PI_ERROR_TIMEOUT = 7u,
    PI_ERROR_STOPPED = 8u,
    PI_ERROR_TIMER_RETRY_EXHAUSTED = 9u,
    PI_ERROR_TIMER_DELTA = 10u,
    PI_ERROR_WINDOW_CONFIG = 11u,
    PI_ERROR_WINDOW_CLEAR = 12u,
    PI_ERROR_PIN_POLICY = 13u,
    PI_ERROR_PEER_FAILED = 15u,
    PI_ERROR_HOST_IO = 16u,
    PI_ERROR_PIN_WAYENABLE = 17u,
    PI_ERROR_PIN_WAYMASK = 18u,
    PI_ERROR_PIN_SHAPE = 19u,
    PI_ERROR_LAYOUT_INVARIANT = 20u
} pi_error_t;

/* 14 is unused. The three PI_ERROR_PIN_* causes above carry the
 * x280_pinned_input_check_policy() codes of the same name; PI_ERROR_PIN_POLICY
 * remains for the pinned failures that name no cause -- a refused preparation,
 * a refused teardown, and a policy left uncertain by a partial one.
 *
 * Code and phase discriminate those three, so they need no codes of their own:
 * PI_ERROR_PIN_POLICY at PI_PHASE_INIT is a refused preparation and at
 * PI_PHASE_CLEANUP a refused teardown. The uncertain policy can never be the
 * code a run reports -- the preparation that sets it records at PI_PHASE_INIT
 * on the way out, and the first record of a run wins.
 *
 * PI_ERROR_LAYOUT_INVARIANT is pi_compute_layout() finding its own arithmetic
 * disagreeing with its own postcondition. It accuses no argument, which is why
 * it is not PI_ERROR_BAD_ARGS: reaching a report it means this implementation
 * is broken, and nothing else can produce it. */

/* --------------------------------------------------------------------------
 * Phases
 * -------------------------------------------------------------------------- */
typedef enum {
    PI_PHASE_INIT = 0u,
    PI_PHASE_WAIT_WARMUP = 1u,
    PI_PHASE_WARMUP = 2u,
    PI_PHASE_WAIT_MEASURE = 3u,
    PI_PHASE_MEASURE = 4u,
    PI_PHASE_DRAIN = 5u,
    PI_PHASE_EXPORT = 6u,
    PI_PHASE_WAIT_COLLECT = 7u,
    PI_PHASE_CLEANUP = 8u,
    PI_PHASE_DONE = 9u,
    PI_PHASE_FAILED = 10u
} pi_phase_t;

/* The shared outcome of an actor's phase and error: DONE is OK, FAILED its
 * error's, every other phase IN_PROGRESS. A refusal of the arguments, the
 * placement or the pinned shape is ARG_LAYOUT_MISMATCH; TIMEOUT and STOPPED
 * (a host STOP) are TIMEOUT; every runtime fault is TRAP, as are NONE under
 * FAILED and an unknown phase or error. */
static inline ubench_outcome_t pi_outcome(pi_phase_t phase, pi_error_t error) {
    switch (phase) {
        case PI_PHASE_DONE: return UBENCH_OUTCOME_OK;
        case PI_PHASE_INIT:
        case PI_PHASE_WAIT_WARMUP:
        case PI_PHASE_WARMUP:
        case PI_PHASE_WAIT_MEASURE:
        case PI_PHASE_MEASURE:
        case PI_PHASE_DRAIN:
        case PI_PHASE_EXPORT:
        case PI_PHASE_WAIT_COLLECT:
        case PI_PHASE_CLEANUP: return UBENCH_OUTCOME_IN_PROGRESS;
        case PI_PHASE_FAILED:
            switch (error) {
                case PI_ERROR_BAD_ARGS:
                case PI_ERROR_MODE_MISMATCH:
                case PI_ERROR_BAD_PLACEMENT:
                case PI_ERROR_CAPACITY:
                case PI_ERROR_PIN_POLICY:
                case PI_ERROR_PIN_WAYENABLE:
                case PI_ERROR_PIN_WAYMASK:
                case PI_ERROR_PIN_SHAPE: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
                case PI_ERROR_TIMEOUT:
                case PI_ERROR_STOPPED: return UBENCH_OUTCOME_TIMEOUT;
                case PI_ERROR_NONE:
                case PI_ERROR_COUNTER_DISTANCE:
                case PI_ERROR_PHASE_OVERRUN:
                case PI_ERROR_TIMER_RETRY_EXHAUSTED:
                case PI_ERROR_TIMER_DELTA:
                case PI_ERROR_WINDOW_CONFIG:
                case PI_ERROR_WINDOW_CLEAR:
                case PI_ERROR_PEER_FAILED:
                case PI_ERROR_HOST_IO:
                case PI_ERROR_LAYOUT_INVARIANT: return UBENCH_OUTCOME_TRAP;
            }
            return UBENCH_OUTCOME_TRAP;
    }
    return UBENCH_OUTCOME_TRAP;
}

/* --------------------------------------------------------------------------
 * Actors
 * -------------------------------------------------------------------------- */
typedef enum {
    PI_ACTOR_HOST = 0u,
    PI_ACTOR_X280 = 1u,
    PI_ACTOR_BRISC = 2u,
    PI_ACTOR_NCRISC = 3u,
    PI_ACTOR_ERISC = 4u
} pi_actor_t;

/* --------------------------------------------------------------------------
 * Assumptions
 * --------------------------------------------------------------------------
 * Each bit names a behaviour the results depend on; the host reports the mask
 * in every run. Item numbers are those of docs/measurement.md "Unconfirmed hardware assumptions". */
/** A write issued after the data writes' acks reaches X280 loads from LIM after
 * the data, because the L2CPU applies accepted writes to LIM in arrival order:
 * owner assumption (item 2); NoC/Counters.md defines the ack for L1 and
 * registers only. */
#define PI_ASSUMED_INBOUND (1u << 0)
/** An X280 load observes an aligned NoC write of 16 B or less into LIM, such as
 * the 4-byte counter, whole: owner assumption (item 14);
 * WormholeB0/NoC/Ordering.md's atomic units are for L1. */
#define PI_ASSUMED_COUNTER_ATOMICITY (1u << 1)
/** A store through a mode-1 window returns only after its NoC write completes:
 * owner assumption (item 3); no X280 document says the store, not a later fence, waits.
 * No packet-injection store relies on it, so the host does not report it; the
 * bit stays reserved so a recorded mask keeps its meaning. */
#define PI_ASSUMED_BRIDGE_COMPLETION (1u << 2)
/** CLINT mtime counts at X280_MTIME_HZ: owner assumption (item 6). */
#define PI_ASSUMED_MTIME_HZ (1u << 3)
/** Samples copied from the pinned cache way into the export region are
 * host-visible: open (item 15). */
#define PI_ASSUMED_PIN_EXPORT (1u << 4)
/** High/low/high reads of the Tensix live timer give one 64-bit value:
 * documented (item 16; TensixTile/DebugTimestamper.md, "The 64-bit counter"). */
#define PI_ASSUMED_LOCAL_TIMER_ORDER (1u << 5)
/** Successive credit stores to one producer's consumed mirror, through one
 * posted static-VC counter window, land in program order: owner assumption
 * (item 21). */
#define PI_ASSUMED_STATIC_VC_ORDER (1u << 6)

/* --------------------------------------------------------------------------
 * ABI tag
 * -------------------------------------------------------------------------- */

/**
 * Builds the `abi_tag` that every args and result record carries.
 *
 * Bits [31:16] are PI_ABI_MAGIC, [15:8] zero, [7:4] the pi_mode_t, and [3:0]
 * the include-creation flag, so one comparison rejects a mismatched firmware
 * variant and uninitialized memory. Usable in a constant expression; each
 * image builds its expected tag from its compile-time defines.
 *
 * @param mode              pi_mode_t value; only the low 4 bits are encoded.
 * @param include_creation  0 or 1; only the low bit is encoded.
 */
#define PI_ABI_TAG(mode, include_creation) \
    (((uint32_t)PI_ABI_MAGIC << 16) | (((uint32_t)(mode) & 0xFu) << 4) | ((uint32_t)(include_creation) & 1u))

/**
 * Tests a received tag against the reader's own mode and include-creation.
 *
 * @param tag               value read from an args or result record.
 * @param mode              reader's compile-time pi_mode_t.
 * @param include_creation  reader's compile-time include-creation flag, 0 or 1.
 * @return 1 when the tag matches exactly; 0 when it does not, and also when
 *         `mode` or `include_creation` is itself out of range, since the
 *         encoding masks those and would otherwise alias a valid tag.
 */
static inline int pi_abi_tag_matches(uint32_t tag, uint32_t mode, uint32_t include_creation) {
    if (mode < (uint32_t)PI_MODE_LATENCY || mode > (uint32_t)PI_MODE_BACKPRESSURE) {
        return 0;
    }
    if (include_creation > 1u) {
        return 0;
    }
    return tag == PI_ABI_TAG(mode, include_creation) ? 1 : 0;
}

/* --------------------------------------------------------------------------
 * Host-owned input records
 * -------------------------------------------------------------------------- */

/**
 * X280 consumer arguments. Host-written before launch, immutable during the
 * run, and the only configuration the X280 image receives: it derives every
 * LIM and L1 address, extent, and expected total from these fields plus
 * pi_compute_layout().
 *
 * `worker_count` (P) and `tile_count` (T) satisfy P == T or P == 2*T. `S` is a
 * power-of-two ring depth in slots; `B` the producer batch; `C` the credit
 * batch in slots; `K` the per-visit credit cap. `W` and `N` are warmup and
 * measured request counts per worker. `poll_cadence` is a power-of-two poll
 * divisor; `l1_tile_base` the byte offset of the worker slices within each
 * producer core's L1; `x280_timeout_ticks` a per-phase mtime budget.
 * `inflight` is how many workers one consumer pass visits together, with
 * every slot load of the pass issued before any is used: 1, 2 or 4, and 1 in
 * latency mode, whose remote timer window follows one tile at a time.
 *
 * `release_num_dests` and `release_rect` are the release group the host's
 * release selector chose: nonzero `release_num_dests` releases the measured
 * phase by one multicast over the virtual rectangle `release_rect`, and must
 * equal `tile_count` with every tile-table entry inside the rectangle; zero
 * releases by one unicast per tile and requires `release_rect` all zero. NoC
 * broadcast reaches only Tensix tiles, because firmware opts every other NIU
 * out through the broadcast opt-out masks (tt-isa-documentation
 * BlackholeA0/NoC/MemoryMap.md, the broadcast-rectangle description and the
 * column and row opt-out masks; NoC/README.md), so the selector never gives
 * other tile types a group.
 */
typedef struct {
    PI_ALIGNAS(UBENCH_LIM_LINE_BYTES) uint32_t abi_tag;
    uint16_t worker_count;
    uint16_t tile_count;
    uint32_t S;
    uint32_t B;
    uint32_t C;
    uint32_t K;
    uint32_t poll_cadence;
    uint32_t l1_tile_base;
    uint64_t W;
    uint64_t N;
    uint64_t x280_timeout_ticks;
    uint32_t run_nonce;
    uint32_t release_num_dests; /* 0: one unicast write per tile */
    x280_noc_rect_t release_rect;
    uint32_t inflight; /* workers one consumer pass visits together: 1, 2 or 4 */
} pi_x280_args_t;

PI_STATIC_ASSERT(sizeof(pi_x280_args_t) == 128u, "pi_x280_args_t must be 128 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_x280_args_t) == UBENCH_LIM_LINE_BYTES, "pi_x280_args_t must be line-aligned");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, abi_tag) == 0u, "offset abi_tag");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, worker_count) == 4u, "offset worker_count");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, tile_count) == 6u, "offset tile_count");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, S) == 8u, "offset S");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, B) == 12u, "offset B");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, C) == 16u, "offset C");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, K) == 20u, "offset K");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, poll_cadence) == 24u, "offset poll_cadence");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, l1_tile_base) == 28u, "offset l1_tile_base");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, W) == 32u, "offset W");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, N) == 40u, "offset N");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, x280_timeout_ticks) == 48u, "offset x280_timeout_ticks");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, run_nonce) == 56u, "offset run_nonce");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, release_num_dests) == 60u, "offset release_num_dests");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, release_rect) == 64u, "offset release_rect");
PI_STATIC_ASSERT(offsetof(pi_x280_args_t, inflight) == 80u, "offset inflight");

/* Producer argument runtime words, PI_PRODUCER_ARG_WORDS * 4 == sizeof(pi_producer_args_t). */
#define PI_PRODUCER_ARG_WORDS 18u

/**
 * Producer arguments, serialized as exactly PI_PRODUCER_ARG_WORDS runtime words.
 *
 * Host-written before launch and immutable during the run. The producer derives
 * its tile, RISC identity, assigned NoC, L1 slice, ring, produced and ready
 * destinations, control word, result, staging, and sample arrays from these
 * fields plus pi_compute_layout(); nothing else is transmitted.
 *
 * `run_nonce` is a host-chosen 32-bit run identity, echoed in
 * pi_producer_result_t so a stale L1 result is rejected. `peer` locates the
 * X280 tile in TRANSLATED coordinate space, resolved for the producer's NoC,
 * which is what its NoC write consumes, matching the tile table's virtual space.
 * `timeout_ticks` is a per-phase budget in the producer's native wall ticks
 * (the live timer on Tensix, the wall clock on an idle ERISC). `dst_chip`/`dst_mesh` address the header the
 * producer seeds and are unused when the host seeds it. Field meanings
 * otherwise match pi_x280_args_t.
 */
typedef struct {
    PI_ALIGNAS(8) uint32_t abi_tag;
    uint32_t run_nonce;
    uint32_t worker_index;
    uint16_t worker_count;
    uint16_t tile_count;
    uint32_t S;
    uint32_t B;
    uint32_t poll_cadence;
    uint16_t dst_chip;
    uint16_t dst_mesh;
    uint32_t l1_tile_base;
    x280_noc_coord_t peer;
    uint32_t pad;
    uint64_t W;
    uint64_t N;
    uint64_t timeout_ticks;
} pi_producer_args_t;

PI_STATIC_ASSERT(sizeof(pi_producer_args_t) == 72u, "pi_producer_args_t must be 72 bytes");
PI_STATIC_ASSERT(
    sizeof(pi_producer_args_t) == PI_PRODUCER_ARG_WORDS * sizeof(uint32_t),
    "pi_producer_args_t must be exactly PI_PRODUCER_ARG_WORDS runtime words");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_producer_args_t) == 8u, "pi_producer_args_t must be 8-aligned");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, abi_tag) == 0u, "offset abi_tag");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, run_nonce) == 4u, "offset run_nonce");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, worker_index) == 8u, "offset worker_index");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, worker_count) == 12u, "offset worker_count");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, tile_count) == 14u, "offset tile_count");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, S) == 16u, "offset S");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, B) == 20u, "offset B");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, poll_cadence) == 24u, "offset poll_cadence");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, dst_chip) == 28u, "offset dst_chip");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, dst_mesh) == 30u, "offset dst_mesh");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, l1_tile_base) == 32u, "offset l1_tile_base");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, peer) == 36u, "offset peer");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, W) == 48u, "offset W");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, N) == 56u, "offset N");
PI_STATIC_ASSERT(offsetof(pi_producer_args_t, timeout_ticks) == 64u, "offset timeout_ticks");

/* The tile table is x280_noc_coord_t[tile_count], one producer tile each in
 * virtual space, which is what the X280 outbound window takes, at the arena's
 * tile-coordinate offset; logical coordinates stay in host report state. A LIM
 * write must cover whole cache lines, so the table's span is
 * `pi_layout_t::tile_coords_bytes` -- align64(8*T), not 8*T -- and the host
 * zeroes the tail past `tile_count` entries rather than shortening the write. */
PI_STATIC_ASSERT(sizeof(x280_noc_coord_t) == 8u, "a tile-table entry must be 8 bytes");

/* --------------------------------------------------------------------------
 * Current-protocol slots and control
 * -------------------------------------------------------------------------- */

/**
 * Produced-progress slot in LIM, one per worker. The owning producer writes
 * `value` by NoC; X280 only reads it. Wraps modulo 2^32 and is compared with
 * pi_counter_distance().
 *
 * `pad` is zeroed and carries no semantic contract, but the 64-byte size is a
 * NoC congruence requirement rather than spare room. The producer publishes
 * `value` with a length-carrying NoC write out of Tensix L1, and that path
 * requires the source and destination addresses to be congruent -- mod 16 per
 * NOC_L1_WRITE_ALIGNMENT_BYTES and the WormholeB0 alignment table, with no
 * Blackhole table published and mod 64 reported for Blackhole DRAM transfers.
 * Every producer's L1 source word is congruent to 0 mod 64, so a 64-byte slot
 * at a 64-aligned base keeps every destination congruent under any of those
 * moduli. A smaller slot leaves the write undefined for the workers it
 * misaligns.
 */
typedef struct {
    PI_ALIGNAS(64) volatile uint32_t value;
    uint8_t pad[60];
} pi_counter_slot_t;

PI_STATIC_ASSERT(sizeof(pi_counter_slot_t) == 64u, "pi_counter_slot_t must be 64 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_counter_slot_t) == 64u, "pi_counter_slot_t must be 64-aligned");
PI_STATIC_ASSERT(offsetof(pi_counter_slot_t, value) == 0u, "offset value");
PI_STATIC_ASSERT(offsetof(pi_counter_slot_t, pad) == 4u, "offset pad");

/**
 * Readiness slot in LIM, one per worker, holding a pi_ready_t. The owning
 * producer writes `value` by NoC; X280 only reads it.
 *
 * The same write path publishes this record, so `pad` and the 64-byte size are
 * the NoC congruence requirement stated on pi_counter_slot_t.
 */
typedef struct {
    PI_ALIGNAS(64) volatile uint32_t value;
    uint8_t pad[60];
} pi_ready_slot_t;

PI_STATIC_ASSERT(sizeof(pi_ready_slot_t) == 64u, "pi_ready_slot_t must be 64 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_ready_slot_t) == 64u, "pi_ready_slot_t must be 64-aligned");
PI_STATIC_ASSERT(offsetof(pi_ready_slot_t, value) == 0u, "offset value");
PI_STATIC_ASSERT(offsetof(pi_ready_slot_t, pad) == 4u, "offset pad");

/**
 * The one host-writable cache line in the arena. `collect_ack_run_value` holds
 * the active full run value only after host collection has succeeded, and zero
 * otherwise; X280 polls it before teardown. The 64-byte size is the required
 * host write granularity and its padding carries no contract.
 */
typedef struct {
    PI_ALIGNAS(UBENCH_LIM_LINE_BYTES) volatile uint64_t collect_ack_run_value;
    uint8_t pad[UBENCH_LIM_LINE_BYTES - 8u];
} pi_host_control_t;

PI_STATIC_ASSERT(sizeof(pi_host_control_t) == UBENCH_LIM_LINE_BYTES, "pi_host_control_t must be one cache line");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_host_control_t) == UBENCH_LIM_LINE_BYTES, "pi_host_control_t must be line-aligned");
PI_STATIC_ASSERT(offsetof(pi_host_control_t, collect_ack_run_value) == 0u, "offset collect_ack_run_value");
PI_STATIC_ASSERT(offsetof(pi_host_control_t, pad) == 8u, "offset pad");

/* --------------------------------------------------------------------------
 * Failure and results
 * -------------------------------------------------------------------------- */

/**
 * Identity of the first failure an actor records; later failures do not
 * overwrite it. `code` is a pi_error_t, `phase` a pi_phase_t, `actor` a
 * pi_actor_t, and `worker_index` is the failing worker, or UINT32_MAX when the
 * failure is not worker-specific. All zero means no failure, which
 * PI_ERROR_NONE and PI_PHASE_INIT encode.
 */
typedef struct {
    PI_ALIGNAS(4) uint32_t code;
    uint32_t phase;
    uint32_t worker_index;
    uint32_t actor;
} pi_failure_t;

PI_STATIC_ASSERT(sizeof(pi_failure_t) == 16u, "pi_failure_t must be 16 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_failure_t) == 4u, "pi_failure_t must be 4-aligned");
PI_STATIC_ASSERT(offsetof(pi_failure_t, code) == 0u, "offset code");
PI_STATIC_ASSERT(offsetof(pi_failure_t, phase) == 4u, "offset phase");
PI_STATIC_ASSERT(offsetof(pi_failure_t, worker_index) == 8u, "offset worker_index");
PI_STATIC_ASSERT(offsetof(pi_failure_t, actor) == 12u, "offset actor");

/**
 * Per-worker consumer progress in LIM, written only by X280. `consumed` and
 * `credited` wrap modulo 2^32 and are compared with pi_counter_distance();
 * `phase_done` counts completions within the current phase and is reset
 * between phases, and `warmup_done` holds the warmup phase's final count. The
 * host cross-checks these against the producer's own result.
 */
typedef struct {
    PI_ALIGNAS(8) uint32_t consumed;
    uint32_t credited;
    uint64_t phase_done;
    uint64_t warmup_done;
} pi_consumer_slot_t;

PI_STATIC_ASSERT(sizeof(pi_consumer_slot_t) == 24u, "pi_consumer_slot_t must be 24 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_consumer_slot_t) == 8u, "pi_consumer_slot_t must be 8-aligned");
PI_STATIC_ASSERT(offsetof(pi_consumer_slot_t, consumed) == 0u, "offset consumed");
PI_STATIC_ASSERT(offsetof(pi_consumer_slot_t, credited) == 4u, "offset credited");
PI_STATIC_ASSERT(offsetof(pi_consumer_slot_t, phase_done) == 8u, "offset phase_done");
PI_STATIC_ASSERT(offsetof(pi_consumer_slot_t, warmup_done) == 16u, "offset warmup_done");

/**
 * One producer's result, written only by the owning RISC into its own L1 slice
 * at PI_L1_RESULT_OFFSET and valid until host collection finishes.
 *
 * `abi_tag` and `run_nonce` must match the launch before any other field is
 * believed. `quiescent` is non-zero once the producer has no publication in
 * flight. The three `*_progress` words are the producer's own evidence, kept
 * separate from pi_consumer_slot_t. `latency_count` and `rtt_count` are the
 * populated element counts of the mode-specific sample arrays. `stall_count`
 * counts measured-phase credit-stall episodes, entries into the ring-full wait
 * after a credit check found none: in PI_MODE_BACKPRESSURE the episodes that
 * ended on an observed credit, which is also the populated count of the
 * stall-sample array; in PI_MODE_THROUGHPUT every entry, with no durations, so
 * a failed run may include one episode that never ended; zero in
 * PI_MODE_LATENCY. `throughput_start`/`throughput_end` are raw reads of the
 * producer's native wall counter taken either side of the measured phase; the host rejects a zero or
 * half-range interval and never aggregates workers. `measured_counter_writes`
 * counts the produced-counter writes issued in the measured phase, one per
 * publication of a non-empty batch, in every mode.
 */
typedef struct {
    PI_ALIGNAS(8) uint32_t abi_tag;
    uint32_t worker_index;
    uint32_t phase;
    uint32_t quiescent;
    uint32_t run_nonce;
    uint32_t reserved_progress;
    uint32_t published_progress;
    uint32_t consumed_progress;
    uint64_t warmup_completed;
    uint64_t measured_completed;
    uint64_t latency_count;
    uint64_t stall_count;
    uint64_t rtt_count;
    uint64_t throughput_start;
    uint64_t throughput_end;
    pi_failure_t failure;
    uint64_t cpu_start;
    uint64_t cpu_end;
    uint64_t measured_counter_writes;
} pi_producer_result_t;

/* Native local stamps; CPU and wall have independent counter origins. */
typedef struct {
    uint64_t wall;
    uint64_t cpu;
} pi_local_stamp_t;

/* Exact local deltas; the producer rejects values exceeding UINT32_MAX. */
typedef struct {
    uint32_t wall;
    uint32_t cpu;
} pi_local_sample_t;

PI_STATIC_ASSERT(sizeof(pi_producer_result_t) == 128u, "pi_producer_result_t must be 128 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_producer_result_t) == 8u, "pi_producer_result_t must be 8-aligned");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, abi_tag) == 0u, "offset abi_tag");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, worker_index) == 4u, "offset worker_index");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, phase) == 8u, "offset phase");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, quiescent) == 12u, "offset quiescent");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, run_nonce) == 16u, "offset run_nonce");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, reserved_progress) == 20u, "offset reserved_progress");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, published_progress) == 24u, "offset published_progress");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, consumed_progress) == 28u, "offset consumed_progress");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, warmup_completed) == 32u, "offset warmup_completed");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, measured_completed) == 40u, "offset measured_completed");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, latency_count) == 48u, "offset latency_count");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, stall_count) == 56u, "offset stall_count");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, rtt_count) == 64u, "offset rtt_count");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, throughput_start) == 72u, "offset throughput_start");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, throughput_end) == 80u, "offset throughput_end");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, failure) == 88u, "offset failure");
PI_STATIC_ASSERT(offsetof(pi_producer_result_t, measured_counter_writes) == 120u, "offset measured_counter_writes");
PI_STATIC_ASSERT(
    PI_L1_RESULT_OFFSET + sizeof(pi_producer_result_t) <= PI_L1_STAGING_OFFSET,
    "producer result must fit between its L1 offset and staging");

/**
 * The run-level result, initialized and written only by X280 at the arena's
 * run-result offset.
 *
 * `state` is a pi_phase_t: PI_PHASE_INIT until X280 finishes, then
 * PI_PHASE_DONE or PI_PHASE_FAILED. `run_value` echoes the host run value.
 * `measured_completed` is the summed measured completions X280 observed.
 * `export_valid_bytes` is the byte count of the latency export region X280
 * filled, zero outside PI_MODE_LATENCY. `data_quiescent`, `windows_cleared`,
 * and `cache_restored` are non-zero once each teardown step has been proved.
 *
 * `aggregate_start_*`/`aggregate_end_*` are X280's own raw reads bracketing
 * the measured phase in every mode: the start when the measured-phase core-sync
 * release returns, the end after the final measured credit is returned. The
 * `mtime` pair is in 50 MHz CLINT ticks and the `rdcycle` pair in hart-0 core
 * cycles, with independent origins. Each is zero when the run failed before
 * reaching it; the host rejects a zero or half-range interval on a successful
 * run. `empty_visits` counts measured-phase round-robin visits that found the
 * visited worker's ring empty; a visit to a worker that has finished its
 * phase is not counted. `released_group_dests` and `released_endpoints` are
 * the release roster X280 bound and executed for the measured phase and any
 * recovery release: the group's receiver count (0: no group) and the unicast
 * endpoint count. `pad` is zeroed by the host and carries no contract.
 */
typedef struct {
    PI_ALIGNAS(UBENCH_LIM_LINE_BYTES) uint32_t abi_tag;
    uint32_t state;
    uint64_t run_value;
    uint64_t measured_completed;
    uint64_t export_valid_bytes;
    pi_failure_t failure;
    uint32_t data_quiescent;
    uint32_t windows_cleared;
    uint32_t cache_restored;
    uint32_t worker_count;
    uint64_t aggregate_start_mtime;
    uint64_t aggregate_end_mtime;
    uint64_t aggregate_start_rdcycle;
    uint64_t aggregate_end_rdcycle;
    uint64_t empty_visits;
    uint32_t released_group_dests;
    uint32_t released_endpoints;
    uint8_t pad[16];
} pi_run_result_t;

PI_STATIC_ASSERT(sizeof(pi_run_result_t) == 128u, "pi_run_result_t must be 128 bytes");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_run_result_t) == UBENCH_LIM_LINE_BYTES, "pi_run_result_t must be line-aligned");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, abi_tag) == 0u, "offset abi_tag");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, state) == 4u, "offset state");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, run_value) == 8u, "offset run_value");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, measured_completed) == 16u, "offset measured_completed");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, export_valid_bytes) == 24u, "offset export_valid_bytes");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, failure) == 32u, "offset failure");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, data_quiescent) == 48u, "offset data_quiescent");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, windows_cleared) == 52u, "offset windows_cleared");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, cache_restored) == 56u, "offset cache_restored");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, worker_count) == 60u, "offset worker_count");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, aggregate_start_mtime) == 64u, "offset aggregate_start_mtime");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, aggregate_end_mtime) == 72u, "offset aggregate_end_mtime");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, aggregate_start_rdcycle) == 80u, "offset aggregate_start_rdcycle");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, aggregate_end_rdcycle) == 88u, "offset aggregate_end_rdcycle");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, empty_visits) == 96u, "offset empty_visits");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, released_group_dests) == 104u, "offset released_group_dests");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, released_endpoints) == 108u, "offset released_endpoints");
PI_STATIC_ASSERT(offsetof(pi_run_result_t, pad) == 112u, "offset pad");

/* --------------------------------------------------------------------------
 * Runtime-argument serialization
 * -------------------------------------------------------------------------- */

/**
 * The punning vehicle for the producer runtime block. A kernel entry
 * may fill `words` from successive get_arg_val<uint32_t>() reads and then use
 * `args` directly, which serializes with no copy at all.
 */
typedef union {
    pi_producer_args_t args;
    uint32_t words[PI_PRODUCER_ARG_WORDS];
} pi_producer_arg_words_t;

PI_STATIC_ASSERT(
    sizeof(pi_producer_arg_words_t) == sizeof(pi_producer_args_t), "pi_producer_arg_words_t must be exactly the args");
PI_STATIC_ASSERT(PI_ALIGNOF(pi_producer_arg_words_t) == 8u, "pi_producer_arg_words_t must be 8-aligned");

/* --------------------------------------------------------------------------
 * Checked counter-distance and sizing helpers
 * -------------------------------------------------------------------------- */

/* Modular u32 forward distance from earlier to later */
static inline uint32_t pi_counter_distance(uint32_t later, uint32_t earlier) {
    return credit_used(later, earlier, UINT32_MAX);
}

/* Align up to next multiple of 64 */
static inline uint64_t pi_align64(uint64_t x) { return (x + 63u) & ~63ULL; }

/** Adds two u64 values. @return 1 and writes `*out`, or 0 on overflow. */
static PI_CONSTEXPR inline int pi_checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > UINT64_MAX - a) {
        return 0;
    }
    *out = a + b;
    return 1;
}

/** Multiplies two u64 values. @return 1 and writes `*out`, or 0 on overflow. */
static PI_CONSTEXPR inline int pi_checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (a != 0u && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

/**
 * Rounds `x` up to a multiple of `align`.
 *
 * @param align power of two, non-zero.
 * @return 1 and writes `*out`, or 0 when `align` is not a non-zero power of
 *         two or the rounding would overflow.
 */
static PI_CONSTEXPR inline int pi_checked_align(uint64_t x, uint64_t align, uint64_t* out) {
    if (align == 0u || (align & (align - 1u)) != 0u) {
        return 0;
    }
    if (x > UINT64_MAX - (align - 1u)) {
        return 0;
    }
    *out = (x + align - 1u) & ~(align - 1u);
    return 1;
}

/* --------------------------------------------------------------------------
 * Shared checked layout
 * -------------------------------------------------------------------------- */

/**
 * Why pi_compute_layout() refused a run, or PI_LAYOUT_OK.
 *
 * One code per class of rejection, not per check: a caller acts on the class,
 * and the offending value carries the rest. PI_LAYOUT_INVARIANT is the
 * function catching itself -- a span the arithmetic above it should have made
 * impossible -- and so is the one code no input can provoke.
 */
typedef enum {
    PI_LAYOUT_OK = 0u,
    PI_LAYOUT_BAD_PARAM = 1u,
    PI_LAYOUT_OVERFLOW = 2u,
    PI_LAYOUT_ALIGNMENT = 3u,
    PI_LAYOUT_LIM_CAPACITY = 4u,
    PI_LAYOUT_EXPORT_CAPACITY = 5u,
    PI_LAYOUT_L1_CAPACITY = 6u,
    PI_LAYOUT_INVARIANT = 7u
} pi_layout_code_t;

/**
 * A layout outcome. `code` is a pi_layout_code_t; on any rejection that can
 * name them cheaply, `value` is the offending quantity and `bound` the limit
 * it failed against. Both are 0 where the code names neither, which is every
 * PI_LAYOUT_OVERFLOW and a domain rejection with no numeric bound.
 */
typedef struct {
    uint32_t code;
    uint64_t value;
    uint64_t bound;
} pi_layout_check_t;

/** Builds a pi_layout_check_t. Pass 0 for a `value` or `bound` the code does
 * not carry. */
static PI_CONSTEXPR inline pi_layout_check_t pi_layout_check(uint32_t code, uint64_t value, uint64_t bound) {
    pi_layout_check_t c = {0u, 0u, 0u};
    c.code = code;
    c.value = value;
    c.bound = bound;
    return c;
}

/** Name of a pi_layout_code_t, for host reporting. Never null. */
static inline const char* pi_layout_code_name(uint32_t code) {
    switch ((pi_layout_code_t)code) {
        case PI_LAYOUT_OK: return "ok";
        case PI_LAYOUT_BAD_PARAM: return "parameter out of domain";
        case PI_LAYOUT_OVERFLOW: return "arithmetic overflow";
        case PI_LAYOUT_ALIGNMENT: return "misaligned base";
        case PI_LAYOUT_LIM_CAPACITY: return "LIM arena capacity";
        case PI_LAYOUT_EXPORT_CAPACITY: return "pinned export capacity";
        case PI_LAYOUT_L1_CAPACITY: return "Tensix L1 capacity";
        case PI_LAYOUT_INVARIANT: return "layout invariant";
    }
    return "unknown layout code";
}

/**
 * The pi_error_t every actor reports for a layout rejection, so host, X280,
 * and producer name one cause rather than three.
 *
 * PI_LAYOUT_INVARIANT maps to PI_ERROR_LAYOUT_INVARIANT rather than to
 * PI_ERROR_BAD_ARGS, which would send an operator auditing arguments that may
 * be entirely valid.
 */
static inline pi_error_t pi_layout_code_to_error(uint32_t code) {
    switch ((pi_layout_code_t)code) {
        case PI_LAYOUT_OK: return PI_ERROR_NONE;
        case PI_LAYOUT_ALIGNMENT: return PI_ERROR_BAD_PLACEMENT;
        case PI_LAYOUT_LIM_CAPACITY:
        case PI_LAYOUT_EXPORT_CAPACITY:
        case PI_LAYOUT_L1_CAPACITY: return PI_ERROR_CAPACITY;
        case PI_LAYOUT_INVARIANT: return PI_ERROR_LAYOUT_INVARIANT;
        case PI_LAYOUT_BAD_PARAM:
        case PI_LAYOUT_OVERFLOW: return PI_ERROR_BAD_ARGS;
    }
    return PI_ERROR_BAD_ARGS;
}

/**
 * Every derived base and extent of one run, as pi_compute_layout() writes it.
 *
 * LIM fields are byte offsets from the arena base; L1 fields are byte offsets
 * within one Tensix core's L1, except `l1_worker_stride` and `l1_tile_bytes`
 * which are extents. Per-worker addresses are `<base>_offset + index *
 * <element size>`, using `ring_stride` for rings and `latency_sample_stride`
 * for the export region.
 */
typedef struct {
    uint32_t worker_count;  /* P */
    uint32_t tile_count;    /* T */
    uint32_t risc_per_tile; /* P / T, 1 or 2 */
    uint32_t risc_shift;    /* log2(risc_per_tile), so 0 or 1 */
    uint32_t risc_mask;     /* risc_per_tile - 1, so 0 or 1 */
    uint32_t staging_depth; /* D = min(B, S) */

    uint64_t args_offset;           /* pi_x280_args_t, 128 B */
    uint64_t host_control_offset;   /* pi_host_control_t, 64 B */
    uint64_t run_result_offset;     /* pi_run_result_t, 128 B */
    uint64_t tile_coords_offset;    /* x280_noc_coord_t[T] */
    uint64_t tile_coords_bytes;     /* align64(8*T): whole lines, tail zeroed */
    uint64_t sync_arrivals_offset;  /* core_sync_slot_t[P] */
    uint64_t consumer_slots_offset; /* pi_consumer_slot_t[P], then pad to 64 */
    uint64_t produced_offset;       /* pi_counter_slot_t[P], 64-aligned */
    uint64_t ready_offset;          /* pi_ready_slot_t[P], 64-aligned */
    uint64_t rings_offset;          /* P rings of `ring_stride` bytes */
    uint64_t ring_stride;           /* S * PI_SLOT_BYTES */
    uint64_t rings_bytes;           /* P * ring_stride */
    uint64_t latency_export_offset; /* 0 outside PI_MODE_LATENCY */
    uint64_t latency_export_bytes;  /* align64(P * 8 * N), 0 otherwise */
    uint64_t latency_sample_stride; /* 8 * N, 0 otherwise */
    uint64_t arena_base_bytes;      /* F: everything up to the export region */
    uint64_t arena_bytes;           /* F + latency_export_bytes */

    uint32_t l1_tile_base;          /* as passed in, 64-byte aligned */
    uint32_t l1_release_offset;     /* shared release word within tile page */
    uint32_t l1_worker_base_offset; /* first per-RISC slice within tile page */
    uint32_t l1_worker_stride;      /* one RISC's slice */
    uint32_t l1_tile_bytes;         /* worker base + risc_per_tile * stride */
    uint32_t l1_starts_offset;      /* within a slice; 0 outside backpressure */
    uint32_t l1_stalls_offset;      /* within a slice; 0 outside backpressure */
    uint32_t l1_rtts_offset;        /* within a slice; 0 outside backpressure */
} pi_layout_t;

/**
 * Validates one run's parameters and derives every LIM and L1 span from them.
 *
 * The single source of the layout formulas; every reader derives its
 * addresses from the result. Fully validates before writing `out`, so no
 * address can be derived from rejected input.
 *
 * @param P             worker count, in [1, PI_MAX_WORKERS]; T must divide it
 *                      with a quotient of 1 (BRISC only) or 2 (BRISC+NCRISC).
 * @param T             tile count, in [1, PI_MAX_TILES].
 * @param S             ring depth in slots; non-zero power of two.
 * @param B             producer batch, non-zero; D = min(B, S) staging slots.
 * @param N             measured requests per worker, non-zero.
 * @param mode          pi_mode_t; PI_MODE_LATENCY adds the export region and
 *                      PI_MODE_BACKPRESSURE the per-worker sample arrays.
 * @param l1_tile_base  byte offset of the worker slices in Tensix L1; must be
 *                      64-byte aligned, as allocate_tile_l1_buffer() enforces.
 * @param out           written only on success; untouched on failure, and so
 *                      untouched by a null pointer, every domain rejection,
 *                      and every overflow.
 * @return `code` PI_LAYOUT_OK, or the class of rejection. PI_LAYOUT_BAD_PARAM
 *         carries the offending parameter and its limit, or 0 where the domain
 *         has no numeric bound; PI_LAYOUT_ALIGNMENT `l1_tile_base` and 64; the
 *         three capacity codes the required and available bytes;
 *         PI_LAYOUT_INVARIANT the computed span and what it should have been;
 *         PI_LAYOUT_OVERFLOW neither. Map it with pi_layout_code_to_error().
 */
static PI_CONSTEXPR inline pi_layout_check_t pi_compute_layout(
    uint32_t P,
    uint32_t T,
    uint32_t S,
    uint32_t B,
    uint64_t N,
    uint32_t mode,
    uint32_t l1_tile_base,
    pi_layout_t* out) {
    uint64_t coords_bytes = 0u;
    uint64_t arrivals_off = 0u;
    uint64_t arrivals_bytes = 0u;
    uint64_t arrivals_end = 0u;
    uint64_t consumer_off = 0u;
    uint64_t consumer_bytes = 0u;
    uint64_t produced_off = 0u;
    uint64_t ready_off = 0u;
    uint64_t slots_bytes = 0u;
    uint64_t ready_end = 0u;
    uint64_t rings_off = 0u;
    uint64_t ring_stride = 0u;
    uint64_t rings_bytes = 0u;
    uint64_t arena_base = 0u;
    uint64_t sample_stride = 0u;
    uint64_t export_off = 0u;
    uint64_t export_bytes = 0u;
    uint64_t arena_bytes = 0u;
    uint64_t staging_bytes = 0u;
    uint64_t slice_end = 0u;
    uint64_t align_s = 0u;
    uint64_t align_n = 0u;
    uint64_t starts_off = 0u;
    uint64_t stalls_off = 0u;
    uint64_t rtts_off = 0u;
    uint64_t worker_stride = 0u;
    uint64_t tile_bytes = 0u;
    uint32_t risc_per_tile = 0u;
    uint32_t D = 0u;

    if (out == NULL) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, 0u, 0u);
    }
    if (P == 0u || P > PI_MAX_WORKERS) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)P, PI_MAX_WORKERS);
    }
    if (T == 0u || T > PI_MAX_TILES) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)T, PI_MAX_TILES);
    }
    if ((P % T) != 0u) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)P, (uint64_t)T);
    }
    risc_per_tile = P / T;
    if (risc_per_tile != 1u && risc_per_tile != 2u) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)risc_per_tile, 2u);
    }
    if (S == 0u || (S & (S - 1u)) != 0u) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)S, 0u);
    }
    if (B == 0u) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)B, 0u);
    }
    if (N == 0u) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, N, 0u);
    }
    if (mode != (uint32_t)PI_MODE_LATENCY && mode != (uint32_t)PI_MODE_THROUGHPUT &&
        mode != (uint32_t)PI_MODE_BACKPRESSURE) {
        return pi_layout_check(PI_LAYOUT_BAD_PARAM, (uint64_t)mode, (uint64_t)PI_MODE_BACKPRESSURE);
    }
    if ((l1_tile_base & 63u) != 0u) {
        return pi_layout_check(PI_LAYOUT_ALIGNMENT, (uint64_t)l1_tile_base, 64u);
    }

    /* LIM arena: fixed records, tile coordinates, per-worker slots, rings. */
    if (!pi_checked_mul(sizeof(x280_noc_coord_t), (uint64_t)T, &coords_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_align(coords_bytes, UBENCH_LIM_LINE_BYTES, &coords_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(320u, coords_bytes, &arrivals_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul(CORE_SYNC_SLOT_BYTES, (uint64_t)P, &arrivals_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(arrivals_off, arrivals_bytes, &arrivals_end)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_align(arrivals_end, 64u, &consumer_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul(sizeof(pi_consumer_slot_t), (uint64_t)P, &consumer_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(consumer_off, consumer_bytes, &produced_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_align(produced_off, 64u, &produced_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul(sizeof(pi_counter_slot_t), (uint64_t)P, &slots_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(produced_off, slots_bytes, &ready_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul(sizeof(pi_ready_slot_t), (uint64_t)P, &slots_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(ready_off, slots_bytes, &ready_end)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_align(ready_end, 64u, &rings_off)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul((uint64_t)S, PI_SLOT_BYTES, &ring_stride)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul((uint64_t)P, ring_stride, &rings_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(rings_off, rings_bytes, &arena_base)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (arena_base > PI_ARENA_BYTES) {
        return pi_layout_check(PI_LAYOUT_LIM_CAPACITY, arena_base, PI_ARENA_BYTES);
    }

    /* Latency export follows the rings and must fit the reserved cache way. */
    arena_bytes = arena_base;
    if (mode == (uint32_t)PI_MODE_LATENCY) {
        if (!pi_checked_mul(8u, N, &sample_stride)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_mul((uint64_t)P, sample_stride, &export_bytes)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_align(export_bytes, UBENCH_LIM_LINE_BYTES, &export_bytes)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (export_bytes > PI_CACHE_LIMIT_BYTES) {
            return pi_layout_check(PI_LAYOUT_EXPORT_CAPACITY, export_bytes, PI_CACHE_LIMIT_BYTES);
        }
        export_off = arena_base;
        if (!pi_checked_add(arena_base, export_bytes, &arena_bytes)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (arena_bytes > PI_ARENA_BYTES) {
            return pi_layout_check(PI_LAYOUT_LIM_CAPACITY, arena_bytes, PI_ARENA_BYTES);
        }
    }

    /* Tensix L1: fixed records, staging, then backpressure sample arrays. */
    D = (B < S) ? B : S;
    if (!pi_checked_mul(PI_SLOT_BYTES, (uint64_t)D, &staging_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(PI_L1_STAGING_OFFSET, staging_bytes, &slice_end)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (mode == (uint32_t)PI_MODE_BACKPRESSURE) {
        if (!pi_checked_mul(sizeof(pi_local_stamp_t), (uint64_t)S, &align_s)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_align(align_s, 64u, &align_s)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_mul(sizeof(pi_local_sample_t), N, &align_n)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_align(align_n, 64u, &align_n)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        starts_off = slice_end;
        if (!pi_checked_add(starts_off, align_s, &stalls_off)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_add(stalls_off, align_n, &rtts_off)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
        if (!pi_checked_add(rtts_off, align_n, &slice_end)) {
            return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
        }
    }
    if (!pi_checked_align(slice_end, 64u, &worker_stride)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_mul((uint64_t)risc_per_tile, worker_stride, &tile_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (!pi_checked_add(64u, tile_bytes, &tile_bytes)) {
        return pi_layout_check(PI_LAYOUT_OVERFLOW, 0u, 0u);
    }
    if (tile_bytes > (uint64_t)UINT32_MAX - (uint64_t)l1_tile_base) {
        return pi_layout_check(PI_LAYOUT_L1_CAPACITY, tile_bytes, (uint64_t)UINT32_MAX - (uint64_t)l1_tile_base);
    }

    /* Invariants of the published layout, not of the arithmetic above. A LIM
     * write covers whole cache lines, so the coordinate table's span must be
     * some number of them; and the consumer bounds every span with one check
     * on arena_bytes, which holds only while the spans are contiguous. Both
     * are unreachable while the steps above stand, and are here to fail at
     * this call rather than as a device-side LIM rejection if one changes. */
    if ((coords_bytes % UBENCH_LIM_LINE_BYTES) != 0u) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, coords_bytes, UBENCH_LIM_LINE_BYTES);
    }
    if (arrivals_off != 320u + coords_bytes) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, arrivals_off, 320u + coords_bytes);
    }
    if ((consumer_off % 64u) != 0u || consumer_off < arrivals_end) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, consumer_off, arrivals_end);
    }

    /* Congruence of the two peer-written arrays, whose requirement is stated
     * on pi_counter_slot_t. Blackhole's noc_parameters.h sets both
     * NOC_L1_{READ,WRITE}_ALIGNMENT_BYTES to 16 and derives L1_ALIGNMENT as
     * their maximum; that constant is an absolute alignment, while the
     * congruence relation itself comes from the alignment table cited above.
     * The transport_bandwidth backends -- the
     * tree's only other Tensix/LIM transfers -- refuse a run unless both the
     * staging source and the peer destination satisfy it. This is that check
     * for packet injection, at the 64 the producer's L1 sources already meet.
     * Unreachable while the alignment above and the record sizes stand, and
     * here so a change to either fails at this call rather than as undefined
     * NoC behaviour on silicon. */
    if ((produced_off % 64u) != 0u) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, produced_off, 64u);
    }
    if ((ready_off % 64u) != 0u) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, ready_off, 64u);
    }
    if (sizeof(pi_counter_slot_t) != 64u || sizeof(pi_ready_slot_t) != 64u) {
        return pi_layout_check(PI_LAYOUT_INVARIANT, sizeof(pi_counter_slot_t), 64u);
    }

    out->worker_count = P;
    out->tile_count = T;
    out->risc_per_tile = risc_per_tile;
    out->risc_shift = (risc_per_tile == 2u) ? 1u : 0u;
    out->risc_mask = risc_per_tile - 1u;
    out->staging_depth = D;

    out->args_offset = 0u;
    out->host_control_offset = 128u;
    out->run_result_offset = 192u;
    out->tile_coords_offset = 320u;
    out->tile_coords_bytes = coords_bytes;
    out->sync_arrivals_offset = arrivals_off;
    out->consumer_slots_offset = consumer_off;
    out->produced_offset = produced_off;
    out->ready_offset = ready_off;
    out->rings_offset = rings_off;
    out->ring_stride = ring_stride;
    out->rings_bytes = rings_bytes;
    out->latency_export_offset = export_off;
    out->latency_export_bytes = export_bytes;
    out->latency_sample_stride = sample_stride;
    out->arena_base_bytes = arena_base;
    out->arena_bytes = arena_bytes;

    out->l1_tile_base = l1_tile_base;
    out->l1_release_offset = 0u;
    out->l1_worker_base_offset = 64u;
    out->l1_worker_stride = (uint32_t)worker_stride;
    out->l1_tile_bytes = (uint32_t)tile_bytes;
    out->l1_starts_offset = (uint32_t)starts_off;
    out->l1_stalls_offset = (uint32_t)stalls_off;
    out->l1_rtts_offset = (uint32_t)rtts_off;
    return pi_layout_check(PI_LAYOUT_OK, 0u, 0u);
}

/**
 * Tile owning `worker_index`, as an index into the tile table.
 *
 * Workers are assigned tile-major with the tile's RISCs adjacent, so with
 * `risc_per_tile` 2 workers 2t and 2t+1 both sit on ordered tile t. Host,
 * producer, and X280 must agree on this or every per-worker number is
 * misattributed, so all three derive it here.
 *
 * @param worker_index must be less than `layout->worker_count`; the caller
 *                     validates its own index, as this is pure index math.
 */
static inline uint32_t pi_worker_tile_index(const pi_layout_t* layout, uint32_t worker_index) {
    /* Shift, not divide or select: this runs once per consumed packet in the
     * consumer, where a fence's memory clobber forces the divisor to reload so
     * a divide cannot be hoisted -- and gcc rewrites the equivalent ternary
     * back into one. risc_shift is exact for the only admitted values. */
    return worker_index >> layout->risc_shift;
}

/**
 * RISC owning `worker_index` within its tile: 0 for BRISC, 1 for NCRISC, and
 * always 0 when `risc_per_tile` is 1. This is also the worker's assigned NoC
 * index, an equivalence the producer enforces on itself.
 *
 * @param worker_index as for pi_worker_tile_index().
 */
static inline uint32_t pi_worker_risc_index(const pi_layout_t* layout, uint32_t worker_index) {
    return worker_index & layout->risc_mask;
}

/**
 * Byte offset of `worker_index`'s slice within its own tile's L1, from which
 * the PI_L1_* records and the layout's staging and sample offsets are reached.
 * Every tile shares one `l1_tile_base`, so this is valid on the owning tile
 * only.
 *
 * @param worker_index as for pi_worker_tile_index().
 */
static inline uint32_t pi_worker_l1_base(const pi_layout_t* layout, uint32_t worker_index) {
    return layout->l1_tile_base + layout->l1_worker_base_offset +
           pi_worker_risc_index(layout, worker_index) * layout->l1_worker_stride;
}

PI_STATIC_ASSERT(
    sizeof(pi_x280_args_t) + sizeof(pi_host_control_t) + sizeof(pi_run_result_t) == 320u,
    "fixed arena records must end where tile coordinates begin");
PI_STATIC_ASSERT(
    CORE_SYNC_SLOT_BYTES + sizeof(pi_consumer_slot_t) + sizeof(pi_counter_slot_t) + sizeof(pi_ready_slot_t) == 160u,
    "per-worker LIM slots must total 160 bytes");
PI_STATIC_ASSERT(
    PI_ARENA_BASE_GOLDEN_P280_T140_S16 <= PI_ARENA_BYTES, "P=280, T=140, S=16 base arena must fit the arena limit");

#if defined(__cplusplus)
/* Canonical shapes evaluated at translation time, so moving a span fails the
 * build instead of leaving the three actors to disagree at run time.
 *
 * abi_tag carries nothing about the layout, so a changed offset
 * passes pi_abi_tag_matches() on every side while host, X280 and Tensix address
 * different memory -- and Metal JIT-compiles the producer from the worktree at
 * launch, so the kernel need not have been built from the same header as the
 * binary that placed the arena. Nothing the device publishes can corroborate
 * its own layout: `arena_bytes` in the report is the host's figure.
 *
 * A failure here means a span moved, and no tag rejects a peer built against
 * the old layout. Updating the expected number removes the only signal that
 * the offsets moved.
 *
 * C11 has no equivalent: _Static_assert takes an integer constant expression,
 * which a call to this function is not. That excludes no actor. The five
 * firmware variants build from src/packet_injection_x280.cpp under
 * -std=c++17 (x280_image.cmake), the Tensix JIT kernel is a .cpp,
 * the host is C++, and no .c translation unit includes this header. The only
 * C11 compilation is the -std=c11 -fsyntax-only -x c portability check in
 * the x280_abi_headers target, which emits no binary, so the tripwire is absent from exactly
 * the one compilation that is nobody's actor. */
static PI_CONSTEXPR inline pi_layout_t pi_golden_layout(
    uint32_t P, uint32_t T, uint32_t S, uint32_t B, uint64_t N, uint32_t mode) {
    pi_layout_t l{};
    pi_compute_layout(P, T, S, B, N, mode, 0u, &l);
    return l;
}

/* Interior offsets, not just the total: a span can move while `arena_bytes`
 * stays put whenever a later align64 absorbs the shift, and it is the spans
 * that the three actors turn into addresses. */
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).sync_arrivals_offset == 1472u,
    "P=280 T=140 S=16 arrival slots start at 1472");
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).consumer_slots_offset == 3712u,
    "P=280 T=140 S=16 consumer slots start at 3712");
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).produced_offset == 10432u,
    "P=280 T=140 S=16 produced slots start at 10432");
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).ready_offset == 28352u,
    "P=280 T=140 S=16 ready slots start at 28352");
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).rings_offset == 46272u,
    "P=280 T=140 S=16 rings start at 46272");

/* Totals across the shapes that exercise different arithmetic: the paired
 * P=2T path, the export region, and P=1, where 24*P is not a whole number of
 * cache lines and the interior alignment steps therefore bite. */
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).arena_bytes == PI_ARENA_BASE_GOLDEN_P280_T140_S16,
    "P=280 T=140 S=16 throughput arena is the published golden");
PI_STATIC_ASSERT(
    PI_ARENA_BASE_GOLDEN_P280_T140_S16 == 619712u, "the published P=280 T=140 S=16 golden is 619712 bytes");
PI_STATIC_ASSERT(
    pi_golden_layout(280u, 140u, 16u, 8u, 58u, PI_MODE_LATENCY).arena_bytes == 749632u,
    "P=280 T=140 S=16 N=58 latency arena is 749632 bytes");
PI_STATIC_ASSERT(
    pi_golden_layout(2u, 1u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).arena_bytes == 4864u,
    "P=2 T=1 S=16 paired arena is 4864 bytes");
PI_STATIC_ASSERT(
    pi_golden_layout(1u, 1u, 16u, 8u, 4096u, PI_MODE_THROUGHPUT).arena_bytes == 2688u,
    "P=1 T=1 S=16 arena is 2688 bytes");

/* These assertions and the runtime goldens in tests/host/test_packet_injection.cpp
 * are halves of one check, and neither subsumes the other: the assertions pin
 * the interior spans at P=280 and the totals elsewhere, while the runtime
 * goldens pin the interior spans at P=1 and P=2 and the strides and extents
 * -- ring_stride, rings_bytes, tile_coords_bytes, l1_tile_bytes, the export
 * triple -- that no total here observes. An interior span moving at a small
 * shape under a total that holds is the coord-padding case, which these
 * assertions alone would miss. */

/* The L1 side, which no arena total observes. */
PI_STATIC_ASSERT(
    pi_golden_layout(4u, 2u, 16u, 32u, 64u, PI_MODE_BACKPRESSURE).l1_worker_stride == 3712u,
    "P=4 T=2 S=16 B=32 N=64 backpressure L1 worker stride is 3712");
PI_STATIC_ASSERT(
    pi_golden_layout(4u, 2u, 16u, 32u, 64u, PI_MODE_BACKPRESSURE).l1_rtts_offset == 3200u,
    "P=4 T=2 S=16 B=32 N=64 backpressure RTT array starts at 3200");
#endif

#endif /* X280_PACKET_INJECTION_ABI_H */
