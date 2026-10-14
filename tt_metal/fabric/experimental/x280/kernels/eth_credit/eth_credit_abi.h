// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/eth_credit/eth_credit_abi.h
 *
 * Wire ABI, constants, enumerations, POD layouts, runtime-argument word
 * indices, 2N-wrap pointer arithmetic, and checked sizing formulas for the
 * credit flow-control benchmark across a direct Ethernet link. Compatible with
 * C11 and C++17. Contains no Metal or host includes.
 */
#ifndef X280_ETH_CREDIT_ABI_H
#define X280_ETH_CREDIT_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "../../include/x280_align.h"
#include "../../include/x280_noc_coord.h"
#include "../shared/core_sync_layout.h"
#include "../shared/credit_counter.h"
#include "../shared/ubench_outcome.h"
#include "../shared/ubench_units.h"

#if defined(__cplusplus)
#define EC_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define EC_ALIGNAS(n) alignas(n)
#define EC_ALIGNOF(t) alignof(t)
#else
#define EC_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#define EC_ALIGNAS(n) _Alignas(n)
#define EC_ALIGNOF(t) _Alignof(t)
#endif

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */
/* Nothing else identifies a layout, so a change to any args or result record
 * changes its magic: a host and an image built from different layouts then
 * reject each other's records. */
#define EC_ABI_MAGIC 0x45434233u    /* 'E' 'C' 'B' '3' */
#define EC_RESULT_MAGIC 0x45435233u /* 'E' 'C' 'R' '3' */

#define EC_MAX_P 4u
#define EC_MAX_DEPTH 1024u
#define EC_X280_HARTS 4u
#define EC_X280_FWD_ARGS_BYTES 128u

/* The shadow header is the slot's first 96 bytes; the payload follows in
 * 16-byte steps because a NoC or TT-link transfer moves whole 16-byte words. */
#define EC_HEADER_BYTES 96u
#define EC_UNIT_BYTES UBENCH_NOC_UNIT_BYTES
/* payload_size_bytes is a uint16 in the production header. */
#define EC_MAX_PAYLOAD_BYTES 65520u
#define EC_MAX_SLOT_BYTES (EC_HEADER_BYTES + EC_MAX_PAYLOAD_BYTES)

#define EC_ETH_CONTROL_ALIGN 512u
#define EC_ETH_L1_BANKS 32u
#define EC_LIM_ALIGN UBENCH_LIM_LINE_BYTES

/* Overlay stream 0 is tt-metal scratch; an Ethernet tile has 32 streams. */
#define EC_STREAM_ID_MIN 1u
#define EC_STREAM_ID_MAX 31u
/* STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE is 17 bits wide on Blackhole. */
#define EC_STREAM_COUNTER_BITS 17u

/* tt_metal/api/tt-metalium/kernel_types.hpp:49, the portable runtime-argument
 * ceiling. */
#define EC_METAL_MAX_RUNTIME_ARGS 341u

/* sizeof(erisc::datamover::handshake::handshake_info_t). */
#define EC_HANDSHAKE_BYTES 32u

#define EC_DEFAULT_POLL_CADENCE 1024u
#define EC_MAX_POLL_CADENCE (1u << 20)

#define EC_RESULT_BYTES 256u
#define EC_MAX_STALL_SITES 4u

/* Barrier workers per chip: E0, E1, and every forwarder but the director. The
 * ready table reserves a slot per possible worker index. */
#define EC_BARRIER_MAX_WORKERS (2u + EC_MAX_P)

/* --------------------------------------------------------------------------
 * Enumerations
 * -------------------------------------------------------------------------- */
typedef enum { EC_STATE_RUNNING = 0u, EC_STATE_DONE = 1u, EC_STATE_FAILED = 2u } ec_state_t;

typedef enum {
    EC_ERROR_NONE = 0u,
    EC_ERROR_STALL_TIMEOUT = 1u, /* detail = ec_site_t */
    EC_ERROR_BAD_SEQUENCE = 3u,  /* detail = expected sequence number */
    EC_ERROR_COUNTER_DISTANCE = 4u,
    EC_ERROR_LINK_DOWN = 5u,
    EC_ERROR_BAD_ARGS = 6u
} ec_error_t;

/* The shared outcome of a kernel's status: DONE is OK, RUNNING IN_PROGRESS,
 * and FAILED its code's -- STALL_TIMEOUT TIMEOUT, BAD_ARGS
 * ARG_LAYOUT_MISMATCH, BAD_SEQUENCE PATH_MISMATCH (a slot held the wrong
 * packet), and COUNTER_DISTANCE, LINK_DOWN or no code TRAP. An unknown state
 * or code is TRAP. */
static inline ubench_outcome_t ec_outcome(ec_state_t state, ec_error_t code) {
    switch (state) {
        case EC_STATE_DONE: return UBENCH_OUTCOME_OK;
        case EC_STATE_RUNNING: return UBENCH_OUTCOME_IN_PROGRESS;
        case EC_STATE_FAILED:
            switch (code) {
                case EC_ERROR_STALL_TIMEOUT: return UBENCH_OUTCOME_TIMEOUT;
                case EC_ERROR_BAD_ARGS: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
                case EC_ERROR_BAD_SEQUENCE: return UBENCH_OUTCOME_PATH_MISMATCH;
                case EC_ERROR_NONE:
                case EC_ERROR_COUNTER_DISTANCE:
                case EC_ERROR_LINK_DOWN: return UBENCH_OUTCOME_TRAP;
            }
            return UBENCH_OUTCOME_TRAP;
    }
    return UBENCH_OUTCOME_TRAP;
}

typedef enum { EC_ROLE_NONE = 0u, EC_ROLE_X280_FWD = 1u, EC_ROLE_E0 = 2u, EC_ROLE_E1 = 3u } ec_role_t;

/* Stall sites, unique across kernels so a STALL_TIMEOUT detail names the wait
 * without its kernel. The first group are the loop sites a result block
 * accumulates; the second are one-shot waits that only ever appear as a
 * timeout detail. */
typedef enum {
    EC_SITE_NONE = 0u,
    EC_SITE_FWD_OUT_FULL = 1u,
    EC_SITE_FWD_W1_STORE = 2u,
    EC_SITE_FWD_IN_EMPTY = 3u,
    EC_SITE_E0_INPUTS_EMPTY = 4u,
    EC_SITE_E0_LINK_CREDITS = 5u,
    EC_SITE_E0_TXQ0_BUSY = 6u,
    EC_SITE_E1_LINK_EMPTY = 7u,
    EC_SITE_E1_IN_FULL = 8u,
    EC_SITE_E1_NOC_BARRIER = 9u,
    EC_SITE_E1_TXQ1_BUSY = 10u,
    EC_SITE_BARRIER = 11u,
    EC_SITE_HANDSHAKE = 12u,
    EC_SITE_DRAIN = 13u,
    EC_SITE_E0_R3_SOURCE = 14u /* E0's R3 source unit, until the NoC has read it */
} ec_site_t;

/* Whether `site` is a cost site: one that runs right after its own issue, so
 * nearly every batch finds work outstanding and its count is not a stall.
 * Its ticks are time spent; a strict-window W1 store always completes. The
 * host leaves cost sites out of the "stalls least" comparison. */
static inline int ec_site_is_cost(uint32_t site) {
    return site == EC_SITE_FWD_W1_STORE || site == EC_SITE_E0_TXQ0_BUSY || site == EC_SITE_E1_TXQ1_BUSY ||
           site == EC_SITE_E1_NOC_BARRIER;
}

/**
 * The site E0 charges a pass that sends nothing, given its link credit to the
 * peer. Credit is tested before input: the forwarder refills E0's tx_fifo
 * only as credit returns, so with none left the input is empty as well and
 * the credit is the cause.
 */
static inline uint32_t ec_e0_idle_site(uint32_t credit) {
    return credit == 0u ? EC_SITE_E0_LINK_CREDITS : EC_SITE_E0_INPUTS_EMPTY;
}

/* ec_x280_fwd_args_t::flags */
#define EC_FWD_FLAG_DIRECTOR (1u << 0)
#define EC_FWD_FLAG_OUTBOUND_ACTIVE (1u << 1)
#define EC_FWD_FLAG_INBOUND_ACTIVE (1u << 2)
/* ec_e0_args_t::flags */
#define EC_E0_FLAG_HANDSHAKE_SENDER (1u << 0)
#define EC_E0_FLAG_OUTBOUND_ACTIVE (1u << 1)

/* --------------------------------------------------------------------------
 * Kernel identity
 * -------------------------------------------------------------------------- */

/**
 * Builds a kernel id: bit 15 is the chip (0 or 1 of the pair), bits [11:8] the
 * ec_role_t, bits [7:0] the forwarder index `p` (0 for E0/E1).
 * Zero is no kernel, because EC_ROLE_NONE is zero.
 */
#define EC_KERNEL_ID(role, index, chip) \
    ((uint16_t)((((uint32_t)(chip) & 1u) << 15) | (((uint32_t)(role) & 0xFu) << 8) | ((uint32_t)(index) & 0xFFu)))

static inline uint32_t ec_kernel_role(uint32_t kernel_id) { return (kernel_id >> 8) & 0xFu; }
static inline uint32_t ec_kernel_index(uint32_t kernel_id) { return kernel_id & 0xFFu; }
static inline uint32_t ec_kernel_chip(uint32_t kernel_id) { return (kernel_id >> 15) & 1u; }

/* --------------------------------------------------------------------------
 * 16-byte control units
 * -------------------------------------------------------------------------- */

/**
 * One pointer, counter or sequence word. Each occupies its own aligned 16-byte
 * unit: the NoC writes L1 in atomic 16-byte units, and a NoC write's source and
 * destination must be congruent mod 16 or it never lands, so a local pointer
 * and the peer shadow it is copied into both sit at offset 0 of a unit.
 */
typedef struct {
    EC_ALIGNAS(16) uint32_t value;
    uint32_t pad[3];
} ec_unit_t;

EC_STATIC_ASSERT(sizeof(ec_unit_t) == EC_UNIT_BYTES, "ec_unit_t must be one 16-byte unit");
EC_STATIC_ASSERT(EC_ALIGNOF(ec_unit_t) == 16u, "ec_unit_t must be 16-aligned");

/**
 * A kernel's terminal state, in its own result block. `state` is an ec_state_t
 * and is 0 (running) as the host initialises it. `origin` is the kernel's own
 * id and `code` its ec_error_t. `detail` is qualified by `code` (see
 * ec_error_t).
 */
typedef struct {
    EC_ALIGNAS(16) uint32_t state;
    uint16_t origin;
    uint16_t code;
    uint32_t detail;
    uint32_t reserved;
} ec_status_t;

EC_STATIC_ASSERT(sizeof(ec_status_t) == EC_UNIT_BYTES, "ec_status_t must be one 16-byte unit");
EC_STATIC_ASSERT(offsetof(ec_status_t, state) == 0u, "offset state");
EC_STATIC_ASSERT(offsetof(ec_status_t, origin) == 4u, "offset origin");
EC_STATIC_ASSERT(offsetof(ec_status_t, code) == 6u, "offset code");
EC_STATIC_ASSERT(offsetof(ec_status_t, detail) == 8u, "offset detail");

/* A 64-bit argument from its two words, as the host splits it. */
static inline uint64_t ec_u64(uint32_t lo, uint32_t hi) { return (uint64_t)lo | ((uint64_t)hi << 32); }

/* A status word from its fields; `origin` and `code` are truncated to 16 bits. */
static inline ec_status_t ec_status_word(uint32_t state, uint32_t origin, uint32_t code, uint32_t detail) {
    ec_status_t s;
    s.state = state;
    s.origin = (uint16_t)origin;
    s.code = (uint16_t)code;
    s.detail = detail;
    s.reserved = 0u;
    return s;
}

/* --------------------------------------------------------------------------
 * Result block
 * -------------------------------------------------------------------------- */

/**
 * Per stall site: passes with no progress, and the ticks they took. The
 * exceptions are the cost sites (ec_site_is_cost), which count batches and
 * the ticks spent on them: EC_SITE_FWD_W1_STORE, EC_SITE_E0_TXQ0_BUSY,
 * EC_SITE_E1_TXQ1_BUSY and EC_SITE_E1_NOC_BARRIER.
 */
typedef struct {
    uint32_t site; /* ec_site_t, EC_SITE_NONE for an unused entry */
    uint32_t pad;
    uint64_t count;
    uint64_t ticks;
} ec_stall_t;

EC_STATIC_ASSERT(sizeof(ec_stall_t) == 24u, "ec_stall_t must be 24 bytes");

/* A stall total that cannot be trusted: an interval at or past half-range (the
 * modulo-64 rule of measurement_delta, rt/measurement.h), or a sum that reached
 * half-range. Every later addition keeps it. */
#define EC_STALL_TICKS_INVALID UINT64_MAX

/**
 * `total` plus `ticks` already measured; EC_STALL_TICKS_INVALID when either is
 * at or past half-range (EC_STALL_TICKS_INVALID is), or the sum reaches it.
 * Both below half-range cannot wrap, so one test of the three top bits decides.
 * Branch-light: the eth_credit forwarder adds inside its timed window.
 */
static inline uint64_t ec_stall_sum(uint64_t total, uint64_t ticks) {
    const uint64_t sum = total + ticks;
    return ((total | ticks | sum) >> 63) ? EC_STALL_TICKS_INVALID : sum;
}

/**
 * `total` plus one counter's modulo-64 interval [start, end], as ec_stall_sum:
 * an interval at or past half-range, which measurement_delta refuses, leaves
 * EC_STALL_TICKS_INVALID.
 */
static inline uint64_t ec_stall_add(uint64_t total, uint64_t start, uint64_t end) {
    return ec_stall_sum(total, end - start);
}

/**
 * Failure evidence. For BAD_SEQUENCE, `p`, `expected` and `seen`; for a
 * timeout, `site` and the last local and shadow pointers it compared.
 */
typedef struct {
    uint32_t p;
    uint32_t expected;
    uint32_t seen;
    uint32_t site;
    uint32_t local_ptr;
    uint32_t shadow_ptr;
    uint32_t pad[2];
} ec_failure_detail_t;

EC_STATIC_ASSERT(sizeof(ec_failure_detail_t) == 32u, "ec_failure_detail_t must be 32 bytes");

/**
 * One kernel's aggregates, written only by that kernel into the cold part of
 * its own control area, and read by the host after the run. `magic` and
 * `kernel_id` must match before any other field is believed; `status` is
 * the kernel's own terminal status. Timestamps are on the kernel's own clock
 * and are never compared with another kernel's.
 */
typedef struct {
    EC_ALIGNAS(16) uint32_t magic;
    uint32_t kernel_id;
    /* A forwarder's packets_in[p] when it stamped first_ts; zero for E0, E1. */
    uint32_t first_batch_packets;
    uint32_t reserved0;
    ec_status_t status;
    uint32_t packets_in[EC_MAX_P];
    uint32_t packets_out[EC_MAX_P];
    ec_stall_t stalls[EC_MAX_STALL_SITES];
    uint64_t first_ts;
    uint64_t last_ts;
    ec_failure_detail_t failure;
    uint8_t reserved_tail[48];
} ec_result_t;

EC_STATIC_ASSERT(sizeof(ec_result_t) == EC_RESULT_BYTES, "ec_result_t must be EC_RESULT_BYTES");
EC_STATIC_ASSERT(offsetof(ec_result_t, status) == 16u, "offset status");
EC_STATIC_ASSERT(offsetof(ec_result_t, packets_in) == 32u, "offset packets_in");
EC_STATIC_ASSERT(offsetof(ec_result_t, packets_out) == 48u, "offset packets_out");
EC_STATIC_ASSERT(offsetof(ec_result_t, stalls) == 64u, "offset stalls");
EC_STATIC_ASSERT(offsetof(ec_result_t, first_ts) == 160u, "offset first_ts");
EC_STATIC_ASSERT(offsetof(ec_result_t, failure) == 176u, "offset failure");

/* --------------------------------------------------------------------------
 * Shape
 * -------------------------------------------------------------------------- */

/**
 * The compile-time shape of a run. ERISC kernels are JIT-built per shape and
 * also receive it as runtime arguments to compare; forwarders take it at run
 * time.
 */
typedef struct {
    uint32_t P;
    uint32_t n_link;
    uint32_t n_deliver;
    uint32_t slot_size;
    uint32_t batch;
} ec_shape_t;

/* --------------------------------------------------------------------------
 * Checked outcomes
 * -------------------------------------------------------------------------- */

typedef enum {
    EC_CHECK_OK = 0u,
    EC_CHECK_BAD_PARAM = 1u,
    EC_CHECK_ALIGNMENT = 2u,
    EC_CHECK_CAPACITY = 3u,
    EC_CHECK_CONGRUENCE = 4u,
    EC_CHECK_STREAM_ID = 5u,
    EC_CHECK_ARG_WORDS = 6u,
    EC_CHECK_COORD = 7u
} ec_check_code_t;

/** Which constraint a check names. */
typedef enum {
    EC_CONSTRAINT_NONE = 0u,
    EC_CONSTRAINT_P = 1u,
    EC_CONSTRAINT_N_LINK = 2u,
    EC_CONSTRAINT_N_DELIVER = 3u,
    EC_CONSTRAINT_SLOT_SIZE = 4u,
    EC_CONSTRAINT_BATCH = 5u,
    EC_CONSTRAINT_ETH_WINDOW = 6u,
    EC_CONSTRAINT_LIM_WINDOW = 7u,
    EC_CONSTRAINT_ETH_BASE = 8u,
    EC_CONSTRAINT_LIM_BASE = 9u,
    EC_CONSTRAINT_DIRECTION = 10u,
    EC_CONSTRAINT_ARGS = 11u,
    EC_CONSTRAINT_PEER_BASE = 12u,
    EC_CONSTRAINT_FWD_AREA = 13u,
    EC_CONSTRAINT_STREAM = 14u,
    EC_CONSTRAINT_READY = 15u,
    EC_CONSTRAINT_HOST_WAIT = 16u
} ec_constraint_t;

/**
 * A check outcome. `code` is an ec_check_code_t and `constraint` the
 * ec_constraint_t it failed; `value` is the offending quantity and `bound` the
 * limit. For EC_CHECK_CAPACITY `value` is the bytes required and `bound` the
 * bytes available, so the overrun is their difference.
 */
typedef struct {
    uint32_t code;
    uint32_t constraint;
    uint64_t value;
    uint64_t bound;
} ec_check_t;

static inline ec_check_t ec_check(uint32_t code, uint32_t constraint, uint64_t value, uint64_t bound) {
    ec_check_t c;
    c.code = code;
    c.constraint = constraint;
    c.value = value;
    c.bound = bound;
    return c;
}

static inline ec_check_t ec_check_ok(void) { return ec_check(EC_CHECK_OK, EC_CONSTRAINT_NONE, 0u, 0u); }

static inline int ec_is_pow2(uint32_t x) { return x != 0u && (x & (x - 1u)) == 0u; }

/**
 * Validates a shape. Depths are powers of two in [1, EC_MAX_DEPTH];
 * `slot_size` is 96 + 16k up to EC_MAX_SLOT_BYTES; 1 <= batch <=
 * min(n_link, n_deliver).
 *
 * @return the first failing constraint with its value and bound, or OK.
 */
static inline ec_check_t ec_validate_shape(const ec_shape_t* s) {
    uint32_t min_depth;
    if (s == NULL) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, 0u, 0u);
    }
    if (s->P == 0u || s->P > EC_MAX_P) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_P, s->P, EC_MAX_P);
    }
    if (!ec_is_pow2(s->n_link) || s->n_link > EC_MAX_DEPTH) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_N_LINK, s->n_link, EC_MAX_DEPTH);
    }
    if (!ec_is_pow2(s->n_deliver) || s->n_deliver > EC_MAX_DEPTH) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_N_DELIVER, s->n_deliver, EC_MAX_DEPTH);
    }
    if (s->slot_size < EC_HEADER_BYTES || s->slot_size > EC_MAX_SLOT_BYTES ||
        ((s->slot_size - EC_HEADER_BYTES) % EC_UNIT_BYTES) != 0u) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_SLOT_SIZE, s->slot_size, EC_MAX_SLOT_BYTES);
    }
    min_depth = s->n_link < s->n_deliver ? s->n_link : s->n_deliver;
    if (s->batch == 0u || s->batch > min_depth) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_BATCH, s->batch, min_depth);
    }
    return ec_check_ok();
}

static inline uint32_t ec_payload_bytes(const ec_shape_t* s) { return s->slot_size - EC_HEADER_BYTES; }

/* --------------------------------------------------------------------------
 * 2N-wrap pointers
 *
 * Every pointer is a uint32 over 0 .. 2N-1 for its FIFO's power-of-two depth
 * N. `n` must be a power of two no larger than EC_MAX_DEPTH, which
 * ec_validate_shape() establishes; these are pure arithmetic and do not check.
 * -------------------------------------------------------------------------- */

static inline uint32_t ec_ptr_mask(uint32_t n) { return 2u * n - 1u; }
static inline uint32_t ec_ptr_advance(uint32_t x, uint32_t k, uint32_t n) { return (x + k) & ec_ptr_mask(n); }
static inline uint32_t ec_ptr_slot(uint32_t x, uint32_t n) { return x & (n - 1u); }
static inline uint32_t ec_ptr_used(uint32_t wr, uint32_t rd, uint32_t n) { return credit_used(wr, rd, ec_ptr_mask(n)); }
static inline int ec_ptr_empty(uint32_t wr, uint32_t rd) { return wr == rd; }
static inline int ec_ptr_full(uint32_t wr, uint32_t rd, uint32_t n) { return ec_ptr_used(wr, rd, n) == n; }

/**
 * Whether a pointer pair read from a FIFO is consistent: both inside the 2N
 * range and no more than N apart. A violation is EC_ERROR_COUNTER_DISTANCE.
 */
static inline int ec_ptr_distance_ok(uint32_t wr, uint32_t rd, uint32_t n) {
    uint32_t used;
    return credit_pair_ok(wr, rd, ec_ptr_mask(n), n, &used);
}

/** Free slots a producer at `wr` may fill against the consumer's `rd`. */
static inline uint32_t ec_ptr_credit(uint32_t wr, uint32_t rd, uint32_t n) { return n - ec_ptr_used(wr, rd, n); }

/* --------------------------------------------------------------------------
 * Barrier worker indices
 *
 * One director per chip; every other participant has a fixed worker index,
 * which is its slot in the director's ready table and the salt of its token.
 * -------------------------------------------------------------------------- */

static inline uint32_t ec_barrier_index_e0(void) { return 0u; }
static inline uint32_t ec_barrier_index_e1(void) { return 1u; }
static inline uint32_t ec_barrier_index_fwd(uint32_t p) { return 2u + p; }

/* --------------------------------------------------------------------------
 * Layouts
 *
 * Every offset is a byte offset from its region's base and a multiple of 16,
 * and every base is aligned to at least 16, so any two units are congruent mod
 * 16. Arrays indexed by `p` or `d` have a 16-byte stride unless named
 * otherwise. Sizes are bounded by ec_validate_shape()'s maxima, so no formula
 * below can overflow 32 bits: the largest, P * N * SLOT, is under 2^29.
 * -------------------------------------------------------------------------- */

/**
 * One Ethernet tile's L1: a control area shared by E0 and E1 at a 512-byte
 * aligned base, followed by E0's tx_fifo slots and E1's rx slots.
 *
 * The control area's unit order and its alignment are intentional, for
 * banking. Blackhole Ethernet L1 is 32 banks x 1024 x 128-bit
 * (EthernetSsDesignSpec, SharePoint Blackhole Documents/Ethernet). The
 * interleave bank = (addr >> 4) & 31 is inferred from the Wormhole Tensix L1
 * MAS ("addr[3:0] as a bank select" on the 16-byte word address) and the Neo L1
 * spec's default low-order interleave; it is not documented for Blackhole
 * Ethernet. Under it, hot unit i of a 512-aligned area lands in bank i. Units
 * are grouped by the core that polls them, E0's then E1's, hot units first, so
 * the 6P hot units have distinct banks for every P <= EC_MAX_P.
 */
typedef struct {
    uint32_t P;

    /* E0 hot units: 3P. */
    uint32_t e0_wr_sh;  /* [P] W2 target, written by forwarder p */
    uint32_t e0_crd;    /* [P] R2 target, written by the peer E1 */
    uint32_t e0_rd_src; /* [P] R3 source */

    /* E1 hot units: 3P. */
    uint32_t e1_out_rd_sh;  /* [P] R1 target, written by forwarder p */
    uint32_t e1_crd_src;    /* [P] R2 source */
    uint32_t e1_out_wr_src; /* [P] W6 source */

    uint32_t hot_units; /* 6P */

    /* Cold units. */
    uint32_t e0_release; /* core_sync release slot */
    uint32_t e1_release;
    uint32_t handshake; /* EC_HANDSHAKE_BYTES, same offset on both tiles */
    uint32_t e0_result; /* ec_result_t */
    uint32_t e1_result;
    uint32_t control_bytes;

    /* Slots, each sized to need: zero bytes when that direction is idle. */
    uint32_t fifo_stride; /* N_LINK * SLOT_SIZE, one p's FIFO */
    uint32_t tx_offset;   /* tx_fifo[p] at tx_offset + p * fifo_stride */
    uint32_t tx_bytes;
    uint32_t rx_offset; /* rx[p] at rx_offset + p * fifo_stride */
    uint32_t rx_bytes;
    uint32_t total_bytes;
} ec_eth_layout_t;

/**
 * Derives an Ethernet tile's layout. `outbound_active` sizes E0's tx_fifo and
 * `inbound_active` E1's rx; the control area is always complete, as E0 runs on
 * both tiles in every run.
 *
 * @param out written only on success.
 * @return OK, or the shape's first failing constraint.
 */
static inline ec_check_t ec_compute_eth_layout(
    const ec_shape_t* s, int outbound_active, int inbound_active, ec_eth_layout_t* out) {
    ec_check_t c = ec_validate_shape(s);
    uint32_t u = 0u;
    ec_eth_layout_t l;
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    if (out == NULL) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, 0u, 0u);
    }
    l.P = s->P;
    l.e0_wr_sh = u * EC_UNIT_BYTES;
    u += s->P;
    l.e0_crd = u * EC_UNIT_BYTES;
    u += s->P;
    l.e0_rd_src = u * EC_UNIT_BYTES;
    u += s->P;
    l.e1_out_rd_sh = u * EC_UNIT_BYTES;
    u += s->P;
    l.e1_crd_src = u * EC_UNIT_BYTES;
    u += s->P;
    l.e1_out_wr_src = u * EC_UNIT_BYTES;
    u += s->P;
    l.hot_units = u;
    l.e0_release = u * EC_UNIT_BYTES;
    u += 1u;
    l.e1_release = u * EC_UNIT_BYTES;
    u += 1u;
    l.handshake = u * EC_UNIT_BYTES;
    u += EC_HANDSHAKE_BYTES / EC_UNIT_BYTES;
    l.e0_result = u * EC_UNIT_BYTES;
    u += EC_RESULT_BYTES / EC_UNIT_BYTES;
    l.e1_result = u * EC_UNIT_BYTES;
    u += EC_RESULT_BYTES / EC_UNIT_BYTES;
    l.control_bytes = u * EC_UNIT_BYTES;

    l.fifo_stride = s->n_link * s->slot_size;
    l.tx_offset = l.control_bytes;
    l.tx_bytes = outbound_active ? s->P * l.fifo_stride : 0u;
    l.rx_offset = l.tx_offset + l.tx_bytes;
    l.rx_bytes = inbound_active ? s->P * l.fifo_stride : 0u;
    l.total_bytes = l.rx_offset + l.rx_bytes;
    *out = l;
    return ec_check_ok();
}

/**
 * One chip's X280 LIM: the per-hart args block, the barrier ready table, and
 * one area per forwarder p at `fwd_offset + p * fwd_stride`. Offsets inside an
 * area are relative to the area.
 */
typedef struct {
    uint32_t args_offset; /* ec_x280_fwd_args_t[EC_X280_HARTS] */
    uint32_t args_bytes;
    uint32_t ready_offset; /* core_sync_slot_t[EC_BARRIER_MAX_WORKERS] */
    uint32_t ready_bytes;
    uint32_t fwd_offset;
    uint32_t fwd_stride;

    uint32_t rd_sh;  /* R3 target, written by E0 */
    uint32_t wr_src; /* this forwarder's tx_fifo write pointer */
    uint32_t wr_sh;  /* W6 target, written by E1 */
    uint32_t rd_src; /* this forwarder's in[p] read pointer */
    uint32_t seq_expected;
    uint32_t release;  /* core_sync release slot */
    uint32_t result;   /* ec_result_t */
    uint32_t staging;  /* W1 header staging, EC_HEADER_BYTES */
    uint32_t in_slots; /* in[p], N_DELIVER x EC_HEADER_BYTES */
    uint32_t area_bytes;
    uint32_t total_bytes;
} ec_x280_layout_t;

/** Derives the X280 LIM layout. @param out written only on success. */
static inline ec_check_t ec_compute_x280_layout(const ec_shape_t* s, ec_x280_layout_t* out) {
    ec_check_t c = ec_validate_shape(s);
    uint32_t u = 0u;
    ec_x280_layout_t l;
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    if (out == NULL) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_NONE, 0u, 0u);
    }
    l.args_offset = 0u;
    l.args_bytes = EC_X280_HARTS * EC_X280_FWD_ARGS_BYTES;
    l.ready_offset = l.args_offset + l.args_bytes;
    l.ready_bytes = x280_align_up(EC_BARRIER_MAX_WORKERS * CORE_SYNC_SLOT_BYTES, EC_LIM_ALIGN);
    l.fwd_offset = l.ready_offset + l.ready_bytes;

    l.rd_sh = u * EC_UNIT_BYTES;
    u += 1u;
    l.wr_src = u * EC_UNIT_BYTES;
    u += 1u;
    l.wr_sh = u * EC_UNIT_BYTES;
    u += 1u;
    l.rd_src = u * EC_UNIT_BYTES;
    u += 1u;
    l.seq_expected = u * EC_UNIT_BYTES;
    u += 1u;
    l.release = u * EC_UNIT_BYTES;
    u += 1u;
    l.result = u * EC_UNIT_BYTES;
    u += EC_RESULT_BYTES / EC_UNIT_BYTES;
    l.staging = u * EC_UNIT_BYTES;
    u += EC_HEADER_BYTES / EC_UNIT_BYTES;
    l.in_slots = u * EC_UNIT_BYTES;
    l.area_bytes = l.in_slots + s->n_deliver * EC_HEADER_BYTES;
    l.fwd_stride = x280_align_up(l.area_bytes, EC_LIM_ALIGN);
    l.total_bytes = l.fwd_offset + s->P * l.fwd_stride;
    *out = l;
    return ec_check_ok();
}

/* --------------------------------------------------------------------------
 * Arguments
 * -------------------------------------------------------------------------- */

/**
 * Forwarder `p` arguments, one per hart in an EC_X280_HARTS array at the LIM
 * layout's args offset. A hart whose record has `magic` zero is not a
 * forwarder and returns. Host-written before launch and immutable.
 *
 * `lim_base` is the absolute LIM address of the X280 layout; every LIM address
 * is derived from it. `eth` is the local Ethernet tile's translated coordinate,
 * which the outbound TLB window takes, and `eth_ctrl_base` is its control area
 * in ERISC L1; the tx_fifo region follows it at the layout's tx_offset. `send_count` and
 * `recv_count` are the packets this forwarder sends on tx_fifo[p] and consumes
 * from in[p]. `stall_budget_ticks` bounds every wait without progress, in
 * mtime ticks. `tlb_base` is this hart's window onto the Ethernet tile.
 * `lim_end` is the end of the LIM the host admitted the layout against, its
 * backed and available LIM; ec_narrow_lim bounds this hart's area by it.
 */
typedef struct {
    EC_ALIGNAS(64) uint32_t magic;
    uint32_t kernel_id;
    uint32_t flags; /* EC_FWD_FLAG_* */
    uint32_t p;
    ec_shape_t shape;
    uint32_t director_p;
    uint32_t run_nonce;
    uint32_t poll_cadence;
    uint32_t send_count;
    uint32_t recv_count;
    uint16_t dst_chip;
    uint16_t dst_mesh;
    uint32_t tlb_base;
    uint64_t stall_budget_ticks;
    uint64_t lim_base;
    x280_noc_coord_t eth;
    uint32_t eth_ctrl_base;
    uint32_t pad;
    uint64_t lim_end;
    uint32_t reserved[6];
} ec_x280_fwd_args_t;

EC_STATIC_ASSERT(
    sizeof(ec_x280_fwd_args_t) == EC_X280_FWD_ARGS_BYTES, "ec_x280_fwd_args_t must be EC_X280_FWD_ARGS_BYTES");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, magic) == 0u, "offset magic");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, kernel_id) == 4u, "offset kernel_id");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, shape) == 16u, "offset shape");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, stall_budget_ticks) == 64u, "offset stall_budget_ticks");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, lim_base) == 72u, "offset lim_base");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, eth) == 80u, "offset eth");
EC_STATIC_ASSERT(offsetof(ec_x280_fwd_args_t, lim_end) == 96u, "offset lim_end");

/*
 * Metal roles take their arguments as runtime words. Each record below is all
 * uint32_t with no padding, so runtime word i is its i-th uint32_t in
 * declaration order: the host serializes a record by copying its bytes, and a
 * kernel reads the words straight into one. The size asserts pin both.
 */

/**
 * The prefix E0 and E1 share, first in both records. `ctrl_base` is this
 * tile's control area. `fwd` is the X280 tile as the role addresses it on its
 * own NoC and `fwd_area[p]` forwarder p's LIM area. `barrier_ready_base` is the
 * director's ready table, where each role arrives at its fixed worker index;
 * `barrier_cap` bounds each barrier poll. `lim_end` is the end of the LIM the
 * host admitted the X280 layout against; ec_narrow_lim bounds every LIM
 * address this record names by it.
 */
typedef struct {
    uint32_t magic, kernel_id, flags;
    ec_shape_t shape;
    uint32_t run_nonce, poll_cadence, stall_budget_lo, stall_budget_hi;
    uint32_t ctrl_base;
    x280_noc_coord_t fwd;
    uint32_t barrier_ready_base;
    uint32_t barrier_cap_lo, barrier_cap_hi;
    uint32_t count[EC_MAX_P];
    uint32_t fwd_area[EC_MAX_P];
    uint32_t stream_id[EC_MAX_P];
    uint32_t lim_end_lo, lim_end_hi;
} ec_erisc_args_t;

/**
 * E0 arguments. tx_fifo sits at the layout's tx_offset past `erisc.ctrl_base`;
 * `peer_ctrl_base`/`peer_rx_base` are the peer tile's control area and rx
 * region. `erisc.stream_id[p]` names the peer tile's pkts[p] counter, which W4
 * increments; `local_stream_id[p]` names this tile's, which E0 resets before
 * its handshake.
 */
typedef struct {
    ec_erisc_args_t erisc;
    uint32_t peer_ctrl_base, peer_rx_base;
    uint32_t local_stream_id[EC_MAX_P];
} ec_e0_args_t;

/**
 * E1 arguments. `rx_base` is this tile's rx region; `peer_ctrl_base` the peer
 * tile's control area, where R2 lands. `erisc.stream_id[p]` names this tile's
 * pkts[p].
 */
typedef struct {
    ec_erisc_args_t erisc;
    uint32_t rx_base, peer_ctrl_base;
} ec_e1_args_t;

#define EC_ERISC_ARG_WORDS 32u
#define EC_E0_ARG_WORDS (EC_ERISC_ARG_WORDS + 6u)
#define EC_E1_ARG_WORDS (EC_ERISC_ARG_WORDS + 2u)

EC_STATIC_ASSERT(sizeof(ec_erisc_args_t) == EC_ERISC_ARG_WORDS * 4u, "the shared prefix has no padding");
EC_STATIC_ASSERT(offsetof(ec_erisc_args_t, shape) == 12u, "offset shape");
EC_STATIC_ASSERT(offsetof(ec_erisc_args_t, count) == 72u, "offset count");
EC_STATIC_ASSERT(offsetof(ec_erisc_args_t, stream_id) == 104u, "offset stream_id");
EC_STATIC_ASSERT(offsetof(ec_erisc_args_t, lim_end_lo) == 120u, "offset lim_end_lo");
EC_STATIC_ASSERT(
    offsetof(ec_e0_args_t, erisc) == 0u && offsetof(ec_e1_args_t, erisc) == 0u,
    "both roles' records begin with the shared prefix");
EC_STATIC_ASSERT(sizeof(ec_e0_args_t) == EC_E0_ARG_WORDS * 4u, "E0 args are exactly EC_E0_ARG_WORDS words");
EC_STATIC_ASSERT(sizeof(ec_e1_args_t) == EC_E1_ARG_WORDS * 4u, "E1 args are exactly EC_E1_ARG_WORDS words");
EC_STATIC_ASSERT(offsetof(ec_e0_args_t, local_stream_id) == 136u, "offset local_stream_id");
EC_STATIC_ASSERT(offsetof(ec_e1_args_t, peer_ctrl_base) == 132u, "offset E1 peer_ctrl_base");
EC_STATIC_ASSERT(EC_E0_ARG_WORDS <= EC_METAL_MAX_RUNTIME_ARGS, "E0 args exceed Metal's runtime-argument ceiling");
EC_STATIC_ASSERT(EC_E1_ARG_WORDS <= EC_METAL_MAX_RUNTIME_ARGS, "E1 args exceed Metal's runtime-argument ceiling");

static inline int ec_shape_equal(const ec_shape_t* a, const ec_shape_t* b) {
    return a->P == b->P && a->n_link == b->n_link && a->n_deliver == b->n_deliver && a->slot_size == b->slot_size &&
           a->batch == b->batch;
}

/* --------------------------------------------------------------------------
 * Tail marker
 *
 * The admission test stamps the last 16 bytes of each W3 slot with a marker
 * naming the packet, which E1 checks as soon as pkts[p] shows it: a W3 whose
 * tail had not committed ahead of its W4 would still hold an older marker.
 * -------------------------------------------------------------------------- */
#define EC_TAIL_MARKER_BYTES 16u
#define EC_TAIL_MARKER_MAGIC 0xEC7A1100u

/** The four marker words for packet `ordinal` (0-based, per p) of forwarder `p`. */
static inline void ec_tail_marker(uint32_t p, uint32_t ordinal, uint32_t out[4]) {
    out[0] = EC_TAIL_MARKER_MAGIC ^ p;
    out[1] = ordinal;
    out[2] = ~ordinal;
    out[3] = p;
}

/* --------------------------------------------------------------------------
 * Argument checks
 *
 * Every address a Metal role writes through comes from its arguments, so each
 * is bounded before use. A kernel passes the memory it may address from its
 * compile-time memory map; the host passes the windows it admitted against.
 * -------------------------------------------------------------------------- */

/** Memory an argument may name: ERISC L1 [l1_lo, l1_hi) and LIM [lim_lo, lim_hi). */
typedef struct {
    uint32_t l1_lo;
    uint32_t l1_hi;
    uint64_t lim_lo;
    uint64_t lim_hi;
} ec_bounds_t;

/** Whether [base, base + bytes) lies inside [lo, hi), without overflow. */
static inline int ec_range_ok(uint64_t base, uint64_t bytes, uint64_t lo, uint64_t hi) {
    return base >= lo && hi >= lo && bytes <= hi - lo && base - lo <= (hi - lo) - bytes;
}

/**
 * Lowers `map`'s LIM end to `lim_end`, the end the host admitted against, so a
 * LIM way the active shape does not back is refused. `map` is the most a
 * kernel may name. @param out written only on success.
 * @return OK, or CAPACITY/LIM_WINDOW with `lim_end` and `map->lim_hi` when
 *         `lim_end` is not in (lim_lo, lim_hi].
 */
static inline ec_check_t ec_narrow_lim(const ec_bounds_t* map, uint64_t lim_end, ec_bounds_t* out) {
    if (lim_end <= map->lim_lo || lim_end > map->lim_hi) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_LIM_WINDOW, lim_end, map->lim_hi);
    }
    *out = *map;
    out->lim_hi = lim_end;
    return ec_check_ok();
}

/**
 * Where an ERISC's control area may sit: 512-aligned and wholly inside L1.
 * Checked before anything is written through `ctrl_base`, the result block
 * included.
 */
static inline ec_check_t ec_check_erisc_ctrl(uint32_t ctrl_base, const ec_eth_layout_t* l, const ec_bounds_t* b) {
    if ((ctrl_base % EC_ETH_CONTROL_ALIGN) != 0u) {
        return ec_check(EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, ctrl_base, EC_ETH_CONTROL_ALIGN);
    }
    if (!ec_range_ok(ctrl_base, l->control_bytes, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, ctrl_base, b->l1_hi);
    }
    return ec_check_ok();
}

/**
 * What an ERISC writes its forwarders through: the X280 coordinate and every
 * forwarder area, each 64-aligned and wholly inside LIM.
 */
static inline ec_check_t ec_check_fwd_targets(
    x280_noc_coord_t fwd, const uint32_t* fwd_area, uint32_t P, const ec_x280_layout_t* lim, const ec_bounds_t* b) {
    uint32_t p;
    if (!x280_noc_coord_valid(fwd)) {
        return ec_check(EC_CHECK_COORD, EC_CONSTRAINT_NONE, fwd.x, fwd.y);
    }
    for (p = 0; p < P; ++p) {
        if ((fwd_area[p] % EC_LIM_ALIGN) != 0u || !ec_range_ok(fwd_area[p], lim->area_bytes, b->lim_lo, b->lim_hi)) {
            return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_FWD_AREA, fwd_area[p], p);
        }
    }
    return ec_check_ok();
}

/**
 * Stream ids for `P` channels: each in [EC_STREAM_ID_MIN, EC_STREAM_ID_MAX],
 * else STREAM_ID/STREAM with the id and its index p; and pairwise distinct,
 * else STREAM_ID/STREAM with the id and the earlier index sharing it, since
 * two channels on one counter would each count the other's packets.
 */
static inline ec_check_t ec_check_stream_ids(const uint32_t* id, uint32_t P) {
    uint32_t p, q;
    for (p = 0; p < P; ++p) {
        if (id[p] < EC_STREAM_ID_MIN || id[p] > EC_STREAM_ID_MAX) {
            return ec_check(EC_CHECK_STREAM_ID, EC_CONSTRAINT_STREAM, id[p], p);
        }
        for (q = 0; q < p; ++q) {
            if (id[q] == id[p]) {
                return ec_check(EC_CHECK_STREAM_ID, EC_CONSTRAINT_STREAM, id[p], q);
            }
        }
    }
    return ec_check_ok();
}

/* The prefix both Metal roles share: identity, the shape against the
 * compile-time one, and the scalar controls, then `lim_end` narrowing `b` into
 * `*narrowed` (ec_narrow_lim) and the ready table against it. Identity comes
 * first, so a record of another layout is refused as BAD_PARAM/ARGS whatever
 * its later words hold. */
static inline ec_check_t ec_check_erisc_prefix(
    const ec_erisc_args_t* a, uint32_t role, const ec_shape_t* expected, const ec_bounds_t* b, ec_bounds_t* narrowed) {
    const uint64_t budget = ec_u64(a->stall_budget_lo, a->stall_budget_hi);
    const uint64_t cap = ec_u64(a->barrier_cap_lo, a->barrier_cap_hi);
    const uint32_t cadence = a->poll_cadence;
    if (a->magic != EC_ABI_MAGIC || ec_kernel_role(a->kernel_id) != role || !ec_shape_equal(&a->shape, expected) ||
        a->run_nonce == 0u || !ec_is_pow2(cadence) || cadence > EC_MAX_POLL_CADENCE || budget == 0u ||
        budget >= (1ull << 63) || cap == 0u) {
        return ec_check(EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS, a->kernel_id, role);
    }
    const ec_check_t lim = ec_narrow_lim(b, ec_u64(a->lim_end_lo, a->lim_end_hi), narrowed);
    if (lim.code != EC_CHECK_OK) {
        return lim;
    }
    if ((a->barrier_ready_base % CORE_SYNC_SLOT_BYTES) != 0u ||
        !ec_range_ok(
            a->barrier_ready_base, EC_BARRIER_MAX_WORKERS * CORE_SYNC_SLOT_BYTES, narrowed->lim_lo, narrowed->lim_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_READY, a->barrier_ready_base, narrowed->lim_hi);
    }
    return ec_check_stream_ids(a->stream_id, expected->P);
}

/* A peer tile's control area, and a slot region of `bytes` at `region` beside
 * it: 16-aligned, past the peer's control area, and inside L1. */
static inline ec_check_t ec_check_peer_region(
    uint32_t peer_ctrl, uint32_t region, uint32_t bytes, const ec_eth_layout_t* l, const ec_bounds_t* b) {
    if ((peer_ctrl % EC_ETH_CONTROL_ALIGN) != 0u || !ec_range_ok(peer_ctrl, l->control_bytes, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_PEER_BASE, peer_ctrl, b->l1_hi);
    }
    if ((region % EC_UNIT_BYTES) != 0u || region < peer_ctrl + l->control_bytes ||
        !ec_range_ok(region, bytes, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_PEER_BASE, region, b->l1_hi);
    }
    return ec_check_ok();
}

/**
 * Validates every E0 argument against the compile-time shape and `b`, the
 * most it may name, with b's LIM end lowered to the record's `lim_end`: the
 * prefix, the control area, tx_fifo at its layout offset and inside L1 when
 * the direction is active, the peer's control area and rx region (16-aligned,
 * past the peer's control area, inside L1), the forwarder targets, and the
 * peer's and this tile's stream ids.
 */
static inline ec_check_t ec_check_e0_args(const ec_e0_args_t* a, const ec_shape_t* expected, const ec_bounds_t* b) {
    const ec_erisc_args_t* e = &a->erisc;
    ec_eth_layout_t l;
    ec_x280_layout_t lim;
    ec_bounds_t nb;
    ec_check_t c = ec_check_erisc_prefix(e, EC_ROLE_E0, expected, b, &nb);
    const int outbound = (e->flags & EC_E0_FLAG_OUTBOUND_ACTIVE) != 0u;
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    c = ec_compute_eth_layout(expected, 1, 1, &l);
    if (c.code == EC_CHECK_OK) {
        c = ec_compute_x280_layout(expected, &lim);
    }
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    c = ec_check_erisc_ctrl(e->ctrl_base, &l, b);
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    if (outbound &&
        !ec_range_ok((uint64_t)e->ctrl_base + l.tx_offset, expected->P * l.fifo_stride, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, (uint64_t)e->ctrl_base + l.tx_offset, b->l1_hi);
    }
    if (outbound) {
        c = ec_check_peer_region(a->peer_ctrl_base, a->peer_rx_base, expected->P * l.fifo_stride, &l, b);
        if (c.code != EC_CHECK_OK) {
            return c;
        }
    }
    c = ec_check_fwd_targets(e->fwd, e->fwd_area, expected->P, &lim, &nb);
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    return ec_check_stream_ids(a->local_stream_id, expected->P);
}

/**
 * Validates every E1 argument against the compile-time shape and `b`, the
 * most it may name, with b's LIM end lowered to the record's `lim_end`: the
 * prefix, the control area, the rx region (16-aligned, past the control area,
 * the whole P x N_LINK slots inside L1), the peer's control area, and the
 * forwarder targets.
 */
static inline ec_check_t ec_check_e1_args(const ec_e1_args_t* a, const ec_shape_t* expected, const ec_bounds_t* b) {
    const ec_erisc_args_t* e = &a->erisc;
    ec_eth_layout_t l;
    ec_x280_layout_t lim;
    ec_bounds_t nb;
    ec_check_t c = ec_check_erisc_prefix(e, EC_ROLE_E1, expected, b, &nb);
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    c = ec_compute_eth_layout(expected, 1, 1, &l);
    if (c.code == EC_CHECK_OK) {
        c = ec_compute_x280_layout(expected, &lim);
    }
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    c = ec_check_erisc_ctrl(e->ctrl_base, &l, b);
    if (c.code != EC_CHECK_OK) {
        return c;
    }
    if ((a->rx_base % EC_UNIT_BYTES) != 0u || a->rx_base < e->ctrl_base + l.control_bytes ||
        !ec_range_ok(a->rx_base, expected->P * l.fifo_stride, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, a->rx_base, b->l1_hi);
    }
    if ((a->peer_ctrl_base % EC_ETH_CONTROL_ALIGN) != 0u ||
        !ec_range_ok(a->peer_ctrl_base, l.control_bytes, b->l1_lo, b->l1_hi)) {
        return ec_check(EC_CHECK_CAPACITY, EC_CONSTRAINT_PEER_BASE, a->peer_ctrl_base, b->l1_hi);
    }
    return ec_check_fwd_targets(e->fwd, e->fwd_area, expected->P, &lim, &nb);
}

/** Bank of an Ethernet L1 byte address under the inferred low-order interleave. */
static inline uint32_t ec_eth_bank(uint32_t addr) { return (addr >> 4) & (EC_ETH_L1_BANKS - 1u); }

EC_STATIC_ASSERT((EC_HEADER_BYTES % EC_UNIT_BYTES) == 0u, "the header must be whole 16-byte units");
EC_STATIC_ASSERT((EC_MAX_PAYLOAD_BYTES % EC_UNIT_BYTES) == 0u, "the payload maximum must be whole units");
EC_STATIC_ASSERT(EC_MAX_PAYLOAD_BYTES <= 0xFFFFu, "payload_size_bytes is a uint16");
EC_STATIC_ASSERT(
    EC_MAX_DEPTH < (1u << (EC_STREAM_COUNTER_BITS - 1u)),
    "pkts[p] never exceeds N_LINK, which must fit the stream counter");
EC_STATIC_ASSERT(
    (uint64_t)EC_MAX_P* EC_MAX_DEPTH* EC_MAX_SLOT_BYTES < (1ull << 32), "the largest slot region must fit 32 bits");
EC_STATIC_ASSERT(sizeof(ec_x280_fwd_args_t) % EC_LIM_ALIGN == 0u, "forwarder args must be whole cache lines");

#endif /* X280_ETH_CREDIT_ABI_H */
