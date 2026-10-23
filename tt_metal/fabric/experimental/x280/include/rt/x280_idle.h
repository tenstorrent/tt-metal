// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_idle.h -- the LIM ping-pong exit path.
 *
 * LIM-resident firmware cannot simply stop. The X280 cannot re-enter reset
 * without a chip-level reset, so the host loads a small resident idle FW into
 * the lowest 4 KiB of LIM once per chip reset (src/lim_idle.c) and then
 * ping-pongs control with successive active firmwares. An active FW that parks
 * instead of jumping back strands the L2CPU out of reset with no recovery short
 * of `tt-smi -r`.
 *
 * WFI is not an alternative: host NOC writes raise no interrupt on this part, so
 * a WFI-parked hart never wakes. Nor is CEASE, which stops the hart until the
 * next reset. The only correct exit is to jump to the idle FW's _start.
 *
 * Static inline so each freestanding firmware target can use it directly.
 */
#ifndef X280_RT_IDLE_H
#define X280_RT_IDLE_H

#include <stdint.h>

#include "rt/x280_hw.h"
#include "x280_hart_protocol.h"

static inline volatile x280_hart_protocol_t* x280_hart_protocol(void) {
    return (volatile x280_hart_protocol_t*)X280_HART_PROTOCOL_BASE;
}

/**
 * The per-hart records. `hartid` is not bounds-checked in these three
 * accessors: a value at or above X280_HART_COUNT returns a pointer outside the
 * protocol block.
 */
static inline volatile x280_hart_control_t* x280_hart_control(uint64_t hartid) {
    return &x280_hart_protocol()->control[hartid];
}

static inline volatile x280_hart_state_t* x280_hart_state(uint64_t hartid) {
    return &x280_hart_protocol()->state[hartid];
}

static inline volatile x280_hart_trap_t* x280_hart_trap(uint64_t hartid) { return &x280_hart_protocol()->trap[hartid]; }

/* This hart's probe state: 0=normal, 1=probe armed, 2=fault occurred. Arming
 * it makes entry.S skip access and illegal-instruction faults on this hart
 * only, so probing an unknown register cannot suppress another hart's real
 * fault report. The host clears it with the rest of the trap record, so an
 * unarmed hart is never left probing. */
static inline volatile uint64_t* x280_hart_probe(uint64_t hartid) { return &x280_hart_trap(hartid)->probe_state; }

static inline void x280_hart_publish_running(uint64_t hartid, uint64_t run_value) {
    volatile x280_hart_state_t* state = x280_hart_state(hartid);
    state->protocol_version = X280_HART_PROTOCOL_VERSION;
    state->run_value = run_value;
    state->heartbeat = X280_HART_ACTIVE_HEARTBEAT(run_value, hartid);
    x280_fence_ow();
    state->phase = X280_HART_PHASE_RUNNING_ACTIVE_FW;
}

/* Has the host asked this hart to finish? The generation must match, so a STOP
 * left over from an earlier session does not end this one. The load fence
 * drains the inbound store buffer; without it a LIM poll can wedge on a stale
 * word even through a volatile. */
static inline int x280_hart_stop_requested(uint64_t hartid, uint64_t run_value) {
    volatile x280_hart_control_t* control = x280_hart_control(hartid);
    x280_fence_ir();
    return control->command == X280_HART_COMMAND_STOP && control->run_value == run_value;
}

/**
 * Spins until the host publishes STOP for `run_value`. Unbounded: there is no
 * timeout and no other exit.
 */
static inline void x280_hart_wait_for_stop(uint64_t hartid, uint64_t run_value) {
    while (!x280_hart_stop_requested(hartid, run_value)) {
    }
}

/* Spin until *word holds `expected`, or until the host's STOP for this
 * generation arrives. Returns 1 if the word was observed and 0 if the wait was
 * abandoned; an abandoning caller must unwind to its return from main rather
 * than proceed as though the word had arrived.
 *
 * For a rendezvous whose counterparty can itself leave on STOP. A bare poll of
 * such a word has one exit and no owner once the counterparty departs, which
 * strands the hart until the next chip reset.
 *
 * The in-loop load fence drains the inbound store buffer; without it a LIM
 * poll can wedge on a stale word even through a volatile. */
static inline int x280_hart_wait_word_or_stop(
    uint64_t hartid, uint64_t run_value, volatile const uint32_t* word, uint32_t expected) {
    for (;;) {
        x280_fence_ir();
        if (*word == expected) {
            return 1;
        }
        if (x280_hart_stop_requested(hartid, run_value)) {
            return 0;
        }
    }
}

/* The argument block the host published for this run, or 0 when the image was
 * enqueued without one. Every hart is told the same address.
 *
 * Opaque: enqueue checked only that the address names a cache-line-aligned
 * block inside the tile's available LIM. Nothing here says the bytes belong to
 * this run, so an image reads its own freshness marker out of the block before
 * acting on the rest of it. */
static inline uint64_t x280_hart_args(uint64_t hartid) {
    x280_fence_ir();
    return x280_hart_control(hartid)->args;
}

/**
 * Host-validated argument extent in bytes for this hart. Zero means no extent
 * was published. Read it before dereferencing x280_hart_args().
 */
static inline uint64_t x280_hart_args_bytes(void) {
    uint64_t hartid;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
    x280_fence_ir();
    return x280_hart_control(hartid)->args_bytes;
}

/* Enqueue cleared the record, so an image that computes nothing publishes zero
 * without calling this. The value lands behind terminal_status, which the
 * harness commits after main returns. */
static inline void x280_hart_set_result(uint64_t hartid, uint64_t result) { x280_hart_state(hartid)->result = result; }

/* Free-running diagnostic stage. Not a commit word: no host predicate reads
 * it, so it needs no fence of its own. */
static inline void x280_hart_set_stage(uint64_t hartid, uint32_t stage) { x280_hart_state(hartid)->stage = stage; }

/* Commit this hart's terminal success. The result is already published; the
 * heartbeat is published before the status word, which is the commit the host
 * reads last. */
static inline void x280_hart_commit_terminal(uint64_t hartid, uint64_t run_value) {
    volatile x280_hart_state_t* state = x280_hart_state(hartid);
    state->heartbeat = X280_HART_ACTIVE_HEARTBEAT(run_value, hartid);
    x280_fence_ow();
    state->terminal_status = X280_HART_STATUS_SUCCESS;
}

/**
 * Publishes RETURNED_TO_IDLE, waits for the host's RETURN_ACK, then enters the
 * resident idle FW with a raw `jr` to X280_IDLE_FW_LOAD_ADDR. The
 * acknowledgement wait is unbounded, like x280_hart_wait_for_stop.
 */
static inline __attribute__((noreturn)) void x280_hart_return_to_idle(uint64_t hartid, uint64_t run_value) {
    volatile x280_hart_control_t* control = x280_hart_control(hartid);
    volatile x280_hart_state_t* state = x280_hart_state(hartid);
    state->run_value = run_value;
    state->heartbeat = X280_HART_ACTIVE_HEARTBEAT(run_value, hartid);
    x280_fence_ow();
    state->phase = X280_HART_PHASE_RETURNED_TO_IDLE;
    for (;;) {
        x280_fence_ir();
        if (control->command == X280_HART_COMMAND_RETURN_ACK && control->run_value == run_value) {
            break;
        }
    }
    __asm__ volatile("jr %0" ::"r"(X280_IDLE_FW_LOAD_ADDR) : "memory");
    __builtin_unreachable();
}

#endif /* X280_RT_IDLE_H */
