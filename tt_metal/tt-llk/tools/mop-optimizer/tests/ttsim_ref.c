// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* ttsim_ref.c -- the expander, ported from ttsim as an oracle.
 *
 * Faithful to ttsim src/tensix.cpp (Apache 2.0, (c) Tenstorrent). The
 * names mirror the original: push_fifo == tensix_push_inst_fifo (eats
 * every NOP), rx == replay_expander (captures into the replay buffer if
 * a load is in flight, expands a REPLAY instruction word against the
 * buffer, else issues it), mop_t0 / mop_t1 == the two arms of
 * mop_expander. The plan walker
 * drives them, translating each tm_planop_t into the ttsim primitive it
 * stands for, so a MOP body and a REPLAY compose exactly as they would
 * in the real frontend.
 *
 * This file knows nothing about tm_expand's internals on purpose. Two
 * independent roads to the same stream; the fuzzer checks they meet.
 */

#include "ttsim_ref.h"

#define RB_SLOTS TM_REPLAY_SLOTS

typedef struct {
    /* Output trace (NOP-free). */
    tm_word_t  *buf;
    uint32_t    n;
    uint32_t    cap;
    int         ok;
    const char *err;

    /* MOP expander config, ttsim's mop_cfg[9] layout, plus the high
     * half of the unpack zmask that a MOP_CFG would have latched. */
    tm_word_t   cfg[9];
    uint32_t    zmask_hi16;

    /* Replay buffer state. */
    tm_word_t   rbuf[RB_SLOTS];
    uint32_t    rp_index;
    uint32_t    rp_left;
    int         rp_exec;
} st_t;

/* tensix_push_inst_fifo: the frontend swallows every NOP, so nothing
 * downstream ever sees one. */
static void
push_fifo(st_t *s, tm_word_t inst)
{
    if (!s->ok) {
        return;
    }
    if (tm_is_nop(inst)) {
        return;
    }
    if (s->n >= s->cap) {
        s->ok  = 0;
        s->err = "ttsim_ref: trace exceeded TM_MAX_WORDS";
        return;
    }
    s->buf[s->n] = inst;
    s->n++;
}

/* replay_expander, for a plain (non-REPLAY) instruction: while a load is
 * in flight the instruction is captured into the buffer (and issued too
 * if execute-while-loading is set); otherwise it just issues. */
static void
rx(st_t *s, tm_word_t inst)
{
    if (!s->ok) {
        return;
    }
    if (s->rp_left > 0u) {
        if (s->rp_index < RB_SLOTS) {
            s->rbuf[s->rp_index] = inst;
        }
        s->rp_index++;
        s->rp_left--;
        if (s->rp_exec) {
            push_fifo(s, inst);
        }
        return;
    }
    /* A REPLAY instruction word acts on the buffer, as in ttsim's
     * replay_expander: load arms the next capture window, run re-emits a
     * range. This is what makes a MOP body of REPLAY words (the matmul's
     * shape) expand correctly. */
    if ((inst >> 24) == 0x04u) {
        uint32_t load = inst & 1u;
        uint32_t exec = (inst >> 1) & 1u;
        uint32_t len  = (inst >> 4) & 0x3FFu;
        uint32_t strt = (inst >> 14) & 0x3FFu;
        if (strt >= RB_SLOTS || len > RB_SLOTS || strt + len > RB_SLOTS) {
            s->ok  = 0;
            s->err = "ttsim_ref: REPLAY word out of buffer range";
            return;
        }
        if (load) {
            s->rp_index = strt;
            s->rp_left  = len;
            s->rp_exec  = (int)exec;
        } else {
            for (uint32_t i = 0u; i < len; i++) {
                push_fifo(s, s->rbuf[strt + i]);
            }
        }
        return;
    }
    push_fifo(s, inst);
}

/* The REPLAY load and run arms, driven straight from the plan IR rather
 * than from an encoded REPLAY word (tm_expand's IR is abstract too). A
 * load sets up the capture window and then feeds the LEN words past it;
 * a run re-issues the captured range. */
static void
replay_load(st_t *s, uint32_t start, uint32_t len, int exec,
            const tm_word_t *words)
{
    s->rp_index = start;
    s->rp_left  = len;
    s->rp_exec  = exec;
    for (uint32_t j = 0u; j < len; j++) {
        rx(s, words ? words[j] : 0u);
    }
}

static void
replay_run(st_t *s, uint32_t start, uint32_t len)
{
    for (uint32_t i = 0u; i < len; i++) {
        push_fifo(s, s->rbuf[(start + i) % RB_SLOTS]);
    }
}

/* mop_expander, template 1: the double loop. One op per inner step;
 * loop_op1 (cfg[6]) alternates the body op and doubles the inner count;
 * the last step is replaced by loop1_last/loop0_last; start/end skip on
 * NOP. Every emit goes through rx so a MOP inside a replay load is
 * captured, exactly as ttsim does it. */
static void
mop_t1(st_t *s)
{
    uint32_t  outer   = s->cfg[0];
    uint32_t  inner   = s->cfg[1];
    tm_word_t start   = s->cfg[2];
    tm_word_t end0    = s->cfg[3];
    tm_word_t end1    = s->cfg[4];
    tm_word_t loop_op = s->cfg[5];
    tm_word_t loop_op1 = s->cfg[6];
    tm_word_t l0last  = s->cfg[7];
    tm_word_t l1last  = s->cfg[8];
    tm_word_t flip    = 0u;

    if (!tm_is_nop(loop_op1)) {
        flip   = loop_op ^ loop_op1;
        inner *= 2u;
    }

    for (uint32_t i = 0u; i < outer && s->ok; i++) {
        if (!tm_is_nop(start)) {
            rx(s, start);
        }
        for (uint32_t j = 0u; j < inner && s->ok; j++) {
            if (j + 1u < inner) {
                rx(s, loop_op);
            } else if (i + 1u < outer) {
                rx(s, l1last);
            } else {
                rx(s, l0last);
            }
            loop_op ^= flip;
        }
        if (!tm_is_nop(end0)) {
            rx(s, end0);
            if (!tm_is_nop(end1)) {
                rx(s, end1);
            }
        }
    }
}

/* mop_expander, template 0: the unpack loop. zmask bit per iteration
 * selects the full unpack (A0[/A1A2A3][/B]) or the skip (skipA[/skipB]).
 * cfg layout: [1]=flags (HasB, HasA123), [2]=B, [3]=A0, [4..6]=A1A2A3,
 * [7]=skipA, [8]=skipB. A clear bit executes (matching tt-mop's
 * confirmed zmask polarity). */
static void
mop_t0(st_t *s, uint32_t count, uint32_t zmask_lo)
{
    uint32_t flags = s->cfg[1];
    uint32_t zmask = zmask_lo | (s->zmask_hi16 << 16);

    for (uint32_t i = 0u; i < count && s->ok; i++) {
        if (zmask & 1u) {
            rx(s, s->cfg[7]);                 /* skipA */
            if (flags & 1u) {
                rx(s, s->cfg[8]);             /* skipB */
            }
        } else {
            rx(s, s->cfg[3]);                 /* A0 */
            if (flags & 2u) {
                rx(s, s->cfg[4]);             /* A1 */
                rx(s, s->cfg[5]);             /* A2 */
                rx(s, s->cfg[6]);             /* A3 */
            }
            if (flags & 1u) {
                rx(s, s->cfg[2]);             /* B */
            }
        }
        zmask >>= 1;
    }
}

tm_stream_t
ttsim_expand(const tm_planop_t *plan, uint32_t n_ops, ka_arena_t *A)
{
    tm_stream_t result;
    st_t        s;
    int         cur = -1;          /* -1 none, 0 unpack, 1 double */

    result.words = NULL;
    result.n     = 0u;
    result.ok    = 0;
    result.err   = NULL;

    if (plan == NULL || A == NULL) {
        result.err = "ttsim_ref: null plan or arena";
        return result;
    }

    s.buf = KA_NEWN(A, tm_word_t, TM_MAX_WORDS);
    if (s.buf == NULL) {
        result.err = "ttsim_ref: arena could not hold TM_MAX_WORDS";
        return result;
    }
    s.n = 0u; s.cap = TM_MAX_WORDS; s.ok = 1; s.err = NULL;
    s.zmask_hi16 = 0u;
    s.rp_index = 0u; s.rp_left = 0u; s.rp_exec = 0;
    for (uint32_t i = 0u; i < 9u; i++) {
        s.cfg[i] = 0u;
    }
    for (uint32_t i = 0u; i < RB_SLOTS; i++) {
        s.rbuf[i] = 0u;
    }

    for (uint32_t k = 0u; k < n_ops && s.ok; k++) {
        const tm_planop_t *op = &plan[k];

        switch (op->kind) {
        case TM_EMIT:
            rx(&s, op->word);          /* passthrough, capturable */
            break;

        case TM_CFG_DOUBLE:
            s.cfg[0] = op->dtmpl.outer_len;
            s.cfg[1] = op->dtmpl.inner_len;
            s.cfg[2] = op->dtmpl.start_op0;
            s.cfg[3] = op->dtmpl.end_op0;
            s.cfg[4] = op->dtmpl.end_op1;
            s.cfg[5] = op->dtmpl.loop_op0;
            s.cfg[6] = op->dtmpl.loop_op1;
            s.cfg[7] = op->dtmpl.loop0_last;
            s.cfg[8] = op->dtmpl.loop1_last;
            cur = 1;
            break;

        case TM_CFG_UNPACK:
            s.cfg[1] = (op->utmpl.unpackB ? 1u : 0u)
                     | (op->utmpl.halo    ? 2u : 0u);
            s.cfg[2] = op->utmpl.B;
            s.cfg[3] = op->utmpl.A0;
            s.cfg[4] = op->utmpl.A1;
            s.cfg[5] = op->utmpl.A2;
            s.cfg[6] = op->utmpl.A3;
            s.cfg[7] = op->utmpl.skipA;
            s.cfg[8] = op->utmpl.skipB;
            s.zmask_hi16 = 0u;
            cur = 0;
            break;

        case TM_MOP_RUN:
            if (cur == 1) {
                mop_t1(&s);
            } else if (cur == 0) {
                s.zmask_hi16 = (op->run_zmask >> 16) & 0xFFFFu;
                mop_t0(&s, op->run_count, op->run_zmask & 0xFFFFu);
            } else {
                s.ok  = 0;
                s.err = "ttsim_ref: MOP_RUN before any template";
            }
            break;

        case TM_REPLAY_LOAD:
            replay_load(&s, op->rp_start, op->rp_len, op->rp_exec,
                        op->rp_words);
            break;

        case TM_REPLAY_RUN:
            replay_run(&s, op->rp_start, op->rp_len);
            break;

        default:
            s.ok  = 0;
            s.err = "ttsim_ref: unknown plan op kind";
            break;
        }
    }

    result.words = s.buf;
    result.n     = s.n;
    result.ok    = s.ok;
    result.err   = s.err;
    return result;
}
