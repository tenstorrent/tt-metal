// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* ttmop_expand.c -- the expander and the verifier.
 *
 * The load-bearing wall. Everything else in the optimiser is allowed to
 * be clever and occasionally wrong; this is the bit that has to be
 * neither. It plays a plan forward exactly as the silicon would, MOP
 * loops unrolled, replay buffer loaded and parroted back, and hands you
 * the Tensix stream that plan actually stands for. The verifier then
 * marches that stream up against the target and, the second they
 * disagree, points at the precise word like a customs officer who has
 * found the one undeclared apple.
 *
 * No globals, not one. The replay buffer and template registers live on
 * the stack inside tm_expand, so two expansions run back to back without
 * leaving fingerprints on each other. You can score a thousand candidate
 * plans and none of them will haunt the next.
 */

#include "ttmop.h"

int
tm_is_nop(tm_word_t w)
{
    return (w >> TM_OP_SHIFT) == TM_OP_NOP;
}

/* The expander's whole state: the output trace plus the replay buffer
 * and its load-in-flight registers. Bundling them lets the MOP expansion
 * route its emitted words through the same replay machinery the frontend
 * uses, so a MOP body that issues a REPLAY expands it, and a MOP issued
 * mid-load is captured. All stack-local inside tm_expand, so two
 * expansions never see each other.
 *
 * The trace is bounded: once it hits TM_MAX_WORDS it stops accepting
 * words and trips the error flag rather than running off the arena. */
typedef struct {
    tm_word_t  *buf;       /* output trace                              */
    uint32_t    n;
    uint32_t    cap;
    int         ok;
    const char *err;

    tm_word_t   rbuf[TM_REPLAY_SLOTS];   /* replay buffer               */
    uint32_t    rp_index;                /* next slot to capture into   */
    uint32_t    rp_left;                 /* words still to capture (>0 = loading) */
    int         rp_exec;                 /* execute-while-loading        */
} tm_exp_t;

/* Append one word to the backend bus trace. The Tensix frontend swallows
 * every NOP before the backend sees it (tensix_push_inst_fifo in ttsim
 * eats opcode 0x02 unconditionally), so the trace we verify against is
 * NOP-free, whatever the source: direct issue, replay, or a MOP body. */
static void
out_put(tm_exp_t *e, tm_word_t w)
{
    if (!e->ok) {
        return;
    }
    if (tm_is_nop(w)) {
        return;
    }
    if (e->n >= e->cap) {
        e->ok  = 0;
        e->err = "expansion exceeded TM_MAX_WORDS";
        return;
    }
    e->buf[e->n] = w;
    e->n++;
}

/* Issue one word into the frontend, mirroring ttsim's replay_expander.
 * Three cases, in order: while a replay load is in flight the word is
 * captured into the buffer (and emitted too if execute-while-loading is
 * set); otherwise a REPLAY instruction word is decoded and acts on the
 * buffer (load sets up the next capture window, run re-emits a range);
 * anything else just issues to the trace. Every emit in the expander
 * goes through here, so MOP bodies, replays, and direct words all
 * compose the way the hardware composes them. */
static void
tm_issue(tm_exp_t *e, tm_word_t w)
{
    if (!e->ok) {
        return;
    }

    if (e->rp_left > 0u) {
        if (e->rp_index < TM_REPLAY_SLOTS) {
            e->rbuf[e->rp_index] = w;
        }
        e->rp_index++;
        e->rp_left--;
        if (e->rp_exec) {
            out_put(e, w);
        }
        return;
    }

    if ((w >> TM_OP_SHIFT) == TM_OP_REPLAY) {
        uint32_t load = w & 1u;
        uint32_t exec = (w >> 1) & 1u;
        uint32_t len  = (w >> 4) & 0x3FFu;
        uint32_t strt = (w >> 14) & 0x3FFu;

        if (strt >= TM_REPLAY_SLOTS || len > TM_REPLAY_SLOTS ||
            strt + len > TM_REPLAY_SLOTS) {
            e->ok  = 0;
            e->err = "REPLAY word out of buffer range";
            return;
        }
        if (load) {
            e->rp_index = strt;
            e->rp_left  = len;
            e->rp_exec  = (int)exec;
        } else {
            for (uint32_t i = 0u; i < len; i++) {
                out_put(e, e->rbuf[strt + i]);
            }
        }
        return;
    }

    out_put(e, w);
}

/* Expand the double-loop template, matching the Tensix MOP expander
 * exactly (tt-isa-documentation MOPExpander.md ExpandTemplate1, and
 * ttsim src/tensix.cpp mop_expander, which agree word for word).
 *
 * The body emits ONE op per inner step, not two. loop_op1 is not a
 * second emitted slot: a non-NOP loop_op1 turns the body into an
 * alternation (the single loop op XOR-flips loop_op0 <-> loop_op1 each
 * step) and DOUBLES the inner count. The last inner step is replaced,
 * not appended: loop1_last on the last inner of an earlier outer
 * iteration, loop0_last on the last inner of the last outer iteration.
 * loop_op carries across outer iterations; it is not reset.
 *
 * start_op / end_op0 / end_op1 are skipped entirely when they are NOPs;
 * end_op1 only fires if end_op0 did. This is documented expander
 * behaviour, not a hardware assumption.
 */
static void
expand_double(tm_exp_t *e, const tm_dtmpl_t *t)
{
    uint32_t  outer = t->outer_len;
    uint32_t  inner = t->inner_len;
    tm_word_t loop_op = t->loop_op0;
    tm_word_t flip = 0u;

    /* Supported ranges, straight from ttsim's verifies (anything outside
     * is UnimplementedFunctionality on the sim, so the optimiser must
     * never propose it). outer in [1,32], inner in [1,64] before the
     * loop_op1 doubling. */
    if (outer < 1u || outer > 32u) {
        e->ok  = 0;
        e->err = "double-loop outer_len outside [1,32]";
        return;
    }
    if (inner < 1u || inner > 64u) {
        e->ok  = 0;
        e->err = "double-loop inner_len outside [1,64]";
        return;
    }
    /* The last-step ops are emitted on every inner loop and must be real
     * instructions; ttsim rejects NOPs here. */
    if (tm_is_nop(t->loop0_last) || tm_is_nop(t->loop1_last)) {
        e->ok  = 0;
        e->err = "double-loop loop0_last/loop1_last must not be NOP";
        return;
    }

    /* A non-NOP loop_op1 alternates the body op and doubles the count. */
    if (!tm_is_nop(t->loop_op1)) {
        flip   = t->loop_op0 ^ t->loop_op1;
        inner *= 2u;
    }

    for (uint32_t oo = 0u; oo < outer; oo++) {
        int last_outer = (oo == outer - 1u);

        if (!tm_is_nop(t->start_op0)) {
            tm_issue(e, t->start_op0);
        }

        for (uint32_t ii = 0u; ii < inner; ii++) {
            if (ii != inner - 1u) {
                tm_issue(e, loop_op);            /* one op per step      */
            } else if (!last_outer) {
                tm_issue(e, t->loop1_last);      /* last inner, earlier outer */
            } else {
                tm_issue(e, t->loop0_last);      /* last inner, last outer    */
            }
            loop_op ^= flip;                     /* alternate            */
            if (!e->ok) {
                return;
            }
        }

        if (!tm_is_nop(t->end_op0)) {
            tm_issue(e, t->end_op0);
            if (!tm_is_nop(t->end_op1)) {
                tm_issue(e, t->end_op1);
            }
        }
        if (!e->ok) {
            return;
        }
    }
}

/* Expand the single-loop unpack template. zmask polarity follows
 * ASSUMPTION 2. count is the number of iterations; the caller has
 * already turned the hardware's "count - 1" back into a real count. */
static void
expand_unpack(tm_exp_t *e, const tm_utmpl_t *t, uint32_t count, uint32_t zmask)
{
    if (count == 0u) {
        return;
    }
    if (count > 32u) {
        e->ok  = 0;
        e->err = "unpack count exceeds 32 (zmask is 32 bits)";
        return;
    }

    for (uint32_t i = 0u; i < count; i++) {
        uint32_t bit = (zmask >> i) & 1u;
#if TM_ZMASK_BIT_EXECUTES
        int execute = (bit != 0u);
#else
        int execute = (bit == 0u);
#endif
        if (execute) {
            tm_issue(e, t->A0);
            if (t->halo) {
                tm_issue(e, t->A1);
                tm_issue(e, t->A2);
                tm_issue(e, t->A3);
            }
            if (t->unpackB) {
                tm_issue(e, t->B);
            }
        } else {
            tm_issue(e, t->skipA);
            if (t->unpackB) {
                tm_issue(e, t->skipB);
            }
        }
        if (!e->ok) {
            return;
        }
    }
}

tm_stream_t
tm_expand(const tm_planop_t *plan, uint32_t n_ops, ka_arena_t *A)
{
    tm_stream_t result;
    tm_exp_t    e;

    /* Template registers, stack-local. The "current template" is whichever
     * CFG op programmed it last, exactly the hardware's program-then-run
     * split. */
    tm_dtmpl_t  cur_d = {0};
    tm_utmpl_t  cur_u = {0};
    int         cur_is_double = -1;   /* -1 none, 0 unpack, 1 double */

    result.words = NULL;
    result.n     = 0u;
    result.ok    = 0;
    result.err   = NULL;

    if (plan == NULL || A == NULL) {
        result.err = "null plan or arena";
        return result;
    }

    e.buf = KA_NEWN(A, tm_word_t, TM_MAX_WORDS);
    if (e.buf == NULL) {
        result.err = "arena could not hold TM_MAX_WORDS";
        return result;
    }
    e.n   = 0u;
    e.cap = TM_MAX_WORDS;
    e.ok  = 1;
    e.err = NULL;
    e.rp_index = 0u;
    e.rp_left  = 0u;
    e.rp_exec  = 0;

    /* Zero the replay buffer so a replay of an unloaded slot is a
     * predictable 0 rather than stack garbage. A plan that replays before
     * loading is a bug we want to be reproducible. */
    for (uint32_t i = 0u; i < TM_REPLAY_SLOTS; i++) {
        e.rbuf[i] = 0u;
    }

    for (uint32_t k = 0u; k < n_ops && e.ok; k++) {
        const tm_planop_t *op = &plan[k];

        switch (op->kind) {
        case TM_EMIT:
            /* Through tm_issue, so a directly-issued REPLAY word expands
             * just as the frontend would expand it. */
            tm_issue(&e, op->word);
            break;

        case TM_CFG_DOUBLE:
            cur_d         = op->dtmpl;
            cur_is_double = 1;
            break;

        case TM_CFG_UNPACK:
            cur_u         = op->utmpl;
            cur_is_double = 0;
            break;

        case TM_MOP_RUN:
            if (cur_is_double == 1) {
                expand_double(&e, &cur_d);
            } else if (cur_is_double == 0) {
                expand_unpack(&e, &cur_u, op->run_count, op->run_zmask);
            } else {
                e.ok  = 0;
                e.err = "MOP_RUN before any template was programmed";
            }
            break;

        case TM_REPLAY_LOAD:
            if (op->rp_start >= TM_REPLAY_SLOTS ||
                op->rp_len   >  TM_REPLAY_SLOTS ||
                op->rp_start + op->rp_len > TM_REPLAY_SLOTS) {
                e.ok  = 0;
                e.err = "replay load out of buffer range";
                break;
            }
            if (op->rp_len > 0u && op->rp_words == NULL) {
                e.ok  = 0;
                e.err = "replay load with len > 0 but no words";
                break;
            }
            /* Set up the capture window and feed the words past it, the
             * way a REPLAY-load word followed by its instructions would.
             * tm_issue captures each (and emits it too if exec is set). */
            e.rp_index = op->rp_start;
            e.rp_left  = op->rp_len;
            e.rp_exec  = op->rp_exec;
            for (uint32_t j = 0u; j < op->rp_len; j++) {
                tm_issue(&e, op->rp_words[j]);
            }
            break;

        case TM_REPLAY_RUN:
            if (op->rp_start >= TM_REPLAY_SLOTS ||
                op->rp_len   >  TM_REPLAY_SLOTS ||
                op->rp_start + op->rp_len > TM_REPLAY_SLOTS) {
                e.ok  = 0;
                e.err = "replay run out of buffer range";
                break;
            }
            for (uint32_t j = 0u; j < op->rp_len; j++) {
                out_put(&e, e.rbuf[op->rp_start + j]);
            }
            break;

        default:
            e.ok  = 0;
            e.err = "unknown plan op kind";
            break;
        }
    }

    result.words = e.buf;
    result.n     = e.n;
    result.ok    = e.ok;
    result.err   = e.err;
    return result;
}

tm_verdict_t
tm_verify(const tm_planop_t *plan, uint32_t n_ops,
          const tm_word_t *target, uint32_t n_target,
          ka_arena_t *A)
{
    tm_verdict_t v;
    tm_stream_t  s;
    uint32_t     lim;

    v.ok         = 0;
    v.n_expanded = 0u;
    v.n_target   = n_target;
    v.diff_at    = 0u;
    v.exp_word   = 0u;
    v.got_word   = 0u;
    v.msg        = NULL;

    s = tm_expand(plan, n_ops, A);
    v.n_expanded = s.n;

    if (!s.ok) {
        v.msg = s.err ? s.err : "expansion failed";
        return v;
    }

    /* Compare the common prefix first, so a mismatch points at the
     * exact word that diverged, which is far more useful to a search
     * pass than "lengths differ". */
    lim = (s.n < n_target) ? s.n : n_target;
    for (uint32_t i = 0u; i < lim; i++) {
        if (s.words[i] != target[i]) {
            v.diff_at  = i;
            v.exp_word = target[i];
            v.got_word = s.words[i];
            v.msg      = "word mismatch";
            return v;
        }
    }

    if (s.n != n_target) {
        v.diff_at = lim;
        v.msg     = (s.n < n_target) ? "expansion shorter than target"
                                     : "expansion longer than target";
        return v;
    }

    v.ok  = 1;
    v.msg = "match";
    return v;
}
