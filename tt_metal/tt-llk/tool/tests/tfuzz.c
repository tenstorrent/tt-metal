// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* tfuzz.c -- differential fuzzer: tm_expand vs the ttsim oracle.
 *
 * For thousands of random but VALID plans, expand through tm_expand (the
 * optimiser's own expander) and through ttsim_expand (ported from
 * Tenstorrent's simulator), and assert the two NOP-free traces are
 * identical. ttsim is the ground truth, so a disagreement is a bug in
 * tm_expand, full stop. This is the guard that would have caught the
 * two-ops-per-inner-step bug on its first single-op config instead of
 * never, the way the hand-written tests did.
 *
 * Determinism: a fixed-seed LCG, no clock, no rand(). Same run every
 * time, so a failure is reproducible from the printed iteration. Configs
 * are generated inside the ranges ttsim accepts (outer [1,32], inner
 * [1,64], unpack count [1,32], loop*_last non-NOP), so both expanders
 * succeed and the comparison is about the stream, not error paths.
 */

#include "tharns.h"
#include "ttmop.h"
#include "ttsim_ref.h"

#define FUZZ_ITERS 4000u

/* A fixed-seed linear congruential generator. glibc's constants. */
static uint32_t
lcg(uint32_t *st)
{
    *st = (*st * 1103515245u) + 12345u;
    return *st;
}

/* A random 32-bit instruction word. When allow_nop is false the opcode
 * is forced into a plain-instruction range that is never 0x02 (NOP), so
 * slots that must not be NOP (loop*_last) stay valid. When true, a NOP
 * turns up about one time in eight, to exercise the skip / eat paths. */
static tm_word_t
rword(uint32_t *st, int allow_nop)
{
    uint32_t opc;
    if (allow_nop && (lcg(st) & 7u) == 0u) {
        opc = TM_OP_NOP;                      /* 0x02 */
    } else {
        opc = 0x10u + (lcg(st) % 0x60u);      /* 0x10 .. 0x6F, never 0x02 */
    }
    return (opc << TM_OP_SHIFT) | (lcg(st) & 0x00FFFFFFu);
}

/* One reusable arena, reset between iterations rather than re-malloced. */
static ka_arena_t *
arena(void)
{
    static ka_arena_t A;
    static int        ready = 0;
    if (!ready) {
        ka_init(&A, NULL, 1u << 21, 0);       /* 2 MiB: two traces fit */
        ready = 1;
    }
    ka_rst(&A);
    return &A;
}

/* Compare two expansions; on mismatch print enough to reproduce. */
static int
agree(const char *shape, uint32_t iter, tm_stream_t a, tm_stream_t b)
{
    if (a.ok != b.ok) {
        printf("\n    [%s #%u] ok differs: tm=%d ttsim=%d (tm.err=%s)\n",
               shape, (unsigned)iter, a.ok, b.ok, a.err ? a.err : "-");
        return 0;
    }
    if (a.n != b.n) {
        printf("\n    [%s #%u] length differs: tm=%u ttsim=%u\n",
               shape, (unsigned)iter, (unsigned)a.n, (unsigned)b.n);
        return 0;
    }
    for (uint32_t i = 0u; i < a.n; i++) {
        if (a.words[i] != b.words[i]) {
            printf("\n    [%s #%u] word %u differs: tm=0x%08X ttsim=0x%08X\n",
                   shape, (unsigned)iter, (unsigned)i,
                   (unsigned)a.words[i], (unsigned)b.words[i]);
            return 0;
        }
    }
    return 1;
}

/* ---- Double-loop template ---- */
static void test_fuzz_double(void)
{
    uint32_t st = 0x2545F491u;
    for (uint32_t it = 0u; it < FUZZ_ITERS; it++) {
        ka_arena_t  *A = arena();
        tm_planop_t  plan[2];
        tm_dtmpl_t  *t = &plan[0].dtmpl;
        tm_stream_t  a, b;

        memset(plan, 0, sizeof(plan));
        plan[0].kind = TM_CFG_DOUBLE;
        t->outer_len  = 1u + (lcg(&st) % 32u);    /* [1,32] */
        t->inner_len  = 1u + (lcg(&st) % 64u);    /* [1,64] */
        t->start_op0  = rword(&st, 1);
        t->end_op0    = rword(&st, 1);
        t->end_op1    = rword(&st, 1);
        t->loop_op0   = rword(&st, 1);
        t->loop_op1   = rword(&st, 1);            /* NOP => no alternation */
        t->loop0_last = rword(&st, 0);            /* must be non-NOP */
        t->loop1_last = rword(&st, 0);
        plan[1].kind  = TM_MOP_RUN;

        a = tm_expand(plan, 2, A);
        b = ttsim_expand(plan, 2, A);
        CHECK(agree("double", it, a, b));
    }
}
TH_REG("fuzz", test_fuzz_double)

/* ---- Unpack template ---- */
static void test_fuzz_unpack(void)
{
    uint32_t st = 0x9E3779B9u;
    for (uint32_t it = 0u; it < FUZZ_ITERS; it++) {
        ka_arena_t  *A = arena();
        tm_planop_t  plan[2];
        tm_utmpl_t  *u = &plan[0].utmpl;
        tm_stream_t  a, b;

        memset(plan, 0, sizeof(plan));
        plan[0].kind = TM_CFG_UNPACK;
        u->unpackB = (int)(lcg(&st) & 1u);
        u->halo    = (int)(lcg(&st) & 1u);
        u->A0 = rword(&st, 1); u->A1 = rword(&st, 1);
        u->A2 = rword(&st, 1); u->A3 = rword(&st, 1);
        u->B  = rword(&st, 1);
        u->skipA = rword(&st, 1); u->skipB = rword(&st, 1);

        plan[1].kind      = TM_MOP_RUN;
        plan[1].run_count = 1u + (lcg(&st) % 32u);   /* [1,32] */
        plan[1].run_zmask = lcg(&st);                /* full 32-bit mask */

        a = tm_expand(plan, 2, A);
        b = ttsim_expand(plan, 2, A);
        CHECK(agree("unpack", it, a, b));
    }
}
TH_REG("fuzz", test_fuzz_unpack)

/* ---- Replay load + run, with a mix of exec and slot ranges ---- */
static void test_fuzz_replay(void)
{
    uint32_t st = 0xB5297A4Du;
    for (uint32_t it = 0u; it < FUZZ_ITERS; it++) {
        ka_arena_t  *A = arena();
        tm_planop_t  plan[3];
        tm_word_t    words[TM_REPLAY_SLOTS];
        tm_stream_t  a, b;
        uint32_t     start = lcg(&st) % TM_REPLAY_SLOTS;
        uint32_t     room  = TM_REPLAY_SLOTS - start;
        uint32_t     len   = 1u + (lcg(&st) % room);

        for (uint32_t i = 0u; i < len; i++) {
            words[i] = rword(&st, 1);            /* NOPs allowed: still buffered */
        }

        memset(plan, 0, sizeof(plan));
        plan[0].kind     = TM_REPLAY_LOAD;
        plan[0].rp_start = start;
        plan[0].rp_len   = len;
        plan[0].rp_exec  = (int)(lcg(&st) & 1u);
        plan[0].rp_words = words;
        plan[1].kind     = TM_REPLAY_RUN;
        plan[1].rp_start = start;
        plan[1].rp_len   = len;
        plan[2].kind     = TM_REPLAY_RUN;        /* replay it a second time */
        plan[2].rp_start = start;
        plan[2].rp_len   = len;

        a = tm_expand(plan, 3, A);
        b = ttsim_expand(plan, 3, A);
        CHECK(agree("replay", it, a, b));
    }
}
TH_REG("fuzz", test_fuzz_replay)

/* ---- MOP-emits-REPLAY: the matmul shape. Record a block, then a
 * double-loop whose body is a REPLAY of that block (optionally
 * alternating with a bare instruction). The REPLAY in the MOP body must
 * expand the same in tm_expand and the oracle. ---- */
static void test_fuzz_mop_replay(void)
{
    uint32_t st = 0x27D4EB2Fu;
    for (uint32_t it = 0u; it < FUZZ_ITERS; it++) {
        ka_arena_t  *A = arena();
        tm_planop_t  plan[3];
        tm_dtmpl_t  *t;
        tm_word_t    block[TM_REPLAY_SLOTS];
        tm_stream_t  a, b;
        uint32_t     len = 1u + (lcg(&st) % TM_REPLAY_SLOTS);
        tm_word_t    replay = TM_REPLAY_RUN_WORD(0u, len);

        for (uint32_t i = 0u; i < len; i++) {
            block[i] = rword(&st, 1);
        }

        memset(plan, 0, sizeof(plan));
        plan[0].kind     = TM_REPLAY_LOAD;     /* record the block */
        plan[0].rp_start = 0u;
        plan[0].rp_len   = len;
        plan[0].rp_exec  = 0;
        plan[0].rp_words = block;

        plan[1].kind = TM_CFG_DOUBLE;
        t = &plan[1].dtmpl;
        t->outer_len  = 1u + (lcg(&st) % 8u);
        t->inner_len  = 1u + (lcg(&st) % 8u);
        t->start_op0  = TM_NOP_WORD;
        t->end_op0    = TM_NOP_WORD;
        t->end_op1    = TM_NOP_WORD;
        t->loop_op0   = replay;
        /* Half the time alternate the REPLAY with a bare instruction. */
        t->loop_op1   = (lcg(&st) & 1u) ? rword(&st, 0) : TM_NOP_WORD;
        t->loop0_last = replay;                /* non-NOP (a REPLAY word) */
        t->loop1_last = replay;
        plan[2].kind  = TM_MOP_RUN;

        a = tm_expand(plan, 3, A);
        b = ttsim_expand(plan, 3, A);
        CHECK(agree("mop_replay", it, a, b));
    }
}
TH_REG("fuzz", test_fuzz_mop_replay)

/* ---- Mixed sequences: direct emits around a double-loop and a replay,
 * exercising that template/buffer state carries across plan ops the same
 * way in both expanders. ---- */
static void test_fuzz_mixed(void)
{
    uint32_t st = 0xC2B2AE35u;
    for (uint32_t it = 0u; it < FUZZ_ITERS; it++) {
        ka_arena_t  *A = arena();
        tm_planop_t  plan[6];
        tm_word_t    words[8];
        tm_dtmpl_t  *t;
        tm_stream_t  a, b;
        uint32_t     len = 1u + (lcg(&st) % 8u);

        for (uint32_t i = 0u; i < len; i++) {
            words[i] = rword(&st, 1);
        }

        memset(plan, 0, sizeof(plan));
        plan[0].kind = TM_EMIT; plan[0].word = rword(&st, 1);

        plan[1].kind = TM_CFG_DOUBLE;
        t = &plan[1].dtmpl;
        t->outer_len  = 1u + (lcg(&st) % 8u);
        t->inner_len  = 1u + (lcg(&st) % 16u);
        t->start_op0  = rword(&st, 1);
        t->end_op0    = rword(&st, 1);
        t->end_op1    = rword(&st, 1);
        t->loop_op0   = rword(&st, 1);
        t->loop_op1   = rword(&st, 1);
        t->loop0_last = rword(&st, 0);
        t->loop1_last = rword(&st, 0);
        plan[2].kind = TM_MOP_RUN;

        plan[3].kind     = TM_REPLAY_LOAD;
        plan[3].rp_start = 0u;
        plan[3].rp_len   = len;
        plan[3].rp_exec  = (int)(lcg(&st) & 1u);
        plan[3].rp_words = words;
        plan[4].kind     = TM_REPLAY_RUN;
        plan[4].rp_start = 0u;
        plan[4].rp_len   = len;

        plan[5].kind = TM_EMIT; plan[5].word = rword(&st, 1);

        a = tm_expand(plan, 6, A);
        b = ttsim_expand(plan, 6, A);
        CHECK(agree("mixed", it, a, b));
    }
}
TH_REG("fuzz", test_fuzz_mixed)
