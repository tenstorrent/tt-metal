// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* texpand.c -- the verifier's own verification.
 *
 * Everything in the optimiser takes tm_expand on faith, so somebody had
 * better not. These tests are the foundation under the foundation. Each
 * one hand-builds a plan, writes down the exact Tensix stream it must
 * produce, and checks the expander agrees to the word. Let a single bit
 * drift here and every plan the optimiser ever scores is marked against
 * a lie, confidently, forever. */

#include "tharns.h"
#include "ttmop.h"

/* A fresh megabyte arena per test. The expander wants room for up to
 * TM_MAX_WORDS (256 KB) plus slack, and a clean arena per test keeps
 * them from leaning on each other. */
static ka_arena_t *fresh(void)
{
    static ka_arena_t A;
    ka_init(&A, NULL, 1u << 20, 0);
    return &A;
}

/* Verify a plan reproduces an expected stream, word for word. */
static int matches(const tm_planop_t *plan, uint32_t n_ops,
                   const tm_word_t *want, uint32_t n_want)
{
    ka_arena_t  *A = fresh();
    tm_verdict_t v = tm_verify(plan, n_ops, want, n_want, A);
    int ok = v.ok;
    if (!ok) {
        printf("    verdict: %s  exp=0x%08X got=0x%08X at=%u (got %u, want %u)\n",
               v.msg ? v.msg : "?", (unsigned)v.exp_word, (unsigned)v.got_word,
               (unsigned)v.diff_at, (unsigned)v.n_expanded, (unsigned)v.n_target);
    }
    ka_free(A);
    return ok;
}

/* ---- NOP predicate ---- */
static void test_is_nop(void)
{
    CHECK(tm_is_nop(TM_NOP_WORD));
    CHECK(!tm_is_nop(0x01000000u));   /* MOP    */
    CHECK(!tm_is_nop(0x04000000u));   /* REPLAY */
}
TH_REG("expand", test_is_nop)

/* ---- NOP-free trace: the frontend swallows every NOP before the
 * backend, so a NOP never appears in the stream we verify against, even
 * a directly-issued one. X, NOP, Y collapses to X, Y. ---- */
static void test_nop_free_trace(void)
{
    tm_planop_t plan[3] = {0};
    tm_word_t   want[2] = { 0x12345678u, 0x00000001u };
    plan[0].kind = TM_EMIT; plan[0].word = 0x12345678u;
    plan[1].kind = TM_EMIT; plan[1].word = TM_NOP_WORD;   /* eaten */
    plan[2].kind = TM_EMIT; plan[2].word = 0x00000001u;
    CHECK(matches(plan, 3, want, 2));
}
TH_REG("expand", test_nop_free_trace)

/* ---- Direct emit is a pass-through ---- */
static void test_emit_passthrough(void)
{
    tm_planop_t plan[3] = {0};
    tm_word_t   want[3] = { 0x12345678u, 0xCAFEBABEu, 0x00000001u };
    plan[0].kind = TM_EMIT; plan[0].word = want[0];
    plan[1].kind = TM_EMIT; plan[1].word = want[1];
    plan[2].kind = TM_EMIT; plan[2].word = want[2];
    CHECK(matches(plan, 3, want, 3));
}
TH_REG("expand", test_emit_passthrough)

/* ---- Double-loop template, including the substitution ladder ----
 * outer=2, inner=2 exercises both replacements: loop1_last on the last
 * inner of the first (non-last) outer iteration, and loop0_last on the
 * last inner of the last outer iteration. All slot values are distinct
 * non-NOP words, so this test does not depend on ASSUMPTION 1. */
static void test_double_loop(void)
{
    tm_planop_t plan[2] = {0};
    tm_dtmpl_t *t = &plan[0].dtmpl;

    plan[0].kind = TM_CFG_DOUBLE;
    t->outer_len  = 2u;
    t->inner_len  = 2u;
    t->start_op0  = 0xAA000001u;
    t->loop_op0   = 0xBB000002u;
    t->loop_op1   = 0xCC000003u;
    t->loop1_last = 0xDD000004u;
    t->loop0_last = 0xEE000005u;
    t->end_op0    = 0xFF000006u;
    t->end_op1    = 0x11000007u;

    plan[1].kind = TM_MOP_RUN;

    {
        tm_word_t want[14] = {
            0xAA000001u, 0xBB000002u, 0xCC000003u, 0xBB000002u, 0xDD000004u,
            0xFF000006u, 0x11000007u,
            0xAA000001u, 0xBB000002u, 0xCC000003u, 0xBB000002u, 0xEE000005u,
            0xFF000006u, 0x11000007u
        };
        CHECK(matches(plan, 2, want, 14));
    }
}
TH_REG("expand", test_double_loop)

/* ---- Single inner op: a clean repeat with no second op. loop_op1 is
 * NOP (no alternation, no doubling), start/end are NOP and so skipped,
 * and loop0_last carries the op on the final step. outer=1, inner=3
 * emits the op three times, nothing more. This is the case the old
 * two-ops-per-iter model got wrong (it appended a fourth op). ---- */
static void test_double_single_op(void)
{
    tm_planop_t plan[2] = {0};
    tm_dtmpl_t *t = &plan[0].dtmpl;

    plan[0].kind  = TM_CFG_DOUBLE;
    t->outer_len  = 1u;
    t->inner_len  = 3u;
    t->start_op0  = TM_NOP_WORD;     /* skipped */
    t->end_op0    = TM_NOP_WORD;     /* skipped */
    t->end_op1    = TM_NOP_WORD;     /* skipped */
    t->loop_op0   = 0xBB000002u;
    t->loop_op1   = TM_NOP_WORD;     /* no alternation */
    t->loop1_last = 0xBB000002u;     /* must be non-NOP; equals the op */
    t->loop0_last = 0xBB000002u;     /* last step replaces, with the op */
    plan[1].kind  = TM_MOP_RUN;

    {
        tm_word_t want[3] = { 0xBB000002u, 0xBB000002u, 0xBB000002u };
        CHECK(matches(plan, 2, want, 3));
    }
}
TH_REG("expand", test_double_single_op)

/* ---- The last inner step REPLACES the body op, it does not append.
 * outer=1, inner=3, distinct loop0_last. Hardware: X X Y (three words).
 * The old model emitted X X X Y; this test fails under it. ---- */
static void test_double_replace_last(void)
{
    tm_planop_t plan[2] = {0};
    tm_dtmpl_t *t = &plan[0].dtmpl;

    plan[0].kind  = TM_CFG_DOUBLE;
    t->outer_len  = 1u;
    t->inner_len  = 3u;
    t->start_op0  = TM_NOP_WORD;
    t->end_op0    = TM_NOP_WORD;
    t->end_op1    = TM_NOP_WORD;
    t->loop_op0   = 0xBB000002u;     /* X */
    t->loop_op1   = TM_NOP_WORD;
    t->loop1_last = 0xCC000003u;     /* non-NOP, unused here (single outer) */
    t->loop0_last = 0xEE000005u;     /* Y, replaces X on the last step */
    plan[1].kind  = TM_MOP_RUN;

    {
        tm_word_t want[3] = { 0xBB000002u, 0xBB000002u, 0xEE000005u };
        CHECK(matches(plan, 2, want, 3));
    }
}
TH_REG("expand", test_double_replace_last)

/* ---- Alternation + doubling: a non-NOP loop_op1 makes the body op
 * flip X<->Z each step and doubles the inner count. outer=1, inner=2
 * becomes four steps X Z X, with the last replaced by loop0_last=Y.
 * Hardware: X Z X Y. ---- */
static void test_double_alternate(void)
{
    tm_planop_t plan[2] = {0};
    tm_dtmpl_t *t = &plan[0].dtmpl;

    plan[0].kind  = TM_CFG_DOUBLE;
    t->outer_len  = 1u;
    t->inner_len  = 2u;              /* doubles to 4 */
    t->start_op0  = TM_NOP_WORD;
    t->end_op0    = TM_NOP_WORD;
    t->end_op1    = TM_NOP_WORD;
    t->loop_op0   = 0xBB000002u;     /* X */
    t->loop_op1   = 0xCC000003u;     /* Z, non-NOP => alternate + double */
    t->loop1_last = 0xDD000004u;     /* non-NOP, unused (single outer) */
    t->loop0_last = 0xEE000005u;     /* Y, replaces the last step */
    plan[1].kind  = TM_MOP_RUN;

    {
        tm_word_t want[4] = {
            0xBB000002u, 0xCC000003u, 0xBB000002u, 0xEE000005u
        };
        CHECK(matches(plan, 2, want, 4));
    }
}
TH_REG("expand", test_double_alternate)

/* ---- Unpack template, zmask selection (confirmed polarity) ----
 * A SET zmask bit SKIPS that iteration, a CLEAR bit executes. zmask
 * 0b1010 over 4 iterations therefore goes execute, skip, execute, skip.
 * (Confirmed against tt-llk; see ASSUMPTION 2 in ttmop.h.) ---- */
static void test_unpack_zmask(void)
{
    tm_planop_t plan[2] = {0};
    tm_utmpl_t *u = &plan[0].utmpl;

    plan[0].kind = TM_CFG_UNPACK;
    u->unpackB = 0;
    u->halo    = 0;
    u->A0      = 0xA0000000u;
    u->skipA   = 0x5A000000u;

    plan[1].kind      = TM_MOP_RUN;
    plan[1].run_count = 4u;
    plan[1].run_zmask = 0xAu;   /* 0b1010 */

    {
        tm_word_t want[4] = {
            0xA0000000u,   /* i=0 bit0=0 -> execute */
            0x5A000000u,   /* i=1 bit1=1 -> skip    */
            0xA0000000u,   /* i=2 bit2=0 -> execute */
            0x5A000000u    /* i=3 bit3=1 -> skip    */
        };
        CHECK(matches(plan, 2, want, 4));
    }
}
TH_REG("expand", test_unpack_zmask)

/* ---- Unpack with halo + B: a full unpack emits A0,A1,A2,A3,B ---- */
static void test_unpack_halo_b(void)
{
    tm_planop_t plan[2] = {0};
    tm_utmpl_t *u = &plan[0].utmpl;

    plan[0].kind = TM_CFG_UNPACK;
    u->unpackB = 1;
    u->halo    = 1;
    u->A0 = 0xA0000000u; u->A1 = 0xA1000000u;
    u->A2 = 0xA2000000u; u->A3 = 0xA3000000u;
    u->B  = 0xB0000000u;
    u->skipA = 0x5A000000u; u->skipB = 0x5B000000u;

    plan[1].kind      = TM_MOP_RUN;
    plan[1].run_count = 2u;
    plan[1].run_zmask = 0x0u;   /* clear bits => both iterations execute */

    {
        tm_word_t want[10] = {
            0xA0000000u, 0xA1000000u, 0xA2000000u, 0xA3000000u, 0xB0000000u,
            0xA0000000u, 0xA1000000u, 0xA2000000u, 0xA3000000u, 0xB0000000u
        };
        CHECK(matches(plan, 2, want, 10));
    }
}
TH_REG("expand", test_unpack_halo_b)

/* ---- Replay: load with execute, then replay ---- */
static void test_replay_exec_then_run(void)
{
    tm_word_t   payload[3] = { 0x0A000000u, 0x0B000000u, 0x0C000000u };
    tm_planop_t plan[2] = {0};

    plan[0].kind     = TM_REPLAY_LOAD;
    plan[0].rp_start = 0u;
    plan[0].rp_len   = 3u;
    plan[0].rp_exec  = 1;            /* execute_while_loading */
    plan[0].rp_words = payload;

    plan[1].kind     = TM_REPLAY_RUN;
    plan[1].rp_start = 0u;
    plan[1].rp_len   = 3u;

    {
        tm_word_t want[6] = {
            0x0A000000u, 0x0B000000u, 0x0C000000u,   /* emitted while loading */
            0x0A000000u, 0x0B000000u, 0x0C000000u    /* emitted on replay     */
        };
        CHECK(matches(plan, 2, want, 6));
    }
}
TH_REG("expand", test_replay_exec_then_run)

/* ---- Replay: load WITHOUT execute emits nothing; replay emits ---- */
static void test_replay_noexec(void)
{
    tm_word_t   payload[2] = { 0xDEAD0000u, 0xBEEF0000u };
    tm_planop_t plan[2] = {0};

    plan[0].kind     = TM_REPLAY_LOAD;
    plan[0].rp_start = 4u;
    plan[0].rp_len   = 2u;
    plan[0].rp_exec  = 0;            /* load only, emit nothing now */
    plan[0].rp_words = payload;

    plan[1].kind     = TM_REPLAY_RUN;
    plan[1].rp_start = 4u;
    plan[1].rp_len   = 2u;

    {
        tm_word_t want[2] = { 0xDEAD0000u, 0xBEEF0000u };
        CHECK(matches(plan, 2, want, 2));
    }
}
TH_REG("expand", test_replay_noexec)

/* ---- The verifier must catch a mismatch and point at the exact word ---- */
static void test_verify_mismatch(void)
{
    ka_arena_t *A = fresh();
    tm_planop_t plan[3] = {0};
    tm_word_t   target[3] = { 0xA0000000u, 0x99999999u, 0xC0000000u };
    tm_verdict_t v;

    plan[0].kind = TM_EMIT; plan[0].word = 0xA0000000u;
    plan[1].kind = TM_EMIT; plan[1].word = 0xB0000000u;   /* != target[1] */
    plan[2].kind = TM_EMIT; plan[2].word = 0xC0000000u;

    v = tm_verify(plan, 3, target, 3, A);
    CHECK(!v.ok);
    CHEQ(v.diff_at, 1u);
    CHEQ(v.exp_word, 0x99999999u);
    CHEQ(v.got_word, 0xB0000000u);
    ka_free(A);
}
TH_REG("expand", test_verify_mismatch)

/* ---- Length mismatch is reported cleanly ---- */
static void test_verify_short(void)
{
    ka_arena_t *A = fresh();
    tm_planop_t plan[2] = {0};
    tm_word_t   target[3] = { 0xA0000000u, 0xB0000000u, 0xC0000000u };
    tm_verdict_t v;

    plan[0].kind = TM_EMIT; plan[0].word = 0xA0000000u;
    plan[1].kind = TM_EMIT; plan[1].word = 0xB0000000u;

    v = tm_verify(plan, 2, target, 3, A);
    CHECK(!v.ok);
    CHEQ(v.n_expanded, 2u);
    CHEQ(v.n_target, 3u);
    ka_free(A);
}
TH_REG("expand", test_verify_short)

/* ---- Running a template before programming one is a clean failure ---- */
static void test_run_before_cfg(void)
{
    ka_arena_t *A = fresh();
    tm_planop_t plan[1] = {0};
    tm_word_t   target[1] = { 0u };
    tm_verdict_t v;

    plan[0].kind = TM_MOP_RUN;
    v = tm_verify(plan, 1, target, 1, A);
    CHECK(!v.ok);              /* expansion should have failed */
    ka_free(A);
}
TH_REG("expand", test_run_before_cfg)

/* ---- Replay outside the buffer range is refused, not scribbled ---- */
static void test_replay_out_of_range(void)
{
    ka_arena_t *A = fresh();
    tm_word_t   payload[4] = { 1u, 2u, 3u, 4u };
    tm_planop_t plan[1] = {0};
    tm_word_t   target[1] = { 0u };
    tm_verdict_t v;

    plan[0].kind     = TM_REPLAY_LOAD;
    plan[0].rp_start = 30u;       /* 30 + 4 > 32 slots */
    plan[0].rp_len   = 4u;
    plan[0].rp_exec  = 1;
    plan[0].rp_words = payload;

    v = tm_verify(plan, 1, target, 1, A);
    CHECK(!v.ok);
    ka_free(A);
}
TH_REG("expand", test_replay_out_of_range)
