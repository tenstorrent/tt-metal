// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* topt.c -- the cost model and the optimiser loop.
 *
 * Two things to prove here. One: the cost model counts RISC-V issues
 * the way the hardware would, so the search is steering by a real
 * number. Two: the optimiser actually reduces that number, and the
 * plan it returns still reproduces the target exactly. A cheaper plan
 * that emits the wrong stream is the one failure mode that matters, so
 * we check the saving and the correctness together. */

#include "tharns.h"
#include "ttmop.h"

static ka_arena_t *fresh(void)
{
    static ka_arena_t A;
    ka_init(&A, NULL, 1u << 20, 0);
    return &A;
}

/* ---- Cost model counts direct emits one apiece ---- */
static void test_cost_naive(void)
{
    tm_planop_t plan[5] = {0};
    for (int i = 0; i < 5; i++) {
        plan[i].kind = TM_EMIT;
        plan[i].word = (tm_word_t)(0x01000000u + (uint32_t)i);
    }
    CHEQ(tm_cost(plan, 5u), 5u);
}
TH_REG("opt", test_cost_naive)

/* ---- Cost model: a load of L plus two runs ----
 * load = 1 REPLAY + L words fed in; each run = 1 REPLAY. With L=3:
 * (1+3) + 1 + 1 = 6. ---- */
static void test_cost_replay(void)
{
    tm_word_t   payload[3] = { 1u, 2u, 3u };
    tm_planop_t plan[3] = {0};

    plan[0].kind = TM_REPLAY_LOAD; plan[0].rp_len = 3u;
    plan[0].rp_words = payload;    plan[0].rp_exec = 1;
    plan[1].kind = TM_REPLAY_RUN;  plan[1].rp_len = 3u;
    plan[2].kind = TM_REPLAY_RUN;  plan[2].rp_len = 3u;

    CHEQ(tm_cost(plan, 3u), 6u);
}
TH_REG("opt", test_cost_replay)

/* ---- The optimiser finds a repeated block and replays it ----
 * Target: A B C X A B C Y A B C. The block ABC (len 3) occurs 3 times.
 * naive = 11. opt = 11 - (3*3) + (3 + 3) = 8. And it must verify. ---- */
static void test_optimise_finds_repeat(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[11] = {
        0xA0u, 0xB0u, 0xC0u,        /* A B C */
        0x11u,                      /* X     */
        0xA0u, 0xB0u, 0xC0u,        /* A B C */
        0x22u,                      /* Y     */
        0xA0u, 0xB0u, 0xC0u         /* A B C */
    };
    tm_optresult_t r;
    tm_verdict_t   v;

    r = tm_optimise(target, 11u, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 11u);
    CHEQ(r.cost_opt, 8u);
    CHECK(r.cost_opt < r.cost_naive);

    /* Belt and braces: re-verify the returned plan against the target. */
    v = tm_verify(r.plan, r.n_ops, target, 11u, A);
    CHECK(v.ok);
    ka_free(A);
}
TH_REG("opt", test_optimise_finds_repeat)

/* ---- No repeats: optimiser returns the verified naive plan ---- */
static void test_optimise_no_repeat(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[5] = { 0x10u, 0x20u, 0x30u, 0x40u, 0x50u };
    tm_optresult_t r;

    r = tm_optimise(target, 5u, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 5u);
    CHEQ(r.cost_opt, 5u);       /* nothing to save */
    CHEQ(r.n_ops, 5u);          /* all direct emits */
    ka_free(A);
}
TH_REG("opt", test_optimise_no_repeat)

/* ---- A longer repeated block is preferred (saves more) ----
 * Target: P Q R S  P Q R S, an 8-word stream that is one 4-block twice.
 * naive = 8. opt = 8 - (2*4) + (4 + 2) = 6. ---- */
static void test_optimise_longer_block(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[8] = {
        0x50u, 0x51u, 0x52u, 0x53u,
        0x50u, 0x51u, 0x52u, 0x53u
    };
    tm_optresult_t r;
    tm_verdict_t   v;

    r = tm_optimise(target, 8u, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 8u);
    CHEQ(r.cost_opt, 6u);
    v = tm_verify(r.plan, r.n_ops, target, 8u, A);
    CHECK(v.ok);
    ka_free(A);
}
TH_REG("opt", test_optimise_longer_block)

/* ---- Multiple disjoint repeats: the coverer factors them all ----
 * Stream: A B C  A B C  X  D E F  D E F. Two distinct length-3 blocks,
 * each twice, with a lone X between. Single-block would take one (save
 * 1) and leave the other direct, landing at 12. The multi-pattern
 * coverer takes both: 13 - 1 - 1 = 11. ---- */
static void test_optimise_multi_block(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[13] = {
        0xA0u, 0xB0u, 0xC0u,        /* A B C */
        0xA0u, 0xB0u, 0xC0u,        /* A B C */
        0x99u,                      /* X     */
        0xD0u, 0xE0u, 0xF0u,        /* D E F */
        0xD0u, 0xE0u, 0xF0u         /* D E F */
    };
    tm_optresult_t r;
    tm_verdict_t   v;

    r = tm_optimise(target, 13u, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 13u);
    CHEQ(r.cost_opt, 11u);          /* both blocks replayed */
    CHECK(r.cost_opt < 12u);        /* strictly better than single-block */
    v = tm_verify(r.plan, r.n_ops, target, 13u, A);
    CHECK(v.ok);
    ka_free(A);
}
TH_REG("opt", test_optimise_multi_block)

/* ---- Buffer budget (thread-sharing constraint) is respected ----
 * Same two-disjoint-block stream as above, but with a budget of only 3
 * slots. Each block is length 3, so only ONE fits. The coverer must
 * take one block (down to 12) and leave the other direct, never
 * overrunning the budget. With the full budget it would reach 11. ---- */
static void test_optimise_budget(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[13] = {
        0xA0u, 0xB0u, 0xC0u,
        0xA0u, 0xB0u, 0xC0u,
        0x99u,
        0xD0u, 0xE0u, 0xF0u,
        0xD0u, 0xE0u, 0xF0u
    };
    tm_optresult_t r;
    tm_verdict_t   v;

    r = tm_optimise_budget(target, 13u, 3u, A);   /* only 3 slots */
    CHECK(r.verified);
    CHEQ(r.cost_opt, 12u);          /* one block replayed, not both */
    v = tm_verify(r.plan, r.n_ops, target, 13u, A);
    CHECK(v.ok);
    ka_free(A);
}
TH_REG("opt", test_optimise_budget)

/* ---- The unified pass pulls both levers in one stream ----
 * A B C  A B C  then X repeated 30 times. The replay lever takes the ABC
 * block (load + run), the MOP lever folds the 30-long run into one
 * double-loop. naive = 36. Replay ABC: load(1+3) + run(1) = 5. MOP run:
 * sync + 9 cfg + run = 11. Total 16, and it must verify. ---- */
static void test_optimise_replay_and_mop(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[36];
    tm_optresult_t r;
    tm_verdict_t   v;

    target[0] = 0xA0u; target[1] = 0xB0u; target[2] = 0xC0u;
    target[3] = 0xA0u; target[4] = 0xB0u; target[5] = 0xC0u;
    for (uint32_t i = 6u; i < 36u; i++) {
        target[i] = 0x26000000u;            /* X, thirty times */
    }

    r = tm_optimise(target, 36u, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 36u);
    CHEQ(r.cost_opt, 16u);
    CHECK(r.cost_opt < r.cost_naive);

    v = tm_verify(r.plan, r.n_ops, target, 36u, A);
    CHECK(v.ok);
    ka_free(A);
}
TH_REG("opt", test_optimise_replay_and_mop)

/* ---- A target carrying a literal NOP cannot round-trip (the frontend
 * eats NOPs), so the optimiser must report it unverified rather than
 * claiming a correctness it does not have. This is the honesty fix in
 * the all-direct fallback: it verifies itself now. ---- */
static void test_optimise_nop_target_unverified(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[3] = { 0x26000000u, TM_NOP_WORD, 0x26000001u };
    tm_optresult_t r = tm_optimise(target, 3u, A);

    CHECK(!r.verified);     /* would have been a false "verified" before */
    ka_free(A);
}
TH_REG("opt", test_optimise_nop_target_unverified)

/* ---- Empty stream is handled, not crashed ---- */
static void test_optimise_empty(void)
{
    ka_arena_t    *A = fresh();
    tm_optresult_t r = tm_optimise(NULL, 0u, A);
    CHEQ(r.cost_naive, 0u);
    CHEQ(r.cost_opt, 0u);
    ka_free(A);
}
TH_REG("opt", test_optimise_empty)
