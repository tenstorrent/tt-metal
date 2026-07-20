// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* tmopfit.c -- the MOP lever: folding single-instruction runs.
 *
 * tm_optimise pulls the replay lever, which cannot compress a run of one
 * repeated instruction (a length-1 block has a negative replay saving).
 * tm_optimise_mop pulls the MOP lever, which can: one programmed
 * double-loop reproduces up to 2048 copies. These tests check it folds
 * the runs it should, leaves alone the ones not worth it, and that every
 * plan it returns reproduces the target exactly (the verifier is the
 * floor under all of it).
 */

#include "tharns.h"
#include "ttmop.h"

static ka_arena_t *fresh(void)
{
    static ka_arena_t A;
    static int        ready = 0;
    if (!ready) {
        ka_init(&A, NULL, 1u << 22, 0);   /* 4 MiB */
        ready = 1;
    }
    ka_rst(&A);
    return &A;
}

/* Fill buf[0..n) with word w. */
static void fill(tm_word_t *buf, uint32_t n, tm_word_t w)
{
    for (uint32_t i = 0u; i < n; i++) {
        buf[i] = w;
    }
}

/* ---- A long single-op run folds, verifies, and gets cheaper ---- */
static void test_mop_basic_run(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[100];
    tm_optresult_t r;

    fill(target, 100u, 0x26000000u);          /* 100 identical MVMUL-ish */
    r = tm_optimise_mop(target, 100u, A);

    CHECK(r.verified);
    CHECK(r.cost_naive == 100u);
    CHECK(r.cost_opt < r.cost_naive);          /* 100 -> ~11 */
    /* 2 ops (cfg + run): 100 = 2 * 50 factors exactly within bounds. */
    CHECK(r.n_ops == 2u);
}
TH_REG("mopfit", test_mop_basic_run)

/* ---- A count <= 64 is covered exactly by outer=1 ---- */
static void test_mop_exact_64(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[64];
    tm_optresult_t r;

    fill(target, 64u, 0x30000000u);
    r = tm_optimise_mop(target, 64u, A);

    CHECK(r.verified);
    CHECK(r.n_ops == 2u);                       /* clean cfg+run, no tail */
    CHECK(r.cost_opt < r.cost_naive);
}
TH_REG("mopfit", test_mop_exact_64)

/* ---- A prime count above the bounds leaves a direct tail, still wins
 * and still verifies. 101 = 2*50 covered + 1 direct. ---- */
static void test_mop_remainder(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[101];
    tm_optresult_t r;

    fill(target, 101u, 0x41000000u);
    r = tm_optimise_mop(target, 101u, A);

    CHECK(r.verified);
    CHECK(r.cost_opt < r.cost_naive);
    CHECK(r.n_ops == 3u);                       /* cfg, run, one tail emit */
}
TH_REG("mopfit", test_mop_remainder)

/* ---- A run too short to clear the MOP setup cost is left direct ---- */
static void test_mop_too_short(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[8];
    tm_optresult_t r;

    fill(target, 8u, 0x26000000u);
    r = tm_optimise_mop(target, 8u, A);

    CHECK(r.verified);
    CHECK(r.cost_opt == r.cost_naive);          /* no fold: stayed direct */
}
TH_REG("mopfit", test_mop_too_short)

/* ---- A run embedded in other instructions: only the run folds ---- */
static void test_mop_run_in_stream(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      target[54];
    tm_optresult_t r;
    tm_verdict_t   v;

    target[0] = 0xAA000001u;
    target[1] = 0xBB000002u;
    fill(&target[2], 50u, 0x26000003u);         /* the run */
    target[52] = 0xCC000004u;
    target[53] = 0xDD000005u;

    r = tm_optimise_mop(target, 54u, A);
    CHECK(r.verified);
    CHECK(r.cost_opt < r.cost_naive);

    /* Re-verify independently, belt and suspenders. */
    v = tm_verify(r.plan, r.n_ops, target, 54u, A);
    CHECK(v.ok);
}
TH_REG("mopfit", test_mop_run_in_stream)

/* ---- The fitter returns nothing for a NOP run ---- */
static void test_mop_nop_run(void)
{
    tm_planop_t frag[2];
    uint32_t    cov = 123u;
    int         got = tm_fit_single_run(TM_NOP_WORD, 500u, frag, &cov);
    CHECK(got == 0);
    CHECK(cov == 0u);
}
TH_REG("mopfit", test_mop_nop_run)

/* ---- Fuzz: any non-NOP op at any count folds to a verified plan that
 * is never more expensive than emitting it directly. ---- */
static void test_mop_fuzz(void)
{
    uint32_t st = 0x1D872B41u;
    for (uint32_t it = 0u; it < 1500u; it++) {
        ka_arena_t    *A = fresh();
        tm_word_t      target[2048];
        tm_optresult_t r;
        uint32_t       count;
        tm_word_t      op;

        st    = (st * 1103515245u) + 12345u;
        count = 1u + (st % 2048u);
        st    = (st * 1103515245u) + 12345u;
        op    = (0x10u + (st % 0x60u)) << TM_OP_SHIFT;  /* never a NOP */
        op   |= (st & 0x00FFFFFFu);

        fill(target, count, op);
        r = tm_optimise_mop(target, count, A);

        CHECK(r.verified);
        CHECK(r.cost_opt <= r.cost_naive);
    }
}
TH_REG("mopfit", test_mop_fuzz)
