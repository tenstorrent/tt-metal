// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* tanchor.c -- the trust anchor: real matmul, real encodings.
 *
 * Everything else in the suite is synthetic or checked against the ttsim
 * oracle. This file ties tt-mop to an actual Tenstorrent kernel. The
 * instruction words here are not invented: they are the exact MVMUL
 * stream the Wormhole B0 matmul records, lifted from
 *   tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h  (matmul_configure_mop)
 * with encodings from TT_OP_MVMUL / TT_OP_SETRWC in ckernel_ops.h, and
 * cross-checked against ttas.
 *
 * How the real matmul issues these: it records the block once into the
 * replay buffer (lltt::record), then programs a MOP whose loop body is a
 * REPLAY of that buffer (lltt::replay_insn). The MOP supplies the loop,
 * the replay buffer supplies the block. So the executed stream is the
 * recorded block repeated inner_loops times, plus the fidelity-reset
 * end op. tt-mop expresses the same thing with its replay lever: load
 * once, run N times. These tests confirm tt-mop reproduces that exact
 * stream, and that its optimiser, handed the raw stream, rediscovers the
 * record-once-replay strategy on its own.
 *
 * MVMUL encoding: TT_OP_MVMUL(clr,mod,addr,dst) = 0x26<<24 | clr<<22 |
 * mod<<19 | addr<<15 | dst. The block below is all CLR_NONE, mod 0,
 * dst 0, so each word is 0x26000000 | (ADDR_MOD << 15).
 */

#include "tharns.h"
#include "ttmop.h"

#define MVMUL(addr) (0x26000000u | ((uint32_t)(addr) << 15))
#define SETRWC_RESET 0x37000008u   /* TT_OP_SETRWC(CLR_NONE,0,0,0,0,SET_F) */

/* The recorded block: the 15 MVMULs of the non-transposed full-tile path
 * (llk_math_matmul.h:378-397) plus the fidelity-reset MVMUL (ADDR_MOD_1).
 * addr_mods: 0 1 0 2 0 1 3 0 0 1 0 2 0 1 3 1. replay_buf_len = 16. */
static const tm_word_t MM_BLOCK[16] = {
    MVMUL(0), MVMUL(1), MVMUL(0), MVMUL(2),
    MVMUL(0), MVMUL(1), MVMUL(3), MVMUL(0),
    MVMUL(0), MVMUL(1), MVMUL(0), MVMUL(2),
    MVMUL(0), MVMUL(1), MVMUL(3), MVMUL(1)
};
#define MM_LEN 16u

static ka_arena_t *fresh(void)
{
    static ka_arena_t A;
    static int        ready = 0;
    if (!ready) {
        ka_init(&A, NULL, 1u << 20, 0);
        ready = 1;
    }
    ka_rst(&A);
    return &A;
}

/* ---- LoFi matmul: inner_loops = 1. One REPLAY of the block, no end op.
 * The plan mirrors the config: record the block (no execute), then run
 * it once. tt-mop must reproduce the 16-MVMUL stream exactly. ---- */
static void test_anchor_lofi(void)
{
    ka_arena_t  *A = fresh();
    tm_planop_t  plan[2];
    tm_verdict_t v;

    memset(plan, 0, sizeof(plan));
    plan[0].kind     = TM_REPLAY_LOAD;     /* lltt::record, no execute */
    plan[0].rp_start = 0u;
    plan[0].rp_len   = MM_LEN;
    plan[0].rp_exec  = 0;
    plan[0].rp_words = MM_BLOCK;
    plan[1].kind     = TM_REPLAY_RUN;      /* MOP runs the REPLAY once */
    plan[1].rp_start = 0u;
    plan[1].rp_len   = MM_LEN;

    v = tm_verify(plan, 2u, MM_BLOCK, MM_LEN, A);   /* macro emission = the block */
    CHECK(v.ok);
}
TH_REG("anchor", test_anchor_lofi)

/* ---- HiFi4 matmul: inner_loops = 4, plus the SETRWC fidelity reset as
 * the MOP end op. Macro emission = the block four times then SETRWC. The
 * plan records once and runs four times, then emits the end op. ---- */
static void test_anchor_hifi4(void)
{
    ka_arena_t  *A = fresh();
    tm_planop_t  plan[6];
    tm_word_t    macro[MM_LEN * 4u + 1u];
    tm_verdict_t v;
    uint32_t     m = 0u;

    for (uint32_t r = 0u; r < 4u; r++) {
        for (uint32_t i = 0u; i < MM_LEN; i++) {
            macro[m++] = MM_BLOCK[i];
        }
    }
    macro[m++] = SETRWC_RESET;

    memset(plan, 0, sizeof(plan));
    plan[0].kind = TM_REPLAY_LOAD;
    plan[0].rp_start = 0u; plan[0].rp_len = MM_LEN;
    plan[0].rp_exec = 0;   plan[0].rp_words = MM_BLOCK;
    for (uint32_t r = 0u; r < 4u; r++) {       /* four REPLAY runs */
        plan[1u + r].kind = TM_REPLAY_RUN;
        plan[1u + r].rp_start = 0u;
        plan[1u + r].rp_len = MM_LEN;
    }
    plan[5].kind = TM_EMIT;                     /* the MOP end op */
    plan[5].word = SETRWC_RESET;

    v = tm_verify(plan, 6u, macro, m, A);
    CHECK(v.ok);
}
TH_REG("anchor", test_anchor_hifi4)

/* ---- The matmul config expressed NATIVELY, the way the C++ actually
 * programs it: record the block once, then a MOP whose loop body is a
 * REPLAY of that block. matmul_configure_mop is ckernel_template(1 outer,
 * inner_loops, lltt::replay_insn(buf,len)) with set_end_op = SETRWC. Here
 * inner_loops = 4 (HiFi4). The expander must expand the REPLAY in the MOP
 * body, giving block x4 + SETRWC, no replay-lever rewrite needed. ---- */
static void test_anchor_hifi4_native(void)
{
    ka_arena_t  *A = fresh();
    tm_planop_t  plan[3];
    tm_dtmpl_t  *t;
    tm_word_t    macro[MM_LEN * 4u + 1u];
    tm_verdict_t v;
    uint32_t     m = 0u;

    for (uint32_t r = 0u; r < 4u; r++) {
        for (uint32_t i = 0u; i < MM_LEN; i++) {
            macro[m++] = MM_BLOCK[i];
        }
    }
    macro[m++] = SETRWC_RESET;

    memset(plan, 0, sizeof(plan));
    plan[0].kind = TM_REPLAY_LOAD;             /* lltt::record(block) */
    plan[0].rp_start = 0u; plan[0].rp_len = MM_LEN;
    plan[0].rp_exec = 0;   plan[0].rp_words = MM_BLOCK;

    plan[1].kind = TM_CFG_DOUBLE;              /* the MOP template */
    t = &plan[1].dtmpl;
    t->outer_len  = 1u;
    t->inner_len  = 4u;                        /* inner_loops = 4 */
    t->start_op0  = TM_NOP_WORD;
    t->end_op0    = SETRWC_RESET;              /* set_end_op */
    t->end_op1    = TM_NOP_WORD;
    t->loop_op0   = TM_REPLAY_RUN_WORD(0u, MM_LEN);   /* lltt::replay_insn */
    t->loop_op1   = TM_NOP_WORD;               /* no alternation */
    t->loop0_last = TM_REPLAY_RUN_WORD(0u, MM_LEN);
    t->loop1_last = TM_REPLAY_RUN_WORD(0u, MM_LEN);

    plan[2].kind = TM_MOP_RUN;

    v = tm_verify(plan, 3u, macro, m, A);
    CHECK(v.ok);
}
TH_REG("anchor", test_anchor_hifi4_native)

/* ---- The optimiser, handed the raw HiFi compute stream (the block four
 * times), must rediscover the matmul's own strategy: record the block
 * once, replay it for the rest. 64 direct -> load(1+16) + 3 runs = 20.
 * This is the keystone: tt-mop arrives at record-once-replay without
 * being told, and the verifier proves it reproduces the stream. ---- */
static void test_anchor_optimiser_rediscovers(void)
{
    ka_arena_t    *A = fresh();
    tm_word_t      stream[MM_LEN * 4u];
    tm_optresult_t r;
    tm_verdict_t   v;
    uint32_t       m = 0u;

    for (uint32_t rr = 0u; rr < 4u; rr++) {
        for (uint32_t i = 0u; i < MM_LEN; i++) {
            stream[m++] = MM_BLOCK[i];
        }
    }

    r = tm_optimise(stream, m, A);
    CHECK(r.verified);
    CHEQ(r.cost_naive, 64u);
    CHEQ(r.cost_opt, 20u);            /* load(17) + 3 runs(3) */
    CHECK(r.cost_opt < r.cost_naive);

    v = tm_verify(r.plan, r.n_ops, stream, m, A);
    CHECK(v.ok);
}
TH_REG("anchor", test_anchor_optimiser_rediscovers)
