// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* ttmop_opt.c -- the cost model and the optimiser.
 *
 * The bounty in one line: issue the same Tensix instructions, but make
 * the poor RISC-V cores type a good deal less. The hardware hands you
 * two levers, the MOP expander and the replay buffer, and this file
 * yanks on both, whichever is cheaper for the stretch of stream in front
 * of it. A block that turns up more than once goes into the replay buffer
 * and gets parroted back for free; a single instruction repeated a heroic
 * number of times gets folded into a MOP loop; everything else is emitted
 * by hand. Direct, replay, MOP, pick your poison per segment.
 *
 * Every plan this pass dreams up is dragged in front of the verifier
 * before it is allowed to win anything. A cheaper plan that emits the
 * wrong stream is not a saving. It is a bug with good PR.
 *
 * The coverer is greedy, in the honest sense: it grabs the highest-saving
 * block that still fits the buffer and does not tread on one already
 * chosen, then the next, and so on down the line. Greedy is provably
 * optimal when the chosen blocks keep to themselves, which is the common
 * case, and the verifier keeps the whole thing honest when they do not.
 * Optimal covering under collision is a problem for a braver day; this
 * already factors every disjoint repeat in a stream, which is most of
 * what real kernels actually throw at it.
 */

#include "ttmop.h"

/* Most repeated-block candidates we will weigh in one stream. Bounded
 * so a pathological input cannot make us allocate forever. */
#define TM_MAX_PATS 256u

/* RISC-V issues to stand up and fire one programmed double-loop: the
 * sync, the nine config stores, and the run. A run pays off only when
 * the MOP covers more copies than this. */
#define TM_MOP_SETUP (TM_COST_MOP_SYNC \
                    + TM_DOUBLE_CFG_SLOTS * TM_COST_MOP_CFG_SLOT \
                    + TM_COST_MOP_RUN)

/* Largest outer*inner not exceeding count, outer in [1,32], inner in
 * [1,64]. Writes the factors through po and pi and returns the product
 * covered. A count of 64 or under is covered exactly with one outer pass;
 * anything that does not factor neatly is covered up to its nearest
 * reachable product, and the shortfall is left for the caller to emit by
 * hand. Primes, as ever, refuse to be reasonable about this. */
static uint32_t
fit_factors(uint32_t count, uint32_t *po, uint32_t *pi)
{
    uint32_t best = 0u, bo = 0u, bi = 0u;

    for (uint32_t o = 1u; o <= 32u; o++) {
        uint32_t i    = count / o;
        uint32_t prod;
        if (i > 64u) {
            i = 64u;
        }
        if (i == 0u) {
            break;                  /* o already exceeds count */
        }
        prod = o * i;
        if (prod > best) {
            best = prod;
            bo   = o;
            bi   = i;
        }
        if (best == count) {
            break;                  /* exact, cannot do better */
        }
    }
    *po = bo;
    *pi = bi;
    return best;
}

uint32_t
tm_cost(const tm_planop_t *plan, uint32_t n_ops)
{
    uint32_t total = 0u;

    if (plan == NULL) {
        return 0u;
    }

    for (uint32_t k = 0u; k < n_ops; k++) {
        const tm_planop_t *op = &plan[k];

        switch (op->kind) {
        case TM_EMIT:
            total += TM_COST_EMIT;
            break;

        case TM_CFG_DOUBLE:
            total += TM_COST_MOP_SYNC
                   + TM_DOUBLE_CFG_SLOTS * TM_COST_MOP_CFG_SLOT;
            break;

        case TM_CFG_UNPACK:
            total += TM_COST_MOP_SYNC
                   + TM_UNPACK_CFG_SLOTS * TM_COST_MOP_CFG_SLOT;
            break;

        case TM_MOP_RUN:
            total += TM_COST_MOP_RUN;
            break;

        case TM_REPLAY_LOAD:
            /* The REPLAY instruction itself, plus the words fed past it
             * to be captured: the core still has to issue those once.
             * Confirmed by load_replay_buf (record, then the LEN
             * instructions follow). */
            total += TM_COST_REPLAY_INSTR + op->rp_len * TM_COST_EMIT;
            break;

        case TM_REPLAY_RUN:
            /* One instruction re-emits the whole buffered range. This
             * single line is the entire point of the toolchain. */
            total += TM_COST_REPLAY_INSTR;
            break;

        default:
            break;
        }
    }
    return total;
}

/* Are the L words at a and b identical? */
static int
block_eq(const tm_word_t *t, uint32_t a, uint32_t b, uint32_t len)
{
    for (uint32_t i = 0u; i < len; i++) {
        if (t[a + i] != t[b + i]) {
            return 0;
        }
    }
    return 1;
}

/* Greedy non-overlapping occurrences of the block t[s .. s+len) across
 * the whole stream, left to right. Writes start positions into occ and
 * returns the count. Left-to-right greedy matches what the plan will
 * actually do walking the stream, so the count is honest. */
static uint32_t
find_occ(const tm_word_t *t, uint32_t n, uint32_t s, uint32_t len,
         uint32_t *occ, uint32_t occ_cap)
{
    uint32_t count = 0u;
    uint32_t i     = 0u;

    if (len == 0u || len > n) {
        return 0u;
    }
    while (i + len <= n) {
        if (block_eq(t, i, s, len)) {
            if (count < occ_cap) {
                occ[count] = i;
            }
            count++;
            i += len;           /* non-overlapping */
        } else {
            i++;
        }
    }
    return count;
}

/* Savings, in RISC-V issues, of replaying a length-L block that occurs
 * K times instead of emitting all of it directly.
 *
 *   direct : K * L
 *   replay : (1 + L) load+execute the first pass, then (K-1) runs
 *   saving : K*L - (L + K)
 *
 * Returns 0 when replay does not help. */
static uint32_t
replay_saving(uint32_t len, uint32_t k)
{
    uint32_t direct, plan_cost;

    if (k < 2u || len == 0u) {
        return 0u;
    }
    direct    = k * len;
    plan_cost = (1u + len) + (k - 1u);
    if (plan_cost >= direct) {
        return 0u;
    }
    return direct - plan_cost;
}

/* The all-direct plan: one EMIT per word. The floor the optimiser
 * improves on, and the fallback if nothing beats it. */
static tm_optresult_t
naive_plan(const tm_word_t *target, uint32_t n, ka_arena_t *A)
{
    tm_optresult_t r;

    r.plan       = NULL;
    r.n_ops      = 0u;
    r.cost_naive = n * TM_COST_EMIT;
    r.cost_opt   = n * TM_COST_EMIT;
    r.verified   = 0;
    r.note       = "no replay opportunity: emitting directly";

    r.plan = KA_NEWN(A, tm_planop_t, n == 0u ? 1u : n);
    if (r.plan == NULL) {
        r.note = "arena exhausted building naive plan";
        return r;
    }
    for (uint32_t i = 0u; i < n; i++) {
        r.plan[i].kind = TM_EMIT;
        r.plan[i].word = target[i];
    }
    r.n_ops = n;

    /* Verify even the all-direct plan rather than trusting it. It
     * reproduces any clean backend trace, but the frontend eats NOPs, so
     * a target carrying a literal NOP would not round-trip. This is the
     * fallback every other path lands on, so it has to report the real
     * answer, not a correctness it has not checked. */
    {
        tm_verdict_t v = tm_verify(r.plan, r.n_ops, target, n, A);
        r.verified = v.ok;
    }
    return r;
}

/* One covering candidate. A replay candidate compresses a multi-
 * instruction block that recurs across the stream; a MOP candidate folds
 * a contiguous run of one repeated instruction into a double-loop. Both
 * compete for the same positions, weighed by the RISC-V issues they
 * save, so the optimiser pulls whichever lever is cheaper per segment. */
typedef enum { CAND_REPLAY = 0, CAND_MOP = 1 } tm_candkind_t;

typedef struct {
    tm_candkind_t kind;
    uint32_t      start;   /* earliest position the candidate covers     */
    uint32_t      len;     /* replay: block length; MOP: copies covered  */
    uint32_t      k;       /* replay: occurrence count (MOP: 1)          */
    uint32_t      saving;  /* RISC-V issues saved vs all-direct          */
    tm_word_t     word;    /* MOP: the repeated instruction              */
} tm_cand_t;

tm_optresult_t
tm_optimise(const tm_word_t *target, uint32_t n, ka_arena_t *A)
{
    return tm_optimise_budget(target, n, TM_REPLAY_SLOTS, A);
}

tm_optresult_t
tm_optimise_budget(const tm_word_t *target, uint32_t n,
                   uint32_t max_slots, ka_arena_t *A)
{
    tm_optresult_t r;
    uint32_t  max_len;
    uint32_t *occ;
    tm_cand_t *cand;
    uint32_t  n_cand = 0u;
    uint8_t  *claimed;     /* position is inside a chosen candidate       */
    uint8_t  *start_op;    /* 0 none, 1 load, 2 run, 3 MOP, at a start    */
    uint32_t *start_len;   /* covered length at a start                   */
    uint32_t *start_slot;  /* replay-buffer slot at a start (replay only) */
    tm_word_t *start_word; /* repeated instruction at a start (MOP only)  */
    uint32_t  next_slot = 0u;
    uint32_t  n_replay = 0u;
    uint32_t  n_mop = 0u;

    if (target == NULL || A == NULL || n == 0u) {
        return naive_plan(target, n, A);
    }

    occ        = KA_NEWN(A, uint32_t, n);
    cand       = KA_NEWN(A, tm_cand_t, TM_MAX_PATS);
    claimed    = KA_NEWN(A, uint8_t,  n);
    start_op   = KA_NEWN(A, uint8_t,  n);
    start_len  = KA_NEWN(A, uint32_t, n);
    start_slot = KA_NEWN(A, uint32_t, n);
    start_word = KA_NEWN(A, tm_word_t, n);
    if (occ == NULL || cand == NULL || claimed == NULL ||
        start_op == NULL || start_len == NULL || start_slot == NULL ||
        start_word == NULL) {
        return naive_plan(target, n, A);
    }
    for (uint32_t i = 0u; i < n; i++) {
        claimed[i]    = 0u;
        start_op[i]   = 0u;
        start_len[i]  = 0u;
        start_slot[i] = 0u;
        start_word[i] = 0u;
    }

    /* Never propose more buffer than physically exists, whatever the
     * caller asked for. */
    if (max_slots > TM_REPLAY_SLOTS) {
        max_slots = TM_REPLAY_SLOTS;
    }

    /* Replay candidates: every distinct beneficial repeated block,
     * weighed once from its earliest occurrence. A block must fit the
     * buffer budget and cannot be longer than half the stream. */
    max_len = n / 2u;
    if (max_len > max_slots) {
        max_len = max_slots;
    }
    for (uint32_t len = max_len; len >= 2u && n_cand < TM_MAX_PATS; len--) {
        for (uint32_t s = 0u; s + len <= n && n_cand < TM_MAX_PATS; s++) {
            uint32_t k = find_occ(target, n, s, len, occ, n);
            uint32_t save;
            if (k < 2u || occ[0] != s) {
                continue;
            }
            save = replay_saving(len, k);
            if (save == 0u) {
                continue;
            }
            cand[n_cand].kind   = CAND_REPLAY;
            cand[n_cand].start  = s;
            cand[n_cand].len    = len;
            cand[n_cand].k      = k;
            cand[n_cand].saving = save;
            cand[n_cand].word   = 0u;
            n_cand++;
        }
    }

    /* MOP candidates: each maximal run of one repeated instruction that
     * a double-loop covers past its setup cost. The replay lever cannot
     * touch these (a length-1 block has a negative replay saving), so
     * this is where MOP earns its keep. */
    {
        uint32_t i = 0u;
        while (i < n && n_cand < TM_MAX_PATS) {
            uint32_t run = 1u;
            while (i + run < n && target[i + run] == target[i]) {
                run++;
            }
            if (!tm_is_nop(target[i])) {
                uint32_t o = 0u, p = 0u;
                uint32_t cov = fit_factors(run, &o, &p);
                if (cov > TM_MOP_SETUP) {
                    cand[n_cand].kind   = CAND_MOP;
                    cand[n_cand].start  = i;
                    cand[n_cand].len    = cov;   /* copies the loop makes */
                    cand[n_cand].k      = 1u;
                    cand[n_cand].saving = cov - TM_MOP_SETUP;
                    cand[n_cand].word   = target[i];
                    n_cand++;
                }
            }
            i += run;
        }
    }

    if (n_cand == 0u) {
        return naive_plan(target, n, A);
    }

    /* Sort candidates by saving, biggest first (selection sort; n_cand
     * is small and bounded). Ties broken toward the longer coverage. */
    for (uint32_t i = 0u; i < n_cand; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1u; j < n_cand; j++) {
            if (cand[j].saving > cand[best].saving ||
                (cand[j].saving == cand[best].saving &&
                 cand[j].len > cand[best].len)) {
                best = j;
            }
        }
        if (best != i) {
            tm_cand_t tmp = cand[i];
            cand[i]       = cand[best];
            cand[best]    = tmp;
        }
    }

    /* Greedy selection. Take a candidate if none of the positions it
     * covers are already claimed, and (replay only) it still fits the
     * buffer budget. The verifier is the backstop, so greedy-by-saving
     * is safe even where it is not provably optimal. */
    for (uint32_t ci = 0u; ci < n_cand; ci++) {
        if (cand[ci].kind == CAND_REPLAY) {
            uint32_t len = cand[ci].len;
            uint32_t k   = find_occ(target, n, cand[ci].start, len, occ, n);
            int      conflict = 0;

            if (k < 2u) {
                continue;
            }
            if (next_slot + len > max_slots) {
                continue;   /* no buffer room left within the budget */
            }
            for (uint32_t j = 0u; j < k && !conflict; j++) {
                for (uint32_t t = 0u; t < len; t++) {
                    if (claimed[occ[j] + t]) {
                        conflict = 1;
                        break;
                    }
                }
            }
            if (conflict) {
                continue;
            }
            for (uint32_t j = 0u; j < k; j++) {
                start_op[occ[j]]   = (j == 0u) ? 1u : 2u;   /* load / run */
                start_len[occ[j]]  = len;
                start_slot[occ[j]] = next_slot;
                for (uint32_t t = 0u; t < len; t++) {
                    claimed[occ[j] + t] = 1u;
                }
            }
            next_slot += len;
            n_replay++;
        } else {
            uint32_t s   = cand[ci].start;
            uint32_t cov = cand[ci].len;
            int      conflict = 0;

            /* If any covered position is already claimed, drop the whole
             * MOP rather than shrinking it to fit the unclaimed prefix.
             * Deliberate: a higher-saving block won this span, and
             * re-fitting a partial run is a refinement, not a correctness
             * need (the verifier guarantees correctness either way). The
             * cost is a little saving left on the table when a run is cut
             * across by a chosen replay block. */
            for (uint32_t t = 0u; t < cov; t++) {
                if (claimed[s + t]) {
                    conflict = 1;
                    break;
                }
            }
            if (conflict) {
                continue;
            }
            start_op[s]   = 3u;           /* MOP */
            start_len[s]  = cov;
            start_word[s] = cand[ci].word;
            for (uint32_t t = 0u; t < cov; t++) {
                claimed[s + t] = 1u;
            }
            n_mop++;
        }
    }

    if (n_replay == 0u && n_mop == 0u) {
        return naive_plan(target, n, A);
    }

    /* Stitch the plan. At a start: a replay load/run, or a programmed MOP
     * double-loop; jumping by the covered length skips its interior.
     * Everywhere else, a direct emit. A run longer than the loop could
     * cover leaves an unclaimed tail, which falls out here as direct. */
    {
        tm_planop_t *plan  = KA_NEWN(A, tm_planop_t, n);
        uint32_t     n_ops = 0u;
        uint32_t     i     = 0u;
        tm_verdict_t v;

        if (plan == NULL) {
            return naive_plan(target, n, A);
        }

        while (i < n) {
            if (start_op[i] == 1u || start_op[i] == 2u) {
                tm_planop_t *op = &plan[n_ops];
                op->rp_start = start_slot[i];
                op->rp_len   = start_len[i];
                if (start_op[i] == 1u) {
                    op->kind     = TM_REPLAY_LOAD;
                    op->rp_exec  = 1;               /* first pass = load */
                    op->rp_words = &target[i];
                } else {
                    op->kind = TM_REPLAY_RUN;
                }
                n_ops++;
                i += start_len[i];
            } else if (start_op[i] == 3u) {
                uint32_t cov = 0u;
                int got = tm_fit_single_run(start_word[i], start_len[i],
                                            &plan[n_ops], &cov);
                /* start_len is a product fit_factors already returned, so
                 * this rebuild always covers it exactly with two ops. */
                n_ops += (uint32_t)got;
                i += start_len[i];
            } else {
                plan[n_ops].kind = TM_EMIT;
                plan[n_ops].word = target[i];
                n_ops++;
                i++;
            }
        }

        /* The non-negotiable step: prove the plan reproduces the target
         * before handing it back. If it somehow does not, the bug is
         * ours and the safe answer is the all-direct plan. */
        v = tm_verify(plan, n_ops, target, n, A);
        if (!v.ok) {
            tm_optresult_t fb = naive_plan(target, n, A);
            fb.note = "optimised plan failed verification; fell back to direct";
            return fb;
        }

        r.plan       = plan;
        r.n_ops      = n_ops;
        r.cost_naive = n * TM_COST_EMIT;
        r.cost_opt   = tm_cost(plan, n_ops);
        r.verified   = 1;
        r.note       = (n_mop > 0u && n_replay > 0u)
                         ? "covered with both replay and MOP loops"
                         : (n_mop > 0u ? "folded runs into MOP loops"
                                       : "replayed repeated blocks");
        return r;
    }
}

/* ================================================================
 *  The MOP lever: fold single-instruction runs into double-loops.
 * ================================================================ */

int
tm_fit_single_run(tm_word_t op, uint32_t count,
                  tm_planop_t *out, uint32_t *covered)
{
    uint32_t o = 0u, i = 0u, cov;

    if (covered != NULL) {
        *covered = 0u;
    }
    if (out == NULL) {
        return 0;
    }
    /* A NOP run expands to nothing (the trace eats NOPs), and a run that
     * cannot be covered past the setup cost is cheaper left direct. */
    if (tm_is_nop(op) || count <= TM_MOP_SETUP) {
        return 0;
    }

    cov = fit_factors(count, &o, &i);
    if (cov <= TM_MOP_SETUP) {
        return 0;
    }

    out[0].kind             = TM_CFG_DOUBLE;
    out[0].word             = 0u;
    out[0].dtmpl.outer_len  = o;
    out[0].dtmpl.inner_len  = i;
    out[0].dtmpl.start_op0  = TM_NOP_WORD;   /* skipped */
    out[0].dtmpl.end_op0    = TM_NOP_WORD;   /* skipped */
    out[0].dtmpl.end_op1    = TM_NOP_WORD;   /* skipped */
    out[0].dtmpl.loop_op0   = op;
    out[0].dtmpl.loop_op1   = TM_NOP_WORD;   /* no alternation */
    out[0].dtmpl.loop0_last = op;            /* non-NOP, the op itself */
    out[0].dtmpl.loop1_last = op;

    out[1].kind = TM_MOP_RUN;

    if (covered != NULL) {
        *covered = cov;
    }
    return 2;
}

tm_optresult_t
tm_optimise_mop(const tm_word_t *target, uint32_t n, ka_arena_t *A)
{
    tm_optresult_t r;
    tm_planop_t   *plan;
    uint32_t       n_ops  = 0u;
    uint32_t       i      = 0u;
    uint32_t       n_mop  = 0u;
    tm_verdict_t   v;

    if (target == NULL || A == NULL || n == 0u) {
        return naive_plan(target, n, A);
    }

    /* Every fold collapses >= TM_MOP_SETUP+1 words into two plan ops, so
     * the plan never has more ops than the stream has words. */
    plan = KA_NEWN(A, tm_planop_t, n);
    if (plan == NULL) {
        return naive_plan(target, n, A);
    }

    while (i < n) {
        uint32_t    run = 1u;
        tm_planop_t frag[2];
        uint32_t    cov = 0u;
        int         got;

        while (i + run < n && target[i + run] == target[i]) {
            run++;
        }

        got = tm_fit_single_run(target[i], run, frag, &cov);
        if (got == 2 && cov > TM_MOP_SETUP && cov <= run) {
            plan[n_ops++] = frag[0];
            plan[n_ops++] = frag[1];
            for (uint32_t t = cov; t < run; t++) {   /* uncovered tail */
                plan[n_ops].kind = TM_EMIT;
                plan[n_ops].word = target[i];
                n_ops++;
            }
            n_mop++;
        } else {
            for (uint32_t t = 0u; t < run; t++) {
                plan[n_ops].kind = TM_EMIT;
                plan[n_ops].word = target[i];
                n_ops++;
            }
        }
        i += run;
    }

    if (n_mop == 0u) {
        return naive_plan(target, n, A);
    }

    v = tm_verify(plan, n_ops, target, n, A);
    if (!v.ok) {
        tm_optresult_t fb = naive_plan(target, n, A);
        fb.note = "MOP plan failed verification; fell back to direct";
        return fb;
    }

    r.plan       = plan;
    r.n_ops      = n_ops;
    r.cost_naive = n * TM_COST_EMIT;
    r.cost_opt   = tm_cost(plan, n_ops);
    r.verified   = 1;
    r.note       = (n_mop == 1u)
                     ? "folded a single-instruction run into a MOP loop"
                     : "folded single-instruction runs into MOP loops";
    return r;
}
