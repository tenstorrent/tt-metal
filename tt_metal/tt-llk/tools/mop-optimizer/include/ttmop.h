// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* ttmop.h -- tt-mop, the open Tensix MOP/replay optimiser
 *
 * Second piece of the toolchain: ttas assembles, tt-mop optimises,
 * tt-emu emulates. The job is to take a flat stream of Tensix
 * instructions, the sort a RISC-V baby core would otherwise issue one
 * weary cycle at a time, and find a cheaper way to produce the exact
 * same stream using the two issue-compression gadgets the hardware
 * bothered to give us: the MOP (macro-op) expander and the replay
 * buffer. Fewer RISC-V issues out the front, identical Tensix bus trace
 * out the back, and nobody downstream any the wiser.
 *
 * The whole optimiser is a search, and a search is only ever as
 * trustworthy as the thing scoring it, so the verifier comes first.
 * Given a plan (some mix of direct emits, MOP runs, and replay), the
 * expander reconstructs the Tensix stream that plan would produce and
 * the verifier checks it equals the target word for word. Get the
 * expander right and every later pass is free to be wrong loudly instead
 * of quietly, which is the only kind of wrong worth having.
 *
 * The expansion rules come straight from tt-llk's ckernel_template.h
 * (Apache 2.0, (c) Tenstorrent). A few points of hardware behaviour the
 * header leaves implicit are written up in the ASSUMPTION notes below.
 * All are now resolved (against the ISA docs, ttsim, and the LLK source);
 * one is kept as a single #define flip in case a future chip inverts it.
 */

#ifndef TTMOP_H
#define TTMOP_H

#include <stdint.h>
#include "kauri.h"

/* A single 32-bit Tensix instruction word. Same encoding model as
 * ttas: opcode in bits [31:24], operands in the low 24. */
typedef uint32_t tm_word_t;

#define TM_OP_SHIFT   24u
#define TM_OP_MOP     0x01u    /* MOP    (run a programmed template)   */
#define TM_OP_REPLAY  0x04u    /* REPLAY (load or replay the buffer)   */
#define TM_OP_NOP     0x02u    /* NOP    (opcode 0x02, no operands)    */
#define TM_NOP_WORD   0x02000000u   /* TT_OP_NOP, the empty template slot */

/* Encode a REPLAY instruction word, the frontend's replay-expander op,
 * matching the Tensix encoding (ttsim src/tensix.cpp): opcode 0x04,
 * start_idx in bits [23:14], len in bits [13:4], execute-while-loading
 * in bit 1, load-mode in bit 0. A MOP template slot can carry one of
 * these: the expander decodes it and acts on the replay buffer, exactly
 * as the real matmul does (its MOP loop body is an lltt::replay_insn).
 * A load word sets up the capture window; a run word re-emits a range. */
#define TM_REPLAY_WORD(start, len, exec, load)            \
    ((TM_OP_REPLAY << TM_OP_SHIFT)                        \
     | (((uint32_t)(start) & 0x3FFu) << 14)               \
     | (((uint32_t)(len)   & 0x3FFu) << 4)                \
     | (((uint32_t)(exec)  & 1u)     << 1)                \
     | ((uint32_t)(load)   & 1u))
#define TM_REPLAY_RUN_WORD(start, len)  TM_REPLAY_WORD((start), (len), 0u, 0u)
#define TM_REPLAY_LOAD_WORD(start, len, exec) TM_REPLAY_WORD((start), (len), (exec), 1u)

/* Capacity ceilings. Fixed and visible, JPL-style: no unbounded
 * anything. A plan or an expansion that wants more than this is told
 * no rather than allowed to wander off the end of an array. */
#define TM_MAX_WORDS    65536u   /* longest stream the expander emits  */
#define TM_REPLAY_SLOTS 32u      /* Wormhole: 16 FPU + 16 SFPU = 32    */

/* =================================================================
 *  Hardware behaviour notes (all resolved; one kept as a flip).
 * ================================================================= */

/* ASSUMPTION 1 -- RESOLVED (2026-06-10), and it never needed silicon.
 * Two facts, both documented, settle it:
 *
 *  (a) The MOP expander SKIPS NOP template slots: start_op, end_op0,
 *      end_op1 are each emitted only when not a NOP (and end_op1 only if
 *      end_op0 also fired). Modelled explicitly in expand_double, since
 *      it changes which later slots fire.
 *  (b) The Tensix frontend then swallows EVERY NOP before the backend
 *      (tensix_push_inst_fifo eats opcode 0x02), so the executed bus
 *      trace is NOP-free regardless of source: direct issue, replay, or
 *      a MOP body. Modelled in out_put, which drops any NOP.
 *
 * Sources, in agreement word for word:
 *   - tt-isa-documentation MOPExpander.md (ExpandTemplate1): the
 *     "(not used if it's a NOP)" notes on StartOp / EndOp0 / EndOp1.
 *   - ttsim src/tensix.cpp: mop_expander's `if (!IS_TENSIX_NOP(...))`
 *     structural guards, and tensix_push_inst_fifo's NOP eat.
 * Resolved in bounty thread #1638: amahmudTT pointed at the ISA doc,
 * mcraigheadTT reviewed the spec and confirmed (commit 9ba1ce0a).
 *
 * DECISION: the optimiser's target is the NOP-free backend trace, the
 * thing a tt-metal PR is judged against. So there is no flip switch here
 * any more; NOPs simply never reach the trace. Double-loop MOP fitting
 * is unblocked. NOTE: a NOP still occupies a replay-buffer SLOT (it is
 * stored on load, just never executed), which the slot budget already
 * accounts for; only execution is suppressed. */

/* ASSUMPTION 2 -- CONFIRMED against tt-llk source (2026-06-04), and it
 * is the OPPOSITE of the misleading ckernel_template.h header comment.
 * A SET zmask bit means SKIP that iteration (the skipA/skipB path); a
 * CLEAR bit means execute the full unpack. So 0 here (set bit => skip).
 *
 * Evidence, three independent confirmations:
 *   - llk_math_transpose_dest.h: "zmask 0-bits: [transpose]. zmask
 *     1-bits: NOP."
 *   - llk_unpack_AB.h: the loop guard is literally `if (!zmask[i])`.
 *   - llk_unpack_AB.h: "iteration 0 zmask will be 0 so unpacker will
 *     execute ... Next iterations have zmask on 1 and execute SKIP."
 * It also matches the run(count) convenience path, which passes
 * zmask=0 to mean "execute all count iterations".
 *
 * Left as a flip in case another chip generation inverts it. */
#ifndef TM_ZMASK_BIT_EXECUTES
#define TM_ZMASK_BIT_EXECUTES 0
#endif

/* ASSUMPTION 3 -- CONFIRMED against tt-llk source (2026-06-04).
 * REPLAY load with execute_while_loading set: a load always stores the
 * words into the buffer, and if the bit is set it ALSO executes them as
 * they load. Replay then re-emits the stored words.
 *
 * Source: ckernel.h::load_replay_buf (tt_llk_blackhole/common/inc),
 * whose own comment reads "EXEC is true to execute when loading
 * (default is false). LEN is the number of instructions to record."
 * Its body issues the record (REPLAY-load) instruction, then issues the
 * LEN instructions, which the core feeds past it. That is exactly the
 * model here: REPLAY_LOAD costs one REPLAY plus the LEN words fed, and
 * with rp_exec set the first pass is produced by the load itself. The
 * optimiser relies on this to make a block's first occurrence the load.
 *
 * The REPLAY word encoding (load-mode bit 0, exec bit 1, len/start
 * fields) now follows ttsim's layout, and tm_issue decodes these when a
 * MOP body emits one, the matmul's actual shape. The differential fuzz
 * against ttsim confirms the two agree, so the bit layout is pinned, not
 * open. */

/* =================================================================
 *  The two MOP template shapes.
 * ================================================================= */

/* Double-loop template (ckernel_template, run as TTI_MOP(1,0,0)).
 * Expansion, matching the Tensix MOP expander exactly (ISA doc
 * MOPExpander.md ExpandTemplate1 == ttsim mop_expander):
 *
 *   loop_op = loop_op0; flip = 0
 *   if loop_op1 is not a NOP:        # alternation + doubling
 *       flip = loop_op0 ^ loop_op1
 *       inner_len *= 2
 *   for o in 0 .. outer_len-1:
 *       if start_op0 not NOP: emit start_op0
 *       for i in 0 .. inner_len-1:
 *           if   i != inner_len-1:   emit loop_op       # one op per step
 *           elif o != outer_len-1:   emit loop1_last
 *           else:                    emit loop0_last
 *           loop_op ^= flip          # carries across outer iterations
 *       if end_op0 not NOP:
 *           emit end_op0
 *           if end_op1 not NOP: emit end_op1
 *
 * The body emits ONE op per step. loop_op1 is the alternate value the
 * single body op flips to and from, not a second emitted slot, and its
 * presence doubles the inner count. The last step REPLACES the body op
 * (loop1_last / loop0_last), it does not append.
 */
typedef struct {
    uint32_t  outer_len;
    uint32_t  inner_len;
    tm_word_t start_op0;
    tm_word_t end_op0;
    tm_word_t end_op1;
    tm_word_t loop_op0;
    tm_word_t loop_op1;     /* NOP => no alternation; else flips with loop_op0 */
    tm_word_t loop0_last;   /* replaces body op on last inner of last outer    */
    tm_word_t loop1_last;   /* replaces body op on last inner, earlier outer   */
} tm_dtmpl_t;

/* Single-loop unpack template (ckernel_unpack_template, run as
 * TT_MOP(0, count-1, zmask)). Per iteration, the zmask bit selects a
 * full unpack or a skip:
 *
 *   for i in 0 .. count-1:
 *       if zmask_bit(i) executes:
 *           emit A0; if halo: emit A1,A2,A3; if unpackB: emit B
 *       else:
 *           emit skipA; if unpackB: emit skipB
 */
typedef struct {
    int       unpackB;      /* emit B / skipB at all            */
    int       halo;         /* emit A1,A2,A3 (else A0 only)      */
    tm_word_t A0, A1, A2, A3;
    tm_word_t B;
    tm_word_t skipA;
    tm_word_t skipB;
} tm_utmpl_t;

/* =================================================================
 *  The plan IR.
 * ================================================================= */

typedef enum {
    TM_EMIT = 0,        /* one literal word straight into the stream   */
    TM_CFG_DOUBLE,      /* program the double-loop template            */
    TM_CFG_UNPACK,      /* program the unpack template                 */
    TM_MOP_RUN,         /* run whichever template was last programmed  */
    TM_REPLAY_LOAD,     /* store `len` words into the buffer at `start`*/
    TM_REPLAY_RUN       /* re-emit `len` words from the buffer at `start`*/
} tm_opkind_t;

typedef struct {
    tm_opkind_t kind;

    /* TM_EMIT */
    tm_word_t   word;

    /* TM_CFG_DOUBLE / TM_CFG_UNPACK carry the template inline. */
    tm_dtmpl_t  dtmpl;
    tm_utmpl_t  utmpl;

    /* TM_MOP_RUN: unpack template uses count + zmask; the double-loop
     * template takes its lengths from the programmed config and
     * ignores these. */
    uint32_t    run_count;
    uint32_t    run_zmask;

    /* TM_REPLAY_LOAD / TM_REPLAY_RUN. For LOAD, `words` points at the
     * `len` words to store (and emit too, if exec is set). */
    uint32_t         rp_start;
    uint32_t         rp_len;
    int              rp_exec;    /* execute_while_loading (LOAD only)  */
    const tm_word_t *rp_words;   /* the words to load   (LOAD only)    */
} tm_planop_t;

/* =================================================================
 *  Expansion + verification.
 * ================================================================= */

/* The reconstructed Tensix stream a plan produces. `words` is arena-
 * owned; do not free it yourself. `ok` is 0 on overflow or a malformed
 * plan, with `err` set to a static reason string. */
typedef struct {
    tm_word_t  *words;
    uint32_t    n;
    int         ok;
    const char *err;
} tm_stream_t;

/* The verifier's answer. `ok` is 1 iff the expanded stream equals the
 * target word for word. On a mismatch, `diff_at` is the first index
 * that differs (or the shorter length, if one stream is a prefix of
 * the other), with exp/got the words at that point. */
typedef struct {
    int         ok;
    uint32_t    n_expanded;
    uint32_t    n_target;
    uint32_t    diff_at;
    tm_word_t   exp_word;   /* target word at diff_at  */
    tm_word_t   got_word;   /* expanded word at diff_at */
    const char *msg;
} tm_verdict_t;

/* Expand a plan into the Tensix stream it produces. Allocates the
 * output from `A`. Pure: no globals, no hidden state between calls. */
tm_stream_t tm_expand(const tm_planop_t *plan, uint32_t n_ops, ka_arena_t *A);

/* Expand the plan and compare it against `target`. The heart of the
 * optimiser: a plan is only allowed to win the search if this says ok. */
tm_verdict_t tm_verify(const tm_planop_t *plan, uint32_t n_ops,
                       const tm_word_t *target, uint32_t n_target,
                       ka_arena_t *A);

/* True if a word is a NOP (opcode 0x02). Used by the NOP-skip path. */
int tm_is_nop(tm_word_t w);

/* =================================================================
 *  Cost model.
 * ================================================================= */

/* What the optimiser is actually minimising: the number of RISC-V
 * instructions a baby core issues to push a plan at the Tensix engine.
 * A direct emit is one issue. A MOP or a replay trades a fixed setup
 * cost for collapsing many emits into one run, so they pay off only
 * past a break-even point, which is the whole game.
 *
 * These constants are the cost model's tuning knobs, same spirit as
 * the hardware ASSUMPTIONs above: best-guess defaults now, calibrated
 * against real RISC-V emission later. Each is a one-line change.
 *
 * Open numbers from the handover, folded in here:
 *   - a single TTI_* emit is assumed to be 1 RISC-V instruction
 *   - mop_sync() is charged once per template program (might be more)
 *   - the 9 / 8 mop_cfg[] stores are charged one issue each
 */
#define TM_COST_EMIT          1u   /* one direct instrn_buffer write    */
#define TM_COST_MOP_SYNC      1u   /* mop_sync() before programming      */
#define TM_COST_MOP_CFG_SLOT  1u   /* one mop_cfg[] register store       */
#define TM_COST_MOP_RUN       1u   /* the TTI_MOP / TT_MOP run instr      */
#define TM_COST_REPLAY_INSTR  1u   /* one REPLAY (load or run) instr      */
#define TM_DOUBLE_CFG_SLOTS   9u   /* double-loop template config stores */
#define TM_UNPACK_CFG_SLOTS   8u   /* unpack template config stores      */

/* RISC-V issue count for a whole plan. The number the optimiser is
 * trying to push down. A replay LOAD costs one REPLAY instruction plus
 * the words fed in to be captured; a replay RUN costs just the one
 * REPLAY instruction, which is exactly where the saving lives. */
uint32_t tm_cost(const tm_planop_t *plan, uint32_t n_ops);

/* =================================================================
 *  The optimiser.
 * ================================================================= */

/* What the optimiser hands back: a plan, the naive all-direct cost it
 * started from, the cost it got down to, and whether the plan actually
 * verified against the target (it always should; if it ever does not,
 * the optimiser falls back to the naive plan and says so). */
typedef struct {
    tm_planop_t *plan;        /* arena-owned                          */
    uint32_t     n_ops;
    uint32_t     cost_naive;  /* every word emitted directly          */
    uint32_t     cost_opt;    /* the plan's RISC-V issue count         */
    int          verified;    /* plan reproduces target word for word  */
    const char  *note;        /* what the pass did, in plain words     */
} tm_optresult_t;

/* Take a target Tensix stream and find a cheaper plan that reproduces
 * it exactly. This first pass does replay-based repeat factoring: it
 * finds the contiguous block whose repeats save the most RISC-V
 * issues, loads it into the replay buffer once (executing the first
 * occurrence as it loads), and replays it at every later occurrence.
 * Everything not covered is emitted directly. The chosen plan is run
 * through the verifier before it is returned; an optimiser that
 * produces a faster-but-wrong stream is worse than useless.
 *
 * MOP fitting and multi-pattern covering (segment DP) are the next
 * passes; both drop into this same find-then-verify shape. */
tm_optresult_t tm_optimise(const tm_word_t *target, uint32_t n, ka_arena_t *A);

/* Same, but bounded to `max_slots` of replay buffer. This is the
 * bounty's thread-sharing constraint: on Wormhole/Blackhole the Math
 * and SFPU threads share one 16+16 buffer, so a Math-thread suggestion
 * that grabs all 32 slots would clash with SFPU. Pass the slots this
 * thread actually owns (e.g. 16) and the coverer will never propose a
 * plan that overruns them. tm_optimise is this with max_slots =
 * TM_REPLAY_SLOTS. */
tm_optresult_t tm_optimise_budget(const tm_word_t *target, uint32_t n,
                                  uint32_t max_slots, ka_arena_t *A);

/* =================================================================
 *  The MOP lever.
 * ================================================================= */

/* Build a [CFG_DOUBLE, MOP_RUN] fragment that emits `op` exactly
 * `*covered` times, where *covered is the largest outer*inner product
 * (outer in [1,32], inner in [1,64]) that does not exceed `count`. The
 * double-loop config is the simple one: start/end NOP (skipped), no
 * alternation, the op on every step. Writes up to two plan ops into
 * `out` (which must hold at least 2) and returns the number written.
 *
 * Returns 0 (writing nothing) when it is not worth it: a NOP op (the
 * trace would be empty), or a count whose best coverage does not clear
 * the MOP setup cost. This is the lever replay cannot pull: a length-1
 * repeat has a negative replay saving, but one MOP run reproduces up to
 * 2048 copies. */
int tm_fit_single_run(tm_word_t op, uint32_t count,
                      tm_planop_t *out, uint32_t *covered);

/* Optimise by folding maximal runs of one repeated instruction into MOP
 * double-loops. Walks the stream, and for each run long enough to beat
 * the MOP setup cost, emits one programmed double-loop plus a direct
 * tail for whatever the loop bounds could not cover; everything else is
 * emitted directly. Verifier-gated like tm_optimise; falls back to the
 * all-direct plan if nothing helps or a plan fails to reproduce the
 * target. Complementary to tm_optimise: that one pulls the replay lever
 * for repeated multi-instruction blocks, this one pulls the MOP lever
 * for high-count single-instruction runs. Unifying the two into one
 * per-segment cost bake-off is the next pass. */
tm_optresult_t tm_optimise_mop(const tm_word_t *target, uint32_t n,
                               ka_arena_t *A);

#endif /* TTMOP_H */
