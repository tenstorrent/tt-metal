// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/pack_block.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_pack_untilize.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm_reuse_dest_srcb.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"

#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_sdpa_bcast_col_srcb_reuse_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_sdpa_bcast_col_srca_srcb_reuse_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_sdpa_api.h"
#endif
#ifdef TRISC_PACK
#include "../../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

// #43563 / tt-blaze#842 (aho/sdpa) reduce-sum WAR-handshake fix. =1 makes the end-of-chunk
// reduce_sum->next-matmul FPU_SFPU handshake UNCONDITIONAL (+ posts FPU_SFPU on the first_chunk
// else-branch so it balances). MEASURED: does NOT reduce the pos8190 iter-alternation (6337/7168
// unchanged with dvalid ON), so left =0. Set =1 to re-enable.
#define FIX_AHO842_43563 0

// #43563 EXP I/O L1 TAP (this investigation): capture, as REAL packed L1 values (zero-flag honored,
// like the LMS tap), the exp INPUT (mm1 - max, i.e. the bcast_sub output RIGHT BEFORE fast_approx_exp)
// and the exp OUTPUT (probs, RIGHT AFTER fast_approx_exp), on CHUNK INDEX==1 only, on every active SDPA
// core. Both are packed into the side CB cb_iter1_dump (threaded in as an extra arg to
// compute_sdpa_chunk) at its BASE tiles: tile 0 = EXP_IN, tile 1 = EXP_OUT. Each is the first score
// tile @ mm1_dst_offset (16 packed rows). The host (test_decoder_block.py) reads these back per-core
// from mla_iter_dump_buffer (cb_iter1_dump base == L1 tile offset 48) into per_core_EXPIN / per_core_
// EXPOUT and a separate iter1-vs-iter2 diff localizes whether the bank-dependence is INTRODUCED at exp
// (EXP_IN identical, EXP_OUT diverges) or INHERITED (EXP_IN diverges). The packs read already-settled
// DEST tiles (MATH stalled on FPU_SFPU) and use cb_iter1_dump (untouched by production), so DEST
// lifetime / handshakes / semaphores are unchanged. Mutually exclusive with SDPA_LMS_TAP_43563 (both
// share cb_iter1_dump's base). 0 = off (default). 1 = enabled.
#define SDPA_EXPIO_TAP_43563 0

// #43563 MM1-DIRECT L1 TAP (this investigation): CONFIRM (not infer from mm1-max) whether the raw QK^T
// matmul output (mm1) itself is bank-dependent iter1-vs-iter2 at pos8190. Capture, as REAL packed L1
// values (zero-flag honored, like the EXP I/O tap), on CHUNK INDEX==1 only, on every active SDPA core:
//   TILE 0 = MM1_RAW : the QK^T matmul output @ mm1_dst_offset, captured RIGHT AFTER sdpa_custom_mm_block
//            and BEFORE reduce_max + bcast_sub touch it (the raw scores).
//   TILE 1 = MAXTILE : the max tile @ max_dst_offset, captured AFTER reduce_max and RIGHT BEFORE
//            bcast_sub consumes it (the max as applied).
// Both packed into the side CB cb_iter1_dump (base == mla_iter_dump_buffer L1 tile offset 48). Host
// (test_decoder_block.py) reads them back per-core and a separate iter1-vs-iter2 diff says whether mm1
// itself diverges (=> matmul/unpack Src carrier) or only the applied max diverges (=> bcast_sub).
// RACE AVOIDANCE: mm1's NEXT consumer is reduce_max (SFPU). We MATH-stall on FPU_SFPU so the matmul is
// settled in DEST, then STALL_PACK on WAIT_SFPU so the packer reads a settled tile, and finally
// STALL_SFPU on PACK so reduce_max (SFPU) cannot start mutating mm1 until the MM1_RAW pack has drained.
// This mirrors the EXP_IN tap's anti-race barrier. Mutually exclusive with SDPA_LMS_TAP / SDPA_EXPIO
// (share cb_iter1_dump base). 0 = off (default). 1 = enabled.
#define SDPA_MM1DIRECT_TAP_43563 1

// #43562/3 BALANCED DUAL-MOV TAP (this investigation). Pairs with QK_DUALMOV_43563 in
// llk_math_sdpa_custom_mm.h: that gated MATH op replaces the QK^T MVMUL with MOVA2D(SrcA=Q)->mm1 tile0
// and MOVB2D(SrcB=K)->mm1 tile1 (the adjacent mm1 region tile, mm1_dst_offset + packed_tile_size), with
// balanced dvalid consumption. RIGHT AFTER the matmul (before reduce_max/bcast_sub touch mm1) we pack
// BOTH tiles into cb_iter1_dump as REAL L1 values: TILE 0 = the Q-readout (SrcA), TILE 1 = the K-readout
// (SrcB). Host (test_decoder_block.py) reads them back per-core into per_core_QDUMP / per_core_KDUMP and
// diffs iter1-vs-iter2 SEPARATELY to localize WHICH Src carries the pos8190 bank-dependence. Same anti-
// race barrier as the MM1DIRECT tap (MATH-stall on FPU_SFPU so the MOVs are settled; STALL_PACK on
// WAIT_SFPU so the packer reads settled tiles; STALL_SFPU on PACK after so reduce_max cannot mutate mm1
// before the packs drain). chunk index==1 only. Reads DEST only; no semaphore change. Mutually exclusive
// with the other cb_iter1_dump taps. 0 = off (default). 1 = enabled.
// NOTE: never produced data -- its partner QK_DUALMOV_43563 deadlocks on device (see that macro's note).
#define SDPA_DUALMOV_TAP_43563 0

// #43563 REAL FIX: fully-define the corr_exp MOVD2B source window. ROOT CAUSE (HW-confirmed): the
// rescale preamble (_llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_, llk_math_sdpa_bcast_col_srca_
// srcb_reuse.h:73-74) issues two MOVD2B(MOV_4_ROWS) at row offsets 0 and 4 => it reads PHYSICAL DEST
// rows 0..7 of the corr_exp tile into SrcB, then the bcast-mul broadcasts col0 of each row into mm2
// (the online-softmax rescale mm2 *= exp(prev_max - cur_max)). MOVD2B reads DEST PHYSICALLY (raw
// Dst16b[row][col], ignoring zero-flags - ISA MOVD2B.md). But non_approx_exp_mul_prev (SFPU) writes
// the correction ONLY into sfpi dst_reg[0] (=> physical row 0, top-4 query rows) and dst_reg[2]
// (=> physical row 4, bottom-4 query rows; SFP_DESTREG_STRIDE=2 so dst_reg[ix] -> physical row ix*2).
// corr_exp physical rows {1,2,3,5,6,7} are NEVER written => undefined bank-dependent leftover, which
// MOVD2B then ships into SrcB col0 -> bank-dependent mm2 -> the iter-parity alternation. (DUMP at
// pos8190 chunk-1 confirmed rows 6,7 col0 diverge iter0-vs-iter1 while rows 0,4 are identical.)
//
// FIX (=1): write the mathematically-correct per-query-row correction into ALL of rows 0..7. The
// reduce stores the top-4 query rows' max in LREG0 (-> corr_exp row 0) and the bottom-4 in LREG2
// (-> corr_exp row 4); the bcast-mul applies corr_exp row r's col0 to mm2 row r, so every row in the
// top sub-tile {0,1,2,3} must carry the top correction (exp_top_4 - 1) and every row in the bottom
// sub-tile {4,5,6,7} the bottom correction (exp_bottom_4 - 1). The "-1" matches the bcast-mul's
// acc_to_dest form (mm2 = mm2*(corr) + mm2 = mm2*exp). Replicating the SAME value the two real rows
// already hold into the leftover rows keeps the rescale mathematically EXACT (not an approximation):
// it is the coherent-full-tile-rewrite that the copy_tile MS fix relied on. REAL SFPU datum write
// (sfpi dst_reg, not a zero-flag) -> iteration-independent by construction. 0 = off (kernel
// unchanged). Defaulted OFF; set 1 to enable the fix.
#define FIX_MOVD2B_SRC_43563 0

// #43563 MM1 (raw QK^T) DEST DUMP. When DUMP_MM1_43563=1, dump the raw mm1 (QK^T scores) region
// AFTER the QK matmul and BEFORE the bcast_sub modifies it (mm1 = mm1 - max), restricted to the
// FIRST chunk only. mm1 are the raw scores (can be negative, magnitude ~tens). We also dump the max
// tile (base=max_dst_offset) for cross-reference: if mm1 is bit-identical bank0-vs-bank1 but max
// differs, the matmul is clean and the SFPU reduce is bank-dependent; if mm1 differs, the matmul is
// bank-dependent. Uses the SAME stall/halt/per-row-float16 reader idiom as DUMP_MS_43563 in
// flash_mla.hpp (MATH-stall on FPU_SFPU so the matmul/reduce producer is settled, dbg_halt, read
// absolute DEST rows via dprint_tensix_dest_reg_row_float16(DataFormat::Float16_b, row), dbg_unhalt).
// DEST is Float16_b. 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_MM1_43563 0

// #43563 reduce-SUM DEST DUMP. When DUMP_SUM_43563=1, dump the SFPU reduce-SUM result region at the
// END of compute_sdpa_chunk (AFTER llk_math_sfpu_sdpa_reduce_sum_row writes sum_dst_offset), restricted
// to the FIRST chunk only. In a SINGLE-CHUNK config (first_chunk==last_chunk==true) this isolates the
// single-chunk reduce-SUM. Mirrors the DUMP_MM1 stall/halt/per-row-float16 idiom (MATH-stall on
// FPU_SFPU so the SFPU producer is settled, dbg_halt, read absolute DEST rows via
// dprint_tensix_dest_reg_row_float16(DataFormat::Float16_b, row), dbg_unhalt). DEST is Float16_b.
// 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_SUM_43563 0

// #43563 SDPA CHAIN DUMP. When DUMP_CHAIN_43563=1, dump the per-stage SDPA results EACH internal
// iteration, tagged by an internal iter_counter that PERSISTS across the decoder's internal
// iterations (function-static, incremented once per SDPA invocation at the mm1 dump on first_chunk).
// Stages dumped here in compute_sdpa_chunk (FIRST chunk only): mm1 (raw QK^T after matmul, before
// bcast_sub), max (after reduce_max), sum (after reduce_sum, end of chunk). The tailout stage is
// dumped in flash_mla.hpp after the cross-core tail reduction. Reuses the proven DPRINT DEST-dump
// idiom (MATH stall on FPU_SFPU, dbg_halt, per-row float16 reader, dbg_unhalt). DEST is Float16_b.
// Lines are greppable: "CHAIN iter=<n> stage=<mm1|max|sum|tailout> row=<r> : <floats>". 0 = off.
#define DUMP_CHAIN_43563 0

// #43563 QK-INPUT DUMP. When DUMP_QKIN_43563=1, dump the INPUTS to the QK^T matmul (cb_k and cb_q),
// the incoming mm1 DEST leftover BEFORE the matmul (mm1_pre), and the raw QK^T output AFTER the matmul
// (mm1_post), restricted to the FIRST chunk only, tagged with the persistent per-internal-iteration
// counter. Purpose: on the diverging cores (chips 6&7, cores (3,1)&(3,3)) the QK^T mm1 output differs
// between internal iteration 0 and 1; this dump resolves whether the matmul's K/Q INPUT is iteration-
// dependent (bug upstream) or whether the matmul produces iter-dependent output from IDENTICAL inputs
// (bug inside the matmul/LLK; reads iter-dependent DEST/SrcB state).
//   - cb_k / cb_q: dumped on UNPACK via TSLICE (UNPACK holds the input-CB read pointer; TSLICE on MATH
//     returns no data, on UNPACK it reads the rd_ptr-relative tile). Placed AFTER cb_wait_front(cb_k)
//     and BEFORE sdpa_custom_mm_block / cb_pop_front. Tag: "QKIN iter=<n> cbk tile=<t> ..." / "... cbq ...".
//   - mm1_pre: rows mm1_dst_offset..+15 read via the per-row float16 reader BEFORE the matmul
//     (incoming leftover the matmul's ZEROACC acts on). Tag: "QKIN iter=<n> mm1_pre row=<r> : ...".
//   - mm1_post: rows mm1_dst_offset..+15 read AFTER sdpa_custom_mm_block (raw QK^T). Tag:
//     "QKIN iter=<n> mm1_post row=<r> : ...".
// Reuses the proven DEST-dump idiom (MATH stall on FPU_SFPU, dbg_halt, per-row float16 reader,
// dbg_unhalt; DEST is Float16_b) and the persistent chain_iter_counter_43563(). 0 = off. Defaulted OFF.
#define DUMP_QKIN_43563 0

// #43563 TAIL-INPUT PROBE. When PROBE_TAILIN_43563=1, in sdpa_tail_ms_reduce, AFTER the two copy_tile
// loads (cb_prev_ms->dst0, cb_worker_ms->dst1) and BEFORE fused_max_sub_exp_add_tile, dump the tail
// INPUT that the cross-core tree reduce is about to consume: dst tile0 (prev) abs rows 0..1 and dst
// tile1 (worker) abs rows 32..33 (the max@col0/sum@col1 the reduce reads). Tag with the persistent
// per-internal-iteration counter (chain_iter_counter_43563), incremented ONCE per invocation here
// (coordinate with DUMP_CHAIN being OFF so this owns the increment). Lines:
// "TAILIN iter=<n> which=<prev|worker> row=<r> : <f16>". Reuses the MATH-stall/halt/per-row-float16
// idiom. 0 = off. Defaulted OFF.
#define PROBE_TAILIN_43563 0

// #43563 TAIL-OUTPUT PROBE. When PROBE_TAILOUT_43563=1, in sdpa_tail_ms_reduce, AFTER
// fused_max_sub_exp_add_tile (the existing tailout location), dump the COMBINED (tree-reduced) cur_ms:
// dst tile2 combined max @ abs row 64 (col0) and combined sum @ abs row 65 (col1), in the !normalize
// branch. Tag: "TAILOUT iter=<n> row=<r> : <f16>". 0 = off. Defaulted OFF.
#define PROBE_TAILOUT_43563 0

// #43563 race-vs-state test: hard all-thread delay/barrier right before the QK^T matmul (1000 NOPs on
// EACH of MATH/UNPACK/PACK). Maximally perturbs the matmul's timing vs every other thread. If the
// iters=1-vs-2 output PCC is unchanged with this on, the bug is timing-INDEPENDENT (deterministic state,
// not a race). 0 = off.
#define BARRIER_QK_43563 0

// #43563 iter-parity REAL-fix candidate: stall MATH until PACK idle before the QK matmul. 1=on.
#define FIX_MATH_WAIT_PACK_43563 0

// #43563 MM2 PER-CHUNK CHAIN DUMP. When DUMP_MM2_CHAIN_43563=1, dump the carried mm2 (PV) DEST
// accumulator on EVERY chunk (not just first_chunk), tagged with the persistent per-internal-iteration
// counter (chain_iter_counter_43563), a per-invocation chunk index (function-static, reset on
// first_chunk), and a stage. Two stages:
//   stage=rescale : right AFTER the !first_chunk rescale correction (mm2 *= exp(prev_max-cur_max);
//                   the sdpa_mul_bcast_col_srca_srcb_reuse_tiles<...,true>(mm2_dst_offset) at ~611).
//                   Only emitted on chunks>=1.
//   stage=pv      : right AFTER the PV accumulate (sdpa_custom_mm_reuse_dest_srcb_block writes
//                   mm2_dst_offset, ~643). Emitted every chunk.
// Dumps a few rows of the mm2 tile (rows mm2_dst_offset+0/+1). Goal: find the FIRST (chunk,stage)
// where internal iter0's mm2 != iter1's mm2. Reuses the proven MATH-stall/halt/per-row-float16 idiom
// (DEST is Float16_b). Lines: "MM2 iter=<n> chunk=<c> stage=<rescale|pv> row=<r> : <f16>". 0 = off.
#define DUMP_MM2_CHAIN_43563 0

// #43563 PV-OPERAND (exp-probs SrcB) DUMP. When DUMP_PROBS_43563=1, dump, RESTRICTED TO chunk index==1
// (the first !first_chunk), the operands of the chunk-1 PV accumulate (mm2 += probs@V):
//   - PROBS: ALL 16 rows of the exp-probs tile at base mm1_dst_offset (the tile MOVD2B reads as SrcB
//            for the PV matmul) RIGHT BEFORE sdpa_custom_mm_reuse_dest_srcb_block. Includes the valid
//            prob rows AND the leftover/non-prob rows so we can see if leftover is bank-dependent.
//            Tag: "PROBS iter=<n> row=<r> : <f16>".
//   - MM2BASE: carried mm2 (rows 0,1 @ base mm2_dst_offset) immediately BEFORE the PV (re-confirm
//              iter0==iter1). Tag: "MM2BASE iter=<n> row=<r> : <f16>".
//   - MM2POST: mm2 (rows 0,1) immediately AFTER the PV accumulate. Tag: "MM2POST iter=<n> row=<r> : <f16>".
// Uses a function-static per-invocation chunk index (reset on first_chunk, ++ at end) and the persistent
// per-internal-iteration counter (chain_iter_counter_43563, ++ on first_chunk; iter tag = counter-1).
// Reuses the proven MATH-stall(FPU_SFPU)/dbg_halt/per-row-float16/dbg_unhalt idiom. DEST is Float16_b.
// 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_PROBS_43563 0

// #43563 MM1+MAX DUMP. When DUMP_MM1MAX_43563=1, dump, RESTRICTED TO chunk index==1 (the first
// !first_chunk), the upstream operands of the chunk-1 exp chain, to climb one level above PROBS:
//   - MM1: the raw QK^T matmul output tile at base mm1_dst_offset, ALL 16 rows, dumped RIGHT AFTER the
//          chunk-1 QK matmul (sdpa_custom_mm_block) and BEFORE reduce_max/bcast-sub.
//          Tag: "MM1 iter=<n> row=<r> : <f16>".
//   - MAX: the running/current max operand used by the chunk-1 bcast-sub, AFTER reduce_max's cross-chunk
//          combine (!first_chunk), rows 0-7 @ base max_dst_offset, BEFORE bcast-sub consumes it.
//          Tag: "MAX iter=<n> row=<r> : <f16>".
//   - SUBEXP: the (scores - max) result in mm1 AFTER bcast-sub, BEFORE exp, rows 0-7.
//          Tag: "SUBEXP iter=<n> row=<r> : <f16>".
// Uses a function-static per-invocation chunk index (reset on first_chunk, ++ at end) and the persistent
// per-internal-iteration counter (chain_iter_counter_43563, ++ on first_chunk; iter tag = counter-1).
// Reuses the proven MATH-stall(FPU_SFPU)/dbg_halt/per-row-float16/dbg_unhalt idiom. DEST is Float16_b.
// 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_MM1MAX_43563 0

// #43563 MM1BASE DUMP. When DUMP_MM1BASE_43563=1, dump the mm1 DEST region the chunk-1 QK^T matmul is
// ABOUT TO accumulate onto (the "accumulation base"), RESTRICTED to chunk index==1 (the first
// !first_chunk). The non-mask QK path's ZEROACC CLR_16 is FLAG-ONLY (sets zero-flags, leaves physical
// data), so the physical base == the leftover already in those DEST rows just BEFORE the matmul. We
// therefore dump mm1_dst_offset..+15 RIGHT BEFORE sdpa_custom_mm_block, at a RACE-FREE point: MATH is
// idle here (we just did t6_semaphore_wait_on_max(FPU_SFPU)) and NO matmul/unpacker handshake is in
// flight, so dbg_halt does not desync SrcB (a halt BETWEEN the ZEROACC and the MVMUL deadlocks the
// SrcB unpacker handshake — confirmed). The dprint per-row reader reads PHYSICAL DEST (ignores
// zero-flags, like the SFPU), so MM1BASE shows the bank-dependent physical residue regardless of the
// (later, flag-only) ZEROACC. MM1OUT (post-matmul) is also dumped to confirm this core diverges under
// instrumentation. Tags: "MM1BASE iter=<n> row=<r> : <f16>" / "MM1OUT iter=<n> row=<r> : <f16>". 0=off.
//
// RESULT (pos8190, chip6 (x=0,y=0), niters=2, MASKDEST=3/CFG=0/AHO842=0): MM1BASE ALL 16 rows
// IDENTICAL iter0-vs-iter1 (non-zero PHYSICAL residue, reader ignored zero-flags), while MM1OUT
// DIVERGED on 15/16 rows. => The accumulation base is CLEAN; the chunk-1 mm1 divergence is NOT
// accumulation-onto-bank-dependent-leftover. Deeper puzzle. Gated OFF in tree.
#define DUMP_MM1BASE_43563 0

// #43563 MOVD2B-SOURCE DUMP. When DUMP_MOVD2BSRC_43563=1, dump, RESTRICTED TO chunk index==1 (the
// first !first_chunk), the EXACT DEST rows the two SrcB-from-DEST MOVD2B loads in compute_sdpa_chunk
// read into SrcB, to test the unifying hypothesis that MOVD2B ships undefined/flag-zeroed (and thus
// bank-dependent) DEST rows into SrcB. Two sources:
//   - CORREXP: the corr_exp tile at corr_exp_dst_offset. The rescale preamble
//     (_llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_) issues two MOVD2B(MOV_4_ROWS) at row
//     offsets 0 and 4 => it reads PHYSICAL rows 0..7. non_approx_exp_mul_prev (SFPU) only writes
//     sfpi dst_reg[0] and dst_reg[2]; the remaining physical rows in 0..7 are NOT written on the
//     !first_chunk path. We dump rows 0..7 (the read window) + rows 8..15 (context) RIGHT BEFORE the
//     rescale preamble (~sdpa.h:799). Tag: "CORREXP iter=<n> row=<r> : <f16>".
//   - PVPROBS: the exp-probs tile at mm1_dst_offset. The PV matmul preamble
//     (_llk_math_sdpa_custom_mm_reuse_dest_srcb_) issues four MOVD2B(MOV_4_ROWS) at offsets 0,4,8,12
//     => it reads PHYSICAL rows 0..15. fast_approx_exp writes the probs. We dump all 16 rows RIGHT
//     BEFORE the PV matmul (~sdpa.h:867). Tag: "PVPROBS iter=<n> row=<r> : <f16>".
// Uses a function-static per-invocation chunk index (reset on first_chunk, ++ at end) and the
// persistent per-internal-iteration counter (chain_iter_counter_43563, ++ on first_chunk; iter tag =
// counter-1). Reuses the proven MATH-stall(FPU_SFPU)/dbg_halt/per-row-float16/dbg_unhalt idiom; the
// per-row reader reads PHYSICAL DEST (ignores zero-flags, exactly like the SFPU) so undefined rows
// show their bank-dependent residue. DEST is Float16_b. 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_MOVD2BSRC_43563 0

#if defined(COMPILE_FOR_TRISC) &&                                                                                    \
    ((defined(DUMP_MM1_43563) && DUMP_MM1_43563) || (defined(DUMP_SUM_43563) && DUMP_SUM_43563) ||                   \
     (defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563) || (defined(DUMP_QKIN_43563) && DUMP_QKIN_43563) ||             \
     (defined(PROBE_TAILIN_43563) && PROBE_TAILIN_43563) || (defined(PROBE_TAILOUT_43563) && PROBE_TAILOUT_43563) || \
     (defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563) || (defined(DUMP_PROBS_43563) && DUMP_PROBS_43563) ||   \
     (defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563) || (defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563) ||     \
     (defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563))
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#endif

#if defined(COMPILE_FOR_TRISC) && defined(DUMP_QKIN_43563) && DUMP_QKIN_43563
// TSLICE (CB-tile DPRINT) for cb_k / cb_q. Only emits data on UNPACK (input-CB read pointer).
#include "api/debug/dprint_tile.h"
#endif

#if defined(COMPILE_FOR_TRISC) &&                                                                                    \
    ((defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563) || (defined(DUMP_QKIN_43563) && DUMP_QKIN_43563) ||             \
     (defined(PROBE_TAILIN_43563) && PROBE_TAILIN_43563) || (defined(PROBE_TAILOUT_43563) && PROBE_TAILOUT_43563) || \
     (defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563) || (defined(DUMP_PROBS_43563) && DUMP_PROBS_43563) ||   \
     (defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563) || (defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563) ||     \
     (defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563))
namespace ckernel {
// Persists across the decoder's internal iterations within ONE kernel execution. Incremented once
// per SDPA invocation (on first_chunk). Read by the chain mm1/max/sum/tailout dumps and the QKIN dump.
inline uint32_t& chain_iter_counter_43563() {
    static uint32_t iter_counter = 0;
    return iter_counter;
}
}  // namespace ckernel
#endif

namespace ckernel {

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles, bool dense = false>
ALWI void sdpa_bcast_col_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_sdpa_init<num_tiles, BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(
        false, false, icb0)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY, dense>(
        icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_reuse_preamble() {
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>()));
}

ALWI void sdpa_bcast_col_reuse_postamble() { MATH((llk_math_sdpa_bcast_col_srcb_reuse_postamble())); }

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles>
ALWI void sdpa_bcast_col_reuse_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in0_cb_id, in_tile_index)));
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in1_cb_id, in_tile_index)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse<eltwise_binary_type, num_tiles, DST_ACCUM_MODE, MATH_FIDELITY>(
        dst_tile_index)));
}

template <uint32_t num_tiles, bool dense = false>
ALWI void sdpa_mul_bcast_col_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_reuse_tiles_init<ELWMUL, num_tiles, dense>(icb0);
}

template <uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_reuse_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    sdpa_bcast_col_reuse_tiles<ELWMUL, num_tiles>(in0_cb_id, in1_cb_id, in_tile_index, dst_tile_index);
}

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY>(
        icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_preamble(uint32_t isrc) {
    UNPACK((llk_unpack_A_sdpa_set_srca_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>(isrc)));
}

template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    uint32_t num_tiles,
    bool skip_signalling = false,
    bool fused_signalling = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse<
          eltwise_binary_type,
          num_tiles,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          skip_signalling,
          fused_signalling>(dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<ELWSUB, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool skip_signalling = false, bool fused_signalling = false>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<ELWSUB, num_tiles, skip_signalling, fused_signalling>(dst_tile_index);
}

template <uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool skip_signalling = false, bool fused_signalling = false>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<ELWMUL, num_tiles, skip_signalling, fused_signalling>(dst_tile_index);
}

template <DataFormat format>
ALWI void sdpa_reduce_row_init() {
    MATH((llk_math_sfpu_sdpa_reduce_row_init<APPROX, DST_ACCUM_MODE, format>()));
}

template <DataFormat format, uint32_t block_width>
ALWI void sdpa_reduce_max_row(uint src_index, uint dst_index, bool prev_max = false) {
    MATH((llk_math_sfpu_sdpa_reduce_max_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_max)));
}

template <DataFormat format, uint32_t block_width>
ALWI void sdpa_reduce_sum_row(uint src_index, uint dst_index, bool prev_sum = false) {
    MATH((llk_math_sfpu_sdpa_reduce_sum_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_sum)));
}

#ifdef TRISC_PACK
// =============================================================================
// #43563 FIX: SFPU literal-zero DEST init.
//
// Root cause (HW-confirmed): the SFPU reduce_max/reduce_sum read strided DEST
// rows that the QK^T matmul did NOT write. Those rows were only "zeroed" via the
// FPU/packer ZERO-FLAG path, which merely SETS a zero-flag and leaves stale data.
// The Blackhole SFPU does NOT honour zero-flags, so it reads stale, DEST-BANK-
// DEPENDENT leftover -> bank-dependent max/sum -> the #43563 iter-parity alternation.
//
// Fix: physically write REAL literal-0 data to every DEST row the SDPA can address
// using SFPU stores (sfpi dst_reg = 0), BEFORE the QK^T matmul. The matmul then
// overwrites the valid mm1 rows; any unwritten leftover row the reduce later reads
// is a genuine, bank-independent 0.
//
// IMPORTANT: this is an SFPU datum write (sfpi::dst_reg[i] = 0 -> SFPSTORE), NOT a
// zero-flag clear. It uses the SAME DEST-base idiom as fast_approx_exp /
// non_approx_exp_mul_prev above (TT_SETC16 DEST_TARGET_REG_CFG_MATH_Offset_ADDR32).
//
// num_rows = number of DEST rows to zero starting at row 0. The tiny-tile (8x32)
// SDPA addresses DEST up to mm1_dst_offset + chunk_size*packed_tile_size rows; the
// caller passes a count that covers the whole addressed region (when in doubt, the
// whole DEST half).
inline void sfpu_zero_dest_43563(uint32_t num_rows) {
    // Write 16-row tile-slots at a time: set the DEST base for each slot, then store
    // literal 0 into all 16 rows (dst_reg[0..15]). This matches the row granularity the
    // reduce addresses (ZERO_ADDR_MOD offsets 0..14 within a tile face).
    constexpr uint32_t rows_per_slot = 16;
    for (uint32_t base = 0; base < num_rows; base += rows_per_slot) {
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base + get_dest_buffer_base());
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (uint32_t r = 0; r < rows_per_slot; r++) {
            sfpi::dst_reg[r] = 0.0f;  // SFPU SFPSTORE of literal 0 (REAL data, not a zero-flag)
        }
    }
}

// #43563 MAX/SUM SPLIT STUB helper. Overwrite ONLY the 2 SFPU datums the reduce stores
// for a single per-core max-or-sum result with a deterministic finite constant. The reduce
// (ckernel_sfpu_sdpa_reduce_row.h) writes its result with the DEST base set to dst_index and
// then SFPSTORE LREG0 @ row 0, SFPSTORE LREG2 @ row 4 (ZERO_ADDR_MOD offsets {0,4}). So to
// stomp the max (or sum) we set the same DEST base (dst_offset + get_dest_buffer_base()) and
// SFPSTORE a constant into dst_reg[0] and dst_reg[4]. This is a REAL SFPU datum write, hence
// iteration-independent by construction. Same DEST-base idiom as sfpu_zero_dest_43563 above.
inline void sfpu_const_subtile_43563(uint32_t dst_offset, float c) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_offset + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    sfpi::dst_reg[0] = c;  // SFPSTORE LREG @ row 0  (matches reduce's first store)
    sfpi::dst_reg[4] = c;  // SFPSTORE LREG @ row 4  (matches reduce's second store)
}

// #43563 candidate FIX helper. The per-core SFPU reduce writes REAL data only into rows {0,4}
// (max, base=max_dst_offset) and {2,6} (sum, base=max_dst_offset, i.e. {0,4} rel sum_dst_offset).
// Rows {1,3,5,7,8..15} of the 16-row MS tile are NEVER written -> they hold bank-dependent leftover,
// which is then packed wholesale (pack_block_contiguous of the full tile) into sdpa_ms_cb and consumed
// by the cross-core tail reduce -> the #43563 iter-parity alternation.
//
// This helper SFPU-stores literal 0 into ONLY the "junk" rows of the MS tile based at tile_base
// (== max_dst_offset), while LEAVING the real datums {0,2,4,6} untouched. REAL SFPU store
// (sfpi dst_reg), not a zero-flag clear -> iteration-independent by construction.
//
// mode: 2 = zero ALL 16 rows (control: reproduces the copy_tile full-tile fix via SFPU; output
//          garbles since the real max/sum are also zeroed).
//       1 = zero the candidate junk set {1,3,5,7,8..15} (preserve {0,2,4,6}).
//       3 = zero ONLY the upper half {8..15} (preserve the whole lower half {0..7}); safer
//          diagnostic to find which rows actually carry the leftover the tail consumes.
inline void sfpu_zero_ms_junk_43563(uint32_t tile_base, uint32_t mode) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, tile_base + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    if (mode == 2) {
        for (uint32_t r = 0; r < 16; r++) {
            sfpi::dst_reg[r] = 0.0f;  // SFPSTORE literal 0 into every row (control)
        }
    } else if (mode == 3) {
        for (uint32_t r = 8; r < 16; r++) {
            sfpi::dst_reg[r] = 0.0f;  // upper half only
        }
    } else {
        // Candidate junk set: everything except {0,2,4,6} (max@{0,4}, sum@{2,6}).
        sfpi::dst_reg[1] = 0.0f;
        sfpi::dst_reg[3] = 0.0f;
        sfpi::dst_reg[5] = 0.0f;
        sfpi::dst_reg[7] = 0.0f;
        sfpi::dst_reg[8] = 0.0f;
        sfpi::dst_reg[9] = 0.0f;
        sfpi::dst_reg[10] = 0.0f;
        sfpi::dst_reg[11] = 0.0f;
        sfpi::dst_reg[12] = 0.0f;
        sfpi::dst_reg[13] = 0.0f;
        sfpi::dst_reg[14] = 0.0f;
        sfpi::dst_reg[15] = 0.0f;
    }
}

// Packer:
// Fast Approx Exp uses 3 constants and LoadMacro
// Non-Approx Exp uses 1 constant for recip. TODO: Look into integrating new polynomial exp in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp

// TODO: Factor this out into a reusable fn in LLK
template <uint32_t scale /* 1.0f in FP32 */>
inline void init_fast_approx_exp_constants() {
    constexpr float LN2_RECIP = 1.4426950408889634f;
    constexpr float A = 256.0f * LN2_RECIP;
    constexpr float B_minus_C = 32500.818359375f;
    constexpr float THRESHOLD = -88.5f;

    constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

    constexpr float A_scaled = A * scale_fp32;
    constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(THRESHOLD_scaled));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(THRESHOLD_scaled));
    TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(A_scaled));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(A_scaled));
    TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(B_minus_C));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(B_minus_C));
    TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3
}

inline void fast_approx_exp(uint32_t dst_index) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::calculate_exponential<true, DST_ACCUM_MODE, true, 8, true>();
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, uint16_t scale_bf16>
inline void non_approx_exp_mul_prev(uint32_t curr_sum_index, uint32_t corr_exp_index) {
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, corr_exp_index + get_dest_buffer_base());
    sfpi::vFloat prev_max_top_4 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat prev_max_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg3];
    sfpi::vFloat curr_max_top_4 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat curr_max_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat sub_top_4 = prev_max_top_4 - curr_max_top_4;
    sfpi::vFloat sub_bottom_4 = prev_max_bottom_4 - curr_max_bottom_4;
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();
    sfpi::vFloat exp_top_4 =
        sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
            sub_top_4, scale_bf16);
    sfpi::vFloat exp_bottom_4 =
        sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
            sub_bottom_4, scale_bf16);
    // Subtract 1. This is because the bcast mul accumulates to dest
    // Without -1: bcast = prev * exp + prev
    // With -1: bcast = prev * (exp - 1) + prev = prev * exp - prev + prev = prev * exp
    sfpi::vFloat corr_top = exp_top_4 - 1.0f;
    sfpi::vFloat corr_bottom = exp_bottom_4 - 1.0f;
    // DEST base is already corr_exp_index (set above). dst_reg[ix] -> physical row ix*2
    // (SFP_DESTREG_STRIDE=2). Real correction: row 0 = top-4 query rows, row 4 = bottom-4.
    dst_reg[0] = corr_top;     // physical row 0
    dst_reg[2] = corr_bottom;  // physical row 4
#if FIX_MOVD2B_SRC_43563
    // #43563 REAL FIX: the rescale preamble's MOVD2B reads PHYSICAL corr_exp rows 0..7 into SrcB and
    // the bcast-mul applies row r's col0 to mm2 row r. Rows {1,2,3,5,6,7} are otherwise UNDEFINED
    // bank-dependent leftover. Replicate the per-sub-tile correction the two real rows hold into the
    // remaining even rows (2,6) of THIS base, then offset the DEST base by +1 row to reach the odd
    // rows (1,3,5,7) - dst_reg can only address even physical rows from a given base. Mathematically
    // exact: every top-sub-tile row gets exp_top_4-1, every bottom-sub-tile row gets exp_bottom_4-1.
    dst_reg[1] = corr_top;     // physical row 2 (top sub-tile)
    dst_reg[3] = corr_bottom;  // physical row 6 (bottom sub-tile)
    // Odd physical rows {1,3,5,7}: shift the DEST base by 1 row.
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, corr_exp_index + 1 + get_dest_buffer_base());
    dst_reg[0] = corr_top;     // physical row 1 (top sub-tile)
    dst_reg[1] = corr_top;     // physical row 3 (top sub-tile)
    dst_reg[2] = corr_bottom;  // physical row 5 (bottom sub-tile)
    dst_reg[3] = corr_bottom;  // physical row 7 (bottom sub-tile)
    // Restore DEST base for the curr_sum write below.
#endif
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, curr_sum_index + get_dest_buffer_base());
    // Load Curr Sum Values
    sfpi::vFloat curr_sum_top_4 = dst_reg[0];
    sfpi::vFloat curr_sum_bottom_4 = dst_reg[2];
    sfpi::vFloat mul_top_4 = curr_sum_top_4 * exp_top_4;
    sfpi::vFloat mul_bottom_4 = curr_sum_bottom_4 * exp_bottom_4;
    dst_reg[0] = mul_top_4;
    dst_reg[2] = mul_bottom_4;
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, uint16_t scale_bf16>
inline void recip_sum(uint32_t curr_sum_index, uint32_t recip_dst_index) {
    // Last op should already be sum offset
    sfpi::vFloat sum_top_4 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat sum_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg2];
    // Init after to avoid trampling cached registers before we use them
    // TODO: Putting the prev regs in the upper regs lets us init ahead of time
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();
    sfpi::vFloat recip_top_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_top_4);
    sfpi::vFloat recip_bottom_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_bottom_4);

    // Subtract 1. This is because the bcast mul accumulates to dest
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, recip_dst_index + get_dest_buffer_base());
    dst_reg[0] = recip_top_4 - 1.0f;
    dst_reg[2] = recip_bottom_4 - 1.0f;
}
#endif

// First chunk controls whether we run the correction path with prev sum, max, out
// Last chunk controls whether we signal out packer to start packing as output is produced
//
// output_granularity controls how often the QK^T*V matmul (sdpa_custom_mm_reuse_dest_srcb_block)
// signals the packer via the FPU->SFPU semaphore. The packer must consume tiles in matching
// groups of output_granularity. num_tiles_v must be divisible by output_granularity.
template <
    uint32_t chunk_size,
    uint32_t num_tiles_k,
    uint32_t num_tiles_v,
    uint32_t scale_fp32,
    uint16_t scale_bf16,
    bool transpose_k,
    bool transpose_v,
    uint32_t packed_tile_size,
    bool exp_approx_mode = false,
    uint32_t output_granularity = 1,
    bool mm_pack_init = true>
void compute_sdpa_chunk(
    uint32_t cb_q,
    uint32_t cb_k,
    uint32_t cb_mask,
    uint32_t cb_out,
    uint32_t mm1_dst_offset,
    uint32_t mm2_dst_offset,
    uint32_t max_dst_offset,
    uint32_t sum_dst_offset,
    uint32_t corr_exp_dst_offset,
    bool first_chunk,
    bool last_chunk,
    bool mask_chunk,
    uint32_t cb_iter1_dump = 0xFFFFFFFF /* #43563 EXP I/O TAP side CB; unused unless SDPA_EXPIO_TAP_43563 */) {
    static_assert(DST_ACCUM_MODE == false, "FP32 destination accumulation mode is not supported");
    static_assert(num_tiles_v % output_granularity == 0, "num_tiles_v must be divisible by output_granularity");
#if defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563
    // #43563 MM2 CHAIN: establish the per-internal-iteration tag and the per-invocation chunk index.
    // The persistent iter counter increments ONCE per SDPA invocation (on first_chunk). The chunk
    // index resets to 0 on first_chunk and increments at the END of this function (below).
    static uint32_t mm2_chain_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        mm2_chain_chunk_43563 = 0;
    }
    uint32_t mm2_chain_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(DUMP_PROBS_43563) && DUMP_PROBS_43563
    // #43563 PROBS: per-invocation chunk index (reset on first_chunk, ++ at end) + persistent iter tag.
    // The iter counter increments ONCE per SDPA invocation (on first_chunk); iter tag = counter-1.
    static uint32_t probs_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        probs_chunk_43563 = 0;
    }
    uint32_t probs_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563
    // #43563 MM1MAX: per-invocation chunk index (reset on first_chunk, ++ at end) + persistent iter tag.
    // The iter counter increments ONCE per SDPA invocation (on first_chunk); iter tag = counter-1.
    static uint32_t mm1max_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        mm1max_chunk_43563 = 0;
    }
    uint32_t mm1max_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(DUMP_QKIN_43563) && DUMP_QKIN_43563
    // #43563 QKIN (pos8190 chunk-1 input dump): per-invocation chunk index (reset on first_chunk,
    // ++ at end) + persistent iter tag. The iter counter increments ONCE per SDPA invocation (on
    // first_chunk); iter tag = counter-1. Restricts the cb_k/cb_q + mm1 dumps to chunk index==1.
    static uint32_t qkin_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        qkin_chunk_43563 = 0;
    }
    uint32_t qkin_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563
    // #43563 MM1BASE: per-invocation chunk index (reset on first_chunk, ++ at end) + persistent iter tag.
    // The iter counter increments ONCE per SDPA invocation (on first_chunk); iter tag = counter-1.
    static uint32_t mm1base_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        mm1base_chunk_43563 = 0;
    }
    uint32_t mm1base_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563
    // #43563 MOVD2BSRC: per-invocation chunk index (reset on first_chunk, ++ at end) + persistent iter
    // tag. The iter counter increments ONCE per SDPA invocation (on first_chunk); iter tag = counter-1.
    static uint32_t movd2bsrc_chunk_43563 = 0;
    if (first_chunk) {
        ckernel::chain_iter_counter_43563()++;
        movd2bsrc_chunk_43563 = 0;
    }
    uint32_t movd2bsrc_iter = ckernel::chain_iter_counter_43563() - 1;
#endif
#if defined(SDPA_EXPIO_TAP_43563) && SDPA_EXPIO_TAP_43563
    // #43563 EXP I/O TAP: per-invocation chunk index (reset on first_chunk, ++ at end). We capture only
    // chunk index==1 (the chunk the L/MS divergence was localized to). No iter tag needed: each
    // num_internal_iterations run writes its own mla_iter_dump_buffer / .pt; the host diffs iter1-vs-iter2.
    static uint32_t expio_chunk_43563 = 0;
    if (first_chunk) {
        expio_chunk_43563 = 0;
    }
#endif
#if defined(SDPA_MM1DIRECT_TAP_43563) && SDPA_MM1DIRECT_TAP_43563
    // #43563 MM1-DIRECT TAP: per-invocation chunk index (reset on first_chunk, ++ at end). We capture only
    // chunk index==1 (the chunk the mm1-max divergence was localized to). No iter tag needed: each
    // num_internal_iterations run writes its own mla_iter_dump_buffer / .pt; the host diffs iter1-vs-iter2.
    static uint32_t mm1direct_chunk_43563 = 0;
    if (first_chunk) {
        mm1direct_chunk_43563 = 0;
    }
#endif
#if defined(SDPA_DUALMOV_TAP_43563) && SDPA_DUALMOV_TAP_43563
    // #43562/3 DUAL-MOV TAP: per-invocation chunk index (reset on first_chunk, ++ at end). Capture only
    // chunk index==1 (the chunk the divergence was localized to). Host diffs iter1-vs-iter2 per .pt.
    static uint32_t dualmov_chunk_43563 = 0;
    if (first_chunk) {
        dualmov_chunk_43563 = 0;
    }
#endif
    PACK((ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_()));
    sdpa_custom_mm_block_init_short<transpose_k, mm_pack_init>(cb_q, cb_k, cb_out, chunk_size);
    cb_wait_front(cb_k, num_tiles_k * chunk_size);
    // Q @ K (FPU)
    // Make sure SFPU of previous chunk is done (sem is zero)
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
#if defined(SFPU_ZERO_DEST_43563) && (SFPU_ZERO_DEST_43563 == 2)
    // #43563 FIX variant 2: per-chunk SFPU literal-zero of the mm1 region the reduce reads,
    // immediately BEFORE this chunk's QK^T matmul (SFPU idle here; FPU_SFPU just hit max). Writes
    // REAL 0 (sfpi dst_reg) so the reduce's strided reads of rows the matmul doesn't write are
    // bank-independent 0 on EVERY chunk (the pre-loop one-shot zero is re-dirtied between chunks).
    PACK((ckernel::sfpu_zero_dest_43563(mm1_dst_offset + chunk_size * 16)));
#endif
#if defined(DUMP_QKIN_43563) && DUMP_QKIN_43563
    // #43563 QK-INPUT DUMP (pos8190): CHUNK INDEX==1 ONLY. Dump the QK^T matmul INPUTS (the K tiles in
    // cb_k and the Q tiles in cb_q the unpacker reads for chunk-1) to decide whether chunk-1's mm1
    // divergence (iter0-vs-iter1) is input-driven (upstream KV-cache/Q) or DEST-accumulation-driven.
    // (1) cb_k / cb_q tiles. UNPACK holds the input-CB read pointer; TSLICE emits no data on MATH/PACK,
    // on UNPACK it samples the rd_ptr-relative tile. cb_k is waited-front above (chunk-1's K tiles are
    // at the rd_ptr front); cb_q is filled by its producer before the matmul. Dump several K + Q tiles.
    if (qkin_chunk_43563 == 1) {
        UNPACK(DPRINT << "CBK iter=" << qkin_iter << " tile=0 : " << TSLICE(cb_k, 0, SliceRange::hw0_32_8()) << ENDL());
        UNPACK(DPRINT << "CBK iter=" << qkin_iter << " tile=1 : " << TSLICE(cb_k, 1, SliceRange::hw0_32_8()) << ENDL());
        UNPACK(DPRINT << "CBK iter=" << qkin_iter << " tile=2 : " << TSLICE(cb_k, 2, SliceRange::hw0_32_8()) << ENDL());
        UNPACK(DPRINT << "CBK iter=" << qkin_iter << " tile=3 : " << TSLICE(cb_k, 3, SliceRange::hw0_32_8()) << ENDL());
        UNPACK(DPRINT << "CBQ iter=" << qkin_iter << " tile=0 : " << TSLICE(cb_q, 0, SliceRange::hw0_32_8()) << ENDL());
        UNPACK(DPRINT << "CBQ iter=" << qkin_iter << " tile=1 : " << TSLICE(cb_q, 1, SliceRange::hw0_32_8()) << ENDL());
    }
#endif
#if defined(BARRIER_QK_43563) && BARRIER_QK_43563
    // #43563 race narrowing: per-thread delay right BEFORE the QK^T matmul. BARRIER_QK_43563 selects
    // which thread(s) to delay: 1=MATH only, 2=UNPACK only, 3=PACK only, 4=all three.
#if (BARRIER_QK_43563 == 1) || (BARRIER_QK_43563 == 4)
    for (uint32_t _bq = 0; _bq < 1000; _bq++) {
        MATH(TTI_NOP);
    }
#endif
#if (BARRIER_QK_43563 == 2) || (BARRIER_QK_43563 == 4)
    for (uint32_t _bq = 0; _bq < 1000; _bq++) {
        UNPACK(TTI_NOP);
    }
#endif
#if (BARRIER_QK_43563 == 3) || (BARRIER_QK_43563 == 4)
    for (uint32_t _bq = 0; _bq < 1000; _bq++) {
        PACK(TTI_NOP);
    }
#endif
#endif
#if defined(FIX_MATH_WAIT_PACK_43563) && FIX_MATH_WAIT_PACK_43563
    // #43563 iter-parity candidate REAL fix (replaces the 1000-NOP stopgap): stall MATH until the PACK
    // thread is idle before the QK^T matmul writes DEST -> MATH cannot start the new matmul while PACK
    // is still draining the PREVIOUS DEST/MS (the custom pack handshake lacks a MATH-side wait-for-PACK).
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK)));
#endif
#if defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563
    // #43563 MM1BASE: chunk index==1 ONLY, RIGHT BEFORE the QK^T matmul. The DEST rows mm1_dst_offset..+15
    // hold the leftover the matmul is about to accumulate onto — the non-mask ZEROACC inside the matmul
    // is FLAG-ONLY, so this physical leftover IS the accumulation base. RACE-FREE point: MATH is idle
    // (FPU_SFPU just hit max, no matmul/unpacker handshake in flight), so dbg_halt is safe here (a halt
    // BETWEEN the in-matmul ZEROACC and the MVMUL deadlocks the SrcB unpacker). The per-row float16
    // reader reads PHYSICAL DEST (ignores zero-flags, like the SFPU) -> shows bank-dependent residue.
    if (mm1base_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "MM1BASE iter=" << mm1base_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
    sdpa_custom_mm_block<transpose_k>(cb_q, cb_k, cb_mask, 0, 0, mm1_dst_offset, num_tiles_k, chunk_size, mask_chunk);
#if defined(SDPA_MM1DIRECT_TAP_43563) && SDPA_MM1DIRECT_TAP_43563
    // #43563 MM1_RAW L1 TAP: chunk index==1 ONLY, RIGHT AFTER the QK^T matmul and BEFORE reduce_max /
    // bcast_sub touch mm1. Pack the first score tile @ mm1_dst_offset (16 packed rows) into cb_iter1_dump
    // TILE 0 as a REAL L1 value. MATH-stall on FPU_SFPU so the FPU matmul is settled in DEST; STALL_PACK
    // on WAIT_SFPU so the packer reads a settled tile. CRITICAL anti-race: mm1's next consumer is
    // reduce_max (SFPU), so AFTER the pack we STALL_SFPU on PACK to keep the SFPU reduce from mutating
    // mm1 before the pack drains (mirrors the EXP_IN tap's barrier). Reads DEST only; no semaphore change.
    if (mm1direct_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous_init(cb_iter1_dump);
        cb_reserve_back(cb_iter1_dump, 2);
        // pack_block_contiguous wants a STANDARD DEST tile slot (0..15); mm1_dst_offset is a packed
        // DEST *row* offset, so convert via packed_tile_size (== 16) to the tile slot.
        pack_block_contiguous(mm1_dst_offset / packed_tile_size, cb_iter1_dump, 1);  // MM1_RAW -> dump tile 0
        // Stall the SFPU pipe until the packer is done so reduce_max (SFPU) cannot mutate mm1 before the
        // MM1_RAW pack reliably captures the settled raw QK^T scores.
        PACK(TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK));
    }
#endif
#if defined(SDPA_DUALMOV_TAP_43563) && SDPA_DUALMOV_TAP_43563
    // #43562/3 DUAL-MOV TAP: chunk index==1 ONLY, RIGHT AFTER the QK dual-MOV (QK_DUALMOV_43563) and
    // BEFORE reduce_max / bcast_sub touch mm1. The dual-MOV wrote Q (SrcA) into mm1 tile0 and K (SrcB)
    // into mm1 tile1 (mm1_dst_offset + packed_tile_size). Pack BOTH as REAL L1 values: tile0 (Q) ->
    // cb_iter1_dump TILE 0, tile1 (K) -> cb_iter1_dump TILE 1. Same anti-race barrier as MM1DIRECT:
    // MATH-stall on FPU_SFPU so the MOVs are settled; STALL_PACK on WAIT_SFPU so the packer reads
    // settled tiles; after both packs, STALL_SFPU on PACK so reduce_max (SFPU) cannot mutate mm1 before
    // the packs drain. Reads DEST only; no semaphore change. Re-arm cb_out packer init for production.
    if (dualmov_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous_init(cb_iter1_dump);
        cb_reserve_back(cb_iter1_dump, 2);
        pack_block_contiguous(mm1_dst_offset / packed_tile_size, cb_iter1_dump, 1);      // QDUMP (SrcA) -> tile 0
        pack_block_contiguous(mm1_dst_offset / packed_tile_size + 1, cb_iter1_dump, 1);  // KDUMP (SrcB) -> tile 1
        cb_push_back(cb_iter1_dump, 2);
        PACK(TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK));
        // Re-arm the production packer MOP for cb_out (the normal output pack does not re-init).
        pack_block_contiguous_init(cb_out);
    }
#endif
#if defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563
    // #43563 MM1OUT: chunk index==1 ONLY, RIGHT AFTER the QK^T matmul. Confirms (cross-check) that this
    // core diverges iter0-vs-iter1 in mm1 under the same instrumentation. MATH-stall on FPU_SFPU so the
    // matmul is settled, halt, per-row float16 read of all 16 rows @ mm1_dst_offset (Float16_b), unhalt.
    if (mm1base_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "MM1OUT iter=" << mm1base_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563
    // #43563 MM1: chunk index==1 ONLY, RIGHT AFTER the QK^T matmul and BEFORE reduce_max/bcast-sub, so
    // mm1 holds the raw QK^T scores. MATH-stall on FPU_SFPU so the matmul is settled in DEST, halt,
    // per-row float16 read (DEST is Float16_b) of all 16 rows @ mm1_dst_offset, unhalt.
    if (mm1max_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "MM1 iter=" << mm1max_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_QKIN_43563) && DUMP_QKIN_43563
    // #43563 QK-INPUT DUMP (mm1_post): raw QK^T scores AFTER the chunk-1 matmul (the known-diverging
    // quantity), dumped alongside the chunk-1 inputs to correlate and to CONFIRM the dumped core
    // actually shows mm1 divergence at pos8190. CHUNK INDEX==1 ONLY. MATH-stall on FPU_SFPU so the
    // matmul is settled, halt, read abs DEST rows mm1_dst_offset..+15 (Float16_b), unhalt.
    if (qkin_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "QKMM1 iter=" << qkin_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif

    // Reduce Max (SFPU)
    PACK((llk_math_sfpu_sdpa_reduce_max_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size>(
        mm1_dst_offset, max_dst_offset, !first_chunk)));
#if defined(DUMP_MM1_43563) && DUMP_MM1_43563
    // #43563 MM1 (raw QK^T) DUMP. FIRST chunk only. We are AFTER the QK matmul + reduce_max (which only
    // READS mm1) and BEFORE bcast_sub (which would do mm1 = mm1 - max), so mm1 still holds the raw QK^T
    // scores. MATH-stall on FPU_SFPU so both the matmul and the reduce are settled in DEST, then halt and
    // read absolute DEST rows with the per-row float16 reader (DEST is Float16_b; the packed tiny tile is
    // 16 rows -> use the per-row reader, NOT dprint_tensix_dest_reg(tile_id)). Dump the first mm1 tile's
    // 16 rows (mm1_dst_offset .. +15) and the max tile's 8 rows (max_dst_offset .. +7) for cross-ref.
    if (first_chunk) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        DPRINT << "MM1DUMP core begin mm1_dst_offset=" << (uint32_t)mm1_dst_offset
               << " max_dst_offset=" << (uint32_t)max_dst_offset << ENDL();
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "MM1DUMP mm1 f16 row " << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
            for (uint16_t r = 0; r < 8; r++) {
                DPRINT << "MM1DUMP max f16 row " << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
            }
        })
        dbg_unhalt();
        DPRINT << "MM1DUMP core end" << ENDL();
    }
#endif
#if defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563
    // #43563 CHAIN DUMP stages mm1 + max. FIRST chunk only. We are AFTER the QK matmul + reduce_max
    // (which only READS mm1) and BEFORE bcast_sub, so mm1 still holds the raw QK^T scores. We
    // increment the persistent iter_counter ONCE here (the start of each SDPA invocation) and tag
    // every stage line with it. MATH-stall on FPU_SFPU so the matmul and reduce are settled, halt,
    // read absolute DEST rows via the per-row float16 reader (DEST is Float16_b), unhalt.
    if (first_chunk) {
        uint32_t chain_iter = ckernel::chain_iter_counter_43563();
        ckernel::chain_iter_counter_43563()++;
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "CHAIN iter=" << chain_iter << " stage=mm1 row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
            for (uint16_t r = 0; r < 8; r++) {
                DPRINT << "CHAIN iter=" << chain_iter << " stage=max row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563
    // #43563 MAX: chunk index==1 ONLY, AFTER reduce_max (incl. the !first_chunk cross-chunk combine) and
    // BEFORE bcast-sub consumes it. This is the running max operand fed to the chunk-1 bcast-sub.
    // MATH-stall on FPU_SFPU so the SFPU reduce is settled in DEST, halt, per-row float16 read of rows
    // 0-7 @ max_dst_offset (DEST is Float16_b), unhalt.
    if (mm1max_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 8; r++) {
                DPRINT << "MAX iter=" << mm1max_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(SDPA_MM1DIRECT_TAP_43563) && SDPA_MM1DIRECT_TAP_43563
    // #43563 MAXTILE L1 TAP: chunk index==1 ONLY, AFTER reduce_max (incl. the !first_chunk cross-chunk
    // combine) and RIGHT BEFORE bcast_sub consumes it. This is the max operand as applied to mm1. Pack
    // the max tile @ max_dst_offset into cb_iter1_dump TILE 1 as a REAL L1 value, then push both tiles.
    // MATH-stall on FPU_SFPU so the SFPU reduce is settled in DEST; STALL_PACK on WAIT_SFPU so the packer
    // reads a settled tile. The pack reads DEST only; no semaphore change. (Max should be bank-invariant.)
    if (mm1direct_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous(max_dst_offset / packed_tile_size, cb_iter1_dump, 1);  // MAXTILE -> dump tile 1
        cb_push_back(cb_iter1_dump, 2);
        // Re-arm the packer MOP for the production output CB (cb_out). The normal (non-tap) output pack
        // in flash_mla.hpp does NOT re-init, so we MUST restore it here.
        pack_block_contiguous_init(cb_out);
    }
#endif
    // Bcast Sub (FPU)
    // Wait for SFPU to finish (sem is 0)
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init<chunk_size>(cb_q);  // For tile shape
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(max_dst_offset);
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles<chunk_size, false>(mm1_dst_offset);
#if defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563
    // #43563 SUBEXP: chunk index==1 ONLY, AFTER bcast-sub (mm1 = mm1 - max) and BEFORE exp consumes it.
    // This is the (scores - max) operand feeding exp. MATH-stall on FPU_SFPU so the FPU bcast-sub is
    // settled in DEST, halt, per-row float16 read of rows 0-7 @ mm1_dst_offset (DEST is Float16_b), unhalt.
    if (mm1max_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 8; r++) {
                DPRINT << "SUBEXP iter=" << mm1max_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
    if (!first_chunk) {
        // Exp Sub (SFPU)
        // Signal FPU that tile is ready
        // This should just init an lreg constant and is what's needed for non-approx exp
        PACK((non_approx_exp_mul_prev<exp_approx_mode, scale_bf16>(sum_dst_offset, corr_exp_dst_offset)));
        PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
        // Bcast Mul (FPU)
        // Wait for SFPU that tile is ready (sem is non-zero)
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
        MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
#if defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563
        // #43563 CORREXP: chunk index==1 ONLY, RIGHT BEFORE the rescale preamble's MOVD2B reads
        // corr_exp into SrcB. The preamble MOVD2Bs read PHYSICAL rows 0..7; non_approx_exp_mul_prev
        // (SFPU) wrote only sfpi dst_reg[0]/dst_reg[2]. We are at a race-free point (MATH just did
        // wait_on_zero(SFPU_FPU), so the SFPU producer is settled, and no MOVD2B/unpacker handshake is
        // in flight). dbg_halt, per-row float16 read of rows 0..15 (the MOVD2B window 0..7 + context
        // 8..15), unhalt. The reader reads PHYSICAL DEST so undefined rows show bank-dependent residue.
        if (movd2bsrc_chunk_43563 == 1) {
            dbg_halt();
            MATH({
                for (uint16_t r = 0; r < 16; r++) {
                    DPRINT << "CORREXP iter=" << movd2bsrc_iter << " row=" << (uint32_t)r << " : ";
                    dprint_tensix_dest_reg_row_float16(
                        (uint32_t)DataFormat::Float16_b, (uint16_t)(corr_exp_dst_offset + r));
                }
            })
            dbg_unhalt();
        }
#endif
        sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset);
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, true>(mm2_dst_offset);
        // FPU has consumed the tile
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
        // Reset to 0
        // No stall since we stalled math already
        MATH((t6_semaphore_get<p_stall::NONE>(SFPU_FPU)));
#if defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563
        // #43563 MM2 CHAIN stage=rescale: AFTER the !first_chunk rescale correction
        // (mm2 *= exp(prev_max - cur_max)). MATH-stall on FPU_SFPU so the bcast-mul is settled in
        // DEST, halt, read abs mm2 DEST rows (Float16_b) via the per-row reader, unhalt.
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 2; r++) {
                DPRINT << "MM2 iter=" << mm2_chain_iter << " chunk=" << mm2_chain_chunk_43563
                       << " stage=rescale row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm2_dst_offset + r));
            }
        })
        dbg_unhalt();
#endif
    }
#if FIX_AHO842_43563
    else {
        // #43563 / tt-blaze#842 fix (aho/sdpa "Test fix race between first and second iter"):
        // on the first_chunk path the !first_chunk correction block (which posts FPU_SFPU) is
        // skipped, so the semaphore was left unposted -> the end-of-chunk reduce_sum->next-matmul
        // handshake had nothing to balance and was itself skipped, leaving the next iteration's QK
        // matmul unsynchronized with this chunk's reduce_sum (WAR hazard on mm1). Post here so the
        // unconditional handshake below can drain the SFPU before the next matmul writes mm1.
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
    }
#endif
#if defined(SDPA_EXPIO_TAP_43563) && SDPA_EXPIO_TAP_43563
    // #43563 EXP_IN L1 TAP: chunk index==1 ONLY, RIGHT BEFORE the exp loop. The bcast_sub (FPU) above
    // wrote (mm1 - max) into the score region; this is the exact tile fast_approx_exp consumes. Pack the
    // first score tile @ mm1_dst_offset (16 packed rows) into cb_iter1_dump TILE 0 as a REAL L1 value.
    // MATH-stall on FPU_SFPU so the FPU bcast_sub is settled in DEST, then STALL_PACK on WAIT_SFPU so the
    // packer reads a settled tile (same handshake the LMS tap uses). Reads DEST only; no semaphore change.
    if (expio_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous_init(cb_iter1_dump);
        cb_reserve_back(cb_iter1_dump, 2);
        // pack_block_contiguous wants a STANDARD DEST tile slot (0..15); mm1_dst_offset is a packed
        // DEST *row* offset, so convert via packed_tile_size (== 16) to the tile slot.
        pack_block_contiguous(mm1_dst_offset / packed_tile_size, cb_iter1_dump, 1);  // EXP_IN -> dump tile 0
        // CRITICAL: the exp loop below issues fast_approx_exp (SFPU) on the SAME PACK thread and OVERWRITES
        // this tile in place. Without a barrier the SFPU exp can run before the packer drains, so the
        // EXP_IN pack races and captures POST-exp probs (observed iter-dependently). Stall the SFPU pipe
        // until the packer is done so EXP_IN reliably captures the PRE-exp (mm1 - max) tile.
        PACK(TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK));
    }
#endif
    // Exp Mul Scale (SFPU)
    PACK((init_fast_approx_exp_constants<scale_fp32>()));
    for (uint32_t i = 0; i < chunk_size; i++) {
        // Wait for FPU that tile is ready (sem is non-zero)
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        // Each tile is 8x32, which is the same as a full 16x16 face
        PACK((fast_approx_exp(mm1_dst_offset + i * packed_tile_size)));
        PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
        // No stall since we waited on sfpu already
        PACK((t6_semaphore_post<p_stall::NONE>(SFPU_FPU)));
    }
#if defined(SDPA_EXPIO_TAP_43563) && SDPA_EXPIO_TAP_43563
    // #43563 EXP_OUT L1 TAP: chunk index==1 ONLY, RIGHT AFTER the exp loop. fast_approx_exp (SFPU) wrote
    // the probs into the score region @ mm1_dst_offset; this is the exact tile the PV matmul's MOVD2B
    // (rows 0..15) and reduce_sum (even rows) consume. Pack the first probs tile into cb_iter1_dump
    // TILE 1 as a REAL L1 value, then push both tiles. MATH-stall on FPU_SFPU so the SFPU exp is settled
    // in DEST; STALL_PACK on WAIT_SFPU so the packer reads a settled tile. Reads DEST only; no sem change.
    if (expio_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous(mm1_dst_offset / packed_tile_size, cb_iter1_dump, 1);  // EXP_OUT -> dump tile 1
        cb_push_back(cb_iter1_dump, 2);
        // Re-arm the packer MOP for the production output CB (cb_out == sdpa_output_cb). The normal
        // (non-tap) output pack in flash_mla.hpp does NOT re-init, so we MUST restore it here.
        pack_block_contiguous_init(cb_out);
    }
#endif

#if defined(DUMP_PROBS_43563) && DUMP_PROBS_43563
    // #43563 PROBS + MM2BASE: chunk index==1 ONLY, RIGHT BEFORE the PV matmul. PROBS = the exp-probs
    // tile at mm1_dst_offset (MOVD2B reads this as SrcB), ALL 16 rows incl. leftover. MM2BASE = the
    // carried mm2 (rows 0,1) about to be accumulated into. MATH-stall on FPU_SFPU so the exp (SFPU)
    // producer is settled in DEST, halt, per-row float16 read (DEST is Float16_b), unhalt.
    if (probs_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "PROBS iter=" << probs_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
            for (uint16_t r = 0; r < 2; r++) {
                DPRINT << "MM2BASE iter=" << probs_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm2_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563
    // #43563 PVPROBS: chunk index==1 ONLY, RIGHT BEFORE the PV matmul's MOVD2B reads the exp-probs
    // tile into SrcB. The PV preamble MOVD2Bs read PHYSICAL rows 0..15. fast_approx_exp (SFPU) wrote
    // the probs above. MATH-stall on FPU_SFPU so the exp SFPU producer is settled in DEST, halt,
    // per-row float16 read of all 16 rows @ mm1_dst_offset (the MOVD2B window), unhalt. The reader
    // reads PHYSICAL DEST so any undefined/leftover rows show their bank-dependent residue.
    if (movd2bsrc_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "PVPROBS iter=" << movd2bsrc_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm1_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
    // MM (FPU)
    sdpa_custom_mm_reuse_dest_srcb_block_init_short(cb_q, cb_k, cb_out, transpose_v, chunk_size, num_tiles_v);
    sdpa_custom_mm_reuse_dest_srcb_block<output_granularity>(
        cb_q,
        cb_k,
        0,
        0,
        mm1_dst_offset,
        mm2_dst_offset,
        transpose_v,
        chunk_size,
        num_tiles_v,
        num_tiles_k,
        last_chunk);
#if defined(DUMP_PROBS_43563) && DUMP_PROBS_43563
    // #43563 MM2POST: chunk index==1 ONLY, immediately AFTER the PV accumulate. MATH-stall on FPU_SFPU
    // so the PV matmul is settled in DEST, halt, per-row float16 read (DEST is Float16_b), unhalt.
    if (probs_chunk_43563 == 1) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 2; r++) {
                DPRINT << "MM2POST iter=" << probs_iter << " row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm2_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563
    // #43563 MM2 CHAIN stage=pv: AFTER the PV accumulate (mm2 += probs@V) for this chunk.
    // MATH-stall on FPU_SFPU so the PV matmul is settled in DEST, halt, read abs mm2 DEST rows
    // (Float16_b) via the per-row reader, unhalt. Emitted on every chunk.
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    dbg_halt();
    MATH({
        for (uint16_t r = 0; r < 2; r++) {
            DPRINT << "MM2 iter=" << mm2_chain_iter << " chunk=" << mm2_chain_chunk_43563
                   << " stage=pv row=" << (uint32_t)r << " : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm2_dst_offset + r));
        }
    })
    dbg_unhalt();
#endif

    // Reduce Sum (SFPU)
    PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
    PACK((llk_math_sfpu_sdpa_reduce_sum_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size, true>(
        mm1_dst_offset, sum_dst_offset, !first_chunk)));
    // Signal SFPU is done for the chunk
    // #43563 / tt-blaze#842 fix (aho/sdpa): when FIX_AHO842_43563=1, make this handshake UNCONDITIONAL
    // (was if(!first_chunk)). It drains the reduce_sum SFPU so the QK matmul of the NEXT chunk/iteration
    // cannot start writing mm1 before this chunk's reduce_sum has finished reading exp(QK) from it (the
    // WAR race). MEASURED not to help pos8190, so gated off (=0) -> restores the if(!first_chunk) guard.
#if FIX_AHO842_43563
    // Wait for FPU to signal (this doesn't block SFPU logic) -> decrement a non-zero semaphore.
    PACK((t6_semaphore_wait_on_zero<p_stall::NONE>(semaphore::FPU_SFPU)));
    // Signal SFPU is done (so QK MM can reuse the space in the next iteration).
    PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
#else
    if (!first_chunk) {
        // Wait for FPU to signal (this doesn't block SFPU logic) -> decrement a non-zero semaphore.
        PACK((t6_semaphore_wait_on_zero<p_stall::NONE>(semaphore::FPU_SFPU)));
        // Signal SFPU is done (so QK MM can reuse the space in the next iteration).
        PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
    }
#endif
#if defined(DUMP_SUM_43563) && DUMP_SUM_43563
    // #43563 reduce-SUM DUMP. FIRST chunk only. We are AFTER the SFPU reduce_sum_row that wrote the
    // per-core SUM result to DEST base sum_dst_offset (= max_dst_offset + 2). MATH-stall on FPU_SFPU so
    // the SFPU producer is settled in DEST, halt, and read absolute DEST rows with the per-row float16
    // reader (DEST is Float16_b). Dump the SUM tile's full 16 rows (sum_dst_offset .. +15) and re-dump
    // the max tile's full 16 rows (max_dst_offset .. +15) for cross-reference.
    if (first_chunk) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        DPRINT << "SUMDUMP core begin sum_dst_offset=" << (uint32_t)sum_dst_offset
               << " max_dst_offset=" << (uint32_t)max_dst_offset << ENDL();
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "SUMDUMP sum f16 row " << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(sum_dst_offset + r));
            }
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "SUMDUMP max f16 row " << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
            }
        })
        dbg_unhalt();
        DPRINT << "SUMDUMP core end" << ENDL();
    }
#endif
#if defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563
    // #43563 CHAIN DUMP stage sum. FIRST chunk only. AFTER the SFPU reduce_sum_row wrote the per-core
    // SUM result to DEST base sum_dst_offset. MATH-stall on FPU_SFPU so the SFPU producer is settled,
    // halt, read absolute DEST rows via the per-row float16 reader. Re-read the persisted iter_counter
    // (already incremented at the mm1 dump above) so this line groups with this invocation's block.
    if (first_chunk) {
        uint32_t chain_iter = ckernel::chain_iter_counter_43563() - 1;
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            for (uint16_t r = 0; r < 16; r++) {
                DPRINT << "CHAIN iter=" << chain_iter << " stage=sum row=" << (uint32_t)r << " : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(sum_dst_offset + r));
            }
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_MM2_CHAIN_43563) && DUMP_MM2_CHAIN_43563
    // #43563 MM2 CHAIN: advance the per-invocation chunk index for the next chunk in this invocation.
    mm2_chain_chunk_43563++;
#endif
#if defined(DUMP_PROBS_43563) && DUMP_PROBS_43563
    // #43563 PROBS: advance the per-invocation chunk index for the next chunk in this invocation.
    probs_chunk_43563++;
#endif
#if defined(DUMP_MM1MAX_43563) && DUMP_MM1MAX_43563
    // #43563 MM1MAX: advance the per-invocation chunk index for the next chunk in this invocation.
    mm1max_chunk_43563++;
#endif
#if defined(DUMP_QKIN_43563) && DUMP_QKIN_43563
    // #43563 QKIN: advance the per-invocation chunk index for the next chunk in this invocation.
    qkin_chunk_43563++;
#endif
#if defined(DUMP_MM1BASE_43563) && DUMP_MM1BASE_43563
    // #43563 MM1BASE: advance the per-invocation chunk index for the next chunk in this invocation.
    mm1base_chunk_43563++;
#endif
#if defined(DUMP_MOVD2BSRC_43563) && DUMP_MOVD2BSRC_43563
    // #43563 MOVD2BSRC: advance the per-invocation chunk index for the next chunk in this invocation.
    movd2bsrc_chunk_43563++;
#endif
#if defined(SDPA_EXPIO_TAP_43563) && SDPA_EXPIO_TAP_43563
    // #43563 EXP I/O TAP: advance the per-invocation chunk index for the next chunk in this invocation.
    expio_chunk_43563++;
#endif
#if defined(SDPA_MM1DIRECT_TAP_43563) && SDPA_MM1DIRECT_TAP_43563
    // #43563 MM1-DIRECT TAP: advance the per-invocation chunk index for the next chunk in this invocation.
    mm1direct_chunk_43563++;
#endif
#if defined(SDPA_DUALMOV_TAP_43563) && SDPA_DUALMOV_TAP_43563
    // #43562/3 DUAL-MOV TAP: advance the per-invocation chunk index for the next chunk in this invocation.
    dualmov_chunk_43563++;
#endif
    cb_pop_front(cb_k, num_tiles_k * chunk_size);
}

template <uint32_t num_tiles_v, bool exp_approx_mode, uint16_t scale_bf16>
void compute_sdpa_recip(uint32_t cb_q, uint32_t sum_dst_offset, uint32_t recip_dst_offset, uint32_t mm2_dst_offset) {
    PACK((recip_sum<exp_approx_mode, scale_bf16>(sum_dst_offset, recip_dst_offset)));
    PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
    MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(recip_dst_offset);
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, false, true>(mm2_dst_offset);
    MATH((t6_semaphore_get<p_stall::MATH>(SFPU_FPU)));
}

// =============================================================================
// SDPA Tail Reduction - Fused SFPI Kernel and Helper
// =============================================================================

#ifdef TRISC_MATH

/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * cur_sum = exp((worker_max - cur_max) * scale) * worker_sum + exp((prev_max - cur_max) * scale) * prev_sum
 * There are 4 results produced:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale), produced in dst_reg[worker_max_base_idx]
 * 3. cur_sum produced in dst_reg[prev_sum_base_idx]
 * 4. cur_max produced in dst_reg[cur_max_base_idx]
 * If final_norm is true, the output is:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[worker_max_base_idx]
 * fused_max_sub_exp_add_tile
 */
template <bool SDPA_EXP_APPROX_MODE, bool final_norm = false>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    // Non-Approx mode for exp initializes recip for final normalization
    static_assert(!(final_norm && SDPA_EXP_APPROX_MODE), "Approx mode must be disabled when final_norm is true");

    // 8 rows
    constexpr int ITERATIONS_HALF_FACE = 2;
    constexpr uint32_t prev_max_base_idx = 0;     // Tile 0, col 0
    constexpr uint32_t prev_sum_base_idx = 1;     // Tile 0, col 1
    constexpr uint32_t worker_max_base_idx = 32;  // Tile 1, col 0
    constexpr uint32_t worker_sum_base_idx = 33;  // Tile 1, col 1
    constexpr uint32_t cur_max_base_idx = 64;     // Tile 2, col 0 (output)
    constexpr uint32_t cur_sum_base_idx = 65;     // Tile 2, col 1 (output)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs for this vector-slot into temporaries to avoid aliasing on dst_reg
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat cur_max;
        v_if(prev_max_vec < worker_max_vec) { cur_max = worker_max_vec; }
        v_else { cur_max = prev_max_vec; }
        v_endif;
        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_max_base_idx] = cur_max;
        }

        // Compute differences
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        // Exponentials of differences
        sfpi::vFloat exp_prev =
            sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker =
            sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_worker, scale_bf16);

        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_sum_base_idx] = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::dst_reg[prev_max_base_idx] = exp_prev;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker;
        } else {
            sfpi::vFloat curr_sum = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::vFloat recip_sum = ckernel::sfpu::sfpu_reciprocal<SDPA_EXP_APPROX_MODE>(curr_sum);
            sfpi::dst_reg[prev_max_base_idx] = exp_prev * recip_sum;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker * recip_sum;
        }
        sfpi::dst_reg += 2;
    }
}

/**
 * Wrapper for fused max-sub-exp-add SFPI kernel.
 * Invokes calculate_fused_max_sub_exp_add_tile via LLK unary SFPU parameters.
 */
template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C, bool final_norm = false>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, final_norm>, idst, vector_mode, scale_bf16);
}
#endif

// =============================================================================
// SDPA Tail Helpers
// =============================================================================

/**
 * Helper 1: MS Reduction Phase
 *
 * Processes MS tiles to compute P1 and P2 scaling factors, sets up SRCB for
 * subsequent L tile broadcast multiply operations.
 *
 * After this call:
 *   - SRCB contains P1 (col 0) and P2 (col 1) ready for broadcast multiply
 *   - If normalize=false: MS output is packed to cb_cur_ms, tile_regs released
 *   - If normalize=true: tile_regs still held (caller can process first L block immediately)
 *
 * @param cb_worker_ms Worker MS tile (MS1) (max in col 0, sum in col 1)
 * @param cb_prev_ms Previous MS tile (MS2) (max in col 0, sum in col 1)
 * @param cb_cur_ms Output MS tile (only used when normalize=false)
 * @param cb_l_for_init CB used for sdpa_mul_bcast_col_reuse_tiles_init
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C,
    bool pop_ms = false,
    bool dense = false>
ALWI void sdpa_tail_ms_reduce(uint32_t cb_worker_ms, uint32_t cb_prev_ms, uint32_t cb_cur_ms, uint32_t cb_l_for_init) {
    copy_tile_to_dst_init_short(cb_worker_ms);
    cb_wait_front(cb_worker_ms, 1);
    cb_wait_front(cb_prev_ms, 1);
    constexpr uint32_t dst_reg_0 = 0;  // prev_ms
    constexpr uint32_t dst_reg_1 = 1;  // worker_ms
    constexpr uint32_t dst_reg_2 = 2;  // cur_ms output

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    tile_regs_acquire();
    copy_tile(cb_prev_ms, 0, dst_reg_0);
    copy_tile(cb_worker_ms, 0, dst_reg_1);
    if constexpr (pop_ms) {
        cb_pop_front(cb_prev_ms, 1);
        cb_pop_front(cb_worker_ms, 1);
    }
#if (defined(PROBE_TAILIN_43563) && PROBE_TAILIN_43563) || (defined(PROBE_TAILOUT_43563) && PROBE_TAILOUT_43563)
    // #43563 TAIL probes. Establish the persistent per-internal-iteration counter for this invocation.
    // When DUMP_CHAIN is NOT active it owns the increment here (once per sdpa_tail_ms_reduce call).
    // The TAILOUT dump below reads (counter-1) to tag the just-incremented invocation, matching the
    // existing CHAIN tailout idiom.
    uint32_t tail_iter = ckernel::chain_iter_counter_43563();
#if !(defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563)
    if constexpr (!normalize) {
        ckernel::chain_iter_counter_43563()++;
    }
#endif
#endif
#if defined(PROBE_TAILIN_43563) && PROBE_TAILIN_43563
    // #43563 TAIL-INPUT PROBE. AFTER copy_tile(cb_prev_ms->dst0) and copy_tile(cb_worker_ms->dst1) and
    // BEFORE fused_max_sub_exp_add_tile: dump the tail INPUT the cross-core reduce will consume.
    // dst tile0 (prev) holds max@col0/sum@col1 in abs rows 0..1; dst tile1 (worker) in abs rows 32..33.
    // Only the !normalize tree-reduce steps consume cross-core MS, so probe there. MATH-stall/halt idiom.
    if constexpr (!normalize) {
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            DPRINT << "TAILIN iter=" << tail_iter << " which=prev row=0 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)0);
            DPRINT << "TAILIN iter=" << tail_iter << " which=prev row=1 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)1);
            DPRINT << "TAILIN iter=" << tail_iter << " which=worker row=0 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)32);
            DPRINT << "TAILIN iter=" << tail_iter << " which=worker row=1 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)33);
        })
        dbg_unhalt();
    }
#endif
    MATH((fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, vector_mode, normalize>(0, scale_bf16)));
#if defined(PROBE_TAILOUT_43563) && PROBE_TAILOUT_43563
    // #43563 TAIL-OUTPUT PROBE. AFTER fused_max_sub_exp_add_tile: combined cur_ms in dst tile2,
    // combined max @ abs row 64 (col0), combined sum @ abs row 65 (col1), in the !normalize branch.
    if constexpr (!normalize) {
        uint32_t tailout_iter = tail_iter;  // already the per-invocation tag (counter incremented above)
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            DPRINT << "TAILOUT iter=" << tailout_iter << " row=0 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)64);
            DPRINT << "TAILOUT iter=" << tailout_iter << " row=1 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)65);
        })
        dbg_unhalt();
    }
#endif
#if defined(DUMP_CHAIN_43563) && DUMP_CHAIN_43563
    // #43563 CHAIN DUMP stage tailout. This runs on the OUTPUT core's cross-core tail reduction. The
    // fused kernel just wrote the COMBINED (tree-reduced) max into DEST cur_max_base_idx=64 (tile 2,
    // col0) and combined sum into cur_sum_base_idx=65 (tile 2, col1) for the !normalize reduction
    // steps. (When normalize=true the final norm path overwrites those with recip-scaled exp factors
    // and does NOT keep the raw combined max/sum, so we only dump the !normalize steps where the
    // combined MS is observable.) Tag with the persisted iter_counter (already incremented in the
    // per-core mm1 dump of THIS invocation). Read abs DEST rows 64..65 with the per-row float16 reader.
    if constexpr (!normalize) {
        uint32_t chain_iter = ckernel::chain_iter_counter_43563() - 1;
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
        dbg_halt();
        MATH({
            // dst_reg_2 == tile 2; combined cur_max @ abs row 64 (col0), cur_sum @ abs row 65 (col1).
            DPRINT << "CHAIN iter=" << chain_iter << " stage=tailout row=0 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)64);
            DPRINT << "CHAIN iter=" << chain_iter << " stage=tailout row=1 : ";
            dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)65);
        })
        dbg_unhalt();
    }
#endif
    // Initialize SRCB reuse for L tile broadcast multiply
    // TODO: Optimize init sequence with copy_tile
    sdpa_mul_bcast_col_reuse_tiles_init<block_size, dense>(cb_l_for_init);
    sdpa_bcast_col_reuse_preamble<normalize>();

    // Not final reduction: pack out stats and release regs
    if constexpr (!normalize) {
        PACK((llk_pack_mop_config<false, false>(cb_cur_ms)));
        tile_regs_commit();
        cb_reserve_back(cb_cur_ms, 1);
        tile_regs_wait();
        pack_tile(dst_reg_2, cb_cur_ms);
        cb_push_back(cb_cur_ms, 1);
        tile_regs_release();
    }
}

/**
 * Helper 2: Process single L block
 *
 * Processes one block of L tiles using P1/P2 already in SRCB from sdpa_tail_ms_reduce.
 * Caller is responsible for cb_wait_front/cb_reserve_back before and cb_push_back/cb_pop_front after.
 *
 * @param cb_l1 First L input CB
 * @param cb_l2 Second L input CB
 * @param cb_l_out Output L CB
 * @param tile_index Starting tile index within the CB (for current block)
 * @param acquire_regs Whether to acquire tile_regs (false if regs already held from MS phase)
 */
template <uint32_t block_size, uint32_t num_blocks, bool untilize = false, bool dense = false, bool manage_cbs = false>
ALWI void sdpa_tail_l_block(
    uint32_t cb_l1, uint32_t cb_l2, uint32_t cb_l_out, uint32_t tile_index, uint32_t block_index, bool acquire_regs) {
    if (acquire_regs) {
        tile_regs_acquire();
    }
    if constexpr (manage_cbs) {
        cb_wait_front(cb_l2, block_size);
        cb_wait_front(cb_l1, block_size);
    }
    sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, tile_index, 0);
    if constexpr (manage_cbs) {
        cb_pop_front(cb_l2, block_size);
        cb_pop_front(cb_l1, block_size);
        if constexpr (!untilize) {
            cb_reserve_back(cb_l_out, block_size);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    if constexpr (untilize) {
        pack_untilize_dest<block_size, block_size * num_blocks, false, false, TILE_C_DIM, 0, dense>(
            cb_l_out, 1, block_index, 8, dense ? 2 : 4);
    } else {
        pack_block_contiguous(0, cb_l_out, block_size);
    }
    if constexpr (manage_cbs) {
        if constexpr (!untilize) {
            cb_push_back(cb_l_out, block_size);
        }
    }
    tile_regs_release();
}

/**
 * Helper 3: Finalize SDPA tail
 *
 * Cleanup: calls postamble and pops MS input tiles.
 * Call this after all L blocks have been processed.
 *
 * @param cb_worker_ms Worker MS tile CB (to pop)
 * @param cb_prev_ms Previous MS tile CB (to pop)
 */
template <bool pop_ms = true>
ALWI void sdpa_tail_finalize(uint32_t cb_worker_ms, uint32_t cb_prev_ms) {
    sdpa_bcast_col_reuse_postamble();
    if constexpr (pop_ms) {
        cb_pop_front(cb_prev_ms, 1);
        cb_pop_front(cb_worker_ms, 1);
    }
}

// =============================================================================
// SDPA Tail - Main function (uses helpers internally)
// =============================================================================

/**
 * SDPA tail reduction combining fused SFPI kernel with srcB reuse broadcast multiply.
 *
 * Implements the following reduction:
 * 1. cb_m_out = max(cb_m2, cb_m1)
 * 2. cb_exp_diff_2 = exp((cb_m1 - cb_m_out) * scale)  [P1]
 * 3. cb_s1 *= cb_exp_diff_2  (s1 * P1)
 * 4. cb_exp_diff_1 = exp((cb_m2 - cb_m_out) * scale)  [P2]
 * 5. cb_s2 *= cb_exp_diff_1  (s2 * P2)
 * 6. cb_s_out = cb_s1 + cb_s2  (s1*P1 + s2*P2)
 * 7. cb_l_out = cb_l1 * P1 + cb_l2 * P2
 *
 * @param cb_worker_max_sum Worker MS tile (MS1) (max in col 0, sum in col 1)
 * @param cb_prev_max_sum Previous MS tile (MS2) (max in col 0, sum in col 1)
 * @param cb_cur_max_sum Output MS tile (only used when normalize=false)
 * @param cb_l1 Worker L tiles
 * @param cb_l2 Previous L tiles
 * @param cb_l_out Output L tiles
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t num_blocks,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C,
    bool dense = false,
    bool untilize = false>
ALWI void sdpa_tail(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    // Phase 1: MS reduction - computes P1/P2, sets up SRCB
    sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, vector_mode, true, dense>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // TODO: Update the tile locs in ms_reduce to enable dense packing during entire reduction
    if constexpr (dense && !untilize) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
    if constexpr (!untilize) {
        pack_block_contiguous_init(cb_l_out);
    }

    // Phase 2: Process all L blocks
    // Untilize requires operating on all blocks at once
    if constexpr (untilize) {
        custom_pack_untilize_dest_init<block_size, num_blocks * block_size, false, TILE_C_DIM, dense>(
            cb_l_out, 8, dense ? 2 : 4);
        cb_reserve_back(cb_l_out, block_size * num_blocks);
    }
    // When normalize=true, first block uses regs still held from MS phase
    if constexpr (normalize) {
        sdpa_tail_l_block<block_size, num_blocks, untilize, dense, true>(cb_l1, cb_l2, cb_l_out, 0, 0, false);
    }
    for (uint32_t i = (normalize ? 1 : 0); i < num_blocks; i++) {
        sdpa_tail_l_block<block_size, num_blocks, untilize, dense, true>(cb_l1, cb_l2, cb_l_out, 0, i, true);
    }
    if constexpr (untilize) {
        cb_push_back(cb_l_out, block_size * num_blocks);
        pack_untilize_uninit(cb_l_out);
    }

    if constexpr (dense && !untilize) {
        // Restore packing stride from tile to tile to 64 rows
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }

    // Phase 3: Finalize (postamble + pop MS)
    sdpa_tail_finalize<false>(cb_worker_max_sum, cb_prev_max_sum);
}

}  // namespace ckernel
