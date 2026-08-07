// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/transpose_wh.h"
#ifdef TRISC_MATH
// SFPU topk: call ckernel::sfpu functors directly via SFPU_UNARY_CALL (no per-op llk_api wrapper layer).
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "experimental/llk_sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#include "experimental/llk_math_generalized_moe_gate_eltwise_binary_api.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face_api.h"
#endif

namespace ckernel {

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void generalized_moe_gate_init(uint32_t icb0, uint32_t icb1) {
    if constexpr (enable_sigmoid) {
        // Init sigmoid (SFPU)
        sigmoid_tile_init<false>();
        // Init transpose wh (FPU)
        transpose_wh_init_short(icb0);
    } else {
        // Init copy add (FPU)
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::Both)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary_init_with_operands<
              EltwiseBinaryType::ELWADD,
              GeneralizedMoeGateEltwiseBinaryMode::COPY,
              MATH_FIDELITY>(icb0, icb1, false)));
        // Init transpose dest addrmods (does not conflict with copy add)
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX, DST_ACCUM_MODE))));
    }
}

// ---- Multi-block (>256) combine helpers ------------------------------------------------------
// Place one field (0=bias, 1=idx, 2=score) of a run sitting in the interm region at rows {0,2}
// (just unpacked from the L1 run stash via copy_tile) into its home region (bias/indices/scores)
// at rows {dst_lo,dst_hi}. Row-selective, so it drops a block's run at the {4,6} merge slot without
// disturbing the run already placed at {0,2}.
template <uint32_t field, uint32_t dst_lo, uint32_t dst_hi, uint32_t src_lo = 0, uint32_t src_hi = 2>
ALWI void generalized_moe_gate_place_field_from_interm() {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_place_field_from_interm,
        (APPROX, DST_ACCUM_MODE, field, src_lo, src_hi, dst_lo, dst_hi),
        0,
        VectorMode::RC_custom)));
}

// Finalize the combine: the two block runs sit at scores/idx/bias {0,2} and {4,6}; bitonically sort
// the 16 candidates -> global top-8 + normalize, then transpose-dest step2 to the output layout.
template <bool is_32bit = false, uint32_t topk = 8, bool output_softmax = false>
ALWI void generalized_moe_gate_combine_finalize(uint32_t eps, uint32_t scale) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_finalize_ungrouped,
        (APPROX, DST_ACCUM_MODE, topk, output_softmax),
        0,
        VectorMode::RC_custom,
        eps,
        scale)));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
}

// Re-init for the combine tail: after the copy_tile unpacks (which reconfigure the unpacker/datacopy
// addrmods), restore the transpose-dest addrmods (for step2) and the topk SFPU setup (for the merge/
// normalize). Mirrors the init that generalized_moe_gate(_init) does before the single-256 pipeline.
template <bool is_32bit = false>
ALWI void generalized_moe_gate_combine_init() {
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
    MATH((SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX, DST_ACCUM_MODE))));
}

// Transpose-dest step2 ONLY (no finalize/normalize): transpose the run from the SFPU "math" layout to the
// "standard" layout, so PACK (pack_untilize) can read it for the L1 stash. step2_init uses num_tiles=3, so
// it transposes all three regions (scores tile0 + idx tile1 + BIAS tile2) — the combine merge needs the bias
// too. Used by process_block_to_run before pack_untilize. (It also "settles" the math state so the NEXT
// block's produce_run/transpose_wh pipeline starts cleanly.)
template <bool is_32bit = false>
ALWI void generalized_moe_gate_step2_only() {
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
}

// Relocate a run between column-pairs within the scores/idx/bias regions (proven copy_topk_run).
template <uint32_t from_lo, uint32_t from_hi, uint32_t to_lo, uint32_t to_hi>
ALWI void generalized_moe_gate_relocate_run() {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_copy_topk_run,
        (APPROX, DST_ACCUM_MODE, from_lo, from_hi, to_lo, to_hi),
        0,
        VectorMode::RC_custom)));
}

// produce_run: for the multi-block (>256) path, end the ungrouped pipeline at merge16_to_run (a
// re-mergeable top-8 RUN at {run_store_lo, run_store_hi}, idx += idx_offset) and SKIP normalize+step2.
// Default (produce_run=false) = the single-256 path: finalize (merge + normalize) + step2.
template <
    bool enable_sigmoid = false,
    bool is_32bit = false,
    bool produce_run = false,
    uint32_t run_store_lo = 0,
    uint32_t run_store_hi = 2,
    uint32_t idx_offset = 0,
    uint32_t topk = 8,
    bool output_softmax = false>
ALWI void generalized_moe_gate(uint32_t icb0, uint32_t icb1, uint32_t eps, uint32_t scale) {
    if constexpr (enable_sigmoid) {
        // Transpose wh (FPU)
        transpose_wh_tile(icb0, 0, 0);
        // Sigmoid (SFPU)
        sigmoid_tile<VectorMode::RC_custom, false>(0);
        // Init add binary reuse (FPU)
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            false, false, icb1)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary_init_with_operands<
              EltwiseBinaryType::ELWADD,
              GeneralizedMoeGateEltwiseBinaryMode::RELOAD,
              MATH_FIDELITY>(icb1, icb1, false)));
        // Add binary reuse (FPU)
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb1, 0)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary<EltwiseBinaryType::ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(
            icb1, icb1, 0)));
        // Init transpose dest addrmods (does not conflict with add binary reuse)
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX, DST_ACCUM_MODE))));
    } else {
        // Copy add (FPU)
        UNPACK((llk_unpack_AB(icb0, icb1, 0, 0)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary<EltwiseBinaryType::ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(
            icb0, icb1, 0)));
    }
    // Set srcb dummy valid for transpose wh (FPU)
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    // Sum top2 (SFPU)
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_sum_top2,
        (APPROX, DST_ACCUM_MODE),
        0,
        VectorMode::RC_custom)));
    // Transpose dest step 0 (FPU) — always runs; puts each group g at DEST row g.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0<DST_ACCUM_MODE, is_32bit>()));
    // Path select — REQUIRED, NO silent default. The including kernel (.cpp) must define GMG_UNGROUPED_TOP8
    // to 1 (ungrouped top-k — every non-DeepSeek model) or 0 (grouped DeepSeek gate). Leaving it undefined is
    // a hard compile error on purpose: it used to fall through to the grouped path with no diagnostic, which
    // is wrong for every non-DeepSeek model. (Note `#if`, not `#ifdef`: `=0` selects grouped, `=1` ungrouped.)
#if !defined(GMG_UNGROUPED_TOP8)
#error \
    "generalized_moe_gate(): define GMG_UNGROUPED_TOP8 (to 1 = ungrouped top-k, or 0 = grouped DeepSeek) in the including kernel before this header — the silent grouped default was removed to avoid selecting the wrong HW sequence."
#endif
#if GMG_UNGROUPED_TOP8
    // TRUE GLOBAL TOP-8 over all 256 experts (ungrouped). post-step0: group g at DEST row g.
    // SFPU merges stay in rows 0-7 (only rows 0-7 are SFPU-addressable); FPU copy4rows (a plain
    // MOVD2B->MOVB2D copy) stashes 4-row blocks in rows 8-15. topA -> {0,2} (rows 0-3), topB -> {4,6}
    // (rows 4-7): row-disjoint. Each copy4rows uses a DISJOINT SrcB scratch window (16/20/24/28) so a
    // later MOVB2D can't read a previous (back-to-back) copy's SrcB leftover.
    //
    // save groups 4-7 source (rows 4-7) -> rows 8-11 (step1<0> below clobbers rows 0-7).
    MATH((llk_math_generalized_moe_gate_copy4rows_init<4, 8, is_32bit, 16>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    // topA = top8(groups 0-3): step1<d2b_dst=0> -> run at rows 0-7 -> merge -> topA at {0,2}.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init<0, 0, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi<DST_ACCUM_MODE, is_32bit>()));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_merge4_top8,
        (APPROX, DST_ACCUM_MODE, 0, 0, 2),
        0,
        VectorMode::RC_custom)));
    // park topA (rows 0-3) -> rows 12-15; restore groups 4-7 (rows 8-11) -> rows 4-7.
    MATH((llk_math_generalized_moe_gate_copy4rows_init<0, 12, is_32bit, 20>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_copy4rows_init<8, 4, is_32bit, 24>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    // topB = top8(groups 4-7): step1_hi<d2b_dst=4> -> run at rows 0-7 -> merge -> topB at {4,6}.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init<4, 0, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi<DST_ACCUM_MODE, is_32bit>()));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_merge4_top8,
        (APPROX, DST_ACCUM_MODE, 0, 4, 6),
        0,
        VectorMode::RC_custom)));
    // restore topA (rows 12-15) -> rows 0-3; now topA@{0,2} (rows 0-3), topB@{4,6} (rows 4-7).
    MATH((llk_math_generalized_moe_gate_copy4rows_init<12, 0, is_32bit, 28>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    if constexpr (produce_run) {
        // Multi-block: emit this block's top-8 as a re-mergeable RUN at {run_store_lo, run_store_hi}
        // (idx += idx_offset for global ids). No normalize/step2 here — the combine does that.
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            generalized_moe_gate_merge16_to_run,
            (APPROX, DST_ACCUM_MODE, run_store_lo, run_store_hi, idx_offset),
            0,
            VectorMode::RC_custom)));
    } else {
        // Single ≤256 block: full bitonic sort of topA{0,2}+topB{4,6} -> global top-8, then keep top-`topk`
        // (zero ranks >= topk before normalize) + normalize over those (softmax over the kept if output_softmax).
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            generalized_moe_gate_finalize_ungrouped,
            (APPROX, DST_ACCUM_MODE, topk, output_softmax),
            0,
            VectorMode::RC_custom,
            eps,
            scale)));
    }
#else
    // Grouped DeepSeek gate: sort_top4 selects top-4 groups, step1 lays them out, top8 merges.
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_sort_top4_groups,
        (APPROX, DST_ACCUM_MODE),
        0,
        VectorMode::RC_custom)));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1<DST_ACCUM_MODE, is_32bit>()));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        generalized_moe_gate_top8,
        (APPROX, DST_ACCUM_MODE),
        0,
        VectorMode::RC_custom,
        eps,
        scale)));
#endif  // GMG_UNGROUPED_TOP8
    // Transpose dest step 2 (FPU) — final output layout. Skipped in produce_run mode (the run stays
    // in the SFPU run layout for the combine; step2 runs once after the combine instead).
    if constexpr (!produce_run) {
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
    }
}

}  // namespace ckernel
