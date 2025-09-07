// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// MATH-side matmul LLK API
#include "llk_math_matmul_api.h"

// PACK-side SFPU EXP LLK API wrapper (routes calls to PACK thread)
#include "llk_pack_sfpu_exp_api.h"

/*************************************************************************
 * LLK MATMUL + EXP (overlapped via double buffering)
 *
 * This header provides minimal LLK utilities to pipeline MATMUL on the MATH
 * thread with EXP on the PACK thread using a double-buffered DST. The
 * expectation is that the kernel orchestrator will:
 *   1) Initialize MATH and PACK via the provided init helpers.
 *   2) In a loop over DST tiles (or sub-blocks), issue PACK EXP on the
 *      previous DST buffer while issuing MATH MATMUL on the current DST.
 *
 * Example usage pattern (pseudo-code):
 *   // once
 *   llk_mmexp_math_init<NUM_FIDELITY, THROTTLE>(in0, in1, transpose, ct, rt, kt);
 *   llk_mmexp_pack_init<>();
 *
 *   // pipeline loop
 *   for (uint32_t i = 0; i < num_tiles; ++i) {
 *       // EXP on the previous output tile (PACK thread)
 *       if (i > 0) {
 *           llk_mmexp_pack<>(prev_dst_index /*i-1*/);
*
}
*  // MATMUL on the current output tile (MATH thread)
    *llk_mmexp_math<NUM_FIDELITY, THROTTLE>(curr_dst_index /*i*/, transpose, ct, rt, kt);
*
}
*  // tail EXP on the last produced tile (PACK thread)
    *llk_mmexp_pack<>(last_dst_index);
*************************************************************************/

    // =============================== INIT ===================================

    // Initialize MATMUL on MATH thread
    template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0>
    inline void llk_mmexp_math_init(
        const std::uint32_t operandA,
        const std::uint32_t operandB,
        const std::uint32_t transpose = 0,
        const std::uint32_t ct_dim = 1,
        const std::uint32_t rt_dim = 1,
        const std::uint32_t kt_dim = 1) {
    llk_math_matmul_init<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(operandA, operandB, transpose, ct_dim, rt_dim, kt_dim);
}

// Initialize EXP on PACK thread
// Matches math-side exp init but intended to be called on PACK to overlap with MATH matmul
template <bool approx = false, bool fast_and_approx = true, uint32_t scale = 0x3F800000>
inline void llk_mmexp_pack_init() {
    ckernel::llk_pack_sfpu_exponential_init<approx, fast_and_approx, scale>();
}

// =============================== ISSUE ==================================

// Issue MATMUL on MATH thread for a DST tile index
// This only executes the compute; data unpack should be handled by the UNPACK thread.
template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0>
inline void llk_mmexp_math(
    const uint dst_index,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    llk_math_matmul<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(dst_index, transpose, ct_dim, rt_dim, kt_dim);
}

// Issue EXP on PACK thread for a completed DST tile index
// EXP operates in-place on the DST register buffer tile
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    int iterations = 8,
    bool is_fp32_dest_acc_en = false>
inline void llk_mmexp_pack(
    const std::uint32_t idst,
    const int vector_mode = 0 /* VectorMode::RC */,
    const std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    ckernel::
        llk_pack_sfpu_exponential<approx, fast_and_approx, scale_en, skip_positive_check, iterations, DST_ACCUM_MODE>(
            idst, vector_mode, scale);
}

// =========================== FUSED OVERLAP ===============================
// Convenience API issuing: PACK EXP on previous DST tile, then MATH MATMUL on current DST tile
// VectorMode::C is common for attention softmax flows but can be changed by callers if needed
template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0>
inline void llk_mmexp_matmul_then_exp(
    const std::uint32_t idst_curr,
    const std::uint32_t idst_prev_for_exp,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
#ifdef TRISC_PACK
    ckernel::llk_pack_sfpu_exponential<>(idst_prev_for_exp, (int)VectorMode::C, p_sfpu::kCONST_1_FP16B);
#endif
    llk_math_matmul<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(idst_curr, transpose, ct_dim, rt_dim, kt_dim);
}
