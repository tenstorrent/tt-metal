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
 * LLK MATMUL + EXP (overlapped via double buffering) - Blackhole
 *************************************************************************/

// =============================== INIT ===================================

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

template <bool approx = false, bool fast_and_approx = true, uint32_t scale = 0x3F800000>
inline void llk_mmexp_pack_init() {
    ckernel::llk_pack_sfpu_exponential_init<approx, fast_and_approx, scale>();
}

// =============================== ISSUE ==================================

template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0>
inline void llk_mmexp_math(
    const uint dst_index,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    llk_math_matmul<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(dst_index, transpose, ct_dim, rt_dim, kt_dim);
}

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
