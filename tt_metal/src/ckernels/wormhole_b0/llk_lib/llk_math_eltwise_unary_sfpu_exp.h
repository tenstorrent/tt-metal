/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_exp.h"
using namespace ckernel;

// New LLK SFPU APIs

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_exponential(uint dst_index, int vector_mode = Dim::RC, int param0 = 0) {

	constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
      (ckernel::sfpu::calculate_exponential<APPROXIMATE, zero_negative, false, first_iterations>,
       ckernel::sfpu::calculate_exponential<APPROXIMATE, zero_negative>,
                                dst_index, vector_mode, param0);

}


template <bool APPROXIMATION_MODE>
inline void configure_programmable_constants(SfpuType operation)
{
    if (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
    }
    else{
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        vConstFloatPrgm2 = 0.863281f;
    }
}




template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);
    configure_programmable_constants<APPROXIMATE>(SfpuType::exponential);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
