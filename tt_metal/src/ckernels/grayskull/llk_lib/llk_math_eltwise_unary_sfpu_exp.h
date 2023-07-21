#pragma once


#include "ckernel_sfpu_init.h"
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
                                (ckernel::sfpu::calculate_sfpu_exponential<APPROXIMATE, zero_negative, false, first_iterations>,
                                ckernel::sfpu::calculate_sfpu_exponential<APPROXIMATE, zero_negative>,
                                dst_index, vector_mode, param0);
}



template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    sfpu::sfpu_init<APPROXIMATE>(SfpuType::exponential);
}
