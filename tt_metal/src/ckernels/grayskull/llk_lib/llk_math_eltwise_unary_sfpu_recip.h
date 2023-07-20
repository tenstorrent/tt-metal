#pragma once
#define OPTIMIZED_COMPILE_RECIP
#ifdef OPTIMIZED_COMPILE_RECIP
#include "ckernel_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_recip.h"
#else
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu.h"
#endif
using namespace ckernel;

// New LLK SFPU APIs

#ifdef OPTIMIZED_COMPILE_RECIP
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_reciprocal(uint dst_index, int vector_mode = Dim::RC) {
	constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_sfpu_reciprocal<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_sfpu_reciprocal<APPROXIMATE>,
                                dst_index, vector_mode);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init() {
    sfpu::sfpu_init_opt<APPROXIMATE>(SfpuType::reciprocal);
}
#else
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_reciprocal(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::reciprocal, APPROXIMATE, Dst>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROXIMATE>();
}

#endif
