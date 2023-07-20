#pragma once
#define OPTIMIZED_COMPILE_SQRT
#ifdef OPTIMIZED_COMPILE_SQRT
#include "ckernel_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_sqrt.h"
#else
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu.h"
#endif
using namespace ckernel;

// New LLK SFPU APIs

#ifdef OPTIMIZED_COMPILE_SQRT
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sqrt(uint dst_index, int vector_mode = Dim::RC) {
    constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_sfpu_sqrt<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_sfpu_sqrt<APPROXIMATE>,
                                dst_index, vector_mode);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt_init() {
    sfpu::sfpu_init_opt<APPROXIMATE>(SfpuType::sqrt);
}
#else
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sqrt(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::sqrt, APPROXIMATE, Dst>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROXIMATE>();
}

#endif
