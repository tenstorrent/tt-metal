#pragma once
#define OPTIMIZED_COMPILE_GELU
#ifdef OPTIMIZED_COMPILE_GELU
#include "ckernel_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_gelu.h"
#else
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu.h"
#endif
using namespace ckernel;

// New LLK SFPU APIs

#ifdef OPTIMIZED_COMPILE_GELU
template <int ITERATIONS = 4>
inline void gelu_helper_wrapper(uint param0=0){
    if ( param0 ) {
	  ckernel::sfpu::calculate_sfpu_gelu<true, ITERATIONS>();
	} else {
	  ckernel::sfpu::calculate_sfpu_gelu<false, ITERATIONS>();
	}
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index, int vector_mode = Dim::RC, int param0=0) {
   	constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (gelu_helper_wrapper<first_iterations>,
                                gelu_helper_wrapper<4>,
                                dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init() {
    sfpu::sfpu_init_opt<APPROXIMATE>(SfpuType::gelu);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(uint dst_index, int vector_mode = Dim::RC) {
	constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_sfpu_gelu_derivative<APPROXIMATE, zero_negative, false, first_iterations>,
                                ckernel::sfpu::calculate_sfpu_gelu_derivative<APPROXIMATE, zero_negative>,
                                dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init() {
    sfpu::sfpu_init_opt<APPROXIMATE>(SfpuType::gelu_derivative);
}

#else
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index, int vector_mode = Dim::RC, int param0=0) {
    llk_math_eltwise_unary_sfpu<SfpuType::gelu, APPROXIMATE, Dst>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>();

}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::gelu_derivative, APPROXIMATE, Dst>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_derivative, APPROXIMATE>();
}

#endif
