#pragma once
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_relu.h"

using namespace ckernel;

// New LLK SFPU APIs

// RELU MAX
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_max(uint dst_index,uint param0, int vector_mode = Dim::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, dst_sync>
                                (ckernel::sfpu::relu_max<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::relu_max<APPROXIMATE>,
                                dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_max_init() {
   ;
}

// RELU MIN
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_min(uint dst_index,uint param0, int vector_mode = Dim::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, dst_sync>
                                (ckernel::sfpu::relu_min<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::relu_min<APPROXIMATE>,
                                dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_min_init() {
   ;
}

//Leaky Relu
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_leaky_relu(uint dst_index,uint param0, int vector_mode = Dim::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, dst_sync>
                                (ckernel::sfpu::calculate_lrelu<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_lrelu<APPROXIMATE>,
                                dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_leaky_relu_init() {
   ;
}
