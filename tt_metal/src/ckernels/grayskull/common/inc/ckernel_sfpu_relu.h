/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// RELU MAX
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::relu_max, APPROXIMATE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void relu_max(uint uint_threshold)
{
    vFloat threshold = Converter::to_float(uint_threshold);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a > threshold) {
            a = threshold;
        }
        v_endif;
        v_if(a < 0.0f) {
            a = 0.0f;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_max(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_max<APPROXIMATE,4>,
				 ckernel::sfpu::relu_max<APPROXIMATE,4>,
				 dst_index, Dim::RC, param0);
}


// RELU MIN

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void relu_min(uint uint_threshold)
{
    vFloat threshold = Converter::to_float(uint_threshold);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a < threshold) {
            a = threshold;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_min_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_min(uint dst_index, uint param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_min<APPROXIMATE,4>,
				 ckernel::sfpu::relu_min<APPROXIMATE,4>,
				 dst_index, Dim::RC, param0);
}

// RELU
//RELU - implemented by relu-min
//relu = relu_min @ threshold = 0
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu(uint dst_index) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_min<APPROXIMATE,4>,
				 ckernel::sfpu::relu_min<APPROXIMATE,4>,
				 dst_index, Dim::RC, 0);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROXIMATE>();
}


//Leaky Relu

// LRELU = LEAKY RELU
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_lrelu(uint slope)
{
    // SFPU microcode
    Converter c_slope;
    c_slope.u = slope;
    vFloat s = c_slope.f;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v *= s;
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_lrelu(uint dst_index, int param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_lrelu<APPROXIMATE,4>,
				 ckernel::sfpu::calculate_lrelu<APPROXIMATE,4>,
				 dst_index, Dim::RC, param0);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_leaky_relu(uint dst_index,int param0){
    llk_math_eltwise_unary_sfpu_lrelu<SfpuType::lrelu, APPROXIMATE, dst_sync>(dst_index,param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lrelu, APPROXIMATE>();
}


}  // namespace sfpu
}  // namespace ckernel
