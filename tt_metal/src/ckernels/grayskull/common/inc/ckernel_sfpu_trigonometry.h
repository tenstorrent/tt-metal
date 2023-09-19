/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"

using namespace sfpi;

namespace ckernel {

namespace sfpu {


template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_tangent_maclaurin_series(vFloat val)
{
    // For input [-1:1]
    // Mclauren series
    // tan(x) = x + (x^3)/3 + (2x^5)/15 + (17x^7)/315 + (62x^9)/2835 + (1382x^11)/155925 + (21844x^13)/6081075 + ...

    vFloat tmp = val;
    vFloat val_square = val * val;

    // x
    vFloat output = tmp;
    // x^3/3
    tmp = tmp * val_square;
    output += 0.3333333333333333 * tmp;
    // (2x^5)/15
    tmp = tmp * val_square;
    output += 0.13333333333333333 * tmp;

    //(17x^7)/315
    tmp = tmp * val_square;
    output += 0.05396825396825397 * tmp;

    //(62x^9)/2835
    tmp = tmp * val_square;
    output += 0.021869488536155203 * tmp;

	// (1382x^11)/155925
    tmp = tmp * val_square;
    output += 0.008863235529902197 * tmp;

	// (21844x^13)/6081075
	tmp = tmp * val_square;
	output += 0.003592128036572481 * tmp;

    // Write out output
    return output;
}

#define PI   (3.14159265358979323846)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        // Range Reduction: It will help us to cover more input range
        v_if(v > PI_2){
            v = v - PI;
        }v_elseif(v < -PI_2){
            v = v + PI;
        }v_else{
            v = v;
        }v_endif;

        v = sfpu_tangent_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}


template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_sfpu_trig() {
    if constexpr (operation == SfpuType::tan) {
        calculate_tangent<APPROXIMATION_MODE, ITERATIONS>();
    }
}


template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_tan_op(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_sfpu_trig<SfpuType::tan, APPROXIMATE, 1>,
                                ckernel::sfpu::calculate_sfpu_trig<SfpuType::tan, APPROXIMATE>,
                                dst_index, vector_mode);

}
}  // namespace sfpu
}  // namespace ckernel
