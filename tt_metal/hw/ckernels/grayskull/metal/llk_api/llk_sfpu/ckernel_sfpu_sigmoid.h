// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL5(coef4,coef3,coef2,coef1,coef0,val) ( (((coef4*val + coef3)*val + coef2)*val + coef1)*val + coef0 )

inline
vFloat sigmoid_piecewise_linear_positive(vFloat val) {
    vFloat result = 0.0f;
    v_if ( val >= +5.0f)  {
        result = 1.0f;
    } v_elseif ( val > 1.0f && val < 5.0f ) {
        result = POLYVAL5(0.00144462f, -0.01055479f, -0.01203685f,  0.24300185f,  0.50437757f,val);
    } v_else {
        result = 0.229f*val + 0.5f; // linear appx as y = 0.229x + 0.5
    }
    v_endif;
    return result;
}

//sigmoid is anti-symmetric and offset by 1
//sigmoid[-x] = 1 - sigmoid[x]
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_sigmoid()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = 0.0f;

        v_if ( val < 0.0f ) {
  	    val = -val;
        }
        v_endif;

	      result = sigmoid_piecewise_linear_positive(val);

	      val = dst_reg[0];
        v_if ( val < 0.0f ) {
            result = 1.0f - result;
        }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }

    return;
}

}  // namespace sfpu
}  // namespace ckernel
