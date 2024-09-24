// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;
namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder(const uint value, const uint recip) {

    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat s = c_value.f;
    vFloat value_tmp = s;
    s = sfpi::abs(s);

    c_value.u = recip;
    vFloat recip_val = c_value.f;
    recip_val = sfpi::abs(recip_val);

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat v = sfpi::abs(val);

        vFloat quotient = v*recip_val;

        vInt tmp = float_to_int16(quotient); //TODO: Replace float_to_int16 to float_to_int32 once it is available
        vFloat newquotient= int32_to_float(tmp);
        v_if (newquotient > quotient){
            newquotient = newquotient - 1;
        }
        v_endif;
        v = v - newquotient * s;

        v_if(val<0 && v!=0){
            v = s - v;
        }
        v_endif;

        v_if(value_tmp<0 && v!=0){
            v = v + value_tmp;
        }
        v_endif;
        v = setsgn(v, value_tmp);
        v_if(s==0){
            v = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;

        constexpr auto iter = 10;
        for(int l=0; l<iter; l++)
        {
            v_if(v>=s){
                v = s - v;
            }
            v_endif;
        }
        v_if(sfpi::abs(v)-s==0.0f){
            v = 0.0f;
        }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
