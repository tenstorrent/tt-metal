// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_floor()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat orig = dst_reg[0];

        vFloat res=0;
        val=sfpi::abs(val);

        for(int i=0; i<10; i++){
            v_if(val>100000){
                val=val-100000;
            }
            v_endif;
        }
        for(int i=0; i<10; i++){
            v_if(val>10000){
                val=val-10000;
            }
            v_endif;
        }
        for(int i=0; i<10; i++){
            v_if(val>1000){
                val=val-1000;
            }
            v_endif;
        }
        for(int i=0; i<10; i++){
            v_if(val>100){
                val=val-100;
            }
            v_endif;
        }
        for(int i=0; i<10; i++){
            v_if(val>10){
                val=val-10;
            }
            v_endif;
        }
        v_if(val>5){
            val=val-5;
        }
        v_endif;
        v_if(val>2){
            val=val-2;
        }
        v_endif;
        v_if(val>2){
            val=val-2;
        }
        v_endif;
        v_if(val>1){
            val=val-1;
        }
        v_endif;
        val=setsgn(val,orig);

        v_if (val>0){
            res = orig-val;
            v_if (orig-res==1){
                res+=1;
            }
            v_endif;
        }
        v_elseif(val<0){
            res = orig-val-1;
        }
        v_endif;
        dst_reg[0] = res;
        dst_reg++;
    }
}


}  // namespace sfpu
}  // namespace ckernel
