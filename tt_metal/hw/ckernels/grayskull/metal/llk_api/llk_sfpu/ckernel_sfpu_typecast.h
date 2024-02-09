// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {

namespace sfpu {


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_to_uint16()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = 0;

        for(int i=0; i<47 ; i++)
        {
                v_if (val == std::numeric_limits<float>::infinity()){
                    result = std::numeric_limits<float>::infinity();
                    val = 0.0f;
                }
                v_elseif (val == -std::numeric_limits<float>::infinity()){
                    result = -std::numeric_limits<float>::infinity();
                    val = 0.0f;
                }
                v_elseif ( val >= 10000.0f  ){
                result += 10000;
                val -= 10000.0f;
                }
                v_elseif ( val >= 1000.0f  ){
                    result += 1000;
                    val -= 1000.0f;
                }
                v_elseif ( val >= 100.0f  ){
                    result += 100;
                    val -= 100.0f;
                }
                v_elseif ( val >= 10.0f ){
                    result += 10;
                    val -= 10.0f;
                }
                v_elseif ( val >= 1.0f){
                    result += 1;
                    val -= 1.0f;
                }v_elseif ( val < 0.0f){
                    result = 0;
                    val = 0.0f;
                }
                v_endif;

        }
        dst_reg[0] = result;

        dst_reg++;
    }

}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_to_uint32()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat result = 0;
        vFloat val = dst_reg[0];

        for(int i = 0; i < 95; i++)
        {
                v_if (val == std::numeric_limits<float>::infinity()){
                    result = std::numeric_limits<float>::infinity();
                    val = 0.0f;
                }
                v_elseif (val == -std::numeric_limits<float>::infinity()){
                    result = -std::numeric_limits<float>::infinity();
                    val = 0.0f;
                }
                v_elseif ( val >= 100000000.0f){
                    result += 100000000;
                    val -= 100000000.0f;
                } v_elseif ( val >= 10000000.0f){
                    result += 10000000;
                    val -= 10000000.0f;
                } v_elseif ( val >= 10000000.0f){
                    result += 10000000;
                    val -= 10000000.0f;
                } v_elseif ( val >= 1000000.0f){
                    result += 1000000;
                    val -= 1000000.0f;
                } v_elseif ( val >= 100000.0f){
                    result += 100000;
                    val -= 100000.0f;
                } v_elseif ( val >= 10000.0f){
                    result += 10000;
                    val -= 10000.0f;
                }v_elseif ( val >= 1000.0f){
                    result += 1000;
                    val -= 1000.0f;
                } v_elseif ( val >= 100.0f){
                    result += 100;
                    val -= 100.0f;
                } v_elseif ( val >= 10.0f){
                    result += 10;
                    val -= 10.0f;
                } v_elseif ( val >= 1.0f){
                    result += 1;
                    val -= 1.0f;
                } v_elseif ( val < 0.0f){
                    result = 0;
                    val = 0.0f;
                }
                v_endif;
        }
        dst_reg[0] = result;

        dst_reg++;
    }

}

template <bool APPROXIMATION_MODE>
void to_uint16_tile_init(){
    ;
}

template <bool APPROXIMATION_MODE>
void to_uint32_tile_init() {
    ;
}
} //ckernel
} // sfpu
