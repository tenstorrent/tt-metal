// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, uint prob, uint scale)
{
    // SFPU microcode

    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

    vUInt rand = l_reg[LRegs::LReg3];

    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        v_if (rand < prob) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        vUInt lfsr = vConstIntPrgm1;
        vUInt tmp = lfsr & rand;
        rand = rand >> 1;
        v_if (tmp != 0) {
            vUInt mask = vConstIntPrgm0;
            rand ^= mask;
        }
        v_endif;

        dst_reg++;
    }

    l_reg[LRegs::LReg3] = rand;
}

inline void _init_dropout_seed_(uint16_t p2){
    FWLOG1("calculate_dropout() -- input seed:%x", p2);

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);

    vInt result = l_reg[LRegs::LReg3];

    vInt tmp = vConstTileId << 10;
    vInt ptis = per_tensix_input_seed;
    result = ~(tmp & ptis) & (tmp | ptis);

    l_reg[LRegs::LReg3] = result;
}

inline void _init_dropout_(const uint seed)
{
    vConstIntPrgm0 = 0xb400;
    vConstIntPrgm1 = 0x1; // binary 0b1 - used to extract LSB

    _init_dropout_seed_(seed);
}

} // namespace sfpu
} // namespace ckernel
