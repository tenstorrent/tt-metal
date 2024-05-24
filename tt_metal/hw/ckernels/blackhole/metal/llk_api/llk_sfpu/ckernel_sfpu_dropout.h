// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(uint prob, uint scale) {
    // SFPU microcode

    vUInt rand = l_reg[LRegs::LReg3];

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        v_if(rand < prob) { dst_reg[0] = vConst0; }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        vUInt lfsr = vConstIntPrgm1;
        vUInt tmp = lfsr & rand;
        rand = rand >> 1;
        v_if(tmp != 0) {
            vUInt mask = vConstIntPrgm0;
            rand ^= mask;
        }
        v_endif;

        dst_reg++;
    }

    l_reg[LRegs::LReg3] = rand;
}

template <bool APPROXIMATION_MODE>
inline void dropout_init(const uint seed) {
    vConstIntPrgm0 = 0xb400;
    vConstIntPrgm1 = 0x1;  // binary 0b1 - used to extract LSB

    _init_dropout_seed_(seed);
}

}  // namespace sfpu
}  // namespace ckernel
