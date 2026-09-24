// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(std::uint32_t probability, std::uint32_t scale) {
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * sFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c =  9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = 0.0f;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, 3, 0);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void dropout_init(const std::uint32_t seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    init_prng_seed(seed);
}

// Op class for dropout: zero each element with the given probability, otherwise scale it. init
// seeds the PRNG.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Dropout : SfpuUnaryOp<Dropout<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = calculate_dropout<APPROXIMATION_MODE, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op(const std::uint32_t seed) {
        dropout_init<APPROXIMATION_MODE>(seed);
    }
};

}  // namespace sfpu
}  // namespace ckernel
