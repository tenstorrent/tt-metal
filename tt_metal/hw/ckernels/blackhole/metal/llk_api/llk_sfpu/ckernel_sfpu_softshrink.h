// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel::sfpu {

inline void softshrink_init() {
    // The λ >= 0 rows step DEST with their last store.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    if (static_cast<int32_t>(param0) >= 0) {
        // x - copysign(λ, x) is x - λ or x + λ with the same rounding, kept where |x| > λ.
        // The zero store sits between the add and its reader, so nothing waits on the add.
        _sfpu_load_imm32_(p_sfpu::LREG2, param0);
#pragma GCC unroll 32
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
            TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LREG3, 1);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPGT(0, p_sfpu::LREG2, p_sfpu::LREG1, 1);
            TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(3, 0, 0, 10);
        }
        return;
    }
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
