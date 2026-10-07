// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// The ttnn unary chain passes an address mode no other op of the chain programs, skips the counter reset on a later
// tile and keeps the rounding constants out of Prgm0-2.
template <uint32_t addr_mod, bool reset_counters, bool prgm_rounding>
inline void _square_init_() {
    // The paired store walks dest through ADDR_MOD_6, which advances by the two rows
    // the loop just wrote (one sfpi row is two dest counter steps), so the loop body
    // needs no separate increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(addr_mod);
    if constexpr (reset_counters) {
        math::reset_counters(p_setrwc::SET_ABD_F);
    }

    if constexpr (prgm_rounding) {
        sfpi::vConstIntPrgm0 = 1;
        sfpi::vConstIntPrgm1 = 0x7fff;
        sfpi::vConstIntPrgm2 = 0xffff0000;
    }
}

inline void square_init() { _square_init_<ADDR_MOD_6, true, true>(); }

sfpi_inline sfpi::vFloat float32_to_bf16_rne_prgm(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm0);
    bits = bits + sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm1) + lsb;
    bits = bits & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm2);
    return sfpi::as<sfpi::vFloat>(bits);
}

sfpi_inline sfpi::vFloat float32_to_bf16_rne_lreg(
    sfpi::vFloat in, sfpi::vUInt one, sfpi::vUInt half, sfpi::vUInt mask) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & one;
    bits = bits + half + lsb;
    bits = bits & mask;
    return sfpi::as<sfpi::vFloat>(bits);
}

// prgm_rounding false: the ttnn unary chain's form, the rounding constants in LRegs.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en = false,
    int ITERATIONS = 8,
    uint32_t addr_mod = ADDR_MOD_6,
    bool prgm_rounding = true>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

    if constexpr (!is_fp32_dest_acc_en && !prgm_rounding) {
        const sfpi::vUInt one = 1;
        const sfpi::vUInt half = 0x7fff;
        const sfpi::vUInt mask = 0xffff0000;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS; d += 2) {
            sfpi::vFloat v0 = sfpi::dst_reg[0];
            sfpi::vFloat v1 = sfpi::dst_reg[1];
            sfpi::dst_reg[0] = float32_to_bf16_rne_lreg(v0 * v0, one, half, mask);
            sfpi::dst_reg[1].mode(addr_mod) = float32_to_bf16_rne_lreg(v1 * v1, one, half, mask);
        }
        return;
    }

#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = v0 * v0;
        sfpi::vFloat r1 = v1 * v1;
        if constexpr (!is_fp32_dest_acc_en) {
            r0 = float32_to_bf16_rne_prgm(r0);
            r1 = float32_to_bf16_rne_prgm(r1);
        }
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1].mode(addr_mod) = r1;
    }
}

}  // namespace ckernel::sfpu
