// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "lltt.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat sfpu_reciprocal_iter(const sfpi::vFloat x) {
    // sfpi::approx_recip(x) will return ±0 for x = ±inf or x ≥ ±2**126, and ±inf for x = ±0.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Optionally improve the approximation using Newton-Raphson.
    if constexpr (max_iter > 0) {
        // Normally, t = 2.0 - x * y, but we negate this (and negate again using y = y * -t later).
        // On Blackhole, when x=0 and y=infinity (and vice versa), t=+NaN regardless of the operand signs.
        // Negating the meaning of t makes it easier to detect NaN using a trivial sign check t>=0.
        // Equivalently, we could use v_if (t >= 2.0) instead, but SFPI doesn't support SFPLE/SFPGT at the moment.
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (max_iter > 1) {
            sfpi::vFloat y1 = y * -t - 0.0f;
            // If t=NaN, then t>=0.  This check consumes the SFPNOP slot of the preceding SFPMAD.
            v_if(t < 0) {
                t = x * y1 - sfpi::vConstFloatPrgm0;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        } else {
            // If t=NaN, then t>=0.  This check cannot be hidden in a SFPNOP slot as it depends on the result of the
            // preceding SFPMAD.
            v_if(t < 0) { y = y * -t - 0.0f; }
            v_endif;
        }
    }

    return y;
}

// Approximate reciprocal, with throughput of 1c/32.
inline void _calculate_reciprocal_fast_7b_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// BF16 reciprocal using a Newton correction on the BF16 LSB.
inline void _calculate_reciprocal_fast_8b_3c_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG1, 5);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    constexpr int y = p_sfpu::LREG0;
    constexpr int x = p_sfpu::LREG1;

    // Macro template 0 uses SFPMAD_MOD1_INDIRECT_VD, so LREG7 selects where
    // the source x copy lands.
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);

    // Pseudocode for the BF16 correction:
    //
    // y = load()
    // x = y
    // y = arecip(y)
    // y[15:0] = 0x8000
    // e = x * y - 1
    // t = e >> 16
    // y += t          # integer add, not FP32 add
    // store(y)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TT_SFPLOADMACRO(
            /*lreg_ind*/ (0 << 2) | y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ ADDR_MOD_7,
            /*dest_reg_addr*/ 0);
        // Macro 0 schedules y = arecip(y) and x = y for the next SFPU issue.
        // Wait before writing y's low 16 bits directly.
        TTI_SFPNOP;
        // Keep the patch and correction in LReg space; macro store/reload
        // scheduling can read a just-written Dst block too soon on Blackhole.
        TTI_SFPLOADI(
            /*lreg_ind*/ y,
            /*instr_mod0*/ sfpi::SFPLOADI_MOD0_LOWER,
            /*imm16*/ 0x8000);
        TTI_SFPMAD(
            /*lreg_src_a*/ x,
            /*lreg_src_b*/ y,
            /*lreg_src_c*/ p_sfpu::LCONST_neg1,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 0);
        TTI_SFPSHFT(
            /*imm12_math*/ (-16) & 0xFFF,
            /*lreg_c*/ x,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 5);
        TTI_SFPIADD(
            /*imm12_math*/ 0,
            /*lreg_c*/ x,
            /*lreg_dest*/ y,
            /*instr_mod1*/ sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(
            /*lreg_ind*/ y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ ADDR_MOD_6,
            /*dest_reg_addr*/ 0);
    }

    TTI_SFPNOP;
#endif
}

// FP32 reciprocal, with throughput of 5c/32.
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // Pseudocode:
    //
    // y = arecip(x)
    // e = 1 - x*y
    // t = e * e + e
    // t2 = t * e + e    # e**3 + e**2 + e
    // t2 = min(t2, 1.0) # replace NaN with 1.0
    // y = t2 * y + y    # y = y * (e**3 + e**2 + e + 1)
    //                   # if y = ±0 or ±inf, then y = y+y
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    //   | Load | Simple                 | MAD                     | Store   |
    // - | -----| ---------------------- | ----------------------- |-------- |
    // 0 | [y]  |                        |                         |         |
    // 1 |      | [y] = arecip(y)        |                         |         |
    // 2 | [e]  |                        |                         |         |
    // 3 |      | [e] L16 = arecip(e)    | e = mad(-e, y, 1.0)     |         |
    // 4 |      |                        |                         |         |
    // 0 |      |                        | [e] = mad(e, e, e)      | [e]     |
    // 1 | [t2] |                        |                         |         |
    // 2 |      |                        | [t2] = mad(t2, e, t2)   | [y] L16 |
    // 3 |      |                        |                         |         |
    // 4 | [z]  | [t2] = swap(t2, 1.0)   |                         |         |
    // 0 |      |                        |                         |         |
    // 1 |      |                        | [z] L16 = mad(t2, z, z) |         |
    // 2 |      |                        |                         |         |
    // 3 |      |                        |                         | [z] L16 |

    lltt::replay(0, 4);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++) {
        lltt::replay(0, 5);
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);
    TTI_SFPNOP;
    lltt::replay(3, 2);

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// ~7b precision; 1c/element
inline void _init_reciprocal_fast_7b_() {
#ifndef DISABLE_SFPLOADMACRO
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple                | Store   |
    // - | ---- | --------------------- | ------- |
    // 0 | [x]  |                       |         |
    // 1 |      | [x] L16 = arecip([x]) |         |
    // 2 |      |                       | [x] L16 |

    TTI_SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP);

    constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
    constexpr std::uint32_t mad_bits = 0;
    constexpr std::uint32_t round_bits = 0;
    constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

    TTI_SFPCONFIG(0, 4, 0);

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for macro 0.
    TTI_SFPCONFIG(0x110, 8, 1);
#endif
}

inline void _init_reciprocal_fast_8b_3c_() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPARECIP(
        /*imm12_math*/ 0,
        /*lreg_c*/ 0,
        /*lreg_dest*/ 12,
        /*instr_mod1*/ sfpi::SFPARECIP_MOD1_RECIP);

    // InstructionTemplate[1]
    TTI_SFPMAD(
        /*lreg_src_a*/ p_sfpu::LCONST_0,
        /*lreg_src_b*/ p_sfpu::LCONST_0,
        /*lreg_src_c*/ 0,
        /*lreg_dest*/ 13,
        /*instr_mod1*/ 8);  // SFPMAD_MOD1_INDIRECT_VD

    // Macro 0: [y]
    // Loads y, schedules y = arecip(y), and copies y to L[LREG7].
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: 0, unused
    //   UsesLoadMod0ForStore: {0,0,0,0},
    //   UnitDelayKind: Simple + MAD use WaitForElapsedInstructions
    // }
    TTI_SFPCONFIG(0x300, 8, 1);
#endif
}

inline void _init_reciprocal_fast_24b_5c_() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int e = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z = p_sfpu::LREG2;
    constexpr int y = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP);

    // InstructionTemplate[1]
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);

    // InstructionTemplate[2]
    // SFPMAD(VA=t2, VB=0 or VD, VC=VD or z)
    TTI_SFPMAD(t2, p_sfpu::LREG0, z, 14, 0);

    // InstructionTemplate[3]
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // Macro 0: [y]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0);
    }

    // Macro 1: [e]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2: [t2]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (2 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (0 << 3) | (4 + 2);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Macro 3: [z]
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0x80 | 0x40 | (1 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0xff0, 8, 1);

    constexpr std::uint32_t prev_offset = -2 & 0x3ff;
    constexpr std::uint32_t offset = 0;

    load_replay_buf(0, 6, [e, t2, z, y, offset, prev_offset] {
        TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, ADDR_MOD_7, prev_offset | (t2 >> 2));
        TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, ADDR_MOD_7, offset | (e >> 2));
        TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_6, prev_offset | (z >> 2));
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_7, prev_offset | (z >> 2));
    });
#endif
}

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return sfpu_reciprocal_iter<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init() {
    if constexpr (!APPROXIMATE) {
        sfpi::vConstFloatPrgm0 = 2.0f;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, bool legacy_compat = false>
inline void calculate_reciprocal() {
    if constexpr (legacy_compat) {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
    } else if constexpr (APPROXIMATION_MODE) {
        _calculate_reciprocal_fast_7b_(ITERATIONS);
    } else if constexpr (is_fp32_dest_acc_en) {
        _calculate_reciprocal_fast_24b_5c_(ITERATIONS);
    } else {
        _calculate_reciprocal_fast_8b_3c_(ITERATIONS);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
void recip_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + reciprocal's ADDR_MOD_6 + counter
    // reset), then the op-specific reciprocal setup below -- one self-contained init, matching exp_init.
    // SDPA runs reciprocal in its softmax after matmul/exp, so the general SFPU state is re-established
    // here, not just reset. Reciprocal uses ADDR_MOD_6 (dest incr 2) on Blackhole.
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        if constexpr (APPROXIMATION_MODE) {
            _init_reciprocal_fast_7b_();
        } else if constexpr (is_fp32_dest_acc_en) {
            _init_reciprocal_fast_24b_5c_();
        } else {
            _init_reciprocal_fast_8b_3c_();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
