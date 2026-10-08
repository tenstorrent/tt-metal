// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_binary_remainder.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel::sfpu {

// FMOD = a - trunc(a / b) * b
// Implemented using 32-bit integer remainder kernel (see ckernel_sfpu_remainder_int32.h)
sfpi_inline void calculate_fmod_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Read inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // FMOD sign handling (result has the same sign as a)
    v_if(a_signed < 0) { r = -r; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_fmod_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // fmod(a, b) = a - trunc(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;
    sfpi::vFloat b_abs = sfpi::abs(b);

    // Compute reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::sfpu_reciprocal_iter<2>(b);

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    sfpi::vFloat trunc_div = _trunc_body_(div_result);

    // Compute fmod = a - trunc(a/b) * b
    sfpi::vFloat result = a - trunc_div * b;

    // Post-correction - fmod result must satisfy |result| < |b|
    // If |result| >= |b|, the truncation was wrong by 1
    sfpi::vFloat result_abs = sfpi::abs(result);

    // If result >= b, we truncated too low, add/subtract b to correct
    v_if(result_abs >= b_abs) {
        // Determine correction direction based on sign of result
        v_if(result >= sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // result was positive and too big
        }
        v_else {
            result = result + b_abs;  // result was negative and too big (magnitude)
        }
        v_endif;
    }
    v_endif;

    // Sign correction - fmod result must have same sign as 'a' (or be zero)
    // If a > 0 and result < 0, the truncation was 1 too high, need to add b
    // If a < 0 and result > 0, the truncation was 1 too low, need to subtract b
    // This fixes cases where a/b ≈ 0.9999999 but rounds to 1 due to reciprocal error
    v_if(a >= sfpi::vFloat(0.0f)) {
        // a is positive, result should be >= 0
        v_if(result < sfpi::vFloat(0.0f)) {
            result = result + b_abs;  // over-truncated
        }
        v_endif;
    }
    v_else {
        // a is negative, result should be <= 0
        v_if(result > sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // under-truncated
        }
        v_endif;
    }
    v_endif;

    // Handle special cases using conditional assignment (NOT early return!)
    // When a == b, fmod(a, b) = 0
    v_if(a == b) { result = sfpi::vFloat(0.0f); }
    v_endif;

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }

    return result;
}

#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void fmod_int32_lm_head(const uint in0, const uint in1) {
    // macro 0: |b| next cycle
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG2 >> 2));
    // macro 1: SFPENCC
    TT_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in0 | (p_sfpu::LREG1 >> 2));
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    // dummy, bf < 0 lanes only; macro 2: e, SFPENCC
    TT_SFPLOADMACRO((2 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 1)
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG5, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG6, sfpi::SFPABS_MOD1_INT);  // with e = 1 - L5 * L0 (macro 2)
    TTI_SFPCAST(p_sfpu::LREG6, p_sfpu::LREG4, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    TTI_SFPGT(0, p_sfpu::LREG4, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 2)
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG12, p_sfpu::LREG4, 0);
    TTI_SFPSHFT(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG7, 5);
    TTI_SFPEXMAN(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPMUL24(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG3, sfpi::SFPMUL24_MOD1_LOWER);
    // dummy; macro 3: SFPENCC
    TT_SFPLOADMACRO((3 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPSHFT(10, p_sfpu::LREG3, p_sfpu::LREG3, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPABS(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 3)
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG5, sfpi::SFPMUL24_MOD1_UPPER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT(23, p_sfpu::LREG5, p_sfpu::LREG5, 7);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

sfpi_inline void fmod_int32_lm_tail(const uint out) {
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG0, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_fmod_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_fmod_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of calculate_fmod_int32_body: 48 issues per row instead of 51; three SFPENCCs
    // share a cycle with the predicated SFPLOADI they close, which then still sees the old lane flags.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 32);
    fmod_int32_lm_head(in0, in1);
    fmod_int32_lm_tail(out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 32);
        fmod_int32_lm_tail(out);
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_fmod(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_fmod_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void fmod_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
#ifndef DISABLE_SFPLOADMACRO
    // A disabled unit uses delay 7 so it cancels no pending instruction.
    constexpr std::uint32_t disabled = 7 << 3;
    // InstructionTemplate[0]: SFPABS in place.
    {
        constexpr std::uint32_t insn = TT_OP_SFPABS(0, 0, 0, sfpi::SFPABS_MOD1_INT);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 0, 0);
    }
    // InstructionTemplate[1]: SFPENCC as in the body.
    {
        constexpr std::uint32_t insn = TT_OP_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 1, 0);
    }
    // InstructionTemplate[2]: VD = -L5 * VD + 1.0.
    {
        constexpr std::uint32_t insn = TT_OP_SFPMAD(p_sfpu::LREG5, 0, p_sfpu::LCONST_1, 0, 1);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 2, 0);
    }
    // Macro 0: |b| in place, next cycle.
    {
        constexpr std::uint32_t simple_bits = (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: SFPENCC with the first 2^31 fixup.
    {
        constexpr std::uint32_t simple_bits = (3 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2: e = 1 - L5 * bf after the fixup, SFPENCC with the second fixup.
    {
        constexpr std::uint32_t simple_bits = (6 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = 0x80 | (2 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Macro 3: SFPENCC with the third fixup.
    {
        constexpr std::uint32_t simple_bits = (5 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }
    // Misc: every unit counts issued instructions; no scheduled store.
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE>
inline void fmod_binary_init() {
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu
