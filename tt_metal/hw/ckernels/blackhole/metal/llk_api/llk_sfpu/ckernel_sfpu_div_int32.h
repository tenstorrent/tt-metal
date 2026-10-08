// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {
#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void div_int32_lm_row(const uint in0, const uint in1, const uint out) {
    // macro 0: in1 to sign-magnitude in place
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG4 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG4 >> 2));
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, in0);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLE(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
    TTI_SFPLE(0, p_sfpu::LREG4, p_sfpu::LREG0, 1);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG7, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPARECIP(0, p_sfpu::LREG7, p_sfpu::LREG2, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG12, p_sfpu::LREG3, 2);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG5, 3);
    TTI_SFPPUSHC(0, 0, 0, sfpi::SFPPUSHC_MOD1_PUSH);
    TTI_SFPGT(0, p_sfpu::LREG3, p_sfpu::LCONST_0, 1);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LREG12, p_sfpu::LREG6, 1);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG2, 2);
    TTI_SFPPOPC(0, 0, 0, sfpi::SFPPOPC_MOD1_POP);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 2);
    TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 2);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG0, 1);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // The inputs are in 2's complement form
        sfpi::vSMag in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vSMag in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vFloat result = 1.0f;

        v_if(in0 == 0 || in0 != in1) {
            sfpi::vFloat float_in0 = sfpi::convert<sfpi::vFloat>(in0, sfpi::RoundMode::Nearest);
            sfpi::vFloat float_in1 = sfpi::convert<sfpi::vFloat>(in1, sfpi::RoundMode::Nearest);
            sfpi::vFloat recip_in1 = sfpu_reciprocal_iter<2>(float_in1);
            result = float_in0 * recip_in1;
            // Residual correction using the remainder (a - q*b) * (1/b). The reciprocal is only
            // accurate to ~1 ulp, so exact quotients (e.g. 28/14) can land 1 ulp off (2.0000001).
            // One residual step snaps exact quotients to the correct value
            // Note: this cannot recover precision already lost when |operand| > 2^24 is rounded during
            // the int32 -> fp32 conversion.
            //   in1 == 0: recip is +/-inf and (a - inf*0) would produce a NaN, corrupting the
            //             intended inf/-inf/NaN result of division by zero.
            v_if(in1 != 0) { result = result + (float_in0 - result * float_in1) * recip_in1; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of the sfpi body above: 25 issues per row instead of 26, and two of the four MAD
    // latency stalls filled by moving SFPPUSHC and SFPPOPC; float(in1) is kept in L7.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 26);
    div_int32_lm_row(in0, in1, out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 26);
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    sfpu_reciprocal_init<false>();
#ifndef DISABLE_SFPLOADMACRO
    // A disabled unit uses delay 7 so it cancels no pending instruction.
    constexpr std::uint32_t disabled = 7 << 3;
    // InstructionTemplate[0]: int32 to sign-magnitude in place.
    {
        constexpr std::uint32_t insn = TT_OP_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 0, 0);
    }
    // Macro 0: in1 to sign-magnitude in place, next cycle.
    {
        constexpr std::uint32_t simple_bits = (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Misc: every unit counts issued instructions; no scheduled store.
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

}  // namespace ckernel::sfpu
