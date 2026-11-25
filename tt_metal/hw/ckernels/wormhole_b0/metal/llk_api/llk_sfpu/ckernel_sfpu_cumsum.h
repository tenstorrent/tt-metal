// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel::sfpu {

// Loads 4 rows of each face pair into LREG<LREG> to LREG<LREG + 3>, transposes them, runs `scan`
// on them, transposes them back and stores them. Passes 0-3 cover faces 0-1 (rows 0, 4, 8, 12),
// passes 4-7 faces 2-3.
template <std::uint32_t PASS, std::uint32_t LREG, typename Scan>
inline void tile_columns_pass(Scan scan) {
    constexpr std::uint32_t row = (PASS % 4) * 4 + (PASS / 4) * 32;
    TTI_SFPLOAD(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPLOAD(LREG + 1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPLOAD(LREG + 2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPLOAD(LREG + 3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
    scan();
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPSTORE(LREG + 1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPSTORE(LREG + 2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPSTORE(LREG + 3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
}

// One pass of scan_tile_columns: even passes scan LREG0-3, odd passes LREG4-7, by replaying the
// scan step.
template <std::uint32_t STEP, std::uint32_t PASS>
inline void scan_tile_columns_pass() {
    tile_columns_pass<PASS, (PASS % 2) * 4>([] { lltt::replay((PASS % 2) * STEP, STEP); });
}

// Sets LREG4-7, which carry the running column values between tiles, to the constant in LREG<VALUE>.
template <std::uint32_t VALUE>
inline void set_carried_lregs() {
    TTI_SFPMOV(0, VALUE, 4, 0);
    TTI_SFPMOV(0, VALUE, 5, 0);
    TTI_SFPMOV(0, VALUE, 6, 0);
    TTI_SFPMOV(0, VALUE, 7, 0);
}

// Scans each column of the tile top to bottom in 8 passes. Passes on LREG0-3 replay slots [0, STEP)
// and passes on LREG4-7 replay [STEP, 2 * STEP), so the init records 2 * STEP slots. LREG4-7 carry
// the running column values between calls; cumsum and cumprod set them on `first`.
template <std::uint32_t STEP>
inline void scan_tile_columns() {
    static_assert(2 * STEP <= ckernel::math::replay_buf_offset, "the scan step must fit in the SFPU replay slots");
    scan_tile_columns_pass<STEP, 0>();
    scan_tile_columns_pass<STEP, 1>();
    scan_tile_columns_pass<STEP, 2>();
    scan_tile_columns_pass<STEP, 3>();
    scan_tile_columns_pass<STEP, 4>();
    scan_tile_columns_pass<STEP, 5>();
    scan_tile_columns_pass<STEP, 6>();
    scan_tile_columns_pass<STEP, 7>();
}

// cumsum_init records an SFPADD and an SFPNOP for each LREG: 8 instructions per pass.
constexpr std::uint32_t cumsum_step = 8;

template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumsum(const bool first) {
    if (first) {
        set_carried_lregs<p_sfpu::LCONST_0>();
    }
    scan_tile_columns<cumsum_step>();
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    lltt::record(0, 2 * cumsum_step);
    // FIXME: These should all be TT_SFP...
    TTI_SFPADD(10, 7, 0, 0, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 0, 1, 1, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 1, 2, 2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 2, 3, 3, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 4, 4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 4, 5, 5, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 5, 6, 6, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 6, 7, 7, 0);
    TTI_SFPNOP;
}

// One row of the compensated (Kahan) scan, the steps of the permuted path's calculate_kahan_step:
// y = x - c, t = s + y, c = (t - s) - y, and c = 0 where t - s is not finite. x, then t, is in
// LREG<ROW>, s in LREG<S> and c in LREG6; LREG4 and LREG5 hold y and t - s.
template <std::uint32_t ROW, std::uint32_t S>
inline void compensated_scan_row() {
    TTI_SFPMAD(6, p_sfpu::LCONST_neg1, ROW, 4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LCONST_1, S, 4, ROW, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(S, p_sfpu::LCONST_neg1, ROW, 5, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(4, p_sfpu::LCONST_neg1, 5, 6, 0);
    // t - s is inf or NaN exactly when its biased exponent is 255: subtract 255 (as a 12-bit
    // immediate) and set c = 0 on the lanes where the result is >= 0.
    TTI_SFPEXEXP(0, 5, 5, sfpi::SFPEXEXP_MOD1_NODEBIAS);
    TTI_SFPIADD(-255 & 0xfff, 5, 5, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 6, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

// One pass of the compensated scan, always on LREG0-3. The two transposes cancel out on LREG4-7, so
// the sum and the compensation stay in LREG7 and LREG6 between passes.
template <std::uint32_t PASS, bool is_fp32_dest_acc_en>
inline void compensated_scan_pass() {
    tile_columns_pass<PASS, 0>([] {
        compensated_scan_row<0, 7>();
        compensated_scan_row<1, 0>();
        compensated_scan_row<2, 1>();
        compensated_scan_row<3, 2>();
        TTI_SFPMOV(0, 3, 7, 0);
        if constexpr (!is_fp32_dest_acc_en) {
            // SFPSTORE truncates into a 16-bit Dest, so round the outputs to nearest first. The sum
            // carried in LREG7 stays fp32.
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, 0, 0, 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, 0, 1, 1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, 0, 2, 2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, 0, 3, 3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
    });
}

// Like calculate_cumsum, with the compensation carried in LREG6 next to the running sum in LREG7. The
// passes are inline: the 4 rows of a pass do not fit the SFPU replay slots that scan_tile_columns uses.
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en>
inline void calculate_cumsum_compensated(const bool first) {
    if (first) {
        // The passes transpose LREG4-7 as a block, so all four are cleared, not only LREG6-7.
        set_carried_lregs<p_sfpu::LCONST_0>();
    }
    compensated_scan_pass<0, is_fp32_dest_acc_en>();
    compensated_scan_pass<1, is_fp32_dest_acc_en>();
    compensated_scan_pass<2, is_fp32_dest_acc_en>();
    compensated_scan_pass<3, is_fp32_dest_acc_en>();
    compensated_scan_pass<4, is_fp32_dest_acc_en>();
    compensated_scan_pass<5, is_fp32_dest_acc_en>();
    compensated_scan_pass<6, is_fp32_dest_acc_en>();
    compensated_scan_pass<7, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_compensated_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}  // namespace ckernel::sfpu
