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

// One pass of scan_tile_columns: loads 4 rows of each face pair into LREG0-3 (even PASS) or LREG4-7
// (odd PASS), transposes them, replays the scan step and stores them back. Passes 0-3 cover faces 0-1
// (rows 0, 4, 8, 12), passes 4-7 faces 2-3.
template <std::uint32_t STEP, std::uint32_t PASS>
inline void scan_tile_columns_pass() {
    constexpr std::uint32_t lreg = (PASS % 2) * 4;
    constexpr std::uint32_t row = (PASS % 4) * 4 + (PASS / 4) * 32;
    TTI_SFPLOAD(lreg, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPLOAD(lreg + 1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPLOAD(lreg + 2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPLOAD(lreg + 3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay((PASS % 2) * STEP, STEP);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(lreg, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPSTORE(lreg + 1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPSTORE(lreg + 2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPSTORE(lreg + 3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
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
        // Clear context for F0
        TTI_SFPMOV(0, 9, 4, 0);
        TTI_SFPMOV(0, 9, 5, 0);
        TTI_SFPMOV(0, 9, 6, 0);
        TTI_SFPMOV(0, 9, 7, 0);
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
    TTI_SFPEXEXP(0, 5, 5, sfpi::SFPEXEXP_MOD1_NODEBIAS);
    TTI_SFPIADD(-255 & 0xfff, 5, 5, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 6, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

// One pass of the compensated scan: loads 4 rows of each face pair into LREG0-3, transposes them,
// scans them and stores them back. The two transposes cancel out on LREG4-7, so the sum and the
// compensation stay in LREG7 and LREG6 between passes.
template <std::uint32_t PASS>
inline void compensated_scan_pass() {
    constexpr std::uint32_t row = (PASS % 4) * 4 + (PASS / 4) * 32;
    TTI_SFPLOAD(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPLOAD(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPLOAD(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPLOAD(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
    compensated_scan_row<0, 7>();
    compensated_scan_row<1, 0>();
    compensated_scan_row<2, 1>();
    compensated_scan_row<3, 2>();
    TTI_SFPMOV(0, 3, 7, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row);
    TTI_SFPSTORE(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 2);
    TTI_SFPSTORE(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 16);
    TTI_SFPSTORE(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, row + 18);
}

// Like calculate_cumsum, with the compensation carried in LREG6 next to the running sum in LREG7. The
// passes are inline: the 4 rows of a pass do not fit the SFPU replay slots that scan_tile_columns uses.
template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_cumsum_compensated(const bool first) {
    if (first) {
        // The passes transpose LREG4-7 as a block, so all four are cleared.
        TTI_SFPMOV(0, p_sfpu::LCONST_0, 4, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, 5, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, 6, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, 7, 0);
    }
    compensated_scan_pass<0>();
    compensated_scan_pass<1>();
    compensated_scan_pass<2>();
    compensated_scan_pass<3>();
    compensated_scan_pass<4>();
    compensated_scan_pass<5>();
    compensated_scan_pass<6>();
    compensated_scan_pass<7>();
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_compensated_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}  // namespace ckernel::sfpu
