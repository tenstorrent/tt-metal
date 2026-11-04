// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// (PMK copy of Stage 7a's kernels/mhc_post_pf7a/mix_sfpu.hpp: identical arithmetic, renamed into the pmk_mhc DST slots)
// One post-mix output tile in ONE SFPU pass over DST (mpf7 DST slots, full-sync fp32 DST):
//   out = y * bcast_col(K col 0) + x_0 * bcast_col(K col 1) + x_1 * bcast_col(K col 16) + x_2 * bcast_col(K col 17)
//         + x_3 * bcast_col(L col 0)
// Per 4-row band: each coefficient is broadcast once (the LLK's BCAST_COL sequence: col-0 lane mask in LREG6 + the
// replayed rotate-add stages 1-2 + stage 3; ckernel_sfpu_binary_bcast.h, recorded by sfpu_bcast_col_init()), then
// multiplied / FMA'd into the band's 4 accumulator registers (4 x 8-column slots). Register use: LREG0 broadcast,
// LREG2 the broadcast's scratch (free between broadcasts: second data register), LREG6 mask, LREG1/3/4/5 accumulators,
// LREG7 data. SFPMAD is a fused multiply-add (one rounding); the op chain rounds the product and the sum separately.
#pragma once

#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_binary_bcast.h"

namespace ckernel::sfpu {

namespace pmk_mix_detail {
constexpr std::uint32_t ACC[4] = {p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5};
constexpr std::uint32_t DAT[2] = {p_sfpu::LREG7, p_sfpu::LREG2};
constexpr std::uint32_t T = DEST_TILE_SIZE_RAW;

inline void bcast(std::uint32_t addr) {
    TT_SFPLOAD(LREG_BCAST, InstrModLoadStore::DEFAULT, ADDR_MOD_7, addr);
    lltt::replay(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);
    _broadcast_stage3_preserve_data_();
}

// acc_k (= or +=) data tile slot k * bcast, k = 0..3
template <bool FIRST>
inline void term(std::uint32_t data_base, const std::uint32_t (&slot)[4]) {
#pragma GCC unroll 4
    for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t d = DAT[k & 1];
        TT_SFPLOAD(d, InstrModLoadStore::DEFAULT, ADDR_MOD_7, data_base + slot[k]);
        TTI_SFPNOP;
        if constexpr (FIRST) {
            TTI_SFPMUL(d, LREG_BCAST, p_sfpu::LCONST_0, ACC[k], 0);
        } else {
            TTI_SFPMAD(d, LREG_BCAST, ACC[k], ACC[k], 0);
        }
    }
}

inline void band(std::uint32_t left, std::uint32_t right) {
    const std::uint32_t slot[4] = {left, left + ODD_COLS_OFFSET, right, right + ODD_COLS_OFFSET};
#if defined(PMK_MIX_DBG) && PMK_MIX_DBG == 1  // bisect: out = x_0 (DST addressing + pack only)
#pragma GCC unroll 4
    for (std::uint32_t k = 0; k < 4; ++k) {
        TT_SFPLOAD(ACC[k], InstrModLoadStore::DEFAULT, ADDR_MOD_7, pmk_mhc::D_X0 * T + slot[k]);
    }
    TTI_SFPNOP;
#pragma GCC unroll 4
    for (std::uint32_t k = 0; k < 4; ++k) {
        TT_SFPSTORE(ACC[k], InstrModLoadStore::DEFAULT, ADDR_MOD_7, pmk_mhc::D_OUT * T + slot[k]);
    }
    return;
#endif
    bcast(pmk_mhc::D_K * T + slot[0]);  // post_s (column 0)
    term<true>(pmk_mhc::D_Y * T, slot);
#if defined(PMK_MIX_DBG) && PMK_MIX_DBG == 2  // bisect: out = post_s * y (one broadcast + MUL)
    TTI_SFPNOP;
    TTI_SFPNOP;
#pragma GCC unroll 4
    for (std::uint32_t k = 0; k < 4; ++k) {
        TT_SFPSTORE(ACC[k], InstrModLoadStore::DEFAULT, ADDR_MOD_7, pmk_mhc::D_OUT * T + slot[k]);
    }
    return;
#endif
    bcast(pmk_mhc::D_K * T + slot[1]);  // C[0][s] (column 1: lane 0 of the odd-column slot)
    term<false>((pmk_mhc::D_X0 + 0) * T, slot);
    bcast(pmk_mhc::D_K * T + slot[2]);  // C[1][s] (column 16)
    term<false>((pmk_mhc::D_X0 + 1) * T, slot);
    bcast(pmk_mhc::D_K * T + slot[3]);  // C[2][s] (column 17)
    term<false>((pmk_mhc::D_X0 + 2) * T, slot);
    bcast(pmk_mhc::D_L * T + slot[0]);  // C[3][s] (column 0)
    term<false>((pmk_mhc::D_X0 + 3) * T, slot);
    TTI_SFPNOP;
    TTI_SFPNOP;
#pragma GCC unroll 4
    for (std::uint32_t k = 0; k < 4; ++k) {
        TT_SFPSTORE(ACC[k], InstrModLoadStore::DEFAULT, ADDR_MOD_7, pmk_mhc::D_OUT * T + slot[k]);
    }
}
}  // namespace pmk_mix_detail

// the SFPU_BINARY_CALL functor (VectorMode::None: called once); the DST slots are fixed (pmk_mhc::D_*)
inline void pmk_mix_tile(std::uint32_t, std::uint32_t, std::uint32_t) {
    for (std::uint32_t b = 0; b < NUM_ROW_BANDS_PER_FACE_HALF; ++b) {
        pmk_mix_detail::band(FACE0_BASE + b * ROW_BAND_STRIDE, FACE1_BASE + b * ROW_BAND_STRIDE);
    }
    for (std::uint32_t b = 0; b < NUM_ROW_BANDS_PER_FACE_HALF; ++b) {
        pmk_mix_detail::band(FACE2_BASE + b * ROW_BAND_STRIDE, FACE3_BASE + b * ROW_BAND_STRIDE);
    }
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
