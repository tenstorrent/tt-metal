// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_cumsum.h"

namespace ckernel {
namespace sfpu {

// cumprod_init records one SFPMUL for each LREG: 4 instructions per pass. No SFPNOPs: Blackhole stalls
// an SFPU instruction that reads the previous SFPMUL's result.
constexpr std::uint32_t cumprod_step = 4;

// The cumsum tile walk, replaying the product chain that cumprod_init records.
template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_cumprod(const bool first) {
    if (first) {
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 4, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 5, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 6, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 7, 0);
    }
    scan_tile_columns<cumprod_step>();
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en>
inline void cumprod_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    load_replay_buf(0, 2 * cumprod_step, [] {
        constexpr std::uint32_t mod1 = is_fp32_dest_acc_en ? 2 : 0;
        TTI_SFPMUL(0, 7, p_sfpu::LCONST_0, 0, mod1);
        TTI_SFPMUL(1, 0, p_sfpu::LCONST_0, 1, mod1);
        TTI_SFPMUL(2, 1, p_sfpu::LCONST_0, 2, mod1);
        TTI_SFPMUL(3, 2, p_sfpu::LCONST_0, 3, mod1);
        TTI_SFPMUL(4, 3, p_sfpu::LCONST_0, 4, mod1);
        TTI_SFPMUL(5, 4, p_sfpu::LCONST_0, 5, mod1);
        TTI_SFPMUL(6, 5, p_sfpu::LCONST_0, 6, mod1);
        TTI_SFPMUL(7, 6, p_sfpu::LCONST_0, 7, mod1);
    });
}

}  // namespace sfpu
}  // namespace ckernel
