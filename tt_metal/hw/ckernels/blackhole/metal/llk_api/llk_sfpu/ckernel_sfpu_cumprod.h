// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_cumsum.h"

namespace ckernel {
namespace sfpu {

// calculate_cumsum's tile walk, replaying the product chain that cumprod_init records.
template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumprod(const bool first) {
    if (first) {
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 4, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 5, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 6, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_1, 7, 0);
    }
    calculate_cumsum<APPROXIMATION_MODE, ITERATIONS>(/*first=*/false);
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumprod_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    load_replay_buf(0, 16, [] {
        TTI_SFPMUL(0, 7, p_sfpu::LCONST_0, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(1, 0, p_sfpu::LCONST_0, 1, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(2, 1, p_sfpu::LCONST_0, 2, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(3, 2, p_sfpu::LCONST_0, 3, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(4, 3, p_sfpu::LCONST_0, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(5, 4, p_sfpu::LCONST_0, 5, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(6, 5, p_sfpu::LCONST_0, 6, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(7, 6, p_sfpu::LCONST_0, 7, 0);
        TTI_SFPNOP;
    });
}

}  // namespace sfpu
}  // namespace ckernel
