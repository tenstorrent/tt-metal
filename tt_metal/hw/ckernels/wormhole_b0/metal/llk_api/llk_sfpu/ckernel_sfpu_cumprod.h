// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_cumsum.h"

namespace ckernel::sfpu {

// cumprod_init records an SFPMUL and an SFPNOP for each LREG: 8 instructions per pass.
constexpr std::uint32_t cumprod_step = 8;

// The cumsum tile walk, replaying the product chain that cumprod_init records.
template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_cumprod(const bool first) {
    if (first) {
        set_carried_lregs<p_sfpu::LCONST_1>();
    }
    scan_tile_columns<cumprod_step>();
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumprod_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    lltt::record(0, 2 * cumprod_step);
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
}

}  // namespace ckernel::sfpu
