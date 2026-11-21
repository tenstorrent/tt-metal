// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity() {
#ifdef DISABLE_SFPLOADMACRO
    // Each row by its own offset: no counter step between rows.
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[d];
        dst_reg[d] = v;
    }
#else
    // Macro 0 (identity_init) stores each loaded row back on the next cycle: one issue slot per row.
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOADMACRO((0 << 2) | p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, d << 1);
    }
    // Covers the last row's scheduled store.
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity_uint() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void identity_init() {
#ifndef DISABLE_SFPLOADMACRO
    // Macro 0: only a store of the loaded register, no delay.
    constexpr std::uint32_t store_bits = (0 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, store_bits << 8);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    // Store format DEFAULT, delays counted in cycles.
    TTI_SFPCONFIG(0, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
