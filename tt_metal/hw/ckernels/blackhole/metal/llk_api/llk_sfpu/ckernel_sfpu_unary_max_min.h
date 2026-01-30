// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

sfpi_inline void load_value_param_float(uint value) { sfpi::vConstIntPrgm0 = value; }

template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_float_body() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

    if constexpr (IS_MAX_OP) {
        // L0 = max(L0, constant); this will only write to L0 since L12 is a constant register.
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, 9); // mod1=9 means set VD=max and VC=min
    } else {
        // L0 = min(L0, constant); this will only write to L0 since L12 is a constant register.
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
}

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min(uint value) {
    load_value_param_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_max_min_float_body<IS_MAX_OP>();
        sfpi::dst_reg++;
    }
}

sfpi_inline void load_value_param_int(uint value) {
    if (value != 0x80000000u) {
        // if msb(value) == 1, we need to invert for SFPSWAP to work
        sfpi::vConstIntPrgm0 = (int)value >= 0 ? value : ~value;
    } else {
        // set L0 = 0x80000000
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
    }
}

template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_int32_body(uint value) {
    if (value != 0x80000000u) {
        if ((int)value >= 0) {
            // if msb(value) == 0, we can safely use SFPSWAP even though it expects sign-magnitude integers
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // mod1=9 means set VD=max and VC=min
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        } else {
            // if msb(value) == 1, we need to invert both values for SFPSWAP to work
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
            TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, IS_MAX_OP ? sfpi::SFPSWAP_MOD1_VEC_MIN_MAX : 9); // mod1=9 means set VD=max and VC=min
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        }
    } else if constexpr (!IS_MAX_OP) {
        // if value == INT_MIN, then min(x, value) = INT_MIN
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    } else {
        // if value == INT_MIN, then max(x, value) = x; do nothing
    }
}

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min_int32(uint value) {
    load_value_param_int(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_max_min_int32_body<IS_MAX_OP>(value);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
