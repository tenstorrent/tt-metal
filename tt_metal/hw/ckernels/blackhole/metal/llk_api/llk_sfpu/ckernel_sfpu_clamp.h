// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_hardtanh.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void clamp_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// out = min(max(x, min_val), max_val)
//
// Both bounds are materialised once per call and the row is sfpi::clamp: SFPLOAD, SFPMOV/SFPSWAP twice, SFPSTORE,
// replayed. This is calculate_hardtanh's body. Measured on Blackhole against the raw-TTI alternative with the bounds
// in L12/L13 (SFPLOAD, two SFPSWAPs against the constant registers, SFPSTORE -- five instructions instead of seven):
// SFPSWAP leaves a one-cycle bubble that only an independent instruction may fill, and the compiler's two bound
// copies sit exactly there, so the seven-instruction body runs at 6.89 cycles/row and the five-instruction one at
// 7.02; interleaving two rows to fill the bubbles was slower still (7.48) and corrupted the int32 arms. Both forms
// gave bit-identical results on every format pair and edge class, so the faster, compiler-scheduled one is kept.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    calculate_hardtanh<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
}

// Row body of the int32 clamp for one (sign of min_val, sign of max_val) pair.
//
// SFPSWAP orders sign-magnitude integers. A two's-complement operand with the sign bit clear compares the same
// way, so it is swapped directly; one with the sign bit set is complemented on both sides of the compare (the
// bound once, by calculate_clamp_int32; the row here) with min and max exchanged, and complemented back. The
// complement that closes the max step and the one that opens the min step cancel when both bounds have the same
// sign, which is why the middle SFPNOT is emitted only when the signs differ.
template <bool MIN_NEG, bool MAX_NEG, int ITERATIONS>
sfpi_inline void calculate_clamp_int32_body() {
    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        if constexpr (MIN_NEG) {
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
            // ~L0 = min(~L0, ~min_val)  <=>  L0 = max(L0, min_val)
            TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        } else {
            // L0 = max(L0, min_val)
            TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);
        }
        if constexpr (MIN_NEG != MAX_NEG) {
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        }
        if constexpr (MAX_NEG) {
            // ~L0 = max(~L0, ~max_val)  <=>  L0 = min(L0, max_val)
            TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        } else {
            // L0 = min(L0, max_val)
            TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        }
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

// out = min(max(x, min_val), max_val) on two's-complement int32 (fp32 dest).
//
// Same one-pass shape as calculate_clamp: both bounds in L12/L13 once per call, then one SFPLOAD, two SFPSWAPs
// and one SFPSTORE per row plus the sign-dependent SFPNOTs described at calculate_clamp_int32_body (0 to 2 per
// row). The bounds are compile-time literals at every production call site, so the dispatch below folds to one
// body.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint min_val, uint max_val) {
    const bool min_neg = static_cast<int>(min_val) < 0;
    const bool max_neg = static_cast<int>(max_val) < 0;
    sfpi::vConstIntPrgm0 = min_neg ? ~min_val : min_val;  // L12
    sfpi::vConstIntPrgm1 = max_neg ? ~max_val : max_val;  // L13
    if (min_neg) {
        if (max_neg) {
            calculate_clamp_int32_body<true, true, ITERATIONS>();
        } else {
            calculate_clamp_int32_body<true, false, ITERATIONS>();
        }
    } else {
        if (max_neg) {
            calculate_clamp_int32_body<false, true, ITERATIONS>();
        } else {
            calculate_clamp_int32_body<false, false, ITERATIONS>();
        }
    }
}

}  // namespace ckernel::sfpu
