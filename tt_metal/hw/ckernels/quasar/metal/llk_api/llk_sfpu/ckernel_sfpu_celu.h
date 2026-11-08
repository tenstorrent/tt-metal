// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu {

inline void celu_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

// celu(x) = x for x>=0, alpha*(exp(x/alpha)-1) for x<0

// NEG_ALPHA: alpha < 0 makes x / alpha positive and unbounded on the x < 0 branch. The helper
// clamps only the negative side, and its 2^k reconstruction writes k + 127 into an 8-bit exponent
// field, so past k = 128 it wraps and the result flips sign instead of overflowing. The clamp must
// stay below 128*ln(2) = 88.72 to avoid the NaN band at k=128 with r<0 (x/alpha in 88.38..88.72).
// At 88.3 the helper is finite (k=127), so alpha * finite is the correct overflow behavior.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_celu(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    // param0 is the alpha bit pattern; the sign test is decided on the RISC, outside the loop.
    const bool neg_alpha = (param0 & 0x80000000u) != 0;
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat arg = alpha_recip * x;
        if (neg_alpha) {
            arg = sfpi::min(arg, 88.3f);
        }
        sfpi::vFloat result = alpha * expm1_cw_clamped(arg);

        if (neg_alpha) {
            v_if(!sfpi::is_finite(x)) { result = x; }
            v_endif;
        }
        v_if(x >= 0.0f) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
