// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_logaddexp.h"
#include "ckernel_sfpu_conversions.h"

namespace ckernel::sfpu {

// logaddexp2(a, b) = max(a, b) + log2(1 + 2^-|a - b|)
//
// The composed form, log2(2^a + 2^b), overflows at |x| > 127 (2^128 is not
// representable) and underflows below -149, even though the result is bounded by its
// own inputs: max(a,b) <= logaddexp2(a,b) <= max(a,b) + 1. Here the argument of the
// power is -|a - b| <= 0, so it lands in (0, 1] and cannot overflow; the magnitude comes
// from max(a, b), which is representable by assumption.
//
// Evaluated through the base-e primitives that ckernel_sfpu_logaddexp.h already relies on:
//     2^-|a - b|    = exp(-|a - b| * ln 2)
//     log2(1 + t)   = log1p(t) * log2(e)
// No new polynomial: log1p sees exactly the same (0, 1] argument range as it does in
// logaddexp, so its existing coefficient set applies unchanged. A variant with log2(e)
// folded into its own fit was simulated against torch.logaddexp2 over 262144 pairs from
// U(-200, 200) and was indistinguishable from this form at every percentile measured; the
// dominant error is the cancellation in max + correction, not the extra multiply.
//
// Written with at most three live vFloat values, for the same register-spill reason
// documented in ckernel_sfpu_logaddexp.h.
//
// Equal infinities and NaN operands are handled by _sfpu_logaddexp_max_ and
// _sfpu_logaddexp_gap_ in ckernel_sfpu_logaddexp.h, shared with logaddexp.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    // ln 2 and log2(e). Both are the correctly rounded float32 nearest values.
    constexpr float LN_TWO = 0.693147182f;
    constexpr float LOG2_E = 1.442695041f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_logaddexp_max_(a, b);
        _sfpu_logaddexp_gap_(a, b);
        // The accurate exponential is required for the same reason as in logaddexp: the
        // approximate body returns 255/256 rather than 1 at zero, and here that error
        // lands on the correction term whose exact value at |a - b| = 0 is 1.
        b = _sfpu_exp_fp32_accurate_(a * -LN_TWO);
        result = result + calculate_log1p_fp32<is_fp32_dest_acc_en>(b) * LOG2_E;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Identical to logaddexp's init, and for the same reason: log1p reads its polynomial
// coefficients from the program constant registers and an SFPU helper called from
// another op's kernel does not carry its own initialisation. The coefficient set differs
// by destination precision, which is why this init is templated.
//
// The base conversion lives in the kernel above, not here, so these constants stay
// exactly the ones log1p expects.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp2_init() {
    // Delegating to log1p_init rather than copying its values keeps one source for the
    // tuned coefficients: a retune of the log1p polynomial reaches this op instead of
    // silently desyncing from it. log1p_init ignores its first two template parameters.
    log1p_init<false /* APPROXIMATION_MODE */, false /* FAST_APPROX */, is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
