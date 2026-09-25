// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_exp2.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_logaddexp.h"

namespace ckernel::sfpu {

// logaddexp2(a, b) = max(a, b) + log2(1 + 2^-|a - b|)
//
// The composed form, log2(2^a + 2^b), overflows at |x| > 127 (2^128 is not
// representable) and underflows below -149, even though the result is bounded by its
// own inputs: max(a,b) <= logaddexp2(a,b) <= max(a,b) + 1. Here the argument of the
// power is -|a - b| <= 0, so it lands in (0, 1] and cannot overflow; the magnitude comes
// from max(a, b), which is representable by assumption.
//
// Evaluated through the primitives that ckernel_sfpu_logaddexp.h already relies on:
//     2^-|a - b|    = exp(-|a - b| * ln 2)    for an fp32 destination
//     2^-|a - b|    by exp2's bfloat16 body   for a bfloat16 destination
//     log2(1 + t)   = log1p(t) * log2(e)
// No new polynomial: log1p sees exactly the same (0, 1] argument range as it does in
// logaddexp, so the same log1p applies, calculate_log1p_fp32 for fp32 and
// _sfpu_logaddexp_log1p_unit_bf16_ for bfloat16, with the coefficients logaddexp's init
// loads. A variant with log2(e) folded into its own fit was simulated against
// torch.logaddexp2 over 262144 pairs from U(-200, 200) and was indistinguishable from this
// form at every percentile measured; the dominant error is the cancellation in
// max + correction, not the extra multiply.
//
// The two inputs are dead once max(a, b) and the gap are known, for the same
// register-spill reason documented in ckernel_sfpu_logaddexp.h.
//
// Equal infinities and NaN operands are handled by _sfpu_logaddexp_max_gap_ in
// ckernel_sfpu_logaddexp.h, shared with logaddexp.
//
// APPROXIMATION_MODE is accepted and ignored, as in log1p_init: the exponential below is
// chosen by the destination precision instead, and is never the approximate body, which is
// not accurate enough here.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    // ln 2 and log2(e). Both are the correctly rounded float32 nearest values.
    constexpr float LN_TWO = 0.693147182f;
    constexpr float LOG2_E = 1.442695041f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result;
        a = _sfpu_logaddexp_max_gap_(a, b, result);
        // The exponential follows the destination precision, as in logaddexp. For fp32 it is
        // the guarded exponential, for the reason given there. For bfloat16, _sfpu_exp2_bf16_
        // is what exp2 uses for a bfloat16 result: logaddexp's exp_21f taken in base 2, so the
        // ln 2 multiply drops out, and like logaddexp's it rounds 2^0 to exactly 1. The
        // approximate body is not usable: its 255/256 at zero lands on the correction term
        // whose exact value at |a - b| = 0 is 1.
        if constexpr (is_fp32_dest_acc_en) {
            b = _sfpu_exp_fp32_accurate_(a * -LN_TWO);
            result = result + calculate_log1p_fp32<true>(b) * LOG2_E;
        } else {
            b = _sfpu_exp2_bf16_(-a);
            result = result + _sfpu_logaddexp_log1p_unit_bf16_(b) * LOG2_E;

            // Rounded as in logaddexp: SFPSTORE would truncate, and the hardware rounds a tie
            // away from zero rather than to even.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Identical to logaddexp's init, and for the same reason: the log1p correction reads its
// polynomial coefficients from the program constant registers and an SFPU helper called
// from another op's kernel does not carry its own initialisation. The coefficient set
// differs by destination precision, which is why this init is templated.
//
// The base conversion lives in the kernel above, not here, so these constants stay
// exactly the ones the log1p correction expects.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp2_init() {
    // Identical setup to logaddexp: both need the log1p coefficients and nothing else. This
    // forwards instead of repeating the setup so the two cannot drift apart.
    calculate_sfpu_logaddexp_init<is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
