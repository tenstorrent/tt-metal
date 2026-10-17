// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, RoundingMode rounding_mode, int ITERATIONS>
inline void calculate_rdiv(const uint value) {
    sfpi::vFloat val = Converter::as_float(value);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat recip;
        if constexpr (APPROXIMATION_MODE) {
            recip = sfpu_reciprocal_iter<0>(in);
        } else {
            if constexpr (is_fp32_dest_acc_en) {
                recip = sfpu_reciprocal_iter<2>(in);
            } else {
                recip = sfpu_reciprocal_iter<1>(in);
                recip = sfpi::convert<sfpi::vFloat16b>(recip, sfpi::RoundMode::Nearest);
            }
        }
        sfpi::vFloat result = recip * val;

        if constexpr (rounding_mode != RoundingMode::None) {
            sfpi::vInt e_in = sfpi::exexp(in, sfpi::ExponentMode::Biased);
            sfpi::vInt e_res = sfpi::exexp(result, sfpi::ExponentMode::Biased);

            sfpi::vFloat q;
            if constexpr (rounding_mode == RoundingMode::Trunc) {
                q = _trunc_body_(result);
            } else {
                q = _floor_body_(result);
            }

            v_if (e_in != 0 && e_in < 253 && e_res != 255) {
                // Fix one-integer errors from the reciprocal product landing just below an integer.
                // Avoid the Newton residual step here; the discrete remainder invariant is enough
                // for the exact-divisible floor/trunc bug and keeps the rounded path smaller.
                sfpi::vFloat r = val - q * in;
                sfpi::vFloat rq = r * recip;  // sign only

                if constexpr (rounding_mode == RoundingMode::Floor) {
                    // floor invariant: remainder shares the divisor's sign, |r| < |in|
                    v_if (rq < 0.0f) { q = q - 1.0f; }
                    v_elseif (sfpi::abs(r) >= sfpi::abs(in)) { q = q + 1.0f; }
                    v_endif;
                } else {
                    // trunc invariant: |r| < |in|
                    v_if (sfpi::abs(r) >= sfpi::abs(in)) {
                        v_if (rq >= 0.0f) { q = q + 1.0f; }
                        v_else { q = q - 1.0f; }
                        v_endif;
                    }
                    v_endif;
                }
            }
            v_endif;
            result = q;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void rdiv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
