// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binary_comp.h"
#include "sfpi.h"

#ifndef ISCLOSE_ATOL_VAL
#define ISCLOSE_ATOL_VAL 1e-8f
#endif
#ifndef ISCLOSE_RTOL_VAL
#define ISCLOSE_RTOL_VAL 1e-5f
#endif
#ifndef ISCLOSE_EQUAL_NAN
#define ISCLOSE_EQUAL_NAN 0
#endif

namespace ckernel::sfpu {

// Compute isclose element-wise: result = |a - b| <= atol + rtol * |b|
//
// ISCLOSE_RTOL_VAL and ISCLOSE_ATOL_VAL are injected as compile-time float
// literal defines by the host program factory (e.g. "1.000000000e-05f").
//
// EQUAL_NAN controls NaN semantics, matching torch.isclose:
//   false (default): any NaN input ⇒ result = 0
//   true:            both NaN     ⇒ result = 1; one NaN ⇒ result = 0
//
// Inputs are expected to be float32 or bfloat16. INT32 tensors must be
// promoted to FLOAT32 before reaching this kernel; invoke_binary_ng_isclose
// handles this via explicit ttnn::typecast calls before dispatch.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool EQUAL_NAN>
inline void calculate_sfpu_isclose(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // abs(a - b)
        sfpi::vFloat diff = a - b;
        sfpi::vInt diff_bits = sfpi::reinterpret<sfpi::vInt>(diff);
        sfpi::vFloat abs_diff = sfpi::reinterpret<sfpi::vFloat>(diff_bits & 0x7FFFFFFF);

        // abs(b) — cache int bits to reuse in the NaN check below
        sfpi::vInt b_bits = sfpi::reinterpret<sfpi::vInt>(b);
        sfpi::vInt b_abs_bits = b_bits & 0x7FFFFFFF;
        sfpi::vFloat abs_b = sfpi::reinterpret<sfpi::vFloat>(b_abs_bits);

        // tolerance = atol + rtol * |b|
        sfpi::vFloat tol = ISCLOSE_ATOL_VAL + ISCLOSE_RTOL_VAL * abs_b;

        // |a - b| <= atol + rtol * |b|
        sfpi::vFloat result = sfpi::vConst0;
        v_if(abs_diff <= tol) { result = sfpi::vConst1; }
        v_endif;

        // Inf fix-up: torch.isclose considers two infinities close only when
        // they are the same infinity (+inf==+inf or -inf==-inf).  The tolerance
        // formula yields inf <= inf = true for mismatched infinities, so we
        // override: if either operand is infinite, result = (a bits == b bits).
        sfpi::vInt a_abs_bits = sfpi::reinterpret<sfpi::vInt>(a) & 0x7FFFFFFF;
        v_if(is_inf(a_abs_bits) || is_inf(b_abs_bits)) {
            result = sfpi::vConst0;
            v_if(sfpi::reinterpret<sfpi::vInt>(a) == b_bits) { result = sfpi::vConst1; }
            v_endif;
        }
        v_endif;

        // NaN fix-up: hardware comparisons may not reliably produce 0 for NaN
        // inputs, so we apply an explicit correction.
        if constexpr (EQUAL_NAN) {
            // Nest "both NaN → 1" inside "any NaN → 0" to avoid evaluating the
            // inner predicate on clean (non-NaN) lanes.
            v_if(is_nan(a_abs_bits) || is_nan(b_abs_bits)) {
                result = sfpi::vConst0;
                v_if(is_nan(a_abs_bits) && is_nan(b_abs_bits)) { result = sfpi::vConst1; }
                v_endif;
            }
            v_endif;
        } else {
            // Any NaN input ⇒ result = 0
            v_if(is_nan(a_abs_bits) || is_nan(b_abs_bits)) { result = sfpi::vConst0; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
