// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binary_comp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Compute isclose element-wise: result = |a - b| <= atol + rtol * |b|
//
// rtol_bits and atol_bits are the IEEE-754 bit-patterns of the tolerance
// scalars, passed as runtime uint32 arguments from the compute kernel and
// converted to float via Converter::as_float().
//
// EQUAL_NAN controls NaN semantics, matching torch.isclose:
//   false (default): any NaN input => result = 0
//   true:            both NaN      => result = 1; one NaN => result = 0
//
// Inputs are expected to be float32 or bfloat16. INT32 tensors must be
// promoted to FLOAT32 before reaching this kernel; invoke_binary_ng_isclose
// handles this via explicit ttnn::typecast calls before dispatch.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool EQUAL_NAN>
inline void calculate_sfpu_isclose(
    const uint32_t dst_index_in0,
    const uint32_t dst_index_in1,
    const uint32_t dst_index_out,
    uint32_t rtol_bits,
    uint32_t atol_bits) {
    constexpr uint32_t dst_tile_size_sfpi = 32;
    // IEEE-754 abs(+inf). abs_bits == inf_bits -> +-Inf; abs_bits > inf_bits -> NaN
    // (NaN has exp == 0xFF and a non-zero mantissa, irrespective of sign or
    // quiet/signaling bit). One comparison classifies both special cases.
    constexpr int32_t inf_bits = 0x7F800000;

    const sfpi::vFloat atol = Converter::as_float(atol_bits);
    const sfpi::vFloat rtol = Converter::as_float(rtol_bits);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Integer views of a, b and their abs bit patterns. Two constraints: the abs
        // bits must stay vInt (a vFloat against inf_bits binds to the float overload,
        // converting it to 2139095040.0f), and the mask cannot be replaced by
        // bit-casting sfpi::abs() because SFPABS float mode leaves -NaN sign-set.
        // vConstIntPrgm0 holds 0x7FFFFFFF, programmed in isclose_init.
        sfpi::vInt a_bits = sfpi::as<sfpi::vInt>(a);
        sfpi::vInt b_bits = sfpi::as<sfpi::vInt>(b);
        sfpi::vInt a_abs_bits = a_bits & sfpi::vConstIntPrgm0;
        sfpi::vInt b_abs_bits = b_bits & sfpi::vConstIntPrgm0;
        sfpi::vFloat b_abs = sfpi::abs(b);

        // abs(a - b) via sign-bit clear.
        sfpi::vFloat abs_diff = sfpi::abs(a - b);

        // tolerance = atol + rtol * |b|
        sfpi::vFloat tol = atol + rtol * b_abs;

        // |a - b| <= atol + rtol * |b|.
        //
        // For finite inputs this is the final answer. For +-Inf inputs the
        // tolerance formula yields inf<=inf=true even for mismatched signs,
        // and for NaN inputs the hardware comparison can be unreliable; both
        // are corrected in the single fix-up branch below.
        sfpi::vFloat result = 0.0f;
        v_if(abs_diff <= tol) { result = 1.0f; }
        v_endif;

        // Single fix-up branch covering every "special" lane (Inf or NaN).
        // Detected by `abs_bits >= inf_bits` which holds iff exp == 0xFF.
        // Inside, we discard the tolerance result and rebuild from scratch:
        //   * matching-sign Inf  -> 1     (a_abs_bits == inf AND a_bits == b_bits)
        //   * both NaN, EQUAL_NAN -> 1    (a_abs_bits >  inf AND b_abs_bits >  inf)
        //   * everything else (mismatched Inf, one-sided NaN, EQUAL_NAN=false
        //     NaN) stays at 0.
        // Folding both old fix-ups into one v_if removes two predicate-stack
        // push/pop pairs from the hot loop.
        v_if(a_abs_bits >= inf_bits || b_abs_bits >= inf_bits) {
            result = 0.0f;
            v_if(a_abs_bits == inf_bits && a_bits == b_bits) { result = 1.0f; }
            v_endif;
            if constexpr (EQUAL_NAN) {
                v_if(a_abs_bits > inf_bits && b_abs_bits > inf_bits) { result = 1.0f; }
                v_endif;
            }
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Programs the sign-clear mask into a constant register so the hot loop does not
// rebuild it with a per-element SFPLOADI inside the replay block.
inline void isclose_init() { sfpi::vConstIntPrgm0 = 0x7FFFFFFF; }

}  // namespace ckernel::sfpu
