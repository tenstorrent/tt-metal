// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_binary_comp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Constant-register contents shared by isclose_init and calculate_sfpu_isclose.
// vConstIntPrgm0: sign-clear mask, x & kIscloseAbsMask is the |x| bit pattern.
// vConstIntPrgm1: abs(+inf) bit pattern.
constexpr std::int32_t kIscloseAbsMask = 0x7FFFFFFF;
constexpr std::int32_t kIscloseInfBits = 0x7F800000;

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
    // vConstIntPrgm1 holds kIscloseInfBits, programmed in isclose_init, so the hot loop
    // reads it as a register operand instead of re-materialising it per use.
    const sfpi::vInt inf_bits = sfpi::vConstIntPrgm1;

    const sfpi::vFloat atol = Converter::as_float(atol_bits);
    const sfpi::vFloat rtol = Converter::as_float(rtol_bits);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Integer views of a, b and their abs bit patterns. Two constraints: the abs
        // bits must stay vInt so the compares against inf_bits below are integer
        // compares on the raw bit patterns, and the mask cannot be replaced by
        // bit-casting sfpi::abs() because SFPABS float mode leaves -NaN sign-set.
        // vConstIntPrgm0 holds kIscloseAbsMask, programmed in isclose_init.
        sfpi::vInt a_bits = sfpi::as<sfpi::vInt>(a);
        sfpi::vInt b_bits = sfpi::as<sfpi::vInt>(b);
        sfpi::vInt a_abs_bits = a_bits & sfpi::vConstIntPrgm0;
        sfpi::vInt b_abs_bits = b_bits & sfpi::vConstIntPrgm0;
        // |b| for the tolerance is the same sign-cleared pattern, so reuse it rather than
        // spend an SFPABS. The two differ only for b = -NaN (SFPABS keeps the sign), a lane
        // the special-value fix-up below overwrites whatever the tolerance compare returned.
        sfpi::vFloat b_abs = sfpi::as<sfpi::vFloat>(b_abs_bits);

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

// Programs the sign-clear mask and the abs(+inf) bit pattern into constant registers so
// the hot loop does not rebuild them with per-element SFPLOADIs inside the replay block.
inline void isclose_init() {
    sfpi::vConstIntPrgm0 = kIscloseAbsMask;
    sfpi::vConstIntPrgm1 = kIscloseInfBits;
}

}  // namespace ckernel::sfpu
