// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel::sfpu {

// 2^31 as float (used for INT32 sign-magnitude conversion edge cases)
constexpr float TWO_POW_31 = 2147483648.0f;

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
// Returns: unsigned remainder r
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(const sfpi::vInt& a_signed, const sfpi::vInt& b_signed) {
    // Get absolute value of b for reciprocal computation
    sfpi::vMag b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle edge case: if conversion results in negative
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    // Compute reciprocal of b
    sfpi::vFloat neg_b_f = sfpi::copyman(-1.0f, b_f);

    sfpi::vFloat inv_b_f = sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1 * neg_b_f;

    sfpi::vFloat scale = sfpi::setman(b_f, 0);

    // First Newton-Raphson iteration: inv_b_f = inv_b_f * (2 - inv_b_f * b_f)
    sfpi::vFloat t = inv_b_f * neg_b_f + 1.0f;
    scale = sfpi::as<sfpi::vFloat>((254 << 23) - sfpi::as<sfpi::vInt>(scale));
    inv_b_f = t * inv_b_f + inv_b_f;

    // Second Newton-Raphson iteration (interleaved with abs(a) computation)
    sfpi::vFloat e = inv_b_f * neg_b_f + 1.0f;
    sfpi::vMag a = sfpi::abs(a_signed);
    inv_b_f = e * inv_b_f + inv_b_f;

    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
    v_endif;

    // Apply scaling factor to finalize reciprocal
    inv_b_f = inv_b_f * scale;

    // Initial quotient approximation : q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vMag q = sfpi::exman(q_f);

    // Recompute b for chunk extraction to reduce register pressure
    b = sfpi::abs(b_signed);

    // 8388608.0f = 2^23 is used as a Bias for mantissa alignment
    sfpi::vFloat MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;

    // Split q and b into 11-bit chunks to compute q * b
    sfpi::vMag MASK_11{0x7ff};
    sfpi::vFloat q1 = sfpi::convert<sfpi::vFloat>(q & MASK_11, sfpi::RoundMode::Nearest);
    sfpi::vFloat q2 = sfpi::convert<sfpi::vFloat>(q >> 11, sfpi::RoundMode::Nearest);
    sfpi::vFloat b1 = sfpi::convert<sfpi::vFloat>((b >> 11) & MASK_11, sfpi::RoundMode::Nearest);
    sfpi::vFloat b0 = sfpi::convert<sfpi::vFloat>(b & MASK_11, sfpi::RoundMode::Nearest);

    // hi = q2 * b0 + q1 * b1 (high part)
    // lo = q1 * b0 (low part)
    sfpi::vFloat hi = q2 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat lo = q1 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    hi = q1 * b1 + hi;

    sfpi::vUInt qb = sfpi::exman(lo) << 11;
    qb += sfpi::exman(hi) << 22;

    // Compute remainder - recompute abs(a_signed)
    a = sfpi::abs(a_signed);
    sfpi::vInt r{a - qb};

    // Use abs(r) for correction computation
    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r), sfpi::RoundMode::Nearest);

    // Compute correction: r / b in float32
    sfpi::vFloat correction_f = r_f * inv_b_f;
    auto correction = sfpi::convert<sfpi::vUInt16>(correction_f, sfpi::RoundMode::Nearest);
    correction_f = sfpi::convert<sfpi::vFloat>(correction, sfpi::RoundMode::Nearest);

    // Recompute b chunks for correction multiplication to reduce register pressure
    b = sfpi::abs(b_signed);
    b0 = sfpi::convert<sfpi::vFloat>(b & MASK_11, sfpi::RoundMode::Nearest);
    b1 = sfpi::convert<sfpi::vFloat>((b >> 11) & MASK_11, sfpi::RoundMode::Nearest);
    sfpi::vFloat b2 = sfpi::convert<sfpi::vFloat>(b >> 22, sfpi::RoundMode::Nearest);

    // tmp = correction * (b2<<22 + b1<<11 + b0)
    sfpi::vFloat low = correction_f * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat mid = correction_f * b1 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat top = correction_f * b2 + MANTISSA_ALIGNMENT_OFFSET;

    sfpi::vInt tmp{sfpi::exman(low) + (sfpi::exman(mid) << 11) + (sfpi::exman(top) << 22)};
    v_if(r < 0) { tmp = -tmp; }
    v_endif;
    r -= tmp;

    // Final adjustment - recompute b to reduce register pressure
    b = sfpi::abs(b_signed);
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;

    return r;
}

// Signed (int32) remainder = a - floor(a / b) * b
sfpi_inline void calculate_remainder_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load signed inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // Remainder sign handling
    sfpi::vInt sign = a_signed ^ b_signed;
    v_if(r != 0) {
        v_if(sign < 0) {
            // When signs differ, floor(a/b) = trunc(a/b) - 1, so remainder needs adjustment
            v_if(a_signed < 0) { r = b_signed - r; }
            v_else { r += b_signed; }
            v_endif;
        }
        v_elseif(a_signed < 0 && b_signed < 0) { r = -r; }
        v_endif;
    }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = r;
}

// Unsigned (uint32) remainder. compute_unsigned_remainder_int32() is exact only when both
// operands are in [0, 2^31) (abs() is a no-op there), so we range-reduce into that regime:
// * b <  2^31: halve a to clear the problematic top bit. With t = a >> 1 (logical) and
//              a = 2*t + (a & 1), a % b = (2*(t % b) + (a & 1)) % b. t < 2^31 for every uint32 a,
//              so the single helper call always sees operands in [0, 2^31).
// * b >= 2^31: a < 2^32 <= 2*b, so a is already in [0, 2b) and needs no helper (a % b = a or a - b).
// Both regimes yield a value x in [0, 2b), reduced by one conditional subtract: x % b =
// (x >=u b) ? x - b : x. The SFPU integer compare only tests sign(x - b), which equals the true
// unsigned x >=u b except when b >= 2^31 and x < 2^31; a second predicate corrects those lanes
// (there x < b, so the remainder is x).
sfpi_inline void calculate_remainder_uint32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load raw 32-bit patterns (interpreted as unsigned)
    sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // Call the helper unconditionally (nesting it inside predication crashes the SFPI rvtt_live
    // pass). t = (uint32)a >> 1 is always < 2^31, so the helper sees valid [0, 2^31) operands; rt
    // is only used on the b < 2^31 lanes, but every lane pays the call.
    sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
    sfpi::vInt rt = compute_unsigned_remainder_int32(t, b);

    // Reload a from DEST instead of keeping it live across the helper
    a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // b < 2^31 uses x = 2*rt + (a & 1); b >= 2^31 keeps x = a
    v_if(b >= 0) { a = rt + rt + (a & 1); }
    v_endif;

    // x % b = (x >=u b) ? x - b : x, valid for both regimes since x in [0, 2b)
    sfpi::vInt r = a;
    v_if(sfpi::vUInt(a) >= sfpi::vUInt(b)) { r = a - b; }
    v_endif;
    // The above compare only tests sign(x - b), matching x >=u b except when b >= 2^31 and x < 2^31
    // Then x < b, remainder = x
    v_if(b < 0 && a >= 0) { r = a; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_remainder_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // remainder(a, b) = a - floor(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * sfpu_reciprocal_iter<2>(b);

    // Compute floor(a/b)
    sfpi::vFloat floor_div = _floor_body_(div_result);

    // Compute remainder = a - floor(a/b) * b
    sfpi::vFloat result = a - floor_div * b;

    // Sign correction: remainder must match the sign of b (or be zero).
    // XOR of the float bit-patterns detects sign mismatch via the MSB,
    // avoiding a compound conditional with four comparisons and an OR.
    v_if(result != sfpi::vFloat(0.0f)) {
        sfpi::vInt signs = sfpi::as<sfpi::vInt>(result) ^ sfpi::as<sfpi::vInt>(b);
        v_and(signs < 0);
        result += b;
    }
    v_endif;

    // Magnitude correction: reciprocal imprecision can cause floor() to be greater than the true floor value.
    v_if(b > sfpi::vFloat(0.0f) && a > sfpi::vFloat(0.0f)) {
        sfpi::vFloat diff = result - b;
        v_if(diff >= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_endif;
    v_if(b < sfpi::vFloat(0.0f) && a < sfpi::vFloat(0.0f)) {
        sfpi::vFloat diff = result - b;
        v_if(diff <= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_endif;

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_uint32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_uint32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_remainder(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_remainder_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void remainder_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void remainder_uint32_init() {
    // Shares the int32 setup: the unsigned path reuses compute_unsigned_remainder_int32().
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void remainder_binary_init() {
    recip_init<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

}  // namespace ckernel::sfpu
