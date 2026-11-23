// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_binary_remainder.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_rounding_ops.h"

namespace ckernel::sfpu {

// FMOD = a - trunc(a / b) * b
// Implemented using 32-bit integer remainder kernel (see ckernel_sfpu_binary_remainder.h)
sfpi_inline void calculate_fmod_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load signed inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // Initialize a fresh value: assignment would retain a_signed for inactive
    // lanes across the helper, increasing register pressure in full LTO builds.
    sfpi::vInt a_reloaded = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // FMOD sign handling (result has the same sign as a)
    v_if(a_reloaded < 0) { r = -r; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_fmod_(sfpi::vFloat a, sfpi::vFloat b) {
    // fmod(a, b) = a - trunc(a/b) * b, evaluated on |a| and |b| so the corrections
    // below are sign-independent; the sign of a is applied once at the end.
    sfpi::vFloat a_abs = sfpi::abs(a);
    sfpi::vFloat b_abs = sfpi::abs(b);

    sfpi::vFloat recip = ckernel::sfpu::sfpu_reciprocal_iter<2>(b_abs);
    sfpi::vFloat trunc_div = _trunc_body_(a_abs * recip);
    sfpi::vFloat result = a_abs - trunc_div * b_abs;

    // The reciprocal error can put trunc(a/b) a unit off (several units for quotients
    // above 2^22). One step down and two steps up bring the result into [0, |b|) when
    // trunc is at most one unit too low or two units too high; larger misses stay out
    // of range.
    v_if(result >= b_abs) { result = result - b_abs; }
    v_endif;
    v_if(result < sfpi::vFloat(0.0f)) { result = result + b_abs; }
    v_endif;
    v_if(result < sfpi::vFloat(0.0f)) { result = result + b_abs; }
    v_endif;

    // Finite a == b needs no override: trunc gives 0 or 1 and the corrections yield 0.
    // a == b = +-inf or NaN gives NaN, as in torch.

    // Zero results take the sign of a too: fmod(-4, 2) = -0, as in torch.
    result = sfpi::copysgn(result, a);

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }

    return result;
}

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_fmod_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_fmod_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_fmod(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_fmod_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void fmod_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void fmod_binary_init() {
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu
