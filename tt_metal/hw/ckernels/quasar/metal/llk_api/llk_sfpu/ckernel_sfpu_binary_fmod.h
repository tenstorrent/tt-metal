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
// Implemented using 32-bit integer remainder kernel (see ckernel_sfpu_remainder_int32.h)
sfpi_inline void calculate_fmod_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Read inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // FMOD sign handling (result has the same sign as a)
    v_if(a_signed < 0) { r = -r; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_fmod_(sfpi::vFloat a, sfpi::vFloat b) {
    // fmod(a, b) = a - trunc(a/b) * b on the magnitudes; the sign of a is applied once at the end.
    sfpi::vFloat a_abs = sfpi::abs(a);
    sfpi::vFloat b_abs = sfpi::abs(b);

    sfpi::vFloat recip = ckernel::sfpu::_sfpu_reciprocal_<2>(b_abs);
    sfpi::vFloat trunc_div = _trunc_body_(a_abs * recip);
    sfpi::vFloat result = a_abs - trunc_div * b_abs;

    // The reciprocal error can leave trunc(a/b) one too low, or up to two too high (quotients above
    // 2^24), so step the remainder back into [0, |b|).
    v_if(result >= b_abs) { result = result - b_abs; }
    v_endif;
    v_if(result < sfpi::vFloat(0.0f)) { result = result + b_abs; }
    v_endif;
    v_if(result < sfpi::vFloat(0.0f)) { result = result + b_abs; }
    v_endif;

    // Handle special cases using conditional assignment (NOT early return!)
    // When a == b, fmod(a, b) = 0
    v_if(a == b) { result = sfpi::vFloat(0.0f); }
    v_endif;

    // fmod takes the sign of a, a zero remainder included (fmod(-6, 3) = -0, as std::fmod).
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
#pragma GCC unroll 8
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
    _init_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
