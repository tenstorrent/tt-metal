// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_binary_remainder.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "llk_math_eltwise_sfpu_op.h"

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
sfpi_inline sfpi::vFloat _sfpu_binary_fmod_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // fmod(a, b) = a - trunc(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;
    sfpi::vFloat b_abs = sfpi::abs(b);

    // Compute reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::sfpu_reciprocal_iter<2>(b);

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    sfpi::vFloat trunc_div = _trunc_body_(div_result);

    // Compute fmod = a - trunc(a/b) * b
    sfpi::vFloat result = a - trunc_div * b;

    // Post-correction - fmod result must satisfy |result| < |b|
    // If |result| >= |b|, the truncation was wrong by 1
    sfpi::vFloat result_abs = sfpi::abs(result);

    // If result >= b, we truncated too low, add/subtract b to correct
    v_if(result_abs >= b_abs) {
        // Determine correction direction based on sign of result
        v_if(result >= sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // result was positive and too big
        }
        v_else {
            result = result + b_abs;  // result was negative and too big (magnitude)
        }
        v_endif;
    }
    v_endif;

    // Sign correction - fmod result must have same sign as 'a' (or be zero)
    // If a > 0 and result < 0, the truncation was 1 too high, need to add b
    // If a < 0 and result > 0, the truncation was 1 too low, need to subtract b
    // This fixes cases where a/b ≈ 0.9999999 but rounds to 1 due to reciprocal error
    v_if(a >= sfpi::vFloat(0.0f)) {
        // a is positive, result should be >= 0
        v_if(result < sfpi::vFloat(0.0f)) {
            result = result + b_abs;  // over-truncated
        }
        v_endif;
    }
    v_else {
        // a is negative, result should be <= 0
        v_if(result > sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // under-truncated
        }
        v_endif;
    }
    v_endif;

    // Handle special cases using conditional assignment (NOT early return!)
    // When a == b, fmod(a, b) = 0
    v_if(a == b) { result = sfpi::vFloat(0.0f); }
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
inline void calculate_fmod_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
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
    sfpu_reciprocal_init<false>();
}

// ---------------------------------------------------------------------------------------------------
// BinaryFmod<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>
//   FORMAT = Int32 -> calculate_fmod_int32 / fmod_int32_init
//   other (float)  -> calculate_sfpu_binary_fmod<.., DST_ACCUM> / fmod_binary_init
//   Backs fmod_int32_tile(_init) / fmod_binary_tile(_init) (api/compute/binary_fmod.h).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct BinaryFmod
    : SfpuBinaryOp<BinaryFmod<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static constexpr bool is_int32 = FORMAT == DataFormat::Int32;

    static void kernel(uint32_t dst_index_in0, uint32_t dst_index_in1, uint32_t dst_index_out) {
        if constexpr (is_int32) {
            calculate_fmod_int32<APPROXIMATION_MODE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
        } else {
            calculate_sfpu_binary_fmod<APPROXIMATION_MODE, ITERATIONS, DST_ACCUM>(
                dst_index_in0, dst_index_in1, dst_index_out);
        }
    }

    static void init_kernel() {
        if constexpr (is_int32) {
            fmod_int32_init<APPROXIMATION_MODE>();
        } else {
            fmod_binary_init<APPROXIMATION_MODE>();
        }
    }
};
}  // namespace ckernel::sfpu
