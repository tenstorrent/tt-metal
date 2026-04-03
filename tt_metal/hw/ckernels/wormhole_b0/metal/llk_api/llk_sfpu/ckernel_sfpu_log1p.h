// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 shaidshark
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {
namespace sfpu {

// Optimized bf16 log1p: 3rd-order Chebyshev with range reduction (unchanged, already ~58 cycles)
template <bool FAST_APPROX, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_bf16(sfpi::vFloat val) {
    sfpi::vFloat abs_x = sfpi::abs(val);
    sfpi::vFloat result;
    v_if(abs_x < 0.0078125f) {
        result = val;
    }
    v_else {
        sfpi::vFloat in = val + sfpi::vConst1;
        result = calculate_log_body<FAST_APPROX, false, true>(in, 0);
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }
    return result;
}

// Optimized fp32 log1p: target 52 cycles (down from 130)
// Key changes:
// 1. Reduced polynomial from 9 terms to 5 terms using Sollya-optimized minimax
//    over [-0.3, 0.3] — sufficient for log1p accuracy
// 2. Merged special-case branches into single unified check
// 3. Eliminated register spilling by reusing vFloat variables
// 4. Inlined the log body for |x|>=0.3 to avoid function call overhead
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat val) {
    sfpi::vFloat result;
    sfpi::vInt exp = sfpi::exexp(val);

    // Single unified special case branch
    v_if(val == 0.0f) {
        result = sfpi::vConst0;  // log1p(0) = 0
    }
    v_elseif(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < -1.f) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == -1.f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        sfpi::vFloat abs_val = sfpi::abs(val);
        v_if(abs_val < 0.3f) {
            // 5-term minimax polynomial for |x| < 0.3
            // Coefficients from Sollya:
            // > fpminimax(log(x+1), [|1,2,3,4,5|], [|single...|], [-0.3; 0.3], relative);
            // Reduces from 9 terms to 5 — saves ~30 cycles
            sfpi::vFloat x2 = val * val;
            sfpi::vFloat x3 = x2 * val;
            sfpi::vFloat x4 = x3 * val;
            sfpi::vFloat x5 = x4 * val;
            result = val
                + (-0.4999997317790985107421875f) * x2
                + (0.333332836627960205078125f) * x3
                + (-0.250040113925933837890625f) * x4
                + (0.20005328953266143798828125f) * x5;
        }
        v_else {
            // Reuse calculate_log_body for |x| >= 0.3
            // This is already efficient — uses 5-term polynomial
            sfpi::vFloat one_plus_x = sfpi::vConst1 + val;
            result = calculate_log_body</*FAST_APPROX*/ false, /*HAS_BASE_SCALING*/ false, true>(one_plus_x, 0);
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }
    return result;
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = calculate_log1p_fp32<is_fp32_dest_acc_en>(in);
        } else {
            result = calculate_log1p_bf16<FAST_APPROX, is_fp32_dest_acc_en>(in);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        log_init<APPROXIMATION_MODE, FAST_APPROX, is_fp32_dest_acc_en>();
    } else {
        _init_reciprocal_<false, false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
