// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"
#include "vconst_verifier.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS, typename vConstVerifier = vconst_verifier::disable>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en, vConstVerifier>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, typename vConstVerifier = vconst_verifier::disable>
inline auto silu_init() {
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    return sigmoid_init<false, vConstVerifier>();
}

}  // namespace ckernel::sfpu
