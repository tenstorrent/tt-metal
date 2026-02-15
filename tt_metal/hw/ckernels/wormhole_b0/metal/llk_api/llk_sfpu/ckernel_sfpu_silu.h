// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS, typename vConstVerifier = vconst_verifier::disable>
inline void calculate_silu() {
    static_assert(!std::is_same_v<vConstVerifier, vconst_verifier::disable>);
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
    if constexpr (!APPROXIMATION_MODE) {
        return _init_sfpu_reciprocal_<false, vConstVerifier>();
    } else {
        return _init_sfpu_reciprocal_<true, vConstVerifier>();
    }
}

}  // namespace ckernel::sfpu
