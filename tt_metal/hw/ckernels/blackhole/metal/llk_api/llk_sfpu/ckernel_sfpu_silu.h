// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

<<<<<<< HEAD
template <bool is_fp32_dest_acc_en, int ITERATIONS>
=======
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
>>>>>>> b47cee158f (silu fix)
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

<<<<<<< HEAD
inline void silu_init() {
    _init_reciprocal_<false, false>();
=======
template <bool APPROXIMATION_MODE>
inline void silu_init() {
    sigmoid_init<false>();
>>>>>>> b47cee158f (silu fix)
}

}  // namespace ckernel::sfpu
