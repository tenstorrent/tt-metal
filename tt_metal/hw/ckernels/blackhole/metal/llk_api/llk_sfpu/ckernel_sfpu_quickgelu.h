// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_quickgelu_appx.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_quickgelu() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // quickgelu(x) = x * sigmoid(1.702 * x)
        sfpi::vFloat scaled_x = x * 1.702f;
        sfpi::vFloat sigmoid_result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(scaled_x);
        sfpi::vFloat result = x * sigmoid_result;

        // Convert to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void quickgelu_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();
    } else {
        quickgelu_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
