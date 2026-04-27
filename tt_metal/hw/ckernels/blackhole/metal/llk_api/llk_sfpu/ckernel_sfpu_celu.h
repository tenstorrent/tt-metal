// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_expm1_cw.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// celu(x) = x for x>=0, alpha*(exp(x/alpha)-1) for x<0

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * expm1_cw_clamped(alpha_recip * x);

        v_if(x >= 0.0f) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void celu_init() {}

}  // namespace ckernel::sfpu
