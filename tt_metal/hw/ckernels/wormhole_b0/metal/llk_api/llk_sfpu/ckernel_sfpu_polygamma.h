// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"

namespace ckernel::sfpu {

/**
 * Fused SFPU kernel for polygamma function: ψ^(n)(x)
 *
 * Computes: ψ^(n)(x) = (-1)^(n+1) * n! * Σ_{k=0}^{NUM_TERMS-1} 1/(x+k)^(n+1)
 *
 * Parameters are passed as bit-cast uint32_t values:
 *   n_packed:     order n (as float bits)
 *   scale_packed: precomputed (-1)^(n+1) * n! (as float bits)
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_polygamma(uint32_t n_packed, uint32_t scale_packed) {
    constexpr int NUM_TERMS = 11;  // Sum 11 terms (k=0..10), matching composite implementation

    // Unpack parameters (C++17 compatible — no std::bit_cast on SFPI compiler)
    float n_float;
    __builtin_memcpy(&n_float, &n_packed, sizeof(float));
    int n = static_cast<int>(n_float);
    int n_plus_1 = n + 1;
    float scale;
    __builtin_memcpy(&scale, &scale_packed, sizeof(float));

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat sum = sfpi::vFloat(0.0f);

        // Accumulate: Σ_{k=0}^{NUM_TERMS-1} 1/(x+k)^(n+1)
        for (int k = 0; k < NUM_TERMS; k++) {
            sfpi::vFloat xi = x + static_cast<float>(k);

            // Compute xi^(n+1) by repeated multiplication
            sfpi::vFloat power = xi;
            for (int j = 1; j < n_plus_1; j++) {
                power = power * xi;
            }

            // 1 / xi^(n+1)
            sum = sum + _sfpu_reciprocal_<2>(power);
        }

        // Apply scale: (-1)^(n+1) * n!
        sfpi::vFloat result = sum * scale;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void polygamma_init() {
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

}  // namespace ckernel::sfpu
