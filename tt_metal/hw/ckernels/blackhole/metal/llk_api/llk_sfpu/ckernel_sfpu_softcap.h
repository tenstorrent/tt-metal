// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// softcap(x) = beta * tanh(x / beta). The caller precomputes 1/beta so the kernel never
// divides (same contract as celu's alpha / alpha_recip).
//
// Always the Sollya polynomial, never _sfpu_tanh_fp32_accurate_: the accurate tanh plus a
// runtime beta exhausts the SFPU LReg file, a hard compile abort rather than a slowdown.
// Use tanh_tile if fp32-grade tanh is needed.
//
// +/-Inf clamps to +/-beta; NaN does not propagate and -0.0 comes back as +0.0 (measured),
// both matching stock calculate_tanh / tanh_tile rather than torch. Leaves rounding to the
// caller, so fused callers can round once at the end.
sfpi_inline sfpi::vFloat _sfpu_softcap_(sfpi::vFloat x, sfpi::vFloat beta, sfpi::vFloat inv_beta) {
    return _sfpu_tanh_polynomial_(x * inv_beta) * beta;
}

inline void softcap_init() { tanh_init</*APPROXIMATION_MODE=*/false, /*is_fp32_dest_acc_en=*/false>(); }

// beta and its reciprocal arrive as fp32 bit patterns.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softcap(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_beta = Converter::as_float(param1);

    for (int d = 0; d < ITERATIONS; d++) {
        // Round after the beta rescale, not before: the multiply amplifies discarded bits.
        sfpi::vFloat result = _sfpu_softcap_(sfpi::dst_reg[0], beta, inv_beta);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
