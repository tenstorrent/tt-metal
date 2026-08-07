// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// softcap(x) = beta * tanh(x / beta), the up half of Moonshot's SiTU activation.
// beta arrives as an fp32 bit pattern with 1/beta precomputed by the caller, so
// the kernel never divides (same contract as celu's alpha / alpha_recip).
//
// Non-finite inputs: +/-Inf clamps to +/-beta. NaN does NOT propagate -- the
// min(., 1.0) inside the tanh polynomial picks 1.0 over NaN, so the result is
// finite (+beta). Stock calculate_tanh behaves the same.
inline void softcap_init() { tanh_init</*APPROXIMATION_MODE=*/false, /*is_fp32_dest_acc_en=*/false>(); }

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_softcap(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_beta = Converter::as_float(param1);

    for (int d = 0; d < ITERATIONS; d++) {
        // Always the Sollya polynomial, never _sfpu_tanh_fp32_accurate_, in both dst
        // modes: the accurate expm1 tanh together with runtime beta exhausts the SFPU
        // LReg file, which is a hard compile abort ("cannot store sfpu register")
        // rather than a slowdown. Callers needing fp32-grade tanh should use tanh_tile.
        //
        // Round once, after the beta rescale -- rounding tanh first would discard bits
        // that the multiply by beta then amplifies.
        sfpi::vFloat result = _sfpu_tanh_polynomial_(sfpi::dst_reg[0] * inv_beta) * beta;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
