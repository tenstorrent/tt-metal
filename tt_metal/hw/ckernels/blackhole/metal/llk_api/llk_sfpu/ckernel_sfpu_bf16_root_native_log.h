// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_bf16_root_native_log_core.h"
}  // namespace sfpi
namespace ckernel::sfpu::bf16 {
// The board's BF16 reciprocal with its constants as immediates, which leaves the
// programmable constant registers to the shared log constants.
sfpi_inline sfpi::vFloat root_native_log_reciprocal(sfpi::vFloat in) {
    // One Newton step from the hardware approximation.
    sfpi::vFloat y = sfpi::approx_recip(in);
    sfpi::vFloat residual = in * y - 2.0f;
    return y * -residual - 0.0f;
}

template <class Config>
inline void init_root_native_log() {
    sfpi::root_native_log_constants<Config>();
}

template <class Config, int Iterations = 32>
inline void calculate_root_native_log() {
    static_assert(Iterations == 32 && Config::kBf16);
    sfpi::root_native_log_tile<Config>(
        [](sfpi::vFloat raw, sfpi::vFloat& result) {
            sfpi::vFloat effective = raw;
            effective = sfpi::raw_daz_action_coordinate(raw);
            sfpi::apply_raw_domain_records<Config, Config::kDomainActionCount - 1>(effective, result);
            sfpi::negative_infinity_terminal<1>(effective, result);
            sfpi::zero_class_terminal<1>(effective, result);
        },
        [](sfpi::vUInt raw, sfpi::vFloat& result) {
            sfpi::negative_nan_class_terminal<Config::kRawNegativeNanClass, false>(raw, result);
        },
        [](sfpi::vFloat x) { return root_native_log_reciprocal(x); });
}
}  // namespace ckernel::sfpu::bf16
