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
    // sfpu_reciprocal_iter<1>.
    sfpi::vFloat negative_x = sfpi::copyman(-1.0f, in);
    sfpi::vFloat y = 0.3232325017452239990234375f * negative_x + 1.4545459747314453125f;
    sfpi::vUInt scale_bits = ~sfpi::as<sfpi::vUInt>(in);
    y = y * negative_x + 2.121212482452392578125f;
    sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);
    sfpi::vFloat t = 1.0f + negative_x * y;
    scale *= 0.5f;
    y = y + y * t;
    y = y * scale;
    return sfpi::copysgn(y, in);
}

template <class Config, int Iterations = 32>
inline void calculate_root_native_log() {
    static_assert(Iterations == 32 && Config::kBf16);
    sfpi::root_native_log_tile<Config>(
        [](sfpi::vFloat raw, sfpi::vFloat& result) {
            sfpi::vFloat effective = raw;
            if constexpr (Config::kEffectiveTerminals) {
                effective = sfpi::raw_daz_action_coordinate(raw);
            }
            if constexpr (Config::kMirroredDomainActions) {
                // x >= bound gives +inf and x <= -bound gives -inf: one test of |x|.
                v_if(sfpi::setsgn(effective, 0) >= Config::kDomainActions[0].bound) {
                    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
                    result = sfpi::copysgn(infinity, effective);
                }
                v_endif;
            } else {
                sfpi::apply_raw_domain_records<Config, Config::kDomainActionCount - 1>(effective, result);
            }
            if constexpr (Config::kEffectiveTerminals) {
                sfpi::negative_infinity_terminal<1>(effective, result);
                sfpi::zero_class_terminal<1>(effective, result);
            }
        },
        [](sfpi::vUInt raw, sfpi::vFloat& result) {
            if constexpr (!Config::kWordTerminals) {
                sfpi::negative_nan_class_terminal<Config::kRawNegativeNanClass, false>(raw, result);
            } else if constexpr (Config::kRawNegativeNanClass == 2) {
                // DEST holds BF16 as sign, mantissa, exponent. Every negative word with an
                // all-ones exponent (-inf and each negative NaN) gives -inf, and a positive
                // subnormal gives +inf.
                sfpi::vUInt exponent_and_sign = raw & 0x80ffu;
                v_if(exponent_and_sign == 0x80ffu) { result = -std::numeric_limits<float>::infinity(); }
                v_elseif(exponent_and_sign == 0u && raw != 0u) { result = std::numeric_limits<float>::infinity(); }
                v_endif;
            } else {
                sfpi::negative_nan_class_terminal<Config::kRawNegativeNanClass, true>(raw, result);
                sfpi::encoded_word_terminal<0x80ffu, 2>(raw, result);
                sfpi::encoded_subnormal_terminal<1, -1>(raw, result);
            }
        },
        [](sfpi::vFloat x) { return root_native_log_reciprocal(x); });
}
}  // namespace ckernel::sfpu::bf16
