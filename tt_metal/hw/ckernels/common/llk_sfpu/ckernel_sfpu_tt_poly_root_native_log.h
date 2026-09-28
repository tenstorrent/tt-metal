// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_min_max.h"
#include "ckernel_sfpu_tt_poly_mirrored_terminals.h"
#include "ckernel_sfpu_tt_poly_root_native_log_core.h"
}  // namespace sfpi
#include "ckernel_sfpu_tt_poly_finite_reciprocal.h"
namespace ckernel::sfpu::ttpoly {
template <class Config, int Iterations = 32>
inline void calculate_root_native_log() {
    static_assert(Iterations == 32 && Config::kBf16);
#if defined(ARCH_BLACKHOLE)
    static_assert(Config::kRawShadow && !Config::kStore && !Config::kSourceTerminal);
    static_assert(Config::kRawShadowBase == 64 && 96 * 2 <= DEST_REGISTER_HALF_SIZE);
#elif defined(ARCH_WORMHOLE)
    static_assert(!Config::kRawShadow && Config::kStore && Config::kSourceTerminal);
    static_assert(Config::kCoreBase + Config::kDegree + 1 == 85 && 85 * 2 <= DEST_REGISTER_HALF_SIZE);
#else
#error "selected root/native log requires BH or WH"
#endif
    sfpu_reciprocal_init<false>();
    sfpi::root_native_log_tile<Config>(
        [](sfpi::vFloat x) { return x; },
        [](sfpi::vFloat raw, sfpi::vFloat& result) {
            sfpi::vFloat effective = raw;
            if constexpr (Config::kRawShadow) {
                effective = sfpi::raw_daz_action_coordinate(raw);
            }
            sfpi::apply_raw_domain_records<Config, Config::kDomainActionCount - 1>(effective, result);
            if constexpr (Config::kRawShadow) {
                sfpi::negative_infinity_terminal<1>(effective, result);
                sfpi::zero_class_terminal<1>(effective, result);
            }
        },
        [](sfpi::vUInt raw, sfpi::vFloat& result) {
            sfpi::negative_nan_class_terminal<Config::kRawNegativeNanClass, Config::kSourceTerminal>(raw, result);
            if constexpr (Config::kSourceTerminal) {
                sfpi::encoded_word_terminal<0x80ffu, 2>(raw, result);
                sfpi::encoded_subnormal_terminal<1, -1>(raw, result);
            }
        },
        [](sfpi::vFloat x) {
            return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat value) {
#if defined(ARCH_WORMHOLE)
                return sfpu_reciprocal_iter<1>(value);
#else
                return value;
#endif
            });
        });
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_ROOT_NATIVE_LOG_SELECTED_V1 1
