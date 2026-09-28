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
}
#include "ckernel_sfpu_tt_poly_finite_reciprocal.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_exponent_bucket_core.h"
}
namespace ckernel::sfpu::ttpoly {
template <class Config>
inline void init_exponent_bucket() {
    sfpu_reciprocal_init<false>();
#if defined(ARCH_BLACKHOLE)
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#elif !defined(ARCH_WORMHOLE)
#error "exponent bucket requires BH or WH"
#endif
}
template <class Config, int Iterations = 32>
inline void calculate_exponent_bucket() {
    static_assert(Iterations == 32, "selected exponent bucket requires a whole tile");
    using Core = sfpi::ExponentBucket<Config>;
#if defined(ARCH_BLACKHOLE)
    static_assert(!Config::kStore && !Config::kFuse);
#elif defined(ARCH_WORMHOLE)
    static_assert(Config::kStore && Config::kFuse);
    static_assert(75 * 2 <= DEST_REGISTER_HALF_SIZE);
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 7; ++i) {
        sfpi::dst_reg[64u + i].template mode<sfpi::DataLayout::F32>() =
            sfpi::vFloat(Config::TT_EXPONENT_BUCKET_COEFFS[i]);
    }
#pragma GCC unroll 8
    for (uint32_t i = 1; i <= 4; ++i) {
        sfpi::dst_reg[70u + i].template mode<sfpi::DataLayout::F32>() =
            sfpi::vFloat(Config::TT_EXPONENT_BUCKET_REFLECTION_COEFFS[2u * i]);
    }
#else
#error "exponent bucket requires BH or WH"
#endif
#if defined(ARCH_WORMHOLE)
#pragma GCC unroll 32
#endif
    for (int d = 0; d < 32; d++) {
        sfpi::vFloat x_raw = sfpi::dst_reg[d];
        sfpi::vFloat y = Core::exponent_bucket_log_derivative_1_eval(x_raw, d + 32, [](sfpi::vFloat x) {
            return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat value) {
#if defined(ARCH_WORMHOLE)
                return sfpu_reciprocal_iter<1>(value);
#else
                return value;
#endif
            });
        });
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#if defined(ARCH_WORMHOLE)
        sfpi::vFloat terminal_input = sfpi::dst_reg[d];
        Core::apply_target_selected_sfpi_class_terminal(terminal_input, y);
        sfpi::dst_reg[d] = y;
#else
        sfpi::dst_reg[d + 32] = y;
#endif
    }
#if defined(ARCH_BLACKHOLE)
    Core::template exponent_bucket_target_class_repair_tti_tile<ADDR_MOD_7, ADDR_MOD_6>([] {});
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_EXPONENT_BUCKET_SELECTED_V1 1
