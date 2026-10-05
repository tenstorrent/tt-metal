// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
}
#include "ckernel_sfpu_bf16_hyperbolic_exp_core.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_hyperbolic_exp() {
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[4]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[3]);
    sfpu_reciprocal_init<false>();
}
template <typename Config, int Iterations = 32>
inline void calculate_hyperbolic_exp() {
    static_assert(Iterations == 32 && Config::kDegree == 4);
    static_assert(Config::kOdd ? Config::kOriginCubicBits == 0x3e2a0000u : Config::kBare);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    if constexpr (Config::kOdd) {
        sfpi::hyperbolic_odd_load_macro_init();
    }
    sfpi::hyperbolic_exp_pins<Config>();
    constexpr unsigned body_slots = Config::kOdd ? 29u : 32u;
    ::ckernel::sfpu::bf16_sfpi::replay(0, body_slots, 1, 1);
    sfpi::hyperbolic_exp_core<ADDR_MOD_7>();
    if constexpr (Config::kOdd) {
        sfpi::hyperbolic_odd_suffix<ADDR_MOD_7, ADDR_MOD_6>();
    } else {
        sfpi::hyperbolic_even_store<ADDR_MOD_6>();
    }
#pragma GCC unroll 32
    for (int row = 1; row < Iterations; ++row) {
        ::ckernel::sfpu::bf16_sfpi::replay(0, body_slots, 0, 0);
        if constexpr (Config::kOdd) {
            sfpi::hyperbolic_odd_suffix<ADDR_MOD_7, ADDR_MOD_6>();
        }
    }
    // The standard whole-tile LLK done restores D-RWC. Restore the borrowed
    // architectural constant here before returning to that owner.
    sfpi::hyperbolic_restore_constant();
}
}  // namespace ckernel::sfpu::bf16
