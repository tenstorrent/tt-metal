// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_bf16_exponent_alu_product_core.h"
#include "sfpi.h"
#include <cstdint>
#include <limits>
namespace ckernel::sfpu::bf16 {
template <class Config>
inline void init_exponent_alu_product() {
    sfpi::vConstFloatPrgm1 = Config::kCoefficients[2] * 0x1p-46f;
    sfpi::vConstFloatPrgm2 = Config::kCoefficients[1] * 0x1p-23f;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    if constexpr (Config::kSymmetric) {
        sfpi::product_symmetric_load_macro_init();
    }
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_exponent_alu_product() {
    static_assert(Iterations == 32, "selected product requires whole-tile ownership");
    if constexpr (Config::kSymmetric) {
        sfpi::product_symmetric_bh<Config, ADDR_MOD_7, ADDR_MOD_6>();
    } else {
        sfpi::product_bounded_bh<Config, ADDR_MOD_7, ADDR_MOD_6>();
    }
}
}  // namespace ckernel::sfpu::bf16
