// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_factored_cw_expm1_core.h"
namespace ckernel::sfpu::ttpoly {
template <class Config, int Iterations = 32>
inline void calculate_factored_cw_expm1() {
    static_assert(Iterations == 32);
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kInverseScaleBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kResidualScaleBits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kHalfBits);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#if defined(ARCH_BLACKHOLE)
    sfpi::factored_cw_bh<Config, ADDR_MOD_7, ADDR_MOD_6>([]() {});
#elif defined(ARCH_WORMHOLE)
    sfpi::factored_cw_wh<Config, ADDR_MOD_3, ADDR_MOD_2>([]() {});
#else
#error "selected factored residual requires BH or WH"
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_FACTORED_CW_EXPM1_SELECTED_V1 1
