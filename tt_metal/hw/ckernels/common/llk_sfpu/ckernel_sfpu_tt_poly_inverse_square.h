// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_inverse_square_core.h"
}
namespace ckernel::sfpu::ttpoly {
template <class Config, int Iterations = 32>
inline void calculate_inverse_square() {
    static_assert(Iterations == 32);
    static_assert(Config::kRepair && !Config::kDirectTail && Config::kFiniteReferencePrecedence);
    sfpu_reciprocal_init<false>();
#if defined(ARCH_BLACKHOLE)
    static_assert(!Config::kWhRepair);
    static_assert(Config::kPositiveZeroFirstRaw == 0x7e80);
    sfpi::vConstFloatPrgm1 = Config::kP2[1];
    sfpi::vConstFloatPrgm2 = Config::kP2[0];
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    sfpi::inverse_square_replay<Config, ADDR_MOD_7, ADDR_MOD_6>();
#else
    static_assert(Config::kWhRepair);
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[d];
        sfpi::vFloat result = sfpi::inverse_square_scalar<Config>(
            x,
            [](sfpi::vFloat value) { return sfpu_reciprocal_iter<1>(value); },
            [](sfpi::vFloat value) { return value; });
        sfpi::dst_reg[d] = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_INVERSE_SQUARE_SELECTED_V1 1
