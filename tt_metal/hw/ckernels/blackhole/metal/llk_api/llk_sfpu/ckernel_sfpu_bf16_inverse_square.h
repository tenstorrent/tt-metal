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
#include "sfpu/ckernel_sfpu_bf16_inverse_square_core.h"
}
namespace ckernel::sfpu::bf16 {
template <class Config>
inline void init_inverse_square() {
    sfpu_reciprocal_init<false>();
    sfpi::vConstFloatPrgm1 = Config::kP2[1];
    sfpi::vConstFloatPrgm2 = Config::kP2[0];
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

template <class Config, int Iterations = 32>
inline void calculate_inverse_square() {
    static_assert(Iterations == 32);
    static_assert(!Config::kDirectTail && Config::kFiniteReferencePrecedence);
    static_assert(!Config::kWhRepair);
    static_assert(!Config::kRepair || Config::kPositiveZeroFirstRaw == 0x7e80);
    sfpi::inverse_square_replay<Config, ADDR_MOD_7, ADDR_MOD_6>();
}
}  // namespace ckernel::sfpu::bf16
