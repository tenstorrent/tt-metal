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
#include "sfpu/ckernel_sfpu_bf16_min_max.h"
}
#include "sfpu/ckernel_sfpu_bf16_finite_reciprocal.h"
#include "sfpu/ckernel_sfpu_bf16_abs_exp_correction_core.h"
#include "sfpu/ckernel_sfpu_bf16_abs_exp_correction.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_abs_exp_correction() {
    sfpi::vConstFloatPrgm0 = Config::kMultiplier;
    sfpi::vConstFloatPrgm1 =
        Config::kExpCoefficients[Config::kExpDegree] * sfpi::correction_exp_scale(Config::kExpDegree);
    sfpi::vConstFloatPrgm2 =
        Config::kExpCoefficients[Config::kExpDegree - 1] * sfpi::correction_exp_scale(Config::kExpDegree - 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
