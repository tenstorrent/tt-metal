// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_cosh_bf16_hyperbolic_exp_core.h"
#include "sfpu/ckernel_sfpu_cosh_bf16_hyperbolic_exp.h"
namespace ckernel::sfpu::bf16::cosh {
template <typename Config>
inline void init_hyperbolic_exp() {
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[4]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[3]);
    sfpu_reciprocal_init<false>();
}
}  // namespace ckernel::sfpu::bf16::cosh
