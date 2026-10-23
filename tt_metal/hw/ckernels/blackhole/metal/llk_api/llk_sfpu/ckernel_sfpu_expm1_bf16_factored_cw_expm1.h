// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_expm1_bf16_factored_cw_expm1_core.h"
namespace ckernel::sfpu::bf16::expm1 {
template <class Config>
inline void init_factored_cw_expm1() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kInverseScaleBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kResidualScaleBits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kHalfBits);
}

template <class Config, int Iterations = 32>
inline void calculate_factored_cw_expm1() {
    static_assert(Iterations == 32);
    ttpoly::expm1::factored_cw_bh<Config, ADDR_MOD_7, ADDR_MOD_7>([]() {});
}
}  // namespace ckernel::sfpu::bf16::expm1
