// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_mish_bf16_exponent_alu_product_core.h"
#include "sfpi.h"
#include <cstdint>
#include <limits>
namespace ckernel::sfpu::bf16::mish {
template <class Config>
inline void init_exponent_alu_product() {
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_exponent_alu_product() {
    static_assert(Iterations == 32, "selected product requires whole-tile ownership");
    ttpoly::mish::product_bounded_wh_tile<Config>();
}
}  // namespace ckernel::sfpu::bf16::mish
