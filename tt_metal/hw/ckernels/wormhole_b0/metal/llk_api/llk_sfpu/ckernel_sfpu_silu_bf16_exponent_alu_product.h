// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_silu_bf16_exponent_alu_product_core.h"
#include "sfpi.h"
#include <cstdint>
#include <limits>
namespace ckernel::sfpu::bf16::silu {
template <class Config>
inline void init_exponent_alu_product() {
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_exponent_alu_product() {
    static_assert(Iterations == 32, "selected product requires whole-tile ownership");
    ttpoly::silu::product_symmetric_wh<Config, ADDR_MOD_3>();
}
}  // namespace ckernel::sfpu::bf16::silu
