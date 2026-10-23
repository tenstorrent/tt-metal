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
    // Both rows load the P2 head's coefficients in the call; the symmetric row's init sets its LOADMACRO sequence.
    ttpoly::silu::product_symmetric_load_macro_init();
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_exponent_alu_product() {
    static_assert(Iterations == 32, "selected product requires whole-tile ownership");
    // Rows step DEST by INCRWC: ADDR_MOD_6 belongs to the FPU ops (matmul) between init and call.
    ttpoly::silu::product_symmetric_bh<Config, ADDR_MOD_7, ADDR_MOD_7, true>();
}
}  // namespace ckernel::sfpu::bf16::silu
