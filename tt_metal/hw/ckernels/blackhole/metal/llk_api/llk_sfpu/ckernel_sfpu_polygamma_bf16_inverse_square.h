// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polygamma_bf16_inverse_square_core.h"
namespace ckernel::sfpu::bf16::polygamma {
template <class Config>
inline void init_inverse_square() {
    // The BH rows load the regularizer's b1/b0 themselves: Prgm1 and Prgm2 belong to stock's init.
    sfpu_reciprocal_init<false>();
}

template <class Config, int Iterations = 32>
inline void calculate_inverse_square() {
    static_assert(Iterations == 32);
    static_assert(!Config::kDirectTail && Config::kFiniteReferencePrecedence);
    static_assert(!Config::kWhRepair);
    static_assert(!Config::kRepair || Config::kPositiveZeroFirstRaw == 0x7e80);
    // Rows step DEST by INCRWC: ADDR_MOD_6 belongs to the FPU ops (matmul) between init and call.
    ttpoly::polygamma::inverse_square_replay<Config, ADDR_MOD_7, ADDR_MOD_7, true>();
}
}  // namespace ckernel::sfpu::bf16::polygamma
