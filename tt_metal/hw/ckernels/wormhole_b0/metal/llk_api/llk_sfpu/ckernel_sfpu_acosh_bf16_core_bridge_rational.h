// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#if defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif
#include "ckernel_sfpu_acosh_bf16_min_max.h"
namespace ckernel::sfpu::bf16::acosh {
template <typename Config, int Iterations = 32>
inline void calculate_core_bridge_rational() {
    static_assert(Iterations == 32, "selected rational program owns a complete tile");
    Config::tile();
}
}  // namespace ckernel::sfpu::bf16::acosh
