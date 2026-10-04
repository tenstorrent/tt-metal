// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
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
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_core_bridge_rational() {
#if defined(TRISC_MATH)
    ckernel::llk_math_sfpu_init_once();
#endif
    ckernel::sfpu::sfpu_reciprocal_init<false>();
}
template <typename Config, int Iterations = 32>
inline void calculate_core_bridge_rational() {
    static_assert(Iterations == 32, "selected rational program owns a complete tile");
    Config::tile();
}
}  // namespace ckernel::sfpu::bf16
