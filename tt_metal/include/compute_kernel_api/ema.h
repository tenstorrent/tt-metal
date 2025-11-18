// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_ema_sfpu_entry.h"
#endif

namespace ckernel {
ALWI void ema_init(uint32_t alpha, uint32_t beta) {
    MATH((llk_math_ema_sfpu_init()));
    MATH((llk_math_ema_sfpu_load_alpha_beta(alpha, beta)));
}

ALWI void ema_clear_previous_output() { MATH((llk_math_ema_sfpu_clear_previous_output())); }

ALWI void ema_tile(uint32_t input_dst_index) { MATH((llk_math_ema_sfpu_tile(input_dst_index))); }

}  // namespace ckernel
