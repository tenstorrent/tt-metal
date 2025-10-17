// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_ema_sfpu_entry.h"
#endif

namespace ckernel {
ALWI void ema_init() { MATH((llk_math_ema_sfpu_init())); }

template <uint32_t input_dst_index>
ALWI void ema_tile(bool first_sample) {
    MATH((llk_math_ema_sfpu<input_dst_index>(first_sample)));
}

}  // namespace ckernel
