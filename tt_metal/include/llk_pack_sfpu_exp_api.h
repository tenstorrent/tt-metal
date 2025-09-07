// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

#ifdef TRISC_PACK
#include "llk_math_eltwise_unary_sfpu_exp.h"
#endif

namespace ckernel {

#ifdef TRISC_PACK

// Initialize SFPU exponential on PACK thread.
// Matches math-side exp init but routed for PACK usage to overlap with MATH matmul.
template <bool approx = false, bool fast_and_approx = true, uint32_t scale = 0x3F800000>
ALWI void llk_pack_sfpu_exponential_init() {
    PACK((llk_math_eltwise_unary_sfpu_exponential_init<approx, fast_and_approx, scale>()));
}

// Issue SFPU exponential on a DST tile from the PACK thread.
// PACK can drive SFPU while MATH continues matmul work on a different DST buffer.
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    int iterations = 8,
    int dst_accum_mode = DST_ACCUM_MODE>
ALWI void llk_pack_sfpu_exponential(
    uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    PACK((llk_math_eltwise_unary_sfpu_exponential<
          approx,
          fast_and_approx,
          scale_en,
          skip_positive_check,
          iterations,
          dst_accum_mode>(idst, vector_mode, scale)));
}

#endif  // TRISC_PACK

}  // namespace ckernel
