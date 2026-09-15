// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Include AFTER frozen compute_common.hpp has parsed the original reciprocal
// API, BEFORE renaming just the final-normalization calls in streaming.hpp.
#if !defined(ARCH_BLACKHOLE) || DST_ACCUM_MODE
#error "Final reciprocal override requires Blackhole BF16 destination"
#endif

namespace ckernel {

template <bool legacy_compat = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sdpa_final_recip8_tile_init() {
    static_assert(!legacy_compat && !is_fp32_dest_acc_en);
#ifdef TRISC_MATH
    static_assert(APPROX, "Keep the original global approximate-math setting");
#endif
    // Both init and calculate explicitly select APPROXIMATION_MODE=false,
    // BF16 DST, non-legacy => Blackhole fast_8b_3c, NOT the FP32 24b routine.
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (false, false, false)));
}

template <bool legacy_compat = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sdpa_final_recip8_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(!legacy_compat && !is_fp32_dest_acc_en);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, false, calculate_reciprocal, (false, false, 8 /* ITERATIONS */, false), idst, vector_mode));
}

}  // namespace ckernel
