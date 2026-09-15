// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Include after FAST compute_common and safe_rescale. The fixed driver maps
// final reciprocal scratch to CB5 and recurrent exponential factors to CB14.
#if !defined(ARCH_BLACKHOLE) || DST_ACCUM_MODE || !defined(SDPA_STREAMING_ACCURACY) || !defined(SDPA_LOFI_SAFE_RESCALE)
#error "Final-scale HiFi4 requires this driver's Blackhole BF16 FAST layout"
#endif
#ifdef RESIDENT_MAIN
#error "MAIN final-scale mapping is not qualified by this private experiment"
#endif

namespace ckernel {
ALWI void sdpa_final_scale_mul_bcast_cols_init(uint32_t a, uint32_t b, uint32_t line = __builtin_LINE()) {
    // b is template-fixed to 5 in normalize_row_streaming, so this folds there.
    if (b == 5) {
        state_configure(a, b, line);
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MathFidelity::HiFi4>(a, b)));
        UNPACK((llk_unpack_AB_init<BroadcastType::COL>(a, b)));
    } else {
        lofi_safe_mul_bcast_cols_init(a, b, line);
    }
}

template <bool fp32 = DST_ACCUM_MODE>
ALWI void sdpa_final_scale_mul_tiles_bcast_cols(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi, uint32_t dst) {
    static_assert(!fp32);
    if (b == 5) {
        MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::COL, fp32,
                                    MathFidelity::HiFi4, EltwiseBinaryReuseDestType::NONE>(a, b, dst, true)));
        UNPACK((llk_unpack_AB<BroadcastType::COL>(a, b, ai, bi)));
    } else {
        lofi_safe_mul_tiles_bcast_cols<fp32>(a, b, ai, bi, dst);
    }
}
}  // namespace ckernel
