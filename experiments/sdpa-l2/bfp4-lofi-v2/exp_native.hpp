// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Include after the selected frozen compute_common.hpp and before the private
// streaming header. Do not include a second physical copy of its SFPU header.
// This helper is opt-in and changes no existing entrypoint through macros.
#if defined(SDPA_LOFI_NATIVE_EXP)
#if !defined(SDPA_FP32_STREAMING) || defined(SDPA_STREAMING_ACCURACY) || defined(RESIDENT_MAIN)
#error "SDPA_LOFI_NATIVE_EXP requires the private cheap-exp FP32 streaming path"
#endif
#if defined(SDPA_LOFI_EXP_DEGREE) || defined(SDPA_DIAG_EXP_MODE) || defined(SDPA_FP32_FUSED_EXP) || \
    defined(SDPA_FP32_L1_MACRO) || defined(SDPA_FP32_REUSE_EXP) || defined(SDPA_FP32_REFINE_MACRO)
#error "SDPA_LOFI_NATIVE_EXP cannot be combined with a different exp implementation"
#endif
#if defined(DISABLE_SFPLOADMACRO)
#error "SDPA_LOFI_NATIVE_EXP requires the native approximate-exp LOADMACRO replay"
#endif

namespace ckernel {

// Required once at the start of EACH QK/exp phase (already done by the caller):
//   exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
// It records replay slots 0..31 and loads native constants L12/L13/L14:
//   A = 256*log2(e)*scale, B = 32500.818359375, shift = 15.
// Do not call init_sdpa_exp_grid: that would select the custom 10-bit grid
// with a 2^-96 exponent offset instead. No other PACK SFPU code may overwrite
// those constants, macro configuration, or replay slots 0..7 before a call.
//
// Despite its name, calculate_sdpa_exp_grid_batch is the same native replay
// engine. With native constants it directly writes ordinary-scale exp to
// FP32 DST, with no polynomial, exponent-restoration MUL, or second DST pass.
// Keep packer ReLU enabled: unclamped very negative inputs produce negatives.
// Keep accurate online rescale exponentials and represented-P sums unchanged.
template <int iterations>
ALWI void exp_native_packthread_tile(uint32_t idst) {
    static_assert(iterations == 32 || iterations == 128, "native SDPA exp supports one or four full tiles");
    static_assert(DST_ACCUM_MODE, "native SDPA exp wrapper requires FP32 DST");
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (iterations), idst, VectorMode::None)));
}

}  // namespace ckernel
#endif  // SDPA_LOFI_NATIVE_EXP
