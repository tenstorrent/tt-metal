// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

// This experimental operation is currently supported on Blackhole.
#if defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_gated_reduce.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Initialize the SFPU state for gated_reduce_tile. Call after other SFPU operations
 * that change the sigmoid constants. Both gate modes use the non-approximate sigmoid.
 *
 * Return value: None
 */
ALWI void gated_reduce_tile_init() { MATH((SFPU_UNARY_INIT_FN(silu, sfpu::sigmoid_init, (false)))); }

// clang-format off
/**
 * Fuse activation and multiplication of two already-reduced tiles in adjacent DEST
 * slots. Apply the optional input scale independently to gate and up, then compute
 * gate * sigmoid(gate) * up, optionally multiplied by out_scale. ClampedSilu first
 * clamps gate from above and uses sigmoid(alpha * gate); Clamp clips up to +/-limit.
 * The intermediate arithmetic stays in FP32, with one final nearest BF16 conversion
 * when DEST is 16-bit. The up slot is unchanged.
 *
 * Acquire DEST and call gated_reduce_tile_init() before this operation. Store gate at
 * idst and up at idst + 1, using Tile32x32 DEST slot spacing even for tiny tiles.
 * For 4/8/16-row, 32-column tiles use ITERATIONS=2/4/8 and VectorMode::R; for a full
 * 32x32 tile use ITERATIONS=8 and VectorMode::RC. Both slots must fit the acquired
 * DEST section. This helper does not reduce the input groups itself.
 *
 * Return value: None
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template | GATE | Gate activation | sfpu::GatedReduceGate | Silu, ClampedSilu | True |
 * | Template | UP | Up activation | sfpu::GatedReduceUp | Identity, Clamp | True |
 * | Template | GATE_SCALE | Scale gate before activation | bool | true, false | True |
 * | Template | UP_SCALE | Scale up before activation | bool | true, false | True |
 * | Template | OUT_SCALE | Scale the product | bool | true, false | True |
 * | Template | ITERATIONS | SFPU vectors per face | int | 2, 4, 8 | False |
 * | Function | idst | Gate DEST tile index; result overwrites this slot | uint32_t | idst + 1 within acquired DEST | True |
 * | Function | scale_bits | FP32 bits of the input scale; ignored for disabled arms | uint32_t | | True |
 * | Function | out_scale_bits | FP32 bits of the output scale; ignored when disabled | uint32_t | | True |
 * | Function | limit_bits | FP32 bits of the clamp limit; ignored without clamping | uint32_t | Nonnegative finite float | True |
 * | Function | alpha_bits | FP32 bits of sigmoid's multiplier; ClampedSilu only | uint32_t | | True |
 * | Function | vector_mode | Faces to visit | VectorMode | R, RC | False |
 */
// clang-format on
template <
    sfpu::GatedReduceGate GATE,
    sfpu::GatedReduceUp UP,
    bool GATE_SCALE,
    bool UP_SCALE,
    bool OUT_SCALE,
    int ITERATIONS = 8>
ALWI void gated_reduce_tile(
    uint32_t idst,
    uint32_t scale_bits,
    uint32_t out_scale_bits,
    uint32_t limit_bits,
    uint32_t alpha_bits,
    VectorMode vector_mode = VectorMode::RC) {
    static_assert(ITERATIONS == 2 || ITERATIONS == 4 || ITERATIONS == 8);
    // The standard unary macro checks the gate index; also check the adjacent up slot.
    MATH(LLK_ASSERT(
        (idst < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>() - 1),
        "gated_reduce requires adjacent gate/up tiles within the acquired DEST section"));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_gated_reduce,
        (GATE, UP, GATE_SCALE, UP_SCALE, OUT_SCALE, DST_ACCUM_MODE, ITERATIONS),
        idst,
        vector_mode,
        scale_bits,
        out_scale_bits,
        limit_bits,
        alpha_bits)));
}

}  // namespace ckernel
#endif  // ARCH_BLACKHOLE
