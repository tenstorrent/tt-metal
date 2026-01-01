// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// Blackhole only: ckernel_sfpu_moe_gate_sqrt_softplus.h is placed under the Blackhole ckernel tree.
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_moe_gate_sqrt_softplus.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the MoE gate's (DeepSeek-V4) router activation fused with the expert-bias add over the
 * tiles at idst_x (logits) and idst_bias (bias), in place:
 *
 *   idst_x    = sqrt(softplus(x))          (beta = 1, threshold = 20, as torch.nn.functional.softplus)
 *   idst_bias = sqrt(softplus(x)) + bias
 *
 * softplus is fp32-accurate (~5e-6 relative error over all inputs, see the LLK header), unlike the
 * stock softplus_tile, whose bf16 polynomial has ~1.6e-3 relative error and drops the tail for |x| > 5.
 * Requires fp32 DEST (fp32_dest_acc_en): there is no bf16 rounding step.
 *
 * Both operands stay in DST. The DST register buffer must be in acquired state via *acquire_dst*
 * call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                              | Type     | Valid Range                                           | Required |
 * |----------------|--------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_x         | The index of the tile in DST holding the logits; receives the score      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_bias      | The index of the tile in DST holding the bias; receives score + bias     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void moe_gate_sqrt_softplus_add_bias_tile(uint32_t idst_x, uint32_t idst_bias) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_moe_gate_sqrt_softplus_add_bias,
        (8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst_x,
        idst_bias,
        idst_bias,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void moe_gate_sqrt_softplus_add_bias_tile_init() {
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::moe_gate_sqrt_softplus_init)));
}

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
