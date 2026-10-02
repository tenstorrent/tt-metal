// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// Blackhole only: ckernel_sfpu_clamped_silu_glu.h is placed under the blackhole ckernel tree. Its
// primitives all have Wormhole counterparts, so enabling Wormhole is a placement and validation
// task rather than a port.
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_clamped_silu_glu.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the element-wise clamped-SwiGLU activation over the tiles at idst0 (gate) and idst1
 * (up), writing the result to odst in DST:
 *
 *   gate_c = min(gate, limit)
 *   up_c   = clamp(up, -limit, limit)
 *   odst   = gate_c * sigmoid(gate_c) * up_c
 *
 * The gate half clamps only from above and the up half clamps both ends, so
 * |odst| <= limit * limit up to the rounding of the packed result. The limit is the compile-time
 * DeepSeek-V4 value of 10, shared by V4 Pro and V4 Flash. This entry point cannot be templated on
 * the Config: ckernel::sfpu::ClampedSiluGluConfigDsV4 is declared only under TRISC_MATH, so naming
 * it in a default template argument fails to compile on the unpack and pack threads. A model with
 * a different limit adds a config beside it and its own thin wrapper here.
 *
 * Both operands stay in DST, so no intermediate is materialized to L1 or DRAM. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                            | Type     | Valid Range                                           | Required |
 * |----------------|------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer holding the gate operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer holding the up operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output          | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode    | The vector mode of the operation                                       | int      | Must be one of the VectorMode values                  | False    |
 */
// clang-format on
ALWI void clamped_silu_glu_tile(
    uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_clamped_silu_glu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */, sfpu::ClampedSiluGluConfigDsV4),
        idst0,
        idst1,
        odst,
        vector_mode)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void clamped_silu_glu_tile_init() {
    // The SfpuType tag only selects whether eltwise_binary_sfpu_configure_addrmod programs
    // ADDR_MOD_6, which it does for the integer-multiply, min/max and compare ops; a pure-sfpi
    // binary op needs none of that, so `unused` is accurate rather than a placeholder. Matches
    // llk_math_eltwise_binary_sfpu_swiglu_init, which passes `unused` for the same reason.
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
}

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
