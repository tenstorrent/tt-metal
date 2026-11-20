// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_copy_dest_values.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Copies all values from the tile in idst_in to the tile in idst_out in the DST register buffer.
 * This is a generalized version that takes a DataFormat template parameter.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type                     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|--------------------------|-------------------------------------------------------|----------|
 * | DATA_FORMAT    | The data format for copy operation                                    | DataFormat               | Any valid DataFormat enum value                       | True     |
 * | ITERATIONS     | Template: 4-row groups per visited face. Iteration k copies face rows 4*(k/2)..4*(k/2)+3 of one column parity (even columns for even k, odd for odd k), so 8 is the whole face and 2 under VectorMode::R is rows 0..3 of all 32 columns. The body advances the DEST counter per iteration and the vector-mode loop re-bases every face on the CR counter, so a partial count composes with any vector mode. Fixed at 8 on Quasar. | int | 1 to 8 | False (8) |
 * | idst_in        | The index of the tile in DST register buffer to copy values from      | uint32_t                 | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | The index of the tile in DST register buffer to copy values to        | uint32_t                 | Must be less than the size of the DST register buffer | True     |
 * | vector_mode    | The faces to visit: RC all four, R faces 0 and 1 (rows 0..15), C faces 0 and 2 (columns 0..15); the rest of idst_out keeps its contents | VectorMode | RC, R, C | False (RC) |
 */
// clang-format on
template <DataFormat DATA_FORMAT, int ITERATIONS = 8>
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    static_assert(ITERATIONS == 8, "copy_dest_values: ITERATIONS is fixed at 8 on Quasar");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (DATA_FORMAT, false /*APPROXIMATE*/),
        idst_in,
        idst_out,
        0 /*unused*/,
        vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (DATA_FORMAT, false /*APPROXIMATE*/, ITERATIONS),
        idst_in,
        idst_out,
        0 /*unused*/,
        vector_mode)));
#endif
}

// clang-format off
/**
 * Copies all values from the tile in idst_in to the tile in idst_out in the DST register buffer.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_in        | The index of the tile in DST register buffer to copy values from      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | The index of the tile in DST register buffer to copy values to        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
[[deprecated("Use copy_dest_values<DataFormat> instead")]]
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out) {
    // Routes through the deprecated 1-template-arg `copy_dest_value<APPROXIMATE>` overload in
    // ckernel::sfpu (the format-agnostic sfpi::vFloat path). New code should use the
    // DataFormat-templated overload above.
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (false /*APPROXIMATE*/),
        idst_in,
        idst_out,
        0 /*unused*/,
        VectorMode::RC)));
}

ALWI void copy_dest_values_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::copy_dest_value_init))); }

}  // namespace ckernel
