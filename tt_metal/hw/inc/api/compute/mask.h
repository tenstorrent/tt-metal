/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void mask_tile_init() {
    MATH(SFPU_UNARY_INIT(mask));  // TODO(AP): move out init
}

// clang-format off
/**
 * Performs element-wise computation of mask on each element of a tile
 * in data and mask DST register. *mask_tile* will mask each element with 0,
 * *mask_posinf_tile* will mask each element with *float(inf)*.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * On Wormhole and Blackhole the mask tile may be any tile of the acquired DST register, before or after the data
 * tile; on Quasar it must be the tile after the data tile.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type       | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | dst_data_index | The index of the tile in DST REG for the data and result                   | uint32_t   | Must be less than the acquired size of DST REG        | True     |
 * | dst_mask_index | The index of the tile in DST REG for the mask                              | uint32_t   | Must be less than the acquired size of DST REG        | True     |
 * | data_format    | The format of the data and mask (supports Float16, Float16_b, and Int32)   | DataFormat | Must be a valid data format                           | False    |
 */
// clang-format on
ALWI void mask_tile(uint32_t idst_data, uint32_t idst2_mask, DataFormat data_format = DataFormat::Float16_b) {
#ifdef ARCH_QUASAR
    // The Quasar bodies read the mask from the tile after the data.
    if (data_format == DataFormat::Float16_b || data_format == DataFormat::Float16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mask, (true /* APPROXIMATE */), idst_data, VectorMode::RC));
    } else if (data_format == DataFormat::Int32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_int_mask, (true /* APPROXIMATE */), idst_data, VectorMode::RC));
    }
#else
    if (data_format == DataFormat::Float16_b || data_format == DataFormat::Float16) {
        MATH(SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_mask,
            (true /* APPROXIMATE */),
            idst_data,
            idst2_mask,
            idst_data,
            VectorMode::RC));
    } else if (data_format == DataFormat::Int32) {
#ifdef ARCH_BLACKHOLE
        // One call per tile: VectorMode::None runs the body once, and 32 iterations cover the four faces.
        MATH(SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_int_mask,
            (true /* APPROXIMATE */, 32 /* ITERATIONS */),
            idst_data,
            idst2_mask,
            idst_data,
            VectorMode::None));
#else
        MATH(SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_int_mask,
            (true /* APPROXIMATE */),
            idst_data,
            idst2_mask,
            idst_data,
            VectorMode::RC));
#endif
    }
#endif
}

ALWI void mask_posinf_tile(uint32_t idst_data, uint32_t idst2_mask) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mask_posinf, (true /* APPROXIMATE */), idst_data, VectorMode::RC));
#else
    MATH(SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_mask_posinf,
        (true /* APPROXIMATE */),
        idst_data,
        idst2_mask,
        idst_data,
        VectorMode::RC));
#endif
}

}  // namespace ckernel
