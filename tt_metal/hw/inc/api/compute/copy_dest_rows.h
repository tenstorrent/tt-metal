// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "sanitizer/api.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_copy_dest_rows.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Copies one 4-row band, all 32 columns, from the tile in idst_in to the tile in idst_out in the DST register buffer:
 * rows 4 * in_band .. 4 * in_band + 3 of the source land in rows 4 * out_band .. 4 * out_band + 3 of the target. Every
 * other row of the target keeps its bits, and the copy is bit-exact (NaN payloads, the sign of zero and denormals
 * included: raw loads and stores). The source and target may be the same tile. Bands 0..3 are the tile's upper half (faces 0 and 1), bands 4..7 the lower
 * half (faces 2 and 3), so a 16x32 tile has bands 0..3.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call, and *copy_dest_values_init* must
 * have run before. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument | Description                                                        | Type     | Valid Range                                           | Required |
 * |----------|--------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_out | The index of the tile in DST register buffer written to            | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | out_band | The 4-row band of that tile written: rows 4 * out_band .. + 3      | uint32_t | 0 to 7                                                | True     |
 * | idst_in  | The index of the tile in DST register buffer read from             | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | in_band  | The 4-row band of that tile read: rows 4 * in_band .. + 3          | uint32_t | 0 to 7                                                | True     |
 */
// clang-format on
ALWI void copy_dest_rows(uint32_t idst_out, uint32_t out_band, uint32_t idst_in, uint32_t in_band) {
    MATH(SAN_HOOK(unsupported()));
    MATH(LLK_ASSERT(out_band < 8 && in_band < 8, "copy_dest_rows: a band index is 0..7"));
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_band,
        (false /*APPROXIMATE*/, DST_ACCUM_MODE),
        idst_in,
        idst_out,
        idst_out,
        VectorMode::None,
        in_band,
        out_band)));
}

// clang-format off
/**
 * Copies one row, all 32 columns, from the tile in idst_in to the tile in idst_out in the DST register buffer: row
 * from_row of the source lands in row to_row of the target. Every other row of the target keeps its bits, and the copy
 * is bit-exact (NaN payloads, the sign of zero and denormals included: raw loads and stores). The source and target may
 * be the same tile. Rows 0..15
 * are the tile's upper half (faces 0 and 1), 16..31 the lower half, so a 16x32 tile has rows 0..15.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call, and *copy_dest_values_init* must
 * have run before. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument | Description                                                        | Type     | Valid Range                                           | Required |
 * |----------|--------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_out | The index of the tile in DST register buffer written to            | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | to_row   | The row of that tile written                                       | uint32_t | 0 to 31                                               | True     |
 * | idst_in  | The index of the tile in DST register buffer read from             | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | from_row | The row of that tile read                                          | uint32_t | 0 to 31                                               | True     |
 */
// clang-format on
ALWI void copy_dest_row(uint32_t idst_out, uint32_t to_row, uint32_t idst_in, uint32_t from_row) {
    MATH(SAN_HOOK(unsupported()));
    MATH(LLK_ASSERT(to_row < 32 && from_row < 32, "copy_dest_row: a row index is 0..31"));
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_row,
        (false /*APPROXIMATE*/, DST_ACCUM_MODE),
        idst_in,
        idst_out,
        idst_out,
        VectorMode::None,
        from_row,
        to_row)));
}

}  // namespace ckernel
