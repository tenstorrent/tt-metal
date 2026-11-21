// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_assert.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_int_sum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void sfpu_sum_int_init() { MATH(SFPU_UNARY_INIT_FN(unused, sfpu::sum_int_init, (APPROX))); }

ALWI void sfpu_sum_int_col(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sum_int_col, (APPROX), idst, VectorMode::R));
}

ALWI void sfpu_sum_int_row(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sum_int_row, (APPROX), idst, VectorMode::C));
}

// clang-format off
/**
 * Adds the Int32 tile at DST index idst + dst_offset onto the tile at DST index idst, in place; the
 * operand tile is left unchanged. Must be preceded by a call to *sfpu_sum_int_init*. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * NOTE: On Quasar the operand tile is always idst + 1, so dst_offset must be 1 there.
 *
 * Return value: None
 *
 * | Argument   | Description                                                        | Type     | Valid Range                                                                | Required |
 * |------------|--------------------------------------------------------------------|----------|----------------------------------------------------------------------------|----------|
 * | idst       | The index of the tile in DST register buffer that receives the sum | uint32_t | Must be less than the size of the DST register buffer                      | True     |
 * | dst_offset | Distance in tiles from idst to the operand tile                    | uint32_t | idst + dst_offset must be less than the size of the DST register buffer; 1 on Quasar | True     |
 */
// clang-format on
ALWI void sfpu_add_int(uint32_t idst, uint32_t dst_offset) {
#ifdef ARCH_QUASAR
    LLK_ASSERT(dst_offset == 1, "sfpu_add_int on Quasar always adds tile idst + 1; dst_offset must be 1");
    MATH(LLK_ASSERT(
        (static_cast<std::uint64_t>(idst) + 1 <
         trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, trisc::DstTileShape::Tile32x32>()),
        "sfpu_add_int operand tile idst + 1 exceeds max dest tiles"));
#else
    MATH(LLK_ASSERT(
        (static_cast<std::uint64_t>(idst) + dst_offset <
         get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "sfpu_add_int operand tile idst + dst_offset exceeds max dest tiles"));
#endif
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, add_int, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, dst_offset));
}

}  // namespace ckernel
