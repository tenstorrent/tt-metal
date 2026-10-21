// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_unary_datacopy_custom_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_A_custom_api.h"
#endif

namespace ckernel {

#ifdef ARCH_BLACKHOLE
// clang-format off
/**
 * Lightweight Blackhole-only copy of a single tile from the input CB into DST at dst_tile_index, for the custom SDPA
 * inner loop. Unlike copy_tile it runs no MOP: the unpacker issues one UNPACR for the tile and the math thread moves
 * it with eight MOVA2D over ADDR_MOD_4, so the matmul / reduce address modifiers and MOP programs survive it and the
 * following op only needs its lightweight reinit (mm_no_mop_reinit_short, reduce_block_max_row_reinit_*). The call
 * programs only the tile address, so the unpacker's SrcA format must already be in_cb_id's (compute_kernel_hw_startup
 * or a format reconfig on that CB); cb_wait_front and acquire_dst apply as for copy_tile.
 *
 * Numerics: like copy_tile, the copy keeps the Src zero-substitution flag disabled so bf16 -0.0 and denormals land
 * in DST unchanged (fp8 sources are flushed instead, whose widened zero carries a residual), and it LEAVES the flag
 * there. Before the next FPU op, run an init / reinit that re-applies the operand-driven default; not all
 * experimental FPU inits do so (see the datacopy lib note in llk_math_eltwise_unary_datacopy_custom.h for the
 * exceptions). Otherwise its MVMUL / GMPOOL keeps denormal Src operands.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | True     |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | True     |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | True     |
 */
// clang-format on
ALWI void copy_tile_custom(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A_custom(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy_custom(in_cb_id, dst_tile_index)));
}
#endif

}  // namespace ckernel
