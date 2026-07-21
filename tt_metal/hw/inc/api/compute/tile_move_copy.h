// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "sanitizer/api.h"
#include "llk_assert.h"

#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Perform the init short for copy tile. This does not reconfigure the unpacker data types.
 * Return value: None
 *
 * | Argument    | Description                                       | Type     | Valid Range                                        | Required |
 * |-------------|---------------------------------------------------|----------|----------------------------------------------------|----------|
 * | cbid        | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31                                            | False    |
 * | transpose   | Flag to perform transpose on SrcA                 | uint32_t | Any positive value will indicate transpose is set  | False    |
 * | transpose_within_16x16_face | Flag to perform transpose within 16x16 face | uint32_t | Any positive value will indicate transpose within 16x16 face is set        | False    |
 */
// clang-format on
ALWI void copy_tile_to_dst_init_short(
    uint32_t cbid,
    uint32_t transpose = 0,
    uint32_t transpose_within_16x16_face = false,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(cbid, call_line);
#else
    LLK_ASSERT(transpose_within_16x16_face == false, "Transpose within face not supported on Quasar");
    LLK_ASSERT(transpose == 0, "Transpose not supported on Quasar");
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, transpose_within_16x16_face, cbid)));
    // 4th template arg is arch-divergent (unpack_to_dest on Quasar, is_int_fpu_en on WH/BH); keep it
    // arch-specific so WH/BH don't wrongly enable the integer-FPU datacopy MOP.
#ifndef ARCH_QUASAR
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(cbid)));
#else
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        cbid)));
#endif
}
/**
 * Perform a init for the copy tile operation. This calls the short init function and initializes packer dst offset
 * registers.
 */
ALWI void copy_tile_init(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    copy_tile_to_dst_init_short(cbid, 0, false, call_line);
}

// clang-format off
/**
 * Return value: None
 *
 * | Argument       | Description                                                       | Type     | Valid Range                                       | Required |
 * |----------------|-------------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | old_cbid       | The identifier of the previous input circular buffer (CB) to SrcA | uint32_t | 0 to 31                                           | True     |
 * | new_cbid       | The identifier of the new input circular buffer (CB) to SrcA      | uint32_t | 0 to 31                                           | True     |
 * | transpose      | Flag to perform transpose on SrcA                                 | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0) {
    LLK_SAN_FUNCTION();
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_cbid, new_cbid)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, new_cbid)));
    copy_tile_to_dst_init_short(new_cbid, transpose);
#endif
}

// clang-format off
/**
 * Copies a single tile from the specified input CB and writes the result to
 * DST at a specified index. The function will employ unpacker to first unpack into SRC
 * registers and then perform move into DST registers, at a specified index.
 * For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
 * previously called to ensure that at least some number n>0 of tiles are available
 * in the input CB. The CB index 0 then references the first tile in the received section of the CB,
 * up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
 * acquire_dst call. This call is blocking and is only available on the compute
 * engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | True     |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | True     |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | True     |
 * */
// clang-format on
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
#endif
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}

// clang-format off
/**
 * Copies a contiguous block of ``ntiles`` tiles from the specified input CB into the
 * DST register buffer starting at a chosen DST index. The source tiles are read from
 * the input CB at positions ``[start_in_tile_index, start_in_tile_index + ntiles)``
 * relative to the CB's current ``fifo_rd_ptr`` — i.e. ``start_in_tile_index`` is a
 * read-only tile offset into the fronted region, NOT into the DST register buffer.
 * The function does NOT advance ``fifo_rd_ptr``; the caller is responsible for
 * ``cb_wait_front(n)`` covering at least ``start_in_tile_index + ntiles`` tiles
 * before the call, and for ``cb_pop_front`` separately if/when the region is no
 * longer needed.
 *
 * Two index parameters, two distinct meanings:
 *   start_in_tile_index   tile offset into the SOURCE CB's fronted region (the read
 *                         starts at fifo_rd_ptr + start_in_tile_index * page_size).
 *   start_dst_tile_index  tile index in the DST register buffer where the first
 *                         copied tile lands (DST is filled sequentially from there
 *                         through start_dst_tile_index + ntiles - 1).
 *
 * The DST register buffer must be in acquired state via ``acquire_dst``. The
 * unpacker / math pipeline executes the copy as a single block to amortize init
 * overhead across the ntiles tiles. This call is blocking and is only available on
 * the compute engine.
 *
 * Operates in tandem with ``cb_reserve_back`` / ``cb_push_back`` / ``cb_wait_front``
 * / ``cb_pop_front`` for the producer-consumer FIFO protocol — the source-offset
 * read here enables index-based access into a fronted region without a per-tile
 * pop, useful for K-block partials reload patterns where the same fronted region
 * holds multiple sub-block slots accessed by offset.
 *
 * NOTE: copy_block_matmul_partials doesn't need explicit initialization function prior
 * to its call. Other op-specific initialization functions (such as ``tilize_init``,
 * ``copy_tile_to_dst_init_short``, etc.) ensure proper initialization of the unpacker
 * / math pipeline.
 *
 * Return value: None
 *
 * | Param Type | Name                 | Description                                              | Type     | Valid Range                                          | Required |
 * |------------|----------------------|----------------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Function   | in_cb_id             | The identifier of the source circular buffer (CB)        | uint32_t | 0 to 31                                              | True     |
 * | Function   | start_in_tile_index  | Tile offset within in_cb_id's fronted region (read base) | uint32_t | start_in_tile_index + ntiles <= fronted-region size  | True     |
 * | Function   | start_dst_tile_index | First DST register index to write                        | uint32_t | start_dst_tile_index + ntiles <= DST size            | True     |
 * | Function   | ntiles               | Number of tiles to copy                                  | uint32_t | start_dst_tile_index + ntiles <= DST size            | True     |
 */
// clang-format on
ALWI void copy_block_matmul_partials(
    uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
#endif
    UNPACK((llk_unpack_A_block<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, start_in_tile_index, ntiles)));
    MATH((llk_math_eltwise_unary_datacopy_block<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        start_dst_tile_index, ntiles, in_cb_id)));
}

}  // namespace ckernel
