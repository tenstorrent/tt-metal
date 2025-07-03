// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

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
 * | transpose   | Flag to perform transpose on SrcA                 | uint32_t | Any positive value will indicate tranpose is set   | False    |
 */
// clang-format on
ALWI void copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, false /*transpose within 16x16 face*/, cbid)));
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, cbid)));
}
/**
 * Perform a init for the copy tile operation. This calls the short init function and initializes packer dst offset
 * registers.
 */
ALWI void copy_tile_init(uint32_t cbid) { copy_tile_to_dst_init_short(cbid); }

// clang-format off
/**
 * Return value: None
 *
 * | Argument       | Description                                                       | Type     | Valid Range                                       | Required |
 * |----------------|-------------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | old_cbid       | The identifier of the previous input circular buffer (CB) to SrcA | uint32_t | 0 to 31                                           | True     |
 * | new_cbid       | The identifier of the new input circular buffer (CB) to SrcA      | uint32_t | 0 to 31                                           | True     |
 * | transpose      | Flag to perform transpose on SrcA                                 | uint32_t | Any positive value will indicate tranpose is set  | False    |
 */
 // clang-format on
ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0) {
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, new_cbid)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, new_cbid)));
    copy_tile_to_dst_init_short(new_cbid, transpose);
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
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}

ALWI void copy_block_matmul_partials(
    uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles) {
    UNPACK((llk_unpack_A_block<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, start_in_tile_index, ntiles, false)));
    MATH((llk_math_eltwise_unary_datacopy_block<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        start_dst_tile_index, ntiles, in_cb_id)));
}

}  // namespace ckernel
