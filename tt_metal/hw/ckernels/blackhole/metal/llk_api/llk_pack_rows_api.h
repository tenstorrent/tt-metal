// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack_rows.h"

/*************************************************************************
 * LLK PACK ROWS
 *
 * This is a non-standard packing operation that requires explicit initialization
 * and uninitialization, similar to pack_untilize.
 *************************************************************************/

/**
 * @brief Initialize the pack rows operation.
 *
 * @param num_rows Total number of rows to pack from the destination register to L1.
 *                 Each row contains 16 datums. Valid range: 1 to 64.
 *
 * This function prepares the packer hardware to pack a specified number of rows
 * from the destination register to L1 memory in row-major format.
 */
inline void llk_pack_rows_init(const std::uint32_t num_rows) { _llk_pack_rows_init_(num_rows); }

/**
 * @brief Pack rows from a destination register to L1 memory.
 *
 * @param dst_index Index in the destination register to read from
 * @param output The output circular buffer identifier
 * @param output_index The index in the output CB to write to
 *
 * This function packs the specified number of rows (configured via llk_pack_rows_init)
 * from the destination register to the output circular buffer.
 */
inline void llk_pack_rows(
    const std::uint32_t dst_index, const std::uint32_t output, const std::uint32_t output_index = 0) {
    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t pack_addr = get_output_tile_address<true, PackMode::Default>(output_id, output_index);
    LLK_ASSERT(
        (dst_index < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_rows_(dst_index, pack_addr);
}

/**
 * @brief Uninitialize the pack rows operation.
 *
 * Restores packer addrmods and counters to a safe default state.
 * Should be called after the pack rows operation is complete.
 */
inline void llk_pack_rows_uninit() { _llk_pack_rows_uninit_(); }
