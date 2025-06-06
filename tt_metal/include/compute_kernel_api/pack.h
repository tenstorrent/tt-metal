// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

// clang-format off
/**
 * Copies a single tile from the DST register buffer at a specified index to a
 * specified CB at a given index. For the out_tile_index to be valid for this
 * call, cb_reserve_back(n) has to be called first to reserve at least some
 * number n > 0 of tiles in the output CB. out_tile_index = 0 then references
 * the first tile in the reserved section of the CB, up to index n - 1, which will
 * then be visible to the consumer in the same order after a cb_push_back call.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Each subsequent pack call will increment the write pointer in the cb by single
 * tile size. The pointer is then again set to a valid position with space for n
 * reserved tiles by another cb_reserve_back call.
 *
 * Operates in tandem with functions cb_reserve_back and cb_push_back.
 *
 * A typical use case is first the producer ensures that there is a number of
 * tiles available in the buffer via cb_reserve_back, then the producer uses
 * the pack_tile call to copy a tile from one of DST slots to a slot in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | ifrom_dst      | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 * | icb            | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | icb_tile       | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                | True     |
 */
// clang-format on
template <bool out_of_order_output = false>
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0) {
    PACK((llk_pack<DST_ACCUM_MODE, out_of_order_output, false>(ifrom_dst, icb, output_tile_index)));
}

// clang-format off
/**
 * Copies a block of tiles from the DST register buffer at a start index to a
 * specified CB at a given start index. cb_reserve_back(n) had to be called first
 * to reserve at least some number n>0 of tiles in the output CB. The out_tile_index = 0
 * then references the first tile in the reserved section of the CB, up to index n-1 that
 * will then be visible to the consumer in the same order after a cb_push_back call.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Each subsequent pack call will increment the write pointer in the cb by ntiles size.
 * The pointer is then again set to a valid position with space for n
 * reserved tiles by another cb_reserve_back call.
 *
 * Operates in tandem with functions cb_reserve_back and cb_push_back.
 *
 * A typical use case is first the producer ensures that there is a number of
 * tiles available in the buffer via cb_reserve_back, then the producer uses
 * the matmul_pack_tile call to copy a block of tiles from the DST slots to the slots in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | ifrom_dst      | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 * | icb            | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | ntiles         | The number of tiles to copy from DST to CB        | uint32_t | Must be less than the size of the CB                | True     |
 */
// clang-format on
ALWI void matmul_pack_tile(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles) {
    PACK((llk_matmul_pack<DST_ACCUM_MODE, false, false>(ifrom_dst, icb, ntiles)));
}

/**
 * Helper function to reconfigure packer output data format.
 */
ALWI void pack_reconfig_data_format(const uint32_t new_operand) {
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_operand)));
}

/**
 * Helper function to reconfigure packer output data format.
 */
ALWI void pack_reconfig_data_format(const uint32_t old_operand, const uint32_t new_operand) {
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_operand, new_operand)));
}

/**
 * Helper function to reconfigure packer l1 accumulation flag
 */
ALWI void pack_reconfig_l1_acc(const uint32_t l1_acc_en) { PACK((llk_pack_reconfig_l1_acc(l1_acc_en))); }

}  // namespace ckernel
