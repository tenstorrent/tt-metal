// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

// clang-format off
/**
 * Copies a single tile from the DEST register buffer at a specified index to a
 * specified CB at a given index. For the out_tile_index to be valid for this
 * call, cb_reserve_back(n) has to be called first to reserve at least some
 * number n > 0 of tiles in the output CB. out_tile_index = 0 then references
 * the first tile in the reserved section of the CB, up to index n - 1, which will
 * then be visible to the consumer in the same order after a cb_push_back call.
 * The DEST register buffer must be in acquired state via *acquire_dst* call.
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
 * the pack_tile call to copy a tile from one of DEST slots to a slot in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 *
 * When the `out_of_order_output` flag is set to true, `pack_tile` behaves like
 * other APIs in that it writes the output tile to the tile index specified by
 * the user in `output_tile_index` within the reserved region of the circular
 * buffer, but relative to the position that might have been previously updated
 * by `cb_push_back` function. If `out_of_order_output` is false (the default),
 * `pack_tile` always operates sequentially: it writes to the next tile index
 * starting from index 0, and the `output_tile_index` parameter is ignored. In
 * this mode, each call to `pack_tile` advances the internal write pointer for
 * the reserved region, which is reset after `cb_push_back` function.
 *
 * NOTE: pack_tile doesn't need explicit initialization function prior to its call. Other op-specific
 * initialization functions (such as `tilize_init`, `reduce_init`, etc.) ensure proper initialization
 * of the packer. The reason for this stems from the fact that MATH and PACK threads need to be explicitly
 * synchronized in the kernels. To ensure this synchronization, tile packing is implemented as a separate
 * API call.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                                       | Type     | Valid Range                                          | Required |
 * |------------|------------------|---------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Template   | out_of_order_output | Whether to allow out-of-order output           | bool     | true/false                                           | False    |
 * | Function   | ifrom_dst        | The index of the tile in the DEST register        | uint32_t | Must be less than the size of the DEST register (16) | True     |
 * | Function   | icb              | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                              | True     |
 * | Function   | output_tile_index| The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                 | False    |
 */
// clang-format on
template <bool out_of_order_output = false>
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0) {
    PACK((llk_pack<DST_ACCUM_MODE, out_of_order_output, false>(ifrom_dst, icb, output_tile_index)));
}

// clang-format off
/**
 * Copies a block of tiles from the DEST register buffer starting at a specified index
 * to a specified circular buffer (CB). The DEST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine. Before calling this function, cb_reserve_back(n) must be called to
 * reserve at least n > 0 tiles in the output CB. Each call to `pack_tile_block` will
 * copy `ntiles` tiles from the DEST register to the reserved region of the CB, starting
 * from index 0. The internal write pointer in the CB is advanced by `ntiles` after each
 * call, and is reset by another cb_push_back call. Operates in tandem with functions
 * cb_reserve_back and cb_push_back.
 *
 * A typical use case is for the producer to ensure that there are enough tiles available
 * in the buffer via cb_reserve_back, then use pack_tile_block to copy a block of tiles
 * from the DEST slots to the reserved space in the CB, and finally call cb_push_back to
 * announce visibility of the reserved section of the circular buffer to the consumer.
 *
 * NOTE: pack_tile_block doesn't need explicit initialization function prior to its call. Other op-specific
 * initialization functions (such as `tilize_init`, `reduce_init`, etc.) ensure proper initialization
 * of the packer. The reason for this stems from the fact that MATH and PACK threads need to be explicitly
 * synchronized in the kernels. To ensure this synchronization, tile packing is implemented as a separate
 * API call.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                       | Type     | Valid Range                                          | Required |
 * |------------|-----------|---------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Function   | ifrom_dst | The index of the first tile in the DEST register  | uint32_t | Must be less than the size of the DEST register (16) | True     |
 * | Function   | icb       | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                              | True     |
 * | Function   | ntiles    | The number of tiles to copy from DEST to CB       | uint32_t | Must be less than the size of the DEST register (16) | True     |
 */
// clang-format on
ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles) {
    PACK((llk_matmul_pack<DST_ACCUM_MODE, false, false>(ifrom_dst, icb, ntiles)));
}

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB ID of the new operand. This function
 * call will always perform the reconfiguration, regardless of the data format of the old operand.
 * If the new CB ID is the same as the current one, reconfiguration will still occur.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name       | Description                        | Type     | Valid Range | Required |
 * |------------|------------|------------------------------------|----------|-------------|----------|
 * | Function   | new_cb_id  | New data format operand value      | uint32_t | Any         | True     |
 */
// clang-format on
ALWI void pack_reconfig_data_format(const uint32_t new_cb_id) {
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb_id)));
}

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB IDs of the old and new operands.
 * This function internally calls the reconfiguration function with the new CB ID, but before it does so,
 * it checks if the old and new data formats are different. If they are the same, it does not perform
 * the reconfiguration. This function is useful when you want to ensure that the packer only reconfigures
 * when different data format is wanted, avoiding unnecessary reconfiguration overhead.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name       | Description                        | Type     | Valid Range | Required |
 * |------------|------------|------------------------------------|----------|-------------|----------|
 * | Function   | old_cb_id  | Previous data format operand value | uint32_t | Any         | True     |
 * | Function   | new_cb_id  | New data format operand value      | uint32_t | Any         | True     |
 */
// clang-format on
ALWI void pack_reconfig_data_format(const uint32_t old_cb_id, const uint32_t new_cb_id) {
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_cb_id, new_cb_id)));
}

// clang-format off
/**
 * Helper function to reconfigure the packer L1 accumulation flag. This function would ideally be called
 * after other initialization functions that initialize the packer for a specific operation.
 * This function configures the packer to accumulate the values it takes from DEST with the ones that
 * are already in L1 at a given CB ID and tile index.
 *
 * The `l1_acc_en` parameter must be set to either 0 (disable accumulation) or 1 (enable accumulation).
 * Other values are not valid.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                        | Type     | Valid Range | Required |
 * |------------|-----------|------------------------------------|----------|-------------|----------|
 * | Function   | l1_acc_en | L1 accumulation enable flag        | uint32_t | 0 or 1      | True     |
 */
// clang-format on
ALWI void pack_reconfig_l1_acc(const uint32_t l1_acc_en) { PACK((llk_pack_reconfig_l1_acc(l1_acc_en))); }

}  // namespace ckernel
