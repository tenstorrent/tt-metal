// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif

namespace ckernel {

// clang-format off
/**
 * A blocking call that waits for the specified number of tiles to be available in the specified circular buffer (CB).
 * This call is used by the consumer of the CB to wait for the producer to fill the CB with at least the specfied number
 * of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired cb_pop_front() call,
 * n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example: 4 calls of
 * cb_wait_front(8) followed by a cb_pop_front(32) would produce incorrect behavior. Instead 4 calls of cb_wait_front()
 * waiting on 8, 16, 24, 32 tiles should be issued.
 *
 * Important note: number of tiles used in all cb_* calls must evenly divide the cb size and must be the same number in
 * all cb_wait_front calls in the same kernel. Example 1: cb_wait_front(32), cb_wait_front(40), cb_pop_front(32+8) tiles
 * on a CB of size 64 would produce incorrect behavior. Example 2: cb_wait_front(3) on a cb of size 32 would also
 * produce incorrect behavior. These limitations are due to performance optimizations in the CB implementation.
 *
 * Important note: CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | ntiles    | The number of tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 * */
// clang-format on
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) { UNPACK((llk_wait_tiles(cbid, ntiles))); }

// clang-format off
/**
 * Pops a specified number of tiles from the front of the specified CB. This
 * also frees this number of tiles in the circular buffer. This call is used by
 * the consumer to free up the space in the CB.
 *
 * We use the convention that the producer pushes tiles into the “back” of the
 * CB queue and the consumer consumes tiles from the “front” of the CB queue.
 *
 * Note that the act of reading of the tile data from the CB does not free up
 * the space in the CB. Waiting on available tiles and popping them is
 * separated in order to allow the consumer to: 1) read the tile data from the
 * CB via multiple reads of sub-tiles 2) access the tiles (or their sub-tiles)
 * that are visible to the consumer by random access of the valid section of
 * the CB
 *
 * Important note: This operation updates the read pointer of the CB, the CB pointer
 * can only be updated from one thread at a time. Example: if compute kernel has cb_pop_front(input_id, 1)
 * and writer kernel also has cb_pop_front(input_id, 1), these calls will produce non-deterministic behavior because
 * cb pointers are not synchronized across threads. Per circular buffer index, only have one thread pop tiles
 * to update the read pointer
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | ntiles    | The number of tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles) { UNPACK((llk_pop_tiles(cbid, ntiles))); }

// clang-format off
/**
 * A blocking call that waits for the specified number of tiles to be free in the specified circular buffer. This call
 * is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | ntiles    | The number of free tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles) {
    PACK((llk_wait_for_free_tiles<false, false, false>(cbid, ntiles)));
}

// clang-format off
/**
 * Pushes a given number of tiles in the back of the specified CB’s queue.
 * Decreases the available space in the circular buffer by this number of
 * tiles. This call is used by the producer to make the tiles visible to the
 * consumer of the CB.
 *
 * We use the convention that the producer pushes tiles into the “back” of the
 * CB queue and the consumer consumes tiles from the “front” of the CB queue.
 *
 * Note that the act of writing the tile data into the CB does not make the
 * tiles visible to the consumer. Writing of the tiles and pushing is separated
 * to allow the producer to: 1) write the tile data to the CB via multiple
 * writes of sub-tiles 2) modify tiles (or sub-tiles) by random access of the
 * valid section of the CB
 *
 * Important note: This operation updates the write pointer of the CB, the CB pointer
 * can only be updated from one thread at a time. Example: if compute kernel has cb_push_back(output_id, 1)
 * and reader kernel also has cb_push_back(output_id, 1), these calls will produce non-deterministic behavior because
 * cb pointers are not synchronized across threads. Per circular buffer index, only have one thread push tiles
 * to update the write pointer
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | ntiles    | The number of tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles) { PACK((llk_push_tiles<false, false>(cbid, ntiles))); }

// clang-format off
/**
 * Gets the L1 address of a tile in the specified circular buffer using mailbox-based
 * synchronization to ensure all compute threads (UNPACK, MATH, PACK) receive the same address.
 *
 * The UNPACK thread reads the tile address and distributes it to MATH and PACK threads
 * via mailbox, ensuring consistent values across all threads.
 *
 * Return value: The L1 address of the tile (same value on all threads)
 *
 * | Argument    | Description                          | Type     | Valid Range | Required |
 * |-------------|--------------------------------------|----------|-------------|----------|
 * | cb_id       | The index of the circular buffer (CB)| uint32_t | 0 to 31     | True     |
 * | tile_index  | The tile index within the CB         | uint32_t | 0 to CB size| True     |
 */
// clang-format on
ALWI uint32_t get_tile_address(uint32_t cb_id, uint32_t tile_index) {
    uint32_t address = 0;

    UNPACK({
        uint32_t operand_id = get_operand_id(cb_id);
        uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr;
        uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
        address = (base_address + offset_address) << 4;  // Convert to byte address

        mailbox_write(ckernel::ThreadId::MathThreadId, address);
        mailbox_write(ckernel::ThreadId::PackThreadId, address);
    })

    MATH(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

    return address;
}

// clang-format off
/**
 * Reads a uint32_t value from a tile in the specified circular buffer at a given element offset.
 * Uses mailbox-based synchronization to ensure all compute threads receive the same value.
 *
 * Return value: The uint32_t value at the specified offset (same value on all threads)
 *
 * | Argument       | Description                                | Type     | Valid Range | Required |
 * |----------------|--------------------------------------------|----------|-------------|----------|
 * | cb_id          | The index of the circular buffer (CB)      | uint32_t | 0 to 31     | True     |
 * | tile_index     | The tile index within the CB               | uint32_t | 0 to CB size| True     |
 * | element_offset | The uint32_t element offset within the tile| uint32_t | >= 0        | True     |
 */
// clang-format on
ALWI uint32_t read_tile_value(uint32_t cb_id, uint32_t tile_index, uint32_t element_offset) {
    uint32_t value = 0;

    UNPACK({
        uint32_t operand_id = get_operand_id(cb_id);
        uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr;
        uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
        uint32_t byte_address = (base_address + offset_address) << 4;  // Convert to byte address

        value = reinterpret_cast<volatile uint32_t*>(byte_address)[element_offset];

        mailbox_write(ckernel::ThreadId::MathThreadId, value);
        mailbox_write(ckernel::ThreadId::PackThreadId, value);
    })

    MATH(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

    return value;
}

}  // namespace ckernel
