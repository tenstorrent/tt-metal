// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/common_runtime_address_map.h"
#include "risc_attribs.h"
#include "debug/waypoint.h"

// The command queue read interface controls reads from the issue region, host owns the issue region write interface
// Commands and data to send to device are pushed into the issue region
struct CQReadInterface {
    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit; // range is inclusive of the limit
    uint32_t issue_fifo_rd_ptr;
    uint32_t issue_fifo_rd_toggle;
};

// The command queue write interface controls writes to the completion region, host owns the completion region read interface
// Data requests from device and event states are written to the completion region
struct CQWriteInterface {
    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit; // range is inclusive of the limit
    uint32_t completion_fifo_wr_ptr;
    uint32_t completion_fifo_wr_toggle;
};

struct CBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit; // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    // Save a cycle during init by writing 0 to the uint32 below
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    // used by packer for in-order packing
    uint32_t fifo_wr_tile_ptr;
};

extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

// NCRISC and BRISC setup read and write
// TRISC sets up read or write
inline void setup_cb_read_write_interfaces(uint32_t tt_l1_ptr *cb_l1_base, uint32_t start_cb_index, uint32_t max_cb_index, bool read, bool write, bool init_wr_tile_ptr) {

    constexpr uint32_t WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;

    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr = cb_l1_base + start_cb_index * WORDS_PER_CIRCULAR_BUFFER_CONFIG;

    for (uint32_t cb_id = start_cb_index; cb_id < max_cb_index; cb_id++) {

        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0];
        uint32_t fifo_size = circular_buffer_config_addr[1];
        uint32_t fifo_num_pages = circular_buffer_config_addr[2];
        uint32_t fifo_page_size = circular_buffer_config_addr[3];
        uint32_t fifo_limit = fifo_addr + fifo_size;

        cb_interface[cb_id].fifo_limit = fifo_limit;  // to check if we need to wrap
        if (write) {
            cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
        }
        if (read) {
            cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
        }
        cb_interface[cb_id].fifo_size = fifo_size;
        cb_interface[cb_id].tiles_acked_received_init = 0;
        if (write) {
            cb_interface[cb_id].fifo_num_pages = fifo_num_pages;
        }
        cb_interface[cb_id].fifo_page_size = fifo_page_size;

        if (init_wr_tile_ptr) {
            cb_interface[cb_id].fifo_wr_tile_ptr = 0;
        }

        circular_buffer_config_addr += WORDS_PER_CIRCULAR_BUFFER_CONFIG;
    }
}

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
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles to be pushed      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_push_back(const int32_t operand, const int32_t num_pages) {

    uint32_t num_words = num_pages * cb_interface[operand].fifo_page_size;

    volatile tt_reg_ptr uint32_t* pages_received_ptr = get_cb_tiles_received_ptr(operand);
    pages_received_ptr[0] += num_pages;

    cb_interface[operand].fifo_wr_ptr += num_words;

    // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
    // producer always writes into contiguous memory, it cannot wrap
    ASSERT(cb_interface[operand].fifo_wr_ptr <= cb_interface[operand].fifo_limit);
    if (cb_interface[operand].fifo_wr_ptr == cb_interface[operand].fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        cb_interface[operand].fifo_wr_ptr -= cb_interface[operand].fifo_size;
    }
}

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
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31 | True     |
 * | num_tiles | The number of tiles to be popped      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_pop_front(int32_t operand, int32_t num_pages) {
    volatile tt_reg_ptr uint32_t* pages_acked_ptr = get_cb_tiles_acked_ptr(operand);
    pages_acked_ptr[0] += num_pages;

    uint32_t num_words = num_pages * cb_interface[operand].fifo_page_size;

    cb_interface[operand].fifo_rd_ptr += num_words;

    // this will basically reset fifo_rd_ptr to fifo_addr -- no other wrap is legal
    // consumer always reads from contiguous memory, it cannot wrap
    ASSERT(cb_interface[operand].fifo_rd_ptr <= cb_interface[operand].fifo_limit);
    if (cb_interface[operand].fifo_rd_ptr == cb_interface[operand].fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        cb_interface[operand].fifo_rd_ptr -= cb_interface[operand].fifo_size;
    }
}

/**
 * Returns a pointer to the beginning of a memory block previously reserved
 * by cb_reserve_back. Note that this call is only valid between calls
 * to cb_reserve_back and cb_push_back. The amount of valid memory
 * is equal to the number of tiles requested in a prior cb_reserve_back call.
 *
 * CB total size must be an even multiple of this call.
 *
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
FORCE_INLINE
uint32_t get_write_ptr(uint32_t operand) {
    // return byte address (fifo_wr_ptr is 16B address)
    uint32_t wr_ptr_bytes = cb_interface[operand].fifo_wr_ptr << 4;
    return wr_ptr_bytes;
}

/**
 * Returns a pointer to the beginning of a memory block previously received
 * by cb_wait_front. Note that this call is only valid between calls
 * to cb_wait_front and cb_pop_front. The amount of valid memory
 * is equal to the number of tiles requested in a prior cb_wait_front call.
 *
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
FORCE_INLINE
uint32_t get_read_ptr(uint32_t operand) {

    // return byte address (fifo_rd_ptr is 16B address)
    uint32_t rd_ptr_bytes = cb_interface[operand].fifo_rd_ptr << 4;
    return rd_ptr_bytes;
}

inline void wait_for_sync_register_value(uint32_t addr, int32_t val) {
    volatile tt_reg_ptr uint32_t* reg_ptr = (volatile uint32_t*)addr;
    int32_t reg_value;
    WAYPOINT("SW");
    do {
        reg_value = reg_ptr[0];
    } while (reg_value != val);
    WAYPOINT("SD");
}

/**
 * A blocking call that waits for the specified number of tiles to be free in the specified circular buffer. This call
 * is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of free tiles to wait for  | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_reserve_back(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    WAYPOINT("CRBW");
    do {
        invalidate_l1_cache();
        // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
        // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
        uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
#ifdef ARCH_GRAYSKULL
        // The following test slows down by 5% when removing the barrier
        // TODO(pgk) investigate GS arbiter WAR in compiler, is this fixing an issue there?
        // models/experimental/stable_diffusion/tests/test_perf_unbatched_stable_diffusion.py::test_perf_bare_metal
        volatile uint32_t local_mem_barrier = pages_acked;
#endif
        uint16_t free_space_pages_wrap =
            cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
        free_space_pages = (int32_t)free_space_pages_wrap;
    } while (free_space_pages < num_pages);
    WAYPOINT("CRBD");
}

/**
 * A blocking call that waits for the specified number of tiles to be available in the specified circular buffer (CB).
 * This call is used by the consumer of the CB to wait for the producer to fill the CB with at least the specified number
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
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles to wait for       | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 * */
FORCE_INLINE
void cb_wait_front(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked = get_cb_tiles_acked_ptr(operand)[0];
    uint32_t pages_received_ptr = (uint32_t) get_cb_tiles_received_ptr(operand);

    uint16_t pages_received;

    WAYPOINT("CWFW");
    do {
        invalidate_l1_cache();
        pages_received = ((uint16_t)reg_read(pages_received_ptr)) - pages_acked;
    } while (pages_received < num_pages);
    WAYPOINT("CWFD");
}
