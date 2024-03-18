// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#define DATA_FORMATS_DEFINED
#endif
#if __has_include("generated_bank_to_noc_coord_mapping.h")
#include "generated_bank_to_noc_coord_mapping.h"
#endif

#include <stdint.h>

#include "circular_buffer.h"
#include "debug/sanitize_noc.h"
#include "debug/status.h"
#include "eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "risc_attribs.h"
#include "third_party/umd/device/tt_silicon_driver_common.hpp"
#include "debug/assert.h"

extern uint8_t noc_index;

/** @file */

/**
 * \private
 */
extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4
#define NOC_DISPATCH_MULTICAST_WRITE_VC 5 // Only to be used by the dispatch cores

inline uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    #if defined(COMPILE_FOR_ERISC)
        return eth_l1_mem::address_map::ERISC_L1_ARG_BASE + (arg_idx << 2);
    #else
        return L1_ARG_BASE + (arg_idx << 2);
    #endif
}

/**
 * Returns the value of an argument from kernel_args array provided during
 * kernel creation using CreateKernel calls.
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | The index of the argument          | uint32_t              | 0 to 255    | True     |
 * | T (template argument) | Data type of the returned argument | Any 4-byte sized type | N/A         | True     |
 */
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

/**
 * Returns the value of a constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateKernel calls.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | The index of the argument          | uint32_t              | 0 to 31     | True     |
 */
#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_##arg_idx

// replicated from ckernels_defs.h, which are currently not included in BRISC / NCRISC builds
// TODO: look into ckernels_defs.h included in NCRISC/BRISC builds
inline __attribute__((always_inline)) constexpr static std::int32_t GET_L1_TILE_SIZE(uint format) {
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::Float16_b): return ((2048 >> 4));
        case ((uint8_t)DataFormat::Float16): return ((2048 >> 4));

        case ((uint8_t)DataFormat::UInt16): return ((2048 >> 4));

        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((1024 >> 4) + (64 >> 4));

        case ((uint8_t)DataFormat::Float32): return ((4096 >> 4));

        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((512 >> 4) + (64 >> 4));

        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((256 >> 4) + (64 >> 4));
        default: return ((1024 >> 4) + (64 >> 4));
    };
}

inline __attribute__((always_inline)) constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::UInt16):
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << 11);
        case ((uint8_t)DataFormat::Float32): return (index << 12);
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << 8) + (index << 6));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << 9) + (index << 6));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << 10) + (index << 6));
    };
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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     |
 * | num_tiles | The number of tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
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

#ifdef DATA_FORMATS_DEFINED

// this API is used by both the reader and writer side of the CB
// it uses unpack_src_format, but because unpack_src_format == pack_dst_format, we can use either
// TODO: this can be made constexpr?
inline std::int32_t get_tile_size(const std::int32_t operand) {
    std::uint32_t input = operand;

    // L1 16B words
    std::uint32_t num_words = GET_L1_TILE_SIZE((uint)unpack_src_format[input]);

    // return bytes
    return num_words << 4;
}

inline DataFormat get_dataformat(const std::int32_t operand) {
    return static_cast<DataFormat>((uint)unpack_src_format[operand]);
}

#endif

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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | operand   | The index of the cirular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
inline __attribute__((always_inline)) uint32_t get_write_ptr(uint32_t operand) {
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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | operand   | The index of the cirular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
inline __attribute__((always_inline)) uint32_t get_read_ptr(uint32_t operand) {

    // return byte address (fifo_rd_ptr is 16B address)
    uint32_t rd_ptr_bytes = cb_interface[operand].fifo_rd_ptr << 4;
    return rd_ptr_bytes;
}

inline void wait_for_sync_register_value(uint32_t addr, int32_t val) {
    volatile tt_reg_ptr uint32_t* reg_ptr = (volatile uint32_t*)addr;
    int32_t reg_value;
    DEBUG_STATUS('S', 'W');
    do {
        reg_value = reg_ptr[0];
    } while (reg_value != val);
    DEBUG_STATUS('S', 'D');
}

/**
 * A blocking call that waits for the specified number of tiles to be free in the specified circular buffer. This call
 * is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of free tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_reserve_back(int32_t operand, int32_t num_pages) {
    kernel_profiler::mark_function_sum_start(CB_RESERVE_BACK_MARKER);
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');
    do {
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
    DEBUG_STATUS('C', 'R', 'B', 'D');
    kernel_profiler::mark_function_sum_end(CB_RESERVE_BACK_MARKER);
}

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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 * */
FORCE_INLINE
void cb_wait_front(int32_t operand, int32_t num_pages) {
    kernel_profiler::mark_function_sum_start(CB_WAIT_FRONT_MARKER);
    uint32_t pages_acked = get_cb_tiles_acked_ptr(operand)[0];
    uint32_t pages_received_ptr = (uint32_t) get_cb_tiles_received_ptr(operand);

    uint16_t pages_received;

    DEBUG_STATUS('C', 'W', 'F', 'W');
    do {
        pages_received = ((uint16_t)reg_read(pages_received_ptr)) - pages_acked;
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'W', 'F', 'D');
    kernel_profiler::mark_function_sum_end(CB_WAIT_FRONT_MARKER);
}

// NOC transfers

// simple APIs

FORCE_INLINE
std::uint64_t get_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        read from/write to via the noc
    */
    return NOC_MULTICAST_ADDR(NOC_X(noc_x_start), NOC_Y(noc_y_start), NOC_X(noc_x_end), NOC_Y(noc_y_end), addr);
}

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */

    return NOC_XY_ADDR(NOC_X(noc_x), NOC_Y(noc_y), addr);
}

/*
    Need an alias to get_noc_addr so that the structs below don't confuse the above get_noc_addr with
    the struct variant
*/
FORCE_INLINE
std::uint64_t get_noc_addr_helper(std::uint32_t noc_xy, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return ((uint64_t)(noc_xy) << 32) | addr;
}



uint64_t get_dram_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0) {
    uint32_t bank_id;
    uint32_t addr;
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
    bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
    addr = (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) * align(page_size, 32)) + bank_base_address + offset;
#else
    bank_id = id & (NUM_DRAM_BANKS - 1);
    addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) * align(page_size, 32)) + bank_base_address + offset;
#endif

    addr += bank_to_dram_offset[bank_id];
    uint32_t noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);

    return noc_addr;
}

uint64_t get_l1_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0) {
    uint32_t bank_id;
    uint32_t addr;
#ifdef IS_NOT_POW2_NUM_L1_BANKS
    bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
    addr = (udivsi3_const_divisor<NUM_L1_BANKS>(id) * align(page_size, 32)) + bank_base_address + offset;
#else
    bank_id = id & (NUM_L1_BANKS - 1);
    addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) * align(page_size, 32)) + bank_base_address + offset;
#endif

    addr += bank_to_l1_offset[bank_id];
    uint32_t noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t get_system_memory_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t base_addr, const uint32_t offset = 0) {
    constexpr static uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;
    uint32_t addr = base_addr + page_size * id + offset;
    uint64_t noc_addr = pcie_core_noc_encoding | addr;
    return noc_addr;
}

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t addr) {
    /*
        Get an encoding which contains the address in L1 on the current core that you want to
        read from/write to via the noc
    */
    return NOC_XY_ADDR(my_x[noc_index], my_y[noc_index], addr);
}

/**
 * Initiates an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). The destination is in L1 memory on the Tensix core
 * executing this function call. Also, see \a noc_async_read_barrier.
 *
 * The source node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument          | Description                                        | Data type | Valid range | required |
 * |-------------------|----------------------------------------------------|-----------|------------------------------------------|----------|
 * | src_noc_addr      | Encoding of the source DRAM location (x,y)+address | uint64_t  | DOX-TODO(ref to explain valid coords)    | Yes      |
 * | dst_local_l1_addr | Address in local L1 memory                         | uint32_t  | 0..1MB                                   | Yes      |
 * | size              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                                   | Yes      |
 */
inline
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_STATUS('N', 'A', 'R', 'W');
    DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);
    DEBUG_SANITIZE_WORKER_ADDR(dst_local_l1_addr, size);
    ncrisc_noc_fast_read_any_len(noc_index, NCRISC_RD_CMD_BUF, src_noc_addr, dst_local_l1_addr, size);
    DEBUG_STATUS('N', 'A', 'R', 'D');
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    DEBUG_STATUS('R', 'P', 'W');
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('R', 'P', 'D');

    DEBUG_STATUS('N', 'A', 'R', 'W');
    DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);
    DEBUG_SANITIZE_WORKER_ADDR(dst_local_l1_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, (uint32_t)src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;

    DEBUG_STATUS('N', 'A', 'R', 'D');
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet_set_state(std::uint64_t src_noc_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    DEBUG_STATUS('R', 'P', 'W');
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('R', 'P', 'D');

    DEBUG_STATUS('N', 'A', 'R', 'W');
    DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, size);

    DEBUG_STATUS('N', 'A', 'R', 'D');
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool inc_num_issued = true>
FORCE_INLINE
void noc_async_read_one_packet_with_state(std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    DEBUG_STATUS('R', 'P', 'W');
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('R', 'P', 'D');

    DEBUG_STATUS('N', 'A', 'R', 'W');

    // TODO: need a way sanitize size + addr w/o directly providing x/y here (grab x/y form state?)
    // DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);
    // DEBUG_SANITIZE_WORKER_ADDR(dst_local_l1_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (inc_num_issued) {
        noc_reads_num_issued[noc_index] += 1;
    }

    DEBUG_STATUS('N', 'A', 'R', 'D');
}

// TODO: write docs
FORCE_INLINE
void noc_async_read_set_state(std::uint64_t src_noc_addr) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    DEBUG_STATUS('N', 'A', 'R', 'W');
    DEBUG_STATUS('R', 'P', 'W');
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('R', 'P', 'D');

    // TODO: need to sanitize in noc_async_read_with_state
    // DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_addr >> 32);

    DEBUG_STATUS('N', 'A', 'R', 'D');
}

// TODO: write docs
template <bool inc_num_issued = true>
FORCE_INLINE
void noc_async_read_with_state(std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_STATUS('N', 'A', 'R', 'W');

    // TODO: need a way sanitize size + addr w/o directly providing x/y here (grab x/y form state?)
    // DEBUG_SANITIZE_NOC_ADDR(src_noc_addr, size);
    DEBUG_SANITIZE_WORKER_ADDR(dst_local_l1_addr, size);

    while (size > NOC_MAX_BURST_SIZE) {
        DEBUG_STATUS('R', 'P', 'W');
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        DEBUG_STATUS('R', 'P', 'D');

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst_local_l1_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_noc_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        size -= NOC_MAX_BURST_SIZE;
        src_noc_addr += NOC_MAX_BURST_SIZE;
        dst_local_l1_addr += NOC_MAX_BURST_SIZE;
        if constexpr (inc_num_issued) {
            noc_reads_num_issued[noc_index] += 1;
        }
    }

    // left-over packet
    DEBUG_STATUS('R', 'P', 'W');
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS('R', 'P', 'D');

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (inc_num_issued) {
        noc_reads_num_issued[noc_index] += 1;
    }

    DEBUG_STATUS('N', 'A', 'R', 'D');
}

FORCE_INLINE
void noc_async_read_inc_num_issued(std::uint32_t num_issued_reads_inc) {
    noc_reads_num_issued[noc_index] += num_issued_reads_inc;
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_write_one_packet(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size) {

    DEBUG_STATUS('N', 'W', 'P', 'W');
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr, size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
    DEBUG_STATUS('N', 'W', 'P', 'D');

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dst_noc_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE,  size);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc_index] += 1;
    noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
 }

// TODO: write docs
// this sets the state for issuing a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE
void noc_async_write_one_packet_set_state(std::uint64_t dst_noc_addr, std::uint32_t size) {

    DEBUG_STATUS('N', 'W', 'P', 'W');
    DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr, size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
    DEBUG_STATUS('N', 'W', 'P', 'D');

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                (non_posted ? NOC_CMD_RESP_MARKED : 0x0);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dst_noc_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE,  size);
 }

// TODO: write docs
// this issues only a single packet with cmd buf state with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE
void noc_async_write_one_packet_with_state(std::uint32_t src_local_l1_addr, std::uint32_t dst_noc_addr) {

    DEBUG_STATUS('N', 'W', 'P', 'W');
    // TODO: need a way sanitize size + addr w/o directly providing x/y here (grab x/y form state?)
    // DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    // DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr, size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
    DEBUG_STATUS('N', 'W', 'P', 'D');

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (non_posted) {
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
    }
 }

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    uint32_t page_size;          // Num bytes in page.

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        uint32_t bank_id;
        uint32_t addr;
        uint32_t noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr =
                (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) * align(this->page_size, 32)) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) * align(this->page_size, 32)) + this->bank_base_address + offset;
#endif
            addr += bank_to_dram_offset[bank_id];
            noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            addr =
                (udivsi3_const_divisor<NUM_L1_BANKS>(id) * align(this->page_size, 32)) + this->bank_base_address + offset;
#else
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = (id >> LOG_BASE_2_OF_NUM_L1_BANKS) * align(this->page_size, 32) + this->bank_base_address + offset;
#endif
            addr += bank_to_l1_offset[bank_id];
            noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0) const {
        noc_async_read(this->get_noc_addr(id, offset), dest_addr, page_size);
    }
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;  // WARNING: This struct is used for optimized get_noc_addr in which case
                                             // you know that bank_unit_size is a power of 2

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        // So far, only using this for DRAM, but will eventually generalize to allow usage in L1 as well
        uint32_t bank_id;
        uint32_t addr;
        uint32_t noc_xy;

#ifdef TEMP_DEBUG2
#endif
        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr =
                (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            addr += bank_to_dram_offset[bank_id];
            noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            addr =
                (udivsi3_const_divisor<NUM_L1_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            addr += bank_to_l1_offset[bank_id];
            noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

template <bool DRAM>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    uint32_t page_size;          // Num bytes in bank unit.
    DataFormat data_format;      // Dataformat

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        uint32_t bank_id;
        uint32_t addr;
        uint32_t noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                   this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                   this->bank_base_address + offset;
#endif
            addr += bank_to_dram_offset[bank_id];
            noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_L1_BANKS>(id)) +
                   this->bank_base_address + offset;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) +
                   this->bank_base_address + offset;
#endif
            addr += bank_to_l1_offset[bank_id];
            noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_tile(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t bank_id;
        uint32_t src_addr;
        uint32_t src_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                       this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                       this->bank_base_address + offset;
#endif
            src_addr += bank_to_dram_offset[bank_id];
            src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_L1_BANKS>(id)) +
                       this->bank_base_address + offset;
#else
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) +
                       this->bank_base_address + offset;
#endif
            src_addr += bank_to_l1_offset[bank_id];
            src_noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        DEBUG_STATUS('N', 'R', 'T', 'W');
        DEBUG_SANITIZE_NOC_ADDR(get_noc_addr_helper(src_noc_xy, src_addr), this->page_size);
        DEBUG_SANITIZE_WORKER_ADDR(dest_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        DEBUG_STATUS('N', 'R', 'T', 'D');

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr) const {
        uint32_t bank_id;
        uint32_t dest_addr;
        uint32_t dest_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            dest_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                        this->bank_base_address;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            dest_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                        this->bank_base_address;
#endif
            dest_addr += bank_to_dram_offset[bank_id];
            dest_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            dest_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_L1_BANKS>(id)) +
                        this->bank_base_address;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            dest_addr =
                MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) + this->bank_base_address;
#endif
            dest_addr += bank_to_l1_offset[bank_id];
            dest_noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        DEBUG_STATUS('N', 'W', 'T', 'W');
        DEBUG_SANITIZE_WORKER_ADDR(src_addr, this->page_size);
        DEBUG_SANITIZE_NOC_ADDR(get_noc_addr_helper(dest_noc_xy, dest_addr), this->page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
        DEBUG_STATUS('N', 'W', 'T', 'D');

        uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
    }
};

// TODO: add noc_async_write_page
// TODO: need static assert + host assert that page size <= 8192, hard constraint
template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    uint32_t log_base_2_of_page_size;          // Num bytes in bank unit.

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t bank_id;
        uint32_t src_addr;
        uint32_t src_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            src_addr = (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            src_addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            src_addr += bank_to_dram_offset[bank_id];
            src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            src_addr = (udivsi3_const_divisor<NUM_L1_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            src_addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            src_addr += bank_to_l1_offset[bank_id];
            src_noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        DEBUG_STATUS('N', 'R', 'P', 'W');
        DEBUG_SANITIZE_NOC_ADDR(get_noc_addr_helper(src_noc_xy, src_addr), log_base_2_of_page_size);
        DEBUG_SANITIZE_WORKER_ADDR(dest_addr, this->log_base_2_of_page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        DEBUG_STATUS('N', 'R', 'P', 'D');

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, 1 << log_base_2_of_page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_read_partial_page(const uint32_t id, uint32_t dest_addr, const uint32_t size, const uint32_t offset) const {
        uint32_t bank_id;
        uint32_t src_addr;
        uint32_t src_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            src_addr = (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            src_addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            src_addr += bank_to_dram_offset[bank_id];
            src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            src_addr = (udivsi3_const_divisor<NUM_L1_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            src_addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            src_addr += bank_to_l1_offset[bank_id];
            src_noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        DEBUG_STATUS('R', 'P', 'W');
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        DEBUG_STATUS('R', 'P', 'D');

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_write_page(const uint32_t id, uint32_t src_addr, const uint32_t write_size_bytes, const uint32_t offset = 0) const {
        uint32_t bank_id;
        uint32_t dest_addr;
        uint32_t dest_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            dest_addr = (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_DRAM_BANKS - 1);
            dest_addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            dest_addr += bank_to_dram_offset[bank_id];
            dest_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
        } else {
#ifdef IS_NOT_POW2_NUM_L1_BANKS
            bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
            dest_addr = (udivsi3_const_divisor<NUM_L1_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#else
            bank_id = id & (NUM_L1_BANKS - 1);
            dest_addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address + offset;
#endif
            dest_addr += bank_to_l1_offset[bank_id];
            dest_noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        }

        DEBUG_STATUS('N', 'W', 'P', 'W');
        DEBUG_SANITIZE_WORKER_ADDR(src_addr, write_size_bytes);
        DEBUG_SANITIZE_NOC_ADDR(get_noc_addr_helper(dest_noc_xy, dest_addr), write_size_bytes);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
        DEBUG_STATUS('N', 'W', 'P', 'D');

        uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE,  write_size_bytes);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
    }
};

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, uint32_t offset = 0) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset);
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, std::uint32_t dst_local_l1_addr, uint32_t offset = 0) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_page(id, dst_local_l1_addr, offset);
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, std::uint32_t dst_local_l1_addr, uint32_t offset = 0) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_tile(id, dst_local_l1_addr, offset);
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the
 * Tensix core executing this function call. The destination is specified using
 * a uint64_t encoding referencing an on-chip node located at NOC coordinates
 * (x,y) and a local address created using get_noc_addr function. Also, see
 * \a noc_async_write_barrier.
 *
 * The destination node can be either a DRAM bank, Tensix core+L1 memory
 * address or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | src_local_l1_addr | Source address in local L1 memory                       | uint32_t | 0..1MB | True     |
 * | dst_noc_addr      | Encoding of the destination DRAM location (x,y)+address | uint64_t | DOX-TODO(insert a reference  to what constitutes valid coords) | True     |
 * | size              | Size of data transfer in bytes | uint32_t | 0..1MB                                                    | True     |
 */
inline
void noc_async_write(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size) {
    DEBUG_STATUS('N', 'A', 'W', 'W');
    DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr, size);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        NOC_UNICAST_WRITE_VC,
        false,
        false,
        1);
    DEBUG_STATUS('N', 'A', 'W', 'D');
}

template <bool DRAM>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, std::uint32_t src_local_l1_addr) {
    s.noc_async_write_tile(id, src_local_l1_addr);
}

FORCE_INLINE
uint32_t get_semaphore(uint32_t semaphore_id) {
    return SEMAPHORE_BASE + semaphore_id * L1_ALIGNMENT;
}

FORCE_INLINE
uint32_t eth_get_semaphore(uint32_t semaphore_id) {
    return eth_l1_mem::address_map::SEMAPHORE_BASE + semaphore_id * L1_ALIGNMENT;
}

inline
void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr) {
    DEBUG_STATUS('N', 'S', 'S', 'W');
    DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr, 4);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        4 /* size in bytes */,
        NOC_UNICAST_WRITE_VC,
        false,
        false,
        1);
    DEBUG_STATUS('N', 'S', 'S', 'D');
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the
 * Tensix core executing this function call to a rectangular destination grid.
 * The destinations are specified using a uint64_t encoding referencing an
 * on-chip grid of nodes located at NOC coordinate range
 * (x_start,y_start,x_end,y_end) and a local address created using
 * *get_noc_multicast_addr* function. Also, *see noc_async_write_barrier*.
 *
 * The destination nodes can only be a set of Tensix cores + L1 memory address.
 * The destination nodes must form a rectangular grid. The destination L1
 * memory address must be the same on all destination nodes.
 *
 * With this API, the multicast sender cannot be part of the multicast
 * destinations. If the multicast sender has to be in the multicast
 * destinations (i.e. must perform a local L1 write), the other API variant
 * *noc_async_write_multicast_loopback_src* can be used.
 *
 * Note: there is no restriction on the number of destinations, i.e. the
 * multicast destinations can span the full chip. However, as mentioned
 * previosuly, the multicast source cannot be part of the destinations. So, the
 * maximum number of destinations is 119.
 *
 * Return value: None
 *
 * | Argument               | Description                                                              | Type     | Valid Range                                                   | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t | 0..1MB                                                        | True     |
 * | dst_noc_addr_multicast | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | size                   | Size of data transfer in bytes | uint32_t | 0..1MB | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t | 0..119                                                        | True     |
 */
inline
void noc_async_write_multicast(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false) {
    DEBUG_STATUS('N', 'M', 'W', 'W');
    DEBUG_SANITIZE_NOC_MULTI_ADDR(dst_noc_addr_multicast, size);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests);
    DEBUG_STATUS('N', 'M', 'W', 'D');
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the
 * Tensix core executing this function call to a rectangular destination grid.
 * The destinations are specified using a uint64_t encoding referencing an
 * on-chip grid of nodes located at NOC coordinate range
 * (x_start,y_start,x_end,y_end) and a local address created using
 * *get_noc_multicast_addr* function. The size of data that is sent is 4 Bytes.
 * This is usually used to set a semaphore value at the destination nodes, as a
 * way of a synchronization mechanism. The same as *noc_async_write_multicast*
 * with preset size of 4 Bytes.
 *
 * Return value: None
 *
 * | Argument               | Description                                                              | Type     | Valid Range                                               | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t | 0..1MB                                                    | True     |
 * | dst_noc_addr_multicast | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting | uint32_t | 0..119                                                    | True     |
 */
inline
void noc_semaphore_set_multicast(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests, bool linked = false) {
    DEBUG_STATUS('N', 'S', 'M', 'W');
    DEBUG_SANITIZE_NOC_MULTI_ADDR(dst_noc_addr_multicast, 4);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests);
    DEBUG_STATUS('N', 'S', 'M', 'D');
}

inline
void noc_semaphore_set_multicast_loopback_src(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests, bool linked = false) {
    DEBUG_STATUS('N', 'S', 'M', 'W');
    DEBUG_SANITIZE_NOC_MULTI_ADDR(dst_noc_addr_multicast, 4);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len_loopback_src(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests);
    DEBUG_STATUS('N', 'S', 'M', 'D');
}

inline
void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false) {
    DEBUG_STATUS('N', 'M', 'L', 'W');
    DEBUG_SANITIZE_NOC_MULTI_ADDR(dst_noc_addr_multicast, size);
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len_loopback_src(
        noc_index,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests);
    DEBUG_STATUS('N', 'M', 'L', 'D');
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_read*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_read* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_read_barrier() {
    DEBUG_STATUS('N', 'R', 'B', 'W');
    while (!ncrisc_noc_reads_flushed(noc_index))
        ;
    DEBUG_STATUS('N', 'R', 'B', 'D');
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_write* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_write_barrier() {
    DEBUG_STATUS('N', 'W', 'B', 'W');
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index))
        ;
    DEBUG_STATUS('N', 'W', 'B', 'D');
}

/**
 * This blocking call waits for all outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to depart, but will not wait
 * for them to complete
*/
FORCE_INLINE
void noc_async_writes_flushed() {
    DEBUG_STATUS('N', 'W', 'B', 'W');
    while (!ncrisc_noc_nonposted_writes_sent(noc_index))
        ;
    DEBUG_STATUS('N', 'W', 'B', 'D');
}

/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True |
 */
FORCE_INLINE
void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    DEBUG_STATUS('N', 'S', 'W');
    while ((*sem_addr) != val)
        ;
    DEBUG_STATUS('N', 'S', 'D');
}

/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal or greater than a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True |
 */
FORCE_INLINE
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    DEBUG_STATUS('N', 'S', 'M', 'W');
    while ((*sem_addr) < val)
        ;
    DEBUG_STATUS('N', 'S', 'M', 'D');
}

/**
 * Sets the value of a local L1 memory address on the Tensix core executing
 * this function to a specific value. This L1 memory address is used as a
 * semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
 * *noc_semaphore_wait*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | Value to set the semaphore to                                  | uint32_t | Any uint32_t value | True |
 */
FORCE_INLINE
void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    // set semaphore value to val
    (*sem_addr) = val;
}

/**
 * The Tensix core executing this function call initiates an atomic increment
 * (with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory
 * address is used as a semaphore of size 4 Bytes, as a synchronization
 * mechanism.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range                                                   | Required |
 * |-----------|----------------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | addr      | Encoding of the destination location (x,y)+address             | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | incr      | The value to increment by                                      | uint32_t | Any uint32_t value                                            | True     |
 */
inline
void noc_semaphore_inc(uint64_t addr, uint32_t incr) {
    /*
    [REFER TO grayskull/noc/noc.h for the documentation of noc_atomic_increment()]
    Generic increment with 32-bit wrap.
  */
    DEBUG_STATUS('N', 'S', 'I', 'W');
    DEBUG_SANITIZE_NOC_ADDR(addr, 4);
    noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, addr, NOC_UNICAST_WRITE_VC, incr, 31 /*wrap*/, false /*linked*/);
    DEBUG_STATUS('N', 'S', 'I', 'D');
}

enum class BufferType: uint8_t {
    DRAM = 0,
    L1 = 1,
    SYSTEM_MEMORY = 2
};

FORCE_INLINE
uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a: b; }


template<bool READ>
FORCE_INLINE void noc_async_sharded_read_write_helper(
                                    const uint32_t num_cores,
                                    const uint32_t page_size,
                                    const uint32_t bank_base_address,
                                    volatile tt_l1_ptr uint32_t* base_command_addr,
                                    const uint32_t addr,
                                    const uint32_t num_pages,
                                    const uint32_t page_id
                                    ){

    uint32_t pages_start = 0;
    uint32_t pages_end = 0;


    //first get to correct core
    uint32_t core_word_id = 0;
    while (not (page_id >= pages_start and page_id < pages_end)) {
        uint32_t num_pages_core = base_command_addr[core_word_id];
        pages_end = pages_start + num_pages_core;
        uint32_t core_id_x = base_command_addr[core_word_id + 1];
        uint32_t core_id_y = base_command_addr[core_word_id + 2];
        if (not (page_id >= pages_start and page_id < pages_end)) {
            pages_start = pages_end;
        }
        core_word_id+=NUM_ENTRIES_PER_SHARD;
    }

    core_word_id-= NUM_ENTRIES_PER_SHARD;

    uint32_t flattened_page_id = page_id;

    uint32_t host_page_id = 0;
    uint32_t host_offset = 0;
    uint32_t core_page_id = (flattened_page_id - pages_start);
    uint32_t core_offset = core_page_id * page_size;

    uint32_t num_pages_left = num_pages;

    while (num_pages_left > 0) {
        uint32_t num_pages_core = base_command_addr[core_word_id];
        pages_end = pages_start + num_pages_core;
        uint32_t core_id_x = base_command_addr[core_word_id + 1];
        uint32_t core_id_y = base_command_addr[core_word_id + 2];


        //now curr_page_id pointing to beginning of section we want in this core
        uint32_t num_pages_write_core = min(pages_end - flattened_page_id, num_pages_left);

        uint32_t size_in_bytes_written = num_pages_write_core *page_size;

        //Writing at beginning of core
        uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
                                        bank_base_address + core_offset);

        if constexpr (READ) {
            noc_async_read(noc_address, addr + host_offset, size_in_bytes_written);
        }
        else{
            noc_async_write(addr + host_offset, noc_address, size_in_bytes_written);
        }

        num_pages_left-= num_pages_write_core;
        host_offset += size_in_bytes_written;
        core_offset = 0;
        pages_start = pages_end;
        flattened_page_id = pages_start;
        core_word_id += NUM_ENTRIES_PER_SHARD;
    }
}

class Buffer {
   private:
    uint32_t bank_base_address;
    uint32_t page_size_;
    uint64_t (*get_noc_addr_helper)(const uint32_t, const uint32_t, const uint32_t, const uint32_t);
    BufferType type;
    bool sharded;

    //sharding
    volatile tt_l1_ptr uint32_t* base_command_addr_;
    uint32_t num_cores_;

    void set_type(const BufferType type) {
        this->type = type;
        switch (type) {
            case BufferType::DRAM:          this->get_noc_addr_helper = get_dram_noc_addr; break;
            case BufferType::L1:            this->get_noc_addr_helper = get_l1_noc_addr; break;
            case BufferType::SYSTEM_MEMORY: this->get_noc_addr_helper = get_system_memory_noc_addr; break;
        }
    }
    uint64_t get_noc_addr_(const uint32_t id, const uint32_t offset = 0) {
        uint64_t noc_addr = this->get_noc_addr_helper(id, this->page_size_, this->bank_base_address, offset);
        return this->get_noc_addr_helper(id, this->page_size_, this->bank_base_address, offset);
    }

   public:

    Buffer(){;}

    Buffer(const BufferType type, const uint32_t bank_base_address, const uint32_t page_size) {
        this->init(type, bank_base_address, page_size);
    }

    Buffer(uint32_t page_size, uint32_t num_cores,  uint32_t addr, volatile tt_l1_ptr uint32_t* command_ptr) {
        this->init_sharded(page_size, num_cores, addr, command_ptr);
    }

    void init(const BufferType type, const uint32_t bank_base_address, const uint32_t page_size) {
        this->set_type(type);
        this->bank_base_address = bank_base_address;
        this->page_size_ = page_size;
        this->sharded = false;
    }

    BufferType get_type() { return this->type; }

    void init_sharded(uint32_t page_size, uint32_t num_cores,  uint32_t addr, volatile tt_l1_ptr uint32_t* command_ptr){
        this->type = BufferType::L1;
        this->bank_base_address = addr;
        this->page_size_ = page_size;
        this->base_command_addr_ = command_ptr;
        this->num_cores_ = num_cores;
        this->sharded = true;

    }
    uint32_t page_size() { return this->page_size_; }


    void noc_async_write_buffer(uint32_t src, const uint32_t id, const uint32_t num_pages, const uint32_t offset=0) {
        #ifndef COMPILE_FOR_ERISC
        // DPRINT << "BUFFER WRITE" << ENDL();
        #endif
        if (this->sharded) {
            noc_async_sharded_read_write_helper<false>(this->num_cores_, this->page_size_,
                                                this->bank_base_address, this->base_command_addr_,
                                                src,  num_pages, id);
        }
        else {
            if (this->type == BufferType::SYSTEM_MEMORY) {
                noc_async_write(src, this->get_noc_addr_(id, offset), this->page_size_ * num_pages);
            }
            else {
                // DPRINT << "BUF WRITE " << num_pages << " at " << (uint32_t)my_x[0] << ", " << (uint32_t)my_y[0] << ENDL();

                // #ifndef COMPILE_FOR_ERISC
                // #endif
                for (uint32_t i = 0; i < num_pages; i++) {
                    uint64_t address = this->get_noc_addr_(id + i, offset);
                    // #ifndef COMPILE_FOR_ERISC
                    // DPRINT << "DRAM BUF WRITE" << ENDL();
                    // uint32_t end = src + page_size_;
                    // for (uint32_t i = src; i < end; i += sizeof(uint32_t)) {
                    //     DPRINT << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(i) << ENDL();
                    // }
                    // DPRINT << ENDL();
                    // #endif
                    noc_async_write(src, address, this->page_size_);
                    src += this->page_size_;
                }
            }
        }

    }

    void noc_async_read_buffer(uint32_t dst, const uint32_t id, const uint32_t num_pages, const uint32_t offset=0) {
        #ifndef COMPILE_FOR_ERISC
        // DPRINT << "BUFFER READ" << ENDL();
        #endif
        if (this->sharded) {
            noc_async_sharded_read_write_helper<true>(this->num_cores_, this->page_size_,
                                            this->bank_base_address, this->base_command_addr_,
                                            dst,  num_pages, id);
        }
        else {
            if (this->type == BufferType::SYSTEM_MEMORY) {
                noc_async_read(this->get_noc_addr_(id, offset), dst, this->page_size_ * num_pages);
            }
            else {
                for (uint32_t i = 0; i < num_pages; i++) {
                    uint64_t address = this->get_noc_addr_(id + i, offset);
                    noc_async_read(address, dst, this->page_size_);
                    dst += this->page_size_;
                }
            }
        }
    }
};
