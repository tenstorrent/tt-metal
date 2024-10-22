// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#define DATA_FORMATS_DEFINED
#endif
#if __has_include("generated_bank_to_noc_coord_mapping.h")
#include "generated_bank_to_noc_coord_mapping.h"
#endif

#include <stdint.h>

#include "core_config.h"
#include "circular_buffer.h"
#include "debug/sanitize_noc.h"
#include "debug/waypoint.h"
#include "eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "risc_attribs.h"
#include "third_party/umd/device/tt_silicon_driver_common.hpp"
#include "debug/assert.h"
#include "dev_msgs.h"

#if defined(KERNEL_BUILD)
constexpr uint8_t noc_index = NOC_INDEX;
constexpr uint8_t noc_mode = NOC_MODE;
#else
extern uint8_t noc_index;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
#endif
extern uint32_t tt_l1_ptr *rta_l1_base;
extern uint32_t tt_l1_ptr *crta_l1_base;
extern uint32_t tt_l1_ptr *sem_l1_base[];

#if defined(KERNEL_BUILD)
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t read_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_RD_CMD_BUF : DYNAMIC_NOC_BRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_WR_CMD_BUF : DYNAMIC_NOC_BRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_WR_REG_CMD_BUF : DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_AT_CMD_BUF : DYNAMIC_NOC_BRISC_AT_CMD_BUF;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t read_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_RD_CMD_BUF : DYNAMIC_NOC_NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_WR_CMD_BUF : DYNAMIC_NOC_NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_WR_REG_CMD_BUF : DYNAMIC_NOC_NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_AT_CMD_BUF : DYNAMIC_NOC_NCRISC_AT_CMD_BUF;
#else // use the default cmf buffers for compute/eth
constexpr uint32_t read_cmd_buf = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NCRISC_AT_CMD_BUF;
#endif
#else // FW build
constexpr uint32_t read_cmd_buf = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NCRISC_AT_CMD_BUF;
#endif

/** @file */

/**
 * \private
 */
extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4
#define NOC_DISPATCH_MULTICAST_WRITE_VC 5 // Only to be used by the dispatch cores

FORCE_INLINE
uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

namespace interleaved_addr_gen {

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_offset_index(uint32_t id) {
    if constexpr (DRAM) {   // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return udivsi3_const_divisor<NUM_DRAM_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_DRAM_BANKS;
#endif
    } else {                // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return udivsi3_const_divisor<NUM_L1_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_L1_BANKS;
#endif
    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index) {
    if constexpr (DRAM) {   // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return id - bank_offset_index * NUM_DRAM_BANKS;
#else
        return id & (NUM_DRAM_BANKS - 1);
#endif
    } else {                // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return id - bank_offset_index * NUM_L1_BANKS;
#else
        return id & (NUM_L1_BANKS - 1);
#endif
    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = noc_index) {
    if constexpr (DRAM) {   // DRAM
        return dram_bank_to_noc_xy[noc][bank_index];
    } else {                // L1
        return l1_bank_to_noc_xy[noc][bank_index];

    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_offset(uint32_t bank_index) {
    if constexpr (DRAM) {   // DRAM
        return bank_to_dram_offset[bank_index];
    } else {                // L1
        return bank_to_l1_offset[bank_index];

    }
}

}  // namespace addrgen


/**
 * Returns the address in L1 for a given runtime argument index for unique (per core) runtime arguments set via SetRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given unique runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | arg_idx        | Unique Runtime argument index                                           | uint32_t | 0 to 255                                       | True     |
 */
static FORCE_INLINE
uint32_t get_arg_addr(int arg_idx) {
    return (uint32_t)&rta_l1_base[arg_idx];;
}

/**
 * Returns the address in L1 for a given runtime argument index for common (all cores) runtime arguments set via SetCommonRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given common runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | arg_idx        | Common Runtime argument index                                           | uint32_t | 0 to 255                                       | True     |
 */
static FORCE_INLINE
uint32_t get_common_arg_addr(int arg_idx) {
    return (uint32_t)&crta_l1_base[arg_idx];
}

/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs() API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range               | Required |
 * |-----------------------|------------------------------------------------|-----------------------|---------------------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 255                  | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A                       | True     |
 */
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

/**
 * Returns the value at a given runtime argument index for common (all cores) runtime arguments set via SetCommonRuntimeArgs() API.
 *
 * Return value: The value associated with the common runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range               | Required |
 * |-----------------------|------------------------------------------------|-----------------------|---------------------------|----------|
 * | arg_idx               | Common Runtime argument index                  | uint32_t              | 0 to 255                  | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A                       | True     |
 */
template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T*)(get_common_arg_addr(arg_idx)));
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


FORCE_INLINE
constexpr static std::int32_t GET_TILE_SIZE(uint format) {
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::Float16_b): return ((2048));
        case ((uint8_t)DataFormat::Float16): return ((2048));

        case ((uint8_t)DataFormat::UInt8): return ((1024));
        case ((uint8_t)DataFormat::UInt16): return ((2048));

        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((1024) + (64));

        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32):
        case ((uint8_t)DataFormat::Float32): return ((4096));

        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((512) + (64));

        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((256) + (64));
        default: return ((1024) + (64));
    };
}

template <uint32_t tile_hw = 1024>
FORCE_INLINE constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    constexpr uint8_t datum_shift = (tile_hw == 1024) ? 10 :
                                    (tile_hw == 512)  ? 9  :
                                    (tile_hw == 256)  ? 8  : 10;

    constexpr uint8_t exp_shift   = (tile_hw == 1024) ? 2  :
                                    (tile_hw == 512)  ? 1  :
                                    (tile_hw == 256)  ? 0  : 2;
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::UInt8): return (index << datum_shift);
        case ((uint8_t)DataFormat::UInt16):
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << (datum_shift + 1));
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32):
        case ((uint8_t)DataFormat::Float32): return (index << (datum_shift + 2));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << (datum_shift - 2)) + (index << (4 + exp_shift)));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << (datum_shift - 1)) + (index << (4 + exp_shift)));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << datum_shift) + (index << (4 + exp_shift)));
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

#ifdef DATA_FORMATS_DEFINED

// this API is used by both the reader and writer side of the CB
// it uses unpack_src_format, but because unpack_src_format == pack_dst_format, we can use either
constexpr inline std::int32_t get_tile_size(const std::int32_t operand) {
    std::uint32_t input = operand;

    // L1 16B words
    std::uint32_t num_words = (uint)unpack_tile_size[input];

    // return bytes
    return num_words;
}

constexpr inline uint32_t get_tile_hw(const std::int32_t operand) {
    std::uint32_t input = operand;
    return (uint32_t)unpack_tile_r_dim[input] * (uint32_t)unpack_tile_c_dim[input];
}

constexpr inline uint32_t get_tile_num_faces(const std::int32_t operand) {
    std::uint32_t input = operand;
    return (uint32_t)unpack_tile_num_faces[input];
}

constexpr inline DataFormat get_dataformat(const std::int32_t operand) {
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

// NOC transfers

// simple APIs

FORCE_INLINE
std::uint64_t get_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr,
    uint8_t noc = noc_index) {
    /*
        Get an encoding which contains tensix core and address you want to
        read from/write to via the noc
    */
    return NOC_MULTICAST_ADDR(DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start), DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end), addr);
}

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr, uint8_t noc = noc_index) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */

    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, noc_x), DYNAMIC_NOC_Y(noc, noc_y), addr);
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
    return ((uint64_t)(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;
}



uint64_t get_dram_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(id, bank_offset_index);
    uint32_t addr = (bank_offset_index * align(page_size, ALLOCATOR_ALIGNMENT)) + bank_base_address + offset + bank_to_dram_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t get_l1_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<false>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<false>(id, bank_offset_index);
    uint32_t addr = (bank_offset_index * align(page_size, ALLOCATOR_ALIGNMENT)) + bank_base_address + offset + bank_to_dram_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<false>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t get_system_memory_noc_addr(const uint32_t id, const uint32_t page_size, const uint32_t base_addr, const uint32_t offset = 0, uint8_t noc = noc_index) {
    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_PCIE_ENCODING(DYNAMIC_NOC_X(noc, PCIE_NOC_X), DYNAMIC_NOC_Y(noc, PCIE_NOC_Y), noc));
    uint32_t addr = base_addr + page_size * id + offset;
    uint64_t noc_addr = pcie_core_noc_encoding | addr;
    return noc_addr;
}

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t addr, uint8_t noc = noc_index) {
    /*
        Get an encoding which contains the address in L1 on the current core that you want to
        read from/write to via the noc
    */
    return NOC_XY_ADDR(my_x[noc], my_y[noc], addr);
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
 * | src_noc_addr      | Encoding of the source NOC location (x,y)+address  | uint64_t  | DOX-TODO(ref to explain valid coords)    | Yes      |
 * | dst_local_l1_addr | Address in local L1 memory                         | uint32_t  | 0..1MB                                   | Yes      |
 * | size              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                                   | Yes      |
 */
inline
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    WAYPOINT("NARW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
    ncrisc_noc_fast_read_any_len(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size);
    WAYPOINT("NARD");
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    WAYPOINT("RPW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RPD");

    WAYPOINT("NARW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_noc_addr);
#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;

    WAYPOINT("NARD");
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet_set_state(std::uint64_t src_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    WAYPOINT("RPW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RPD");

    WAYPOINT("NARW");

#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);

    WAYPOINT("NARD");
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool inc_num_issued = true>
FORCE_INLINE
void noc_async_read_one_packet_with_state(std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    WAYPOINT("RPW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RPD");

    WAYPOINT("NARW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, src_noc_addr, dst_local_l1_addr);

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (inc_num_issued) {
        noc_reads_num_issued[noc] += 1;
    }

    WAYPOINT("NARD");
}

// TODO: write docs
FORCE_INLINE
void noc_async_read_set_state(std::uint64_t src_noc_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    WAYPOINT("NARW");
    WAYPOINT("RPW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RPD");

#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    WAYPOINT("NARD");
}

// TODO: write docs
template <bool inc_num_issued = true>
FORCE_INLINE
void noc_async_read_with_state(std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    WAYPOINT("NARW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc, src_noc_addr, dst_local_l1_addr, size);

    while (size > NOC_MAX_BURST_SIZE) {
        WAYPOINT("RPW");
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("RPD");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        size -= NOC_MAX_BURST_SIZE;
        src_noc_addr += NOC_MAX_BURST_SIZE;
        dst_local_l1_addr += NOC_MAX_BURST_SIZE;
        if constexpr (inc_num_issued) {
            noc_reads_num_issued[noc] += 1;
        }
    }

    // left-over packet
    WAYPOINT("RPW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RPD");

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (inc_num_issued) {
        noc_reads_num_issued[noc] += 1;
    }

    WAYPOINT("NARD");
}

FORCE_INLINE
void noc_async_read_inc_num_issued(std::uint32_t num_issued_reads_inc, uint8_t noc = noc_index) {
    noc_reads_num_issued[noc] += num_issued_reads_inc;
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_write_one_packet(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {

    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE,  size);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;  // num_dests
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_write_multicast_one_packet(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index) {
    WAYPOINT("NMPW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field =
                            NOC_CMD_CPY | NOC_CMD_WR |
                            NOC_CMD_VC_STATIC |
                            NOC_CMD_STATIC_VC(NOC_MULTICAST_WRITE_VC) |
                            (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) |
                            NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr_multicast);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr_multicast >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr_multicast >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE,  size);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += num_dests;
}

// TODO: write docs
// this sets the state for issuing a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE
void noc_async_write_one_packet_set_state(std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {

    WAYPOINT("NWPW");
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                NOC_CMD_STATIC_VC(vc) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                (non_posted ? NOC_CMD_RESP_MARKED : 0x0);

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE,  size);
}

// TODO: write docs
// this issues only a single packet with cmd buf state with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE
void noc_async_write_one_packet_with_state(std::uint32_t src_local_l1_addr, std::uint32_t dst_noc_addr, uint8_t noc = noc_index) {

    WAYPOINT("NWPW");
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, dst_noc_addr, src_local_l1_addr);

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (non_posted) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;  // num_dests
    }
}

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    const uint32_t page_size;    // Num bytes in page.
    const uint32_t aligned_page_size = align(page_size, ALLOCATOR_ALIGNMENT);

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return (bank_offset_index * this->aligned_page_size) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(this->get_noc_addr(id, offset), dest_addr, page_size, noc);
    }
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;  // WARNING: This struct is used for optimized get_noc_addr in which case
                                             // you know that bank_unit_size is a power of 2
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT
                                                   ? this->log_base_2_of_page_size
                                                   : LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT;

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    // TODO: Remove page_size from argument list. This can be derived from data_format
    uint32_t page_size;          // Num bytes in bank unit.
    DataFormat data_format;      // Data format

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return MUL_WITH_TILE_SIZE<tile_hw>((uint)this->data_format, bank_offset_index) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_tile(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NRTW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("NRTD");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NWTW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        WAYPOINT("NWTD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;  // num_dests
    }
};

// TODO: add noc_async_write_page
// TODO: need static assert + host assert that page size <= 8192, hard constraint
template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;             // Base address for the whole tensor.
    const uint32_t log_base_2_of_page_size; // Num bytes in bank unit.
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT
                                                   ? this->log_base_2_of_page_size
                                                   : LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT;

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {

        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NRPW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, 1 << this->aligned_log_base_2_of_page_size);
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("NRPD");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, 1 << this->aligned_log_base_2_of_page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_read_partial_page(const uint32_t id, uint32_t dest_addr, const uint32_t size, const uint32_t offset, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("RPW");
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("RPD");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, size);

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_page(const uint32_t id, uint32_t src_addr, const uint32_t write_size_bytes, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NWPW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, write_size_bytes);
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        WAYPOINT("NWPD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE,  write_size_bytes);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;  // num_dests
    }
};

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, std::uint32_t dst_local_l1_addr, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_page(id, dst_local_l1_addr, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, std::uint32_t dst_local_l1_addr, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_tile(id, dst_local_l1_addr, offset, noc);
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
 * | Argument          | Description                                             | Type     | Valid Range                                                    | Required |
 * |-------------------|---------------------------------------------------------|----------|----------------------------------------------------------------|----------|
 * | src_local_l1_addr | Source address in local L1 memory                       | uint32_t | 0..1MB                                                         | True     |
 * | dst_noc_addr      | Encoding of the destination NOC location (x,y)+address  | uint64_t | DOX-TODO(insert a reference  to what constitutes valid coords) | True     |
 * | size              | Size of data transfer in bytes                          | uint32_t | 0..1MB                                                         | True     |
 */
template<uint32_t max_page_size=NOC_MAX_BURST_SIZE + 1>
inline
void noc_async_write(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_one_packet(src_local_l1_addr, dst_noc_addr, size);
    } else {
        WAYPOINT("NAWW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr,size);
        ncrisc_noc_fast_write_any_len(
            noc,
            write_cmd_buf,
            src_local_l1_addr,
            dst_noc_addr,
            size,
            NOC_UNICAST_WRITE_VC,
            false,
            false,
            1,
            true);
        WAYPOINT("NAWD");
    }
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, std::uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    s.noc_async_write_tile(id, src_local_l1_addr, noc);
}

template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
FORCE_INLINE
uint32_t get_semaphore(uint32_t semaphore_id) {
    return (uint32_t)sem_l1_base[static_cast<int>(type)] + semaphore_id * L1_ALIGNMENT;
}

inline
void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, uint8_t noc = noc_index) {
    WAYPOINT("NSSW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len(
        noc,
        write_reg_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr,
        4 /* size in bytes */,
        NOC_UNICAST_WRITE_VC,
        false,
        false,
        1,
        true);
    WAYPOINT("NSSD");
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
 * Note: The number of destinations needs to be non-zero. Besides that,
 * there is no restriction on the number of destinations, i.e. the
 * multicast destinations can span the full chip. However, as mentioned
 * previously, the multicast source cannot be part of the destinations. So, the
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
template<uint32_t max_page_size=NOC_MAX_BURST_SIZE + 1>
inline
void noc_async_write_multicast(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index) {
    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast_one_packet(src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, linked, multicast_path_reserve);
    } else {
        WAYPOINT("NMWW");
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr,size);
        ncrisc_noc_fast_write_any_len(
            noc,
            write_cmd_buf,
            src_local_l1_addr,
            dst_noc_addr_multicast,
            size,
            NOC_MULTICAST_WRITE_VC,
            true,
            linked,
            num_dests,
            multicast_path_reserve);
        WAYPOINT("NMWD");
    }
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
 * With this API, the multicast sender cannot be part of the multicast
 * destinations. If the multicast sender has to be in the multicast
 * destinations (i.e. must perform a local L1 write), the other API variant
 * *noc_semaphore_set_multicast_loopback_src* can be used.
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
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests, bool linked = false, bool multicast_path_reserve = true, uint8_t noc = noc_index) {
    WAYPOINT("NSMW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len(
        noc,
        write_reg_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        multicast_path_reserve);
    WAYPOINT("NSMD");
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
 * Note: With this API, sending data only to the source node (when num_dests
 * is 1) may result in unexpected behaviour. For some parameters, hangs have
 * been observed. For some other parameters, nothing may happen. Consider using
 * regular non multicast operations such as *noc_async_write* in this case.
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
void noc_semaphore_set_multicast_loopback_src(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests, bool linked = false, bool multicast_path_reserve = true, uint8_t noc = noc_index) {
    WAYPOINT("NSMW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len_loopback_src(
        noc,
        write_reg_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        multicast_path_reserve);
    WAYPOINT("NSMD");
}

inline
void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index) {
    WAYPOINT("NMLW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len_loopback_src(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        multicast_path_reserve);
    WAYPOINT("NMLD");
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_read*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_read* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
void noc_async_read_barrier(uint8_t noc = noc_index) {
    WAYPOINT("NRBW");
    // BH cache is write-through so reader must invalidate if reading any address that was previously read
    do {
        invalidate_l1_cache();
    } while (!ncrisc_noc_reads_flushed(noc));
    WAYPOINT("NRBD");
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
void noc_async_write_barrier(uint8_t noc = noc_index) {
    WAYPOINT("NWBW");
    while (!ncrisc_noc_nonposted_writes_flushed(noc))
        ;
    WAYPOINT("NWBD");
}

/**
 * This blocking call waits for all outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to depart, but will not wait
 * for them to complete
*/
FORCE_INLINE
void noc_async_writes_flushed(uint8_t noc = noc_index) {
    WAYPOINT("NWFW");
    while (!ncrisc_noc_nonposted_writes_sent(noc))
        ;
    WAYPOINT("NWFD");
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
void noc_async_atomic_barrier(uint8_t noc_idx = noc_index) {
    WAYPOINT("NABW");
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_idx))
        ;
    WAYPOINT("NABD");
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
    WAYPOINT("NSW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) != val);
    WAYPOINT("NSD");
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
    WAYPOINT("NSMW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
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
 * Initiates an asynchronous write of a 32-bit value to a NOC destination.
 * Typically used for writing registers, but can be used for memory locations as well.
 * The destination is specified as a 64-bit NOC address (see \a noc_async_write).
 * The advantage over using \a noc_async_write is that we don't a Tensix L1
 * memory source location; the write value is written directly into a register.
 * Unlike using \a noc_async_write, there are also no address alignment concerns.
 * Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a DRAM bank, Tensix core+L1 memory
 * address or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range                                                   | Required |
 * |-----------|----------------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | addr      | Encoding of the destination location (x,y)+address             | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | val       | The value to be written                                        | uint32_t | Any uint32_t value                                            | True     |
 * | be        | Byte-enable                                                    | uint8_t  | 0x1-0xF                                                       | False    |
 */
FORCE_INLINE
void noc_inline_dw_write(uint64_t addr, uint32_t val, uint8_t be = 0xF, uint8_t noc = noc_index) {

    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_ADDR(noc, addr, 4);
    noc_fast_write_dw_inline(
                noc,
                write_reg_cmd_buf,
                val,
                addr,
                be, // byte-enable
                NOC_UNICAST_WRITE_VC,
                false, // mcast
                false // posted
            );
    WAYPOINT("NWID");
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
FORCE_INLINE
void noc_semaphore_inc(uint64_t addr, uint32_t incr, uint8_t noc_id = noc_index) {
    /*
    [REFER TO grayskull/noc/noc.h for the documentation of noc_atomic_increment()]
    Generic increment with 32-bit wrap.
  */
    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment<noc_mode>(noc_id, write_at_cmd_buf, addr, NOC_UNICAST_WRITE_VC, incr, 31 /*wrap*/, false /*linked*/, false /*posted*/, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    WAYPOINT("NSID");
}

inline void RISC_POST_HEARTBEAT(uint32_t &heartbeat) {
  invalidate_l1_cache();
  volatile uint32_t* ptr = (volatile uint32_t*)(0x1C);
  heartbeat++;
  ptr[0] = 0xAABB0000 | (heartbeat & 0xFFFF);
}

FORCE_INLINE
uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a: b; }

template <bool use_vc>
FORCE_INLINE
uint32_t noc_async_read_tile_dram_sharded_set_state(uint32_t bank_base_address, uint32_t page_size, uint32_t bank_id = 0, const uint32_t vc = 0, uint8_t noc = noc_index) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = bank_base_address + bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc][bank_id];

    WAYPOINT("NRTW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRTD");

    if constexpr(use_vc) {
        uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, page_size);  // len_bytes

    return src_addr_;
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_with_state(uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index) {
    uint32_t src_addr_;

    src_addr_ = src_base_addr + src_addr;

    WAYPOINT("NRTW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRTD");

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr_);      // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_with_state_with_trid(uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index) {
    WAYPOINT("NRDW");
    #ifndef ARCH_GRAYSKULL
    ncrisc_noc_fast_read_with_transaction_id(noc, read_cmd_buf, src_base_addr, src_addr, dest_addr, trid);
    #endif
    WAYPOINT("NRDD");
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_set_trid(uint32_t trid = 0, uint8_t noc = noc_index) {
    WAYPOINT("NSTW");
    #ifndef ARCH_GRAYSKULL
    ncrisc_noc_set_transaction_id(noc, read_cmd_buf, trid);
    #endif
    WAYPOINT("NSTD");
}

FORCE_INLINE
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NBTW");
    #ifndef ARCH_GRAYSKULL
    do {
        invalidate_l1_cache();
    } while (!ncrisc_noc_read_with_transaction_id_flushed(noc, trid));
    #endif
    WAYPOINT("NBTD");
}
