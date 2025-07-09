// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#define DATA_FORMATS_DEFINED
#endif
#include <noc/noc_parameters.h>

#include <algorithm>
#include <stdint.h>

#include "core_config.h"
#include "circular_buffer.h"
#include "dataflow_cmd_bufs.h"
#include "debug/sanitize_noc.h"
#include "debug/waypoint.h"
#include "eth_l1_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "risc_attribs.h"
#include "utils/utils.h"
#include "debug/assert.h"
#include "compile_time_args.h"
#include "dev_msgs.h"
#include "dataflow_api_common.h"
#include "dataflow_api_addrgen.h"
#include "accessor/tensor_accessor.h"
#include "tools/profiler/kernel_profiler.hpp"

// clang-format off
/**
 * Returns the absolute logical X coordinate value that this kernel is running on. The absolute coordinate
 * is the one relative to the origin of the physical grid.
 *
 * Return value: X coordinate value.
 */
// clang-format on
inline uint8_t get_absolute_logical_x() {
    extern uint8_t my_logical_x_;  // Set in FW
    return my_logical_x_;
}

// clang-format off
/**
 * Returns the absolute logical Y coordinate value that this kernel is running on. The absolute coordinate
 * is the one relative to the origin of the physical grid.
 *
 * Return value: Y coordinate value.
 */
// clang-format on
inline uint8_t get_absolute_logical_y() {
    extern uint8_t my_logical_y_;  // Set in FW
    return my_logical_y_;
}

// clang-format off
/**
 * Returns the relative logical X coordinate value that this kernel is running on. The relative coordinate
 * is with respect to the origin of the sub device for this core type.
 *
 * Return value: X coordinate value.
 */
// clang-format on
inline uint8_t get_relative_logical_x() {
    extern uint8_t my_relative_x_;  // Set in FW
    return my_relative_x_;
}

// clang-format off
/**
 * Returns the relative logical Y coordinate value that this kernel is running on. The relative coordinate
 * is with respect to the origin of the sub device for this core type.
 *
 * Return value: Y coordinate value.
 */
// clang-format on
inline uint8_t get_relative_logical_y() {
    extern uint8_t my_relative_y_;  // Set in FW
    return my_relative_y_;
}

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for unique (per core) runtime arguments set via
 * SetRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given unique runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Unique Runtime argument index                                           | uint32_t | 0 to 255    | True     |
 */
// clang-format on
static FORCE_INLINE uint32_t get_arg_addr(int arg_idx) { return (uint32_t)&rta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for common (all cores) runtime arguments set via
 * SetCommonRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given common runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Common Runtime argument index                                           | uint32_t | 0 to 255    | True     |
 */
// clang-format on
static FORCE_INLINE uint32_t get_common_arg_addr(int arg_idx) { return (uint32_t)&crta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs()
 * API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 255    | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A         | True     |
 */
// clang-format on
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

// clang-format off
/**
 * Returns the value at a given runtime argument index for common (all cores) runtime arguments set via
 * SetCommonRuntimeArgs() API.
 *
 * Return value: The value associated with the common runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | Common Runtime argument index                  | uint32_t              | 0 to 255    | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A         | True     |
 */
// clang-format on
template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_common_arg_addr(arg_idx)));
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
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to be pushed      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
FORCE_INLINE
void cb_push_back(const int32_t operand, const int32_t num_pages) {
    uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;

    volatile tt_reg_ptr uint32_t* pages_received_ptr = get_cb_tiles_received_ptr(operand);
    pages_received_ptr[0] += num_pages;

    get_local_cb_interface(operand).fifo_wr_ptr += num_words;

    // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
    // producer always writes into contiguous memory, it cannot wrap
    ASSERT(get_local_cb_interface(operand).fifo_wr_ptr <= get_local_cb_interface(operand).fifo_limit);
    if (get_local_cb_interface(operand).fifo_wr_ptr == get_local_cb_interface(operand).fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        get_local_cb_interface(operand).fifo_wr_ptr -= get_local_cb_interface(operand).fifo_size;
    }
}

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
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to be popped      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
FORCE_INLINE
void cb_pop_front(int32_t operand, int32_t num_pages) {
    volatile tt_reg_ptr uint32_t* pages_acked_ptr = get_cb_tiles_acked_ptr(operand);
    pages_acked_ptr[0] += num_pages;

    uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(operand).fifo_rd_ptr += num_words;

    // this will basically reset fifo_rd_ptr to fifo_addr -- no other wrap is legal
    // consumer always reads from contiguous memory, it cannot wrap
    ASSERT(get_local_cb_interface(operand).fifo_rd_ptr <= get_local_cb_interface(operand).fifo_limit);
    if (get_local_cb_interface(operand).fifo_rd_ptr == get_local_cb_interface(operand).fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        get_local_cb_interface(operand).fifo_rd_ptr -= get_local_cb_interface(operand).fifo_size;
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

// clang-format off
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
 * |-----------|---------------------------------------|----------|-------------|----------|
 * | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
FORCE_INLINE
uint32_t get_write_ptr(uint32_t operand) {
    // return byte address (fifo_wr_ptr is 16B address)
    uint32_t wr_ptr_bytes = get_local_cb_interface(operand).fifo_wr_ptr;
    return wr_ptr_bytes;
}

// clang-format off
/**
 * Returns a pointer to the beginning of a memory block previously received
 * by cb_wait_front. Note that this call is only valid between calls
 * to cb_wait_front and cb_pop_front. The amount of valid memory
 * is equal to the number of tiles requested in a prior cb_wait_front call.
 *
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range | Required |
 * |-----------|---------------------------------------|----------|-------------|----------|
 * | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
FORCE_INLINE
uint32_t get_read_ptr(uint32_t operand) {
    // return byte address (fifo_rd_ptr is 16B address)
    uint32_t rd_ptr_bytes = get_local_cb_interface(operand).fifo_rd_ptr;
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

// clang-format off
/**
 * A non-blocking call that checks if the specified number of pages are available for reservation at the back of the
 * circular buffer. This call is used by the producer to see if the consumer has freed up the desired space (in pages).
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: true if the specified number of pages are available
 *
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of free tiles to wait for  | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
FORCE_INLINE
bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked_ptr = (uint32_t)get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
    uint16_t free_space_pages_wrap = get_local_cb_interface(operand).fifo_num_pages - (pages_received - pages_acked);
    return num_pages <= static_cast<int32_t>(free_space_pages_wrap);
}

// clang-format off
/**
 * A blocking call that waits for the specified number of tiles to be free in the specified circular buffer. This call
 * is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of free tiles to wait for  | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
// clang-format on
FORCE_INLINE
void cb_reserve_back(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked_ptr = (uint32_t)get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    WAYPOINT("CRBW");
    do {
        // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
        // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
        invalidate_l1_cache();
        uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
        uint16_t free_space_pages_wrap =
            get_local_cb_interface(operand).fifo_num_pages - (pages_received - pages_acked);
        free_space_pages = (int32_t)free_space_pages_wrap;
    } while (free_space_pages < num_pages);
    WAYPOINT("CRBD");
}

// clang-format off
/**
 * A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
 * buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
 * specified number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
 * cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
 * 4 calls of cb_wait_front(8) followed by a cb_pop_front(32) would produce incorrect behavior. Instead 4 calls of
 * cb_wait_front() waiting on 8, 16, 24, 32 tiles should be issued.
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
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to check for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 */
// clang-format on
FORCE_INLINE
bool cb_pages_available_at_front(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked = get_cb_tiles_acked_ptr(operand)[0];
    uint32_t pages_received_ptr = (uint32_t)get_cb_tiles_received_ptr(operand);

    uint16_t pages_received = ((uint16_t)reg_read(pages_received_ptr)) - pages_acked;
    return num_pages <= pages_received;
}

// clang-format off
/**
 * A blocking call that waits for the specified number of tiles to be available in the specified circular buffer (CB).
 * This call is used by the consumer of the CB to wait for the producer to fill the CB with at least the specified
 * number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
 * cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
 * 4 calls of cb_wait_front(8) followed by a cb_pop_front(32) would produce incorrect behavior. Instead 4 calls of
 * cb_wait_front() waiting on 8, 16, 24, 32 tiles should be issued.
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
 * | Argument  | Description                           | Type     | Valid Range                                                                                       | Required |
 * |-----------|---------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the circular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to wait for       | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 */
// clang-format on
FORCE_INLINE
void cb_wait_front(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked = get_cb_tiles_acked_ptr(operand)[0];
    uint32_t pages_received_ptr = (uint32_t)get_cb_tiles_received_ptr(operand);

    uint16_t pages_received;

    WAYPOINT("CWFW");
    do {
        pages_received = ((uint16_t)reg_read(pages_received_ptr)) - pages_acked;
    } while (pages_received < num_pages);
    WAYPOINT("CWFD");
}

// #######################################################################################
// #################################### NOC transfers ####################################
// #######################################################################################

// clang-format off
/**
 * Initiates an asynchronous read for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_read for more details.
 */
// clang-format on
FORCE_INLINE
void noc_async_read_one_packet(
    std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    WAYPOINT("RP2W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP2D");

    WAYPOINT("NAOW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
    ncrisc_noc_fast_read<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size);
    WAYPOINT("NAOD");
}

// clang-format off
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
 * | Argument                          | Description                                        | Data type | Valid range                      | required |
 * |-----------------------------------|----------------------------------------------------|-----------|----------------------------------|----------|
 * | src_noc_addr                      | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | dst_local_l1_addr                 | Address in local L1 memory                         | uint32_t  | 0..1MB                           | True     |
 * | size                              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                           | True     |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1                           | False    |
 * | max_page_size (template argument) | Maximum size of a single transaction in bytes      | uint32_t  | Any uint32_t number              | False    |
 */
// clang-format on
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ,src_noc_addr,size, -1);

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet(src_noc_addr, dst_local_l1_addr, size, noc);
    } else {
        WAYPOINT("NARW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
        ncrisc_noc_fast_read_any_len<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size);
        WAYPOINT("NARD");
    }
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet_set_state(std::uint64_t src_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_SET_STATE, src_noc_addr, size, -1);

    WAYPOINT("RP3W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP3D");

    WAYPOINT("NASW");

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc,
        read_cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);

    WAYPOINT("NASD");
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool inc_num_issued = true>
FORCE_INLINE void noc_async_read_one_packet_with_state(
    std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_WITH_STATE, static_cast<uint64_t>(src_noc_addr), 0, -1);
    if constexpr (inc_num_issued) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
        }
    }

    WAYPOINT("RP4W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP4D");

    WAYPOINT("NATW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, src_noc_addr, dst_local_l1_addr);

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (inc_num_issued) {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_reads_num_issued[noc] += 1;
        }
    }

    WAYPOINT("NATD");
}

// TODO: write docs
FORCE_INLINE
void noc_async_read_set_state(std::uint64_t src_noc_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_SET_STATE,src_noc_addr,0,-1);

    WAYPOINT("RP5W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP5D");

    WAYPOINT("NAUW");

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc,
        read_cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    WAYPOINT("NAUD");
}

// TODO: write docs
template <bool inc_num_issued = true>
FORCE_INLINE void noc_async_read_with_state(
    std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_WITH_STATE,src_noc_addr,size,-1);

    WAYPOINT("NAVW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc, src_noc_addr, dst_local_l1_addr, size);

    while (size > NOC_MAX_BURST_SIZE) {
        if constexpr (inc_num_issued) {
            if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
            }
        }
        WAYPOINT("RP6W");
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("RP6D");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        size -= NOC_MAX_BURST_SIZE;
        src_noc_addr += NOC_MAX_BURST_SIZE;
        dst_local_l1_addr += NOC_MAX_BURST_SIZE;
        if constexpr (inc_num_issued) {
            if constexpr (noc_mode == DM_DEDICATED_NOC) {
                noc_reads_num_issued[noc] += 1;
            }
        }
    }

    if constexpr (inc_num_issued) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
        }
    }
    // left-over packet
    WAYPOINT("RP7W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP7D");

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (inc_num_issued) {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_reads_num_issued[noc] += 1;
        }
    }

    WAYPOINT("NAVD");
}

FORCE_INLINE
void noc_async_read_inc_num_issued(std::uint32_t num_issued_reads_inc, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, num_issued_reads_inc);
    } else {
        noc_reads_num_issued[noc] += num_issued_reads_inc;
    }
}

// clang-format off
/**
 * Initiates an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_write for more details.
 */
// clang-format on
FORCE_INLINE
void noc_async_write_one_packet(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index) {
    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    ncrisc_noc_fast_write<noc_mode>(
        noc, write_cmd_buf, src_local_l1_addr, dst_noc_addr, size, NOC_UNICAST_WRITE_VC, false, false, 1, true);
}

// clang-format off
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
 * | Argument                          | Description                                             | Type     | Valid Range                      | Required |
 * |-----------------------------------|---------------------------------------------------------|----------|----------------------------------|----------|
 * | src_local_l1_addr                 | Source address in local L1 memory                       | uint32_t | 0..1MB                           | True     |
 * | dst_noc_addr                      | Encoding of the destination NOC location (x,y)+address  | uint64_t | Results of \a get_noc_addr calls | True     |
 * | size                              | Size of data transfer in bytes                          | uint32_t | 0..1MB                           | True     |
 * | noc                               | Which NOC to use for the transaction                    | uint8_t  | 0 or 1                           | False    |
 * | max_page_size (template argument) | Maximum size of a single transaction in bytes           | uint32_t | Any uint32_t number              | False    |
 */
// clang-format on
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_, dst_noc_addr, size, NOC_UNICAST_WRITE_VC);

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_one_packet(src_local_l1_addr, dst_noc_addr, size, noc);
    } else {
        WAYPOINT("NAWW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc, write_cmd_buf, src_local_l1_addr, dst_noc_addr, size, NOC_UNICAST_WRITE_VC, false, false, 1, true);
        WAYPOINT("NAWD");
    }
}

// clang-format off
/**
 * Initiates an asynchronous multicast write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_write_multicast for more details.
 */
// clang-format on
FORCE_INLINE
void noc_async_write_multicast_one_packet(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_MULTICAST, dst_noc_addr_multicast, size, NOC_MULTICAST_WRITE_VC);
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    ncrisc_noc_fast_write<noc_mode>(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true /* mcast */,
        linked,
        num_dests,
        true /* multicast_path_reserve */);
}

// clang-format off
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
 * maximum number of destinations is number of cores - 1.
 *
 * Return value: None
 *
 * | Argument                          | Description                                                              | Type     | Valid Range                                | Required |
 * |-----------------------------------|--------------------------------------------------------------------------|----------|--------------------------------------------|----------|
 * | src_local_l1_addr                 | Source address in local L1 memory                                        | uint32_t | 0..1MB                                     | True     |
 * | dst_noc_addr_multicast            | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | Results of \a get_noc_multicast_addr calls | True     |
 * | size                              | Size of data transfer in bytes                                           | uint32_t | 0..1MB                                     | True     |
 * | num_dests                         | Number of destinations that the multicast source is targetting           | uint32_t | 0..(number of cores -1)                    | True     |
 * | linked                            | Whether the transaction is linked                                        | bool     | true or false                              | False    |
 * | noc                               | Which NOC to use for the transaction                                     | uint8_t  | 0 or 1                                     | False    |
 * | max_page_size (template argument) | Maximum size of a single transaction in bytes                            | uint32_t | Any uint32_t number                        | False    |
 */
// clang-format on
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_MULTICAST, dst_noc_addr_multicast, size, NOC_MULTICAST_WRITE_VC);

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast_one_packet(src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, linked);
    } else {
        WAYPOINT("NMWW");
        NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc,
            write_cmd_buf,
            src_local_l1_addr,
            dst_noc_addr_multicast,
            size,
            NOC_MULTICAST_WRITE_VC,
            true /* mcast */,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NMWD");
    }
}

// TODO: write docs
// this sets the state for issuing a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE void noc_async_write_one_packet_set_state(
    std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_SET_STATE, dst_noc_addr, size, vc);

    WAYPOINT("NWPW");
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                             0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                             (non_posted ? NOC_CMD_RESP_MARKED : 0x0);

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_cmd_buf,
        NOC_RET_ADDR_COORDINATE,
        (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, size);
}

// TODO: write docs
// this issues only a single packet with cmd buf state with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
template <bool non_posted = true>
FORCE_INLINE void noc_async_write_one_packet_with_state(
    std::uint32_t src_local_l1_addr, std::uint32_t dst_noc_addr, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_STATE, 0ull, 0, -1);

    if constexpr (non_posted) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    } else {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        }
    }
    WAYPOINT("NWPW");
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, dst_noc_addr, src_local_l1_addr);

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (non_posted) {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;  // num_dests
        }
    } else {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_posted_writes_num_issued[noc] += 1;
        }
    }
}

template <typename DSpec>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const TensorAccessor<DSpec>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    noc_async_read(s.get_noc_addr(id, offset, noc), dst_local_l1_addr, s.page_size, noc);
}

template <typename DSpec>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const TensorAccessor<DSpec>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    noc_async_read(s.get_noc_addr(id, offset, noc), dst_local_l1_addr, s.page_size, noc);
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    noc_async_read(s.get_noc_addr(id, offset, noc), dst_local_l1_addr, s.page_size, noc);
}

template <typename DSpec>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const TensorAccessor<DSpec>& s,
    std::uint32_t src_local_l1_addr,
    const uint32_t write_size_bytes,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    noc_async_write(src_local_l1_addr, s.get_noc_addr(id, offset, noc), write_size_bytes, noc);
}

template <bool DRAM>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& s,
    std::uint32_t src_local_l1_addr,
    const uint32_t write_size_bytes,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    noc_async_write(src_local_l1_addr, s.get_noc_addr(id, offset, noc), write_size_bytes, noc);
}

template <typename DSpec>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const TensorAccessor<DSpec>& s, std::uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    noc_async_write(src_local_l1_addr, s.get_noc_addr(id, 0, noc), s.page_size, noc);
}

template <bool DRAM>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, std::uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    noc_async_write(src_local_l1_addr, s.get_noc_addr(id, 0, noc), s.page_size, noc);
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, id, s.page_size, -1);

    noc_async_read(s.get_noc_addr(id, offset), dst_local_l1_addr, s.page_size, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, id, s.page_size, -1);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    uint32_t src_addr = s.get_addr(id, bank_offset_index, bank_index, offset);
    uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

    WAYPOINT("NRTW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dst_local_l1_addr, s.page_size);
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRTD");

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, s.page_size);            // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t src_local_l1_addr,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::WRITE_, id, s.page_size, NOC_UNICAST_WRITE_VC);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
    }
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    uint32_t dest_addr = s.get_addr(id, bank_offset_index, bank_index);
    uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

    WAYPOINT("NWTW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(
        noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_local_l1_addr, s.page_size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWTD");

    constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                       NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
                                       0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                       0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                       NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);            // (uint32_t)dest_addr
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);  // dest_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, s.page_size);            // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;  // num_dests
    }
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const InterleavedPow2AddrGenFast<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    uint32_t src_addr = s.get_addr(id, bank_offset_index, bank_index, offset);
    uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

    WAYPOINT("NRPW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(
        noc, get_noc_addr_helper(src_noc_xy, src_addr), dst_local_l1_addr, 1 << s.aligned_log_base_2_of_page_size);
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRPD");

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, 1 << s.aligned_log_base_2_of_page_size);  // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

template <bool DRAM>
FORCE_INLINE void noc_async_read_partial_page(
    const uint32_t id,
    const InterleavedPow2AddrGenFast<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    const uint32_t size,
    const uint32_t offset,
    uint8_t noc = noc_index) {
    // Note: This is not used anywhere in tt-metal
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    uint32_t src_addr = s.get_addr(id, bank_offset_index, bank_index, offset);
    uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

    WAYPOINT("RP1W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP1D");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dst_local_l1_addr, size);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);                   // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

template <bool DRAM>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const InterleavedPow2AddrGenFast<DRAM>& s,
    std::uint32_t src_local_l1_addr,
    const uint32_t write_size_bytes,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    // Note: This is not used anywhere in tt-metal
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
    }
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    uint32_t dest_addr = s.get_addr(id, bank_offset_index, bank_index, offset);
    uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(
        noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_local_l1_addr, write_size_bytes);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                       NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
                                       0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                       0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                       NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);            // (uint32_t)dest_addr
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);  // dest_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, write_size_bytes);       // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;  // num_dests
    }
}

template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
FORCE_INLINE uint32_t get_semaphore(uint32_t semaphore_id) {
    return (uint32_t)sem_l1_base[static_cast<int>(type)] + semaphore_id * L1_ALIGNMENT;
}

inline void noc_semaphore_set_remote(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, uint8_t noc = noc_index) {
    WAYPOINT("NSSW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len<noc_mode>(
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

// clang-format off
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
 * | Argument               | Description                                                              | Type     | Valid Range                                | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|--------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t | 0..1MB                                     | True     |
 * | dst_noc_addr_multicast | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | Results of \a get_noc_multicast_addr calls | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t | 0..(number of cores - 1)                   | True     |
 * | linked                 | Whether the transaction is linked                                        | bool     | true or false                              | False    |
 * | noc                    | Which NOC to use for the transaction                                     | uint8_t  | 0 or 1                                     | False    |
 */
// clang-format on
inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    WAYPOINT("NSNW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len<noc_mode>(
        noc,
        write_reg_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        true /* multicast_path_reserve */);
    WAYPOINT("NSND");
}
// clang-format off
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
 * | Argument               | Description                                                              | Type     | Valid Range                                | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|--------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t | 0..1MB                                     | True     |
 * | dst_noc_addr_multicast | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | Results of \a get_noc_multicast_addr calls | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t | 0..(number of cores)                       | True     |
 * | linked                 | Whether the transaction is linked                                        | bool     | true or false                              | False    |
 * | noc                    | Which NOC to use for the transaction                                     | uint8_t  | 0 or 1                                     | False    |
 */
// clang-format on
inline void noc_semaphore_set_multicast_loopback_src(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    WAYPOINT("NSLW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, 4);
    ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
        noc,
        write_reg_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        true /* multicast_path_reserve */);
    WAYPOINT("NSLD");
}

inline void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    constexpr bool multicast_path_reserve = true;

    NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_MULTICAST, dst_noc_addr_multicast, size, NOC_MULTICAST_WRITE_VC);

    WAYPOINT("NMLW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
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
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | noc      | Which NOC to use for the transaction | uint8_t  | 0 or 1      | False    |
 */
void noc_async_read_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_START);

    WAYPOINT("NRBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        while (!ncrisc_dynamic_noc_reads_flushed(noc)) {
            invalidate_l1_cache();
        }
    } else {
        while (!ncrisc_noc_reads_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NRBD");

    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_END);
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_write* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | noc      | Which NOC to use for the transaction | uint8_t  | 0 or 1      | False    |
 */
FORCE_INLINE
void noc_async_write_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_START);

    WAYPOINT("NWBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc)) {
            invalidate_l1_cache();
        }
    } else {
        while (!ncrisc_noc_nonposted_writes_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NWBD");

    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_END);
}

/**
 * This blocking call waits for all outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to depart, but will not wait
 * for them to complete
 */
FORCE_INLINE
void noc_async_writes_flushed(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_FLUSH);

    WAYPOINT("NWFW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        while (!ncrisc_dynamic_noc_nonposted_writes_sent(noc)) {
            invalidate_l1_cache();
        }
    } else {
        while (!ncrisc_noc_nonposted_writes_sent(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NWFD");
}

/**
 * This blocking call waits for all outstanding enqueued posted *noc_async_write*
 * calls issued on the current Tensix core to depart, but will not wait
 * for them to complete
 */
FORCE_INLINE
void noc_async_posted_writes_flushed(uint8_t noc = noc_index) {
    WAYPOINT("NPWW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        while (!ncrisc_dynamic_noc_posted_writes_sent(noc)) {
            invalidate_l1_cache();
        }
    } else {
        while (!ncrisc_noc_posted_writes_sent(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NPWD");
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
    RECORD_NOC_EVENT(NocEventType::ATOMIC_BARRIER);

    WAYPOINT("NABW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_idx)) {
            invalidate_l1_cache();
        }
    } else {
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_idx));
    }
    invalidate_l1_cache();
    WAYPOINT("NABD");
}

/**
 * This blocking call waits for all the outstanding read, write, and atomic NOC
 * transactions issued on the current Tensix core to complete. After returning
 * from this call all transaction queues will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_full_barrier(uint8_t noc_idx = noc_index) {
    invalidate_l1_cache();
    RECORD_NOC_EVENT(NocEventType::FULL_BARRIER);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        WAYPOINT("NFBW");
        while (!ncrisc_dynamic_noc_reads_flushed(noc_idx));
        WAYPOINT("NFCW");
        while (!ncrisc_dynamic_noc_nonposted_writes_sent(noc_idx));
        WAYPOINT("NFDW");
        while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc_idx));
        WAYPOINT("NFEW");
        while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_idx));
        WAYPOINT("NFFW");
        while (!ncrisc_dynamic_noc_posted_writes_sent(noc_idx));
        WAYPOINT("NFBD");
    } else {
        WAYPOINT("NFBW");
        while (!ncrisc_noc_reads_flushed(noc_idx));
        WAYPOINT("NFCW");
        while (!ncrisc_noc_nonposted_writes_sent(noc_idx));
        WAYPOINT("NFDW");
        while (!ncrisc_noc_nonposted_writes_flushed(noc_idx));
        WAYPOINT("NFEW");
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_idx));
        WAYPOINT("NFFW");
        while (!ncrisc_noc_posted_writes_sent(noc_idx));
        WAYPOINT("NFBD");
    }
}

// clang-format off
/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                            | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory   | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore      | uint32_t | Any uint32_t value | True     |
 */
// clang-format on
FORCE_INLINE
void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT);

    WAYPOINT("NSW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) != val);
    WAYPOINT("NSD");
}

// clang-format off
/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal or greater than a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                            | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory   | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore      | uint32_t | Any uint32_t value | True     |
 */
// clang-format on
FORCE_INLINE
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT);

    WAYPOINT("NSMW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
}

// clang-format off
/**
 * Sets the value of a local L1 memory address on the Tensix core executing
 * this function to a specific value. This L1 memory address is used as a
 * semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
 * *noc_semaphore_wait*.
 *
 * Return value: None
 *
 * | Argument  | Description                             | Type     | Valid Range        |Required |
 * |-----------|-----------------------------------------|----------|--------------------|---------|
 * | sem_addr  | Semaphore address in local L1 memory    | uint32_t | 0..1MB             | True    |
 * | val       | Value to set the semaphore to           | uint32_t | Any uint32_t value | True    |
 */
// clang-format on
FORCE_INLINE
void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_SET);

    // set semaphore value to val
    (*sem_addr) = val;
}

// clang-format off
/**
 * Initiates an asynchronous write of a 32-bit value to a NOC destination.
 * Typically used for writing registers, but can be used for memory locations as well.
 * The destination is specified as a 64-bit NOC address (see \a noc_async_write).
 * The advantage over using \a noc_async_write is that we don't a Tensix L1
 * memory source location; the write value is written directly into a register.
 * Unlike using \a noc_async_write, there are also no address alignment concerns.
 * Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument  | Description                                            | Type     | Valid Range                      | Required |
 * |-----------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | addr      | Encoding of the destination location (x,y)+address     | uint64_t | Results of \a get_noc_addr calls | True     |
 * | val       | The value to be written                                | uint32_t | Any uint32_t value               | True     |
 * | be        | Byte-enable                                            | uint8_t  | 0x1-0xF                          | False    |
 */
// clang-format on
template <bool write_to_stream_reg = false, bool posted = false>
FORCE_INLINE void noc_inline_dw_write(
    uint64_t addr, uint32_t val, uint8_t be = 0xF, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_ADDR(noc, addr, 4);
    // This API does not support DRAM addresses
    DEBUG_SANITIZE_NO_DRAM_ADDR(noc, addr, 4);
#ifdef ARCH_BLACKHOLE
    // On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
    // time. If one port on the receipient has no back-pressure then the transaction will hang because there is no
    // mechanism to allow one memory port to move ahead of another. To workaround this hang, we emulate inline writes on
    // Blackhole by writing the value to be written to local L1 first and then issue a noc async write.
    ASSERT((addr & 0x3) == 0);
    if constexpr (write_to_stream_reg) {
        noc_fast_write_dw_inline<noc_mode>(
            noc,
            write_at_cmd_buf,
            val,
            addr,
            be,  // byte-enable
            vc,
            false,  // mcast
            posted  // posted
        );
        WAYPOINT("NWID");
        return;
    }

    ASSERT(be == 0xF);
    uint32_t src_addr = noc_get_interim_inline_value_addr(noc, addr);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, src_addr, 4);
    noc_async_writes_flushed(noc);
    volatile tt_l1_ptr uint32_t* interim_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    *interim_addr_ptr = val;
    ncrisc_noc_fast_write_any_len<noc_mode>(
        noc,
        write_cmd_buf,
        src_addr,
        addr,
        4,
        vc,
        false,  // mcast
        false,  // linked
        1,      // num_dests
        true,   // multicast_path_reserve
        posted  // posted
    );
    noc_async_writes_flushed(noc);
#else
    noc_fast_write_dw_inline<noc_mode>(
        noc,
        write_at_cmd_buf,
        val,
        addr,
        be,  // byte-enable
        vc,
        false,  // mcast
        posted  // posted
    );
#endif
    WAYPOINT("NWID");
}

// on BH this api can only write to stream register, writing to L1 will cause hangs!
template <bool posted = false, bool set_val = false>
FORCE_INLINE void noc_inline_dw_write_set_state(
    uint64_t addr,
    uint32_t val = 0,
    uint8_t be = 0xF,
    uint8_t cmd_buf = write_at_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NWIW");
    // DEBUG_SANITIZE_NOC_ADDR is not needed here because it doesn't send out the request
    // The address could be set here or later in noc_inline_dw_write_with_state

    uint32_t noc_cmd_field = NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_WR_INLINE |
                             0x0 | (posted ? 0x0 : NOC_CMD_RESP_MARKED);

    uint32_t be32 = be;
    uint32_t be_shift = (addr & (NOC_WORD_BYTES - 1));
    // If we're given a misaligned address, don't write to the bytes in the word below the address
    be32 = (be32 << be_shift);

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (set_val) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, addr & 0xFFFFFFFF);
#ifdef ARCH_BLACKHOLE
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
    WAYPOINT("NWID");
}

// on BH this api can only write to stream register, writing to L1 will cause hangs!
template <
    bool update_addr_lo = false,
    bool update_counter = true,
    bool posted = false,
    bool update_addr_hi = false,
    bool update_val = false>
FORCE_INLINE void noc_inline_dw_write_with_state(
    uint32_t val, uint32_t addr = 0, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index) {
    // only either hi or lo address should be getting updated
    static_assert("Error: Only High or Low address update is supported" && (update_addr_lo && update_addr_hi) == 0);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
            } else {
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
            }
        }
    }
    WAYPOINT("NWIW");
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (update_addr_lo) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, addr);
    } else if constexpr (update_addr_hi) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, addr);
    }
    if constexpr (update_val) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += 1;
            } else {
                noc_nonposted_writes_num_issued[noc] += 1;
                noc_nonposted_writes_acked[noc] += 1;
            }
        }
    }
    WAYPOINT("NWID");
}

// clang-format off
/**
 * The Tensix core executing this function call initiates an atomic increment
 * (with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory
 * address is used as a semaphore of size 4 Bytes, as a synchronization
 * mechanism. Refer to <arch>/noc/noc.h for the documentation of noc_atomic_increment.
 *
 * Return value: None
 *
 * | Argument                   | Description                                                      | Type     | Valid Range                      | Required |
 * |----------------------------|------------------------------------------------------------------|----------|----------------------------------|----------|
 * | addr                       | Encoding of the destination location (x,y)+address               | uint64_t | Results of \a get_noc_addr calls | True     |
 * | incr                       | The value to increment by                                        | uint32_t | Any uint32_t value               | True     |
 * | noc_id                     | Which NOC to use for the transaction                             | uint8_t  | 0 or 1                           | False    |
 * | vc                         | Which NOC to use for the transaction                             | uint8_t  | 0-3 (Unicast VCs)                | False    |
 * | posted (template argument) | Whether the call is posted or nonposted (i.e. needs to be acked) | uint32_t | true or false                    | False    |
 */
// clang-format on
template <bool posted = false>
FORCE_INLINE void noc_semaphore_inc(
    uint64_t addr, uint32_t incr, uint8_t noc_id = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::SEMAPHORE_INC, addr, 0, vc);

    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment<noc_mode>(
        noc_id,
        write_at_cmd_buf,
        addr,
        vc,
        incr,
        31 /*wrap*/,
        false /*linked*/,
        posted /*posted*/,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);
    WAYPOINT("NSID");
}

inline void RISC_POST_HEARTBEAT(uint32_t& heartbeat) {
    invalidate_l1_cache();
    volatile uint32_t* ptr = (volatile uint32_t*)(0x1C);
    heartbeat++;
    ptr[0] = 0xAABB0000 | (heartbeat & 0xFFFF);
}

template <bool use_vc>
FORCE_INLINE uint32_t noc_async_read_tile_dram_sharded_set_state(
    uint32_t bank_base_address,
    uint32_t page_size,
    uint32_t bank_id = 0,
    const uint32_t vc = 0,
    uint8_t noc = noc_index) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = bank_base_address + bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc][bank_id];

    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ_DRAM_SHARDED_SET_STATE, uint64_t(src_noc_xy) << 32, page_size, (use_vc) ? vc : -1);

    WAYPOINT("NRTW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRTD");

    if constexpr (use_vc) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, page_size);              // len_bytes

    return src_addr_;
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_with_state(
    uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_DRAM_SHARDED_WITH_STATE);

    uint32_t src_addr_;
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    src_addr_ = src_base_addr + src_addr;

    WAYPOINT("NRTW");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("NRTD");

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr_);  // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

template <bool skip_ptr_update = false>
FORCE_INLINE void noc_async_read_tile_dram_sharded_with_state_with_trid(
    uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_DRAM_SHARDED_WITH_STATE);

    WAYPOINT("NRDW");
    ncrisc_noc_fast_read_with_transaction_id<noc_mode, skip_ptr_update>(
        noc, read_cmd_buf, src_base_addr, src_addr, dest_addr, trid);
    WAYPOINT("NRDD");
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_set_trid(uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_SET_TRID);

    WAYPOINT("NSTW");
    ncrisc_noc_set_transaction_id(noc, read_cmd_buf, trid);
    WAYPOINT("NSTD");
}

FORCE_INLINE
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NBTW");
    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_WITH_TRID);
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NBTD");
}

template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_set_state(
    std::uint64_t dst_noc_addr,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID_SET_STATE, dst_noc_addr, 0, vc);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    WAYPOINT("NAWD");
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                             0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                             (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
}

template <bool update_counter = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_with_state(
    std::uint32_t src_local_l1_addr,
    std::uint32_t dst_noc_addr,
    std::uint32_t size,
    std::uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
            } else {
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
            }
        }
    }
    WAYPOINT("NWPW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID_WITH_STATE, 0ull, size, -1);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    WAYPOINT("NWPD");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_STATE(noc, dst_noc_addr, src_local_l1_addr, size);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += 1;
            } else {
                noc_nonposted_writes_num_issued[noc] += 1;
                noc_nonposted_writes_acked[noc] += 1;
            }
        }
    }
}

template <bool update_counter = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr,
    std::uint32_t size,
    std::uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
            } else {
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
                inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
            }
        }
    }
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, dst_noc_addr, size, -1);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                             0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                             (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (update_counter) {
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += 1;
            } else {
                noc_nonposted_writes_num_issued[noc] += 1;
                noc_nonposted_writes_acked[noc] += 1;
            }
        }
    }
}

FORCE_INLINE
void noc_async_write_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NWTW");
    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NWTD");
}
