// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#define DATA_FORMATS_DEFINED
#endif

#include <algorithm>
#include <stdint.h>
#include <tuple>
#include <utility>
#include <type_traits>

#include "internal/dataflow/dataflow_api_addrgen.h"
#include "core_config.h"
#include "internal/circular_buffer_interface.h"
#include "eth_l1_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"
#include "hostdev/dev_msgs.h"
#include "api/tensor/tensor_accessor.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/sanitize.h"

#if !defined(KERNEL_BUILD)
// This file uses noc_mode, which isn't defined in the firmware build.
#error "dataflow_api.h is only supported in kernel build. Firmware build should use low-level APIs instead."
#endif

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
 * Helper function to check if an address is in L1 memory space (not register space).
 * L1 addresses must be below NOC_REG_SPACE_START_ADDR.
 */
// clang-format on
bool is_l1_address(uint64_t addr) { return ((addr & 0xFFFFFFFF) < NOC_REG_SPACE_START_ADDR); }

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for unique (per core) runtime arguments set via
 * SetRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given unique runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Unique Runtime argument index                                           | uint32_t | 0 to 341    | True     |
 */
// clang-format on
static FORCE_INLINE uintptr_t get_arg_addr(int arg_idx) { return (uintptr_t)&rta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the address in L1 for a given runtime argument index for common (all cores) runtime arguments set via
 * SetCommonRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given common runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-------------|----------|
 * | arg_idx        | Common Runtime argument index                                           | uint32_t | 0 to 341    | True     |
 */
// clang-format on
static FORCE_INLINE uintptr_t get_common_arg_addr(int arg_idx) { return (uintptr_t)&crta_l1_base[arg_idx]; }

// clang-format off
/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs()
 * API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 341    | True     |
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
 * | arg_idx               | Common Runtime argument index                  | uint32_t              | 0 to 341    | True     |
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

inline void wait_for_sync_register_value(uintptr_t addr, int32_t val) {
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
    uintptr_t pages_acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(operand);

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
    uintptr_t pages_acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(operand);

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
    uintptr_t pages_received_ptr = (uintptr_t)get_cb_tiles_received_ptr(operand);

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
    uintptr_t pages_received_ptr = (uintptr_t)get_cb_tiles_received_ptr(operand);

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
 *
 * Return value: None
 *
 * | Argument                          | Description                                        | Data type | Valid range                      | required |
 * |-----------------------------------|----------------------------------------------------|-----------|----------------------------------|----------|
 * | src_noc_addr                      | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | dst_local_l1_addr                 | Address in local L1 memory                         | uint32_t  | 0..1MB                           | True     |
 * | size                              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                           | True     |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1                           | False    |
 */
// clang-format on
template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_one_packet(
    uint64_t src_noc_addr,
    uint32_t dst_local_l1_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t read_req_vc = NOC_UNICAST_WRITE_VC) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ, dst_local_l1_addr, src_noc_addr, size, -1, false);
    }

    WAYPOINT("RP2W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP2D");

    WAYPOINT("NAOW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
    ncrisc_noc_fast_read<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size, read_req_vc);
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
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true>
inline void noc_async_read(
    uint64_t src_noc_addr,
    uint32_t dst_local_l1_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t read_req_vc = NOC_UNICAST_WRITE_VC) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ, dst_local_l1_addr, src_noc_addr, size, -1, false);
    }

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);
    } else {
        WAYPOINT("NARW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
        ncrisc_noc_fast_read_any_len<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size, read_req_vc);
        WAYPOINT("NARD");
    }
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous read for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_read_set_state for more details.
 *
 * Return value: None
 *
 * | Argument                          | Description                                        | Data type | Valid range                              | required |
 * |-----------------------------------|----------------------------------------------------|-----------|------------------------------------------|----------|
 * | src_noc_addr                      | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls         | True     |
 * | size                              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                                   | True     |
 * | vc                                | Which VC to use for the transaction                | uint32_t  | 0-3 (Unicast VCs)                        | False    |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1                                   | False    |
 * | max_page_size (template argument) | Maximum size of a single transaction in bytes      | uint32_t  | Any uint32_t number                      | False    |
 * | use_vc (template argument)        | Enable custom VC usage                             | bool      | True or False                            | False    |
 */
// clang-format on
template <bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_set_state(
    uint64_t src_noc_addr, uint32_t size, const uint32_t vc = 0, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ_SET_STATE, 0, src_noc_addr, size, (use_vc) ? static_cast<int8_t>(vc) : -1, false);

    WAYPOINT("NASW");
    ncrisc_noc_read_set_state<noc_mode, true /* one_packet */, use_vc>(noc, read_cmd_buf, src_noc_addr, size, vc);
    WAYPOINT("NASD");
}

// clang-format off
/**
 * Initiates an asynchronous read for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_read_with_state for more details.
 *
 * Return value: None
 *
 * | Argument                          | Description                                        | Data type | Valid range         | required |
 * |-----------------------------------|----------------------------------------------------|-----------|-------------------- |----------|
 * | src_local_l1_addr                 | Address in local L1 memory on source core          | uint32_t  | 0..1MB              | True     |
 * | dst_local_l1_addr                 | Address in local L1 memory on destination core     | uint32_t  | 0..1MB              | True     |
 * | vc                                | Which VC to use for the transaction                | uint32_t  | 0-3 (Unicast VCs)   | False    |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1              | False    |
 * | inc_num_issued (template argument)| Whether issued read counter should be increment    | uint32_t  | Any uint32_t number | False    |
 * | use_vc (template argument)        | Enable custom VC usage                             | bool      | True or False       | False    |
 */
// clang-format on
template <bool inc_num_issued = true, bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_with_state(
    uint32_t src_local_l1_addr, uint32_t dst_local_l1_addr, const uint32_t vc = 0, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ_WITH_STATE,
        dst_local_l1_addr,
        static_cast<uint64_t>(src_local_l1_addr),
        0,
        (use_vc) ? static_cast<int8_t>(vc) : -1,
        false);

    WAYPOINT("NATW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, src_local_l1_addr, dst_local_l1_addr);

    ncrisc_noc_read_with_state<noc_mode, inc_num_issued, true /* one_packet */>(
        noc, read_cmd_buf, src_local_l1_addr, dst_local_l1_addr);

    WAYPOINT("NATD");
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a noc_async_read_with_state, which will issue the actual read request.
 * \a noc_async_read can be used instead if the state preservation is not
 * needed. Also, see \a noc_async_read_barrier.
 *
 * The source node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                          | Description                                        | Data type | Valid range                              | required |
 * |-----------------------------------|----------------------------------------------------|-----------|------------------------------------------|----------|
 * | src_noc_addr                      | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls         | True     |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1                                   | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_read_set_state(uint64_t src_noc_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_SET_STATE, 0, src_noc_addr, 0, -1, false);

    WAYPOINT("NAUW");
    ncrisc_noc_read_set_state<noc_mode>(noc, read_cmd_buf, src_noc_addr);
    WAYPOINT("NAUD");
}

// clang-format off
/**
 * Initiates an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function must be preceded by a call to
 * \a noc_async_read_set_state. This function is used to issue the actual
 * read request after the state has been set up. \a noc_async_read can be
 * used instead if the state preservation is not needed. Also, see
 * \a noc_async_read_barrier.
 *
 * Return value: None
 *
 * | Argument                          | Description                                        | Data type | Valid range         | required |
 * |-----------------------------------|----------------------------------------------------|-----------|-------------------- |----------|
 * | src_local_l1_addr                 | Address in local L1 memory on source core          | uint32_t  | 0..1MB              | True     |
 * | dst_local_l1_addr                 | Address in local L1 memory on destination core     | uint32_t  | 0..1MB              | True     |
 * | size                              | Size of data transfer in bytes                     | uint32_t  | 0..1MB              | True     |
 * | noc                               | Which NOC to use for the transaction               | uint8_t   | 0 or 1              | False    |
 * | inc_num_issued (template argument)| Whether issued read counter should be increment    | uint32_t  | Any uint32_t number | False    |
 */
// clang-format on
template <bool inc_num_issued = true>
FORCE_INLINE void noc_async_read_with_state(
    uint32_t src_local_l1_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ_WITH_STATE, dst_local_l1_addr, static_cast<uint64_t>(src_local_l1_addr), size, -1, false);

    WAYPOINT("NAVW");

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc, src_local_l1_addr, dst_local_l1_addr, size);

    ncrisc_noc_read_any_len_with_state<noc_mode, inc_num_issued>(
        noc, read_cmd_buf, src_local_l1_addr, dst_local_l1_addr, size);

    WAYPOINT("NAVD");
}

// clang-format off
/**
 * Increments the number of issued reads counter. This is used to manually increment the number of issued reads counter.
 *
 * Return value: None
 *
 * | Argument                   | Description                            | Type     | Valid Range         | Required |
 * |----------------------------|----------------------------------------|----------|---------------------|----------|
 * | num_issued_reads_inc       | Number of reads to increment by        | uint32_t | Any uint32_t number | True     |
 * | noc                        | Which NOC's counters to increment      | uint8_t  | 0 or 1              | False    |
 */
// clang-format on
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
 *
 * Return value: None
 *
 * | Argument                               | Description                                            | Type     | Valid Range                      | Required |
 * |----------------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | src_local_l1_addr                      | Source address in local L1 memory                      | uint32_t | 0..1MB                           | True     |
 * | dst_noc_addr                           | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | True     |
 * | size                                   | Size of data transfer in bytes                         | uint32_t | 0..1MB                           | True     |
 * | noc                                    | Which NOC to use for the transaction                   | uint8_t  | 0 or 1                           | False    |
 * | vc                                     | Which VC to use for the transaction                    | uint8_t  | 0-3 (Unicast VCs)                | False    |
 * | enable_noc_tracing (template argument) | NOC tracing enable                                     | bool     | true or false                    | False    |
 * | posted (template argument)             | Whether the write is posted (i.e. no ack required)     | bool     | true or false                    | False    |
 */
// clang-format on
template <bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr,
    std::uint32_t size,
    uint8_t noc = noc_index,
    uint32_t vc = NOC_UNICAST_WRITE_VC) {
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_, src_local_l1_addr, dst_noc_addr, size, vc, posted);
    }

    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    ncrisc_noc_fast_write<noc_mode>(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        vc,
        false /* mcast */,
        false /* linked */,
        1 /* num_dests */,
        true /* multicast_path_reserve */,
        posted);
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
 * | Argument                               | Description                                             | Type     | Valid Range                      | Required |
 * |----------------------------------------|---------------------------------------------------------|----------|----------------------------------|----------|
 * | src_local_l1_addr                      | Source address in local L1 memory                       | uint32_t | 0..1MB                           | True     |
 * | dst_noc_addr                           | Encoding of the destination NOC location (x,y)+address  | uint64_t | Results of \a get_noc_addr calls | True     |
 * | size                                   | Size of data transfer in bytes                          | uint32_t | 0..1MB                           | True     |
 * | noc                                    | Which NOC to use for the transaction                    | uint8_t  | 0 or 1                           | False    |
 * | max_page_size (template argument)      | Maximum size of a single transaction in bytes           | uint32_t | Any uint32_t number              | False    |
 * | enable_noc_tracing (template argument) | NOC tracing enable                                      | bool     | true or false                    | False    |
 * | posted (template argument)             | Whether the write is posted (i.e. no ack required)      | bool     | true or false                    | False    |
 */
// clang-format on
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true, bool posted = false>
inline void noc_async_write(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t vc = NOC_UNICAST_WRITE_VC) {
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_, src_local_l1_addr, dst_noc_addr, size, vc, posted);
    }

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_one_packet<false, posted>(src_local_l1_addr, dst_noc_addr, size, noc, vc);
    } else {
        WAYPOINT("NAWW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc, write_cmd_buf, src_local_l1_addr, dst_noc_addr, size, vc, false, false, 1, true, posted);
        WAYPOINT("NAWD");
    }
}

// clang-format off
/**
 * Initiates an asynchronous multicast write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Refer to \a noc_async_write_multicast for more details.
 */
// clang-format on
template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_write_multicast_one_packet(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    if constexpr (enable_noc_tracing) {
        NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
        RECORD_NOC_EVENT_WITH_ADDR(
            NocEventType::WRITE_MULTICAST,
            src_local_l1_addr,
            dst_noc_addr_multicast,
            size,
            NOC_MULTICAST_WRITE_VC,
            false);
    }
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
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::WRITE_MULTICAST, src_local_l1_addr, dst_noc_addr_multicast, size, NOC_MULTICAST_WRITE_VC, false);

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast_one_packet<false>(src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, linked);
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

// clang-format off
/**
 * Sets the stateful registers for an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size)
 * to a specified destination node located at NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a noc_async_write_one_packet_with_state, which will issue the actual
 * write request. \a noc_async_write can be used instead if the state preservation is not
 * needed. Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                       | Description                                            | Data type | Valid range                      | required |
 * |--------------------------------|--------------------------------------------------------|-----------|----------------------------------|----------|
 * | dst_noc_addr                   | Encoding of the destination NOC location (x,y)+address | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | size                           | Size of data transfer in bytes                         | uint32_t  | 0..1MB                           | True     |
 * | noc                            | Which NOC to use for the transaction                   | uint8_t   | 0 or 1                           | False    |
 * | vc                             | Which VC to use for the transaction                    | uint8_t   | 0-3                              | False    |
 * | posted (template argument)     | Whether the write is posted (i.e. no ack required)     | bool      | true or false                    | False    |
 */
// clang-format on
template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_set_state(
    uint64_t dst_noc_addr, uint32_t size, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_SET_STATE, 0, dst_noc_addr, size, vc, posted);

    WAYPOINT("NWPW");
    ncrisc_noc_write_set_state<posted, true /* one_packet */>(noc, write_cmd_buf, dst_noc_addr, size, vc);
    WAYPOINT("NWPD");
}

// clang-format off
/**
 * Initiates an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size)
 * to a specified destination node located at NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function must be preceded by a call to
 * \a noc_async_write_one_packet_set_state. This function is used to issue the actual
 * write request after the state has been set up.
 * \a noc_async_write can be used instead if the state preservation is not needed. Also, see \a noc_async_write_barrier.
 *
 * Return value: None
 *
 * | Argument                       | Description                                        | Data type | Valid range   | required |
 * |--------------------------------|----------------------------------------------------|-----------|---------------|----------|
 * | src_local_l1_addr              | Address in local L1 memory on source core          | uint32_t  | 0..1MB        | True     |
 * | dst_local_l1_addr              | Address in local L1 memory on destination core     | uint32_t  | 0..1MB        | True     |
 * | noc                            | Which NOC to use for the transaction               | uint8_t   | 0 or 1        | False    |
 * | posted (template argument)     | Whether the write is posted (i.e. no ack required) | bool      | true or false | False    |
 */
// clang-format on
template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_state(
    uint32_t src_local_l1_addr, uint32_t dst_local_l1_addr, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_STATE, src_local_l1_addr, 0ull, 0, -1, posted);

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, dst_local_l1_addr, src_local_l1_addr);

    WAYPOINT("NWPW");
    ncrisc_noc_write_with_state<noc_mode, posted, true /* update_counter */, true /* one_packet */>(
        noc, write_cmd_buf, src_local_l1_addr, dst_local_l1_addr);
    WAYPOINT("NWPD");
}

// clang-format off
/**
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the AddrGen object.
 * This function is the generic implementation that can be used with any address generator that provides
 * the get_noc_addr method and either a page_size or a log_base_2_of_page_size
 * member variable. It is designed to be flexible and can be used with various address generators.
 * Note that providing the size argument is optional, and if provided,
 * it will override the default page size of the address generator.
 *
 * Return value: None
 *
 * | Argument                     | Description                          | Data type | Valid range                                    | required |
 * |------------------------------|--------------------------------------|-----------|------------------------------------------------|----------|
 * | id                           | Page id                              | uint32_t  | Any uint32_t number                            | True     |
 * | addrgen                      | Address generator object             | AddrGen   | N/A                                            | True     |
 * | dst_local_l1_addr            | Address in local L1 memory           | uint32_t  | 0..1MB                                         | True     |
 * | offset                       | Custom address offset                | uint32_t  | 0..1MB                                         | False    |
 * | noc                          | Which NOC to use for the transaction | uint8_t   | 0 or 1                                         | False    |
 * | AddrGen (template parameter) | Address generator class              | typename  | Any AddrGen class in \a dataflow_api_addrgen.h | True     |
 */
// clang-format on
template <typename AddrGen, bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const AddrGen& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    static_assert(
        has_required_addrgen_traits_v<AddrGen>,
        "AddrGen must have get_noc_addr() and either page_size or log_base_2_of_page_size member variable");

    uint32_t page_size;
    if constexpr (has_page_size_v<AddrGen>) {
        page_size = addrgen.page_size;
    } else {
        page_size = (1 << addrgen.log_base_2_of_page_size);
    }
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, page_size, -1, false);
    }
    noc_async_read<NOC_MAX_BURST_SIZE + 1, false>(
        addrgen.get_noc_addr(id, offset, noc), dst_local_l1_addr, page_size, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_read_page instead.
 *
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the InterleavedAddrGen object.
 * This function is a convenience wrapper around noc_async_read_page for InterleavedAddrGen objects.
 * Refer to template <typename AddrGen> noc_async_read_page for a generic implementation and more details.
 */
// clang-format on
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_read_page instead.")]]
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, addrgen.page_size, -1, false);
    noc_async_read_page<InterleavedAddrGen<DRAM>, false>(id, addrgen, dst_local_l1_addr, offset, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_read_page instead.
 *
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the InterleavedAddrGen object.
 * This function is a convenience wrapper around noc_async_read_page for InterleavedAddrGen objects.
 * Refer to template <typename AddrGen> noc_async_read_page for a generic implementation and more details.
 */
// clang-format on
template <typename DSpec>
[[deprecated("Use <typename AddrGen> noc_async_read_page instead.")]]
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const TensorAccessor<DSpec>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, addrgen.page_size, -1, false);
    noc_async_read_page<TensorAccessor<DSpec>, false>(id, addrgen, dst_local_l1_addr, offset, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_read_page instead.
 *
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the InterleavedAddrGen object.
 * This function is a convenience wrapper around noc_async_read_page for InterleavedAddrGen objects.
 * Refer to template <typename AddrGen> noc_async_read_page for a generic implementation and more details.
 */
// clang-format on
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_read_page instead.")]]
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, addrgen.page_size, -1, false);
    noc_async_read_page<InterleavedAddrGen<DRAM>, false>(id, addrgen, dst_local_l1_addr, offset, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_read_page instead.
 *
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the InterleavedAddrGenFast object.
 * This function is a convenience wrapper around noc_async_read_page for InterleavedAddrGenFast objects.
 * Refer to template <typename AddrGen> noc_async_read_page for a generic implementation and more details.
 *
 * Extra arguments:
 *
 * | Argument                     | Description         | Data type | Valid range         | required |
 * |------------------------------|---------------------|-----------|---------------------|----------|
 * | tile_hw (template parameter) | Tile height x width | uint32_t  | Any uint32_t number | True     |
 */
// clang-format on
template <bool DRAM, uint32_t tile_hw>
[[deprecated("Use <typename AddrGen> noc_async_read_page instead.")]]
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, addrgen.page_size, -1, false);
    noc_async_read_page<InterleavedAddrGenFast<DRAM, tile_hw>, false>(id, addrgen, dst_local_l1_addr, offset, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_read_page instead.
 *
 * Initiates an asynchronous read for a single packet with transaction size and source location determined by the InterleavedPow2AddrGenFast object.
 * This function is a convenience wrapper around noc_async_read_page for InterleavedPow2AddrGenFast objects.
 * Refer to template <typename AddrGen> noc_async_read_page for a generic implementation and more details.
 */
// clang-format on
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_read_page instead.")]]
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const InterleavedPow2AddrGenFast<DRAM>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(
        NocEventType::READ, dst_local_l1_addr, id, addrgen, offset, 1 << addrgen.log_base_2_of_page_size, -1, false);
    noc_async_read_page<InterleavedPow2AddrGenFast<DRAM>, false>(id, addrgen, dst_local_l1_addr, offset, noc);
}

// clang-format off
/**
 * Initiates an asynchronous write for a single packet with transaction size and destination location determined by the AddrGen object.
 * This function is the generic implementation that can be used with any address generator that provides
 * the get_noc_addr method and either a page_size or a log_base_2_of_page_size member variable.
 * It is designed to be flexible and can be used with various address generators.
 * Note that providing the size argument is optional, and if provided,
 * it will override the default page size of the address generator.
 *
 * Return value: None
 *
 * | Argument                                | Description                                             | Data type | Valid range                                    | required |
 * |-----------------------------------------|---------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | id                                      | Page id                                                 | uint32_t  | Any uint32_t number                            | True     |
 * | addrgen                                 | Address generator object                                | AddrGen   | N/A                                            | True     |
 * | src_local_l1_addr                       | Address in local L1 memory                              | uint32_t  | 0..1MB                                         | True     |
 * | size                                    | Size of data in bytes                                   | uint32_t  | 0..NOC_MAX_BURST_SIZE MB                       | False    |
 * | offset                                  | Custom address offset                                   | uint32_t  | 0..1MB                                         | False    |
 * | noc                                     | Which NOC to use for the transaction                    | uint8_t   | 0 or 1                                         | False    |
 * | AddrGen (template parameter)            | Address generator class                                 | typename  | Any AddrGen class in \a dataflow_api_addrgen.h | True     |
 * | enable_noc_tracing (template parameter) | NOC tracing enable                                      | bool      | true or false                                  | False    |
 * | posted (template parameter)             | Whether the write is posted (i.e. no ack required)      | bool      | true or false                                  | False    |
 */
// clang-format on
template <typename AddrGen, bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const AddrGen& addrgen,
    uint32_t src_local_l1_addr,
    uint32_t size = 0,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    static_assert(
        has_required_addrgen_traits_v<AddrGen>,
        "AddrGen must have get_noc_addr() and either page_size or log_base_2_of_page_size member variable");

    uint32_t page_size;
    if constexpr (has_page_size_v<AddrGen>) {
        page_size = addrgen.page_size;
    } else {
        page_size = (1 << addrgen.log_base_2_of_page_size);
    }
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ID(
            NocEventType::WRITE_,
            src_local_l1_addr,
            id,
            addrgen,
            offset,
            size ? size : page_size,
            NOC_UNICAST_WRITE_VC,
            posted);
    }
    noc_async_write<NOC_MAX_BURST_SIZE + 1, false, posted>(
        src_local_l1_addr, addrgen.get_noc_addr(id, offset, noc), size ? size : page_size, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_write_page instead.
 *
 * Initiates an asynchronous write for a single packet with custom transaction size, and destination location determined by the InterleavedAddrGen object.
 * This function is a convenience wrapper around noc_async_write_page for InterleavedAddrGen objects.
 * Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
 *
 * Extra arguments:
 *
 * | Argument          | Description                          | Data type                  | Valid range              | required |
 * |-------------------|--------------------------------------|----------------------------|--------------------------|----------|
 * | write_size_bytes  | Size of data transfer in bytes       | uint32_t                   | 0..NOC_MAX_BURST_SIZE MB | True     |
 */
// clang-format on
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_write_page instead.")]]
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& addrgen,
    uint32_t src_local_l1_addr,
    const uint32_t write_size_bytes,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(
        NocEventType::WRITE_, src_local_l1_addr, id, addrgen, offset, write_size_bytes, NOC_UNICAST_WRITE_VC, false);
    noc_async_write_page<InterleavedAddrGen<DRAM>, false>(
        id, addrgen, src_local_l1_addr, write_size_bytes, offset, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_write_page instead.
 *
 * Initiates an asynchronous write for a single packet with transaction size and destination location determined by the InterleavedAddrGen object.
 * This function is a convenience wrapper around noc_async_write_page for InterleavedAddrGen objects.
 * Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
 */
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_write_page instead.")]]
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGen<DRAM>& addrgen, uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::WRITE_, src_local_l1_addr, id, addrgen, 0 /* offset */, addrgen.page_size, NOC_UNICAST_WRITE_VC, false);
    noc_async_write_page<InterleavedAddrGen<DRAM>, false>(
        id, addrgen, src_local_l1_addr, addrgen.page_size, 0 /* offset */, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_write_page instead.
 *
 * Initiates an asynchronous write for a single packet with transaction size and destination location determined by the InterleavedAddrGenFast object.
 * This function is a convenience wrapper around noc_async_write_page for InterleavedAddrGenFast objects.
 * Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
 *
 * Extra arguments:
 *
 * | Argument                     | Description         | Data type | Valid range         | required |
 * |------------------------------|---------------------|-----------|---------------------|----------|
 * | tile_hw (template parameter) | Tile height x width | uint32_t  | Any uint32_t number | True     |
 */
template <bool DRAM, uint32_t tile_hw>
[[deprecated("Use <typename AddrGen> noc_async_write_page instead.")]]
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& addrgen,
    uint32_t src_local_l1_addr,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(NocEventType::WRITE_, src_local_l1_addr, id, addrgen, 0 /* offset */, addrgen.page_size, NOC_UNICAST_WRITE_VC, false);
    noc_async_write_page<InterleavedAddrGenFast<DRAM, tile_hw>, false>(
        id, addrgen, src_local_l1_addr, addrgen.page_size, 0 /* offset */, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_write_page instead.
 *
 * Initiates an asynchronous write for a single packet with transaction size and destination location determined by the TensorAccessor object.
 * This function is a convenience wrapper around noc_async_write_page for TensorAccessor objects.
 * Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
 */
// clang-format on
template <typename DSpec>
[[deprecated("Use <typename AddrGen> noc_async_write_page instead.")]]
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const TensorAccessor<DSpec>& addrgen, uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(
        NocEventType::WRITE_,
        src_local_l1_addr,
        id,
        addrgen,
        0 /* offset */,
        addrgen.page_size,
        NOC_UNICAST_WRITE_VC,
        false);
    noc_async_write_page<TensorAccessor<DSpec>, false>(
        id, addrgen, src_local_l1_addr, addrgen.page_size, 0 /* offset */, noc);
}

// clang-format off
/**
 * THIS API IS DEPRECATED AND WILL BE REMOVED SOON. Use <typename AddrGen> noc_async_write_page instead.
 *
 * Initiates an asynchronous write for a single packet with transaction size and destination location determined by the InterleavedPow2AddrGenFast object.
 * This function is a convenience wrapper around noc_async_write_page for InterleavedPow2AddrGenFast objects.
 * Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
 *
 * Extra arguments:
 *
 * | Argument          | Description                    | Data type | Valid range              | required |
 * |-------------------|--------------------------------|-----------|--------------------------|----------|
 * | write_size_bytes  | Size of data transfer in bytes | uint32_t  | 0..NOC_MAX_BURST_SIZE MB | True     |
 *
 */
// clang-format on
template <bool DRAM>
[[deprecated("Use <typename AddrGen> noc_async_write_page instead.")]]
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,
    const InterleavedPow2AddrGenFast<DRAM>& addrgen,
    uint32_t src_local_l1_addr,
    const uint32_t write_size_bytes,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ID(
        NocEventType::WRITE_, src_local_l1_addr, id, addrgen, offset, write_size_bytes, NOC_UNICAST_WRITE_VC, false);
    noc_async_write_page<InterleavedPow2AddrGenFast<DRAM>, false>(
        id, addrgen, src_local_l1_addr, write_size_bytes, offset, noc);
}

// clang-format off
/**
 * Initiates an asynchronous read of a shard from a source noc address into a local L1 address.
 * The size of the transaction and the source address are determined by the TensorAccessor object.
 * This function only works for sharded tensors.
 *
 * Return value: None
 *
 * | Argument                   | Description                                      | Type           | Valid Range                                              | Required |
 * |----------------------------|--------------------------------------------------|----------------|----------------------------------------------------------|----------|
 * | shard_id                   | Row-major index of a shard in the sharded tensor | uint32_t       | Any uint32_t number                                      | True     |
 * | s                          | TensorAccessor object                            | TensorAccessor | Any TensorAccessor object, refer to \a tensor_accessor.h | True     |
 * | dst_local_l1_addr          | Destination address in local L1 memory           | uint32_t       | 0..1MB                                                   | True     |
 * | noc                        | Which NOC to use for the transaction             | uint8_t        | 0 or 1                                                   | False    |
 */
// clang-format on
template <typename DSpec>
FORCE_INLINE void noc_async_read_shard(
    const uint32_t shard_id, const TensorAccessor<DSpec>& s, std::uint32_t dst_local_l1_addr, uint8_t noc = noc_index) {
    auto shard_volume = s.dspec().shard_volume();
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ,
        dst_local_l1_addr,
        s.get_shard_noc_addr(shard_id, noc),
        s.page_size * shard_volume,
        -1,
        false);
    noc_async_read<NOC_MAX_BURST_SIZE + 1, false>(
        s.get_shard_noc_addr(shard_id, noc), dst_local_l1_addr, s.page_size * shard_volume, noc);
}

// clang-format off
/**
 * Initiates an asynchronous write of a shard from a local L1 address to a destination noc address.
 * The size of the transaction and the destination address are determined by the TensorAccessor object.
 * This function only works for sharded tensors.
 *
 * Return value: None
 *
 * | Argument                    | Description                                        | Type           | Valid Range                                              | Required |
 * |-----------------------------|----------------------------------------------------|----------------|----------------------------------------------------------|----------|
 * | shard_id                    | Row-major index of a shard in the sharded tensor   | uint32_t       | Any uint32_t number                                      | True     |
 * | s                           | TensorAccessor object                              | TensorAccessor | Any TensorAccessor object, refer to \a tensor_accessor.h | True     |
 * | src_local_l1_addr           | Source address in local L1 memory                  | uint32_t       | 0..1MB                                                   | True     |
 * | noc                         | Which NOC to use for the transaction               | uint8_t        | 0 or 1                                                   | False    |
 * | DSpec (template parameter)  | DistributionSpec type                              | typename       | Any DistributionSpec object                              | False    |
 * | posted (template parameter) | Whether the write is posted (i.e. no ack required) | bool           | true or false                                            | False    |
 */
// clang-format on
template <typename DSpec, bool posted = false>
FORCE_INLINE void noc_async_write_shard(
    const uint32_t shard_id, const TensorAccessor<DSpec>& s, std::uint32_t src_local_l1_addr, uint8_t noc = noc_index) {
    auto shard_volume = s.dspec().shard_volume();
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::WRITE_,
        src_local_l1_addr,
        s.get_shard_noc_addr(shard_id, noc),
        s.page_size * shard_volume,
        NOC_UNICAST_WRITE_VC,
        posted);
    noc_async_write<NOC_MAX_BURST_SIZE + 1, false, posted>(
        src_local_l1_addr, s.get_shard_noc_addr(shard_id, noc), s.page_size * shard_volume, noc);
}

// clang-format off
/**
 * Returns the local address of the semaphore with the given id.
 *
 * Return value: Local address of the semaphore (uint32_t)
 *
 * | Argument                  | Description                | Type                     | Valid Range              | Required |
 * |---------------------------|----------------------------|--------------------------|--------------------------|----------|
 * | semaphore_id              | Semaphore id               | uint32_t                 | 0..2^20-1                | True     |
 * | type (template parameter) | Type of the core           | ProgrammableCoreType     | Any ProgrammableCoreType | False    |
 */
// clang-format on
template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
FORCE_INLINE uint32_t get_semaphore(uint32_t semaphore_id) {
    return (uint32_t)sem_l1_base[static_cast<int>(type)] + semaphore_id * L1_ALIGNMENT;
}

// clang-format off
/**
 * Initiates an asynchronous write from a source address in L1 memory on the
 * Tensix core executing this function call to a single destination node.
 * The size of data that is sent is 4 Bytes. This is usually used to set a
 * semaphore value at the destination node, as a way of synchronization.
 *
 * Return value: None
 *
 * | Argument               | Description                          | Type     | Valid Range                     | Required |
 * |------------------------|--------------------------------------|----------|---------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory    | uint32_t | 0..1MB                          | True     |
 * | dst_noc_addr           | Destination NOC address              | uint64_t | Results of \a get_noc_addr call | True     |
 * | noc                    | Which NOC to use for the transaction | uint8_t  | 0 or 1                          | False    |
 */
// clang-format on
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

// clang-format off
/**
 * Initiates an asynchronous write from a source address in L1 memory on the
 * Tensix core executing this function call to a rectangular destination grid.
 * This API is the same as *noc_async_write_multicast* but with the multicast
 * sender being part of the multicast destinations. Refer to *noc_async_write_multicast* for more details.
 *
 * Return value: None
 *
 * | Argument                          | Description                                                              | Type     | Valid Range                                | Required |
 * |-----------------------------------|--------------------------------------------------------------------------|----------|--------------------------------------------|----------|
 * | src_local_l1_addr                 | Source address in local L1 memory                                        | uint32_t | 0..1MB                                     | True     |
 * | dst_noc_addr_multicast            | Encoding of the destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | Results of \a get_noc_multicast_addr calls | True     |
 * | size                              | Size of data transfer in bytes                                           | uint32_t | 0..1MB                                     | True     |
 * | num_dests                         | Number of destinations that the multicast source is targeting            | uint32_t | 0..(number of cores -1)                    | True     |
 * | linked                            | Whether the transaction is linked                                        | bool     | true or false                              | False    |
 * | noc                               | Which NOC to use for the transaction                                     | uint8_t  | 0 or 1                                     | False    |
 */
// clang-format on
inline void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    constexpr bool multicast_path_reserve = true;

    NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::WRITE_MULTICAST, src_local_l1_addr, dst_noc_addr_multicast, size, NOC_MULTICAST_WRITE_VC, false);

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
 * | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
 */
void noc_async_read_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_START, false);

    WAYPOINT("NRBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_reads_flushed(noc));
    } else {
        while (!ncrisc_noc_reads_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NRBD");

    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_END, false);
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
 * | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
 */
FORCE_INLINE
void noc_async_write_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_START, false);

    WAYPOINT("NWBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
    } else {
        while (!ncrisc_noc_nonposted_writes_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NWBD");

    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_END, false);
}

/**
 * This blocking call waits for all outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to depart, but will not wait
 * for them to complete
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
 */
FORCE_INLINE
void noc_async_writes_flushed(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_FLUSH, false);

    WAYPOINT("NWFW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_writes_sent(noc));
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
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
 */
FORCE_INLINE
void noc_async_posted_writes_flushed(uint8_t noc = noc_index) {
    WAYPOINT("NPWW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_posted_writes_sent(noc));
    } else {
        while (!ncrisc_noc_posted_writes_sent(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NPWD");
}

/**
 * This blocking call waits for all the outstanding enqueued atomic
 * transactions issued on the current Tensix core to complete. After returning
 * from this call the atomic transaction queue will be empty for the current
 * Tensix core.
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | noc_idx  | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
 */
FORCE_INLINE
void noc_async_atomic_barrier(uint8_t noc_idx = noc_index) {
    RECORD_NOC_EVENT(NocEventType::ATOMIC_BARRIER, false);

    WAYPOINT("NABW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_idx));
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
    RECORD_NOC_EVENT(NocEventType::FULL_BARRIER, false);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        WAYPOINT("NFBW");
        while (!ncrisc_dynamic_noc_reads_flushed(noc_idx)) {
            invalidate_l1_cache();
        }
        WAYPOINT("NFCW");
        while (!ncrisc_dynamic_noc_nonposted_writes_sent(noc_idx)) {
            invalidate_l1_cache();
        }
        WAYPOINT("NFDW");
        while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc_idx)) {
            invalidate_l1_cache();
        }
        WAYPOINT("NFEW");
        while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_idx)) {
            invalidate_l1_cache();
        }
        WAYPOINT("NFFW");
        while (!ncrisc_dynamic_noc_posted_writes_sent(noc_idx)) {
            invalidate_l1_cache();
        }
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
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT, false);

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
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT, false);

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
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_SET, false);

    // set semaphore value to val
    (*sem_addr) = val;
}

// clang-format off
/**
 * Initiates an asynchronous write of a 32-bit value to a NOC destination.
 * Typically used for writing registers, but can be used for memory locations as well.
 * The destination is specified as a 64-bit NOC address (see \a noc_async_write).
 * The advantage over using \a noc_async_write is that we don't use a Tensix L1
 * memory source location; the write value is written directly into a register.
 * Unlike using \a noc_async_write, there are also no address alignment concerns.
 * Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Return value: None
 *
 * | Argument                                 | Description                                                | Type     | Valid Range                      | Required |
 * |------------------------------------------|------------------------------------------------------------|----------|----------------------------------|----------|
 * | addr                                     | Encoding of the destination location (x,y)+address         | uint64_t | Results of \a get_noc_addr calls | True     |
 * | val                                      | The value to be written                                    | uint32_t | Any uint32_t value               | True     |
 * | be                                       | Byte-enable                                                | uint8_t  | 0x1-0xF                          | False    |
 * | noc                                      | NOC to use for the transaction                             | uint8_t  | 0 or 1                           | False    |
 * | vc                                       | Virtual channel to use for the transaction                 | uint8_t  | 0-3 (Unicast VCs)                | False    |
 * | customized_src_addr                      | Custom source address for storing the value to be written  | uint32_t | Any uint32_t value               | False    |
 * |                                          | (required when `flush` is false)                           |          |                                  |          |
 * | dst_type            (template parameter) | Whether the write is targeting L1 or a Stream Register     | InlineWriteDst     | DEFAULT, L1, REG       | False    |
 * | posted              (template parameter) | Whether the call is posted (i.e. ack requirement)          | bool     | true or false                    | False    |
 * | flush               (template parameter) | Whether to flush the NOC transaction before issuing the    | bool     | true or false                    | False    |
 * |                                          | write (`false` callers must prevent races on the caller    |          |                                  |          |
 * |                                          | side)                                                      |          |                                  |          |
 *
 * When `flush` is disabled the caller is responsible for providing a valid `customized_src_addr` scratch location and
 * ensuring no outstanding inline write uses that address before issuing another write.
 */
// clang-format on
template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true>
FORCE_INLINE void noc_inline_dw_write(
    uint64_t addr,
    uint32_t val,
    uint8_t be = 0xF,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t customized_src_addr = 0) {
    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_ADDR(noc, addr, 4);
    DEBUG_SANITIZE_NO_DRAM_ADDR(noc, addr, 4);
#if defined(ARCH_BLACKHOLE) && defined(WATCHER_ENABLED)
    if constexpr (dst_type == InlineWriteDst::L1) {
        if constexpr (!flush) {
            ASSERT(customized_src_addr != 0);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, customized_src_addr, 4);
        } else {
            uint32_t src_addr = noc_get_interim_inline_value_addr(noc, addr);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, src_addr, 4);
        }
    }
#endif

    noc_fast_write_dw_inline<noc_mode, dst_type, flush>(
        noc,
        write_at_cmd_buf,
        val,
        addr,
        be,  // byte-enable
        vc,
        false,   // mcast
        posted,  // posted
        customized_src_addr);
    WAYPOINT("NWID");
}

// clang-format off
/**
 * Sets the stateful registers for an inline write of a 32-bit value to a NOC destination.
 * This function is used to set up the state for \a noc_inline_dw_write_with_state, which will issue the actual
 * write request. The 32-bit value and part of the destination address can be set later in \a noc_inline_dw_write_with_state.
 * \a noc_inline_dw_write can be used instead if the state preservation is not
 * needed. Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                     | Description                                                | Type     | Valid Range                      | Required |
 * |------------------------------|------------------------------------------------------------|----------|----------------------------------|----------|
 * | addr                         | Encoding of the destination location (x,y)+address         | uint64_t | Results of \a get_noc_addr calls | True     |
 * | val                          | The value to be written                                    | uint32_t | Any uint32_t value               | False    |
 * | be                           | Byte-enable                                                | uint8_t  | 0x1-0xF                          | False    |
 * | cmd_buf                      | Command buffer to use for the transaction                  | uint8_t  | 0-3                              | False    |
 * | noc                          | NOC to use for the transaction                             | uint8_t  | 0 or 1                           | False    |
 * | vc                           | Virtual channel to use for the transaction                 | uint8_t  | 0-3 (Unicast VCs)                | False    |
 * | posted (template parameter)  | Whether the call is posted (i.e. ack requirement)          | bool     | true or false                    | False    |
 * | set_val (template parameter) | Whether to set the value for the write here                | bool     | true or false                    | False    |
 */
// clang-format on
template <bool posted = false, bool set_val = false>
FORCE_INLINE void noc_inline_dw_write_set_state(
    uint64_t addr,
    uint32_t val = 0,
    uint8_t be = 0xF,
    uint8_t cmd_buf = write_at_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NWIW");
    noc_fast_write_dw_inline_set_state<posted, set_val>(noc, cmd_buf, addr, be, vc, val);
    WAYPOINT("NWID");
}

// clang-format off
/**
 * Initiates an inline write of a 32-bit value to a NOC destination.
 * This function must be preceded by a call to \a noc_inline_dw_write_set_state.
 * This function is used to issue the actual write request after the state has been set up.
 * The 32-bit value and part of the destination address can also be set in this API (Only either hi or lo address should be getting updated).
 * \a noc_inline_dw_write can be used instead if the state preservation is not
 * needed. Also, see \a noc_async_write_barrier.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                                   | Description                                            | Data type | Valid range   | required |
 * |--------------------------------------------|--------------------------------------------------------|-----------|---------------|----------|
 * | val                                        | The value to be written                                | uint32_t  | Any uint32_t  | True     |
 * | addr                                       | The local address to write to (if not set in state)    | uint32_t  | 0..1MB        | False    |
 * | cmd_buf                                    | Command buffer to use for the transaction              | uint8_t   | 0-3           | False    |
 * | noc                                        | NOC to use for the transaction                         | uint8_t   | 0 or 1        | False    |
 * | update_addr_lo (template parameter)        | Whether to update the lower 32 bits of the address     | bool      | true or false | False    |
 * | update_counter (template parameter)        | Whether to update the write counters                   | bool      | true or false | False    |
 * | posted (template parameter)                | Whether the call is posted (i.e. ack requirement)      | bool      | true or false | False    |
 * | update_addr_hi (template parameter)        | Whether to update the upper 32 bits of the address     | bool      | true or false | False    |
 * | update_val (template parameter)            | Whether to set the value to be written                 | bool      | true or false | False    |
 * | dst_type (template parameter)              | Whether the write is targeting L1 or a Stream Register | InlineWriteDst| DEFAULT, L1, REG | False    |
 */
// clang-format on
template <
    bool update_addr_lo = false,
    bool update_counter = true,
    bool posted = false,
    bool update_addr_hi = false,
    bool update_val = false,
    InlineWriteDst dst_type = InlineWriteDst::DEFAULT>
FORCE_INLINE void noc_inline_dw_write_with_state(
    uint32_t val, uint32_t addr = 0, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758: always update counter for blackhole as a temporary
    // workaround for avoiding hangs in fabric router, as counters will be checked inside the
    // noc_fast_spoof_write_dw_inline, will remove this restriction once all inline write change to stream reg write.
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif

    WAYPOINT("NWIW");
    noc_fast_write_dw_inline_with_state<
        noc_mode,
        update_addr_lo,
        update_addr_hi,
        update_val,
        posted,
        update_counter_in_callee,
        dst_type>(noc, cmd_buf, val, addr);
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
 * | vc                         | Which VC to use for the transaction                              | uint8_t  | 0-3 (Unicast VCs)                | False    |
 * | posted (template argument) | Whether the call is posted or nonposted (i.e. needs to be acked) | uint32_t | true or false                    | False    |
 */
// clang-format on
template <bool posted = false>
FORCE_INLINE void noc_semaphore_inc(
    uint64_t addr, uint32_t incr, uint8_t noc_id = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::SEMAPHORE_INC, 0, addr, 0, vc, posted);

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
    // Posting heartbeat at this address is only needed for Wormhole
#if !defined(ARCH_BLACKHOLE)
    invalidate_l1_cache();
    volatile uint32_t* ptr = (volatile uint32_t*)(0x1C);
    heartbeat++;
    ptr[0] = 0xAABB0000 | (heartbeat & 0xFFFF);
#endif
}

// clang-format off
/**
 * Initiates an asynchronous read for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * Must first set the transaction id using \a noc_async_read_set_trid and the stateful registers
 * using an API such as \a noc_async_read_one_packet_set_state.
 *
 * Return value: None
 *
 * | Argument                            | Description                                    | Data type | Valid range   | required |
 * |-------------------------------------|------------------------------------------------|-----------|---------------|----------|
 * | src_base_addr                       | Base address of source location                | uint32_t  | 0..1MB        | True     |
 * | src_addr                            | Address in local L1 memory on source core      | uint32_t  | 0..1MB        | True     |
 * | dest_addr                           | Address in local L1 memory on destination core | uint32_t  | 0..1MB        | True     |
 * | trid                                | Transaction id for the transaction             | uint32_t  | 0x0 - 0xF     | False    |
 * | noc                                 | Which NOC to use for the transaction           | uint8_t   | 0 or 1        | False    |
 * | skip_ptr_update (template argument) | Whether to skip updating counters              | bool      | true or false | False    |
 */
// clang-format on
template <bool skip_ptr_update = false, bool skip_cmdbuf_chk = false>
FORCE_INLINE void noc_async_read_one_packet_with_state_with_trid(
    uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_WITH_STATE_AND_TRID, false);

    WAYPOINT("NRDW");
    ncrisc_noc_fast_read_with_transaction_id<noc_mode, skip_ptr_update, skip_cmdbuf_chk>(
        noc, read_cmd_buf, src_base_addr, src_addr, dest_addr, trid);
    WAYPOINT("NRDD");
}

// clang-format off
/**
 * Sets the transaction id for a noc read.
 *
 * Return value: None
 *
 * | Argument | Description                                        | Data type | Valid range | Required |
 * |----------|----------------------------------------------------|-----------|-------------|----------|
 * | trid     | Transaction id for the transaction                 | uint32_t  | 0x0 - 0xF   | False    |
 * | noc      | Which NOC to use for the transaction               | uint32_t  | 0 or 1      | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_read_set_trid(uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_SET_TRID, false);

    WAYPOINT("NSTW");
    ncrisc_noc_set_transaction_id(noc, read_cmd_buf, trid);
    WAYPOINT("NSTD");
}

// clang-format off
/**
 * Sets the transaction id for a noc write.
 *
 * Return value: None
 *
 * | Argument | Description                                        | Data type | Valid range | Required |
 * |----------|----------------------------------------------------|-----------|-------------|----------|
 * | trid     | Transaction id for the transaction                 | uint32_t  | 0x0 - 0xF   | False    |
 * | noc      | Which NOC to use for the transaction               | uint32_t  | 0 or 1      | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_write_set_trid(uint32_t trid = 0, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_SET_TRID, false);
    WAYPOINT("NWSW");
    ncrisc_noc_set_transaction_id(noc, write_cmd_buf, trid);
    WAYPOINT("NWSD");
}

// clang-format off
/**
 * This blocking call waits for all the outstanding enqueued read transactions
 * issued on the current Tensix core with the given transaction id to complete.
 * After returning from this call there will be no outstanding read transactions
 * with the given transaction id.
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | trid     | Transaction id for the transaction   | uint32_t | 0x0 - 0xF   | True     |
 * | noc      | Which NOC to use for the transaction | uint8_t  | 0 or 1      | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NBTW");
    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_WITH_TRID, false);
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NBTD");
}

// clang-format off
/**
 * Initiates an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size)
 * with a transaction ID. Refer to \a noc_async_write_one_packet for more details.
 *
 * Extra arguments:
 *
 * | Argument                           | Description                                        | Type     | Valid Range       | Required |
 * |------------------------------------|----------------------------------------------------|----------|-------------------|----------|
 * | trid                               | Transaction ID to be used for the write operation  | uint32_t | 0-15              | True     |
 * | cmd_buf                            | Command buffer to use for the transaction          | uint8_t  | 0-3               | False    |
 * | vc                                 | VC to use for the transaction                      | uint8_t  | 0-3 (Unicast VCs) | False    |
 * | update_counter (template argument) | Whether to update write counters or not            | bool     | true or false     | False    |
 * | posted (template argument)         | Whether the write is posted (i.e. ack requirement) | bool     | true or false     | False    |
 */
// clang-format on
template <bool update_counter = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr,
    uint32_t size,
    uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_local_l1_addr, dst_noc_addr, size, -1, posted);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, cmd_buf));

#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758: always update counter for blackhole as a temporary
    // workaround for avoiding hangs in fabric router, as counters will be checked inside the
    // noc_fast_spoof_write_dw_inline, will remove this restriction once all inline write change to stream reg write.
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif

    ncrisc_noc_fast_write<noc_mode, true /* use_trid */, update_counter_in_callee>(
        noc,
        cmd_buf,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        vc,
        false,  // mcast
        false,  // linked
        1,      // num_dests
        true,   // multicast_path_reserve
        posted,
        trid);
    WAYPOINT("NWPD");
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size)
 * to a specified destination node located at NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * noc_async_write_one_packet_with_trid_with_state, which will issue the actual
 * write request. \a noc_async_write_one_packet_with_trid can be used instead if the state preservation is not
 * needed.
 *
 * The destination node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                   | Description                                            | Data type | Valid range                      | required |
 * |----------------------------|--------------------------------------------------------|-----------|----------------------------------|----------|
 * | dst_noc_addr               | Encoding of the destination NOC location (x,y)+address | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | cmd_buf                    | Command buffer to use for the transaction              | uint8_t   | 0-3                              | False    |
 * | noc                        | NOC to use for the transaction                         | uint8_t   | 0 or 1                           | False    |
 * | vc                         | VC to use for the transaction                          | uint8_t   | 0-3 (Unicast VCs)                | False    |
 * | posted (template argument) | Whether the write is posted (i.e. ack requirement)     | bool      | true or false                    | False    |
 */
// clang-format on
template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_set_state(
    uint64_t dst_noc_addr,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NAWW");
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID_SET_STATE, 0, dst_noc_addr, 0, vc, posted);

    ncrisc_noc_write_set_state<posted, false /* one_packet */>(noc, cmd_buf, dst_noc_addr, 0 /* len_bytes */, vc);
    WAYPOINT("NAWD");
}

// clang-format off
/**
 * Initiates an asynchronous write for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size)
 * with a transaction ID. This function must be preceded by a call to
 * \a noc_async_write_one_packet_with_trid_set_state. This function is used to issue the actual
 * write request after the state has been set up.
 * \a noc_async_write_one_packet_with_trid can be used instead if the state preservation is not needed.
 * Also, see \a noc_async_write_barrier.
 *
 * Return value: None
 *
 * | Argument                           | Description                                        | Data type | Valid range   | required |
 * |------------------------------------|----------------------------------------------------|-----------|---------------|----------|
 * | src_local_l1_addr                  | Address in local L1 memory on source core          | uint32_t  | 0..1MB        | True     |
 * | dst_local_l1_addr                  | Address in local L1 memory on destination core     | uint32_t  | 0..1MB        | True     |
 * | size                               | Size of the data transfer in bytes                 | uint32_t  | 0..1MB        | True     |
 * | trid                               | Transaction ID to be used for the transaction      | uint32_t  | 0-15          | True     |
 * | cmd_buf                            | Command buffer to use for the transaction          | uint8_t   | 0-3           | False    |
 * | noc                                | NOC to use for the transaction                     | uint8_t   | 0 or 1        | False    |
 * | update_counter (template argument) | Whether to update write counters or not            | bool      | true or false | False    |
 * | posted (template argument)         | Whether the write is posted (i.e. ack requirement) | bool      | true or false | False    |
 */
// clang-format on
template <bool update_counter = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_with_state(
    uint32_t src_local_l1_addr,
    uint32_t dst_local_l1_addr,
    uint32_t size,
    uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index) {
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID_WITH_STATE, src_local_l1_addr, 0ull, size, -1, posted);

    // In order to sanitize, need to grab full noc addr + xfer size from state.
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_STATE(noc, dst_local_l1_addr, src_local_l1_addr, size);

#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758: always update counter for blackhole as a temporary
    // workaround for avoiding hangs in fabric router, as counters will be checked inside the
    // noc_fast_spoof_write_dw_inline, will remove this restriction once all inline write change to stream reg write.
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif

    WAYPOINT("NWPW");
    ncrisc_noc_set_transaction_id(noc, cmd_buf, trid);
    ncrisc_noc_write_with_state<noc_mode, posted, update_counter_in_callee>(
        noc, cmd_buf, src_local_l1_addr, dst_local_l1_addr, size);
    WAYPOINT("NWPD");
}

// clang-format off
/**
 * This blocking call waits for all the outstanding enqueued write transactions
 * issued on the current Tensix core with the given transaction id to complete.
 * After returning from this call there will be no outstanding write transactions
 * with the given transaction id.
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | trid     | Transaction id for the transaction   | uint32_t | 0x0 - 0xF   | True     |
 * | noc      | Which NOC to use for the transaction | uint8_t  | 0 or 1      | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_write_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NWTW");
    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NWTD");
}

// clang-format off
/**
 * This blocking call waits for all outstanding enqueued write transactions
 * with the given transaction id to depart, but will not wait
 * for them to complete.
 *
 * Return value: None
 *
 * | Argument | Description                          | Type     | Valid Range | Required |
 * |----------|--------------------------------------|----------|-------------|----------|
 * | trid     | Transaction id for the transaction   | uint32_t | 0x0 - 0xF   | True     |
 * | noc      | Which NOC to use for the transaction | uint8_t  | 0 or 1      | False    |
 */
// clang-format on
FORCE_INLINE
void noc_async_write_flushed_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_FLUSH_WITH_TRID, false);
    WAYPOINT("NFTW");
    while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NFTD");
}

// clang-format off
/**
 * This resets the barrier counter for a given transaction id on a given NOC using a mask.
 * Only the N bits up to the number of transaction ids are used.
 *
 * Return value: None
 *
 * | Argument | Description                               | Type     | Valid Range      | Required |
 * |----------|-------------------------------------------|----------|------------------|----------|
 * | id_mask  | Transaction id mask for the transaction   | uint32_t | 0x0 - 0xFFFFFFFF | False    |
 * | noc      | Which NOC to use for the transaction      | uint8_t  | 0 or 1           | False    |
 */
// clang-format on
FORCE_INLINE
void reset_noc_trid_barrier_counter(uint32_t id_mask = NOC_CLEAR_OUTSTANDING_REQ_MASK, uint32_t noc = noc_index) {
    noc_clear_outstanding_req_cnt(noc, id_mask);
}
