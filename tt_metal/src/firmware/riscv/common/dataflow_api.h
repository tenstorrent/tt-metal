#pragma once

#if __has_include("chlkc_unpack_data_format.h")
    #include "chlkc_unpack_data_format.h"
    #include "chlkc_pack_data_format.h"
    #define DATA_FORMATS_DEFINED
#endif

#include <stdint.h>
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/bank_to_noc_coord_mapping.h"
// #include "frameworks/tt_dispatch/impl/command.hpp"
#include "circular_buffer.h"

#include "debug_print.h"
/*
 * This is a trick with Doxygen to force it to not expand the always_inline
 * attribute property. We turn on predefine-only expansion with MACRO_EXPANSION
 * and EXPAND_ONLY_PREDEF set to YES. Then, we send in __DOXYGEN__ and
 * FORCE_INLINE to remove it from the source entirely when fed in Doxygen.
 *
 * However, this should not affect what the actual source code declaration,
 * and functions that were declared such will still be alwyas_inline.
 */
#if __DOXYGEN__
    #define FORCE_INLINE
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

/** @file */

/**
 * \private
 */
CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];
CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];

CBReadInterface cq_read_interface;

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4
// for GS is 8, need to set to a different value for WH and future architectures
#define NUM_DRAM_BANKS 8
#define NUM_L1_BANKS 128
#define LOG_BASE_2_OF_NUM_DRAM_BANKS 3
#define LOG_BASE_2_OF_NUM_L1_BANKS 7

// dram channel to x/y lookup tables
// TODO: these should be constexpr compile-time init'd, but it doesn't work on BRISC yet
uint32_t dram_bank_to_noc_x[NUM_DRAM_BANKS];
uint32_t dram_bank_to_noc_y[NUM_DRAM_BANKS];
uint32_t dram_bank_to_noc_xy[NUM_DRAM_BANKS];

uint8_t shuffled_l1_bank_ids[NUM_L1_BANKS];

uint32_t l1_bank_to_noc_x[NUM_L1_BANKS];
uint32_t l1_bank_to_noc_y[NUM_L1_BANKS];
uint32_t l1_bank_to_noc_xy[NUM_L1_BANKS];

int32_t l1_bank_to_l1_offset[NUM_L1_BANKS];

// GS RISC-V RTL bug workaround (l1 reads followed by local mem reads causes a hang)
// in ncrisc.cc/brisc.cc: volatile uint32_t local_mem_barrier;
void write_to_local_mem_barrier(uint32_t data) {
    local_mem_barrier = data;
}

constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    return L1_ARG_BASE + (arg_idx<<2);
}

/**
 * Returns the value of an argument from kernel_args array provided during
 * kernel creation using CreateDataMovementKernel, CreateComputeKernel calls.
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
    return *((volatile T*)(get_arg_addr(arg_idx)));
}

/**
 * Returns the value of a constexpr argument from kernel_compile_time_args array provided during kernel creation using CreateDataMovementKernel, CreateComputeKernel calls.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | The index of the argument          | uint32_t              | 0 to 31     | True     |
 */
#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_ ## arg_idx

void init_dram_bank_to_noc_coord_lookup_tables() {
// this mapping is for GS
// TODO: generalize for other architectures
// Dram channel 0: 1, 0
// Dram channel 1: 1, 6
// Dram channel 2: 4, 0
// Dram channel 3: 4, 6
// Dram channel 4: 7, 0
// Dram channel 5: 7, 6
// Dram channel 6: 10, 0
// Dram channel 7: 10, 6
    dram_bank_to_noc_x[0] = dram_bank_to_noc_x[1] = 1;
    dram_bank_to_noc_x[2] = dram_bank_to_noc_x[3] = 4;
    dram_bank_to_noc_x[4] = dram_bank_to_noc_x[5] = 7;
    dram_bank_to_noc_x[6] = dram_bank_to_noc_x[7] = 10;

    dram_bank_to_noc_y[0] = dram_bank_to_noc_y[2] = dram_bank_to_noc_y[4] = dram_bank_to_noc_y[6] = 0;
    dram_bank_to_noc_y[1] = dram_bank_to_noc_y[3] = dram_bank_to_noc_y[5] = dram_bank_to_noc_y[7] = 6;

    dram_bank_to_noc_xy[0] = (NOC_Y(0) << NOC_ADDR_NODE_ID_BITS) | NOC_X(1);
    dram_bank_to_noc_xy[1] = (NOC_Y(6) << NOC_ADDR_NODE_ID_BITS) | NOC_X(1);
    dram_bank_to_noc_xy[2] = (NOC_Y(0) << NOC_ADDR_NODE_ID_BITS) | NOC_X(4);
    dram_bank_to_noc_xy[3] = (NOC_Y(6) << NOC_ADDR_NODE_ID_BITS) | NOC_X(4);
    dram_bank_to_noc_xy[4] = (NOC_Y(0) << NOC_ADDR_NODE_ID_BITS) | NOC_X(7);
    dram_bank_to_noc_xy[5] = (NOC_Y(6) << NOC_ADDR_NODE_ID_BITS) | NOC_X(7);
    dram_bank_to_noc_xy[6] = (NOC_Y(0) << NOC_ADDR_NODE_ID_BITS) | NOC_X(10);
    dram_bank_to_noc_xy[7] = (NOC_Y(6) << NOC_ADDR_NODE_ID_BITS) | NOC_X(10);
}

void init_l1_bank_to_noc_coord_lookup_tables() {
    int id = 0;
    int remapped_id;

    init_shuffled_l1_bank_id_mapping(shuffled_l1_bank_ids);

    // Single bank cores
    for (uint32_t y = 1; y < 11; y++) {
        if (y == 6) continue;
        for (uint32_t x = 1; x < 13; x++) {
            remapped_id = shuffled_l1_bank_ids[id];
            l1_bank_to_noc_x[remapped_id] = x;
            l1_bank_to_noc_y[remapped_id] = y;
            l1_bank_to_noc_xy[remapped_id] = (NOC_Y(y) << NOC_ADDR_NODE_ID_BITS) | NOC_X(x);
            l1_bank_to_l1_offset[remapped_id] = 0;
            id++;
        }
    }

    // Storage cores
    for (uint32_t x = 2; x < 13; x++) {
        if (x == 7) continue;
        remapped_id = shuffled_l1_bank_ids[id];
        l1_bank_to_noc_x[remapped_id] = x;
        l1_bank_to_noc_y[remapped_id] = 11;
        l1_bank_to_noc_xy[remapped_id] = (NOC_Y(11) << NOC_ADDR_NODE_ID_BITS) | NOC_X(x);
        l1_bank_to_l1_offset[remapped_id] = -512 * 1024; // Bank 0 of storage core allocated top down from 512KB
        id++;
        remapped_id = shuffled_l1_bank_ids[id];
        l1_bank_to_noc_x[remapped_id] = x;
        l1_bank_to_noc_y[remapped_id] = 11;
        l1_bank_to_noc_xy[remapped_id] = (NOC_Y(11) << NOC_ADDR_NODE_ID_BITS) | NOC_X(x);
        l1_bank_to_l1_offset[remapped_id] = 0; // Bank 1 of storage core allocated top down from 1MB, like all other worker cores
        id++;
    }
}

// only BRISC to call this
void init_sync_registers() {

    volatile uint* tiles_received_ptr;
    volatile uint* tiles_acked_ptr;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
      tiles_received_ptr = get_cb_tiles_received_ptr(operand);
      tiles_received_ptr[0] = 0;
      tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
      tiles_acked_ptr[0] = 0;
    }
}

// can be used on NCRICS and/or BRISC, as both can act as tile producers into Tensix
void setup_cb_read_write_interfaces() {

  volatile std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {

    // write_to_local_mem_barrier are needed on GS because of the RTL bug
    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
    std::uint32_t fifo_addr = circular_buffer_config_addr[0];
    std::uint32_t fifo_size = circular_buffer_config_addr[1];
    std::uint32_t fifo_size_tiles = circular_buffer_config_addr[2];
    write_to_local_mem_barrier(fifo_size_tiles);

    cb_write_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
    cb_write_interface[cb_id].fifo_wr_ptr = fifo_addr;
    cb_write_interface[cb_id].fifo_size = fifo_size;
    cb_write_interface[cb_id].fifo_size_tiles = fifo_size_tiles;

    circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
  }

  circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {

    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
    std::uint32_t fifo_addr = circular_buffer_config_addr[0];
    std::uint32_t fifo_size = circular_buffer_config_addr[1];
    //std::uint32_t fifo_size_tiles = circular_buffer_config_addr[2]; // unused
    write_to_local_mem_barrier(fifo_size);

    cb_read_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
    cb_read_interface[cb_id].fifo_rd_ptr = fifo_addr;
    cb_read_interface[cb_id].fifo_size = fifo_size;

    circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
  }
}


// Only the read interface is set up on the device... the write interface
// belongs to host
void setup_cq_read_write_interface() {
    uint fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4; // The fifo starts after the pointer addresses
    uint fifo_size = ((1024 * 1024 * 1024) >> 4) - fifo_addr;

    cq_read_interface.fifo_limit = fifo_addr + fifo_size - 1;
    cq_read_interface.fifo_rd_ptr = fifo_addr;
    cq_read_interface.fifo_size = fifo_size;

    // Setting up here rather than in init sync registers function
    // since these are not registers, rather they are L1 values
    // Read ptr
    get_cq_read_ptr()[0] = fifo_addr;

    // Write ptr
    get_cq_write_ptr()[0] = fifo_addr;
}

// replicated from ckernels_defs.h, which are currently not included in BRISC / NCRISC builds
// TODO: look into ckernels_defs.h included in NCRISC/BRISC builds
inline __attribute__((always_inline))
constexpr static std::int32_t GET_L1_TILE_SIZE(uint format) {
    switch (format&0x1F) {
        case ((uint8_t)DataFormat::Float16_b): return ((2048>>4));
        case ((uint8_t)DataFormat::Float16):   return ((2048>>4));

        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((1024>>4)+(64>>4));

        case ((uint8_t)DataFormat::Float32): return ((4096>>4));

        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((512>>4)+(64>>4));

        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((256>>4)+(64>>4));
        default: return ((1024>>4)+(64>>4));
    };
}

inline __attribute__((always_inline))
constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    switch (format&0x1F) {
        case ((uint8_t)DataFormat::Bfp8_b): return ((index<<10)+(index<<6));
        //Keep default as Bfp8?
        default: return ((index<<10)+(index<<6));
    };
}

#ifdef DATA_FORMATS_DEFINED
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
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_push_back(const std::int32_t operand, const std::int32_t num_tiles) {

    const std::uint32_t input = operand;
    std::uint32_t num_words;

    // FIXME: indexing into the array via "input" var doesn't work, it seems only this function is broken
    // on NCRISC, it may work on BRISC (tbd by running the reader on BRISC)
    // However, indexing via constants 0,1,2 works
    #if 1
    // TODO: this was fixed on NCRISC but may still be broken on BRISC
    num_words =  num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[input]); // this doesn't work
    #else
    // temp workaround for input=0,1,2 (likely low-perf due to conditionals)
    if (input == 0) {
        num_words =  num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[0]);
    } else if (input == 1) {
        num_words =  num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[1]);
    } else if (input == 2) {
        num_words =  num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[2]);
    } else {
        // fallback to the format of input 0 for inputs > 2
        num_words =  num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[0]);
    }
    #endif

    volatile std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    tiles_received_ptr[0] += num_tiles;

    cb_write_interface[input].fifo_wr_ptr += num_words;

    // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
    // producer always writes into contiguous memory, it cannot wrap
    if (cb_write_interface[input].fifo_wr_ptr > cb_write_interface[input].fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        cb_write_interface[input].fifo_wr_ptr -= cb_write_interface[input].fifo_size;
    }
}

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
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
 */
FORCE_INLINE
void cb_pop_front(std::int32_t operand, std::int32_t num_tiles) {

    volatile std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    tiles_acked_ptr[0] += num_tiles;

    std::uint32_t output = operand;

    std::uint32_t num_words = num_tiles * GET_L1_TILE_SIZE((uint)pack_dst_format[output]);

    cb_read_interface[output].fifo_rd_ptr += num_words;

    // this will basically reset fifo_rd_ptr to fifo_addr -- no other wrap is legal
    // consumer always reads from contiguous memory, it cannot wrap
    if (cb_read_interface[output].fifo_rd_ptr > cb_read_interface[output].fifo_limit) {
        // TODO: change this to fifo_wr_ptr
        cb_read_interface[output].fifo_rd_ptr -= cb_read_interface[output].fifo_size;
    }
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
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 */
inline __attribute__((always_inline))
uint32_t get_write_ptr(std::int32_t operand) {
    std::uint32_t input = operand;
    // return byte address (fifo_wr_ptr is 16B address)
    std::uint32_t wr_ptr_bytes = cb_write_interface[input].fifo_wr_ptr << 4;
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
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 */
inline __attribute__((always_inline))
uint32_t get_read_ptr(std::int32_t operand) {
    std::uint32_t output = operand;

    // return byte address (fifo_wr_ptr is 16B address)
    std::uint32_t rd_ptr_bytes = cb_read_interface[output].fifo_rd_ptr << 4;
    return rd_ptr_bytes;
}

inline void wait_for_sync_register_value(std::uint32_t addr, std::int32_t val) {
    volatile std::uint32_t* reg_ptr = (volatile std::uint32_t*) addr;
    std::int32_t reg_value;
    do {
        reg_value = reg_ptr[0];
    } while (reg_value != val);
}

/**
 * A blocking call that waits for the specified number of tiles to be free in the specified circular buffer. This call is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.
 *
 * CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of free tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 */
FORCE_INLINE
void cb_reserve_back(std::int32_t operand, std::int32_t num_tiles) {
    std::uint32_t input = operand;

    volatile std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t tiles_received = tiles_received_ptr[0];

    std::int32_t free_space_tiles;
    do {
        // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
        // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
        std::uint16_t tiles_acked = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_acked_ptr);
        std::uint16_t free_space_tiles_wrap = cb_write_interface[input].fifo_size_tiles - (tiles_received - tiles_acked);
        free_space_tiles = (std::int32_t) free_space_tiles_wrap;
    } while (free_space_tiles < num_tiles);
}

/**
 * A blocking call that waits for the specified number of tiles to be available in the specified circular buffer (CB). This call is used by the consumer of the CB to wait for the producer to fill the CB with at least the specfied number of tiles.
 * Important note: in case multiple calls of cb_wait_front(n) are issued without a paired cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles.
 * Example: 4 calls of cb_wait_front(8) followed by a cb_pop_front(32) would produce incorrect behavior. Instead 4 calls of cb_wait_front() waiting on 8, 16, 24, 32 tiles should be issued.
 *
 * Important note: number of tiles used in all cb_* calls must evenly divide the cb size and must be the same number in all cb_wait_front calls in the same kernel.
 * Example 1: cb_wait_front(32), cb_wait_front(40), cb_pop_front(32+8) tiles on a CB of size 64 would produce incorrect behavior.
 * Example 2: cb_wait_front(3) on a cb of size 32 would also produce incorrect behavior.
 * These limitations are due to performance optimizations in the CB implementation.
 *
 * Important note: CB total size must be an even multiple of the argument passed to this call.
 *
 * Return value: None
 *
 * | Argument  | Description                          | Type     | Valid Range                                                                                       | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
 * | num_tiles | The number of tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
 * */
FORCE_INLINE
void cb_wait_front(std::int32_t operand, std::int32_t num_tiles) {
    //std::uint32_t output = operand_to_output_index(operand);
    std::uint32_t output = operand;

    volatile std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    // "tiles_poppped" doesn't change while we wait for tiles to be pushed to CB
    std::uint16_t tiles_acked = tiles_acked_ptr[0];

    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;
    std::uint16_t tiles_received;
    std::uint16_t num_tiles_recv;

    do {
        tiles_received = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}

// NOC transfers

// simple APIs

FORCE_INLINE
std::uint64_t get_noc_multicast_addr(std::uint32_t noc_x_start, std::uint32_t noc_y_start, std::uint32_t noc_x_end, std::uint32_t noc_y_end, std::uint32_t addr) {
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
std::uint64_t get_noc_addr_helper(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return NOC_XY_ADDR(NOC_X(noc_x), NOC_Y(noc_y), addr);
}

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address; // Base address for the whole tensor.
    uint32_t page_size; // Num bytes in page.

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {

        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;
        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = mulsi3(id >> LOG_BASE_2_OF_NUM_DRAM_BANKS, this->page_size) + this->bank_base_address + offset;

            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = mulsi3(id >> LOG_BASE_2_OF_NUM_L1_BANKS, this->page_size) + this->bank_base_address + offset;
            addr += l1_bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_x, noc_y, addr);
        return noc_addr;
    }

};


template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size; // WARNING: This struct is used for optimized get_noc_addr in which case you know that bank_unit_size is a power of 2

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id) const {

        // So far, only using this for DRAM, but will eventually generalize to allow usage in L1 as well
        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;

        #ifdef TEMP_DEBUG2
        // DPRINT << this->bank_base_address << ENDL();
        #endif
        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address;
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address;
            addr += l1_bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_x, noc_y, addr);
        return noc_addr;
    }

};

template <bool DRAM>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address; // Base address for the whole tensor.
    uint32_t page_size; // Num bytes in bank unit.
    DataFormat data_format; // Dataformat

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;
        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) + this->bank_base_address + offset;
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) + this->bank_base_address + offset;
            addr += l1_bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = get_noc_addr_helper(noc_x, noc_y, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_tile(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t src_addr;
        uint32_t src_noc_xy;

        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) + this->bank_base_address + offset;
            src_noc_xy = dram_bank_to_noc_xy[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) + this->bank_base_address + offset;
            src_addr += l1_bank_to_l1_offset[bank_id];
            src_noc_xy = l1_bank_to_noc_xy[bank_id];
        }

        while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));

        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr); // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy); // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, this->page_size); // len_bytes
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[loading_noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr) const {
        uint32_t dest_addr;
        uint32_t dest_noc_xy;

        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            dest_addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) + this->bank_base_address;
            dest_noc_xy = dram_bank_to_noc_xy[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            dest_addr = MUL_WITH_TILE_SIZE((uint) this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) + this->bank_base_address;
            dest_addr += l1_bank_to_l1_offset[bank_id];
            dest_noc_xy = l1_bank_to_noc_xy[bank_id];
        }

        while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_REG_CMD_BUF));
        uint32_t noc_cmd_field =
          NOC_CMD_CPY | NOC_CMD_WR |
          NOC_CMD_VC_STATIC  |
          NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
          0x0 | // (linked ? NOC_CMD_VC_LINKED : 0x0)
          0x0 | // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
          NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, dest_addr); // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dest_noc_xy); // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE, this->page_size); // len_bytes
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[loading_noc] += 1;
        noc_nonposted_writes_acked[loading_noc] += 1; // num_dests
    }

};


template <bool DRAM>
FORCE_INLINE
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM> &s, uint32_t offset = 0) {
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
FORCE_INLINE
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM> &s) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    // DPRINT << s.bank_base_address << ',' << ' ' << uint(DRAM) << ENDL();
    return s.get_noc_addr(id);
}

template <bool DRAM>
FORCE_INLINE
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, uint32_t offset = 0) {
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

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t addr) {
    /*
        Get an encoding which contains the address in L1 on the current core that you want to
        read from/write to via the noc
    */
    return NOC_XY_ADDR(my_x[loading_noc], my_y[loading_noc], addr);
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
 * | Argument          | Description                                        | Data type | Valid range                              | required |
 * |-------------------|----------------------------------------------------|-----------|------------------------------------------|----------|
 * | src_noc_addr      | Encoding of the source DRAM location (x,y)+address | uint64_t  | DOX-TODO(insert ref to explain valid coords) | Yes      |
 * | dst_local_l1_addr | Address in local L1 memory                         | uint32_t  | 0..1MB                                   | Yes      |
 * | size              | Size of data transfer in bytes                     | uint32_t  | 0..1MB                                   | Yes      |
 */
FORCE_INLINE
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                                        src_noc_addr,
                                        dst_local_l1_addr,
                                        size);
}

template <bool DRAM>
FORCE_INLINE
void noc_async_read_tile(const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, std::uint32_t dst_local_l1_addr, uint32_t offset = 0) {
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
 * | Argument          | Description                                             | Type     | Valid Range                                               | Required |
 * |-------------------|---------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | src_local_l1_addr | Source address in local L1 memory                       | uint32_t | 0..1MB                                                    | True     |
 * | dst_noc_addr      | Encoding of the destination DRAM location (x,y)+address | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | size              | Size of data transfer in bytes                          | uint32_t | 0..1MB                                                    | True     |
 */
FORCE_INLINE
void noc_async_write(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr,  std::uint32_t size) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr, size,
                            NOC_UNICAST_WRITE_VC, false, false, 1);
}

template <bool DRAM>
FORCE_INLINE
void noc_async_write_tile(const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, std::uint32_t src_local_l1_addr) {
    s.noc_async_write_tile(id, src_local_l1_addr);
}

FORCE_INLINE
void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr, 4 /* size in bytes */,
                            NOC_UNICAST_WRITE_VC, false, false, 1);
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
 * | size                   | Size of data transfer in bytes                                           | uint32_t | 0..1MB                                                        | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t | 0..119                                                        | True     |
 */
FORCE_INLINE
void noc_async_write_multicast(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t size, std::uint32_t num_dests) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr_multicast, size,
                            NOC_MULTICAST_WRITE_VC, true, false, num_dests);
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
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t | 0..119                                                    | True     |
 */
FORCE_INLINE
void noc_semaphore_set_multicast(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr_multicast, 4 /*size in bytes*/,
                            NOC_MULTICAST_WRITE_VC, true, false, num_dests);
}

FORCE_INLINE
void noc_async_write_multicast_loopback_src(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t size, std::uint32_t num_dests) {
        ncrisc_noc_fast_write_any_len_loopback_src(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr_multicast, size,
                            NOC_MULTICAST_WRITE_VC, true, false, num_dests);
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
    while (!ncrisc_noc_reads_flushed(loading_noc));
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_writ*e
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_write* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_write_barrier()  {
    while (!ncrisc_noc_nonposted_writes_flushed(loading_noc));
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
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True     |
 */
FORCE_INLINE
void noc_semaphore_wait(volatile uint32_t* sem_addr, uint32_t val)  {
    while((*sem_addr) != val);
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
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True     |
 * | val       | Value to set the semaphore to                                  | uint32_t | Any uint32_t value | True     |
 */
FORCE_INLINE
void noc_semaphore_set(volatile uint32_t* sem_addr, uint32_t val)  {
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
 * | Argument  | Description                                                    | Type     | Valid Range                                               | Required |
 * |-----------|----------------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | addr      | Encoding of the destination location (x,y)+address             | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | incr      | The value to increment by                                      | uint32_t | Any uint32_t value                                        | True     |
 */
FORCE_INLINE
void noc_semaphore_inc(uint64_t addr, uint32_t incr){
    /*
    [REFER TO grayskull/noc/noc.h for the documentation of noc_atomic_increment()]
    Generic increment with 32-bit wrap.
  */
    noc_fast_atomic_increment(loading_noc, NCRISC_AT_CMD_BUF, addr, incr, 31 /*wrap*/, false /*linked*/);
}

// optimized NOC transfer APIs
inline void noc_fast_read(uint32_t src_addr, uint32_t dest_addr) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

inline void noc_fast_read_set_src_xy(uint64_t src_addr) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_addr >> 32);
}

inline void noc_fast_read_set_len(uint32_t len_bytes) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
}

inline void noc_fast_read_inc_num_issued(uint32_t num_issued) {
    // while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));
    noc_reads_num_issued[loading_noc] += num_issued;
}

// a fast write that assumes a single-dest (ie unicast)
inline void noc_fast_write(uint32_t src_addr, uint64_t dest_addr) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

inline void noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF));
    uint32_t noc_cmd_field =
      NOC_CMD_CPY | NOC_CMD_WR |
      NOC_CMD_VC_STATIC  |
      NOC_CMD_STATIC_VC(vc) |
      (linked ? NOC_CMD_VC_LINKED : 0x0) |
      (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
      NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
}

inline void noc_fast_write_set_dst_xy(uint64_t dest_addr) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, dest_addr >> 32);
}

inline void noc_fast_write_set_len(uint32_t len_bytes) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
}

inline void noc_fast_write_inc_num_dests(uint32_t num_issued) {
    // while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF));
    noc_nonposted_writes_num_issued[loading_noc] += num_issued;
    noc_nonposted_writes_acked[loading_noc] += num_issued;
}

inline void noc_prepare_deassert_reset_flag(uint32_t l1_addr) {
    reinterpret_cast<volatile uint32_t*>(l1_addr)[0] = uint32_t(TENSIX_DEASSERT_SOFT_RESET);
}

inline void noc_prepare_assert_reset_flag(uint32_t l1_addr) {
    reinterpret_cast<volatile uint32_t*>(l1_addr)[0] = uint32_t(TENSIX_ASSERT_SOFT_RESET);
}


// Command queue APIs
FORCE_INLINE
void cq_wait_front() {

    u32 fifo_wr_ptr;
    do {
        fifo_wr_ptr = get_cq_write_ptr()[0];
    } while (cq_read_interface.fifo_rd_ptr == fifo_wr_ptr);
}

FORCE_INLINE
void cq_pop_front(u32 cmd_size_16B) {
    cq_read_interface.fifo_rd_ptr += cmd_size_16B;

    if (cq_read_interface.fifo_rd_ptr > cq_read_interface.fifo_limit) {
        cq_read_interface.fifo_rd_ptr -= cq_read_interface.fifo_size;
    }

    uint32_t pcie_noc_x = NOC_X(0);
    uint32_t pcie_noc_y = NOC_Y(4); // These are the PCIE core coordinates
    uint64_t pcie_address =
        get_noc_addr(pcie_noc_x, pcie_noc_y, HOST_CQ_READ_PTR);  // For now, we are writing to host hugepages at offset 0 (nothing else currently writing to it)

    u32 rd_ptr = cq_read_interface.fifo_rd_ptr;
    volatile u32* rd_ptr_ptr = get_cq_read_ptr();

    rd_ptr_ptr[0] = rd_ptr;
    noc_async_write(u32(rd_ptr_ptr), pcie_address, 4);
    noc_async_write_barrier();
}
