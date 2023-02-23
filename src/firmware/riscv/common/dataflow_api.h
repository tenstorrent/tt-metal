#pragma once

#if __has_include("chlkc_unpack_data_format.h")
    #include "chlkc_unpack_data_format.h"
    #include "chlkc_pack_data_format.h"
    #define DATA_FORMATS_DEFINED
#endif

#include <stdint.h>
#include "hostdevcommon/common_runtime_address_map.h"
#include "circular_buffer.h"

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

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4
// for GS is 8, need to set to a different value for WH and future architectures
#define NUM_DRAM_CHANNELS 8
// dram channel to x/y lookup tables
// TODO: these should be constexpr compile-time init'd, but it doesn't work on BRISC yet
uint32_t dram_channel_to_noc_x[NUM_DRAM_CHANNELS];
uint32_t dram_channel_to_noc_y[NUM_DRAM_CHANNELS];

// GS RISC-V RTL bug workaround (l1 reads followed by local mem reads causes a hang)
// in ncrisc.cc/brisc.cc: volatile uint32_t local_mem_barrier;
void write_to_local_mem_barrier(uint32_t data) {
    local_mem_barrier = data;
}

constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    return L1_ARG_BASE + (arg_idx<<2);
}

template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile T*)(get_arg_addr(arg_idx)));
}

#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_ ## arg_idx

void init_dram_channel_to_noc_coord_lookup_tables() {
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
    dram_channel_to_noc_x[0] = dram_channel_to_noc_x[1] = 1;
    dram_channel_to_noc_x[2] = dram_channel_to_noc_x[3] = 4;
    dram_channel_to_noc_x[4] = dram_channel_to_noc_x[5] = 7;
    dram_channel_to_noc_x[6] = dram_channel_to_noc_x[7] = 10;

    dram_channel_to_noc_y[0] = dram_channel_to_noc_y[2] = dram_channel_to_noc_y[4] = dram_channel_to_noc_y[6] = 0;
    dram_channel_to_noc_y[1] = dram_channel_to_noc_y[3] = dram_channel_to_noc_y[5] = dram_channel_to_noc_y[7] = 6;
}



// only BRISC to call this
void init_sync_registers() {

    volatile std::uint32_t* tiles_received_ptr;
    volatile std::uint32_t* tiles_acked_ptr;

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

// replicated from ckernels_defs.h, which are currently not included in BRISC / NCRISC builds
// TODO: look into ckernels_defs.h included in NCRISC/BRISC builds
constexpr static std::int32_t GET_L1_TILE_SIZE(uint format) {
    switch (format&0x1F) {
        case ((uint8_t)DataFormat::Float32): return ((4096>>4));
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return ((2048>>4));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((1024>>4)+(64>>4));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((512>>4)+(64>>4));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((256>>4)+(64>>4));
        default: return ((1024>>4)+(64>>4));
    };
}

#ifdef DATA_FORMATS_DEFINED
inline void cb_push_back(const std::int32_t operand, const std::int32_t num_tiles) {

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

inline void cb_pop_front(std::int32_t operand, std::int32_t num_tiles) {

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

inline uint32_t get_write_ptr(std::int32_t operand) {
    std::uint32_t input = operand;
    // return byte address (fifo_wr_ptr is 16B address)
    std::uint32_t wr_ptr_bytes = cb_write_interface[input].fifo_wr_ptr << 4;
    return wr_ptr_bytes;
}

inline uint32_t get_read_ptr(std::int32_t operand) {
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

inline void cb_reserve_back(std::int32_t operand, std::int32_t num_tiles) {
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

inline void cb_wait_front(std::int32_t operand, std::int32_t num_tiles) {
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

FORCE_INLINE
std::uint64_t get_noc_addr(
    uint32_t id, uint32_t bank_base_address, uint32_t num_used_banks,
    uint32_t log_base_2_of_num_used_banks, uint32_t log_base_2_of_bank_unit_size) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.
        bank_base_address: Base address for the whole tensor.
        num_used_banks: How many memory banks we are using.
        log_base_2_of_num_used_banks: log(num_used_banks).
        log_base_2_of_bank_unit_size: log(bank_unit_size).
    */
    uint32_t bank_id = id & (num_used_banks - 1);

    // So far, only using this for DRAM, but will eventually generalize to allow usage in L1 as well
    uint32_t dram_x = dram_channel_to_noc_x[bank_id];
    uint32_t dram_y = dram_channel_to_noc_y[bank_id];
    uint32_t dram_addr = ((id >> log_base_2_of_num_used_banks) << log_base_2_of_bank_unit_size) + bank_base_address;
    uint64_t noc_addr = get_noc_addr(dram_x, dram_y, dram_addr);
    return noc_addr;
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
 * This is the large detailed description that's why. Note these blocks start with two *s rather than 1.
 *
 * What the heck is going on.
 *
 */
FORCE_INLINE
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                                        src_noc_addr,
                                        dst_local_l1_addr,
                                        size);
}

FORCE_INLINE
void noc_async_write(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr,  std::uint32_t size) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr, size,
                            NOC_UNICAST_WRITE_VC, false, false, 1);
}

FORCE_INLINE
void noc_async_write_multicast(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t size, std::uint32_t num_dests) {
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr_multicast, size,
                            NOC_MULTICAST_WRITE_VC, true, false, num_dests);
}

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

FORCE_INLINE
void noc_async_read_barrier() {
    while (!ncrisc_noc_reads_flushed(loading_noc));
}

FORCE_INLINE
void noc_async_write_barrier()  {
    while (!ncrisc_noc_nonposted_writes_flushed(loading_noc));
}

FORCE_INLINE
void noc_semaphore_wait(volatile uint32_t* sem_addr, uint32_t val)  {
    while((*sem_addr) != val);
}

FORCE_INLINE
void noc_semaphore_set(volatile uint32_t* sem_addr, uint32_t val)  {
    // set semaphore value to val
    (*sem_addr) = val;
}

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
