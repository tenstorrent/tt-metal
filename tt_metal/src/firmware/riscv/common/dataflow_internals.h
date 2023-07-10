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
#include "debug_print.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"

extern uint8_t loading_noc;

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

extern CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];
extern CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
extern CQReadInterface cq_read_interface;

/**
 * \private
 */

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4

// dram channel to x/y lookup tables
// The number of banks is generated based off device we are running on --> controlled by allocator
extern uint8_t dram_bank_to_noc_x[NUM_DRAM_BANKS];
extern uint8_t dram_bank_to_noc_y[NUM_DRAM_BANKS];
extern uint32_t dram_bank_to_noc_xy[NUM_DRAM_BANKS];

extern uint8_t l1_bank_to_noc_x[NUM_L1_BANKS];
extern uint8_t l1_bank_to_noc_y[NUM_L1_BANKS];
extern uint32_t l1_bank_to_noc_xy[NUM_L1_BANKS];

namespace dataflow_internal {

// GS RISC-V RTL bug workaround (l1 reads followed by local mem reads causes a hang)
// in ncrisc.cc/brisc.cc: volatile uint32_t local_mem_barrier;
void write_to_local_mem_barrier(uint32_t data) { local_mem_barrier = data; }

constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    return L1_ARG_BASE + (arg_idx << 2);
}

void init_dram_bank_to_noc_coord_lookup_tables() {
    init_dram_bank_coords(dram_bank_to_noc_x, dram_bank_to_noc_y);
    for (uint8_t i = 0; i < NUM_DRAM_BANKS; i++) {
        dram_bank_to_noc_xy[i] = ((NOC_Y(dram_bank_to_noc_y[i]) << NOC_ADDR_NODE_ID_BITS) | NOC_X(dram_bank_to_noc_x[i])) << (NOC_ADDR_LOCAL_BITS - 32);
    }
}

void init_l1_bank_to_noc_coord_lookup_tables() {
    init_l1_bank_coords(l1_bank_to_noc_x, l1_bank_to_noc_y, bank_to_l1_offset);
    for (uint16_t i = 0; i < NUM_L1_BANKS; i++) {
        l1_bank_to_noc_xy[i] = ((NOC_Y(l1_bank_to_noc_y[i]) << NOC_ADDR_NODE_ID_BITS) | NOC_X(l1_bank_to_noc_x[i])) << (NOC_ADDR_LOCAL_BITS - 32);
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

        circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG;  // move by 3 uint32's
    }

    circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

    for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        std::uint32_t fifo_addr = circular_buffer_config_addr[0];
        std::uint32_t fifo_size = circular_buffer_config_addr[1];
        // std::uint32_t fifo_size_tiles = circular_buffer_config_addr[2]; // unused
        write_to_local_mem_barrier(fifo_size);

        cb_read_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
        cb_read_interface[cb_id].fifo_rd_ptr = fifo_addr;
        cb_read_interface[cb_id].fifo_size = fifo_size;

        circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG;  // move by 3 uint32's
    }
}

// Only the read interface is set up on the device... the write interface
// belongs to host
void setup_cq_read_write_interface() {
    uint fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4;  // The fifo starts after the pointer addresses
    uint fifo_size = ((1024 * 1024 * 1024) >> 4) - fifo_addr;

    cq_read_interface.fifo_limit = fifo_addr + fifo_size - 1;
    cq_read_interface.fifo_rd_ptr = fifo_addr;
    cq_read_interface.fifo_size = fifo_size;
}

inline void wait_for_sync_register_value(std::uint32_t addr, std::int32_t val) {
    volatile std::uint32_t* reg_ptr = (volatile std::uint32_t*)addr;
    std::int32_t reg_value;
    do {
        reg_value = reg_ptr[0];
    } while (reg_value != val);
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

FORCE_INLINE
void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr) {
    ncrisc_noc_fast_write_any_len(
        loading_noc,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        4 /* size in bytes */,
        NOC_UNICAST_WRITE_VC,
        false,
        false,
        1);
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
 * | Argument               | Description                                                              | Type     |
 * Valid Range                                               | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t |
 * 0..1MB                                                    | True     | | dst_noc_addr_multicast | Encoding of the
 * destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | DOX-TODO(insert a reference to what constitutes
 * valid coords) | True     | | num_dests              | Number of destinations that the multicast source is targetting
 * | uint32_t | 0..119                                                    | True     |
 */
FORCE_INLINE
void noc_semaphore_set_multicast(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast, std::uint32_t num_dests) {
    ncrisc_noc_fast_write_any_len(
        loading_noc,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4 /*size in bytes*/,
        NOC_MULTICAST_WRITE_VC,
        true,
        false,
        num_dests);
}

FORCE_INLINE
void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests) {
    ncrisc_noc_fast_write_any_len_loopback_src(
        loading_noc,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true,
        false,
        num_dests);
}

/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *dataflow_internal::noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        |
 * Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True |
 */
FORCE_INLINE
void noc_semaphore_wait(volatile uint32_t* sem_addr, uint32_t val) {
    while ((*sem_addr) != val)
        ;
}

/**
 * Sets the value of a local L1 memory address on the Tensix core executing
 * this function to a specific value. This L1 memory address is used as a
 * semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
 * *dataflow_internal::noc_semaphore_wait*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        |
 * Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | Value to set the semaphore to                                  | uint32_t | Any uint32_t value | True |
 */
FORCE_INLINE
void noc_semaphore_set(volatile uint32_t* sem_addr, uint32_t val) {
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
 * | Argument  | Description                                                    | Type     | Valid Range | Required |
 * |-----------|----------------------------------------------------------------|----------|-----------------------------------------------------------|----------|
 * | addr      | Encoding of the destination location (x,y)+address             | uint64_t | DOX-TODO(insert a reference
 * to what constitutes valid coords) | True     | | incr      | The value to increment by | uint32_t | Any uint32_t
 * value                                        | True     |
 */
FORCE_INLINE
void noc_semaphore_inc(uint64_t addr, uint32_t incr) {
    /*
    [REFER TO grayskull/noc/noc.h for the documentation of noc_atomic_increment()]
    Generic increment with 32-bit wrap.
  */
    noc_fast_atomic_increment(loading_noc, NCRISC_AT_CMD_BUF, addr, incr, 31 /*wrap*/, false /*linked*/);
}

// optimized NOC transfer APIs
inline void noc_fast_read(uint32_t src_addr, uint32_t dest_addr) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF))
        ;
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

inline void noc_fast_read_set_src_xy(uint64_t src_addr) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF))
        ;
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_addr >> 32);
}

inline void noc_fast_read_set_len(uint32_t len_bytes) {
    while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF))
        ;
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
}

inline void noc_fast_read_inc_num_issued(uint32_t num_issued) {
    // while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF));
    noc_reads_num_issued[loading_noc] += num_issued;
}

// a fast write that assumes a single-dest (ie unicast)
inline void noc_fast_write(uint32_t src_addr, uint64_t dest_addr) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF))
        ;
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

inline void noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF))
        ;
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) |
                             (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) | NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
}

inline void noc_fast_write_set_dst_xy(uint64_t dest_addr) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF))
        ;
    NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, dest_addr >> 32);
}

inline void noc_fast_write_set_len(uint32_t len_bytes) {
    while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_CMD_BUF))
        ;
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

} // namespace dataflow_internal
