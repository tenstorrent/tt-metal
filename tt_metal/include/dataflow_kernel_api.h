#pragma once

#include <stdint.h>

#include "dataflow_internals.h"
// #include "debug_print.h"

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

/**
 * Returns the value of a constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateDataMovementKernel, CreateComputeKernel calls.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | The index of the argument          | uint32_t              | 0 to 31     | True     |
 */
#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_##arg_idx

int __multiply(int n, int m) {
    int res = 0, count = 0;
    while (m) {
        if ((m & 1) == 1)
            res += (n << count);
        count++;
        m >>= 1;
    }
    return res;
}

int __min(int a, int b) {
    if (a < b)
        return a;
    else
        return b;
}

/**
 * \private
 */

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
    return *((volatile T*)(dataflow_internal::get_arg_addr(arg_idx)));
}

// replicated from ckernels_defs.h, which are currently not included in BRISC / NCRISC builds
// TODO: look into ckernels_defs.h included in NCRISC/BRISC builds
inline __attribute__((always_inline)) constexpr static std::int32_t GET_L1_TILE_SIZE(uint format) {
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::Float16_b): return ((2048 >> 4));
        case ((uint8_t)DataFormat::Float16): return ((2048 >> 4));

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
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << 11);
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << 10) + (index << 6));
    };
}

#ifdef DATA_FORMATS_DEFINED
// this API is used by both the reader and writer side of the CB
// it uses unpack_src_format, but because unpack_src_format == pack_dst_format for a given CB,
// we can use either.
// TODO: this can be made constexpr?
inline std::int32_t get_tile_size(const std::int32_t operand) {
    std::uint32_t input = operand;

    // L1 16B words
    std::uint32_t num_words = GET_L1_TILE_SIZE((uint)unpack_src_format[input]);

    // return bytes
    return num_words << 4;
}
#endif  // DATA_FORMATS_DEFINED

namespace dataflow {

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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     | | num_tiles | The number of
 * tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that
 * fit into the CB) | True     |
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
    num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[input]);  // this doesn't work
#else
    // temp workaround for input=0,1,2 (likely low-perf due to conditionals)
    if (input == 0) {
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[0]);
    } else if (input == 1) {
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[1]);
    } else if (input == 2) {
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[2]);
    } else {
        // fallback to the format of input 0 for inputs > 2
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[0]);
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
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     | | num_tiles | The number of
 * tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that
 * fit into the CB) | True     |
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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     |
 */
inline __attribute__((always_inline)) uint32_t get_write_ptr(std::int32_t operand) {
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
 * | Argument  | Description                          | Type     | Valid Range | Required |
 * |-----------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------|----------|
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     |
 */
inline __attribute__((always_inline)) uint32_t get_read_ptr(std::int32_t operand) {
    std::uint32_t output = operand;

    // return byte address (fifo_rd_ptr is 16B address)
    std::uint32_t rd_ptr_bytes = cb_read_interface[output].fifo_rd_ptr << 4;
    return rd_ptr_bytes;
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
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     | | num_tiles | The number of free
 * tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit
 * into the CB) |          |
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
        std::uint16_t tiles_acked = (std::uint16_t)reg_read_barrier((std::uint32_t)tiles_acked_ptr);
        std::uint16_t free_space_tiles_wrap =
            cb_write_interface[input].fifo_size_tiles - (tiles_received - tiles_acked);
        free_space_tiles = (std::int32_t)free_space_tiles_wrap;
    } while (free_space_tiles < num_tiles);
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
 * | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31 | True     | | num_tiles | The number of
 * tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that
 * fit into the CB) |          |
 * */
FORCE_INLINE
void cb_wait_front(std::int32_t operand, std::int32_t num_tiles) {
    // std::uint32_t output = operand_to_output_index(operand);
    std::uint32_t output = operand;

    volatile std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    // "tiles_poppped" doesn't change while we wait for tiles to be pushed to CB
    std::uint16_t tiles_acked = tiles_acked_ptr[0];

    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;
    std::uint16_t tiles_received;
    std::uint16_t num_tiles_recv;

    do {
        tiles_received = (std::uint16_t)reg_read_barrier((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}

// NOC transfers

// simple APIs

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return NOC_XY_ADDR(NOC_X(noc_x), NOC_Y(noc_y), addr);
}

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    uint32_t page_size;          // Num bytes in page.

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;
        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr =
                mulsi3(udivsi3_const_divisor<NUM_DRAM_BANKS>(id), this->page_size) + this->bank_base_address + offset;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#else
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = mulsi3(id >> LOG_BASE_2_OF_NUM_DRAM_BANKS, this->page_size) + this->bank_base_address + offset;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#endif
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = mulsi3(id >> LOG_BASE_2_OF_NUM_L1_BANKS, this->page_size) + this->bank_base_address + offset;
            addr += bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = dataflow_internal::get_noc_addr_helper(noc_x, noc_y, addr);
        return noc_addr;
    }
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;  // WARNING: This struct is used for optimized get_noc_addr in which case
                                             // you know that bank_unit_size is a power of 2

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id) const {
        // So far, only using this for DRAM, but will eventually generalize to allow usage in L1 as well
        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;

#ifdef TEMP_DEBUG2
#endif
        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr =
                (udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << this->log_base_2_of_page_size) + this->bank_base_address;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#else
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#endif
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = ((id >> LOG_BASE_2_OF_NUM_L1_BANKS) << this->log_base_2_of_page_size) + this->bank_base_address;
            addr += bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = dataflow_internal::get_noc_addr_helper(noc_x, noc_y, addr);
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
        uint32_t addr;
        uint32_t noc_x;
        uint32_t noc_y;
        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                   this->bank_base_address + offset;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#else
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                   this->bank_base_address + offset;
            addr += bank_to_dram_offset[bank_id];
            noc_x = dram_bank_to_noc_x[bank_id];
            noc_y = dram_bank_to_noc_y[bank_id];
#endif
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) +
                   this->bank_base_address + offset;
            addr += bank_to_l1_offset[bank_id];
            noc_x = l1_bank_to_noc_x[bank_id];
            noc_y = l1_bank_to_noc_y[bank_id];
        }

        uint64_t noc_addr = dataflow_internal::get_noc_addr_helper(noc_x, noc_y, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_tile(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t src_addr;
        uint32_t src_noc_xy;

        if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                       this->bank_base_address + offset;
            src_addr += bank_to_dram_offset[bank_id];
            src_noc_xy = dram_bank_to_noc_xy[bank_id];
#else
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                       this->bank_base_address + offset;
            src_addr += bank_to_dram_offset[bank_id];
            src_noc_xy = dram_bank_to_noc_xy[bank_id];
#endif
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            src_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) +
                       this->bank_base_address + offset;
            src_addr += bank_to_l1_offset[bank_id];
            src_noc_xy = l1_bank_to_noc_xy[bank_id];
        }

        while (!ncrisc_noc_fast_read_ok(loading_noc, NCRISC_RD_CMD_BUF))
            ;

        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[loading_noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr) const {
        uint32_t dest_addr;
        uint32_t dest_noc_xy;

        if constexpr (DRAM) {
            uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
            dest_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, udivsi3_const_divisor<NUM_DRAM_BANKS>(id)) +
                        this->bank_base_address;
#else
            dest_addr = MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) +
                        this->bank_base_address;
            dest_addr += bank_to_dram_offset[bank_id];
#endif
            dest_noc_xy = dram_bank_to_noc_xy[bank_id];
        } else {
            uint32_t bank_id = id & (NUM_L1_BANKS - 1);
            dest_addr =
                MUL_WITH_TILE_SIZE((uint)this->data_format, id >> LOG_BASE_2_OF_NUM_L1_BANKS) + this->bank_base_address;
            dest_addr += bank_to_l1_offset[bank_id];
            dest_noc_xy = l1_bank_to_noc_xy[bank_id];
        }

        while (!ncrisc_noc_fast_write_ok(loading_noc, NCRISC_WR_REG_CMD_BUF))
            ;
        uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(loading_noc, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[loading_noc] += 1;
        noc_nonposted_writes_acked[loading_noc] += 1;  // num_dests
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
FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id);
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

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t addr) {
    /*
        Get an encoding which contains the address in L1 on the current core that you want to
        read from/write to via the noc
    */
    return NOC_XY_ADDR(my_x[loading_noc], my_y[loading_noc], addr);
}

FORCE_INLINE
std::uint64_t get_noc_addr_rm(
    uint32_t row, uint32_t col, uint32_t bank_base_address, uint32_t num_used_banks, uint32_t W) {
    uint32_t bank_id = row & (num_used_banks - 1);
    uint32_t dram_x = dram_bank_to_noc_x[bank_id];
    uint32_t dram_y = dram_bank_to_noc_y[bank_id];
    // >>3 is because of 8 banks
    // TODO(AP): replace multiply with increments
    uint32_t dram_addr = bank_base_address + (__multiply(row >> 3, (W << 1))) + (col << 1);
    std::uint64_t noc_addr = get_noc_addr(dram_x, dram_y, dram_addr);
    return noc_addr;
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
 * | src_noc_addr      | Encoding of the source DRAM location (x,y)+address | uint64_t  | DOX-TODO(insert ref to explain
 * valid coords) | Yes      | | dst_local_l1_addr | Address in local L1 memory                         | uint32_t  |
 * 0..1MB                                   | Yes      | | size              | Size of data transfer in bytes | uint32_t
 * | 0..1MB                                   | Yes      |
 */
FORCE_INLINE
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF, src_noc_addr, dst_local_l1_addr, size);
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
 * | src_local_l1_addr | Source address in local L1 memory                       | uint32_t | 0..1MB | True     | |
 * dst_noc_addr      | Encoding of the destination DRAM location (x,y)+address | uint64_t | DOX-TODO(insert a reference
 * to what constitutes valid coords) | True     | | size              | Size of data transfer in bytes | uint32_t |
 * 0..1MB                                                    | True     |
 */
FORCE_INLINE
void noc_async_write(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size) {
    ncrisc_noc_fast_write_any_len(
        loading_noc,
        NCRISC_WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        NOC_UNICAST_WRITE_VC,
        false,
        false,
        1);
}

template <bool DRAM>
FORCE_INLINE void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGenFast<DRAM>& s, std::uint32_t src_local_l1_addr) {
    s.noc_async_write_tile(id, src_local_l1_addr);
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
 * | Argument               | Description                                                              | Type     |
 * Valid Range                                                   | Required |
 * |------------------------|--------------------------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | src_local_l1_addr      | Source address in local L1 memory                                        | uint32_t |
 * 0..1MB                                                        | True     | | dst_noc_addr_multicast | Encoding of the
 * destinations nodes (x_start,y_start,x_end,y_end)+address | uint64_t | DOX-TODO(insert a reference to what constitutes
 * valid coords) | True     | | size                   | Size of data transfer in bytes | uint32_t | 0..1MB | True     |
 * | num_dests              | Number of destinations that the multicast source is targetting           | uint32_t |
 * 0..119                                                        | True     |
 */
FORCE_INLINE
void noc_async_write_multicast(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests) {
    ncrisc_noc_fast_write_any_len(
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
 * This blocking call waits for all the outstanding enqueued *noc_async_read*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_read* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_read_barrier() {
    while (!ncrisc_noc_reads_flushed(loading_noc))
        ;
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
void noc_async_write_barrier() {
    while (!ncrisc_noc_nonposted_writes_flushed(loading_noc))
        ;
}

// Command queue APIs
FORCE_INLINE
void cq_wait_front() {
    while (cq_read_interface.fifo_rd_ptr == get_cq_write_ptr()[0] and
           cq_read_interface.fifo_rd_toggle == get_cq_write_toggle()[0])
        ;
}

FORCE_INLINE
void notify_host_of_cq_read_pointer() {
    // These are the PCIE core coordinates
    u64 pcie_address = get_noc_addr(0, 4, HOST_CQ_READ_PTR);  // For now, we are writing to host hugepages at offset
                                                              // 0 (nothing else currently writing to it)
    u32 rd_ptr = cq_read_interface.fifo_rd_ptr;
    volatile u32* rd_ptr_addr = get_cq_read_ptr();
    rd_ptr_addr[0] = rd_ptr;
    noc_async_write(CQ_READ_PTR, pcie_address, 4);
    noc_async_write_barrier();
}

FORCE_INLINE
void notify_host_of_cq_read_toggle() {
    u64 pcie_address = get_noc_addr(0, 4, HOST_CQ_READ_TOGGLE_PTR);  // For now, we are writing to host hugepages at
                                                                     // offset 0 (nothing else currently writing to it)
    cq_read_interface.fifo_rd_toggle = not cq_read_interface.fifo_rd_toggle;
    volatile u32* rd_toggle_ptr = get_cq_read_toggle();
    rd_toggle_ptr[0] = cq_read_interface.fifo_rd_toggle;

    noc_async_write(CQ_READ_TOGGLE, pcie_address, 4);
    noc_async_write_barrier();
}

FORCE_INLINE
void cq_pop_front(u32 cmd_size_B) {
    // First part of equation aligns to nearest multiple of 32, and then we shift to make it a 16B addr. Both
    // host and device are consistent in updating their pointers in this way, so they won't get out of sync. The
    // alignment is necessary because we can only read/write from/to 32B aligned addrs in host<->dev communication.
    u32 cmd_size_16B = (((cmd_size_B - 1) | 31) + 1) >> 4;
    cq_read_interface.fifo_rd_ptr += cmd_size_16B;

    notify_host_of_cq_read_pointer();
}

}  // namespace dataflow
