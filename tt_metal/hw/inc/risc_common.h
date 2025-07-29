// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _RISC_COMMON_H_
#define _RISC_COMMON_H_

#include <stdint.h>

#include <cstdint>

#include "eth_l1_address_map.h"
#include "limits.h"
#include "mod_div_lib.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "stream_io_map.h"
#include "tensix.h"

#define NOC_X(x) NOC_0_X(noc_index, noc_size_x, (x))
#define NOC_Y(y) NOC_0_Y(noc_index, noc_size_y, (y))
#define DYNAMIC_NOC_X(noc, x) NOC_0_X(noc, noc_size_x, (x))
#define DYNAMIC_NOC_Y(noc, y) NOC_0_Y(noc, noc_size_y, (y))
#define NOC_X_PHYS_COORD(x) NOC_0_X_PHYS_COORD(noc_index, noc_size_x, x)
#define NOC_Y_PHYS_COORD(y) NOC_0_Y_PHYS_COORD(noc_index, noc_size_y, y)

#define TILE_WORD_2_BIT ((256 + 64 + 32) >> 4)
#define TILE_WORD_4_BIT ((512 + 64 + 32) >> 4)
#define TILE_WORD_8_BIT ((32 * 32 * 1 + 64 + 32) >> 4)
#define TILE_WORD_16_BIT ((32 * 32 * 2 + 32) >> 4)
#define TILE_WORD_32_BIT ((32 * 32 * 4 + 32) >> 4)

const uint32_t STREAM_RESTART_CHECK_MASK = (0x1 << 3) - 1;

const uint32_t MAX_TILES_PER_PHASE = 2048;

// These values are defined in each core type's FW .cc file

// Virtual X coordinate
extern uint8_t my_x[NUM_NOCS];

// Virtual Y coordinate
extern uint8_t my_y[NUM_NOCS];

inline void WRITE_REG(uint32_t addr, uint32_t val) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    ptr[0] = val;
}

inline uint32_t READ_REG(uint32_t addr) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    return ptr[0];
}

inline uint32_t dram_io_incr_ptr(uint32_t curr_ptr, uint32_t incr, uint32_t buf_size_q_slots) {
    uint32_t next_ptr = curr_ptr + incr;
    uint32_t double_buf_size_q_slots = 2 * buf_size_q_slots;
    if (next_ptr >= double_buf_size_q_slots) {
        next_ptr -= double_buf_size_q_slots;
    }
    return next_ptr;
}

inline __attribute__((always_inline)) uint32_t dram_io_empty(uint32_t rd_ptr, uint32_t wr_ptr) {
    return (rd_ptr == wr_ptr);
}

inline __attribute__((always_inline)) uint32_t
dram_io_local_empty(uint32_t local_rd_ptr, uint32_t rd_ptr, uint32_t wr_ptr) {
    if (rd_ptr == wr_ptr) {
        return true;
    }

    uint32_t case1 = rd_ptr < wr_ptr && (local_rd_ptr < rd_ptr || local_rd_ptr >= wr_ptr);
    uint32_t case2 = rd_ptr > wr_ptr && wr_ptr <= local_rd_ptr && local_rd_ptr < rd_ptr;

    return case1 || case2;
}

inline uint32_t dram_io_full(uint32_t rd_ptr, uint32_t wr_ptr, uint32_t buf_size_q_slots) {
    uint32_t wr_ptr_reduced_by_q_slots = wr_ptr - buf_size_q_slots;
    uint32_t rd_ptr_reduced_by_q_slots = rd_ptr - buf_size_q_slots;
    uint32_t case1 = (wr_ptr_reduced_by_q_slots == rd_ptr);
    uint32_t case2 = (rd_ptr_reduced_by_q_slots == wr_ptr);
    return case1 || case2;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_inc_wrap(uint32_t buf_ptr, uint32_t inc, uint32_t buf_size) {
    uint32_t result = buf_ptr + inc;
    if (result >= buf_size) {
        result -= buf_size;
    }
    return result;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_dec_wrap(uint32_t buf_ptr, uint32_t dec, uint32_t buf_size) {
    uint32_t result = buf_ptr;
    if (dec > result) {
        result += buf_size;
    }
    result -= dec;
    return result;
}

// This definition of reg_read conflicts with the one in
// tt_metal/third_party/tt_llk_wormhole_b0/common/inc/ckernel.h, which trisc
// kernels bring into the global namespace using "using namespace ckernel".
#if !defined(COMPILE_FOR_TRISC)  // BRISC, NCRISC, ERISC, IERISC
inline __attribute__((always_inline)) uint32_t reg_read(uint32_t addr) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(addr);
    return p_reg[0];
}
#endif

inline void assert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    uint32_t trisc_reset_mask = RISCV_SOFT_RESET_0_TRISCS;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 | trisc_reset_mask);
}

inline void deassert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    uint32_t trisc_reset_mask = RISCV_SOFT_RESET_0_TRISCS;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 & ~trisc_reset_mask);
}

inline void deassert_all_reset() { WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NONE); }

inline void assert_just_ncrisc_reset() { WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NCRISC); }

inline uint32_t special_mult(uint32_t a, uint32_t special_b) {
    if (special_b == TILE_WORD_8_BIT) {
        return a * TILE_WORD_8_BIT;
    } else if (special_b == TILE_WORD_16_BIT) {
        return a * TILE_WORD_16_BIT;
    } else if (special_b == TILE_WORD_4_BIT) {
        return a * TILE_WORD_4_BIT;
    } else if (special_b == TILE_WORD_2_BIT) {
        return a * TILE_WORD_2_BIT;
    } else if (special_b == TILE_WORD_32_BIT) {
        return a * TILE_WORD_32_BIT;
    }

    while (true);
    return 0;
}

// Invalidates Blackhole's entire L1 cache
// Blackhole L1 cache is a small write-through cache (4x16B L1 lines). The cache covers all of L1 (no
// MMU or range registers).
//  Writing an address on one proc and reading it from another proc only requires the reader to invalidate.
//  Need to invalidate any address written by noc that may have been previously read by riscv
inline __attribute__((always_inline)) void invalidate_l1_cache() {
#if defined(ARCH_BLACKHOLE) && !defined(DISABLE_L1_DATA_CACHE)
    asm("fence");
#endif
}

// risc_init function isn't required for TRISCS
#if !defined(COMPILE_FOR_TRISC)  // BRISC, NCRISC, ERISC, IERISC
#include "noc_nonblocking_api.h"

inline void risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = MY_NOC_ENCODING(n);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}
#endif  // !defined(COMPILE_FOR_TRISC)

// Helper function to wait for a specified number of cycles, safe to call in erisc kernels.
#if defined(COMPILE_FOR_ERISC)
#include "erisc.h"
#endif
inline void riscv_wait(uint32_t cycles) {
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock = 0;
    do {
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
}

// Flush i$ on ethernet riscs
inline __attribute__((always_inline)) void flush_erisc_icache() {
#ifdef ARCH_BLACKHOLE
#pragma GCC unroll 2048
    for (int i = 0; i < 2048; i++) {
        asm("nop");
    }
#else
#pragma GCC unroll 128
    for (int i = 0; i < 128; i++) {
        asm("nop");
    }
#endif
}

#endif
