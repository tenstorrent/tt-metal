// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file llk_intf_api.hpp
 * @brief Low-Level Kernel (LLK) Interface API for tile counter and buffer management
 *
 * This file provides comprehensive API functions for managing LLK interface counters
 * and buffer operations in the overlay system. It includes both fast custom instruction
 * variants and standard memory-mapped register access functions.
 */

#pragma once
#include "overlay_reg.h"
#include "rocc_instructions.hpp"

#define LLK_REG_SIZE TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE
#define COUNTER_REG_SIZE TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE

/**
 * @enum llk_interface_e
 * @brief Enumeration of available LLK interfaces
 *
 * Defines the available Low-Level Kernel interfaces in the system.
 * Each interface can manage multiple tile counters independently.
 */
enum llk_interface_e {
    LLK_INTERFACE_0 = 0,
    LLK_INTERFACE_1 = 1,
    LLK_INTERFACE_2 = 2,
    LLK_INTERFACE_3 = 3,
    LLK_INTERFACE_SIZE = 4
};

/**
 * @enum llk_intf_counter_id
 * @brief Enumeration of LLK interface counter IDs
 *
 * Defines the available counter indices within each LLK interface.
 * Each interface supports up to 16 independent counters (0-15).
 */
enum llk_intf_counter_id {
    LLK_INTF_COUNTER_0 = 0,
    LLK_INTF_COUNTER_1 = 1,
    LLK_INTF_COUNTER_2 = 2,
    LLK_INTF_COUNTER_3 = 3,
    LLK_INTF_COUNTER_4 = 4,
    LLK_INTF_COUNTER_5 = 5,
    LLK_INTF_COUNTER_6 = 6,
    LLK_INTF_COUNTER_7 = 7,
    LLK_INTF_COUNTER_8 = 8,
    LLK_INTF_COUNTER_9 = 9,
    LLK_INTF_COUNTER_10 = 10,
    LLK_INTF_COUNTER_11 = 11,
    LLK_INTF_COUNTER_12 = 12,
    LLK_INTF_COUNTER_13 = 13,
    LLK_INTF_COUNTER_14 = 14,
    LLK_INTF_COUNTER_15 = 15,
    LLK_INTF_COUNTER_SIZE = 16
};

// Legacy(slow) LLK Interface
inline void llk_reg_write(uint64_t addr, uint64_t data) {
    LLK_INTF_WRITE(addr, data);
    asm volatile("fence" : : : "memory");
}

inline uint64_t llk_reg_read(uint64_t addr) {
    uint64_t data = LLK_INTF_READ(addr);
    asm volatile("fence" : : : "memory");
    return data;
}

// New (fast) LLK Interface
inline void fast_llk_reg_write(uint64_t addr, uint64_t data) { LLK_INTF_WRITE(addr, data); }

inline uint64_t fast_llk_reg_read(uint64_t addr) {
    uint64_t data = LLK_INTF_READ(addr);
    return data;
}

// -----------------------------------------------------------------------------
// Legacy(slow) interface API below.
// -----------------------------------------------------------------------------

inline __attribute__((always_inline)) void llk_intf_reset(uint64_t llk_if, uint64_t counter) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                           counter * COUNTER_REG_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_OFFSET)) = 1;
}

inline __attribute__((always_inline)) void llk_intf_inc_posted(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                           counter * COUNTER_REG_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET)) = inc;
}

inline __attribute__((always_inline)) void llk_intf_inc_acked(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                           counter * COUNTER_REG_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET)) = inc;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_occupancy(uint64_t llk_if, uint64_t counter) {
    return *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                                  llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
                                  TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET));
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_free_space(uint64_t llk_if, uint64_t counter) {
    return *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                                  llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
                                  TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET));
}

inline __attribute__((always_inline)) void llk_intf_set_capacity(uint64_t llk_if, uint64_t counter, uint64_t cap) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                           counter * COUNTER_REG_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET)) =
        cap;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_capacity(uint64_t llk_if, uint64_t counter) {
    return *((
        volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                            counter * COUNTER_REG_SIZE +
                            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET));
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_posted(uint64_t llk_if, uint64_t counter) {
    return *(
        (volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                             counter * COUNTER_REG_SIZE +
                             TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_OFFSET));
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_acked(uint64_t llk_if, uint64_t counter) {
    return *(
        (volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                             counter * COUNTER_REG_SIZE +
                             TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_OFFSET));
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_error(uint64_t llk_if, uint64_t counter) {
    return *(
        (volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
                             counter * COUNTER_REG_SIZE +
                             TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_OFFSET));
}

// -----------------------------------------------------------------------------
// Fast interface API below.
// Implements identical functions as above and expands to use all 4 LLK IFs
// Important: after bulk of these functions add memory fence
// -----------------------------------------------------------------------------

inline __attribute__((always_inline)) void fast_llk_intf_reset(uint64_t llk_if, uint64_t counter) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR,
        1);
}

inline __attribute__((always_inline)) void fast_llk_intf_inc_posted(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR,
        inc);
}

inline __attribute__((always_inline)) void fast_llk_intf_inc_acked(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR,
        inc);
}

inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_occupancy(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR);
}

inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_free_space(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR);
}

inline __attribute__((always_inline)) void fast_llk_intf_set_capacity(uint64_t llk_if, uint64_t counter, uint64_t cap) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR,
        cap);
}

inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_capacity(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR);
}

inline __attribute__((always_inline)) uint16_t fast_llk_intf_read_posted(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);
}

inline __attribute__((always_inline)) uint16_t fast_llk_intf_read_acked(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_ADDR);
}

inline __attribute__((always_inline)) uint16_t fast_llk_intf_read_error(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_ADDR);
}
