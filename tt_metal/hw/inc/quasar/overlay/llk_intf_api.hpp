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

// -----------------------------------------------------------------------------
// Fast interface API below.
// Implements identical functions as bellow with fast_llk_reg_read and fast_llk_reg_write
// Important: after bulk of these functions add memory fence
// -----------------------------------------------------------------------------
/**
 * @brief Fast write to LLK register without memory fence
 *
 * Performs a high-performance write operation to an LLK register using
 * custom instruction without memory fence. Use when bulk operations
 * are performed and fence will be added later.
 *
 * @param addr Register address to write to
 * @param data Data value to write
 *
 * @note No memory fence - add fence after bulk operations
 */
inline void fast_llk_reg_write(uint64_t addr, uint64_t data) { LLK_INTF_WRITE(addr, data); }

/**
 * @brief Fast read from LLK register without memory fence
 *
 * Performs a high-performance read operation from an LLK register using
 * custom instruction without memory fence. Use when bulk operations
 * are performed and fence will be added later.
 *
 * @param addr Register address to read from
 * @return Data value read from the register
 *
 * @note No memory fence - add fence after bulk operations
 */
inline uint64_t fast_llk_reg_read(uint64_t addr) {
    uint64_t data = LLK_INTF_READ(addr);
    return data;
}

/**
 * @brief Reset specific LLK interface counter (fast version)
 *
 * Resets the specified counter in the given LLK interface using
 * fast register access without memory fence.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_reset(uint64_t llk_if, uint64_t counter) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR,
        1);
}

/**
 * @brief Increment posted counter (fast version)
 *
 * Increments the posted transaction counter for the specified
 * LLK interface and counter by the given amount.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @param inc Increment value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_inc_posted(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR,
        inc);
}

/**
 * @brief Read posted counter value (fast version)
 *
 * Reads the current posted transaction counter value for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Current posted counter value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint16_t fast_llk_intf_read_posted(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR);
}

/**
 * @brief Increment acknowledged counter (fast version)
 *
 * Increments the acknowledged transaction counter for the specified
 * LLK interface and counter by the given amount.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @param inc Increment value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_inc_acked(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR,
        inc);
}

/**
 * @brief Read acknowledged counter value (fast version)
 *
 * Reads the current acknowledged transaction counter value for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Current acknowledged counter value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint16_t fast_llk_intf_read_acked(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR);
}

/**
 * @brief Get buffer occupancy (fast version)
 *
 * Returns the current occupancy level of the buffer for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Current buffer occupancy
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_occupancy(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR);
}

/**
 * @brief Get buffer free space (fast version)
 *
 * Returns the current free space available in the buffer for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Current buffer free space
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_free_space(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR);
}

/**
 * @brief Set buffer capacity (fast version)
 *
 * Sets the total capacity of the buffer for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @param cap Buffer capacity value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_set_capacity(uint64_t llk_if, uint64_t counter, uint64_t cap) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR,
        cap);
}

/**
 * @brief Get buffer capacity (fast version)
 *
 * Returns the total capacity of the buffer for the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Buffer capacity value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_capacity(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR);
}

/**
 * @brief Set occupancy threshold (fast version)
 *
 * Sets the occupancy threshold for flow control on the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @param thresh Occupancy threshold value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_set_occupancy_thresh(
    uint64_t llk_if, uint64_t counter, uint64_t thresh) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_AVAIL_THRESHOLD_REG_ADDR,
        thresh);
}

/**
 * @brief Get occupancy threshold (fast version)
 *
 * Returns the occupancy threshold for flow control on the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Occupancy threshold value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_occupancy_thresh(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_AVAIL_THRESHOLD_REG_ADDR);
}

/**
 * @brief Set free space threshold (fast version)
 *
 * Sets the free space threshold for flow control on the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @param thresh Free space threshold value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) void fast_llk_intf_set_free_space_thresh(
    uint64_t llk_if, uint64_t counter, uint64_t thresh) {
    fast_llk_reg_write(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_FREE_THRESHOLD_REG_ADDR,
        thresh);
}

/**
 * @brief Get free space threshold (fast version)
 *
 * Returns the free space threshold for flow control on the specified
 * LLK interface and counter.
 *
 * @param llk_if LLK interface ID (0-3)
 * @param counter Counter ID (0-15)
 * @return Free space threshold value
 *
 * @note Uses fast register access - add memory fence after bulk operations
 */
inline __attribute__((always_inline)) uint64_t fast_llk_intf_get_free_space_thresh(uint64_t llk_if, uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
        counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_FREE_THRESHOLD_REG_ADDR);
}

// -----------------------------------------------------------------------------
// MMIO (slow) interface API below.
// -----------------------------------------------------------------------------

inline void llk_reg_write(uint64_t addr, uint64_t data) {
    LLK_INTF_WRITE(addr, data);
    asm volatile("fence" : : : "memory");
}
inline uint64_t llk_reg_read(uint64_t addr) {
    uint64_t data = LLK_INTF_READ(addr);
    asm volatile("fence" : : : "memory");
    return data;
}

inline __attribute__((always_inline)) void llk_intf_reset(uint64_t llk_if, uint64_t counter) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_OFFSET)) = 1;
}

inline __attribute__((always_inline)) void llk_intf_inc_posted(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET)) = inc;
}

inline __attribute__((always_inline)) void llk_intf_inc_acked(uint64_t llk_if, uint64_t counter, uint64_t inc) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET)) = inc;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_occupancy(uint64_t llk_if, uint64_t counter) {
    return *(
        (volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                             llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                             counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                             TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET));
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_free_space(uint64_t llk_if, uint64_t counter) {
    return *(
        (volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                             llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                             counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                             TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET));
}

inline __attribute__((always_inline)) void llk_intf_set_capacity(uint64_t llk_if, uint64_t counter, uint64_t cap) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET)) =
        cap;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_capacity(uint64_t llk_if, uint64_t counter) {
    return *((
        volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                            llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET));
}

inline __attribute__((always_inline)) void llk_intf_set_occupancy_thresh(
    uint64_t llk_if, uint64_t counter, uint64_t thresh) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_AVAIL_THRESHOLD_REG_OFFSET)) =
        thresh;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_occupancy_thresh(uint64_t llk_if, uint64_t counter) {
    return *((
        volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                            llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_AVAIL_THRESHOLD_REG_OFFSET));
}

inline __attribute__((always_inline)) void llk_intf_set_free_space_thresh(
    uint64_t llk_if, uint64_t counter, uint64_t thresh) {
    *((volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                           llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                           counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_FREE_THRESHOLD_REG_OFFSET)) =
        thresh;
}

inline __attribute__((always_inline)) uint64_t llk_intf_get_free_space_thresh(uint64_t llk_if, uint64_t counter) {
    return *((
        volatile uint32_t*)(TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                            llk_if * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE +
                            counter * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
                            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__TILES_FREE_THRESHOLD_REG_OFFSET));
}
