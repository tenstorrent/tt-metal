// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file llk_intf_api.hpp
 * @brief Quasar low-level kernel interface tile-counter API.
 *
 * Provides direct memory-mapped access and custom-instruction access to the
 * four LLK interfaces and their tile counters. The legacy custom-instruction
 * helpers fence each operation; the fast helpers leave fence placement to the
 * caller so several operations can be grouped behind one fence.
 */

#pragma once

#include <cstdint>

#include "meta/registers/overlay_reg.h"
#include "rocc_instructions.hpp"

// Retain these macros for compatibility with existing callers.
#define LLK_REG_SIZE TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE
#define COUNTER_REG_SIZE TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE

namespace overlay {

/** Available LLK interfaces. */
enum llk_interface_e {
    LLK_INTERFACE_0 = 0,
    LLK_INTERFACE_1 = 1,
    LLK_INTERFACE_2 = 2,
    LLK_INTERFACE_3 = 3,
    LLK_INTERFACE_SIZE = 4,
};

/** Counter indices within an LLK interface. */
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
    LLK_INTF_COUNTER_SIZE = 16,
};

// Fenced custom-instruction register access.

/** Write an LLK interface register and fence subsequent memory operations. */
inline void llk_reg_write(std::uint64_t addr, std::uint64_t data) {
    LLK_INTF_WRITE(addr, data);
    asm volatile("fence" : : : "memory");
}

/** Read an LLK interface register and fence subsequent memory operations. */
inline std::uint64_t llk_reg_read(std::uint64_t addr) {
    std::uint64_t data = LLK_INTF_READ(addr);
    asm volatile("fence" : : : "memory");
    return data;
}

// Unfenced custom-instruction register access.

/** Write an LLK interface register without emitting a memory fence. */
inline void fast_llk_reg_write(std::uint64_t addr, std::uint64_t data) { LLK_INTF_WRITE(addr, data); }

/** Read an LLK interface register without emitting a memory fence. */
inline std::uint64_t fast_llk_reg_read(std::uint64_t addr) {
    std::uint64_t data = LLK_INTF_READ(addr);
    return data;
}

// Direct memory-mapped interface.

/** Reset one tile counter in an LLK interface. */
inline __attribute__((always_inline)) void llk_intf_reset(std::uint64_t llk_if, std::uint64_t counter) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_OFFSET) =
        1;
}

/** Increment the posted count for one tile counter. */
inline __attribute__((always_inline)) void llk_intf_inc_posted(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET) =
        inc;
}

/** Increment the acknowledged count for one tile counter. */
inline __attribute__((always_inline)) void llk_intf_inc_acked(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET) =
        inc;
}

/** Return the occupancy value for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_occupancy(
    std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET);
}

/** Return the free-space value for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_free_space(
    std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET);
}

/** Set the capacity for one tile counter. */
inline __attribute__((always_inline)) void llk_intf_set_capacity(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t cap) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET) = cap;
}

/** Return the configured capacity for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_capacity(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET);
}

/** Return the current posted count for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_posted(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_OFFSET);
}

/** Return the current acknowledged count for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_acked(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_OFFSET);
}

/** Return the error status for one tile counter. */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_error(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_OFFSET);
}

// Unfenced custom-instruction interface. Callers must emit a memory fence after
// a group of operations when ordering is required.

/** Reset one tile counter without emitting a memory fence. */
inline __attribute__((always_inline)) void fast_llk_intf_reset(std::uint64_t llk_if, std::uint64_t counter) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR,
        1);
}

/** Increment the posted count without emitting a memory fence. */
inline __attribute__((always_inline)) void fast_llk_intf_inc_posted(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR,
        inc);
}

/** Increment the acknowledged count without emitting a memory fence. */
inline __attribute__((always_inline)) void fast_llk_intf_inc_acked(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR,
        inc);
}

/** Return the occupancy value without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_occupancy(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR);
}

/** Return the free-space value without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_free_space(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR);
}

/** Set the capacity without emitting a memory fence. */
inline __attribute__((always_inline)) void fast_llk_intf_set_capacity(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t cap) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR,
        cap);
}

/** Return the configured capacity without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_capacity(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR);
}

/** Return the 16-bit posted count without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_posted(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);
}

/** Return the 16-bit acknowledged count without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_acked(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_ADDR);
}

/** Return the 16-bit error status without emitting a memory fence. */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_error(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_ADDR);
}

}  // namespace overlay
