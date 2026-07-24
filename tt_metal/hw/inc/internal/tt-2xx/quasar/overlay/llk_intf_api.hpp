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
 *
 * @warning Interface and counter indices are not checked at runtime. Callers
 *          must use an interface in [0, LLK_INTERFACE_SIZE) and a counter in
 *          [0, LLK_INTF_COUNTER_SIZE).
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

/**
 * @brief Write an LLK interface register through the custom-instruction path.
 * @param addr LLK register address accepted by the custom instruction.
 * @param data Raw value to write.
 * @note Emits a RISC-V `fence` instruction immediately after the write.
 */
inline void llk_reg_write(std::uint64_t addr, std::uint64_t data) {
    LLK_INTF_WRITE(addr, data);
    asm volatile("fence" : : : "memory");
}

/**
 * @brief Read an LLK interface register through the custom-instruction path.
 * @param addr LLK register address accepted by the custom instruction.
 * @return Raw 64-bit value returned by the custom instruction.
 * @note Emits a RISC-V `fence` instruction immediately after the read.
 */
inline std::uint64_t llk_reg_read(std::uint64_t addr) {
    std::uint64_t data = LLK_INTF_READ(addr);
    asm volatile("fence" : : : "memory");
    return data;
}

// Unfenced custom-instruction register access.

/**
 * @brief Write an LLK interface register through the fast custom-instruction path.
 * @param addr LLK register address accepted by the custom instruction.
 * @param data Raw value to write.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline void fast_llk_reg_write(std::uint64_t addr, std::uint64_t data) { LLK_INTF_WRITE(addr, data); }

/**
 * @brief Read an LLK interface register through the fast custom-instruction path.
 * @param addr LLK register address accepted by the custom instruction.
 * @return Raw 64-bit value returned by the custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline std::uint64_t fast_llk_reg_read(std::uint64_t addr) {
    std::uint64_t data = LLK_INTF_READ(addr);
    return data;
}

// Direct memory-mapped interface.

/**
 * @brief Reset one tile counter through its memory-mapped reset register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @note Performs a volatile 32-bit MMIO write and emits no explicit fence.
 */
inline __attribute__((always_inline)) void llk_intf_reset(std::uint64_t llk_if, std::uint64_t counter) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_OFFSET) =
        1;
}

/**
 * @brief Increment the posted count through its memory-mapped register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param inc Increment value; the volatile MMIO write uses its low 32 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) void llk_intf_inc_posted(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET) =
        inc;
}

/**
 * @brief Increment the acknowledged count through its memory-mapped register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param inc Increment value; the volatile MMIO write uses its low 32 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) void llk_intf_inc_acked(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET) =
        inc;
}

/**
 * @brief Read the occupancy value through the memory-mapped posted register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit MMIO value, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_occupancy(
    std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET);
}

/**
 * @brief Read the free-space value through the memory-mapped acknowledged register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit MMIO value, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_free_space(
    std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_OFFSET);
}

/**
 * @brief Set one tile counter's capacity through its memory-mapped register.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param cap Capacity value; the volatile MMIO write uses its low 32 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) void llk_intf_set_capacity(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t cap) {
    *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET) = cap;
}

/**
 * @brief Read one tile counter's configured capacity through MMIO.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit capacity register, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_capacity(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET);
}

/**
 * @brief Read one tile counter's current posted count through MMIO.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit posted-count register, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_posted(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_OFFSET);
}

/**
 * @brief Read one tile counter's current acknowledged count through MMIO.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit acknowledged-count register, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_acked(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_OFFSET);
}

/**
 * @brief Read one tile counter's error status through MMIO.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The 32-bit error-status register, zero-extended to 64 bits.
 * @note Emits no explicit memory fence.
 */
inline __attribute__((always_inline)) std::uint64_t llk_intf_get_error(std::uint64_t llk_if, std::uint64_t counter) {
    return *reinterpret_cast<volatile std::uint32_t*>(
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR + llk_if * LLK_REG_SIZE +
        counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_OFFSET);
}

// Unfenced custom-instruction interface. Callers must emit a memory fence after
// a group of operations when ordering is required.

/**
 * @brief Reset one tile counter through the fast custom-instruction path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) void fast_llk_intf_reset(std::uint64_t llk_if, std::uint64_t counter) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR,
        1);
}

/**
 * @brief Increment one tile counter's posted count through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param inc Increment value passed to the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) void fast_llk_intf_inc_posted(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR,
        inc);
}

/**
 * @brief Increment one tile counter's acknowledged count through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param inc Increment value passed to the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) void fast_llk_intf_inc_acked(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t inc) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR,
        inc);
}

/**
 * @brief Read one tile counter's occupancy through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return Raw 64-bit value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_occupancy(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR);
}

/**
 * @brief Read one tile counter's free space through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return Raw 64-bit value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_free_space(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR);
}

/**
 * @brief Set one tile counter's capacity through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @param cap Capacity value passed to the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) void fast_llk_intf_set_capacity(
    std::uint64_t llk_if, std::uint64_t counter, std::uint64_t cap) {
    fast_llk_reg_write(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
            TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR,
        cap);
}

/**
 * @brief Read one tile counter's configured capacity through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return Raw 64-bit value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint64_t fast_llk_intf_get_capacity(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR);
}

/**
 * @brief Read one tile counter's posted count through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The low 16 bits of the value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_posted(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);
}

/**
 * @brief Read one tile counter's acknowledged count through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The low 16 bits of the value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_acked(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_ADDR);
}

/**
 * @brief Read one tile counter's error status through the fast path.
 * @param llk_if LLK interface index in [0, LLK_INTERFACE_SIZE).
 * @param counter Tile-counter index in [0, LLK_INTF_COUNTER_SIZE).
 * @return The low 16 bits of the value returned by the LLK custom instruction.
 * @warning This function does not emit a memory fence. The caller is
 *          responsible for ordering this operation when required.
 */
inline __attribute__((always_inline)) std::uint16_t fast_llk_intf_read_error(
    std::uint64_t llk_if, std::uint64_t counter) {
    return fast_llk_reg_read(
        llk_if * LLK_REG_SIZE + counter * COUNTER_REG_SIZE +
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ERROR_STATUS_REG_ADDR);
}

}  // namespace overlay
