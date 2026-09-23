// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0
/**
 * @file addrgen_api.hpp
 * @brief Hardware Address Generator API for Overlay Command Buffers
 *
 * This file provides a comprehensive API for controlling hardware address generators
 * within the overlay system. Each command buffer has a dedicated address generator
 * that can be programmed to generate complex memory access patterns.
 *
 * ## Architecture Overview
 *
 * The address generation functionality is implemented using custom RISC-V ROCC
 * (Rocket Chip Custom Coprocessor) instructions. Since each address generator
 * requires unique opcodes that must be known at compile time, the address generator
 * id is a template parameter:
 * - `xxx_addrgen<ADDRGEN_0>()` - Functions for address generator 0
 * - `xxx_addrgen<ADDRGEN_1>()` - Functions for address generator 1
 * `xxx_addrgen_0()` / `xxx_addrgen_1()` aliases are provided at the end of the file.
 *
 * ## Hardware Loop Implementation
 *
 * Each address generator implements following loops in hardware:
 *
 * ```c
 * for (base = base_start; ; base += face_size) {
 *   for (outer = outer_start; outer < outer_end; outer += outer_stride) {
 *     for (inner = inner_start; inner < inner_end; inner += inner_stride) {
 *       for (bank = bank_start; bank < bank_end; bank += bank_stride) {
 *         yield address = base + outer + inner + (bank_id << bank_offset);
 *       }
 *     }
 *   }
 * }
 * ```
 * Difference to SW loops is that after the condition is met, e.g. inner < inner_end, new idx is equal
 * (inner + inner_stride) mod inner_end, not inner_end.
 *
 * This structure allows for efficient generation of complex memory access patterns
 * including strided access, multi-dimensional arrays, and bank-interleaved memory.
 *
 * ## Bank Ordering
 *
 * The API supports different bank ordering modes via the `bank_order_e` enumeration:
 * - `BANK_INNER`: Bank iteration in the innermost loop
 * - `BANK_MIDDLE`: Bank iteration in the middle loop level
 * - `BANK_OUTER`: Bank iteration in the outermost loop level
 *
 * @note This API depends on the ROCC instruction definitions from rocc_instructions.hpp
 */

#pragma once

#include <type_traits>
#include "rocc_instructions.hpp"

#define ADDRGEN_0 0
#define ADDRGEN_1 1

namespace overlay {

enum bank_order_e { BANK_INNER = 0, BANK_MIDDLE, BANK_OUTER };

/*
 * Focused configuration structs for address generators.
 * Each struct contains only the parameters needed for a specific configuration type.
 * This approach provides zero memory overhead and explicit configuration.
 *
 * Usage: Call only the setup functions you need with the appropriate config struct.
 * E.g. setup_src_inner_loop_addrgen_0(LoopConfig{.stride = 64, .end_addr = 1024});
 *
 */

/* Banking configuration for source or destination */
struct BankingConfig {
    uint32_t endpoint_id_shift;
    uint32_t size;
    uint32_t skip{1};
    uint32_t base{0};
    uint32_t offset{0};
    bank_order_e bank_order{BANK_INNER};
};

/* Loop configuration for inner or outer loops */
struct LoopConfig {
    /* Amount of increase per loop */
    uint64_t stride;
    /* Ending condition for the loop, not inclusive */
    uint64_t end_addr;
    /* Starting offset for the loop */
    uint64_t addr_offset{0};
};

/* Note: Face size and base start use direct uint64_t parameters for simplicity */

#define ADDRGEN_0 0
#define ADDRGEN_1 1

/* Guard for the ADDRGEN template parameter: the __builtin_riscv_ttrocc_addrgen_* builtins accept any
 * constant id and silently encode an out-of-range one as a different instruction. */
template <uint32_t ADDRGEN>
using addrgen_id_t = std::enable_if_t<ADDRGEN == ADDRGEN_0 || ADDRGEN == ADDRGEN_1>;

/* __builtin_riscv_ttrocc_addrgen_push_both{,_pop_x} (sfpi 7.79.0) put the address generator id in a
 * register instead of the immediate field, which the assembler rejects. Emit the instruction directly. */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void ttrocc_addrgen_push_both() {
    asm volatile("tt.rocc.addrgen_push_both %0" ::"i"(ADDRGEN));
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void ttrocc_addrgen_push_both_pop_x(uint64_t skip_src, uint64_t skip_dest) {
    asm volatile("tt.rocc.addrgen_push_both_pop_x %0, %1, %2" ::"i"(ADDRGEN), "r"(skip_src), "r"(skip_dest));
}

/*
 * @fn reset_addrgen<ADDRGEN>()
 *
 * @brief Defines an inline reset functions for resetting address generator state
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void reset_addrgen() {
    __builtin_riscv_ttrocc_addrgen_reset(ADDRGEN);
}

/*
 * @fn reset_counters_addrgen<ADDRGEN>()
 *
 * @brief Defines an inline reset counters functions which resets only the address generator counters
 * while keeping the base addresses, sizes, and strides intact.
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void reset_counters_addrgen() {
    __builtin_riscv_ttrocc_addrgen_reset_counters(ADDRGEN);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_banking_addrgen(const BankingConfig& cfg) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = cfg.endpoint_id_shift;
    misc.f.src_bank_order = cfg.bank_order;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET / 8, cfg.offset);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET / 8, cfg.base);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET / 8, cfg.size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET / 8, cfg.skip);
}

/*
 * @fn setup_src_banking_addrgen<ADDRGEN>()
 *
 * @brief Defines an inline functions for setting up banking for source
 * of address generator, works together with ATT
 *
 * @param endpoint_id_shift Bank offset
 * @param size              Size for the banking for loop
 * @param skip              Step for the banking for loop
 * @param base              Starting index of the banking for loop
 * @param offset  Offset to the index of the banking for loop
 *
 * @example
 * If we have 3 banks (common scenario is to define bank 0 as local L1):
 * (0, 0, 0), (1, 2, 0),(2, 2, 1)
 * Since first bank is local L1 we want to skip it in this example and only iterate through other 2 banks,
 * this can be done with following configuration
 * size = 2
 * skip = 1
 * base = 1
 * offset = 0
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_banking_addrgen(
    uint32_t endpoint_id_shift, uint32_t size, uint32_t skip = 1, uint32_t base = 0, uint32_t current_endpoint = 0) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = endpoint_id_shift;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET / 8, current_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET / 8, skip);
}
/*
 * @fn setup_src_face_size_loop_addrgen<ADDRGEN>()
 *
 * @brief Function for setting face size (outer most loop)
 *
 * @param Face size for most outer loop
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_face_size_addrgen(uint64_t face_size) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET / 8, face_size);
}
/*
 * @fn setup_src_base_start_addrgen<ADDRGEN>()
 *
 * @brief Function for setting base start  (outer most loop)
 *
 * @param Base start address for most outer loop
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_base_start_addrgen(uint64_t base_start) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base_start);
}
/*
 * @fn setup_src_inner_loop_addrgen<ADDRGEN>()
 *
 * @brief Function for setting parameters of inner loop of address generator
 *
 * @param stride Loop step
 * @param size Loop ending condition
 * @addr_offset Loop starting value for addr
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_ADDRESS_REG_OFFSET / 8, addr_offset);
}
/*
 * @fn setup_src_outer_loop_addrgen<ADDRGEN>()
 *
 * @brief Function for setting parameters of outer loop of address generator
 *
 * @param stride Loop step
 * @param size Loop ending condition
 * @addr_offset Loop starting value for addr
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_ADDRESS_REG_OFFSET / 8, addr_offset);
}

/*
 * @fn setup_dest_banking_addrgen<ADDRGEN>()
 *
 * @brief Defines an inline functions for setting up banking for destination
 * of address generator, works together with ATT
 *
 * @param endpoint_id_shift Bank offset
 * @param size              Size for the banking for loop
 * @param skip              Step for the banking for loop
 * @param base              Starting index of the banking for loop
 * @param offset  Offset to the index of the banking for loop
 *
 * @example
 * If we have 3 banks (common scenario is to define bank 0 as local L1):
 * (0, 0, 0), (1, 2, 0),(2, 2, 1)
 * Since first bank is local L1 we want to skip it in this example and only iterate through other 2 banks,
 * this can be done with following configuration
 * size = 2
 * skip = 1
 * base = 1
 * offset = 0
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_banking_addrgen(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = endpoint_id_shift;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET / 8, current_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET / 8, base_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET / 8, skip);
}
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_banking_addrgen(const BankingConfig& cfg) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = cfg.endpoint_id_shift;
    misc.f.dst_bank_order = cfg.bank_order;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET / 8, cfg.offset);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET / 8, cfg.base);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET / 8, cfg.size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET / 8, cfg.skip);
}
/*
 * @fn setup_dest_face_size_addrgen<ADDRGEN>()
 *
 * @brief Function for setting face size (outer most loop)
 *
 * @param Face size for most outer loop
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen(uint64_t face_size) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET / 8, face_size);
}
/*
 * @fn setup_dest_base_start_addrgen<ADDRGEN>()
 *
 * @brief Function for setting destination base start (outer most loop)
 *
 * @param Base start address for most outer loop of destination
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen(uint64_t base_start) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base_start);
}
/*
 * @fn setup_dest_inner_loop_addrgen<ADDRGEN>()
 *
 * @brief Function for setting parameters of inner loop of address generator
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_ADDRESS_REG_OFFSET / 8, addr_offset);
}
/*
 * @fn setup_dest_outer_loop_addrgen<ADDRGEN>()
 *
 * @brief Function for configuring outer loop of destination of  address generator
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_ADDRESS_REG_OFFSET / 8, addr_offset);
}

/*
 * Additional focused setup functions using config structs
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen(const LoopConfig& cfg) {
    setup_src_inner_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end_addr, cfg.addr_offset);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen(const LoopConfig& cfg) {
    setup_src_outer_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end_addr, cfg.addr_offset);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen(const LoopConfig& cfg) {
    setup_dest_inner_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end_addr, cfg.addr_offset);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen(const LoopConfig& cfg) {
    setup_dest_outer_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end_addr, cfg.addr_offset);
}

/* Face size and base start functions use direct parameters - no struct wrappers needed */

/*
 * @fn peek_src_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for source without popping
 * and triggering new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t peek_src_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_peek_src(ADDRGEN);
}

/*
 * @fn pop_src_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for source popping it
 * and triggering new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t pop_src_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_pop_src(ADDRGEN);
}
/*
 * @fn pop_src_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for source popping it
 * and triggers/skips (pop_amount-1) src addresses afterwards
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t pop_src_addrgen(uint64_t pop_amount) {
    return __builtin_riscv_ttrocc_addrgen_pop_x_src(ADDRGEN, pop_amount);
}
/*
 * @fn peek_dest_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for destination without popping
 * and triggering new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t peek_dest_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_peek_dest(ADDRGEN);
}

/*
 * @fn pop_dest_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for destination popping it
 * and triggering new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_pop_dest(ADDRGEN);
}

/*
 * @fn pop_dest_addrgen<ADDRGEN>()
 *
 * @brief Reads current generated address for destination popping it
 * and triggering new pop_amount-1 address to be generated and trown away
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen(uint64_t pop_amount) {
    return __builtin_riscv_ttrocc_addrgen_pop_x_dest(ADDRGEN, pop_amount);
}

/*
 * @fn push_src_addrgen<ADDRGEN>()
 *
 * @brief Pushes generated address from address generator to command buffer
 * and triggers new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void push_src_addrgen() {
    __builtin_riscv_ttrocc_addrgen_push_src(ADDRGEN);
}

/*
 * @fn push_dest_addrgen<ADDRGEN>()
 *
 * @brief Pushes generated address from address generator to command buffer
 * and triggers new address to be generated
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void push_dest_addrgen() {
    __builtin_riscv_ttrocc_addrgen_push_dest(ADDRGEN);
}

/*
 * @fn push_both_addrgen<ADDRGEN>()
 *
 * @brief Pushes generated address from address generator to command buffer
 * and triggers new address to be generated, for both source and destination
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void push_both_addrgen() {
    ttrocc_addrgen_push_both<ADDRGEN>();
}
/*
 * @fn pop_both_addrgen<ADDRGEN>()
 *
 * @brief Returns the generated dest address and skips (x-1) dest addresses afterwards
 * Result holds 32 bits of each dest and source addresses - {dest[31:0], src[31:]}
 *
 */
template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void pop_both_addrgen(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    __builtin_riscv_ttrocc_addrgen_pop_both(ADDRGEN, src_pop_amount, dest_pop_amount);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void add_src_banking_addrgen(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = endpoint_id_shift;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET / 8, current_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET / 8, base_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET / 8, skip);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base_addr);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET / 8, size);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base_addr);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET / 8, inner_stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET / 8, inner_end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_STRIDE_REG_OFFSET / 8, outer_stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_END_REG_OFFSET / 8, outer_end);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void add_dest_banking_addrgen(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;
    misc.f.bank_offset = endpoint_id_shift;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET / 8, current_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET / 8, base_endpoint);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET / 8, skip);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base_addr);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET / 8, size);
}

template <uint32_t ADDRGEN, typename = addrgen_id_t<ADDRGEN>>
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base_addr);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET / 8, inner_stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET / 8, inner_end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_STRIDE_REG_OFFSET / 8, outer_stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_END_REG_OFFSET / 8, outer_end);
}

/* Per-addrgen aliases: addrgen_0/_1 spellings of the ADDRGEN-templated functions above. */
inline __attribute__((always_inline)) void reset_addrgen_0() { return reset_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void reset_addrgen_1() { return reset_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void reset_counters_addrgen_0() { return reset_counters_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void reset_counters_addrgen_1() { return reset_counters_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void setup_src_banking_addrgen_0(const BankingConfig& cfg) {
    return setup_src_banking_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_src_banking_addrgen_1(const BankingConfig& cfg) {
    return setup_src_banking_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_src_banking_addrgen_0(
    uint32_t endpoint_id_shift, uint32_t size, uint32_t skip = 1, uint32_t base = 0, uint32_t current_endpoint = 0) {
    return setup_src_banking_addrgen<ADDRGEN_0>(endpoint_id_shift, size, skip, base, current_endpoint);
}
inline __attribute__((always_inline)) void setup_src_banking_addrgen_1(
    uint32_t endpoint_id_shift, uint32_t size, uint32_t skip = 1, uint32_t base = 0, uint32_t current_endpoint = 0) {
    return setup_src_banking_addrgen<ADDRGEN_1>(endpoint_id_shift, size, skip, base, current_endpoint);
}
inline __attribute__((always_inline)) void setup_src_face_size_addrgen_0(uint64_t face_size) {
    return setup_src_face_size_addrgen<ADDRGEN_0>(face_size);
}
inline __attribute__((always_inline)) void setup_src_face_size_addrgen_1(uint64_t face_size) {
    return setup_src_face_size_addrgen<ADDRGEN_1>(face_size);
}
inline __attribute__((always_inline)) void setup_src_base_start_addrgen_0(uint64_t base_start) {
    return setup_src_base_start_addrgen<ADDRGEN_0>(base_start);
}
inline __attribute__((always_inline)) void setup_src_base_start_addrgen_1(uint64_t base_start) {
    return setup_src_base_start_addrgen<ADDRGEN_1>(base_start);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_0(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_src_inner_loop_addrgen<ADDRGEN_0>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_1(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_src_inner_loop_addrgen<ADDRGEN_1>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_0(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_src_outer_loop_addrgen<ADDRGEN_0>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_1(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_src_outer_loop_addrgen<ADDRGEN_1>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_0(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return setup_dest_banking_addrgen<ADDRGEN_0>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_1(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return setup_dest_banking_addrgen<ADDRGEN_1>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_0(const BankingConfig& cfg) {
    return setup_dest_banking_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_1(const BankingConfig& cfg) {
    return setup_dest_banking_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen_0(uint64_t face_size) {
    return setup_dest_face_size_addrgen<ADDRGEN_0>(face_size);
}
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen_1(uint64_t face_size) {
    return setup_dest_face_size_addrgen<ADDRGEN_1>(face_size);
}
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen_0(uint64_t base_start) {
    return setup_dest_base_start_addrgen<ADDRGEN_0>(base_start);
}
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen_1(uint64_t base_start) {
    return setup_dest_base_start_addrgen<ADDRGEN_1>(base_start);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_0(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_0>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_1(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_1>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_0(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_0>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_1(
    uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_1>(stride, size, addr_offset);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_src_inner_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_src_inner_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_src_outer_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_src_outer_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) uint64_t peek_src_addrgen_0() { return peek_src_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) uint64_t peek_src_addrgen_1() { return peek_src_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) uint64_t pop_src_addrgen_0() { return pop_src_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) uint64_t pop_src_addrgen_1() { return pop_src_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) uint64_t pop_src_addrgen_0(uint64_t pop_amount) {
    return pop_src_addrgen<ADDRGEN_0>(pop_amount);
}
inline __attribute__((always_inline)) uint64_t pop_src_addrgen_1(uint64_t pop_amount) {
    return pop_src_addrgen<ADDRGEN_1>(pop_amount);
}
inline __attribute__((always_inline)) uint64_t peek_dest_addrgen_0() { return peek_dest_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) uint64_t peek_dest_addrgen_1() { return peek_dest_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen_0() { return pop_dest_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen_1() { return pop_dest_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen_0(uint64_t pop_amount) {
    return pop_dest_addrgen<ADDRGEN_0>(pop_amount);
}
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen_1(uint64_t pop_amount) {
    return pop_dest_addrgen<ADDRGEN_1>(pop_amount);
}
inline __attribute__((always_inline)) void push_src_addrgen_0() { return push_src_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_src_addrgen_1() { return push_src_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void push_dest_addrgen_0() { return push_dest_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_dest_addrgen_1() { return push_dest_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void push_both_addrgen_0() { return push_both_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_both_addrgen_1() { return push_both_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void pop_both_addrgen_0(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    return pop_both_addrgen<ADDRGEN_0>(src_pop_amount, dest_pop_amount);
}
inline __attribute__((always_inline)) void pop_both_addrgen_1(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    return pop_both_addrgen<ADDRGEN_1>(src_pop_amount, dest_pop_amount);
}
inline __attribute__((always_inline)) void add_src_banking_addrgen_0(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return add_src_banking_addrgen<ADDRGEN_0>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void add_src_banking_addrgen_1(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return add_src_banking_addrgen<ADDRGEN_1>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen_0(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    return setup_src_1D_stride_addrgen<ADDRGEN_0>(base_addr, size, stride);
}
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen_1(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    return setup_src_1D_stride_addrgen<ADDRGEN_1>(base_addr, size, stride);
}
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen_0(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    return setup_src_2D_stride_addrgen<ADDRGEN_0>(base_addr, size, inner_end, inner_stride, outer_end, outer_stride);
}
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen_1(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    return setup_src_2D_stride_addrgen<ADDRGEN_1>(base_addr, size, inner_end, inner_stride, outer_end, outer_stride);
}
inline __attribute__((always_inline)) void add_dest_banking_addrgen_0(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return add_dest_banking_addrgen<ADDRGEN_0>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void add_dest_banking_addrgen_1(
    uint32_t endpoint_id_shift,
    uint32_t size,
    uint32_t skip = 1,
    uint32_t base_endpoint = 0,
    uint32_t current_endpoint = 0) {
    return add_dest_banking_addrgen<ADDRGEN_1>(endpoint_id_shift, size, skip, base_endpoint, current_endpoint);
}
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen_0(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    return setup_dest_1D_stride_addrgen<ADDRGEN_0>(base_addr, size, stride);
}
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen_1(
    uint64_t base_addr, uint64_t size, uint64_t stride) {
    return setup_dest_1D_stride_addrgen<ADDRGEN_1>(base_addr, size, stride);
}
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen_0(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    return setup_dest_2D_stride_addrgen<ADDRGEN_0>(base_addr, size, inner_end, inner_stride, outer_end, outer_stride);
}
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen_1(
    uint64_t base_addr,
    uint64_t size,
    uint64_t inner_end,
    uint64_t inner_stride,
    uint64_t outer_end,
    uint64_t outer_stride) {
    return setup_dest_2D_stride_addrgen<ADDRGEN_1>(base_addr, size, inner_end, inner_stride, outer_end, outer_stride);
}

}  // namespace overlay
