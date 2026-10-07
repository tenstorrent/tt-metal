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
 * @note Built on cmdbuff_api.hpp: address generator N feeds command buffer N
 */

#pragma once

#include "cmdbuff_api.hpp"

namespace overlay {

/* Address generator id, the template parameter of every *_addrgen<ADDRGEN>() function (an enum so that an
 * out-of-range id is a compile error, see CmdBuf). Address generator N feeds command buffer N: its base
 * address registers live in that command buffer, and push_*() hand addresses to it. */
enum AddrGen : uint32_t { ADDRGEN_0 = CMDBUF_0, ADDRGEN_1 = CMDBUF_1 };

/* Command buffer paired with an address generator */
constexpr CmdBuf paired_cmdbuf(AddrGen addrgen) { return static_cast<CmdBuf>(static_cast<uint32_t>(addrgen)); }

enum bank_order_e { BANK_INNER = 0, BANK_MIDDLE, BANK_OUTER };

/*
 * Configuration structs for address generators.
 *
 * Usage: call only the setup functions you need with the appropriate config struct.
 * E.g. setup_src_inner_loop_addrgen<ADDRGEN_0>({.stride = 64, .end = 1024});
 */

/* Banking loop of the source or destination, used together with the ATT */
struct BankingConfig {
    /* Bit position of the endpoint (bank) id in the generated address. Shared by source and destination:
     * the last setup_*_banking_addrgen() call wins. */
    uint32_t endpoint_id_shift;
    /* Number of banks to iterate over */
    uint32_t size;
    /* Step between bank indices */
    uint32_t skip{1};
    /* First bank index of the loop */
    uint32_t base{0};
    /* Bank index the generator starts on (BANK_CURRENT) */
    uint32_t current{0};
    /* Loop level of the banking loop */
    bank_order_e bank_order{BANK_INNER};
};

/* Inner or outer loop of the source or destination */
struct LoopConfig {
    /* Address increase per iteration */
    uint64_t stride;
    /* End of the loop, exclusive */
    uint64_t end;
    /* Starting value of the loop */
    uint64_t start{0};
};

/*
 * @fn reset_addrgen<ADDRGEN>()
 *
 * @brief Resets all address generator registers to their RDL defaults
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void reset_addrgen() {
    __builtin_riscv_ttrocc_addrgen_reset(ADDRGEN);
}

/*
 * @fn reset_counters_addrgen<ADDRGEN>()
 *
 * @brief Resets only the address generator counters, keeping base addresses, sizes and strides
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void reset_counters_addrgen() {
    __builtin_riscv_ttrocc_addrgen_reset_counters(ADDRGEN);
}

/*
 * @fn setup_src_banking_addrgen<ADDRGEN>()
 *
 * @brief Sets up the banking loop of the source
 *
 * @param cfg Banking loop
 *
 * @example
 * With 3 banks where bank 0 is the local L1, iterating over the other 2 banks only:
 * {.endpoint_id_shift = ..., .size = 2, .skip = 1, .base = 1, .current = 0}
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_banking_addrgen(const BankingConfig& cfg) {
    // MISC also holds the destination bank order: read-modify-write it.
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val =
        __builtin_riscv_ttrocc_addrgen_rd_reg(ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8);
    misc.f.bank_offset = cfg.endpoint_id_shift;
    misc.f.src_bank_order = cfg.bank_order;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET / 8, cfg.current);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET / 8, cfg.base);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET / 8, cfg.size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET / 8, cfg.skip);
}

/*
 * @fn setup_dest_banking_addrgen<ADDRGEN>()
 *
 * @brief Sets up the banking loop of the destination (see setup_src_banking_addrgen)
 *
 * @param cfg Banking loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_banking_addrgen(const BankingConfig& cfg) {
    // MISC also holds the source bank order: read-modify-write it.
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val =
        __builtin_riscv_ttrocc_addrgen_rd_reg(ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8);
    misc.f.bank_offset = cfg.endpoint_id_shift;
    misc.f.dst_bank_order = cfg.bank_order;
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET / 8, misc.val);

    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET / 8, cfg.current);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET / 8, cfg.base);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET / 8, cfg.size);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET / 8, cfg.skip);
}

/*
 * @fn setup_src_face_size_addrgen<ADDRGEN>()
 *
 * @brief Sets the source face size, the base increment of the outermost loop
 *
 * @param face_size Face size in bytes
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_face_size_addrgen(uint64_t face_size) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET / 8, face_size);
}

/*
 * @fn setup_dest_face_size_addrgen<ADDRGEN>()
 *
 * @brief Sets the destination face size, the base increment of the outermost loop
 *
 * @param face_size Face size in bytes
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen(uint64_t face_size) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET / 8, face_size);
}

/*
 * @fn setup_src_base_start_addrgen<ADDRGEN>()
 *
 * @brief Sets the source base start address of the outermost loop (SRC_BASE of the paired command buffer)
 *
 * @param base_start Base start address
 *
 * @note reset_cmdbuf() clears it: reset the command buffer first
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_base_start_addrgen(uint64_t base_start) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        paired_cmdbuf(ADDRGEN), TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base_start);
}

/*
 * @fn setup_dest_base_start_addrgen<ADDRGEN>()
 *
 * @brief Sets the destination base start address of the outermost loop (DEST_BASE of the paired command
 * buffer)
 *
 * @param base_start Base start address
 *
 * @note reset_cmdbuf() clears it: reset the command buffer first
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen(uint64_t base_start) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        paired_cmdbuf(ADDRGEN), TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base_start);
}

/*
 * @fn setup_src_inner_loop_addrgen<ADDRGEN>()
 *
 * @brief Sets the source inner loop
 *
 * @param stride Address increase per iteration
 * @param end End of the loop, exclusive
 * @param start Starting value of the loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET / 8, end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_ADDRESS_REG_OFFSET / 8, start);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen(const LoopConfig& cfg) {
    setup_src_inner_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end, cfg.start);
}

/*
 * @fn setup_src_outer_loop_addrgen<ADDRGEN>()
 *
 * @brief Sets the source outer loop
 *
 * @param stride Address increase per iteration
 * @param end End of the loop, exclusive
 * @param start Starting value of the loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_END_REG_OFFSET / 8, end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_ADDRESS_REG_OFFSET / 8, start);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen(const LoopConfig& cfg) {
    setup_src_outer_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end, cfg.start);
}

/*
 * @fn setup_dest_inner_loop_addrgen<ADDRGEN>()
 *
 * @brief Sets the destination inner loop
 *
 * @param stride Address increase per iteration
 * @param end End of the loop, exclusive
 * @param start Starting value of the loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET / 8, end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_ADDRESS_REG_OFFSET / 8, start);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen(const LoopConfig& cfg) {
    setup_dest_inner_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end, cfg.start);
}

/*
 * @fn setup_dest_outer_loop_addrgen<ADDRGEN>()
 *
 * @brief Sets the destination outer loop
 *
 * @param stride Address increase per iteration
 * @param end End of the loop, exclusive
 * @param start Starting value of the loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_STRIDE_REG_OFFSET / 8, stride);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_END_REG_OFFSET / 8, end);
    __builtin_riscv_ttrocc_addrgen_wr_reg(
        ADDRGEN, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_ADDRESS_REG_OFFSET / 8, start);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen(const LoopConfig& cfg) {
    setup_dest_outer_loop_addrgen<ADDRGEN>(cfg.stride, cfg.end, cfg.start);
}

/*
 * @fn setup_src_1D_stride_addrgen<ADDRGEN>()
 *
 * @brief Sets up a 1D strided source: base start, inner loop, and a face size equal to the loop end so the
 * pattern repeats from base_addr + end
 *
 * @param base_addr Base start address (see setup_src_base_start_addrgen)
 * @param loop Inner loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen(uint64_t base_addr, const LoopConfig& loop) {
    setup_src_base_start_addrgen<ADDRGEN>(base_addr);
    setup_src_inner_loop_addrgen<ADDRGEN>(loop);
    setup_src_face_size_addrgen<ADDRGEN>(loop.end);
}

/*
 * @fn setup_src_2D_stride_addrgen<ADDRGEN>()
 *
 * @brief Sets up a 2D strided source: base start, face size, inner and outer loop
 *
 * @param base_addr Base start address (see setup_src_base_start_addrgen)
 * @param face_size Face size in bytes
 * @param inner Inner loop
 * @param outer Outer loop
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    setup_src_base_start_addrgen<ADDRGEN>(base_addr);
    setup_src_inner_loop_addrgen<ADDRGEN>(inner);
    setup_src_outer_loop_addrgen<ADDRGEN>(outer);
    setup_src_face_size_addrgen<ADDRGEN>(face_size);
}

/*
 * @fn setup_dest_1D_stride_addrgen<ADDRGEN>()
 *
 * @brief Sets up a 1D strided destination (see setup_src_1D_stride_addrgen)
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen(uint64_t base_addr, const LoopConfig& loop) {
    setup_dest_base_start_addrgen<ADDRGEN>(base_addr);
    setup_dest_inner_loop_addrgen<ADDRGEN>(loop);
    setup_dest_face_size_addrgen<ADDRGEN>(loop.end);
}

/*
 * @fn setup_dest_2D_stride_addrgen<ADDRGEN>()
 *
 * @brief Sets up a 2D strided destination (see setup_src_2D_stride_addrgen)
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    setup_dest_base_start_addrgen<ADDRGEN>(base_addr);
    setup_dest_inner_loop_addrgen<ADDRGEN>(inner);
    setup_dest_outer_loop_addrgen<ADDRGEN>(outer);
    setup_dest_face_size_addrgen<ADDRGEN>(face_size);
}

/*
 * @fn peek_src_addrgen<ADDRGEN>()
 *
 * @return Current generated source address, without generating a new one
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t peek_src_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_peek_src(ADDRGEN);
}

/*
 * @fn pop_src_addrgen<ADDRGEN>()
 *
 * @brief Returns the current generated source address and generates the next one; with pop_amount,
 * skips (pop_amount - 1) further source addresses
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t pop_src_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_pop_src(ADDRGEN);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t pop_src_addrgen(uint64_t pop_amount) {
    return __builtin_riscv_ttrocc_addrgen_pop_x_src(ADDRGEN, pop_amount);
}

/*
 * @fn peek_dest_addrgen<ADDRGEN>()
 *
 * @return Current generated destination address, without generating a new one
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t peek_dest_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_peek_dest(ADDRGEN);
}

/*
 * @fn pop_dest_addrgen<ADDRGEN>()
 *
 * @brief Returns the current generated destination address and generates the next one; with pop_amount,
 * skips (pop_amount - 1) further destination addresses
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen() {
    return __builtin_riscv_ttrocc_addrgen_pop_dest(ADDRGEN);
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t pop_dest_addrgen(uint64_t pop_amount) {
    return __builtin_riscv_ttrocc_addrgen_pop_x_dest(ADDRGEN, pop_amount);
}

/*
 * @fn pop_both_addrgen<ADDRGEN>()
 *
 * @brief Returns the current generated source and destination addresses and advances both, skipping
 * (src_pop_amount - 1) source and (dest_pop_amount - 1) destination addresses
 *
 * @return {dest[31:0], src[31:0]}
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t pop_both_addrgen(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    return __builtin_riscv_ttrocc_addrgen_pop_both(ADDRGEN, src_pop_amount, dest_pop_amount);
}

/*
 * @fn push_src_addrgen<ADDRGEN>()
 *
 * @brief Pushes the generated source address to the paired command buffer and generates the next one
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void push_src_addrgen() {
    __builtin_riscv_ttrocc_addrgen_push_src(ADDRGEN);
}

/*
 * @fn push_dest_addrgen<ADDRGEN>()
 *
 * @brief Pushes the generated destination address to the paired command buffer and generates the next one
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void push_dest_addrgen() {
    __builtin_riscv_ttrocc_addrgen_push_dest(ADDRGEN);
}

/* __builtin_riscv_ttrocc_addrgen_push_both{,_pop_x} (sfpi 7.79.0) put the address generator id in a
 * register instead of the immediate field, which the assembler rejects. Emit the instruction directly. */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void ttrocc_addrgen_push_both() {
    asm volatile("tt.rocc.addrgen_push_both %0" ::"i"(ADDRGEN));
}
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void ttrocc_addrgen_push_both_pop_x(uint64_t skip_src, uint64_t skip_dest) {
    asm volatile("tt.rocc.addrgen_push_both_pop_x %0, %1, %2" ::"i"(ADDRGEN), "r"(skip_src), "r"(skip_dest));
}

/*
 * @fn push_both_addrgen<ADDRGEN>()
 *
 * @brief Pushes the generated source and destination addresses to the paired command buffer and generates
 * the next ones
 */
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) void push_both_addrgen() {
    ttrocc_addrgen_push_both<ADDRGEN>();
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
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_0(const BankingConfig& cfg) {
    return setup_dest_banking_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_banking_addrgen_1(const BankingConfig& cfg) {
    return setup_dest_banking_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_src_face_size_addrgen_0(uint64_t face_size) {
    return setup_src_face_size_addrgen<ADDRGEN_0>(face_size);
}
inline __attribute__((always_inline)) void setup_src_face_size_addrgen_1(uint64_t face_size) {
    return setup_src_face_size_addrgen<ADDRGEN_1>(face_size);
}
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen_0(uint64_t face_size) {
    return setup_dest_face_size_addrgen<ADDRGEN_0>(face_size);
}
inline __attribute__((always_inline)) void setup_dest_face_size_addrgen_1(uint64_t face_size) {
    return setup_dest_face_size_addrgen<ADDRGEN_1>(face_size);
}
inline __attribute__((always_inline)) void setup_src_base_start_addrgen_0(uint64_t base_start) {
    return setup_src_base_start_addrgen<ADDRGEN_0>(base_start);
}
inline __attribute__((always_inline)) void setup_src_base_start_addrgen_1(uint64_t base_start) {
    return setup_src_base_start_addrgen<ADDRGEN_1>(base_start);
}
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen_0(uint64_t base_start) {
    return setup_dest_base_start_addrgen<ADDRGEN_0>(base_start);
}
inline __attribute__((always_inline)) void setup_dest_base_start_addrgen_1(uint64_t base_start) {
    return setup_dest_base_start_addrgen<ADDRGEN_1>(base_start);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_0(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_src_inner_loop_addrgen<ADDRGEN_0>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_1(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_src_inner_loop_addrgen<ADDRGEN_1>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_src_inner_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_src_inner_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_src_inner_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_0(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_src_outer_loop_addrgen<ADDRGEN_0>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_1(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_src_outer_loop_addrgen<ADDRGEN_1>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_src_outer_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_src_outer_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_src_outer_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_0(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_0>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_1(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_1>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_inner_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_dest_inner_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_0(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_0>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_1(
    uint64_t stride, uint64_t end, uint64_t start = 0) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_1>(stride, end, start);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_0(const LoopConfig& cfg) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_0>(cfg);
}
inline __attribute__((always_inline)) void setup_dest_outer_loop_addrgen_1(const LoopConfig& cfg) {
    return setup_dest_outer_loop_addrgen<ADDRGEN_1>(cfg);
}
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen_0(uint64_t base_addr, const LoopConfig& loop) {
    return setup_src_1D_stride_addrgen<ADDRGEN_0>(base_addr, loop);
}
inline __attribute__((always_inline)) void setup_src_1D_stride_addrgen_1(uint64_t base_addr, const LoopConfig& loop) {
    return setup_src_1D_stride_addrgen<ADDRGEN_1>(base_addr, loop);
}
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen_0(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    return setup_src_2D_stride_addrgen<ADDRGEN_0>(base_addr, face_size, inner, outer);
}
inline __attribute__((always_inline)) void setup_src_2D_stride_addrgen_1(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    return setup_src_2D_stride_addrgen<ADDRGEN_1>(base_addr, face_size, inner, outer);
}
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen_0(uint64_t base_addr, const LoopConfig& loop) {
    return setup_dest_1D_stride_addrgen<ADDRGEN_0>(base_addr, loop);
}
inline __attribute__((always_inline)) void setup_dest_1D_stride_addrgen_1(uint64_t base_addr, const LoopConfig& loop) {
    return setup_dest_1D_stride_addrgen<ADDRGEN_1>(base_addr, loop);
}
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen_0(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    return setup_dest_2D_stride_addrgen<ADDRGEN_0>(base_addr, face_size, inner, outer);
}
inline __attribute__((always_inline)) void setup_dest_2D_stride_addrgen_1(
    uint64_t base_addr, uint64_t face_size, const LoopConfig& inner, const LoopConfig& outer) {
    return setup_dest_2D_stride_addrgen<ADDRGEN_1>(base_addr, face_size, inner, outer);
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
inline __attribute__((always_inline)) uint64_t pop_both_addrgen_0(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    return pop_both_addrgen<ADDRGEN_0>(src_pop_amount, dest_pop_amount);
}
inline __attribute__((always_inline)) uint64_t pop_both_addrgen_1(uint64_t src_pop_amount, uint64_t dest_pop_amount) {
    return pop_both_addrgen<ADDRGEN_1>(src_pop_amount, dest_pop_amount);
}
inline __attribute__((always_inline)) void push_src_addrgen_0() { return push_src_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_src_addrgen_1() { return push_src_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void push_dest_addrgen_0() { return push_dest_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_dest_addrgen_1() { return push_dest_addrgen<ADDRGEN_1>(); }
inline __attribute__((always_inline)) void push_both_addrgen_0() { return push_both_addrgen<ADDRGEN_0>(); }
inline __attribute__((always_inline)) void push_both_addrgen_1() { return push_both_addrgen<ADDRGEN_1>(); }

}  // namespace overlay
