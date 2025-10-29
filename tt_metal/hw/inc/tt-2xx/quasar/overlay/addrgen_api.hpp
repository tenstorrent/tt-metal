// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
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
 * requires unique opcodes that must be known at compile time, this API uses
 * C++ macros to generate separate function variants for each address generator:
 * - `xxx_addrgen_0()` - Functions for address generator 0
 * - `xxx_addrgen_1()` - Functions for address generator 1
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

#include "rocc_instructions.hpp"

#define ADDRGEN_0 0
#define ADDRGEN_1 1

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

#define DEFINE_ADDR_GEN(buf_name, cmdbuf)                                                                            \
                                                                                                                     \
    /*                                                                                                               \
     * @def reset_addrgen_0()                                                                                        \
     * @def reset_addrgen_1()                                                                                        \
     *                                                                                                               \
     * @brief Defines an inline reset functions for reseting address generator state                                 \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void reset_##buf_name() { ADDRGEN_RESET(cmdbuf); }                         \
                                                                                                                     \
    /*                                                                                                               \
     * @def reset_counters_addrgen_0()                                                                               \
     * @def reset_counters_addrgen_1()                                                                               \
     *                                                                                                               \
     * @brief Defines an inline reset counters functions which resets only the address generator counters            \
     * while keeping the base addresses, sizes, and strides intact.                                                  \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void reset_counters_##buf_name() { ADDRGEN_RESET_COUNTERS(cmdbuf); }       \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_src_banking_##buf_name(const BankingConfig& cfg) {              \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = cfg.endpoint_id_shift;                                                                  \
        misc.f.src_bank_order = cfg.bank_order;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET, cfg.offset);    \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET, cfg.base);         \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET, cfg.size);         \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET, cfg.skip);         \
    }                                                                                                                \
                                                                                                                     \
    /*                                                                                                               \
     * @def setup_src_banking_addrgen_0()                                                                            \
     * @def setup_src_banking_addrgen_1()                                                                            \
     *                                                                                                               \
     * @brief Defines an inline functions for setting up banking for source                                          \
     * of address generator, works together with ATT                                                                 \
     *                                                                                                               \
     * @param endpoint_id_shift Bank offset                                                                          \
     * @param size              Size for the banking for loop                                                        \
     * @param skip              Step for the banking for loop                                                        \
     * @param base              Starting index of the banking for loop                                               \
     * @param offset  Offset to the index of the banking for loop                                                    \
     *                                                                                                               \
     * @example                                                                                                      \
     * If we have 3 banks (common scenario is to define bank 0 as local L1):                                         \
     * (0, 0, 0), (1, 2, 0),(2, 2, 1)                                                                                \
     * Since first bank is local L1 we want to skip it in this example and only iterate trough other 2 banks,        \
     * this can be done with following configuration                                                                 \
     * size = 2                                                                                                      \
     * skip = 1                                                                                                      \
     * base = 1                                                                                                      \
     * offset = 0                                                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_banking_##buf_name(                                         \
        uint32_t endpoint_id_shift,                                                                                  \
        uint32_t size,                                                                                               \
        uint32_t skip = 1,                                                                                           \
        uint32_t base = 0,                                                                                           \
        uint32_t current_endpoint = 0) {                                                                             \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = endpoint_id_shift;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(                                                                                              \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET, current_endpoint);         \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET, base);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET, skip);             \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_src_face_size_loop_addrgen_0()                                                                     \
     * @def setup_src_face_size_oop_addrgen_1()                                                                      \
     *                                                                                                               \
     * @brief Function for setting face size (outer most loop)                                                       \
     *                                                                                                               \
     * @param Face size for most outer loop                                                                          \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_face_size_##buf_name(uint64_t face_size) {                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET, face_size);        \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_src_base_start_addrgen_0()                                                                         \
     * @def setup_src_base_start_addrgen_1()                                                                         \
     *                                                                                                               \
     * @brief Function for setting base start  (outer most loop)                                                     \
     *                                                                                                               \
     * @param Base start address for most outer loop                                                                 \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_base_start_##buf_name(uint64_t base_start) {                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET, base_start);                 \
    }                                                                                                                \
    /*                                                                                                               \
    /*                                                                                                               \
     * @def setup_src_inner_loop_addrgen_0()                                                                         \
     * @def setup_src_inner_loop_addrgen_1()                                                                         \
     *                                                                                                               \
     * @brief Function for setting parameters of inner loop of address generator                                     \
     *                                                                                                               \
     * @param stride Loop step                                                                                       \
     * @param size Loop ending condition                                                                             \
     * @addr_offset Loop starting value for addr                                                                     \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_inner_loop_##buf_name(                                      \
        uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {                                                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET, stride);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_ADDRESS_REG_OFFSET, addr_offset);  \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_src_outer_loop_addrgen_0()                                                                         \
     * @def setup_src_outer_loop_addrgen_1()                                                                         \
     *                                                                                                               \
     * @brief Function for setting parameters of outer loop of address generator                                     \
     *                                                                                                               \
     * @param stride Loop step                                                                                       \
     * @param size Loop ending condition                                                                             \
     * @addr_offset Loop starting value for addr                                                                     \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_outer_loop_##buf_name(                                      \
        uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {                                                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_STRIDE_REG_OFFSET, stride);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_END_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_ADDRESS_REG_OFFSET, addr_offset);  \
    }                                                                                                                \
                                                                                                                     \
    /*                                                                                                               \
     * @def setup_dest_banking_addrgen_0()                                                                           \
     * @def setup_dest_banking_addrgen_1()                                                                           \
     *                                                                                                               \
     * @brief Defines an inline functions for setting up banking for destination                                     \
     * of address generator, works together with ATT                                                                 \
     *                                                                                                               \
     * @param endpoint_id_shift Bank offset                                                                          \
     * @param size              Size for the banking for loop                                                        \
     * @param skip              Step for the banking for loop                                                        \
     * @param base              Starting index of the banking for loop                                               \
     * @param offset  Offset to the index of the banking for loop                                                    \
     *                                                                                                               \
     * @example                                                                                                      \
     * If we have 3 banks (common scenario is to define bank 0 as local L1):                                         \
     * (0, 0, 0), (1, 2, 0),(2, 2, 1)                                                                                \
     * Since first bank is local L1 we want to skip it in this example and only iterate trough other 2 banks,        \
     * this can be done with following configuraiton                                                                 \
     * size = 2                                                                                                      \
     * skip = 1                                                                                                      \
     * base = 1                                                                                                      \
     * offset = 0                                                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_dest_banking_##buf_name(                                        \
        uint32_t endpoint_id_shift,                                                                                  \
        uint32_t size,                                                                                               \
        uint32_t skip = 1,                                                                                           \
        uint32_t base_endpoint = 0,                                                                                  \
        uint32_t current_endpoint = 0) {                                                                             \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = endpoint_id_shift;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(                                                                                              \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET, current_endpoint);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET, base_endpoint);   \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET, skip);            \
    }                                                                                                                \
    inline __attribute__((always_inline)) void setup_dest_banking_##buf_name(const BankingConfig& cfg) {             \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = cfg.endpoint_id_shift;                                                                  \
        misc.f.dst_bank_order = cfg.bank_order;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET, cfg.offset);   \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET, cfg.base);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET, cfg.size);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET, cfg.skip);        \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_dest_face_size_addrgen_0()                                                                         \
     * @def setup_dest_face_size_addrgen_1()                                                                         \
     *                                                                                                               \
     * @brief Function for setting face size (outer most loop)                                                       \
     *                                                                                                               \
     * @param Face size for most outer loop                                                                          \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_dest_face_size_##buf_name(uint64_t face_size) {                 \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET, face_size);       \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_dest_base_start_addrgen_0()                                                                        \
     * @def setup_dest_base_start_addrgen_1()                                                                        \
     *                                                                                                               \
     * @brief Function for setting destination base start (outer most loop)                                          \
     *                                                                                                               \
     * @param Base start address for most outer loop of destionation                                                 \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_dest_base_start_##buf_name(uint64_t base_start) {               \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET, base_start);                \
    }                                                                                                                \
    /*                                                                                                               \
    /*                                                                                                               \
     * @def setup_dest_inner_loop_addrgen_0()                                                                        \
     * @def setup_dest_inner_loop_addrgen_1()                                                                        \
     *                                                                                                               \
     * @brief Function for setting parameters of inner loop of address generator                                     \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_dest_inner_loop_##buf_name(                                     \
        uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {                                                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET, stride);       \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_ADDRESS_REG_OFFSET, addr_offset); \
    }                                                                                                                \
    /*                                                                                                               \
     * @def setup_dest_outer_loop_addrgen_0()                                                                        \
     * @def setup_dest_outer_loop_addrgen_1()                                                                        \
     *                                                                                                               \
     * @brief Function for configuring outer loop of destination of  address generator                               \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_dest_outer_loop_##buf_name(                                     \
        uint64_t stride, uint64_t size, uint64_t addr_offset = 0) {                                                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_STRIDE_REG_OFFSET, stride);       \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_END_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_ADDRESS_REG_OFFSET, addr_offset); \
    }                                                                                                                \
                                                                                                                     \
    /*                                                                                                               \
     * Additional focused setup functions using config structs                                                       \
     */                                                                                                              \
    inline __attribute__((always_inline)) void setup_src_inner_loop_##buf_name(const LoopConfig& cfg) {              \
        setup_src_inner_loop_##buf_name(cfg.stride, cfg.end_addr, cfg.addr_offset);                                  \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_src_outer_loop_##buf_name(const LoopConfig& cfg) {              \
        setup_src_outer_loop_##buf_name(cfg.stride, cfg.end_addr, cfg.addr_offset);                                  \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_dest_inner_loop_##buf_name(const LoopConfig& cfg) {             \
        setup_dest_inner_loop_##buf_name(cfg.stride, cfg.end_addr, cfg.addr_offset);                                 \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_dest_outer_loop_##buf_name(const LoopConfig& cfg) {             \
        setup_dest_outer_loop_##buf_name(cfg.stride, cfg.end_addr, cfg.addr_offset);                                 \
    }                                                                                                                \
                                                                                                                     \
    /* Face size and base start functions use direct parameters - no struct wrappers needed */                       \
                                                                                                                     \
    /*                                                                                                               \
     * @def peek_src_addrgen_0()                                                                                     \
     * @def peek_src_addrgen_1()                                                                                     \
     *                                                                                                               \
     * @brief Reads current generated address for source without poping                                              \
     * and triggering new address to be generated                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t peek_src_##buf_name() { return ADDRGEN_PEEK_SRC(cmdbuf); }        \
                                                                                                                     \
    /*                                                                                                               \
     * @def pop_src_addrgen_0()                                                                                      \
     * @def pop_src_addrgen_1()                                                                                      \
     *                                                                                                               \
     * @brief Reads current generated address for source popping it                                                  \
     * and triggering new address to be generated                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t pop_src_##buf_name() { return ADDRGEN_POP_SRC(cmdbuf); }          \
    /*                                                                                                               \
     * @def pop_src_addrgen_0()                                                                                      \
     * @def pop_src_addrgen_1()                                                                                      \
     *                                                                                                               \
     * @brief Reads current generated address for source popping it                                                  \
     * and triggers/skips (pop_amount-1) src addresses afterwards                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t pop_src_##buf_name(uint64_t pop_amount) {                         \
        return ADDRGEN_POP_X_SRC(cmdbuf, pop_amount);                                                                \
    }                                                                                                                \
    /*                                                                                                               \
     * @def peek_dest_addrgen_0()                                                                                    \
     * @def peek_dest_addrgen_1()                                                                                    \
     *                                                                                                               \
     * @brief Reads current generated address for destination without poping                                         \
     * and triggering new address to be generated                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t peek_dest_##buf_name() { return ADDRGEN_PEEK_DEST(cmdbuf); }      \
                                                                                                                     \
    /*                                                                                                               \
     * @def pop_dest_addrgen_0()                                                                                     \
     * @def pop_dest_addrgen_1()                                                                                     \
     *                                                                                                               \
     * @brief Reads current generated address for destination poping it                                              \
     * and triggering new address to be generated                                                                    \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t pop_dest_##buf_name() { return ADDRGEN_POP_DEST(cmdbuf); }        \
                                                                                                                     \
    /*                                                                                                               \
     * @def pop_dest_addrgen_0()                                                                                     \
     * @def pop_dest_addrgen_1()                                                                                     \
     *                                                                                                               \
     * @brief Reads current generated address for destination poping it                                              \
     * and triggering new pop_amount-1 address to be generated and trown away                                        \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) uint64_t pop_dest_##buf_name(uint64_t pop_amount) {                        \
        return ADDRGEN_POP_X_DEST(cmdbuf, pop_amount);                                                               \
    }                                                                                                                \
                                                                                                                     \
    /*                                                                                                               \
     * @def push_src_addrgen_0()                                                                                     \
     * @def push_src_addrgen_1()                                                                                     \
     *                                                                                                               \
     * @brief Pushes generated address from address generator to command buffer                                      \
     * and triggers new address to be generated                                                                      \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void push_src_##buf_name() { ADDRGEN_PUSH_SRC(cmdbuf); }                   \
                                                                                                                     \
    /*                                                                                                               \
     * @def push_dest_addrgen_0()                                                                                    \
     * @def push_dest_addrgen_1()                                                                                    \
     *                                                                                                               \
     * @brief Pushes generated address from address generator to command buffer                                      \
     * and triggers new address to be generated                                                                      \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void push_dest_##buf_name() { ADDRGEN_PUSH_DEST(cmdbuf); }                 \
                                                                                                                     \
    /*                                                                                                               \
     * @def push_both_addrgen_0()                                                                                    \
     * @def push_both_addrgen_1()                                                                                    \
     *                                                                                                               \
     * @brief Pushes generated address from address generator to command buffer                                      \
     * and triggers new address to be generated, for both source and destination                                     \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline __attribute__((always_inline)) void push_both_##buf_name() { ADDRGEN_PUSH_BOTH(cmdbuf); }                 \
    /*                                                                                                               \
     * @def pop_both_addrgen_0()                                                                                     \
     * @def pop_both_addrgen_1()                                                                                     \
     *                                                                                                               \
     * @brief Returns the generated dest address and skips (x-1) dest addresses afterwards                           \
     * Result holds 32 bits of each dest and source addresses - {dest[31:0], src[31:]}                               \
     *                                                                                                               \
     * @note This macro creates 2 inline functions, 1 per each address generator                                     \
     */                                                                                                              \
    inline                                                                                                           \
        __attribute__((always_inline)) void pop_both_##buf_name(uint64_t src_pop_amount, uint64_t dest_pop_amount) { \
        ADDRGEN_POP_BOTH(cmdbuf, src_pop_amount, dest_pop_amount);                                                   \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void add_src_banking_##buf_name(                                           \
        uint32_t endpoint_id_shift,                                                                                  \
        uint32_t size,                                                                                               \
        uint32_t skip = 1,                                                                                           \
        uint32_t base_endpoint = 0,                                                                                  \
        uint32_t current_endpoint = 0) {                                                                             \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = endpoint_id_shift;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(                                                                                              \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET, current_endpoint);         \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_BASE_REG_OFFSET, base_endpoint);    \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SIZE_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_SKIP_REG_OFFSET, skip);             \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_src_1D_stride_##buf_name(                                       \
        uint64_t base_addr, uint64_t size, uint64_t stride) {                                                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET, base_addr);                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET, stride);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET, size);             \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_src_2D_stride_##buf_name(                                       \
        uint64_t base_addr,                                                                                          \
        uint64_t size,                                                                                               \
        uint64_t inner_end,                                                                                          \
        uint64_t inner_stride,                                                                                       \
        uint64_t outer_end,                                                                                          \
        uint64_t outer_stride) {                                                                                     \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET, base_addr);                  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_STRIDE_REG_OFFSET, inner_stride);  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_END_REG_OFFSET, inner_end);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_FACE_SIZE_REG_OFFSET, size);             \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_STRIDE_REG_OFFSET, outer_stride);  \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_END_REG_OFFSET, outer_end);        \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void add_dest_banking_##buf_name(                                          \
        uint32_t endpoint_id_shift,                                                                                  \
        uint32_t size,                                                                                               \
        uint32_t skip = 1,                                                                                           \
        uint32_t base_endpoint = 0,                                                                                  \
        uint32_t current_endpoint = 0) {                                                                             \
        TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;                                                                         \
        misc.val = TT_ROCC_ADDRESS_GEN_MISC_REG_DEFAULT;                                                             \
        misc.f.bank_offset = endpoint_id_shift;                                                                      \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET, misc.val);                  \
                                                                                                                     \
        ADDRGEN_WR_REG(                                                                                              \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_CURRENT_REG_OFFSET, current_endpoint);        \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_BASE_REG_OFFSET, base_endpoint);   \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SIZE_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_BANK_SKIP_REG_OFFSET, skip);            \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_dest_1D_stride_##buf_name(                                      \
        uint64_t base_addr, uint64_t size, uint64_t stride) {                                                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET, base_addr);                 \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET, stride);       \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET, size);            \
    }                                                                                                                \
                                                                                                                     \
    inline __attribute__((always_inline)) void setup_dest_2D_stride_##buf_name(                                      \
        uint64_t base_addr,                                                                                          \
        uint64_t size,                                                                                               \
        uint64_t inner_end,                                                                                          \
        uint64_t inner_stride,                                                                                       \
        uint64_t outer_end,                                                                                          \
        uint64_t outer_stride) {                                                                                     \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET, base_addr);                 \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_STRIDE_REG_OFFSET, inner_stride); \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_INNER_END_REG_OFFSET, inner_end);       \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_FACE_SIZE_REG_OFFSET, size);            \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_STRIDE_REG_OFFSET, outer_stride); \
        ADDRGEN_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_OUTER_END_REG_OFFSET, outer_end);       \
    }

DEFINE_ADDR_GEN(addrgen_0, ADDRGEN_0)
DEFINE_ADDR_GEN(addrgen_1, ADDRGEN_1)

#undef DEFINE_ADDR_GEN
