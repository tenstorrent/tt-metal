// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "xcustom_test.hpp"
#include "overlay_reg.h"

/**
 * @defgroup misc_instructions Miscellaneous Instructions
 * @brief Debug, utility, and interface instructions
 * @{
 */

/**
 * @brief Posts a debug code value for debugging purposes
 * @param val The debug code value to post
 *
 * This macro is non-functional and used for debugging. It posts a debug code
 * that can be observed in hardware debug interfaces.
 *
 * @details
 *
 * **Example with val=0x1234:**
 * ```
 * funct7 = (32 + 1) = 33 (0x21)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x21  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Value register     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10000100000001011001000000101101
 * Hex:    0x8405802D
 * ```
 */
#define DBG_POSTCODE(val)                       \
    {                                           \
        ROCC_INSTRUCTION_S(2, (val), (32 + 1)); \
    }

/**
 * @brief Creates a NOC (Network-on-Chip) fence to ensure memory consistency
 *
 * Causes a NOC fence which ensures that all inbound flits after this fence instruction,
 * that are already in the local NIU, have been committed to L1.
 *
 * @note When calling this you must call: asm volatile ("fence" : : : "memory");
 *
 * @details
 *
 * **Example:**
 * ```
 * funct7 = (32 + 2) = 34 (0x22)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x22  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10001000000000000000000000101101
 * Hex:    0x8800002D
 * ```
 */
#define NOC_FENCE()                    \
    {                                  \
        ROCC_INSTRUCTION(2, (32 + 2)); \
    }

/**
 * @brief Writes directly to the LLK  Interface
 * @param addr The address to write to in the LLK interface
 * @param val The value to write
 *
 * Provides fast access to the LLK Interface for direct writes.
 *
 * @details
 *
 * **Example with addr=0x1000, val=0x5678:**
 * ```
 * funct7 = (32 + 3) = 35 (0x23)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x23  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Value register     |
 * | rs1    | 19-15 | 0x0B  | Address register   |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10001100000001011100000000101101
 * Hex:    0x8C05C02D
 * ```
 */
#define LLK_INTF_WRITE(addr, val)                        \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (addr), (val), (32 + 3)); \
    }

/**
 * @brief Reads directly from the LLK  Interface
 * @param addr The address to read from in the LLK interface
 * @return The value read from the LLK interface
 *
 * Provides fast access to the LLK Interface for direct reads.
 *
 * @details
 * **Example with addr=0x1000, result in a0:**
 * ```
 * funct7 = (32 + 3) = 35 (0x23)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x23  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Address register   |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10001100000001011010000000101101
 * Hex:    0x8C05A02D
 * ```
 */
#define LLK_INTF_READ(addr)                             \
    [&]() {                                             \
        uint64_t result;                                \
        ROCC_INSTRUCTION_DS(2, result, addr, (32 + 3)); \
        return result;                                  \
    }()

/**
 * @brief Writes to FDS register file
 * @param addr The register address to write to
 * @param val The value to write to the register
 *
 * Provides access to the FDS register file for configuration and control.
 *
 * @details
 * **Example with addr=0x2000, val=0x9ABC:**
 * ```
 * funct7 = (32 + 4) = 36 (0x24)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x24  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Value register     |
 * | rs1    | 19-15 | 0x0B  | Address register   |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10010000000001011100000000101101
 * Hex:    0x9005C02D
 * ```
 */
#define FDS_INTF_WRITE(addr, val)                        \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (addr), (val), (32 + 4)); \
    }

/**
 * @brief Reads from FDS register file
 * @param addr The register address to read from
 * @return The value read from the register
 *
 * Provides access to the FDS register file for status and configuration reading.
 *
 * @details
 * **Example with addr=0x2000, result in a0:**
 * ```
 * funct7 = (32 + 4) = 36 (0x24)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x24  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Address register   |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 10010000000001011010000000101101
 * Hex:    0x9005A02D
 * ```
 */
#define FDS_INTF_READ(addr)                             \
    [&]() {                                             \
        uint64_t result;                                \
        ROCC_INSTRUCTION_DS(2, result, addr, (32 + 4)); \
        return result;                                  \
    }()

/** @} */  // end of misc_instructions

/**
 * @defgroup context_switch Context Switch Operations
 * @brief Context switching functionality for core management
 * @{
 */

/**
 * @brief Allocates a slot on the context switch RAM
 * @return Slot ID if successful, 0 if no slot is available
 *
 * Allocates a slot on the context switch RAM for storing core context.
 * Only even cores have this functionality.
 *
 * @details
 * **Example with result in a0:**
 * ```
 * funct7 = 3 (0x03)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x03  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 00001100000000001010000000101101
 * Hex:    0x0C002A2D
 * ```
 */
#define CS_ALLOC                          \
    [&]() {                               \
        uint64_t result;                  \
        ROCC_INSTRUCTION_D(2, result, 3); \
        return result;                    \
    }()

/**
 * @brief Deallocates the given context switch slot
 * @param context The slot ID to deallocate
 *
 * Deallocates the specified slot in the context switch RAM.
 * @warning You should not pass slot 0 to this function.
 * Only even cores have this functionality.
 *
 * @details
 * **Example with context=5:**
 * ```
 * funct7 = 2 (0x02)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x02  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Context register   |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 00001000000001011001000000101101
 * Hex:    0x0805802D
 * ```
 */
#define CS_DEALLOC(context)                  \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 2); \
    }

/**
 * @brief Saves the current context to the given slot
 * @param context The slot ID to save the context to (must be non-zero)
 *
 * Saves the current core context to the specified slot in context switch RAM.
 * @note Call asm volatile ("fence" : : : "memory"); after this instruction.
 *
 * @details
 * **Example with context=3:**
 * ```
 * funct7 = 1 (0x01)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x01  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Context register   |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 00000100000001011001000000101101
 * Hex:    0x0405802D
 * ```
 */
#define CS_SAVE(context)                     \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 1); \
    }

/**
 * @brief Restores the context from the given slot
 * @param context The slot ID to restore the context from (must be non-zero)
 *
 * Restores the core context from the specified slot in context switch RAM.
 * @note Call asm volatile ("fence" : : : "memory"); after this instruction.
 *
 * @details
 * **Example with context=7:**
 * ```
 * funct7 = 0 (0x00)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x00  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Context register   |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2D  | CUSTOM_2          |
 *
 * Binary: 00000000000001011001000000101101
 * Hex:    0x0005802D
 * ```
 */
#define CS_RESTORE(context)                  \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 0); \
    }

/** @} */  // end of context_switch

/**
 * @defgroup address_generation Address Generation
 * @brief Address generator operations for memory access patterns
 * @{
 */

/**
 * @brief Writes to an address generator register
 * @param cmdbuf The command buffer ID (0-based)
 * @param reg The register number to write to
 * @param val The value to write to the register
 *
 * Configures address generator registers for memory access pattern generation.
 *
 * @details
 * **Example with cmdbuf=1, reg=16, val=0x1234:**
 * ```
 * funct7 = (1*32) + (16/8) = 32 + 2 = 34 (0x22)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x22  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Value register     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10001000000001011001000000101011
 * Hex:    0x8805802B
 * ```
 */
#define ADDRGEN_WR_REG(cmdbuf, reg, val)                               \
    {                                                                  \
        ROCC_INSTRUCTION_S(1, (val), (((cmdbuf) * 32) + ((reg) / 8))); \
    }

/**
 * @brief Reads from an address generator register
 * @param cmdbuf The command buffer ID (0-based)
 * @param reg The register number to read from
 * @return The value read from the register
 *
 * Reads the current value of an address generator register.
 *
 * @details
 * **Example with cmdbuf=0, reg=8, result in a0:**
 * ```
 * funct7 = (0*32) + (8/8) = 0 + 1 = 1 (0x01)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x01  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 00000100000000001010000000101011
 * Hex:    0x04002A2B
 * ```
 */
#define ADDRGEN_RD_REG(cmdbuf, reg)                                     \
    [&]() {                                                             \
        uint64_t result;                                                \
        ROCC_INSTRUCTION_D(1, result, (((cmdbuf) * 32) + ((reg) / 8))); \
        return result;                                                  \
    }()

/**
 * @brief Resets all address generator registers to default values
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Resets all address generator registers to their default values as specified
 * in the RDL (Register Description Language) file.
 *
 * @details
 * **Example with cmdbuf=1:**
 * ```
 * funct7 = (1*32) + 26 = 32 + 26 = 58 (0x3A)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3A  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11101000000000000000000000101011
 * Hex:    0xE800002B
 * ```
 */
#define ADDRGEN_RESET(cmdbuf)                      \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 26)); \
    }

/**
 * @brief Resets address generator counters while preserving base, size, and strides
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Resets only the address generator counters while keeping the base addresses,
 * sizes, and strides intact.
 *
 * @details
 * **Example with cmdbuf=0:**
 * ```
 * funct7 = (0*32) + 26 = 0 + 26 = 26 (0x1A)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x1A  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Zero value         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 01101000000000000000000000101011
 * Hex:    0x6800002B
 * ```
 */
#define ADDRGEN_RESET_COUNTERS(cmdbuf)                    \
    {                                                     \
        ROCC_INSTRUCTION_S(1, (0), ((cmdbuf) * 32 + 26)); \
    }

/**
 * @brief Returns the generated source address without generating a new one
 * @param cmdbuf The command buffer ID (0-based)
 * @return The current source address
 *
 * Peeks at the current source address without advancing the address generator.
 *
 * @details
 * **Example with cmdbuf=2, result in a0:**
 * ```
 * funct7 = (2*32) + 31 = 64 + 31 = 95 (0x5F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x5F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10111110000000001010000000101011
 * Hex:    0xBE002A2B
 * ```
 */
#define ADDRGEN_PEEK_SRC(cmdbuf)                             \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(1, result, ((cmdbuf) * 32 + 31)); \
        return result;                                       \
    }()

/**
 * @brief Returns the generated source address and generates a new one
 * @param cmdbuf The command buffer ID (0-based)
 * @return The current source address
 *
 * Returns the current source address and advances the address generator
 * to generate the next address.
 *
 * @details
 * **Example with cmdbuf=1, result in a0:**
 * ```
 * funct7 = (1*32) + 31 = 32 + 31 = 63 (0x3F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Zero value         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111100000000001010000000101011
 * Hex:    0xFC002A2B
 * ```
 */
#define ADDRGEN_POP_SRC(cmdbuf)                                  \
    [&]() {                                                      \
        uint64_t result;                                         \
        ROCC_INSTRUCTION_DS(1, result, 0, ((cmdbuf) * 32 + 31)); \
        return result;                                           \
    }()

/**
 * @brief Returns the generated source address and skips addresses
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_src Number of source addresses to skip (skip_src-1 addresses)
 * @return The current source address
 *
 * Returns the current source address and skips (skip_src-1) source addresses
 * in the sequence.
 *
 * @details
 * **Example with cmdbuf=0, skip_src=3, result in a0:**
 * ```
 * funct7 = (0*32) + 31 = 0 + 31 = 31 (0x1F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x1F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Skip count         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 01111100000001011010000000101011
 * Hex:    0x7C05A02B
 * ```
 */
#define ADDRGEN_POP_X_SRC(cmdbuf, skip_src)                        \
    [&]() {                                                        \
        uint64_t rs1 = skip_src;                                   \
        uint64_t result;                                           \
        ROCC_INSTRUCTION_DS(1, result, rs1, ((cmdbuf) * 32 + 31)); \
        return result;                                             \
    }()

/**
 * @brief Returns the generated destination address without generating a new one
 * @param cmdbuf The command buffer ID (0-based)
 * @return The current destination address
 *
 * Peeks at the current destination address without advancing the address generator.
 *
 * @details
 * **Example with cmdbuf=1, result in a0:**
 * ```
 * funct7 = (1*32) + 30 = 32 + 30 = 62 (0x3E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111000000000001010000000101011
 * Hex:    0xF8002A2B
 * ```
 */
#define ADDRGEN_PEEK_DEST(cmdbuf)                            \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(1, result, ((cmdbuf) * 32 + 30)); \
        return result;                                       \
    }()

/**
 * @brief Returns the generated destination address and generates a new one
 * @param cmdbuf The command buffer ID (0-based)
 * @return The current destination address
 *
 * Returns the current destination address and advances the address generator
 * to generate the next address.
 *
 * @details
 * **Example with cmdbuf=2, result in a0:**
 * ```
 * funct7 = (2*32) + 30 = 64 + 30 = 94 (0x5E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x5E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Zero value         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10111100000000001010000000101011
 * Hex:    0xBC002A2B
 * ```
 */
#define ADDRGEN_POP_DEST(cmdbuf)                                 \
    [&]() {                                                      \
        uint64_t result;                                         \
        ROCC_INSTRUCTION_DS(1, result, 0, ((cmdbuf) * 32 + 30)); \
        return result;                                           \
    }()

/**
 * @brief Returns the generated destination address and skips addresses
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_dest Number of destination addresses to skip (skip_dest-1 addresses)
 * @return The current destination address
 *
 * Returns the current destination address and skips (skip_dest-1) destination
 * addresses in the sequence.
 *
 * @details
 * **Example with cmdbuf=0, skip_dest=5, result in a0:**
 * ```
 * funct7 = (0*32) + 30 = 0 + 30 = 30 (0x1E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x1E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Skip count         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 01111000000001011010000000101011
 * Hex:    0x7805A02B
 * ```
 */
#define ADDRGEN_POP_X_DEST(cmdbuf, skip_dest)                      \
    [&]() {                                                        \
        uint64_t rs1 = skip_dest;                                  \
        uint64_t result;                                           \
        ROCC_INSTRUCTION_DS(1, result, rs1, ((cmdbuf) * 32 + 30)); \
        return result;                                             \
    }()

/**
 * @brief Returns both source and destination addresses with skipping
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_src Number of source addresses to skip
 * @param skip_dest Number of destination addresses to skip
 * @return Packed addresses: {dest[31:0], src[31:0]}
 *
 * Returns both generated addresses and skips the specified number of addresses.
 * The result contains 32 bits of each destination and source address packed together.
 *
 * @details
 * **Example with cmdbuf=1, skip_src=2, skip_dest=3, result in a0:**
 * ```
 * funct7 = (1*32) + 25 = 32 + 25 = 57 (0x39)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x39  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Skip dest count    |
 * | rs1    | 19-15 | 0x0B  | Skip src count     |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11100100000001011100000000101011
 * Hex:    0xE405C02B
 * ```
 */
#define ADDRGEN_POP_BOTH(cmdbuf, skip_src, skip_dest)                    \
    [&]() {                                                              \
        uint64_t rs1 = skip_src;                                         \
        uint64_t rs2 = skip_dest;                                        \
        uint64_t result;                                                 \
        ROCC_INSTRUCTION_DSS(1, result, rs1, rs2, ((cmdbuf) * 32 + 25)); \
        return result;                                                   \
    }()

/**
 * @brief Pushes source address to command buffer and generates new address
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Pushes the generated source address to the command buffer associated with
 * this address generator and generates a new address.
 *
 * @details
 * **Example with cmdbuf=0:**
 * ```
 * funct7 = (0*32) + 28 = 0 + 28 = 28 (0x1C)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x1C  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 01110000000000000000000000101011
 * Hex:    0x7000002B
 * ```
 */
#define ADDRGEN_PUSH_SRC(cmdbuf)                   \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 28)); \
    }

/**
 * @brief Pushes source address and generates multiple new addresses
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_src Number of new source addresses to generate
 *
 * Pushes the generated source address to the command buffer and generates
 * (skip_src) new addresses.
 *
 * @details
 * **Example with cmdbuf=1, skip_src=4:**
 * ```
 * funct7 = (1*32) + 28 = 32 + 28 = 60 (0x3C)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3C  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Skip count         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11110000000001011001000000101011
 * Hex:    0xF005802B
 * ```
 */
#define ADDRGEN_PUSH_SRC_POP_X(cmdbuf, skip_src)          \
    {                                                     \
        uint64_t rs1 = skip_src;                          \
        ROCC_INSTRUCTION_S(1, rs1, ((cmdbuf) * 32 + 28)); \
    }

/**
 * @brief Pushes destination address to command buffer and generates new address
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Pushes the generated destination address to the command buffer associated with
 * this address generator and generates a new address.
 *
 * @details
 * **Example with cmdbuf=2:**
 * ```
 * funct7 = (2*32) + 27 = 64 + 27 = 91 (0x5B)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x5B  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10110110000000000000000000101011
 * Hex:    0xB600002B
 * ```
 */
#define ADDRGEN_PUSH_DEST(cmdbuf)                  \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 27)); \
    }

/**
 * @brief Pushes destination address and generates multiple new addresses
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_dest Number of new destination addresses to generate
 *
 * Pushes the generated destination address to the command buffer and generates
 * (skip_dest) new addresses.
 *
 * @details
 * **Example with cmdbuf=0, skip_dest=6:**
 * ```
 * funct7 = (0*32) + 27 = 0 + 27 = 27 (0x1B)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x1B  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Skip count         |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 01101100000001011001000000101011
 * Hex:    0x6C05802B
 * ```
 */
#define ADDRGEN_PUSH_DEST_POP_X(cmdbuf, skip_dest)        \
    {                                                     \
        uint64_t rs1 = skip_dest;                         \
        ROCC_INSTRUCTION_S(1, rs1, ((cmdbuf) * 32 + 27)); \
    }

/**
 * @brief Pushes both addresses to command buffer and generates new addresses
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Pushes both generated addresses to the command buffer associated with
 * this address generator and generates new addresses.
 *
 * @details
 * **Example with cmdbuf=1:**
 * ```
 * funct7 = (1*32) + 29 = 32 + 29 = 61 (0x3D)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3D  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11110100000000000000000000101011
 * Hex:    0xF400002B
 * ```
 */
#define ADDRGEN_PUSH_BOTH(cmdbuf)                  \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 29)); \
    }

/**
 * @brief Pushes both addresses and generates multiple new addresses
 * @param cmdbuf The command buffer ID (0-based)
 * @param skip_src Number of new source addresses to generate
 * @param skip_dest Number of new destination addresses to generate
 *
 * Pushes both generated addresses to the command buffer and generates
 * new source and destination addresses.
 *
 * @details
 * **Example with cmdbuf=2, skip_src=1, skip_dest=2:**
 * ```
 * funct7 = (2*32) + 29 = 64 + 29 = 93 (0x5D)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x5D  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Skip dest count    |
 * | rs1    | 19-15 | 0x0B  | Skip src count     |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10111010000001011100000000101011
 * Hex:    0xBA05C02B
 * ```
 */
#define ADDRGEN_PUSH_BOTH_POP_X(cmdbuf, skip_src, skip_dest)    \
    {                                                           \
        uint64_t rs1 = skip_src;                                \
        uint64_t rs2 = skip_dest;                               \
        ROCC_INSTRUCTION_SS(1, rs1, rs2, ((cmdbuf) * 32 + 29)); \
    }

/** @} */  // end of address_generation

/**
 * @defgroup command_buffers Command Buffer Operations
 * @brief Command buffer management for transaction processing
 * @{
 */

/**
 * @brief Writes to a command buffer register
 * @param cmdbuf The command buffer ID (0-based)
 * @param reg The register number to write to
 * @param val The value to write to the register
 *
 * Configures command buffer registers for transaction processing.
 *
 * @details
 * **Example with cmdbuf=1, reg=16, val=0x1234:**
 * ```
 * funct7 = (1*64) + (16/8) = 64 + 2 = 66 (0x42)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x42  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Value register     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10000100000001011001000000001011
 * Hex:    0x8405800B
 * ```
 */
#define CMDBUF_WR_REG(cmdbuf, reg, val)                                \
    {                                                                  \
        ROCC_INSTRUCTION_S(0, (val), (((cmdbuf) * 64) + ((reg) / 8))); \
    }

/**
 * @brief Reads from a command buffer register
 * @param cmdbuf The command buffer ID (0-based)
 * @param reg The register number to read from
 * @return The value read from the register
 *
 * Reads the current value of a command buffer register.
 *
 * @details
 * **Example with cmdbuf=0, reg=24, result in a0:**
 * ```
 * funct7 = (0*64) + (24/8) = 0 + 3 = 3 (0x03)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x03  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 00001100000000001010000000001011
 * Hex:    0x0C002A0B
 * ```
 */
#define CMDBUF_RD_REG(cmdbuf, reg)                                      \
    [&]() {                                                             \
        uint64_t result;                                                \
        ROCC_INSTRUCTION_D(0, result, (((cmdbuf) * 64) + ((reg) / 8))); \
        return result;                                                  \
    }()

/**
 * @brief Gets available VC space for the currently programmed request VC
 * @param cmdbuf The command buffer ID (0-based)
 * @return Amount of available VC space
 *
 * Returns the amount of virtual channel space available for the currently
 * programmed request virtual channel.
 *
 * @details
 * **Example with cmdbuf=1, result in a0:**
 * ```
 * funct7 = (1*64) + 62 = 64 + 62 = 126 (0x7E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11111100000000001010000000001011
 * Hex:    0xFC002A0B
 * ```
 */
#define CMDBUF_GET_VC_SPACE(cmdbuf)                          \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 62)); \
        return result;                                       \
    }()

/**
 * @brief Gets available VC space for a specific virtual channel
 * @param cmdbuf The command buffer ID (0-based)
 * @param vc The virtual channel number to query
 * @return Amount of available VC space for the specified VC
 *
 * Returns the amount of virtual channel space available for the specified
 * virtual channel.
 *
 * @details
 * **Example with cmdbuf=0, vc=5, result in a0:**
 * ```
 * funct7 = (0*64) + 62 = 0 + 62 = 62 (0x3E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | VC number          |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11111000000001011010000000001011
 * Hex:    0xF805A00B
 * ```
 */
#define CMDBUF_GET_VC_SPACE_VC(cmdbuf, vc)                          \
    [&]() {                                                         \
        uint64_t result;                                            \
        ROCC_INSTRUCTION_DS(0, result, (vc), ((cmdbuf) * 64 + 62)); \
        return result;                                              \
    }()

/**
 * @brief Gets number of outstanding write transactions sent
 * @param cmdbuf The command buffer ID (0-based)
 * @return Number of outstanding write transactions
 *
 * Returns the number of outstanding write transactions that need to be sent
 * from the local core using the programmed write-ack transaction ID.
 *
 * @details
 * **Example with cmdbuf=2, result in a0:**
 * ```
 * funct7 = (2*64) + 61 = 128 + 61 = 189 (0xBD)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xBD  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10111101000000001010000000001011
 * Hex:    0xBD002A0B
 * ```
 */
#define CMDBUF_WR_SENT(cmdbuf)                               \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 61)); \
        return result;                                       \
    }()

/**
 * @brief Gets number of outstanding write transactions for a specific transaction ID
 * @param cmdbuf The command buffer ID (0-based)
 * @param trid The transaction ID to query
 * @return Number of outstanding write transactions for the specified TRID
 *
 * Returns the number of outstanding write transactions that need to be sent
 * from the local core using the given transaction ID.
 *
 * @details
 * **Example with cmdbuf=0, trid=7, result in a0:**
 * ```
 * funct7 = (0*64) + 61 = 0 + 61 = 61 (0x3D)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3D  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Transaction ID     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11110100000001011010000000001011
 * Hex:    0xF405A00B
 * ```
 */
#define CMDBUF_WR_SENT_TRID(cmdbuf, trid)                             \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 61)); \
        return result;                                                \
    }()

/**
 * @brief Gets number of pending transaction acknowledgments
 * @param cmdbuf The command buffer ID (0-based)
 * @return Number of pending transaction acknowledgments
 *
 * Returns the number of acknowledgments that are yet to be received for
 * non-posted read or write transactions using the programmed tr-ack transaction ID.
 *
 * @details
 * **Example with cmdbuf=1, result in a0:**
 * ```
 * funct7 = (1*64) + 60 = 64 + 60 = 124 (0x7C)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7C  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11111000000000001010000000001011
 * Hex:    0xF8002A0B
 * ```
 */
#define CMDBUF_TR_ACK(cmdbuf)                                \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 60)); \
        return result;                                       \
    }()

/**
 * @brief Gets number of pending transaction acknowledgments for a specific TRID
 * @param cmdbuf The command buffer ID (0-based)
 * @param trid The transaction ID to query
 * @return Number of pending transaction acknowledgments for the specified TRID
 *
 * Returns the number of acknowledgments that are yet to be received for
 * non-posted read or write transactions using the given transaction ID.
 *
 * @details
 * **Example with cmdbuf=2, trid=3, result in a0:**
 * ```
 * funct7 = (2*64) + 60 = 128 + 60 = 188 (0xBC)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xBC  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Transaction ID     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10111100000001011010000000001011
 * Hex:    0xBC05A00B
 * ```
 */
#define CMDBUF_TR_ACK_TRID(cmdbuf, trid)                              \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 60)); \
        return result;                                                \
    }()

/**
 * @brief Resets all command buffer registers to default values
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Resets all command buffer registers to their default values as specified
 * in the RDL (Register Description Language) file.
 *
 * @details
 * **Example with cmdbuf=1:**
 * ```
 * funct7 = (1*64) + 59 = 64 + 59 = 123 (0x7B)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7B  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11110110000000000000000000001011
 * Hex:    0xF600000B
 * ```
 */
#define CMDBUF_RESET(cmdbuf)                       \
    {                                              \
        ROCC_INSTRUCTION(0, ((cmdbuf) * 64 + 59)); \
    }

/**
 * @brief Gets available IDMA VC space for the currently programmed request VC
 * @param cmdbuf The command buffer ID (0-based)
 * @return Amount of available IDMA VC space
 *
 * Returns the amount of IDMA (Internal DMA) virtual channel space available
 * for the currently programmed request virtual channel.
 *
 * @details
 * **Example with cmdbuf=2, result in a0:**
 * ```
 * funct7 = (2*64) + 58 = 128 + 58 = 186 (0xBA)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xBA  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10111010000000001010000000001011
 * Hex:    0xBA002A0B
 * ```
 */
#define CMDBUF_IDMA_GET_VC_SPACE(cmdbuf)                     \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 58)); \
        return result;                                       \
    }()

/**
 * @brief Gets available IDMA VC space for a specific virtual channel
 * @param cmdbuf The command buffer ID (0-based)
 * @param vc The virtual channel number to query
 * @return Amount of available IDMA VC space for the specified VC
 *
 * Returns the amount of IDMA virtual channel space available for the specified
 * virtual channel.
 *
 * @details
 * **Example with cmdbuf=0, vc=4, result in a0:**
 * ```
 * funct7 = (0*64) + 58 = 0 + 58 = 58 (0x3A)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3A  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | VC number          |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11101000000001011010000000001011
 * Hex:    0xE805A00B
 * ```
 */
#define CMDBUF_IDMA_GET_VC_SPACE_VC(cmdbuf, vc)                     \
    [&]() {                                                         \
        uint64_t result;                                            \
        ROCC_INSTRUCTION_DS(0, result, (vc), ((cmdbuf) * 64 + 58)); \
        return result;                                              \
    }()

/**
 * @brief Gets number of pending IDMA transaction acknowledgments
 * @param cmdbuf The command buffer ID (0-based)
 * @return Number of pending IDMA transaction acknowledgments
 *
 * Returns the number of IDMA acknowledgments that are yet to be received for
 * non-posted read or write transactions using the programmed tr-ack transaction ID.
 *
 * @details
 * **Example with cmdbuf=1, result in a0:**
 * ```
 * funct7 = (1*64) + 57 = 64 + 57 = 121 (0x79)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x79  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11110010000000001010000000001011
 * Hex:    0xF2002A0B
 * ```
 */
#define CMDBUF_IDMA_TR_ACK(cmdbuf)                           \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 57)); \
        return result;                                       \
    }()

/**
 * @brief Gets number of pending IDMA transaction acknowledgments for a specific TRID
 * @param cmdbuf The command buffer ID (0-based)
 * @param trid The transaction ID to query
 * @return Number of pending IDMA transaction acknowledgments for the specified TRID
 *
 * Returns the number of IDMA acknowledgments that are yet to be received for
 * non-posted read or write transactions using the given transaction ID.
 *
 * @details
 * **Example with cmdbuf=2, trid=6, result in a0:**
 * ```
 * funct7 = (2*64) + 57 = 128 + 57 = 185 (0xB9)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xB9  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Transaction ID     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10111001000001011010000000001011
 * Hex:    0xB905A00B
 * ```
 */
#define CMDBUF_IDMA_TR_ACK_TRID(cmdbuf, trid)                         \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 57)); \
        return result;                                                \
    }()

/**
 * @brief Issues a transaction using the command buffer
 * @param cmdbuf The command buffer ID (0-based)
 *
 * Issues a transaction using the currently configured command buffer settings.
 *
 * @details
 * **Example with cmdbuf=0:**
 * ```
 * funct7 = (0*64) + 63 = 0 + 63 = 63 (0x3F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x3F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11111100000000000000000000001011
 * Hex:    0xFC00000B
 * ```
 */
#define CMDBUF_ISSUE_TRANS(cmdbuf)                 \
    {                                              \
        ROCC_INSTRUCTION(0, ((cmdbuf) * 64 + 63)); \
    }

/**
 * @brief Issues an inline transaction with immediate data
 * @param cmdbuf The command buffer ID (0-based)
 * @param val The immediate value for the inline transaction
 *
 * Issues an inline transaction with immediate data. See issue_write_inline_
 * in software/pkernels/common/cmdbuf_accel.hpp for encoding details.
 *
 * @details
 * **Example with cmdbuf=1, val=0xABCD:**
 * ```
 * funct7 = (1*64) + 63 = 64 + 63 = 127 (0x7F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Inline value        |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11111110000001011001000000001011
 * Hex:    0xFE05800B
 * ```
 */
#define CMDBUF_ISSUE_INLINE_TRANS(cmdbuf, val)              \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (val), ((cmdbuf) * 64 + 63)); \
    }

/**
 * @brief Issues an inline transaction with extended parameters
 * @param cmdbuf The command buffer ID (0-based)
 * @param val The immediate value for the inline transaction
 * @param rs2 The second source register containing destination address and transaction parameters
 *
 * Issues an inline transaction with extended parameters. See issue_write_inline_len_
 * in software/pkernels/common/cmdbuf_accel.hpp for encoding details.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`)
 * - Bits [60]: Has XY flag (`has_xy`)
 * - Bits [59:57]: Size in bytes minus 1 (`size_bytes-1`)
 * - Bits [47:32]: Destination coordinates (`dest_coords`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `dest_addr  | dest_coords << 32| ((size_bytes-1) << 57) | (has_xy << 60) | (posted << 61) | (snoop << 62) |
 * (flush << 63)`
 *
 * **Example with cmdbuf=2, val=0x1234, dest_addr=0x5678, size_bytes=0x1000, flags=0:**
 * ```
 * rs2 = 0x5678 | ((0x1000-1) << 57) | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x5678 | (0xFFF << 57)
 *
 * funct7 = (2*64) + 63 = 128 + 63 = 191 (0xBF)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xBF  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Dest addr + params |
 * | rs1    | 19-15 | 0x0B  | Inline value        |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10111111000001011100000000001011
 * Hex:    0xBF05C00B
 * ```
 */
#define CMDBUF_ISSUE_INLINE_ADDR_TRANS(cmdbuf, val, rs2)            \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (val), (rs2), ((cmdbuf) * 64 + 63)); \
    }

/**
 * @brief Issues a read transaction using only destination address and length
 * @param cmdbuf The command buffer ID (0-based)
 * @param rs1 The first source register containing destination address and length
 *
 * Issues a read transaction using only destination address and length. This variant can be used if you only want to
 * change the destination address and length while using default values for source address and transaction flags.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(len_bytes << 32) | dest_addr`
 *
 * **Example with cmdbuf=0, len_bytes=0x1000, dest_addr=0x1000:**
 * ```
 * rs1 = (0x1000 << 32) | 0x1000 = 0x100000001000
 *
 * funct7 = (0*64) + 56 = 0 + 56 = 56 (0x38)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x38  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Dest addr + length |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11100000000001011001000000001011
 * Hex:    0xE005800B
 * ```
 */
#define CMDBUF_ISSUE_READ1_TRANS(cmdbuf, rs1)               \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (rs1), ((cmdbuf) * 64 + 56)); \
    }

/**
 * @brief Issues a read transaction with both addresses and transaction parameters
 * @param cmdbuf The command buffer ID (0-based)
 * @param rs1 The first source register containing destination address and length
 * @param rs2 The second source register containing source address and transaction flags
 *
 * Issues a read transaction with both source and destination addresses.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(len_bytes << 32) | dest_addr`
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`) - not used in this function
 * - Bits [60]: Has XY flag (`has_xy`)  - if set, will look at coordinates from address in rs2
 * - Bits [59:32]: Source coordinates, created with NOC_XY_COORD macro
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(flush << 63) | (snoop << 62) | (posted << 61) | (has_xy << 60) | ((src_coordinate) << 32) | (src_addr)`
 *
 * **Example with cmdbuf=1, len_bytes=0x1000, src_addr=0x2000, dest_addr=0x100, flags=0:**
 * ```
 * rs1 = (0x1000 << 32) | 0x100 = 0x1000000000100
 * rs2 = 0x2000 | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x2000
 *
 * funct7 = (1*64) + 56 = 64 + 56 = 120 (0x78)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x78  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Src addr + flags   |
 * | rs1    | 19-15 | 0x0B  | Dest addr + length |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11110000000001011100000000001011
 * Hex:    0xF005C00B
 * ```
 */
#define CMDBUF_ISSUE_READ2_TRANS(cmdbuf, rs1, rs2)                  \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (rs1), (rs2), ((cmdbuf) * 64 + 56)); \
    }

/**
 * @brief Issues a write transaction using only source address and length
 * @param cmdbuf The command buffer ID (0-based)
 * @param rs1 The first source register containing source address and length
 *
 * Issues a write transaction using only source address and length. This variant can be used if you only want to change
 * the source address and length while using default values for destination address and transaction flags.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(len_bytes << 32) | src_addr`
 *
 * **Example with cmdbuf=2, len_bytes=0x1000, src_addr=0x3000:**
 * ```
 * rs1 = (0x1000 << 32) | 0x3000 = 0x100000003000
 *
 * funct7 = (2*64) + 55 = 128 + 55 = 183 (0xB7)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0xB7  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Src addr + length  |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 10110111000001011001000000001011
 * Hex:    0xB705800B
 * ```
 */
#define CMDBUF_ISSUE_WRITE1_TRANS(cmdbuf, rs1)              \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (rs1), ((cmdbuf) * 64 + 55)); \
    }

/**
 * @brief Issues a write transaction with both addresses and transaction parameters
 * @param cmdbuf The command buffer ID (0-based)
 * @param rs1 The first source register containing source address and length
 * @param rs2 The second source register containing destination address and transaction flags
 *
 * Issues a write transaction with both source and destination addresses.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(len_bytes << 32) | src_addr`
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`)
 * - Bits [60]: Has XY flag (`has_xy`)  - if set, will look at coordinates from address in rs1
 * - Bits [59:32]: Destination coordinates, created with NOC_XY_COORD macro
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(flush << 63) | (snoop << 62) | (posted << 61) | (has_xy << 60) | (dest_coordinate << 32) | dest_addr`
 *
 * **Example with cmdbuf=0, len_bytes=0x1000, src_addr=0x4000, dest_addr=0x200, flags=0:**
 * ```
 * rs1 = (0x1000 << 32) | 0x4000 = 0x1000000004000
 * rs2 = 0x200 | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x200
 *
 * funct7 = (0*64) + 55 = 0 + 55 = 55 (0x37)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x37  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Dest addr + flags  |
 * | rs1    | 19-15 | 0x0B  | Src addr + length  |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x0B  | CUSTOM_0          |
 *
 * Binary: 11011100000001011100000000001011
 * Hex:    0xDC05C00B
 * ```
 */
#define CMDBUF_ISSUE_WRITE2_TRANS(cmdbuf, rs1, rs2)                 \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (rs1), (rs2), ((cmdbuf) * 64 + 55)); \
    }

/** @} */  // end of command_buffers

/**
 * @defgroup simple_command_buffers Simple Command Buffer Operations
 * @brief Simplified command buffer operations for basic use cases
 *
 * These macros provide simplified access to command buffer operations
 * without requiring explicit command buffer ID parameters.
 * @{
 */

/**
 * @brief Writes to a simple command buffer register
 * @param reg The register number to write to
 * @param val The value to write to the register
 *
 * Simplified command buffer register write operation.
 *
 * @details
 * **Example with reg=16, val=0x1234:**
 * ```
 * funct7 = 64 + (16/8) = 64 + 2 = 66 (0x42)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x42  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Value register     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10000100000001011001000000101011
 * Hex:    0x8405802B
 * ```
 */
#define SCMDBUF_WR_REG(reg, val)                          \
    {                                                     \
        ROCC_INSTRUCTION_S(1, (val), (64 + ((reg) / 8))); \
    }

/**
 * @brief Reads from a simple command buffer register
 * @param reg The register number to read from
 * @return The value read from the register
 *
 * Simplified command buffer register read operation.
 *
 * @details
 * **Example with reg=24, result in a0:**
 * ```
 * funct7 = 64 + (24/8) = 64 + 3 = 67 (0x43)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x43  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 10000110000000001010000000101011
 * Hex:    0x86002A2B
 * ```
 */
#define SCMDBUF_RD_REG(reg)                                \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_D(1, result, (64 + ((reg) / 8))); \
        return result;                                     \
    }()

/**
 * @brief Gets available VC space for the simple command buffer
 * @return Amount of available VC space
 *
 * Returns the amount of virtual channel space available for the currently
 * programmed request virtual channel in the simple command buffer.
 *
 * @details
 * **Example with result in a0:**
 * ```
 * funct7 = 64 + 62 = 126 (0x7E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111100000000001010000000101011
 * Hex:    0xFC002A2B
 * ```
 */
#define SCMDBUF_GET_VC_SPACE()                    \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 62)); \
        return result;                            \
    }()

/**
 * @brief Gets available VC space for a specific virtual channel in simple command buffer
 * @param vc The virtual channel number to query
 * @return Amount of available VC space for the specified VC
 *
 * Returns the amount of virtual channel space available for the specified
 * virtual channel in the simple command buffer.
 *
 * @details
 * **Example with vc=5, result in a0:**
 * ```
 * funct7 = 64 + 62 = 126 (0x7E)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7E  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | VC number          |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111100000001011010000000101011
 * Hex:    0xFC05A02B
 * ```
 */
#define SCMDBUF_GET_VC_SPACE_VC(vc)                      \
    [&]() {                                              \
        uint64_t result;                                 \
        ROCC_INSTRUCTION_DS(1, result, (vc), (64 + 62)); \
        return result;                                   \
    }()

/**
 * @brief Gets number of outstanding write transactions sent from simple command buffer
 * @return Number of outstanding write transactions
 *
 * Returns the number of outstanding write transactions that need to be sent
 * from the local core using the programmed write-ack transaction ID.
 *
 * @details
 * **Example with result in a0:**
 * ```
 * funct7 = 64 + 61 = 125 (0x7D)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7D  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111010000000001010000000101011
 * Hex:    0xFA002A2B
 * ```
 */
#define SCMDBUF_WR_SENT()                         \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 61)); \
        return result;                            \
    }()

/**
 * @brief Gets number of outstanding write transactions for a specific transaction ID
 * @param trid The transaction ID to query
 * @return Number of outstanding write transactions for the specified TRID
 *
 * Returns the number of outstanding write transactions that need to be sent
 * from the local core using the given transaction ID.
 *
 * @details
 * **Example with trid=7, result in a0:**
 * ```
 * funct7 = 64 + 61 = 125 (0x7D)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7D  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Transaction ID     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111010000001011010000000101011
 * Hex:    0xFA05A02B
 * ```
 */
#define SCMDBUF_WR_SENT_TRID(trid)                         \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(1, result, (trid), (64 + 61)); \
        return result;                                     \
    }()

/**
 * @brief Gets number of pending transaction acknowledgments for simple command buffer
 * @return Number of pending transaction acknowledgments
 *
 * Returns the number of acknowledgments that are yet to be received for
 * non-posted read or write transactions using the programmed tr-ack transaction ID.
 *
 * @details
 * **Example with result in a0:**
 * ```
 * funct7 = 64 + 60 = 124 (0x7C)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7C  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111000000000001010000000101011
 * Hex:    0xF8002A2B
 * ```
 */
#define SCMDBUF_TR_ACK()                          \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 60)); \
        return result;                            \
    }()

/**
 * @brief Gets number of pending transaction acknowledgments for a specific TRID
 * @param trid The transaction ID to query
 * @return Number of pending transaction acknowledgments for the specified TRID
 *
 * Returns the number of acknowledgments that are yet to be received for
 * non-posted read or write transactions using the given transaction ID.
 *
 * @details
 * **Example with trid=3, result in a0:**
 * ```
 * funct7 = 64 + 60 = 124 (0x7C)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7C  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Transaction ID     |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 1     | rd used            |
 * | rd     | 11-7  | 0x0A  | Result register    |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111000000001011010000000101011
 * Hex:    0xF805A02B
 * ```
 */
#define SCMDBUF_TR_ACK_TRID(trid)                          \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(1, result, (trid), (64 + 60)); \
        return result;                                     \
    }()

/**
 * @brief Resets all simple command buffer registers to default values
 *
 * Resets all simple command buffer registers to their default values as specified
 * in the RDL (Register Description Language) file.
 *
 * @details
 * **Example:**
 * ```
 * funct7 = 64 + 59 = 123 (0x7B)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7B  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11110110000000000000000000101011
 * Hex:    0xF600002B
 * ```
 */
#define SCMDBUF_RESET()                 \
    {                                   \
        ROCC_INSTRUCTION(1, (64 + 59)); \
    }

/**
 * @brief Issues a transaction using the simple command buffer
 *
 * Issues a transaction using the currently configured simple command buffer settings.
 *
 * @details
 * **Example:**
 * ```
 * funct7 = 64 + 63 = 127 (0x7F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x00  | Unused (x0)       |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 0     | rs1 not used       |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111110000000000000000000101011
 * Hex:    0xFE00002B
 * ```
 */
#define SCMDBUF_ISSUE_TRANS()           \
    {                                   \
        ROCC_INSTRUCTION(1, (64 + 63)); \
    }

/**
 * @brief Issues an inline transaction with immediate data using simple command buffer
 * @param val The immediate value for the inline transaction
 *
 * Issues an inline transaction with immediate data using the simple command buffer.
 *
 * @details
 * **Example with val=0xABCD:**
 * ```
 * funct7 = 64 + 63 = 127 (0x7F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7F  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Inline value        |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111110000001011001000000101011
 * Hex:    0xFE05802B
 * ```
 */
#define SCMDBUF_ISSUE_INLINE_TRANS(val)          \
    {                                            \
        ROCC_INSTRUCTION_S(1, (val), (64 + 63)); \
    }

/**
 * @brief Issues an inline transaction with extended parameters using simple command buffer
 * @param val The immediate value for the inline transaction
 * @param rs2 The second source register containing destination address and transaction parameters
 *
 * Issues an inline transaction with extended parameters using the simple command buffer.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`)
 * - Bits [60]: Has XY flag (`has_xy`)
 * - Bits [59:57]: Size in bytes minus 1 (`size_bytes-1`)
 * - Bits [47:32]: Destination coordinates (`dest_coords`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `dest_addr  | dest_coords << 32| ((size_bytes-1) << 57) | (has_xy << 60) | (posted << 61) | (snoop << 62) |
 * (flush << 63)`
 *
 * **Example with val=0x1234, dest_addr=0x5678, size_bytes=0x1000, flags=0:**
 * ```
 * rs2 = 0x5678 | ((0x1000-1) << 57) | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x5678 | (0xFFF << 57)
 *
 * funct7 = 64 + 63 = 127 (0x7F)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x7F  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Dest addr + params |
 * | rs1    | 19-15 | 0x0B  | Inline value        |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11111110000001011100000000101011
 * Hex:    0xFE05C02B
 * ```
 */
#define SCMDBUF_ISSUE_INLINE_ADDR_TRANS(val, rs2)        \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (val), (rs2), (64 + 63)); \
    }

/**
 * @brief Issues a read transaction using only destination address and length using simple command buffer
 * @param rs1 The first source register containing destination address and length
 *
 * Issues a read transaction using only destination address and length. This variant can be used if you only want to
 * change the destination address and length while using default values for source address and transaction flags.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(len_bytes << 32) | dest_addr`
 *
 * **Example with len_bytes=0x1000, dest_addr=0x5000:**
 * ```
 * rs1 = (0x1000 << 32) | 0x5000 = 0x100000005000
 *
 * funct7 = 64 + 56 = 120 (0x78)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x78  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Dest addr + length |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11110000000001011001000000101011
 * Hex:    0xF005802B
 * ```
 */
#define SCMDBUF_ISSUE_READ1_TRANS(rs1)           \
    {                                            \
        ROCC_INSTRUCTION_S(1, (rs1), (64 + 56)); \
    }

/**
 * @brief Issues a read transaction with both addresses and transaction parameters using simple command buffer
 * @param rs1 The first source register containing destination address and length
 * @param rs2 The second source register containing source address and transaction flags
 *
 * Issues a read transaction with both source and destination addresses.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(len_bytes << 32) | dest_addr`
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`) - not used in this function
 * - Bits [60]: Has XY flag (`has_xy`)  - if set, will look at coordinates from address in rs2
 * - Bits [59:32]: Source coordinates, created with NOC_XY_COORD macro
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(flush << 63) | (snoop << 62) | (posted << 61) | (has_xy << 60) | ((src_coordinate) << 32) | (src_addr)`
 *
 * **Example with len_bytes=0x1000, src_addr=0x2000, dest_addr=0x100, flags=0:**
 * ```
 * rs1 = (0x1000 << 32) | 0x100 = 0x1000000000100
 * rs2 = 0x2000 | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x2000
 *
 * funct7 = 64 + 56 = 120 (0x78)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x78  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Src addr + flags   |
 * | rs1    | 19-15 | 0x0B  | Dest addr + length |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11110000000001011100000000101011
 * Hex:    0xF005C02B
 * ```
 */
#define SCMDBUF_ISSUE_READ2_TRANS(rs1, rs2)              \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (rs1), (rs2), (64 + 56)); \
    }

/**
 * @brief Issues a write transaction using only source address and length using simple command buffer
 * @param rs1 The first source register containing source address and length
 *
 * Issues a write transaction using only source address and length. This variant can be used if you only want to change
 * the source address and length while using default values for destination address and transaction flags.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(len_bytes << 32) | src_addr`
 *
 * **Example with len_bytes=0x1000, src_addr=0x7000:**
 * ```
 * rs1 = (0x1000 << 32) | 0x7000 = 0x100000007000
 *
 * funct7 = 64 + 55 = 119 (0x77)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x77  | Function code      |
 * | rs2    | 24-20 | 0x00  | Unused (x0)       |
 * | rs1    | 19-15 | 0x0B  | Src addr + length  |
 * | xs2    | 12    | 0     | rs2 not used       |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11101110000001011001000000101011
 * Hex:    0xEE05802B
 * ```
 */
#define SCMDBUF_ISSUE_WRITE1_TRANS(rs1)          \
    {                                            \
        ROCC_INSTRUCTION_S(1, (rs1), (64 + 55)); \
    }

/**
 * @brief Issues a write transaction with both addresses and transaction parameters using simple command buffer
 * @param rs1 The first source register containing source address and length
 * @param rs2 The second source register containing destination address and transaction flags
 *
 * Issues a write transaction with both source and destination addresses.
 *
 * @details
 * **Parameter Encoding:**
 *
 * **rs1 (64-bit):**
 * - Bits [63:32]: Length in bytes (`len_bytes`)
 * - Bits [31:0]: Source address (`src_addr`)
 * - Format: `(len_bytes << 32) | src_addr`
 *
 * **rs2 (64-bit):**
 * - Bits [63]: Flush flag (`flush`)
 * - Bits [62]: Snoop flag (`snoop`)
 * - Bits [61]: Posted flag (`posted`)
 * - Bits [60]: Has XY flag (`has_xy`)  - if set, will look at coordinates from address in rs2
 * - Bits [59:32]: Destination coordinates, created with NOC_XY_COORD macro
 * - Bits [31:0]: Destination address (`dest_addr`)
 * - Format: `(flush << 63) | (snoop << 62) | (posted << 61) | (has_xy << 60) | (dest_coordinate << 32) | dest_addr`
 *
 * **Example with len_bytes=0x1000, src_addr=0x4000, dest_addr=0x200, flags=0:**
 * ```
 * rs1 = (0x1000 << 32) | 0x4000 = 0x1000000004000
 * rs2 = 0x200 | (0 << 60) | (0 << 61) | (0 << 62) | (0 << 63) = 0x200
 *
 * funct7 = 64 + 55 = 119 (0x77)
 *
 * Instruction Fields:
 * | Field  | Bits  | Value | Description        |
 * |--------|-------|-------|-------------------|
 * | funct7 | 31-25 | 0x77  | Function code      |
 * | rs2    | 24-20 | 0x0C  | Dest addr + flags  |
 * | rs1    | 19-15 | 0x0B  | Src addr + length  |
 * | xs2    | 12    | 1     | rs2 used           |
 * | xs1    | 13    | 1     | rs1 used           |
 * | xd     | 14    | 0     | rd not used        |
 * | rd     | 11-7  | 0x00  | Unused (x0)       |
 * | opcode | 6-0   | 0x2B  | CUSTOM_1          |
 *
 * Binary: 11101110000001011100000000101011
 * Hex:    0xEE05C02B
 * ```
 */
#define SCMDBUF_ISSUE_WRITE2_TRANS(rs1, rs2)             \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (rs1), (rs2), (64 + 55)); \
    }
