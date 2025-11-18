// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "xcustom_test.hpp"
#include "overlay_reg.h"
/////////////////
// MISC
/////////////////

// Non functional
#define DBG_POSTCODE(val)                       \
    {                                           \
        ROCC_INSTRUCTION_S(2, (val), (32 + 1)); \
    }

// Causes a NOC fence which ensures that all inbound flits after this fence instruction, that are already in the local
// niu, have been commited to L1 When calling this you must call: asm volatile ("fence" : : : "memory");
#define NOC_FENCE()                    \
    {                                  \
        ROCC_INSTRUCTION(2, (32 + 2)); \
    }

// Fast LLK Interface
// Writes directly to the LLK Interface
#define LLK_INTF_WRITE(addr, val)                        \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (addr), (val), (32 + 3)); \
    }

// Fast LLK Interface
// Reads directly from the LLK Interface
#define LLK_INTF_READ(addr)                             \
    [&]() {                                             \
        uint64_t result;                                \
        ROCC_INSTRUCTION_DS(2, result, addr, (32 + 3)); \
        return result;                                  \
    }()

// FDS Register Interface
// Writes to FDS register file
#define FDS_INTF_WRITE(addr, val)                        \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (addr), (val), (32 + 4)); \
    }

// Reads from FDS register file
#define FDS_INTF_READ(addr)                             \
    [&]() {                                             \
        uint64_t result;                                \
        ROCC_INSTRUCTION_DS(2, result, addr, (32 + 4)); \
        return result;                                  \
    }()

// Tiles to Process Threshold Instructions
// Set threshold for IDMA_TR_ACK tiles to process interrupts for a specific TRID
#define SET_TILES_TO_PROCESS_THRES_IDMA_TR_ACK(trid, val) \
    {                                                     \
        ROCC_INSTRUCTION_SS(2, (trid), (val), (32 + 5));  \
    }

// Read threshold for IDMA_TR_ACK tiles to process interrupts for a specific TRID
#define GET_TILES_TO_PROCESS_THRES_IDMA_TR_ACK(trid)      \
    [&]() {                                               \
        uint64_t result;                                  \
        ROCC_INSTRUCTION_DS(2, result, (trid), (32 + 5)); \
        return result;                                    \
    }()

// Set threshold for WR_SENT tiles to process interrupts for a specific TRID
#define SET_TILES_TO_PROCESS_THRES_WR_SENT(trid, val)    \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (trid), (val), (32 + 6)); \
    }

// Read threshold for WR_SENT tiles to process interrupts for a specific TRID
#define GET_TILES_TO_PROCESS_THRES_WR_SENT(trid)          \
    [&]() {                                               \
        uint64_t result;                                  \
        ROCC_INSTRUCTION_DS(2, result, (trid), (32 + 6)); \
        return result;                                    \
    }()

// Set threshold for TR_ACK tiles to process interrupts for a specific TRID
#define SET_TILES_TO_PROCESS_THRES_TR_ACK(trid, val)     \
    {                                                    \
        ROCC_INSTRUCTION_SS(2, (trid), (val), (32 + 7)); \
    }

// Read threshold for TR_ACK tiles to process interrupts for a specific TRID
#define GET_TILES_TO_PROCESS_THRES_TR_ACK(trid)           \
    [&]() {                                               \
        uint64_t result;                                  \
        ROCC_INSTRUCTION_DS(2, result, (trid), (32 + 7)); \
        return result;                                    \
    }()

/////////////////
// Context Switch
/////////////////

// Allocates a slot on the context switch ram
// Returns 0 is not slot is available, or the slot id
// Only even cores have this functionality
#define CS_ALLOC                          \
    [&]() {                               \
        uint64_t result;                  \
        ROCC_INSTRUCTION_D(2, result, 3); \
        return result;                    \
    }()

// Deallocates the given slot
// You should not pass slot 0 to this
// Only even cores have this functionality
#define CS_DEALLOC(context)                  \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 2); \
    }

// Saves the context to the given slot, slot id must be non-zero
// call asm volatile ("fence" : : : "memory"); after this
#define CS_SAVE(context)                     \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 1); \
    }

// Restores the context from the given slot, slot id must be non-zero
// call asm volatile ("fence" : : : "memory"); after this
#define CS_RESTORE(context)                  \
    {                                        \
        ROCC_INSTRUCTION_S(2, (context), 0); \
    }

/////////////////
// Address Gen
/////////////////

// Write to a address generator register
#define ADDRGEN_WR_REG(cmdbuf, reg, val)                               \
    {                                                                  \
        ROCC_INSTRUCTION_S(1, (val), (((cmdbuf) * 32) + ((reg) / 8))); \
    }

// Read from an address generator register
#define ADDRGEN_RD_REG(cmdbuf, reg)                                     \
    [&]() {                                                             \
        uint64_t result;                                                \
        ROCC_INSTRUCTION_D(1, result, (((cmdbuf) * 32) + ((reg) / 8))); \
        return result;                                                  \
    }()

// Reset all address generator registers to a default value
// The default value is given in the rdl file.
#define ADDRGEN_RESET(cmdbuf)                      \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 26)); \
    }

// Reset the address generator counters, but keep the base, size and strides
#define ADDRGEN_RESET_COUNTERS(cmdbuf)                    \
    {                                                     \
        ROCC_INSTRUCTION_S(1, (0), ((cmdbuf) * 32 + 26)); \
    }

// Returns the generated src address without generating a new one
#define ADDRGEN_PEEK_SRC(cmdbuf)                             \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(1, result, ((cmdbuf) * 32 + 31)); \
        return result;                                       \
    }()

// Returns the generated src address and generates a new one
#define ADDRGEN_POP_SRC(cmdbuf)                                  \
    [&]() {                                                      \
        uint64_t result;                                         \
        ROCC_INSTRUCTION_DS(1, result, 1, ((cmdbuf) * 32 + 31)); \
        return result;                                           \
    }()

// Returns the generated src address and skips (skip_src-1) src addresses afterwards
#define ADDRGEN_POP_X_SRC(cmdbuf, skip_src)                        \
    [&]() {                                                        \
        uint64_t rs1 = skip_src;                                   \
        uint64_t result;                                           \
        ROCC_INSTRUCTION_DS(1, result, rs1, ((cmdbuf) * 32 + 31)); \
        return result;                                             \
    }()

// Returns the generated dest address without generating a new one
#define ADDRGEN_PEEK_DEST(cmdbuf)                            \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(1, result, ((cmdbuf) * 32 + 30)); \
        return result;                                       \
    }()

// Returns the generated dest address and generates a new one
#define ADDRGEN_POP_DEST(cmdbuf)                                 \
    [&]() {                                                      \
        uint64_t result;                                         \
        ROCC_INSTRUCTION_DS(1, result, 1, ((cmdbuf) * 32 + 30)); \
        return result;                                           \
    }()

// Returns the generated dest address and skips (skip_dest-1) dest addresses afterwards
#define ADDRGEN_POP_X_DEST(cmdbuf, skip_dest)                      \
    [&]() {                                                        \
        uint64_t rs1 = skip_dest;                                  \
        uint64_t result;                                           \
        ROCC_INSTRUCTION_DS(1, result, rs1, ((cmdbuf) * 32 + 30)); \
        return result;                                             \
    }()

// Returns the generated dest address and skips (x-1) dest addresses afterwards
// Result holds 32 bits of each dest and source addresses - {dest[31:0], src[31:]}
#define ADDRGEN_POP_BOTH(cmdbuf, skip_src, skip_dest)                    \
    [&]() {                                                              \
        uint64_t rs1 = skip_src;                                         \
        uint64_t rs2 = skip_dest;                                        \
        uint64_t result;                                                 \
        ROCC_INSTRUCTION_DSS(1, result, rs1, rs2, ((cmdbuf) * 32 + 25)); \
        return result;                                                   \
    }()

// Push the generated src address to the command buffer asccociated with this address generator
// and generate a new address
#define ADDRGEN_PUSH_SRC(cmdbuf)                   \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 28)); \
    }

// Push the generated src address to the command buffer asccociated with this address generator
// and generate (skip_src) new addresses
#define ADDRGEN_PUSH_SRC_POP_X(cmdbuf, skip_src)          \
    {                                                     \
        uint64_t rs1 = skip_src;                          \
        ROCC_INSTRUCTION_S(1, rs1, ((cmdbuf) * 32 + 28)); \
    }

// Push the generated dest address to the command buffer asccociated with this address generator
// and generate a new address
#define ADDRGEN_PUSH_DEST(cmdbuf)                  \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 27)); \
    }

// Push the generated dest address to the command buffer asccociated with this address generator
// and generate (skip_dest) new addresses
#define ADDRGEN_PUSH_DEST_POP_X(cmdbuf, skip_dest)        \
    {                                                     \
        uint64_t rs1 = skip_dest;                         \
        ROCC_INSTRUCTION_S(1, rs1, ((cmdbuf) * 32 + 27)); \
    }

// Push both generated addresses to the command buffer asccociated with this address generator
// and generate new addresses
#define ADDRGEN_PUSH_BOTH(cmdbuf)                  \
    {                                              \
        ROCC_INSTRUCTION(1, ((cmdbuf) * 32 + 29)); \
    }

// Push both generated addresses to the command buffer asccociated with this address generator
// and generate new src and dest addresses
#define ADDRGEN_PUSH_BOTH_POP_X(cmdbuf, skip_src, skip_dest)    \
    {                                                           \
        uint64_t rs1 = skip_src;                                \
        uint64_t rs2 = skip_dest;                               \
        ROCC_INSTRUCTION_SS(1, rs1, rs2, ((cmdbuf) * 32 + 29)); \
    }

/////////////////
// CMD Bufs
/////////////////

// Write to a command buffer register
#define CMDBUF_WR_REG(cmdbuf, reg, val)                                \
    {                                                                  \
        ROCC_INSTRUCTION_S(0, (val), (((cmdbuf) * 64) + ((reg) / 8))); \
    }

// Read a command buffer register
#define CMDBUF_RD_REG(cmdbuf, reg)                                      \
    [&]() {                                                             \
        uint64_t result;                                                \
        ROCC_INSTRUCTION_D(0, result, (((cmdbuf) * 64) + ((reg) / 8))); \
        return result;                                                  \
    }()

// Returns the amount of vc space for the currently programmed request VC
#define CMDBUF_GET_VC_SPACE(cmdbuf)                          \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 62)); \
        return result;                                       \
    }()

// Returns the amount of vc space for the specified VC
#define CMDBUF_GET_VC_SPACE_VC(cmdbuf, vc)                          \
    [&]() {                                                         \
        uint64_t result;                                            \
        ROCC_INSTRUCTION_DS(0, result, (vc), ((cmdbuf) * 64 + 62)); \
        return result;                                              \
    }()

// Returns the number of outstanding write transactions that need to be sent from the local
// core using the programmed wr-ack transaction id
#define CMDBUF_WR_SENT(cmdbuf)                               \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 61)); \
        return result;                                       \
    }()

// Returns the number of outstanding write transactions that need to be sent from the local
// core using the given transaction id
#define CMDBUF_WR_SENT_TRID(cmdbuf, trid)                             \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 61)); \
        return result;                                                \
    }()

// Returns the number of acks that are yet to be recieved for non-posted read or write transactions
// using the programmed tr-ack transaction id
#define CMDBUF_TR_ACK(cmdbuf)                                \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 60)); \
        return result;                                       \
    }()

// Returns the number of acks that are yet to be recieved for non-posted read or write transactions
// using the given transaction id
#define CMDBUF_TR_ACK_TRID(cmdbuf, trid)                              \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 60)); \
        return result;                                                \
    }()

// Reset all command buffer registers to their default values as given by the rdl file.
#define CMDBUF_RESET(cmdbuf)                       \
    {                                              \
        ROCC_INSTRUCTION(0, ((cmdbuf) * 64 + 59)); \
    }

// Returns the amount of idma vc space for the currently programmed request VC
#define CMDBUF_IDMA_GET_VC_SPACE(cmdbuf)                     \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 58)); \
        return result;                                       \
    }()

// Returns the amount of idma vc space for the specified VC
#define CMDBUF_IDMA_GET_VC_SPACE_VC(cmdbuf, vc)                     \
    [&]() {                                                         \
        uint64_t result;                                            \
        ROCC_INSTRUCTION_DS(0, result, (vc), ((cmdbuf) * 64 + 58)); \
        return result;                                              \
    }()

// Returns the number of idma acks that are yet to be recieved for non-posted read or write transactions
// using the programmed tr-ack transaction id
#define CMDBUF_IDMA_TR_ACK(cmdbuf)                           \
    [&]() {                                                  \
        uint64_t result;                                     \
        ROCC_INSTRUCTION_D(0, result, ((cmdbuf) * 64 + 57)); \
        return result;                                       \
    }()

// Returns the number of idma acks that are yet to be recieved for non-posted read or write transactions
// using the given transaction id
#define CMDBUF_IDMA_TR_ACK_TRID(cmdbuf, trid)                         \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 57)); \
        return result;                                                \
    }()

// Issues the transaction
#define CMDBUF_ISSUE_TRANS(cmdbuf)                 \
    {                                              \
        ROCC_INSTRUCTION(0, ((cmdbuf) * 64 + 63)); \
    }

// Issues an inline transaction
#define CMDBUF_ISSUE_INLINE_TRANS(cmdbuf, val)              \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (val), ((cmdbuf) * 64 + 63)); \
    }

// Issues an inline transaction with extended params
#define CMDBUF_ISSUE_INLINE_ADDR_TRANS(cmdbuf, val, rs2)            \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (val), (rs2), ((cmdbuf) * 64 + 63)); \
    }

// Issues a read transaction
// This preprograms the command buffer to a set of default values that allow for simple read operations
// This variant can be used if you only want to change the dest address
#define CMDBUF_ISSUE_READ1_TRANS(cmdbuf, rs1)               \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (rs1), ((cmdbuf) * 64 + 56)); \
    }

// Issues a read transaction
// This preprograms the command buffer to a set of default values that allow for simple read operations
#define CMDBUF_ISSUE_READ2_TRANS(cmdbuf, rs1, rs2)                  \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (rs1), (rs2), ((cmdbuf) * 64 + 56)); \
    }

// Issues a write transaction
// This preprograms the command buffer to a set of default values that allow for simple write operations
// This variant can be used if you only want to change the src address
#define CMDBUF_ISSUE_WRITE1_TRANS(cmdbuf, rs1)              \
    {                                                       \
        ROCC_INSTRUCTION_S(0, (rs1), ((cmdbuf) * 64 + 55)); \
    }

// Issues a write transaction
// This preprograms the command buffer to a set of default values that allow for simple write operations
#define CMDBUF_ISSUE_WRITE2_TRANS(cmdbuf, rs1, rs2)                 \
    {                                                               \
        ROCC_INSTRUCTION_SS(0, (rs1), (rs2), ((cmdbuf) * 64 + 55)); \
    }

// Read tiles to process TR_ACK counter for a specific transaction ID
#define CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(cmdbuf, trid)             \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 54)); \
        return result;                                                \
    }()

// Read tiles to process WR_SENT counter for a specific transaction ID
#define CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(cmdbuf, trid)            \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 53)); \
        return result;                                                \
    }()

// Read tiles to process IDMA_TR_ACK counter for a specific transaction ID
#define CMDBUF_READ_TILES_TO_PROCESS_IDMA_TR_ACK(cmdbuf, trid)        \
    [&]() {                                                           \
        uint64_t result;                                              \
        ROCC_INSTRUCTION_DS(0, result, (trid), ((cmdbuf) * 64 + 52)); \
        return result;                                                \
    }()

// Clear tiles to process TR_ACK counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(cmdbuf, trid)   \
    {                                                        \
        ROCC_INSTRUCTION_S(0, (trid), ((cmdbuf) * 64 + 51)); \
    }

// Clear tiles to process WR_SENT counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(cmdbuf, trid)  \
    {                                                        \
        ROCC_INSTRUCTION_S(0, (trid), ((cmdbuf) * 64 + 50)); \
    }

// Clear tiles to process IDMA_TR_ACK counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define CMDBUF_CLEAR_TILES_TO_PROCESS_IDMA_TR_ACK(cmdbuf, trid) \
    {                                                           \
        ROCC_INSTRUCTION_S(0, (trid), ((cmdbuf) * 64 + 49));    \
    }

/////////////////
// Simple CMD Buf
/////////////////

// For instruction on using the simple command buffers
// see the comments on the regular command buffers above

#define SCMDBUF_WR_REG(reg, val)                          \
    {                                                     \
        ROCC_INSTRUCTION_S(1, (val), (64 + ((reg) / 8))); \
    }

#define SCMDBUF_RD_REG(reg)                                \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_D(1, result, (64 + ((reg) / 8))); \
        return result;                                     \
    }()

#define SCMDBUF_GET_VC_SPACE()                    \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 62)); \
        return result;                            \
    }()

#define SCMDBUF_GET_VC_SPACE_VC(vc)                      \
    [&]() {                                              \
        uint64_t result;                                 \
        ROCC_INSTRUCTION_DS(1, result, (vc), (64 + 62)); \
        return result;                                   \
    }()

#define SCMDBUF_WR_SENT()                         \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 61)); \
        return result;                            \
    }()

#define SCMDBUF_WR_SENT_TRID(trid)                         \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(1, result, (trid), (64 + 61)); \
        return result;                                     \
    }()

#define SCMDBUF_TR_ACK()                          \
    [&]() {                                       \
        uint64_t result;                          \
        ROCC_INSTRUCTION_D(1, result, (64 + 60)); \
        return result;                            \
    }()

#define SCMDBUF_TR_ACK_TRID(trid)                          \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(1, result, (trid), (64 + 60)); \
        return result;                                     \
    }()

#define SCMDBUF_RESET()                 \
    {                                   \
        ROCC_INSTRUCTION(1, (64 + 59)); \
    }

#define SCMDBUF_ISSUE_TRANS()           \
    {                                   \
        ROCC_INSTRUCTION(1, (64 + 63)); \
    }

#define SCMDBUF_ISSUE_INLINE_TRANS(val)          \
    {                                            \
        ROCC_INSTRUCTION_S(1, (val), (64 + 63)); \
    }

#define SCMDBUF_ISSUE_INLINE_ADDR_TRANS(val, rs2)        \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (val), (rs2), (64 + 63)); \
    }

#define SCMDBUF_ISSUE_READ1_TRANS(rs1)           \
    {                                            \
        ROCC_INSTRUCTION_S(1, (rs1), (64 + 56)); \
    }

#define SCMDBUF_ISSUE_READ2_TRANS(rs1, rs2)              \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (rs1), (rs2), (64 + 56)); \
    }

#define SCMDBUF_ISSUE_WRITE1_TRANS(rs1)          \
    {                                            \
        ROCC_INSTRUCTION_S(1, (rs1), (64 + 55)); \
    }

#define SCMDBUF_ISSUE_WRITE2_TRANS(rs1, rs2)             \
    {                                                    \
        ROCC_INSTRUCTION_SS(1, (rs1), (rs2), (64 + 55)); \
    }

// Read tiles to process TR_ACK counter for a specific transaction ID
#define SCMDBUF_READ_TILES_TO_PROCESS_TR_ACK(trid)         \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(0, result, (trid), (64 + 54)); \
        return result;                                     \
    }()

// Read tiles to process WR_SENT counter for a specific transaction ID
#define SCMDBUF_READ_TILES_TO_PROCESS_WR_SENT(trid)        \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(0, result, (trid), (64 + 53)); \
        return result;                                     \
    }()

// Read tiles to process IDMA_TR_ACK counter for a specific transaction ID
#define SCMDBUF_READ_TILES_TO_PROCESS_IDMA_TR_ACK(trid)    \
    [&]() {                                                \
        uint64_t result;                                   \
        ROCC_INSTRUCTION_DS(0, result, (trid), (64 + 52)); \
        return result;                                     \
    }()

// Clear tiles to process TR_ACK counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define SCMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(trid) \
    {                                               \
        ROCC_INSTRUCTION_S(0, (trid), (64 + 51));   \
    }

// Clear tiles to process WR_SENT counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define SCMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(trid) \
    {                                                \
        ROCC_INSTRUCTION_S(0, (trid), (64 + 50));    \
    }

// Clear tiles to process IDMA_TR_ACK counter for a specific transaction ID
// must run __asm__ volatile("nop") after to give time for the clear to propagate
#define SCMDBUF_CLEAR_TILES_TO_PROCESS_IDMA_TR_ACK(trid) \
    {                                                    \
        ROCC_INSTRUCTION_S(0, (trid), (64 + 49));        \
    }

#endif
