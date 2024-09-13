// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/sanitize_noc.h
//
// This file implements a method sanitize noc addresses.
// Malformed addresses (out of range offsets, bad XY, etc) are stored in L1
// where the watcher thread can log the result.  The device then soft-hangs in
// a spin loop.
//
// All functionaly gated behind defined WATCHER_ENABLED
//
#pragma once

#include "dprint.h"

// Add the ability to skip NOC logging, we can't have the tunneling cores stalling waiting for the
// print server.
#if !defined(SKIP_NOC_LOGGING)
#define LOG_LEN(l) DPRINT << NOC_LOG_XFER(l);
#define LOG_READ_LEN_FROM_STATE(noc_id) LOG_LEN(NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define LOG_WRITE_LEN_FROM_STATE(noc_id) LOG_LEN(NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));
#else
#define LOG_LEN(l)
#define LOG_READ_LEN_FROM_STATE(noc_id)
#define LOG_WRITE_LEN_FROM_STATE(noc_id)
#endif

#if (                                                                                          \
    defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)) &&                                                        \
    (defined(WATCHER_ENABLED)) && (!defined(WATCHER_DISABLE_NOC_SANITIZE))

#include "watcher_common.h"

extern uint8_t noc_index;

#include "dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"

// A couple defines for specifying read/write and multi/unicast
#define DEBUG_SANITIZE_NOC_READ true
#define DEBUG_SANITIZE_NOC_WRITE false
typedef bool debug_sanitize_noc_dir_t;
#define DEBUG_SANITIZE_NOC_MULTICAST true
#define DEBUG_SANITIZE_NOC_UNICAST false
typedef bool debug_sanitize_noc_cast_t;

// Helper function to get the core type from noc coords.
AddressableCoreType get_core_type(uint8_t noc_id, uint8_t x, uint8_t y) {
    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);

    for (uint32_t idx = 0; idx < MAX_NON_WORKER_CORES; idx++) {
        uint8_t core_x = core_info->non_worker_cores[idx].x;
        uint8_t core_y = core_info->non_worker_cores[idx].y;
        if (x == NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_x) &&
            y == NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_y)) {
            return core_info->non_worker_cores[idx].type;
        }
    }

    for (uint32_t idx = 0; idx < MAX_HARVESTED_ROWS; idx++) {
        uint16_t harvested_y = core_info->harvested_y[idx];
        if (y == NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) harvested_y)) {
            return AddressableCoreType::HARVESTED;
        }
    }

    // Tensix
    if (noc_id == 0) {
        if (x >= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) 1) &&
            x <= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_info->noc_size_x - 1) &&
            y >= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) 1) &&
            y <= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_info->noc_size_y - 1)) {
            return AddressableCoreType::TENSIX;
        }
    } else {
        if (x <= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) 1) &&
            x >= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_info->noc_size_x - 1) &&
            y <= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) 1) &&
            y >= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_info->noc_size_y - 1)) {
            return AddressableCoreType::TENSIX;
        }
    }

    return AddressableCoreType::UNKNOWN;
}

// TODO(PGK): remove soft reset when fw is downloaded at init
#define DEBUG_VALID_REG_ADDR(a, l)                                                      \
    (((((a) >= NOC_OVERLAY_START_ADDR) &&                                               \
       ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
      ((a) == RISCV_DEBUG_REG_SOFT_RESET_0)) &&                                         \
     (l) == 4)
#define DEBUG_VALID_WORKER_ADDR(a, l) ((a >= MEM_L1_BASE) && (a + l <= MEM_L1_BASE + MEM_L1_SIZE) && ((a) + (l) > (a)))
inline bool debug_valid_pcie_addr(uint64_t addr, uint64_t len) {
    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    return ((addr) >= core_info->noc_pcie_addr_base) && ((addr) + (len) <= core_info->noc_pcie_addr_end) &&
           ((addr) + (len) > (addr));
}
inline bool debug_valid_dram_addr(uint64_t addr, uint64_t len) {
    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    return ((addr) >= core_info->noc_dram_addr_base) && ((addr) + (len) <= core_info->noc_dram_addr_end) &&
           ((addr) + (len) > (addr));
}

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && ((a) + (l) <= MEM_ETH_BASE + MEM_ETH_SIZE))

// Note:
//  - this isn't racy w/ the host so long as invalid is written last
//  - this isn't racy between riscvs so long as each gets their own noc_index
inline void debug_sanitize_post_noc_addr_and_hang(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    uint16_t invalid) {
    debug_sanitize_noc_addr_msg_t tt_l1_ptr *v = *GET_MAILBOX_ADDRESS_DEV(watcher.sanitize_noc);

    if (v[noc_id].invalid == DebugSanitizeNocInvalidOK) {
        v[noc_id].noc_addr = noc_addr;
        v[noc_id].l1_addr = l1_addr;
        v[noc_id].len = len;
        v[noc_id].which = debug_get_which_riscv();
        v[noc_id].multicast = multicast;
        v[noc_id].invalid = invalid;
    }

    // Update launch msg to show that we've exited.
    tt_l1_ptr launch_msg_t *launch_msg = GET_MAILBOX_ADDRESS_DEV(launch);
    launch_msg->go.run = RUN_MSG_DONE;

#if defined(COMPILE_FOR_ERISC)
    // For erisc, we can't hang the kernel/fw, because the core doesn't get restarted when a new
    // kernel is written. In this case we'll do an early exit back to base FW.
    internal_::disable_erisc_app();
    erisc_early_exit(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE);
#endif

    while (1) { ; }
}

// Return value is the alignment mask for the type of core the noc address points
// to. Need to do this because L1 alignment needs to match the noc address alignment requirements,
// even if it's different than the inherent L1 alignment requirements.
// Direction is specified because reads and writes may have different L1 requirements (see noc_parameters.h).
uint32_t debug_sanitize_noc_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t noc_len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    // Different encoding of noc addr depending on multicast vs unitcast
    uint8_t x, y;
    if (multicast) {
        x = (uint8_t) NOC_MCAST_ADDR_START_X(noc_addr);
        y = (uint8_t) NOC_MCAST_ADDR_START_Y(noc_addr);
    } else {
        x = (uint8_t) NOC_UNICAST_ADDR_X(noc_addr);
        y = (uint8_t) NOC_UNICAST_ADDR_Y(noc_addr);
    }
    uint64_t noc_local_addr = NOC_LOCAL_ADDR(noc_addr);
    AddressableCoreType core_type = get_core_type(noc_id, x, y);

    // Extra check for multicast
    if (multicast) {
        uint8_t x_end = (uint8_t) NOC_MCAST_ADDR_END_X(noc_addr);
        uint8_t y_end = (uint8_t) NOC_MCAST_ADDR_END_Y(noc_addr);

        AddressableCoreType end_core_type = get_core_type(noc_id, x_end, y_end);

        // Multicast supports workers only
        if (core_type != AddressableCoreType::TENSIX || end_core_type != AddressableCoreType::TENSIX || (x > x_end || y > y_end)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_id, noc_addr, l1_addr, noc_len, multicast, DebugSanitizeNocInvalidMulticast);
        }
    }

    // Check noc addr, we save the alignment requirement from the noc src/dst because the L1 address
    // needs to match alignment.
    // Reads and writes may have different alignment requirements, see noc_parameters.h for details.
    uint32_t alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_L1_READ_ALIGNMENT_BYTES : NOC_L1_WRITE_ALIGNMENT_BYTES) - 1;  // Default alignment, only override in ceratin cases.
    uint32_t invalid = multicast ? DebugSanitizeNocInvalidMulticast : DebugSanitizeNocInvalidUnicast;
    if (core_type == AddressableCoreType::PCIE) {
        alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_PCIE_READ_ALIGNMENT_BYTES : NOC_PCIE_WRITE_ALIGNMENT_BYTES) - 1;
        if (!debug_valid_pcie_addr(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_id, noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else if (core_type == AddressableCoreType::DRAM) {
        alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_DRAM_READ_ALIGNMENT_BYTES : NOC_DRAM_WRITE_ALIGNMENT_BYTES) - 1;
        if (!debug_valid_dram_addr(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_id, noc_addr, l1_addr, noc_len, multicast, invalid);
        }
#ifndef ARCH_GRAYSKULL
    } else if (core_type == AddressableCoreType::ETH) {
        if (!DEBUG_VALID_REG_ADDR(noc_local_addr, noc_len) && !DEBUG_VALID_ETH_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_id, noc_addr, l1_addr, noc_len, multicast, invalid);
        }
#endif
    } else if (core_type == AddressableCoreType::TENSIX) {
        if (!DEBUG_VALID_REG_ADDR(noc_local_addr, noc_len) && !DEBUG_VALID_WORKER_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_id, noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else {
        // Bad XY
        debug_sanitize_post_noc_addr_and_hang(noc_id, noc_addr, l1_addr, noc_len, multicast, invalid);
    }

    return alignment_mask;
}

void debug_sanitize_worker_addr(uint8_t noc_id, uint32_t addr, uint32_t len) {
    // Regs are exempt from standard L1 validation
    if (DEBUG_VALID_REG_ADDR(addr, len))
        return;

    if (!DEBUG_VALID_WORKER_ADDR(addr, len)) {
        debug_sanitize_post_noc_addr_and_hang(noc_id, addr, 0, len, false, DebugSanitizeNocInvalidL1);
    }
}

void debug_sanitize_noc_and_worker_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t worker_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    // Check noc addr, get any extra alignment req for worker.
    uint32_t alignment_mask = debug_sanitize_noc_addr(noc_id, noc_addr, worker_addr, len, multicast, dir);

    // Check worker addr
    debug_sanitize_worker_addr(noc_id, worker_addr, len);

    // Check alignment, but not for reg addresses.
    if (!DEBUG_VALID_REG_ADDR(worker_addr, len)) {
        if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_id, noc_addr, worker_addr, len, multicast, DebugSanitizeNocInvalidAlignment);
        }
    }
}

// TODO: Clean these up with #7453
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_FROM_STATE(noc_id)                                   \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                         \
        noc_id,                                                                                  \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) |   \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO)),       \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO),                        \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_id)                               \
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(                                                     \
        noc_id,                                                                               \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID) << 32) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO)),     \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO),                    \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_id, cmd_buf)                                   \
    DEBUG_SANITIZE_NOC_ADDR(                                                                  \
        noc_id,                                                                               \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_MID) << 32) |      \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_LO)),              \
        4);
#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l)                                                      \
    debug_sanitize_noc_addr(noc_id, a, 0, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_id, noc_a, worker_a, l, multicast, dir)  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, multicast, dir); \
    LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, noc_a, worker_a, l)                                                  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l);                                                                                                  \
    debug_insert_delay((uint8_t)TransactionRead);
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_id, noc_a, worker_a, l)                                              \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l);                                                                                                    \
    debug_insert_delay((uint8_t)TransactionRead);
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)                                                  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_WRITE); \
    LOG_LEN(l);                                                                                                   \
    debug_insert_delay((uint8_t)TransactionWrite)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)                                              \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_WRITE); \
    LOG_LEN(l);                                                                                                     \
    debug_insert_delay((uint8_t)TransactionWrite);

#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)         \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                                    \
        noc_id,                                                                                             \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                           \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc_id, noc_a_lower, worker_a, l)               \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                                    \
        noc_id,                                                                                             \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                           \
        l);
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)           \
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(                                                                      \
        noc_id,                                                                                                \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                              \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_INSERT_DELAY(transaction_type) debug_insert_delay(transaction_type)

// Delay for debugging purposes
inline void debug_insert_delay(uint8_t transaction_type) {
#if defined(WATCHER_DEBUG_DELAY)
    debug_insert_delays_msg_t tt_l1_ptr *v = GET_MAILBOX_ADDRESS_DEV(watcher.debug_insert_delays);

    bool delay = false;
    switch (transaction_type) {
        case TransactionRead: delay = (v[0].read_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        case TransactionWrite: delay = (v[0].write_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        case TransactionAtomic: delay = (v[0].atomic_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        default: break;
    }
    if (delay) {
        // WATCHER_DEBUG_DELAY is a compile time constant passed with -D
        riscv_wait (WATCHER_DEBUG_DELAY);
        v[0].feedback |= (1 << transaction_type); // Mark that we have delayed on this transaction type
    }
#endif  // WATCHER_DEBUG_DELAY
}

#else  // !WATCHER_ENABLED

#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_id, noc_a, worker_a, l, multicast, dir) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a) \
    LOG_READ_LEN_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc_id, noc_a_lower, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a) \
    LOG_WRITE_LEN_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_id, cmd_buf)
#define DEBUG_INSERT_DELAY(transaction_type)

#endif  // WATCHER_ENABLED
