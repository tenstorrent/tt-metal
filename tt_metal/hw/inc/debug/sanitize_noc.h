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

#include "watcher_common.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE)

extern uint8_t noc_index;

#include "noc_addr_ranges_gen.h"
#include "noc_parameters.h"
#include "noc_overlay_parameters.h"

// A couple defines for specifying read/write and multi/unicast
#define DEBUG_SANITIZE_NOC_READ true
#define DEBUG_SANITIZE_NOC_WRITE false
typedef bool debug_sanitize_noc_dir_t;
#define DEBUG_SANITIZE_NOC_MULTICAST true
#define DEBUG_SANITIZE_NOC_UNICAST false
typedef bool debug_sanitize_noc_cast_t;

// TODO(PGK): remove soft reset when fw is downloaded at init
#define DEBUG_VALID_REG_ADDR(a, l) \
    ( \
        ( \
            ( \
                ((a) >= NOC_OVERLAY_START_ADDR) && \
                ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS) \
            ) || \
            ((a) == RISCV_DEBUG_REG_SOFT_RESET_0) \
        ) \
        && (l) == 4 \
    )
#define DEBUG_VALID_WORKER_ADDR(a, l) \
    ( \
        (a >= MEM_L1_BASE) && \
        (a + l <= MEM_L1_BASE + MEM_L1_SIZE) && \
        ((a) + (l) > (a)) \
    )
#define DEBUG_VALID_PCIE_ADDR(a, l) (((a) >= NOC_PCIE_ADDR_BASE) && \
                                     ((a) + (l) <= NOC_PCIE_ADDR_END) && \
                                     ((a) + (l) > (a)))
#define DEBUG_VALID_DRAM_ADDR(a, l) (((a) >= NOC_DRAM_ADDR_BASE) && \
                                     ((a) + (l) <= NOC_DRAM_ADDR_END) && \
                                     ((a) + (l) > (a)))

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && \
                                    ((a) + (l) <= MEM_ETH_BASE + MEM_ETH_SIZE))

// Note:
//  - this isn't racy w/ the host so long as invalid is written last
//  - this isn't racy between riscvs so long as each gets their own noc_index
inline void debug_sanitize_post_noc_addr_and_hang(
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    uint16_t invalid
) {
    debug_sanitize_noc_addr_msg_t tt_l1_ptr *v = *GET_MAILBOX_ADDRESS_DEV(sanitize_noc);

    if (v[noc_index].invalid == DebugSanitizeNocInvalidOK) {
        v[noc_index].noc_addr = noc_addr;
        v[noc_index].l1_addr = l1_addr;
        v[noc_index].len = len;
        v[noc_index].which = debug_get_which_riscv();
        v[noc_index].multicast = multicast;
        v[noc_index].invalid = invalid;
    }

#if defined(COMPILE_FOR_ERISC)
    // For erisc, we can't hang the kernel/fw, because the core doesn't get restarted when a new
    // kernel is written. In this case we'll do an early exit back to base FW.
    internal_::disable_erisc_app();
    erisc_early_exit(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE);
#endif

    while(1) {
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
    }
}

// Return value is the alignment mask for the type of core the noc address points
// to. Need to do this because L1 alignment needs to match the noc address alignment requirements,
// even if it's different than the inherent L1 alignment requirements. Note that additional
// alignment restrictions only apply for writes from L1, so need to specify direction as well.
uint32_t debug_sanitize_noc_addr(
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t noc_len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    // Different encoding of noc addr depending on multicast vs unitcast
    uint32_t x, y;
    if (multicast) {
        x = NOC_MCAST_ADDR_START_X(noc_addr);
        y = NOC_MCAST_ADDR_START_Y(noc_addr);
    } else {
        x = NOC_UNICAST_ADDR_X(noc_addr);
        y = NOC_UNICAST_ADDR_Y(noc_addr);
    }
    uint64_t noc_local_addr = NOC_LOCAL_ADDR_OFFSET(noc_addr);

    // Extra check for multicast
    if (multicast) {
        uint32_t x_end = NOC_MCAST_ADDR_END_X(noc_addr);
        uint32_t y_end = NOC_MCAST_ADDR_END_Y(noc_addr);

        if (!NOC_WORKER_XY_P(x, y) ||
            !NOC_WORKER_XY_P(x_end, y_end) ||
            (x > x_end || y > y_end)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_addr, l1_addr, noc_len,
                multicast, DebugSanitizeNocInvalidMulticast
            );
        }
    }

    // Check noc addr, we save the alignment requirement from the noc src/dst because the L1 address
    // needs to match alignment.
    uint32_t alignment_mask = NOC_L1_ALIGNMENT_BYTES-1; // Default alignment, only override in ceratin cases.
    uint32_t invalid = multicast? DebugSanitizeNocInvalidMulticast : DebugSanitizeNocInvalidUnicast;
    if (NOC_PCIE_XY_P(x, y)) {
        // Additional alignment restriction only applies to reads
        if (dir == DEBUG_SANITIZE_NOC_READ)
            alignment_mask = NOC_PCIE_ALIGNMENT_BYTES - 1;
        if (!DEBUG_VALID_PCIE_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else if (NOC_DRAM_XY_P(x, y)) {
        // Additional alignment restriction only applies to reads
        if (dir == DEBUG_SANITIZE_NOC_READ)
            alignment_mask = NOC_DRAM_ALIGNMENT_BYTES - 1;
        if (!DEBUG_VALID_DRAM_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else if (NOC_ETH_XY_P(x, y)) {
        if (!DEBUG_VALID_ETH_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else if (NOC_WORKER_XY_P(x, y)) {
        if (!DEBUG_VALID_REG_ADDR(noc_local_addr, noc_len) && !DEBUG_VALID_WORKER_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, l1_addr, noc_len, multicast, invalid);
        }
    } else {
        // Bad XY
        debug_sanitize_post_noc_addr_and_hang(noc_addr, l1_addr, noc_len, multicast, invalid);
    }

    return alignment_mask;
}

void debug_sanitize_worker_addr(uint32_t addr, uint32_t len) {
    // Regs are exempt from standard L1 validation
    if (DEBUG_VALID_REG_ADDR(addr, len))
        return;

    if (!DEBUG_VALID_WORKER_ADDR(addr, len)) {
        debug_sanitize_post_noc_addr_and_hang(addr, 0, len, false, DebugSanitizeNocInvalidL1);
    }
}

void debug_sanitize_noc_and_worker_addr(
    uint64_t noc_addr,
    uint32_t worker_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir
) {
    // Check noc addr, get any extra alignment req for worker.
    uint32_t alignment_mask = debug_sanitize_noc_addr(noc_addr, worker_addr, len, multicast, dir);

    // Check worker addr
    debug_sanitize_worker_addr(worker_addr, len);

    // Check alignment, but not for reg addresses.
    if (!DEBUG_VALID_REG_ADDR(worker_addr, len)) {
        if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, worker_addr, len, multicast, DebugSanitizeNocInvalidAlignment);
        }
    }
}

// TODO: should be able clean up uses of the first three macros and remove them.
#define DEBUG_SANITIZE_WORKER_ADDR(a, l) \
    debug_sanitize_worker_addr(a, l)
#define DEBUG_SANITIZE_NOC_ADDR(a, l) \
    debug_sanitize_noc_addr(a, 0, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l) \
    debug_sanitize_noc_addr(a, 0, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_READ)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_a, worker_a, l, multicast, dir) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, multicast, dir)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ)
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_READ)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_WRITE)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_WRITE)

#else // !WATCHER_ENABLED

#define DEBUG_SANITIZE_WORKER_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_a, worker_a, l, multicast, dir)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_a, worker_a, l)
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_a, worker_a, l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_a, worker_a, l)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_a, worker_a, l)

#endif // WATCHER_ENABLED

#endif // TENSIX_FIRMWARE
