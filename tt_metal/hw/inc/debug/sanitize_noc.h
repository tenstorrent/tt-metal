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

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)

#if defined(WATCHER_ENABLED)

extern uint8_t noc_index;

#include "noc_addr_ranges_gen.h"
#include "noc_parameters.h"
#include "noc_overlay_parameters.h"

#define DEBUG_VALID_L1_ADDR(a, l) ((a >= MEM_L1_BASE) && \
                                   (a + l <= MEM_L1_BASE + MEM_L1_SIZE) && \
                                   ((a) + (l) > (a)) && \
                                   ((a % NOC_L1_ALIGNMENT_BYTES) == 0))

// TODO(PGK): remove soft reset when fw is downloaded at init
#define DEBUG_VALID_REG_ADDR(a)                                                        \
    ((((a) >= NOC_OVERLAY_START_ADDR) &&                                               \
      ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
     ((a) == RISCV_DEBUG_REG_SOFT_RESET_0))
#define DEBUG_VALID_WORKER_ADDR(a, l) (DEBUG_VALID_L1_ADDR(a, l) || (DEBUG_VALID_REG_ADDR(a) && (l) == 4))
#define DEBUG_VALID_PCIE_ADDR(a, l) (((a) >= NOC_PCIE_ADDR_BASE) && \
                                     ((a) + (l) <= NOC_PCIE_ADDR_END) && \
                                     ((a) + (l) > (a)) && \
                                     ((a % NOC_PCIE_ALIGNMENT_BYTES) == 0))
#define DEBUG_VALID_DRAM_ADDR(a, l) (((a) >= NOC_DRAM_ADDR_BASE) && \
                                     ((a) + (l) <= NOC_DRAM_ADDR_END) && \
                                     ((a) + (l) > (a)) && \
                                     ((a % NOC_DRAM_ALIGNMENT_BYTES) == 0))

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && \
                                    ((a) + (l) <= MEM_ETH_BASE + MEM_ETH_SIZE) && \
                                    ((a % NOC_L1_ALIGNMENT_BYTES) == 0))

// Note:
//  - this isn't racy w/ the host so long as invalid is written last
//  - this isn't racy between riscvs so long as each gets their own noc_index
inline void debug_sanitize_post_noc_addr_and_hang(uint64_t a, uint32_t l, uint32_t invalid)
{
    debug_sanitize_noc_addr_msg_t tt_l1_ptr *v = *GET_MAILBOX_ADDRESS_DEV(sanitize_noc);

    if (v[noc_index].invalid == DebugSanitizeNocInvalidOK) {
        v[noc_index].addr = a;
        v[noc_index].len = l;
        v[noc_index].which = debug_get_which_riscv();
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

inline void debug_sanitize_worker_addr(uint32_t a, uint32_t l)
{
    if (!DEBUG_VALID_WORKER_ADDR(a, l)) {
        debug_sanitize_post_noc_addr_and_hang(a, l, DebugSanitizeNocInvalidL1);
    }
}

inline void debug_sanitize_noc_addr(uint32_t x, uint32_t y, uint64_t la, uint64_t a, uint32_t l, uint32_t invalid)
{
    if (NOC_PCIE_XY_P(x, y)) {
        if (!DEBUG_VALID_PCIE_ADDR(la, l)) {
            debug_sanitize_post_noc_addr_and_hang(a, l, invalid);
        }
    } else if (NOC_DRAM_XY_P(x, y)) {
        if (!DEBUG_VALID_DRAM_ADDR(la, l)) {
            debug_sanitize_post_noc_addr_and_hang(a, l, invalid);
        }
    } else if (NOC_ETH_XY_P(x, y)) {
        if (!DEBUG_VALID_ETH_ADDR(la, l)) {
            debug_sanitize_post_noc_addr_and_hang(a, l, invalid);
        }
    } else if (NOC_WORKER_XY_P(x, y)) {
        if (!DEBUG_VALID_WORKER_ADDR(la, l)) {
            debug_sanitize_post_noc_addr_and_hang(a, l, invalid);
        }
    } else {
        // Bad XY
        debug_sanitize_post_noc_addr_and_hang(a, l, invalid);
    }
}

void debug_sanitize_noc_uni_addr(uint64_t a, uint32_t l)
{
    uint32_t x = NOC_UNICAST_ADDR_X(a);
    uint32_t y = NOC_UNICAST_ADDR_Y(a);
    uint64_t la = NOC_LOCAL_ADDR_OFFSET(a);
    debug_sanitize_noc_addr(x, y, la, a, l, DebugSanitizeNocInvalidUnicast);
}

void debug_sanitize_noc_multi_addr(uint64_t a, uint32_t l)
{
    uint32_t xs = NOC_MCAST_ADDR_START_X(a);
    uint32_t ys = NOC_MCAST_ADDR_START_Y(a);
    uint32_t xe = NOC_MCAST_ADDR_END_X(a);
    uint32_t ye = NOC_MCAST_ADDR_END_Y(a);

    if (!NOC_WORKER_XY_P(xs, ys) ||
        !NOC_WORKER_XY_P(xe, ye) ||
        (xs > xe || ys > ye)) {
        debug_sanitize_post_noc_addr_and_hang(a, l, DebugSanitizeNocInvalidMulticast);
    }
    // Use one XY to determine the type for validating the address
    uint64_t la = NOC_LOCAL_ADDR_OFFSET(a);
    debug_sanitize_noc_addr(xs, ys, la, a, l, DebugSanitizeNocInvalidMulticast);
}

#define DEBUG_SANITIZE_WORKER_ADDR(a, l) debug_sanitize_worker_addr(a, l)
#define DEBUG_SANITIZE_NOC_ADDR(a, l) debug_sanitize_noc_uni_addr(a, l)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l) debug_sanitize_noc_multi_addr(a, l)

#else // !WATCHER_ENABLED

#define DEBUG_SANITIZE_WORKER_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_MULTI_LOOPBACK_ADDR(a, l)

#endif // WATCHER_ENABLED

#endif // TENSIX_FIRMWARE
