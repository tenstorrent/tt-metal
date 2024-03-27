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

#if defined(WATCHER_ENABLED)

extern uint8_t noc_index;

#include "noc_addr_ranges_gen.h"
#include "noc_parameters.h"
#include "noc_overlay_parameters.h"

#define DEBUG_VALID_L1_ADDR(a, l, extra_alignment_bytes) \
    ( \
        (a >= MEM_L1_BASE) && \
        (a + l <= MEM_L1_BASE + MEM_L1_SIZE) && \
        ((a) + (l) > (a)) && \
        ((a % NOC_L1_ALIGNMENT_BYTES) == 0) && \
        ((a % extra_alignment_bytes) == 0) \
    )

// TODO(PGK): remove soft reset when fw is downloaded at init
#define DEBUG_VALID_REG_ADDR(a)                                                        \
    ((((a) >= NOC_OVERLAY_START_ADDR) &&                                               \
      ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
     ((a) == RISCV_DEBUG_REG_SOFT_RESET_0))
#define DEBUG_VALID_WORKER_ADDR(a, l, extra_alignment_bytes) \
    ( \
        DEBUG_VALID_L1_ADDR(a, l, extra_alignment_bytes) || \
        (DEBUG_VALID_REG_ADDR(a) && (l) == 4) \
    )
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

// Return value is the alignment requirement (in bytes) for the type of core the noc address points
// to. Need to do this because L1 alignment needs to match the noc address alignment requirements,
// even if it's different than the inherent L1 alignment requirements.
uint32_t debug_sanitize_noc_addr(uint64_t noc_addr, uint32_t noc_len, bool multicast) {
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
            debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, DebugSanitizeNocInvalidMulticast);
        }
    }

    // Check noc addr, we save the alignment requirement from the noc src/dst because the L1 address
    // needs to match alignment (even if it's different than the L1 alignment requirements).
    uint32_t extra_alignment_req = NOC_L1_ALIGNMENT_BYTES;
    uint32_t invalid = multicast? DebugSanitizeNocInvalidMulticast : DebugSanitizeNocInvalidUnicast;
    if (NOC_PCIE_XY_P(x, y)) {
        extra_alignment_req = NOC_PCIE_ALIGNMENT_BYTES;
        if (!DEBUG_VALID_PCIE_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, invalid);
        }
    } else if (NOC_DRAM_XY_P(x, y)) {
        extra_alignment_req = NOC_DRAM_ALIGNMENT_BYTES;
        if (!DEBUG_VALID_DRAM_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, invalid);
        }
    } else if (NOC_ETH_XY_P(x, y)) {
        if (!DEBUG_VALID_ETH_ADDR(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, invalid);
        }
    } else if (NOC_WORKER_XY_P(x, y)) {
        if (!DEBUG_VALID_WORKER_ADDR(noc_local_addr, noc_len, NOC_L1_ALIGNMENT_BYTES)) {
            debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, invalid);
        }
    } else {
        // Bad XY
        debug_sanitize_post_noc_addr_and_hang(noc_addr, noc_len, invalid);
    }

    return extra_alignment_req;
}

void debug_sanitize_worker_addr(uint32_t addr, uint32_t len, uint32_t extra_alignment_req) {
    if (!DEBUG_VALID_WORKER_ADDR(addr, len, extra_alignment_req)) {
        debug_sanitize_post_noc_addr_and_hang(addr, len, DebugSanitizeNocInvalidL1);
    }
}

void debug_sanitize_noc_and_worker_addr(uint64_t noc_addr, uint32_t worker_addr, uint32_t len, bool multicast) {
    // Check noc addr, get any extra alignment req for worker.
    uint32_t extra_alignment_req = debug_sanitize_noc_addr(noc_addr, len, multicast);

    // Check worker addr
    debug_sanitize_worker_addr(worker_addr, len, extra_alignment_req);
}

#define DEBUG_SANITIZE_WORKER_ADDR(a, l) \
    debug_sanitize_worker_addr(a, l, NOC_L1_ALIGNMENT_BYTES)
#define DEBUG_SANITIZE_NOC_ADDR(a, l) \
    debug_sanitize_noc_addr(a, l, false)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l) \
    debug_sanitize_noc_addr(a, l, true)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, false)
#define DEBUG_SANITIZE_NOC_MULTI_TRANSACTION(noc_a, worker_a, l) \
    debug_sanitize_noc_and_worker_addr(noc_a, worker_a, l, true)

#else // !WATCHER_ENABLED

#define DEBUG_SANITIZE_WORKER_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_MULTI_ADDR(a, l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_a, worker_a, l)
#define DEBUG_SANITIZE_NOC_MULTI_TRANSACTION(noc_a, worker_a, l)

#endif // WATCHER_ENABLED

#endif // TENSIX_FIRMWARE
