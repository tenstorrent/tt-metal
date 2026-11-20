// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC atomic increment (noc_semaphore_inc, the call behind tt-metal semaphores and fabric credits) from a Keraunos CCE
// hart to a word in another core's SRAM, then waits for every increment to be acked.
//
// Status words (uint32, uncached alias): 0 started, 1 issued count, 4 done.
//
// Runtime args:
//   0, 1: destination NOC x, y
//   2:    destination SRAM offset (the target core's own offset, as the host uses it)
//   3:    increments to issue
//   4:    increment value
//   5:    status block, local SRAM offset
//   6:    1 = posted (what tt-metal fabric issues; no ack to wait for), 0 = non-posted, waits for every ack

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    const uint32_t dst_x = get_arg_val<uint32_t>(0);
    const uint32_t dst_y = get_arg_val<uint32_t>(1);
    const uint32_t dst_offset = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t incr = get_arg_val<uint32_t>(4);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(5));

    status[0] = 0x1D3A0001;
    const uint64_t dst = get_noc_addr(dst_x, dst_y, dst_offset);
    const uint32_t posted = get_arg_val<uint32_t>(6);
    for (uint32_t i = 0; i < count; i++) {
        if (posted) {
            noc_semaphore_inc<true>(dst, incr);
        } else {
            noc_semaphore_inc<false>(dst, incr);
        }
        status[1] = i + 1;
    }
    status[2] = 0x1D3A0002;  // all issued
    if (!posted) {
        noc_async_atomic_barrier();
    }
    status[4] = 0x1D3A0003;
}
