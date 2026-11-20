// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC (non-IDMA) traffic from a Keraunos CCE hart through the overlay command buffer, to find out whether the plain
// NOC path works on a CCE at all (noc_semaphore_inc from a CCE issues once and never completes).
//
// Status words (uint32, uncached alias): 0 started, 1 issued count, 2 all issued, 4 done.
//
// Runtime args:
//   0, 1: this core's NOC x, y
//   2, 3: destination NOC x, y
//   4:    destination SRAM offset (the target core's own offset, as the host uses it)
//   5:    source SRAM offset in this core (mode 0)
//   6:    mode: 0 = one non-posted NOC write of `len` bytes, waiting for its ack;
//               1 = `count` posted NOC atomic increments of `incr` (no ack to wait for)
//   7:    len (mode 0) or count (mode 1)
//   8:    incr (mode 1)
//   9:    status block, local SRAM offset

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

using namespace overlay;

void kernel_main() {
    const uint32_t my_x = get_arg_val<uint32_t>(0);
    const uint32_t my_y = get_arg_val<uint32_t>(1);
    const uint32_t dst_x = get_arg_val<uint32_t>(2);
    const uint32_t dst_y = get_arg_val<uint32_t>(3);
    const uint32_t dst_offset = get_arg_val<uint32_t>(4);
    const uint32_t src_offset = get_arg_val<uint32_t>(5);
    const uint32_t mode = get_arg_val<uint32_t>(6);
    const uint32_t n = get_arg_val<uint32_t>(7);
    const uint32_t incr = get_arg_val<uint32_t>(8);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(9));

    status[0] = 0x1D3A0001;
    const uint64_t my_coord = NOC_XY_COORD(my_x, my_y);
    const uint64_t dst_coord = NOC_XY_COORD(dst_x, dst_y);
    reset_cmdbuf<CMDBUF_0>();
    if (mode == 0) {
        // The source is this core's own SRAM in its cached view, which on CCE1 starts at ETH_CCE_LOCAL_BASE.
        noc_write_cmdbuf<CMDBUF_0>(my_coord, ETH_CCE_LOCAL_BASE + src_offset, dst_coord, dst_offset, n);
        status[1] = 1;
        status[2] = 0x1D3A0002;
        while (!noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>()) {
        }
    } else {
        for (uint32_t i = 0; i < n; i++) {
            noc_atomic_increment_cmdbuf<CMDBUF_0>(dst_coord, dst_offset, incr);
            status[1] = i + 1;
        }
        status[2] = 0x1D3A0002;
    }
    status[4] = 0x1D3A0003;
}
