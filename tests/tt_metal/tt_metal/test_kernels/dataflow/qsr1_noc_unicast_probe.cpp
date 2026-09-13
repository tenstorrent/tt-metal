// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qsr.s1 bring-up probe (device-initiated inter-tile NoC traffic). Writes `value` into local L1 at
// `result_addr`, then unicasts that word (a) to `result_addr` on the worker at NoC (dst_x, dst_y) and
// (b) to `dram_addr` on the DRAM tile at NoC (dram_x, dram_y). The host seeds sentinels everywhere and
// reads all tiles back to see WHERE the two writes actually landed.

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t value = get_arg(args::value);
    const uint32_t result_addr = get_arg(args::result_addr);
    const uint32_t dst_x = get_arg(args::dst_x);
    const uint32_t dst_y = get_arg(args::dst_y);
    const uint32_t dram_x = get_arg(args::dram_x);
    const uint32_t dram_y = get_arg(args::dram_y);
    const uint32_t dram_addr = get_arg(args::dram_addr);

    CoreLocalMem<uint32_t> buf(result_addr);
    buf[0] = value;
    flush_l2_cache_line(result_addr);

    // Posted writes, deliberately WITHOUT a barrier: on qsr.s1 a barrier after a remote write never
    // returns (probe run p4), which would hide where the data went. The host waits, then reads back.
    const uint64_t dst_l1 = get_noc_addr(dst_x, dst_y, result_addr, noc_index);
    noc_async_write(result_addr, dst_l1, sizeof(uint32_t));

    const uint64_t dst_dram = get_noc_addr(dram_x, dram_y, dram_addr, noc_index);
    noc_async_write(result_addr, dst_dram, sizeof(uint32_t));

    // Self-diagnostics for the host: what does this core believe about itself and the destination?
    //   +8  my_x[noc_index] | my_y << 8 | noc_index << 16  (firmware's own idea of its NoC node id)
    //   +12 raw NOC_NODE_ID register as read through the cmd-buf register path
    //   +16 the DEST_COORD value programmed for (dst_x, dst_y)
    //   +20 the DEST_ADDR (low 32 bits) programmed
    CoreLocalMem<uint32_t> diag(result_addr + 8);
    diag[0] = (uint32_t)my_x[noc_index] | ((uint32_t)my_y[noc_index] << 8) | ((uint32_t)noc_index << 16);
    diag[1] = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    diag[2] = (uint32_t)(dst_l1 >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    diag[3] = (uint32_t)dst_l1;
    flush_l2_cache_line(result_addr + 8);
    flush_l2_cache_line(result_addr + 16);

    // Local "kernel reached the end" flag, one word after the value.
    CoreLocalMem<uint32_t> done(result_addr + 4);
    done[0] = 0xD0DE0001u;
    flush_l2_cache_line(result_addr + 4);
}
