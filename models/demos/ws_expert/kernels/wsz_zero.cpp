// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert: zeroes L1 ranges on this core (bench_util.zero_pending: the zeros of tensors allocated without a host
// write, inside a trace). Each range: its first block zeroed by stores, the rest copied from it by local NoC writes.
// CT: 0 ranges (every core the same count; unused ranges have 0 bytes). RT: (address, bytes) pairs (4-B multiples).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t blk = 2048;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t addr = get_arg_val<uint32_t>(2 * r), bytes = get_arg_val<uint32_t>(1 + 2 * r);
        const uint32_t z = bytes < blk ? bytes : blk;
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < z / 4; ++i) {
            p[i] = 0;
        }
        for (uint32_t off = z; off < bytes; off += z) {
            const uint32_t len = bytes - off < z ? bytes - off : z;
            noc_async_write(addr, get_noc_addr(my_x[noc_index], my_y[noc_index], addr + off), len);
        }
    }
    noc_async_write_barrier();
}
