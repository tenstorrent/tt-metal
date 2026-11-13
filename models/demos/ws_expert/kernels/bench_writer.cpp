// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbench writer (NCRISC): pops the output CB (16) block by block; the last pass's NOUT tiles go to DRAM.
// RT: 0 out addr, 1 total tiles popped, 2 NOUT (the tail written out), 3 pop granularity.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ao = get_arg_val<uint32_t>(0);
    const uint32_t total = get_arg_val<uint32_t>(1);
    const uint32_t nout = get_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(3);
    constexpr auto argso = TensorAccessorArgs<0>();
    const auto so = TensorAccessor(argso, ao, get_tile_size(16));
    for (uint32_t i = 0; i < total; i += g) {
        cb_wait_front(16, g);
        if (i + nout >= total) {  // the tail
            uint32_t l1 = get_read_ptr(16);
            for (uint32_t j = 0; j < g; ++j) {
                const uint32_t t = i + j - (total - nout);
                noc_async_write_page(t, so, l1);
                l1 += get_tile_size(16);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(16, g);
    }
}
