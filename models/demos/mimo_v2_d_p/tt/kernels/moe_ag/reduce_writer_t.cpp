// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE local reduce with a tiled output, writer (BRISC): each tile row (c_16, TPR tiles) of this core's
// tokens [g0, g0 + n) (32-aligned) into the bf16 TILE [T, H] partials.
// CT: 0 TPR (tiles per tile row = H / 32)   RT: 0 out addr, 1 g0, 2 n
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPR = get_compile_time_arg_val(0);
    const InterleavedAddrGen<true> o = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 2048};
    const uint32_t g0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    for (uint32_t b = 0; b < n / 32; ++b) {
        cb_wait_front(tt::CBIndex::c_16, TPR);
        const uint32_t src = get_read_ptr(tt::CBIndex::c_16);
        const uint32_t tr = g0 / 32 + b;
        for (uint32_t t = 0; t < TPR; ++t) {
            noc_async_write(src + t * 2048, get_noc_addr(tr * TPR + t, o), 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(tt::CBIndex::c_16, TPR);
    }
    noc_async_full_barrier();
}
