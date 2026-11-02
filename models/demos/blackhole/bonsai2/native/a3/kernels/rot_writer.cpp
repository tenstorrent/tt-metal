// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill rotation, writer: output tile i of unit (b, r) to page r*kt + b*32 + i.
// Runtime args: y_addr, u0, nu.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t ytb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = 16;
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = ytb};
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t b = u / R, r = u % R;
        for (uint32_t i = 0; i < 32; ++i) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), get_noc_addr(r * kt + b * 32 + i, yg), ytb);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
