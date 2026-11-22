// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused short conv, writer: fp32 x -> ring slot (dense pages), output row (row 0 of a row-form tensor).
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_xf = 16;

    const uint32_t slot = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_tile = get_arg_val<uint32_t>(2);
    const uint32_t out_es = get_arg_val<uint32_t>(3);

    cb_wait_front(cb_xf, T);
    rowio::write_pages(slot, 4096, 0, T, get_read_ptr(cb_xf));
    cb_wait_front(cb_out, T);
    rowio::scatter_row(out_addr, out_tile, out_es, ntc, 0, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_xf, T);
    cb_pop_front(cb_out, T);
}
