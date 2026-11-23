// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention sink, writer: per row R, head h's tile to page h * RT + R of [1, H, T, 1].
// Runtime args: out, n_rows, R per row.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_named_compile_time_arg_val("h");
    constexpr uint32_t RT = get_named_compile_time_arg_val("rt");
    constexpr uint32_t cb_out = 16;
    const uint32_t out = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = 2048};
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t R = get_arg_val<uint32_t>(2 + i);
        for (uint32_t h = 0; h < H; ++h) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), oa.get_noc_addr(h * RT + R), 2048);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
