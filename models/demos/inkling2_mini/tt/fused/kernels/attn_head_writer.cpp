// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, head role, writer: each item's bf16 head row -> row r of tiles t0 .. t0 + 3 of up to two
// destination tensors (interleaved DRAM, or one L1 shard at a given NOC core).
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t ndst = get_named_compile_time_arg_val("ndst");
    constexpr uint32_t cb_out = 19;

    uint32_t a = 0;
    for (uint32_t i = 0; i < nitems; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t d = 0; d < ndst; ++d) {
            const uint32_t base = get_arg_val<uint32_t>(a++);
            const uint32_t l1 = get_arg_val<uint32_t>(a++);
            const uint32_t x = get_arg_val<uint32_t>(a++);
            const uint32_t y = get_arg_val<uint32_t>(a++);
            const uint32_t t0 = get_arg_val<uint32_t>(a++);
            const uint32_t r = get_arg_val<uint32_t>(a++);
            attnio::scatter_head(base, 2048, 2, t0, r, src, l1, x, y);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
