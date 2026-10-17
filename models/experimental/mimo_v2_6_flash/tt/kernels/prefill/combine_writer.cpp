// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC of the prefill MoE combine: each item's 4 output tiles into the [T, H] bf16 tile output.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t block = 4;
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = tile_bytes};
    for (uint32_t i = first; i < first + count; i++) {
        const uint32_t tt = i / (nt / block);
        const uint32_t c0 = (i % (nt / block)) * block;
        cb_wait_front(cb_out, block);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < block; c++) {
            noc_async_write(src, out_gen.get_noc_addr(tt * nt + c0 + c, 0), tile_bytes);
            src += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, block);
    }
}
