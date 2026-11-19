// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 combine "rm" -- writer (moe_combine_rm_reader.cpp): unit u's 32 PT bf16 tiles = tile row u / H_SPLIT,
// tile columns (u % H_SPLIT) 32 PT .. of y [C, H] TILE. Compile-time: PT, HT (= H / 32), H_SPLIT. Runtime: y address,
// first unit, unit count.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t H_SPLIT = get_compile_time_arg_val(2);
    constexpr uint32_t CB_OUT = 16;
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t NT = 32 * PT;
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);
    if (nu == 0) {
        return;
    }
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = TILE_BYTES};
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t chunk = u / H_SPLIT;
        const uint32_t base = chunk * HT + (u - chunk * H_SPLIT) * NT;
        cb_wait_front(CB_OUT, NT);
        const uint32_t rp = get_read_ptr(CB_OUT);
        for (uint32_t x = 0; x < NT; ++x) {
            noc_async_write(rp + x * TILE_BYTES, yg.get_noc_addr(base + x), TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, NT);
    }
}
