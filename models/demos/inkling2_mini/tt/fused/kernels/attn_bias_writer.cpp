// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, bias role, writer: mask tiles j0 .. j0 + nj - 1 -> mask [1, 1, 24, 1024] bf16 (DRAM).
// has_sink: also the sdpa sink tile [24, 32] bf16, row h = sink logit h (projection row 0, 24 columns).
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_sink = get_named_compile_time_arg_val("has_sink");
    constexpr uint32_t sink_tile = get_named_compile_time_arg_val("sink_tile");  // projection tile of sink col 0
    constexpr uint32_t cb_mask = 21, cb_sink = 6, cb_stage = 25;

    const uint32_t mask = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t proj = get_arg_val<uint32_t>(2);
    const uint32_t sink = get_arg_val<uint32_t>(3);

    if constexpr (has_sink) {
        const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
        cb_reserve_back(cb_sink, 1);
        const uint32_t w = get_write_ptr(cb_sink);
        attnio::zero_l1(w, 2048);
        attnio::gather_row(proj, 2048, 2, sink_tile, 1, 0, stage + 256, stage);  // barriers (zeros too)
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + 256);
        for (uint32_t h = 0; h < 24; ++h) {
            const uint32_t word = static_cast<uint32_t>(v[h]) * 0x10001u;
            const uint32_t face = h < 16 ? 0 : 2;
            volatile tt_l1_ptr uint32_t* row =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + (face * 256 + (h & 15) * 16) * 2);
            for (uint32_t k = 0; k < 8; ++k) {
                row[k] = word;        // face (cols 0..15)
                row[128 + k] = word;  // face + 1 (cols 16..31)
            }
        }
        const InterleavedAddrGen<true> dst = {.bank_base_address = sink, .page_size = 2048};
        noc_async_write(w, dst.get_noc_addr(0), 2048);
    }

    const InterleavedAddrGen<true> dst = {.bank_base_address = mask, .page_size = 2048};
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_mask, 1);
        noc_async_write(get_read_ptr(cb_mask), dst.get_noc_addr(j0 + j), 2048);
        noc_async_write_barrier();
        cb_pop_front(cb_mask, 1);
    }
    noc_async_write_barrier();
}
