// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused near partial, reader. Core (g, i) of kv group g takes slot tiles j0 .. j0 + nj - 1 of the 1024-slot ring:
// a ones tile, q_g (4 tiles of the kv-group q [1, 8, 32, 128]), the sink block tile, the
// mask tiles of its slots, then the ring's K tiles (4 per slot tile) and V tiles of head g in slot-tile order.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t kv_page = get_named_compile_time_arg_val("kv_page");  // cache tile bytes
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");            // slot tiles of this core
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_sink = 4, cb_ones = 6;
    constexpr uint32_t ring_tiles = 32, dt = 4;  // slot tiles of the ring, head-dim tiles

    const uint32_t q = get_arg_val<uint32_t>(0);
    const uint32_t mask = get_arg_val<uint32_t>(1);
    const uint32_t sink = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);
    const uint32_t v = get_arg_val<uint32_t>(4);
    const uint32_t g = get_arg_val<uint32_t>(5);
    const uint32_t ones = get_arg_val<uint32_t>(6);
    const uint32_t j0 = get_arg_val<uint32_t>(7);

    cb_reserve_back(cb_ones, 1);
    attnio::read_pages(ones, 2048, 0, 1, get_write_ptr(cb_ones));
    cb_reserve_back(cb_q, dt);
    attnio::read_pages(q, 2048, g * dt, dt, get_write_ptr(cb_q));
    cb_reserve_back(cb_sink, 1);
    attnio::read_pages(sink, 4096, g, 1, get_write_ptr(cb_sink));
    cb_reserve_back(cb_mask, nj);
    attnio::read_pages(mask, 2048, g * ring_tiles + j0, nj, get_write_ptr(cb_mask));
    noc_async_read_barrier();
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_q, dt);
    cb_push_back(cb_sink, 1);
    cb_push_back(cb_mask, nj);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t cb = pass == 0 ? cb_k : cb_v;
        const uint32_t base = pass == 0 ? k : v;
        for (uint32_t j = 0; j < nj; ++j) {
            cb_reserve_back(cb, dt);
            attnio::read_pages(base, kv_page, (g * ring_tiles + j0 + j) * dt, dt, get_write_ptr(cb));
            noc_async_read_barrier();
            cb_push_back(cb, dt);
        }
    }
}
