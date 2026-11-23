// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused near partial, reader. Core (g, i) of kv group g takes slot tiles j0 .. j0 + nj - 1 of the 1024-slot ring:
// a ones tile, q_g (4 tiles of the kv-group q [1, 8, 32, 128]), the sink block tile, the
// mask tiles of its slots, then the ring's K tiles (4 per slot tile) and V tiles of head g in slot-tile order.
// pend (multi-token verify, token ntok - 1): rows 0 .. ntok - 1 of the pending K / V tiles (pending [2, 8, 32, 128])
// are copied into the ring tiles at slots (pos0 + t) % 1024 before they go to compute, so the token sees the ring
// plain decode would see at its step.
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
    constexpr uint32_t pend = get_named_compile_time_arg_val("pend");
    uint32_t pend_l1 = 0, pos0 = 0, ntok = 0;
    if constexpr (pend) {
        constexpr uint32_t cb_pend = 10;
        pend_l1 = get_write_ptr(cb_pend);
        const uint32_t pend_addr = get_arg_val<uint32_t>(8);
        const uint32_t pos_addr = get_arg_val<uint32_t>(9);
        ntok = get_arg_val<uint32_t>(10);
        const InterleavedAddrGen<true> pa = {.bank_base_address = pend_addr, .page_size = kv_page};
        for (uint32_t kv = 0; kv < 2; ++kv) {
            for (uint32_t c = 0; c < dt; ++c) {
                noc_async_read(pa.get_noc_addr((kv * 8 + g) * dt + c), pend_l1 + (kv * dt + c) * kv_page, kv_page);
            }
        }
        const InterleavedAddrGen<true> pp = {.bank_base_address = pos_addr, .page_size = 4};
        noc_async_read(pp.get_noc_addr(0, 0), pend_l1 + 2 * dt * kv_page, 16);
        noc_async_read_barrier();
        pos0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pend_l1 + 2 * dt * kv_page);
    }

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
            if constexpr (pend) {
                for (uint32_t t = 0; t < ntok; ++t) {
                    const uint32_t slot = (pos0 + t) % (ring_tiles * 32);
                    if ((slot >> 5) != j0 + j) {
                        continue;
                    }
                    for (uint32_t c = 0; c < dt; ++c) {
                        attnio::copy_tile_row(
                            pend_l1 + (pass * dt + c) * kv_page,
                            t,
                            get_write_ptr(cb) + c * kv_page,
                            slot & 31,
                            kv_page);
                    }
                }
            }
            cb_push_back(cb, dt);
        }
    }
}
