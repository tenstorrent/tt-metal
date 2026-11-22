// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Near partial combine, writer (kv group g): o tiles -> pages 4g .. of o_near [1, 8, 3, 128] fp32, the r tile ->
// page g of [1, 8, 3, 1] fp32, rows 0..2 of the bf16 sink tile -> rows 3g .. 3g + 2 of the far call's sink [24, 32].
#include "attn_io.hpp"

namespace {
// rows 0..2 of a bf16 tile at L1 src -> rows 3g .. 3g + 2 of page `page` of a bf16 TILE tensor
FORCE_INLINE void rows3(uint32_t src, const InterleavedAddrGen<true>& d, uint32_t page, uint32_t g) {
    for (uint32_t k = 0; k < 3; ++k) {
        const uint32_t h = 3 * g + k, f0 = (h >> 4) * 2;
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write(
                src + (f * 256 + k * 16) * 2, d.get_noc_addr(page, ((f0 + f) * 256 + (h & 15) * 16) * 2), 32);
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_out = 17, cb_st = 19, cb_r = 20, dt = 4;
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t sink = get_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(3);

    const InterleavedAddrGen<true> od = {.bank_base_address = out, .page_size = 4096};
    for (uint32_t t = 0; t < dt; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), od.get_noc_addr(g * dt + t), 4096);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    {
        cb_wait_front(cb_st, 1);
        const InterleavedAddrGen<true> sd = {.bank_base_address = sink, .page_size = 2048};
        rows3(get_read_ptr(cb_st), sd, 0, g);
        cb_wait_front(cb_r, 1);
        const InterleavedAddrGen<true> rd = {.bank_base_address = r, .page_size = 4096};
        noc_async_write(get_read_ptr(cb_r), rd.get_noc_addr(g), 4096);
        noc_async_write_barrier();
        cb_pop_front(cb_st, 1);
        cb_pop_front(cb_r, 1);
    }
}
