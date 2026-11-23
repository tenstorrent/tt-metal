// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Conv window for a multi-token verify pass (single-input mode, data movement only, one core per projection-input
// shard): token t's lag group goes to rows 4 t .. 4 t + 3 of each window tile: row 4 t + l = the input of token
// t - l (the pass's input rows xs[t - l], or the window state's row l - t - 1 for t - l < 0), i.e. the rows plain
// decode's window would hold at token t's step. The rest of the tile is zero. The tile goes to this core's L1 shard
// of the projection input and to `scratch` (DRAM, for the commit); the window state is not written.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");  // window tiles of this core
    constexpr uint32_t R = get_named_compile_time_arg_val("R");            // tokens, <= 4
    constexpr uint32_t cb_old = 0, cb_new = 1, cb_stage = 24;

    const uint32_t hist = get_arg_val<uint32_t>(0);
    const uint32_t shard = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);  // first window tile of this core
    const uint32_t scratch = get_arg_val<uint32_t>(3);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const InterleavedAddrGen<true> h = {.bank_base_address = hist, .page_size = 2048};
    const InterleavedAddrGen<true> sc = {.bank_base_address = scratch, .page_size = 2048};
    const uint32_t old = get_write_ptr(cb_old), nw = get_write_ptr(cb_new);
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(h.get_noc_addr(t0 + j), old + j * 2048, 2048);
        for (uint32_t t = 0; t < R; ++t) {
            const InterleavedAddrGen<true> x = {.bank_base_address = get_arg_val<uint32_t>(4 + t), .page_size = 2048};
            for (uint32_t f = 0; f < 2; ++f) {  // input row: 64B block at face f row 0
                noc_async_read(x.get_noc_addr(t0 + j, f * 512), stage + ((j * R + t) * 2 + f) * 64, 64);
            }
        }
        attnio::zero_l1(nw + j * 2048, 2048);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        const uint32_t o = old + j * 2048, n = nw + j * 2048;
        for (uint32_t t = 0; t < R; ++t) {
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t row = 4 * t + l;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t src = (t >= l) ? stage + ((j * R + (t - l)) * 2 + f) * 64
                                                  : o + f * 512 + (l - t - 1) * 32;  // window state row l - t - 1
                    noc_async_read(get_noc_addr(src), n + f * 512 + row * 32, 32);
                }
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_write(nw + j * 2048, get_noc_addr(shard + j * 2048), 2048);
        noc_async_write(nw + j * 2048, sc.get_noc_addr(t0 + j), 2048);
    }
    noc_async_write_barrier();
}
