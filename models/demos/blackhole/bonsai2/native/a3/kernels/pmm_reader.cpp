// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Short-prefill ternary matmul (A3 at M>1), reader. y [M, N] = x' [M, K] @ (codes * scales), M <= 128.
// Per K super-block G (32 K-tiles = 8 scale groups): this block of x' (MT x 32 tiles) once per core, then for
// each of this core's weight columns its 32 code tiles (contiguous K-stick in its DRAM bank) and the one
// fp32 scale_rows tile holding the block's 8 scale rows.
// Runtime args: x_addr, w_addr, s_addr, c0, nc.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t nc = get_arg_val<uint32_t>(4);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t cpb = get_compile_time_arg_val(3);
    constexpr uint32_t wtb = get_compile_time_arg_val(4);
    constexpr uint32_t s_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t xtb = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = 0, cb_c = 1, cb_s = 2;
    const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = xtb};
    for (uint32_t G = 0; G < ng; ++G) {
        cb_reserve_back(cb_x, MT * 32);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t t = 0; t < 32; ++t, l1 += xtb) {
                noc_async_read(get_noc_addr(m * kt + G * 32 + t, xg), l1, xtb);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MT * 32);
        for (uint32_t c = c0; c < c0 + nc; ++c) {
            const uint32_t bank = c / cpb, j = c % cpb;
            cb_reserve_back(cb_s, 1);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, s_addr + (j * s_tiles + G / 4) * 4096),
                           get_write_ptr(cb_s), 4096);
            cb_reserve_back(cb_c, 32);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + (j * kt + G * 32) * wtb),
                           get_write_ptr(cb_c), 32 * wtb);
            noc_async_read_barrier();
            cb_push_back(cb_s, 1);
            cb_push_back(cb_c, 32);
        }
    }
}
