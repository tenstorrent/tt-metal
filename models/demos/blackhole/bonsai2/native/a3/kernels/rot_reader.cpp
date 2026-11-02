// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill rotation x' = H(D x), reader. Unit u = (b, r), b-major: b = u / R, r = u % R (r = 32-row
// tile row, b = 1024-wide block). Per unit: the 32 x tiles of (r, b) into cb_x. When b changes: the
// block's 64 rotation tiles (pn tensor, tile b*64 + 2j + neg = (-1)^neg diag(d_bj) H32 / 32) into cb_pn.
// Runtime args: x_addr, pn_addr, u0, nu.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t pn_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);
    constexpr uint32_t R = get_compile_time_arg_val(0);   // tile rows of x
    constexpr uint32_t kt = get_compile_time_arg_val(1);  // tile columns of x
    constexpr uint32_t xtb = get_compile_time_arg_val(2);
    constexpr uint32_t ptb = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = 0, cb_pn = 1;
    const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = xtb};
    const InterleavedAddrGen<true> pg{.bank_base_address = pn_addr, .page_size = ptb};
    uint32_t cur_b = 0xffffffff;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t b = u / R, r = u % R;
        if (b != cur_b) {
            cb_reserve_back(cb_pn, 64);
            uint32_t l1 = get_write_ptr(cb_pn);
            for (uint32_t t = 0; t < 64; ++t, l1 += ptb) {
                noc_async_read(get_noc_addr(b * 64 + t, pg), l1, ptb);
            }
            noc_async_read_barrier();
            cb_push_back(cb_pn, 64);
            cur_b = b;
        }
        cb_reserve_back(cb_x, 32);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t j = 0; j < 32; ++j, l1 += xtb) {
            noc_async_read(get_noc_addr(r * kt + b * 32 + j, xg), l1, xtb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, 32);
    }
}
