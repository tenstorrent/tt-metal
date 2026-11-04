// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill rotation x' = blockdiag(H_1024) D x, reader (NCRISC). This core rotates 1024-block b of row tiles
// m_first .. m_first + m_count - 1: the block's 64 weight tiles (per input tile a of the block: +-diag(d_a) H32 / 32)
// into c_1 once, then per row tile the block's 32 input tiles into c_0.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t Kt = get_common_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t m_first = get_arg_val<uint32_t>(1);
    const uint32_t m_count = get_arg_val<uint32_t>(2);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto h_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_h = tt::CBIndex::c_1;
    constexpr uint32_t TB = 2048;
    if (m_count == 0) {
        return;
    }
    const auto hs = TensorAccessor(h_args, h_addr, TB);
    cb_reserve_back(cb_h, 64);
    const uint32_t lh = get_write_ptr(cb_h);
    for (uint32_t t = 0; t < 64; ++t) {
        noc_async_read(hs.get_noc_addr(64 * b + t), lh + t * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_h, 64);
    const auto xs = TensorAccessor(x_args, x_addr, TB);
#ifdef GATE
    // GATE: x is head-major [1, H, T, HD tiles] (attention output), the gate token-major [T, Kt] (c_3)
    const uint32_t g_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t Mt = get_common_arg_val<uint32_t>(4), HD = get_common_arg_val<uint32_t>(5);
    constexpr auto g_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto gs = TensorAccessor(g_args, g_addr, TB);
    constexpr uint32_t cb_g = tt::CBIndex::c_3;
#endif
    for (uint32_t m = m_first; m < m_first + m_count; ++m) {
        cb_reserve_back(cb_x, 32);
        const uint32_t l1 = get_write_ptr(cb_x);
#ifdef GATE
        cb_reserve_back(cb_g, 32);
        const uint32_t lg = get_write_ptr(cb_g);
        for (uint32_t a = 0; a < 32; ++a) {
            const uint32_t n = 32 * b + a;
            noc_async_read(xs.get_noc_addr(((n / HD) * Mt + m) * HD + n % HD), l1 + a * TB, TB);
            noc_async_read(gs.get_noc_addr(m * Kt + n), lg + a * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_g, 32);
#else
        for (uint32_t a = 0; a < 32; ++a) {
            noc_async_read(xs.get_noc_addr(m * Kt + 32 * b + a), l1 + a * TB, TB);
        }
        noc_async_read_barrier();
#endif
        cb_push_back(cb_x, 32);
    }
}
