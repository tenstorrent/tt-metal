// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill (residual add +) RMSNorm + rotation (norm_rot.py), writer (BRISC). In production order: HAS_ADD: h (the new
// residual); then per 1024-wide block WRITE_N: n (the normalised activation), and x' (rotated).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xr_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t Kt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1), b1 = get_arg_val<uint32_t>(2);
    constexpr auto xr_args = TensorAccessorArgs<0>();
    constexpr auto n_args = TensorAccessorArgs<xr_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_out = 16, cb_n = 17, cb_hout = 18;
    if (me >= rows) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_out);
    auto drain = [&](uint32_t cb, auto& dst, uint32_t page) {
        cb_wait_front(cb, 1);
        noc_async_write(get_read_ptr(cb), dst.get_noc_addr(page), TB);
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    };
#ifdef HAS_ADD
    const auto hs = TensorAccessor(h_args, h_addr, TB);
    for (uint32_t t = 32 * b0; t < 32 * b1; ++t) {
        drain(cb_hout, hs, me * Kt + t);
    }
#endif
    const auto xs = TensorAccessor(xr_args, xr_addr, TB);
#ifdef WRITE_N
    const auto ns = TensorAccessor(n_args, n_addr, TB);
#endif
    for (uint32_t b = b0; b < b1; ++b) {
#ifdef WRITE_N
        for (uint32_t t = 0; t < 32; ++t) {
            drain(cb_n, ns, me * Kt + 32 * b + t);
        }
#endif
        for (uint32_t t = 0; t < 32; ++t) {
            drain(cb_out, xs, me * Kt + 32 * b + t);
        }
    }
    noc_async_write_barrier();
}
