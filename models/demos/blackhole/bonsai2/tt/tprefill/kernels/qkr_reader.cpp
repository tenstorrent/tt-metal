// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill per-head RMSNorm + partial RoPE (qk_rope.py), reader (NCRISC). The const tiles once; per item (r, h), items
// r-major: the head's Dt tiles of row tile r into c_0 and cos / sin tiles (r, 0..1) into c_2.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: x, consts, cos, sin, items, ncores, grid_x, H, Mt, Dt, Rt (cos / sin tile columns)
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(3);
    uint32_t first, n_items;
    core_span(
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        get_common_arg_val<uint32_t>(6),
        1,
        first,
        n_items);
    const uint32_t H = get_common_arg_val<uint32_t>(7), Mt = get_common_arg_val<uint32_t>(8);
    const uint32_t Dt = get_common_arg_val<uint32_t>(9), Rt = get_common_arg_val<uint32_t>(10);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_x = 0, cb_c = 1, cb_cs = 2;
#ifdef KV_CACHE
    constexpr auto v_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    const auto vs = TensorAccessor(v_args, get_common_arg_val<uint32_t>(11), get_tile_size(cb_x));
    constexpr uint32_t cb_v = 5;
#endif
    if (n_items == 0) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_x), CT = get_tile_size(cb_c);
    const auto xs = TensorAccessor(x_args, x_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, CT);
#ifdef KV_CACHE
    const uint32_t c_off = get_common_arg_val<uint32_t>(12);  // the consts' first tile (one tensor for many ops)
#else
    const uint32_t c_off = get_common_arg_val<uint32_t>(11);
#endif
    const auto coss = TensorAccessor(cos_args, cos_addr, TB);
    const auto sins = TensorAccessor(sin_args, sin_addr, TB);
    cb_reserve_back(cb_c, 1 + Dt);
    for (uint32_t t = 0; t < 1 + Dt; ++t) {
        noc_async_read(cs.get_noc_addr(c_off + t), get_write_ptr(cb_c) + t * CT, CT);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, 1 + Dt);
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t r = it / H, h = it % H;
        cb_reserve_back(cb_x, Dt);
        const uint32_t lx = get_write_ptr(cb_x);
        for (uint32_t j = 0; j < Dt; ++j) {
            noc_async_read(xs.get_noc_addr((h * Mt + r) * Dt + j), lx + j * TB, TB);
        }
        cb_reserve_back(cb_cs, 4);
        const uint32_t lc = get_write_ptr(cb_cs);
        noc_async_read(coss.get_noc_addr(r * Rt + 0), lc, TB);
        noc_async_read(coss.get_noc_addr(r * Rt + 1), lc + TB, TB);
        noc_async_read(sins.get_noc_addr(r * Rt + 0), lc + 2 * TB, TB);
        noc_async_read(sins.get_noc_addr(r * Rt + 1), lc + 3 * TB, TB);
        noc_async_read_barrier();
        cb_push_back(cb_x, Dt);
        cb_push_back(cb_cs, 4);
#ifdef KV_CACHE
        // head h's V tiles (head-major [1, H, 1, D]): the compute packs them bf8 for the cache write
        cb_reserve_back(cb_v, Dt);
        for (uint32_t j = 0; j < Dt; ++j) {
            noc_async_read(vs.get_noc_addr(h * Dt + j), get_write_ptr(cb_v) + j * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Dt);
#endif
    }
}
