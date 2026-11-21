// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode RMSNorm + rotation, writer (BRISC): the rotated block (row-major 2048 B) into each of the C copies of x' at
// 2048 b, and the normalised block back into row 0 of activation tiles 32 b .. 32 b + 31 (32 B per face row: the
// normalised activation lives in L1, where 32-B writes are fine).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"
#ifndef NR_CB_X  // CB indices (the fused form inside a ternary decode linear moves them off the linear's)
#define NR_CB_X 0
#define NR_CB_C 2
#define NR_CB_SC 3
#define NR_CB_Y 16
#endif
#ifndef NR_FIRST  // the fused form's cores start after the linear's
#define NR_FIRST 0
#endif

#ifndef NR_BPC
#define NR_BPC 1
#endif

void kernel_main() {
#ifdef NR_TO_HUB
    // fused into a ternary decode linear (decode_linear3.py rotated()): each own x' block b into the hub's (core 0's)
    // buffer at 2048 b and its counter bumped (as rot_writer.cpp ROT_TO_HUB); NR_XNT: also the tiled normalised block
    // into tile b of xnt. common args: hub x, hub y, buffer, counter, grid_x, B, xnt
    const uint32_t hx = get_common_arg_val<uint32_t>(0), hy = get_common_arg_val<uint32_t>(1);
    const uint32_t hbuf = get_common_arg_val<uint32_t>(2), hcnt = get_common_arg_val<uint32_t>(3);
    const uint32_t c = core_index(get_common_arg_val<uint32_t>(4)) - NR_FIRST;
    const uint32_t B = get_common_arg_val<uint32_t>(5);
    const uint32_t b0 = c * NR_BPC, nb = (B - b0 < NR_BPC) ? B - b0 : NR_BPC;
#ifdef NR_XNT
    constexpr auto t_args = TensorAccessorArgs<0>();
    const auto xnt = TensorAccessor(t_args, get_common_arg_val<uint32_t>(6), 2048);
#endif
    constexpr uint32_t cb_yh = NR_CB_Y;
    for (uint32_t i = 0; i < nb; ++i) {
#ifdef NR_XNT
        cb_wait_front(20, 1);
        noc_async_write(get_read_ptr(20), xnt.get_noc_addr(b0 + i), 2048);
#endif
        cb_wait_front(cb_yh, 1);
        noc_async_write(get_read_ptr(cb_yh), get_noc_addr(hx, hy, hbuf + (b0 + i) * 2048), 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hx, hy, hcnt), 1);
        cb_pop_front(cb_yh, 1);
#ifdef NR_XNT
        cb_pop_front(20, 1);
#endif
    }
    noc_async_atomic_barrier();
#else
    // common args: out (x' copies), xn, grid_x, xnt (the normalised blocks tiled, B tiles); one block per core
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(2)) - NR_FIRST;
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr auto n_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    const auto xnt = TensorAccessor(t_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), ROW_BYTES);
    const auto xn = TensorAccessor(n_args, get_common_arg_val<uint32_t>(1), 2048);
    constexpr uint32_t cb_y = NR_CB_Y, cb_xn = 17;
    cb_wait_front(cb_xn, 1);
    const uint32_t n = get_read_ptr(cb_xn);
    for (uint32_t jh = 0; jh < 32; ++jh) {
        const uint64_t t = xn.get_noc_addr(32 * b + jh);
        noc_async_write(n + jh * 64, t, 32);
        noc_async_write(n + jh * 64 + 32, t + 512, 32);
    }
    cb_wait_front(cb_y, 1);
    const uint32_t l1 = get_read_ptr(cb_y);
#ifdef X_PAGED
    // x' as a [B, 1024] L1 tensor (2-KB pages over the cores' L1): block b is page b, written once
    noc_async_write(l1, out.get_noc_addr(b), 2048);
#else
    for (uint32_t c = 0; c < C; ++c) {
        noc_async_write(l1, out.get_noc_addr(c, b * 2048), 2048);
    }
#endif
    cb_wait_front(20, 1);  // the tiled normalised block -> tile b of xnt
    noc_async_write(get_read_ptr(20), xnt.get_noc_addr(b), 2048);
    noc_async_write_barrier();
    cb_pop_front(20, 1);
    cb_pop_front(cb_y, 1);
    cb_pop_front(cb_xn, 1);
#endif
}
