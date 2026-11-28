// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode rotation, writer (BRISC): the rotated block (row-major, 2048 B) into each of the C copies of x', at 2048 b.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"
#ifndef ROT_CB_X  // CB indices (the fused form inside a ternary decode linear moves them off the linear's)
#define ROT_CB_X 0
#define ROT_CB_M 1
#define ROT_CB_H 2
#define ROT_CB_SCR 3
#define ROT_CB_U 4
#define ROT_CB_Y 16
#endif
#ifndef ROT_FIRST  // the fused form's rotation cores start after the linear's
#define ROT_FIRST 0
#endif

void kernel_main() {
#ifdef ROT_TO_HUB
    // fused into a ternary decode linear (decode_linear3.py rotated()): the block goes into the hub's (the linear's
    // core 0) L1 x' buffer at 2048 b and bumps its counter; the hub multicasts the whole x' once (tdec_stream.hpp
    // XHub: one multicast instead of one per rotation core, which serialise: 17 senders ~60 us vs ~5 us)
    // common args: hub x, hub y, buffer, counter, grid_x
    const uint32_t hx = get_common_arg_val<uint32_t>(0), hy = get_common_arg_val<uint32_t>(1);
    const uint32_t buf = get_common_arg_val<uint32_t>(2), cnt = get_common_arg_val<uint32_t>(3);
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(4)) - ROT_FIRST;
    constexpr uint32_t cb_y_h = ROT_CB_Y;
    cb_wait_front(cb_y_h, 1);
    noc_async_write(get_read_ptr(cb_y_h), get_noc_addr(hx, hy, buf + b * 2048), 2048);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hx, hy, cnt), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_y_h, 1);
#else
#ifdef ROT_COMMON
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(1));
#else
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);
#endif
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    constexpr uint32_t cb_y = ROT_CB_Y;
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
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
#endif
}
