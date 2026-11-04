// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill gates (forward.py GateTiles), writer: g_bh tiles (hr, r) for hr = 0, 1, then beta_bh tiles
// (h, r) for h = 0 .. 47.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t g_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t rows = get_common_arg_val<uint32_t>(2);
    const uint32_t me = get_arg_val<uint32_t>(0);
    constexpr auto g_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_g = 16, cb_beta = 17;
    if (me >= rows) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_g);
    const auto gs = TensorAccessor(g_args, g_addr, TB);
    const auto bs = TensorAccessor(b_args, b_addr, TB);
    auto drain = [&](uint32_t cb, auto& dst, uint32_t page) {
        cb_wait_front(cb, 1);
        noc_async_write(get_read_ptr(cb), dst.get_noc_addr(page), TB);
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    };
    for (uint32_t hr = 0; hr < 2; ++hr) {
        drain(cb_g, gs, hr * rows + me);
    }
    for (uint32_t h = 0; h < 48; ++h) {
        drain(cb_beta, bs, h * rows + me);
    }
    noc_async_write_barrier();
}
