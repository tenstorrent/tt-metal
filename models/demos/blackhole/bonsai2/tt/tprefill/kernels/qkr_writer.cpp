// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill per-head RMSNorm + partial RoPE (qk_rope.py), writer (BRISC): per item the compute's tiles 2 .. Dt - 1, then
// 0 and 1, to the head-major output.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, items, ncores, grid_x, H, Mt, Dt
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first, n_items;
    core_span(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        1,
        first,
        n_items);
    const uint32_t H = get_common_arg_val<uint32_t>(4), Mt = get_common_arg_val<uint32_t>(5);
    const uint32_t Dt = get_common_arg_val<uint32_t>(6);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16;
    const uint32_t TB = get_tile_size(cb_out);
    const auto os = TensorAccessor(o_args, o_addr, TB);
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t r = it / H, h = it % H, base = (h * Mt + r) * Dt;
        for (uint32_t k = 0; k < Dt; ++k) {
            const uint32_t j = k < Dt - 2 ? k + 2 : k - (Dt - 2);
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(base + j), TB);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
