// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill output, writer (BRISC): item (h, tt)'s 4 bf16 tiles to (tt, 4h .. 4h+3) of [1, T, H * 128].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, Nitems, ncores, grid_x, Tt, Wt
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), get_common_arg_val<uint32_t>(3), 1, first, n);
    const uint32_t Tt = get_common_arg_val<uint32_t>(4), Wt = get_common_arg_val<uint32_t>(5);
    constexpr auto a = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(a, get_common_arg_val<uint32_t>(0), 2048);
    constexpr uint32_t cb_out = 16;
    for (uint32_t it = first; it < first + n; ++it) {
        const uint32_t h = it / Tt, tt = it % Tt;
        cb_wait_front(cb_out, 4);
        for (uint32_t d = 0; d < 4; ++d) {
            noc_async_write(get_read_ptr(cb_out) + d * 2048, out.get_noc_addr(tt * Wt + h * 4 + d), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 4);
    }
}
