// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv (conv2_compute.cpp), writer (BRISC): per segment of (column, time tiles) the
// output tiles in time order; a segment starting at the column's first tile also writes the column's new conv state
// right after its first output tile (in place: this core's reader read the old state first).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, state, Dt, ncores, grid_x, Tt
    const uint32_t Dt = get_common_arg_val<uint32_t>(2);
    const uint32_t Tt = get_common_arg_val<uint32_t>(5);
    uint32_t first, n;
    core_span(Dt * Tt, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), 1, first, n);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 2048);
    constexpr uint32_t cb_out = 16, cb_ns = 17;
    for (uint32_t u = first; u < first + n; ++u) {
        const uint32_t ct = u / Tt, tt = u % Tt;
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(tt * Dt + ct), 2048);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
        if (tt == 0) {
            cb_wait_front(cb_ns, 1);
            noc_async_write(get_read_ptr(cb_ns), ss.get_noc_addr(ct), 2048);
            noc_async_writes_flushed();
            cb_pop_front(cb_ns, 1);
        }
    }
    noc_async_write_barrier();
}
