// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv, writer (BRISC): per channel column, the Tt output tiles in time order, then the
// new conv state tile (in place: the reader read the old one first, on this same core).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, state, Dt, ncores, grid_x, Tt
    const uint32_t Dt = get_common_arg_val<uint32_t>(2);
    uint32_t first, n;
    core_span(Dt, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), 1, first, n);
    const uint32_t Tt = get_common_arg_val<uint32_t>(5);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 2048);
    constexpr uint32_t cb_out = 16, cb_ns = 17;
    for (uint32_t ct = first; ct < first + n; ++ct) {
        for (uint32_t tt = 0; tt < Tt; ++tt) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(tt * Dt + ct), 2048);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
        cb_wait_front(cb_ns, 1);
        noc_async_write(get_read_ptr(cb_ns), ss.get_noc_addr(ct), 2048);
        noc_async_write_barrier();
        cb_pop_front(cb_ns, 1);
    }
    noc_async_write_barrier();
}
