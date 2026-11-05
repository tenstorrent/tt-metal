// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv (conv2_compute.cpp), writer (BRISC): per segment of (column, time tiles) the
// output tiles in time order; a segment starting at the column's first tile also writes the column's new conv state
// right after its first output tile (in place: this core's reader read the old state first). Output tile columns
// ct < W0 go to out, W0 <= ct < W0 + W1 to out_b, the rest to out_c (each [1, T, its columns]; unsplit: W0 = Dt).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, state, Dt, ncores, grid_x, Tt, out_b, out_c, W0, W1
    const uint32_t Dt = get_common_arg_val<uint32_t>(2);
    const uint32_t Tt = get_common_arg_val<uint32_t>(5);
    uint32_t first, n;
    core_span(Dt * Tt, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), 1, first, n);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(6), 2048);
    const auto oc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(7), 2048);
    const uint32_t W0 = get_common_arg_val<uint32_t>(8), W1 = get_common_arg_val<uint32_t>(9);
    const uint32_t W2 = Dt - W0 - W1;
    auto dest = [&](uint32_t tt, uint32_t ct) -> uint64_t {
        if (ct < W0) {
            return os.get_noc_addr(tt * W0 + ct);
        }
        if (ct < W0 + W1) {
            return ob.get_noc_addr(tt * W1 + ct - W0);
        }
        return oc.get_noc_addr(tt * W2 + ct - W0 - W1);
    };
    constexpr uint32_t cb_out = 16, cb_ns = 17;
    for (uint32_t u = first; u < first + n; ++u) {
        const uint32_t ct = u / Tt, tt = u % Tt;
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), dest(tt, ct), 2048);
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
