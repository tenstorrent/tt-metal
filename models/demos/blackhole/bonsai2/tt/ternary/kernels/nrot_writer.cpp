// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode RMSNorm + rotation, writer (BRISC): the rotated block (row-major 2048 B) into each of the C copies of x' at
// 2048 b, and the normalised block back into row 0 of activation tiles 32 b .. 32 b + 31 (32 B per face row: the
// normalised activation lives in L1, where 32-B writes are fine).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"

void kernel_main() {
    // common args: out (x' copies), xn, grid_x
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(2));
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr auto n_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), ROW_BYTES);
    const auto xn = TensorAccessor(n_args, get_common_arg_val<uint32_t>(1), 2048);
    constexpr uint32_t cb_y = 16, cb_xn = 17;
    cb_wait_front(cb_xn, 1);
    const uint32_t n = get_read_ptr(cb_xn);
    for (uint32_t jh = 0; jh < 32; ++jh) {
        const uint64_t t = xn.get_noc_addr(32 * b + jh);
        noc_async_write(n + jh * 64, t, 32);
        noc_async_write(n + jh * 64 + 32, t + 512, 32);
    }
    cb_wait_front(cb_y, 1);
    const uint32_t l1 = get_read_ptr(cb_y);
    for (uint32_t c = 0; c < C; ++c) {
        noc_async_write(l1, out.get_noc_addr(c, b * 2048), 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
    cb_pop_front(cb_xn, 1);
}
