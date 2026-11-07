// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dot-product finish, writer (one core): writes the result tile. keep_old: first copies the output's current tile
// to `old` (r.z of the previous iteration, which the p update divides by). A runtime argument, not a define, so the
// three finishes share one binary.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0), old_addr = get_arg_val<uint32_t>(1);
    const uint32_t keep_old = get_arg_val<uint32_t>(2);
    constexpr auto ao = TensorAccessorArgs<0>();
    constexpr auto aold = TensorAccessorArgs<ao.next_compile_time_args_offset()>();
    const auto OUT = TensorAccessor(ao, out_addr, 4096);
    const auto OLD = TensorAccessor(aold, old_addr, 4096);
    if (keep_old) {
        cb_reserve_back(CB_T, 1);
        const uint32_t t = get_write_ptr(CB_T);
        noc_async_read(OUT.get_noc_addr(0), t, 4096);
        noc_async_read_barrier();
        noc_async_write(t, OLD.get_noc_addr(0), 4096);
        noc_async_write_barrier();
    }
    cb_wait_front(CB_OUT, 1);
    noc_async_write(get_read_ptr(CB_OUT), OUT.get_noc_addr(0), 4096);
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 1);
}
