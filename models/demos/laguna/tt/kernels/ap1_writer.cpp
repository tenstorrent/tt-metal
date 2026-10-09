// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode attention prologue, writer (Laguna): the 4 output tiles (q, or k) to their tensor; role 1 also
// writes the 4 v tiles the reader gathered.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr auto v_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    cb_wait_front(16, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_write(get_read_ptr(16) + j * 2048, o.get_noc_addr(j), 2048);
    }
    if constexpr (role == 1) {
        const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(1), 2048);
        cb_wait_front(5, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write(get_read_ptr(5) + j * 2048, v.get_noc_addr(j), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(5, 4);
    }
    noc_async_write_barrier();
    cb_pop_front(16, 4);
}
