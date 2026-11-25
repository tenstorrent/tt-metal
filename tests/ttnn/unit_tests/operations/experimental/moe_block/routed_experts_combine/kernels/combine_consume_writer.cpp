// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combine consume microbench, writer (NCRISC): each token's bf16 sum (row_tiles pages of 2048 B) to its output row.
//
// Compile-time args: row_tiles, tokens, then the output's TensorAccessorArgs. Runtime args: the output's address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tokens = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t row_bytes = row_tiles * 2048;
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0), row_bytes);
    for (uint32_t t = 0; t < tokens; ++t) {
        cb_wait_front(cb_out, row_tiles);
        noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(t), row_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, row_tiles);
    }
}
