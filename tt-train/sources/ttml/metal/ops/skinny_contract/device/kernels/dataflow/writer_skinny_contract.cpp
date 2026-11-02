// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_contract writer: per chunk of the core's column range `cols` output tiles (a shorter chunk: ncols), pages
// c + j (tile (0, n) of [r, N], or (n, 0) of [N, r]).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 2;
constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr uint32_t cols = get_compile_time_arg_val(2);
constexpr auto out_args = TensorAccessorArgs<3>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t n_count = get_arg_val<uint32_t>(arg++);
    const auto out_acc = TensorAccessor(out_args, out_addr, get_tile_size(cb_out));
    const uint32_t tile_bytes = get_tile_size(cb_out);
    for (uint32_t c = n_begin; c < n_begin + n_count; c += cols) {
        const uint32_t ncols = (c + cols <= n_begin + n_count) ? cols : (n_begin + n_count - c);
        cb_wait_front(cb_out, cols);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < ncols; ++j) {
            noc_async_write_page(c + j, out_acc, l1);
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, cols);
    }
}
