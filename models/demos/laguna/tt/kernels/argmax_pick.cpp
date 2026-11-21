// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 3 (one core): the D chips' candidates gathered side by side (g [R, 8 D] uint32: page = row, chip
// d's [key, global column] at words 8 d, 8 d + 1) -> per row the column of the largest key (equal keys: the lower
// chip, i.e. the lower column) -> tok [1, R_out] uint32.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t D = get_compile_time_arg_val(1);
    constexpr uint32_t R_out = get_compile_time_arg_val(2);  // rows written (the token buffer's width)
    constexpr auto g_args = TensorAccessorArgs<3>();
    constexpr auto t_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0), D * 32);
    const auto t = TensorAccessor(t_args, get_common_arg_val<uint32_t>(1), R_out * 4);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t row = 0; row < R_out; ++row) {
        noc_async_read(g.get_noc_addr(row), buf + row * D * 32, D * 32);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + R * D * 32);
    for (uint32_t row = 0; row < R_out; ++row) {
        const uint32_t base = row * D * 8;
        uint32_t bk = c[base], bi = c[base + 1];
        for (uint32_t d = 1; d < D; ++d) {
            const uint32_t k = c[base + d * 8];
            if (k > bk) {
                bk = k;
                bi = c[base + d * 8 + 1];
            }
        }
        out[row] = bi;
    }
    noc_async_write(buf + R * D * 32, t.get_noc_addr(0), R_out * 4);
    noc_async_write_barrier();
}
