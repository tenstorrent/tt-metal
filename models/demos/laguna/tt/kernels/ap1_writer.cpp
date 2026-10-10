// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention prologue, writer (Laguna): row b's (= the core's logical x) 4 output tiles (q, or k) to tiles
// 4b .. 4b + 3 of their tensor; role 1 also writes the 4 v tiles the reader gathered.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t hp = get_compile_time_arg_val(1);           // q heads per part (even), role 0
    constexpr uint32_t total_heads = get_compile_time_arg_val(2);  // q heads in all
    constexpr uint32_t q_parts = get_compile_time_arg_val(3);      // 1: whole tiles
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr auto v_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t t0 = get_absolute_logical_x() * 4;
    cb_wait_front(16, 4);
    if (role == 0 && q_parts > 1) {
        // this part's head rows only (other parts write theirs): 64-byte pieces of 2 rows (hp and h0 even)
        const uint32_t h0 = get_absolute_logical_y() * hp;
        for (uint32_t h = h0; h < h0 + hp && h < total_heads; h += 2) {
            const uint32_t off = (h < 16 ? 0 : 1024) + (h % 16) * 32;
            const uint32_t len = (h + 1 < total_heads) ? 64 : 32;
            for (uint32_t j = 0; j < 4; ++j) {
                noc_async_write(get_read_ptr(16) + j * 2048 + off, o.get_noc_addr(t0 + j) + off, len);
                noc_async_write(get_read_ptr(16) + j * 2048 + off + 512, o.get_noc_addr(t0 + j) + off + 512, len);
            }
        }
    } else {
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write(get_read_ptr(16) + j * 2048, o.get_noc_addr(t0 + j), 2048);
        }
    }
    if constexpr (role == 1) {
        const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(1), 2048);
        cb_wait_front(5, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write(get_read_ptr(5) + j * 2048, v.get_noc_addr(t0 + j), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(5, 4);
    }
    noc_async_write_barrier();
    cb_pop_front(16, 4);
}
