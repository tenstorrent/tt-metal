// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention prologue, writer (Laguna): per token row b (rpc per core), its 4 output tiles (q, or k) to tiles
// 4 (b % 32) .. + 3 of its 32-row chunk's tensor (more than 32 rows: one q / k / v per chunk); role 1 also writes the
// 4 v tiles the reader gathered.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t hp = get_compile_time_arg_val(1);           // q heads per part (even), role 0
    constexpr uint32_t total_heads = get_compile_time_arg_val(2);  // q heads in all
    constexpr uint32_t q_parts = get_compile_time_arg_val(3);      // 1: whole tiles
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t base = get_compile_time_arg_val(5);  // first row-major core index of this role
    constexpr uint32_t rpc = get_compile_time_arg_val(6);   // rows per core (consecutive)
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto v_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t L = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - base;
    const uint32_t part = L % q_parts;
    for (uint32_t i = 0; i < rpc; ++i) {
        // row -> its 32-row chunk's tensors (common args: chunk c's output, then its v), local row row % 32
        const uint32_t row = (L / q_parts) * rpc + i, c = row / 32;
        const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2 * c), 2048);
        const uint32_t t0 = (row % 32) * 4;
        cb_wait_front(16, 4);
        if (role == 0 && q_parts > 1) {
            // this part's head rows only (other parts write theirs): 64-byte pieces of 2 rows (hp and h0 even)
            const uint32_t h0 = part * hp;
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
            const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2 * c + 1), 2048);
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
}
