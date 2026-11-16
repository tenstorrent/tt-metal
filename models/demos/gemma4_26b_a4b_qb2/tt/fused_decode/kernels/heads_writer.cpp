// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 q / k / v head norm + RoPE, writer (BRISC) (fused_decode/heads.py: HeadsB1): packed row r of the head's
// output tile -> row `head` of column tile r of the tensor's one-core height-sharded output ([1, 1, heads, D], heads
// on the rows; the layout the cache update and SDPA take): face 0 / face 1 rows, 32 B each.
//
// runtime args: [shard_noc_x, shard_noc_y, shard_addr, head]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t nx = get_arg_val<uint32_t>(0);
    const uint32_t ny = get_arg_val<uint32_t>(1);
    const uint32_t shard = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t page = 2048, face = 512;
    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    const uint32_t row = (head >> 4) * 2 * face + (head & 15) * 32;
    for (uint32_t r = 0; r < R; ++r) {
        noc_async_write(src + 32 * r, get_noc_addr(nx, ny, shard + page * r + row), 32);
        noc_async_write(src + face + 32 * r, get_noc_addr(nx, ny, shard + page * r + row + face), 32);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
