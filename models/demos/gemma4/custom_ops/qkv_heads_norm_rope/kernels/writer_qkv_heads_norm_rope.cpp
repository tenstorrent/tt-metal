// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for QkvHeadsNormRope: per tensor of this core's work, the cols_t output tiles (the normalized, rotated
// head rows of user u) go to user u's shard of that tensor's output (height-sharded: the shard sits at the same L1
// address on every shard core; this core is usually not the shard core, so the tiles travel over the NoC).
// compile-time args: [cb_out, cols_t]
// runtime args: [shard_noc_x, shard_noc_y, t0, t_count, out_addr_q, out_addr_k, out_addr_v]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t nx = get_arg_val<uint32_t>(0);
    const uint32_t ny = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t_count = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cols_t = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = 2048;
    CircularBuffer co(cb_out);
    for (uint32_t t = t0; t < t0 + t_count; ++t) {
        const uint32_t out_addr = get_arg_val<uint32_t>(4 + t);
        co.wait_front(cols_t);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < cols_t; ++c) {
            noc_async_write(src + c * tile_bytes, get_noc_addr(nx, ny, out_addr + c * tile_bytes), tile_bytes);
        }
        noc_async_write_barrier();
        co.pop_front(cols_t);
    }
}
