// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 LM head + greedy, reader (NCRISC) (fused_decode/lmhead.py: LmHeadGreedyB1).
//
// x: row 0 of the final normed hidden ([1, 1, 1, hidden] BF16 TILE, L1) as kt 1x32 activation tiles (cb_x, once).
// Weights: this core's `cols` vocab column tiles [j0, j0 + cols) of its DRAM bank (moe_decode down_layout: column j at
// byte j * kt * tile_bytes of the bank's shard), streamed one column (kt tiles) per cb_w page group.
//
// runtime args: [x_addr, w_addr, bank, j0, cols]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t cols = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);

    cb_reserve_back(cb_x, kt);
    uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kt; ++t) {
        noc_async_read(s_x.get_noc_addr(t, 0), l1, 32);
        noc_async_read(s_x.get_noc_addr(t, 512), l1 + 32, 32);
        l1 += 64;
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    constexpr uint32_t col_bytes = kt * tile_bytes;
    for (uint32_t j = 0; j < cols; ++j) {
        cb_reserve_back(cb_w, kt);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank, w_addr + (j0 + j) * col_bytes), get_write_ptr(cb_w), col_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_w, kt);
    }
}
