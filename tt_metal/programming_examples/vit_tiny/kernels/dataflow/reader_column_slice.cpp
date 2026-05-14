// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads a column slice from a 2D tiled matrix.
// Input: [Mt, total_Wt] tiles. Reads columns [start_col .. start_col + slice_Wt - 1].
// Pushes tiles directly to cb_out (c_16) for writer to consume.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t total_Wt = get_arg_val<uint32_t>(2);
    uint32_t start_col = get_arg_val<uint32_t>(3);
    uint32_t slice_Wt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = 16;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);

    for (uint32_t row = 0; row < Mt; row++) {
        for (uint32_t col = 0; col < slice_Wt; col++) {
            uint32_t tile_id = row * total_Wt + start_col + col;
            cb_reserve_back(cb_out, 1);
            uint32_t l1_addr = get_write_ptr(cb_out);
            noc_async_read_tile(tile_id, s, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_out, 1);
        }
    }
}
