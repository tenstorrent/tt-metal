// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads tiles in transposed layout order for 2D matrix transpose.
// Input: [Mt, Nt] tiles. Output will be [Nt, Mt] tiles.
// For output tile (out_r, out_c), reads input tile (out_c, out_r).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = 0;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);

    // Read in transposed order: output is [Nt, Mt]
    for (uint32_t out_r = 0; out_r < Nt; out_r++) {
        for (uint32_t out_c = 0; out_c < Mt; out_c++) {
            uint32_t src_tile_id = out_c * Nt + out_r;
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(src_tile_id, s, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
