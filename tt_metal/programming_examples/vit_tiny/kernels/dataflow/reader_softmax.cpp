// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads input tiles row by row (Wt tiles per row, Mt rows).
// Also fills a scaler CB with a tile of 1.0 values for reduce operations.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);

    // Fill scaler CB with a tile of 1.0 values
    // The reduce operation uses this as a scaling factor
    cb_reserve_back(cb_scaler, 1);
    uint32_t scaler_addr = get_write_ptr(cb_scaler);
    // Fill the first row of each face with 1.0 (bfloat16 = 0x3F80)
    // This is what the reduce API expects for SUM/MAX scaling
    volatile tt_l1_ptr uint16_t* scaler_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scaler_addr);
    for (uint32_t i = 0; i < 32 * 32; i++) {
        scaler_ptr[i] = 0x3F80;  // 1.0 in bfloat16
    }
    cb_push_back(cb_scaler, 1);

    // Read input tiles row by row
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            uint32_t tile_idx = mt * Wt + wt;
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, s, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
