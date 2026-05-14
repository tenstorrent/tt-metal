// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicore softmax reader: fills scaler CB with 1.0, then reads assigned rows.
// Runtime args: src_addr, start_row, num_rows, Wt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_row = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);

    // Fill scaler CB with 1.0 (bfloat16 = 0x3F80)
    cb_reserve_back(cb_scaler, 1);
    auto* scaler_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < 1024; i++) {
        scaler_ptr[i] = 0x3F80;
    }
    cb_push_back(cb_scaler, 1);

    // Read input tiles for assigned rows
    for (uint32_t mt = 0; mt < num_rows; mt++) {
        uint32_t row = start_row + mt;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            uint32_t tile_id = row * Wt + wt;
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(tile_id, s, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
