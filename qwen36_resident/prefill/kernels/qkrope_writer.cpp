// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// q / k head norm + partial RoPE, writer (BRISC): each group's 8 output tiles back over its input tiles
// (in place; every group reads its tiles before any are written). Also writes the reduce scaler.
// Runtime args: 0 first group, 1 groups, 2 t address.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scaler = 4, cb_out = 16, Dt = 8;
    constexpr uint32_t tile = get_tile_size(cb_out);
    const uint32_t first = get_arg_val<uint32_t>(0), groups = get_arg_val<uint32_t>(1);
    const InterleavedAddrGenFast<true> t{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = tile, .data_format = DataFormat::Float16_b};
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t w = 0; w < 512; w++) {
        sc[w] = (w % 128) < 8 ? 0x3f803f80 : 0;  // bf16 1.0 in the first row of each face
    }
    cb_push_back(cb_scaler, 1);
    for (uint32_t g = first; g < first + groups; g++) {
        cb_wait_front(cb_out, Dt);
        for (uint32_t d = 0; d < Dt; d++) {
            noc_async_write_tile(g * Dt + d, t, get_read_ptr(cb_out) + d * tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Dt);
    }
}
