// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN gates, writer (BRISC): g and beta tiles (r, 0..1) of the [C, 48] fp32 outputs.
// Runtime args: 0 first row tile, 1 row tiles, 2 g, 3 beta addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_g = 16, cb_beta = 17;
    constexpr uint32_t tile = get_tile_size(cb_g);
    const InterleavedAddrGenFast<true> g{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> beta{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = tile, .data_format = DataFormat::Float32};
    for (uint32_t r = first; r < first + rows; r++) {
        for (uint32_t c = 0; c < 2; c++) {
            cb_wait_front(cb_g, 1);
            noc_async_write_tile(r * 2 + c, g, get_read_ptr(cb_g));
            noc_async_write_barrier();
            cb_pop_front(cb_g, 1);
        }
        for (uint32_t c = 0; c < 2; c++) {
            cb_wait_front(cb_beta, 1);
            noc_async_write_tile(r * 2 + c, beta, get_read_ptr(cb_beta));
            noc_async_write_barrier();
            cb_pop_front(cb_beta, 1);
        }
    }
}
