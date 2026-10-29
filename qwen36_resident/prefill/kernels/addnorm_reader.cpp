// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Residual add + RMSNorm, reader (NCRISC): this core's W tiles of row tile r, columns [c0, c0 + W), of the
// residual stream x and the branch output b ([C, H] token-major tiles, Ht tiles per row).
// Compile-time args: 0 Ht, 1 W. Runtime args: 0 r, 1 c0, 2 x address, 3 b address.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_b = 1;
    constexpr uint32_t tile = get_tile_size(cb_x);
    const uint32_t r = get_arg_val<uint32_t>(0), c0 = get_arg_val<uint32_t>(1);
    const InterleavedAddrGenFast<true> x{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> b{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = tile, .data_format = DataFormat::Float16_b};
    constexpr uint32_t step = 2;  // tiles per push, matching the compute's DEST groups
    for (uint32_t i = 0; i < W; i += step) {
        cb_reserve_back(cb_x, step);
        cb_reserve_back(cb_b, step);
        for (uint32_t t = 0; t < step; t++) {
            noc_async_read_tile(r * Ht + c0 + i + t, x, get_write_ptr(cb_x) + t * tile);
            noc_async_read_tile(r * Ht + c0 + i + t, b, get_write_ptr(cb_b) + t * tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, step);
        cb_push_back(cb_b, step);
    }
}
