// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN output gate over one prefill chunk, reader (NCRISC): y = rmsnorm_per_head(o) * silu(z), from the
// delta-rule output o (head-major [heads, C, 128] tiles, bf16 or fp32 as cb_o) to token-major rows [C, heads * 128]. A
// group is one (row tile t, head h): o tiles (h, t, 0..3) and z tiles (t, z0 + 4h + 0..3) of the projection output p
// (Pw tiles wide). A head-major tile holds the same 32 tokens x 32 dims as its token-major place, so the layout change
// is only the tile order. First the reduce scaler (1/128, row mean).
//
// Compile-time args: 0 Rt, 1 heads, 2 Pw, 3 z0 (tiles). Runtime args: 0 first group, 1 groups,
// 2 o address, 3 p address.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t heads = get_compile_time_arg_val(1);
    constexpr uint32_t Pw = get_compile_time_arg_val(2);
    constexpr uint32_t z0 = get_compile_time_arg_val(3);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t groups = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_o = 0, cb_z = 1, cb_scaler = 2;
    constexpr uint32_t o_tile = get_tile_size(cb_o), z_tile = get_tile_size(cb_z);
    const InterleavedAddrGenFast<true> o{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = o_tile, .data_format = get_dataformat(cb_o)};
    const InterleavedAddrGenFast<true> p{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = z_tile, .data_format = DataFormat::Float16_b};

    wh_generate_reduce_scaler<true>(cb_scaler, 0x3c003c00);  // bf16 1/128, twice
    for (uint32_t g = first; g < first + groups; g++) {
        const uint32_t t = g / heads, h = g % heads;
        cb_reserve_back(cb_o, 4);
        cb_reserve_back(cb_z, 4);
        for (uint32_t d = 0; d < 4; d++) {
            noc_async_read_tile((h * Rt + t) * 4 + d, o, get_write_ptr(cb_o) + d * o_tile);
            noc_async_read_tile(t * Pw + z0 + h * 4 + d, p, get_write_ptr(cb_z) + d * z_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_o, 4);
        cb_push_back(cb_z, 4);
    }
}
