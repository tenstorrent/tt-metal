// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per output tile (branch b, column j): the block tile j, the residual tile hidden_tiles * b + j and the injection
// tile with its column b copied into column 0 (rows 0..31), which the compute column-broadcasts.
// Named compile-time args: cb_block, cb_res, cb_coef, hidden_tiles, branches.  Compile-time args: TensorAccessorArgs
// for block, residual, injection, chained from 0.
// Runtime args: 0 block addr, 1 residual addr, 2 injection addr, 3 first unit, 4 units; unit u = (column u / branches,
// branch u % branches).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t block_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t injection_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_block = get_named_compile_time_arg_val("cb_block");
    constexpr uint32_t cb_res = get_named_compile_time_arg_val("cb_res");
    constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t branches = get_named_compile_time_arg_val("branches");
    constexpr auto block_args = TensorAccessorArgs<0>();
    constexpr auto residual_args = TensorAccessorArgs<block_args.next_compile_time_args_offset()>();
    constexpr auto injection_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();

    const auto block = TensorAccessor(block_args, block_addr);
    const auto residual = TensorAccessor(residual_args, residual_addr);
    const auto injection = TensorAccessor(injection_args, injection_addr);
    const uint32_t tile_bytes = get_tile_size(cb_block);
    Noc noc;
    DataflowBuffer blk(cb_block);
    DataflowBuffer res(cb_res);
    DataflowBuffer coef(cb_coef);

    for (uint32_t u = first; u < first + units; ++u) {
        FUSED_ZONE("fz_gw_r_unit");
        const uint32_t b = u % branches;
        const uint32_t j = u / branches;
        blk.reserve_back(1);
        res.reserve_back(1);
        coef.reserve_back(1);
        noc.async_read(block, blk, tile_bytes, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read(residual, res, tile_bytes, {.page_id = hidden_tiles * b + j}, {.offset_bytes = 0});
        noc.async_read(injection, coef, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // bf16 tile element (r, c) is 16-bit word (r >> 4) * 512 + (c >> 4) * 256 + (r & 15) * 16 + (c & 15)
        volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(coef.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row0 = (r >> 4) * 512 + (r & 15) * 16;
            tile[row0] = tile[row0 + b];
        }
        blk.push_back(1);
        res.push_back(1);
        coef.push_back(1);
    }
}
