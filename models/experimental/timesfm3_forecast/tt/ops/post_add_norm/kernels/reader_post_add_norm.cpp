// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads the scaler and the gain row once, then per tile row: the whole o row, then the residual row, in blocks.
// Compile-time args: Wt, blk, accessor args of o, residual, gain. Runtime: o, res, gain addresses, rows, start.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

    Noc noc;
    CircularBuffer cb_o(tt::CBIndex::c_0);
    CircularBuffer cb_r(tt::CBIndex::c_1);
    CircularBuffer cb_scaler(tt::CBIndex::c_2);
    CircularBuffer cb_g(tt::CBIndex::c_3);
    const uint32_t o_bytes = get_tile_size(tt::CBIndex::c_0);
    const uint32_t r_bytes = get_tile_size(tt::CBIndex::c_1);
    const uint32_t g_bytes = get_tile_size(tt::CBIndex::c_3);
    const auto o = TensorAccessor(o_args, o_addr);
    const auto res = TensorAccessor(r_args, r_addr);
    const auto g = TensorAccessor(g_args, g_addr);

    cb_scaler.reserve_back(1);
    volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scaler.get_write_ptr());
    for (uint32_t i = 0; i < 512; ++i) {
        scaler[i] = 0;
    }
    constexpr uint32_t one_one_bf16 = 0x3F803F80;
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t i = 0; i < 8; ++i) {
            scaler[face * 128 + i] = one_one_bf16;
        }
    }
    cb_scaler.push_back(1);

    cb_g.reserve_back(Wt);
    for (uint32_t j = 0; j < Wt; ++j) {
        noc.async_read(g, cb_g, g_bytes, {.page_id = j}, {.offset_bytes = j * g_bytes});
    }
    noc.async_read_barrier();
    cb_g.push_back(Wt);

    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_o.reserve_back(blk);
            for (uint32_t d = 0; d < blk; ++d) {
                noc.async_read(o, cb_o, o_bytes, {.page_id = row * Wt + j0 + d}, {.offset_bytes = d * o_bytes});
            }
            noc.async_read_barrier();
            cb_o.push_back(blk);
        }
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_r.reserve_back(blk);
            for (uint32_t d = 0; d < blk; ++d) {
                noc.async_read(res, cb_r, r_bytes, {.page_id = row * Wt + j0 + d}, {.offset_bytes = d * r_bytes});
            }
            noc.async_read_barrier();
            cb_r.push_back(blk);
        }
    }
}
