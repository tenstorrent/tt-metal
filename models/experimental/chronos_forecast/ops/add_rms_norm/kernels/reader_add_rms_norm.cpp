// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for [y = a + b,] n = rms_norm(y): each tile row of a (and b) in blocks of blk tiles, plus the
// reduce scaler tile (1.0 in row 0 of each face; the 1/W scale is applied in DST).
//
// Compile-time args: Wt, blk, fuse_add, TensorAccessorArgs(a), TensorAccessorArgs(b)
// Runtime args: a_addr, b_addr, num_rows, start_row

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr bool fuse_add = get_compile_time_arg_val(2) != 0;
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t a_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t b_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t scaler_cb_id = tt::CBIndex::c_2;

    Noc noc;
    CircularBuffer cb_a(a_cb_id);
    CircularBuffer cb_b(b_cb_id);
    CircularBuffer cb_scaler(scaler_cb_id);
    const uint32_t a_tile_bytes = get_tile_size(a_cb_id);
    const uint32_t b_tile_bytes = fuse_add ? get_tile_size(b_cb_id) : 0;
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);

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

    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_a.reserve_back(blk);
            if constexpr (fuse_add) {
                cb_b.reserve_back(blk);
            }
            for (uint32_t d = 0; d < blk; ++d) {
                const uint32_t page = row * Wt + j0 + d;
                noc.async_read(a, cb_a, a_tile_bytes, {.page_id = page}, {.offset_bytes = d * a_tile_bytes});
                if constexpr (fuse_add) {
                    noc.async_read(b, cb_b, b_tile_bytes, {.page_id = page}, {.offset_bytes = d * b_tile_bytes});
                }
            }
            noc.async_read_barrier();
            cb_a.push_back(blk);
            if constexpr (fuse_add) {
                cb_b.push_back(blk);
            }
        }
    }
}
