// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// post_attention, one core per local head: the head's gate tiles (the second 256 columns of its 512 in the qg
// projection) and the head's attention rows placed into an otherwise zero tile row (the chain's slice + tilize with
// zero padding).  CBs: 0 gate (bf16, 8), 1 attention (bf16, 8), 3 row scratch (bf16, 1).
// Compile-time args: TensorAccessorArgs attention (ROW_MAJOR [1, 32, rows, 256]), qg.
// Runtime args: 0 attention, 1 qg addresses, 2 rows, 3 head, 4 first tile of the qg projection in qg (0 for the
// separate qg shard; the qg window's first tile in the merged projection shard).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t att_addr = get_arg_val<uint32_t>(0);
    const uint32_t qg_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t qg_first = get_arg_val<uint32_t>(4);
    constexpr uint32_t CB_GATE = 0, CB_ATT = 1, CB_ROW = 3, HEAD_TILES = 8, GATE_FIRST = 8;
    constexpr auto att_args = TensorAccessorArgs<0>();
    constexpr auto qg_args = TensorAccessorArgs<att_args.next_compile_time_args_offset()>();
    const auto att = TensorAccessor(att_args, att_addr);
    const auto qg = TensorAccessor(qg_args, qg_addr);
    using tile_rows::ROW_BYTES;
    using tile_rows::TILE_BYTES;

    cb_reserve_back(CB_GATE, HEAD_TILES);
    cb_reserve_back(CB_ATT, HEAD_TILES);
    const uint32_t gate_l1 = get_write_ptr(CB_GATE), att_l1 = get_write_ptr(CB_ATT);
    {
        FUSED_ZONE("fz_qs_pa_r_zero");
        Noc noc;
        DataflowBuffer att_cb(CB_ATT);
        noc.async_write_zeros(att_cb, HEAD_TILES * TILE_BYTES);
        noc.write_zeros_l1_barrier();
    }
    {
        FUSED_ZONE("fz_qs_pa_r_rows");
        for (uint32_t t = 0; t < HEAD_TILES; ++t) {
            noc_async_read_page(qg_first + 2 * HEAD_TILES * head + GATE_FIRST + t, qg, gate_l1 + t * TILE_BYTES);
        }
        // DRAM reads land whole (64-byte aligned) rows in the scratch; the RISC places the chunks into the tile faces
        cb_reserve_back(CB_ROW, 1);
        const uint32_t row_l1 = get_write_ptr(CB_ROW);
        for (uint32_t lane = 0; lane < rows; ++lane) {
            noc_async_read(att.get_noc_addr(head * rows + lane, 0), row_l1, HEAD_TILES * 2 * ROW_BYTES);
            noc_async_read_barrier();
            invalidate_l1_cache();
            tile_rows::place_row(row_l1, att_l1, HEAD_TILES, lane);
        }
        noc_async_read_barrier();
        cb_push_back(CB_GATE, HEAD_TILES);
        cb_push_back(CB_ATT, HEAD_TILES);
    }
}
