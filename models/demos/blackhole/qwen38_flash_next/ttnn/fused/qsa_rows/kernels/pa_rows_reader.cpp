// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 (qsa_rows_post_attention) reader, one core per (local head h, tile column t): the head's gate
// tile t (the second 256 columns of its 512 in the qg projection: tile qg_first + 16 h + 8 + t) and the head's
// attention rows' 32-column window t placed into one tile (the chain's slice to the local heads + tilize), row by
// row from the ROW_MAJOR [1, 32 heads, rows, 256] sparse output.  CBs: 0 gate (1 tile), 1 attention (1 tile),
// 3 row scratch (64 B per lane, one tile of room).
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/tile_rows.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_GATE = 0, CB_ATT = 1, CB_ROW = 3;
constexpr uint32_t HEAD_TILES = 8, GATE_FIRST = 8, WINDOW_BYTES = 64;  // 32 bf16 of a 256-column head row
}  // namespace

void kernel_main() {
    constexpr auto att_args = TensorAccessorArgs<0>();
    constexpr auto qg_args = TensorAccessorArgs<att_args.next_compile_time_args_offset()>();
    const uint32_t att_addr = get_arg_val<uint32_t>(0);
    const uint32_t qg_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);  // the tile's rows in the sparse output (32 on the verify tile)
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t column = get_arg_val<uint32_t>(4);  // tile column t of the head's 8
    const uint32_t qg_first = get_arg_val<uint32_t>(5);
    const auto att = TensorAccessor(att_args, att_addr);  // page = one head row of 256 bf16
    const auto qg = TensorAccessor(qg_args, qg_addr);     // page = one tile of the qg shard
    using tile_rows::TILE_BYTES;
    cb_reserve_back(CB_GATE, 1);
    cb_reserve_back(CB_ATT, 1);
    cb_reserve_back(CB_ROW, 1);
    const uint32_t gate_l1 = get_write_ptr(CB_GATE), att_l1 = get_write_ptr(CB_ATT), row_l1 = get_write_ptr(CB_ROW);
    {
        FUSED_ZONE("fz_qr_pa_r_gate");
        noc_async_read_page(qg_first + 2 * HEAD_TILES * head + GATE_FIRST + column, qg, gate_l1);
        tile_rows::fill_words(att_l1, TILE_BYTES / 4, 0);  // rows past the sparse output's are the chain's zero pad
    }
    {
        FUSED_ZONE("fz_qr_pa_r_rows");
        // every lane's 64-byte window of the head's row lands in the scratch (64-byte aligned), then the RISC places
        // its two 32-byte halves into the tile's faces (tile_rows::chunk_offset)
        for (uint32_t lane = 0; lane < rows; ++lane) {
            noc_async_read(
                att.get_noc_addr(head * rows + lane, column * WINDOW_BYTES),
                row_l1 + lane * WINDOW_BYTES,
                WINDOW_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t lane = 0; lane < rows; ++lane) {
            for (uint32_t half = 0; half < 2; ++half) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    row_l1 + lane * WINDOW_BYTES + half * tile_rows::ROW_BYTES);
                volatile tt_l1_ptr uint32_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(att_l1 + tile_rows::chunk_offset(lane, half));
                for (uint32_t k = 0; k < tile_rows::ROW_BYTES / 4; ++k) {
                    d[k] = s[k];
                }
            }
        }
    }
    cb_push_back(CB_GATE, 1);
    cb_push_back(CB_ATT, 1);
}
