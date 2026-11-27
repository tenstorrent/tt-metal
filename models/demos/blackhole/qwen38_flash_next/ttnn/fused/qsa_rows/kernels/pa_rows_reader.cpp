// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 reader, one core per (local head h, tile column t): the head's gate tile t (tile
// qg_first + 16 h + 8 + t of the qg projection) and the head's attention rows' 32-column window t placed into one
// tile, row by row from the ROW_MAJOR [1, 32 heads, rows, 256] sparse output.  CBs: 0 gate (1 tile), 1 attention
// (1 tile), 3 row scratch (64 B per lane, one tile of room).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/tile_rows.h"

namespace {
constexpr uint32_t cb_gate = 0, cb_att = 1, cb_row = 3;
constexpr uint32_t head_tiles = 8, gate_first_tile = 8, window_bytes = 64;  // 32 bf16 of a 256-column head row
}  // namespace

void kernel_main() {
    constexpr auto attention_args = TensorAccessorArgs<0>();
    constexpr auto qg_args = TensorAccessorArgs<attention_args.next_compile_time_args_offset()>();
    const uint32_t attention_addr = get_arg_val<uint32_t>(0);
    const uint32_t qg_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);  // the tile's rows in the sparse output (32 on the verify tile)
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t column = get_arg_val<uint32_t>(4);  // tile column t of the head's 8
    const uint32_t qg_first = get_arg_val<uint32_t>(5);
    const auto attention = TensorAccessor(attention_args, attention_addr);  // page = one head row of 256 bf16
    const auto qg = TensorAccessor(qg_args, qg_addr);                       // page = one tile of the qg shard
    using tile_rows::tile_bytes;
    const Noc noc;
    CircularBuffer(cb_gate).reserve_back(1);
    CircularBuffer(cb_att).reserve_back(1);
    CircularBuffer(cb_row).reserve_back(1);
    // cb_row is scratch: reserved once and never pushed
    const uint32_t attention_l1 = CircularBuffer(cb_att).get_write_ptr(),
                   row_l1 = CircularBuffer(cb_row).get_write_ptr();
    {
        noc.async_read(
            qg,
            CircularBuffer(cb_gate),
            qg.get_aligned_page_size(),
            {.page_id = qg_first + 2 * head_tiles * head + gate_first_tile + column},
            {});
        tile_rows::fill_words(attention_l1, tile_bytes / 4, 0);  // rows past the sparse output's are zero padding
    }
    {
        // every lane's 64-byte window of the head's row lands in the scratch (64-byte aligned), then the RISC places
        // its two 32-byte halves into the tile's faces (tile_rows::chunk_offset)
        for (uint32_t lane = 0; lane < rows; ++lane) {
            noc.async_read(
                attention,
                CoreLocalMem<uint32_t>(row_l1),
                window_bytes,
                {.page_id = head * rows + lane, .offset_bytes = column * window_bytes},
                {.offset_bytes = lane * window_bytes});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t lane = 0; lane < rows; ++lane) {
            for (uint32_t half = 0; half < 2; ++half) {
                const CoreLocalMem<const volatile uint32_t> src_words(
                    row_l1 + lane * window_bytes + half * tile_rows::half_row_bytes);
                const CoreLocalMem<volatile uint32_t> dst_words(attention_l1 + tile_rows::chunk_offset(lane, half));
                for (uint32_t k = 0; k < tile_rows::half_row_bytes / 4; ++k) {
                    dst_words[k] = src_words[k];
                }
            }
        }
    }
    CircularBuffer(cb_gate).push_back(1);
    CircularBuffer(cb_att).push_back(1);
}
