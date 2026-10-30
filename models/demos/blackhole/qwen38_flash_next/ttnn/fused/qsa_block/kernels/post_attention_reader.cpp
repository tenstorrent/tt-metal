// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// post_attention, one core per local head: the head's gate tiles (the second 256 columns of its 512 in the qg
// projection) and the head's attention rows placed into an otherwise zero tile row (slice + tilize with zero padding).
// CBs: 0 gate (bf16, 8), 1 attention (bf16, 8), 3 row scratch (bf16, 1).
// Compile-time args: TensorAccessorArgs attention (ROW_MAJOR [1, 32, rows, 256]), qg.
// Runtime args: 0 attention, 1 qg addresses, 2 rows, 3 head, 4 first tile of the qg projection in qg.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"

void kernel_main() {
    const uint32_t attention_addr = get_arg_val<uint32_t>(0);
    const uint32_t qg_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t qg_first = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_gate = 0, cb_att = 1, cb_row = 3, head_tiles = 8, gate_first_tile = 8;
    constexpr auto attention_args = TensorAccessorArgs<0>();
    constexpr auto qg_args = TensorAccessorArgs<attention_args.next_compile_time_args_offset()>();
    const auto attention = TensorAccessor(attention_args, attention_addr);
    const auto qg = TensorAccessor(qg_args, qg_addr);
    using tile_rows::half_row_bytes;
    using tile_rows::tile_bytes;
    const Noc noc;

    CircularBuffer(cb_gate).reserve_back(head_tiles);
    CircularBuffer(cb_att).reserve_back(head_tiles);
    const uint32_t attention_l1 = CircularBuffer(cb_att).get_write_ptr();
    noc.async_write_zeros(CoreLocalMem<uint32_t>(attention_l1), head_tiles * tile_bytes, {});
    noc.write_zeros_l1_barrier();
    for (uint32_t t = 0; t < head_tiles; ++t) {
        noc.async_read(
            qg,
            CircularBuffer(cb_gate),
            qg.get_aligned_page_size(),
            {.page_id = qg_first + 2 * head_tiles * head + gate_first_tile + t},
            {.offset_bytes = t * tile_bytes});
    }
    // DRAM reads land whole (64-byte aligned) rows in the scratch; the RISC places the chunks into the tile faces
    CircularBuffer(cb_row).reserve_back(1);
    // cb_row is scratch: reserved once and never pushed
    const uint32_t row_l1 = CircularBuffer(cb_row).get_write_ptr();
    for (uint32_t lane = 0; lane < rows; ++lane) {
        noc.async_read(
            attention,
            CoreLocalMem<uint32_t>(row_l1),
            head_tiles * 2 * half_row_bytes,
            {.page_id = head * rows + lane, .offset_bytes = 0},
            {});
        noc.async_read_barrier();
        invalidate_l1_cache();
        tile_rows::place_row(row_l1, attention_l1, head_tiles, lane);
    }
    noc.async_read_barrier();
    CircularBuffer(cb_gate).push_back(head_tiles);
    CircularBuffer(cb_att).push_back(head_tiles);
}
