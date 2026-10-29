// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes form of gdn_rows_scan's writer: one (value head, lane) item per core, the whole head
// (block_tiles = head_tiles), the compute kernel unchanged (row r is tile row t = lane * R + r). Row t of each o tile
// goes into the zeroed o-rows tiles, the state after r + 1 rows to prefix slot t, and only lane u's rows of the gated
// output are written; the last lane's cores also write the pad rows B * R .. 31 as exact zeros. One core owns one lane
// of one head. Compile-time args: 0 rows (R, rows per lane), 1 block_tiles (= 4, the builder's vbt), 2 lanes (B); then
// TensorAccessorArgs of prefix ([B * R, 12, 128, 128] fp32) and out ([1, 1, 32, 1536] bf16).  Runtime args: 0 prefix,
// 1 out addresses, 2 items, then (head, lane) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t cb_outs = 25, cb_outg = 26, cb_orows = 29, cb_obf = 31;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
constexpr uint32_t lanes = get_compile_time_arg_val(2);
constexpr uint32_t value_heads = 12, head_tiles = 4, block_state_tiles = head_tiles * block_tiles;
constexpr uint32_t head_state_tiles = head_tiles * head_tiles, tile_rows = 32;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096;
constexpr uint32_t face_bytes = fused_rows::face_bytes, half_row_bytes = fused_rows::half_row_bytes;
static_assert(block_tiles == head_tiles, "the lanes form runs the whole head on one core");
static_assert(rows >= 1 && lanes >= 1 && lanes * rows <= tile_rows, "B lanes x R rows in one tile");

// Lane u's state after row + 1 rows -> prefix slot row0 + row: the head's `block_state_tiles` tiles from cb_outs.
template <typename Accessor>
FORCE_INLINE void drain_state(const Accessor& prefix, uint32_t row, uint32_t row0, uint32_t head) {
    const Noc noc;
    CircularBuffer(cb_outs).wait_front(block_state_tiles);
    for (uint32_t kt = 0; kt < head_tiles; ++kt) {
        for (uint32_t c = 0; c < block_tiles; ++c) {
            const uint32_t page = ((row0 + row) * value_heads + head) * head_state_tiles + kt * head_tiles + c;
            noc.async_write(
                CircularBuffer(cb_outs),
                prefix,
                prefix.get_aligned_page_size(),
                {.offset_bytes = (kt * block_tiles + c) * fp32_tile_bytes},
                {.page_id = page});
        }
    }
    noc.async_write_barrier();
    CircularBuffer(cb_outs).pop_front(block_state_tiles);
}

// Tile row `row` of the bf16 tile at `src_l1` into the same row of page `page` of `out`: two 32 B face pieces.
template <typename Accessor>
void write_tile_row(const Accessor& out, uint32_t page, uint32_t src_l1, uint32_t row) {
    const Noc noc;
    const uint32_t offset = fused_rows::bf16_row_offset(row);
    noc.async_write(
        CoreLocalMem<uint32_t>(src_l1 + offset), out, half_row_bytes, {}, {.page_id = page, .offset_bytes = offset});
    noc.async_write(
        CoreLocalMem<uint32_t>(src_l1 + offset + face_bytes),
        out,
        half_row_bytes,
        {},
        {.page_id = page, .offset_bytes = offset + face_bytes});
}
}  // namespace

void kernel_main() {
    constexpr auto prefix_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t row0 = lane * rows;

        CircularBuffer(cb_orows).reserve_back(head_tiles);
        // Taken before the push below moves the write pointer: the pad rows are written from here after it.
        const uint32_t o_rows = CircularBuffer(cb_orows).get_write_ptr();
        fused_rows::fill_words(
            o_rows, head_tiles * bf16_tile_bytes / 4, 0);  // every row not lane u's stays exactly zero

        for (uint32_t r = 0; r < rows; ++r) {
            CircularBuffer(cb_obf).wait_front(block_tiles);
            const uint32_t o_tiles = CircularBuffer(cb_obf).get_read_ptr();
            for (uint32_t j = 0; j < block_tiles; ++j) {
                fused_rows::copy_row_bf16(o_rows + j * bf16_tile_bytes, o_tiles + j * bf16_tile_bytes, row0 + r);
            }
            CircularBuffer(cb_obf).pop_front(block_tiles);
            if (r + 1 < rows) {
                drain_state(prefix, r, row0, head);
            }
        }
        // the gated norm may start; the last prefix state drains meanwhile
        CircularBuffer(cb_orows).push_back(head_tiles);
        drain_state(prefix, rows - 1, row0, head);
        {
            CircularBuffer(cb_outg).wait_front(head_tiles);
            const uint32_t gated = CircularBuffer(cb_outg).get_read_ptr();
            // lane u's rows of the head's four gated column tiles
            for (uint32_t c = 0; c < head_tiles; ++c) {
                const uint32_t page = head * head_tiles + c;
                for (uint32_t r = 0; r < rows; ++r) {
                    write_tile_row(out, page, gated + c * bf16_tile_bytes, row0 + r);
                }
                if (lane + 1 == lanes) {  // the pad rows past B * R: exact zeros (the o-rows tiles' untouched rows)
                    for (uint32_t t = lanes * rows; t < tile_rows; ++t) {
                        write_tile_row(out, page, o_rows + c * bf16_tile_bytes, t);
                    }
                }
            }
            noc.async_write_barrier();
            CircularBuffer(cb_outg).pop_front(head_tiles);
        }
    }
}
