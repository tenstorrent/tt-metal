// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_norm`` writer.  Per work unit u = head * NC + tile row: the four gated tiles to ``gated``
// [1, 1, T, 1536] at the token-major pages c * 48 + 4h + d (the fold of the chain's twelve head slices and the
// concat).  With ``rows`` < T the tile row's rows past the valid ones are zeroed in L1 first: the chain multiplies
// them by an exact 0.0, which is +0 under the bf16 multiply's zero clamp.
//
// The history (``commit_rows_full``): the units of the last tile row with head < 10 each take eight of the
// projection's 80 q|k|v column tiles (columns 8h .. 8h + 7) from cb_proj, copy their rows 29..31 into rows 0..2 of a
// zeroed tile and write each to its column's page of ``history_next`` [1, 1, 32, 2560].  A bf16 tile row r is two
// 32-byte segments: face (r >> 4) * 2 + {0, 1} at element offset (r & 15) * 16.
// CBs: cb_proj (5, the projection tiles), cb_out (16, the gated tiles), cb_hist (17, the history scratch).
// Compile-time args: TensorAccessorArgs of gated, history_next, chained from 0.  Runtime args: 0 gated address,
// 1 history_next address, 2 chunks (NC), 3 rows, 4 history, 5 units on this core, 6 first unit.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_proj = 5, cb_out = 16, cb_hist = 17;
constexpr uint32_t tile_rows = 32;
constexpr uint32_t head_tiles = 4;    // 128 / 32
constexpr uint32_t value_tiles = 48;  // 1536 / 32
constexpr uint32_t history_heads = 10;
constexpr uint32_t history_columns = 8;
constexpr uint32_t history_rows = 3;
constexpr uint32_t bf16_tile_bytes = 2048;
constexpr uint32_t face_bytes = 512;     // 16 x 16 bf16
constexpr uint32_t half_row_bytes = 32;  // 16 bf16 of one tile row inside one face
constexpr uint32_t half_row_words = half_row_bytes / 4;

// byte offset of the left (segment 0) or right (segment 1) half of tile row `row` inside a bf16 tile
constexpr uint32_t row_half(uint32_t row, uint32_t segment) {
    return ((row >> 4) * 2 + segment) * face_bytes + (row & 15) * half_row_bytes;
}

void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = value;
    }
}

void copy_words(uint32_t dst_addr, uint32_t src_addr, uint32_t words) {
    CoreLocalMem<volatile uint32_t> dst(dst_addr);
    CoreLocalMem<volatile uint32_t> src(src_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = src[k];
    }
}
}  // namespace

void kernel_main() {
    constexpr auto gated_args = TensorAccessorArgs<0>();
    constexpr auto history_next_args = TensorAccessorArgs<gated_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t gated_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_next_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t rows = get_arg_val<uint32_t>(arg++);
    const uint32_t history = get_arg_val<uint32_t>(arg++);
    const uint32_t units = get_arg_val<uint32_t>(arg++);
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);

    const auto gated = TensorAccessor(gated_args, gated_addr);
    const auto history_next = TensorAccessor(history_next_args, history_next_addr);
    const Noc noc;
    // The history scratch: reserved once and never pushed, so its write pointer is stable for the whole kernel.
    CircularBuffer(cb_hist).reserve_back(history_columns);
    const uint32_t history_l1 = CircularBuffer(cb_hist).get_write_ptr();

    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t unit = first_unit + i;
        const uint32_t head = unit / chunks;
        const uint32_t chunk = unit - head * chunks;

        CircularBuffer(cb_out).wait_front(head_tiles);
        const uint32_t gated_l1 = CircularBuffer(cb_out).get_read_ptr();
        const uint32_t start_row = chunk * tile_rows;
        const uint32_t valid = rows <= start_row ? 0 : (rows - start_row < tile_rows ? rows - start_row : tile_rows);
        if (valid < tile_rows) {
            for (uint32_t d = 0; d < head_tiles; ++d) {
                for (uint32_t row = valid; row < tile_rows; ++row) {
                    fill_words(gated_l1 + d * bf16_tile_bytes + row_half(row, 0), half_row_words, 0);
                    fill_words(gated_l1 + d * bf16_tile_bytes + row_half(row, 1), half_row_words, 0);
                }
            }
        }
        {
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc.async_write(
                    CircularBuffer(cb_out),
                    gated,
                    gated.get_aligned_page_size(),
                    {.offset_bytes = d * bf16_tile_bytes},
                    {.page_id = chunk * value_tiles + head * head_tiles + d});
            }
            noc.async_write_barrier();
        }
        CircularBuffer(cb_out).pop_front(head_tiles);

        if (history != 0 && chunk + 1 == chunks && head < history_heads) {
            CircularBuffer(cb_proj).wait_front(history_columns);
            const uint32_t projected_l1 = CircularBuffer(cb_proj).get_read_ptr();
            for (uint32_t j = 0; j < history_columns; ++j) {
                const uint32_t history_tile = history_l1 + j * bf16_tile_bytes;
                fill_words(history_tile, bf16_tile_bytes / 4, 0);
                for (uint32_t row = 0; row < history_rows; ++row) {
                    const uint32_t source_row = tile_rows - history_rows + row;
                    for (uint32_t segment = 0; segment < 2; ++segment) {
                        copy_words(
                            history_tile + row_half(row, segment),
                            projected_l1 + j * bf16_tile_bytes + row_half(source_row, segment),
                            half_row_words);
                    }
                }
                noc.async_write(
                    CoreLocalMem<uint32_t>(history_tile),
                    history_next,
                    history_next.get_aligned_page_size(),
                    {},
                    {.page_id = head * history_columns + j});
            }
            noc.async_write_barrier();
            CircularBuffer(cb_proj).pop_front(history_columns);
        }
    }
}
