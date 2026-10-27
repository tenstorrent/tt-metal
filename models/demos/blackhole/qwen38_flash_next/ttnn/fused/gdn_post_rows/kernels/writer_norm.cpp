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
// zeroed tile and write it to page col of ``history_next`` [1, 1, 32, 2560].  A bf16 tile row r is two 32-byte
// segments: face (r >> 4) * 2 + {0, 1} at element offset (r & 15) * 16.
// CBs: cb_proj (5, the projection tiles), cb_out (16, the gated tiles), cb_hist (17, the history scratch).
// Compile-time args: TensorAccessorArgs of gated, history_next, chained from 0.  Runtime args: 0 gated address,
// 1 history_next address, 2 chunks (NC), 3 rows, 4 history, 5 units on this core, 6 first unit.

#include "api/dataflow/dataflow_api.h"
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

// byte offset of the left (segment 0) or right (segment 1) half of tile row r inside a bf16 tile
constexpr uint32_t row_half(uint32_t row, uint32_t segment) {
    return ((row >> 4) * 2 + segment) * face_bytes + (row & 15) * half_row_bytes;
}

void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}

void copy_words(uint32_t dst, uint32_t src, uint32_t words) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t k = 0; k < words; ++k) {
        d[k] = s[k];
    }
}
}  // namespace

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto history_next_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_next_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t rows = get_arg_val<uint32_t>(arg++);
    const uint32_t history = get_arg_val<uint32_t>(arg++);
    const uint32_t units = get_arg_val<uint32_t>(arg++);
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);

    const auto out = TensorAccessor(out_args, out_addr);
    const auto history_next = TensorAccessor(history_next_args, history_next_addr);
    cb_reserve_back(cb_hist, history_columns);
    const uint32_t hist_l1 = get_write_ptr(cb_hist);

    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t unit = first_unit + i;
        const uint32_t head = unit / chunks;
        const uint32_t chunk = unit - head * chunks;

        cb_wait_front(cb_out, head_tiles);
        const uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t start_row = chunk * tile_rows;
        const uint32_t valid = rows <= start_row ? 0 : (rows - start_row < tile_rows ? rows - start_row : tile_rows);
        if (valid < tile_rows) {
            for (uint32_t d = 0; d < head_tiles; ++d) {
                for (uint32_t r = valid; r < tile_rows; ++r) {
                    fill_words(l1 + d * bf16_tile_bytes + row_half(r, 0), half_row_words, 0);
                    fill_words(l1 + d * bf16_tile_bytes + row_half(r, 1), half_row_words, 0);
                }
            }
        }
        {
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_write_page(chunk * value_tiles + head * head_tiles + d, out, l1 + d * bf16_tile_bytes);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, head_tiles);

        if (history != 0 && chunk + 1 == chunks && head < history_heads) {
            cb_wait_front(cb_proj, history_columns);
            const uint32_t p_l1 = get_read_ptr(cb_proj);
            for (uint32_t j = 0; j < history_columns; ++j) {
                const uint32_t tile = hist_l1 + j * bf16_tile_bytes;
                fill_words(tile, bf16_tile_bytes / 4, 0);
                for (uint32_t r = 0; r < history_rows; ++r) {
                    const uint32_t source = tile_rows - history_rows + r;
                    for (uint32_t segment = 0; segment < 2; ++segment) {
                        copy_words(
                            tile + row_half(r, segment),
                            p_l1 + j * bf16_tile_bytes + row_half(source, segment),
                            half_row_words);
                    }
                }
                noc_async_write_page(head * history_columns + j, history_next, tile);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_proj, history_columns);
        }
    }
}
