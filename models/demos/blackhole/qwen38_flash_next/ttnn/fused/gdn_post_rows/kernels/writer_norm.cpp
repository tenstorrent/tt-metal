// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_norm`` writer.  Per work unit u = head * NC + tile row: the four gated tiles to ``gated``
// [1, 1, T, 1536] at the token-major pages c * 48 + 4h + d (the fold of the chain's twelve head slices and the
// concat).  With ``rows`` < T the tile row's rows past the valid ones are zeroed in L1 first: the chain multiplies
// them by an exact 0.0, which is +0 under the bf16 multiply's zero clamp.
//
// The history (``commit_rows_full``): the units of the last tile row with head < 10 each take eight of the
// projection's 80 q|k|v column tiles (columns 8h .. 8h + 7) from CB_PROJ, copy their rows 29..31 into rows 0..2 of a
// zeroed tile and write it to page col of ``history_next`` [1, 1, 32, 2560].  A bf16 tile row r is two 32-byte
// segments: face (r >> 4) * 2 + {0, 1} at element offset (r & 15) * 16.  Data movement: bitwise.
// CBs: CB_PROJ (5, the projection tiles), CB_OUT (16, the gated tiles), CB_HIST (17, the history scratch).
// Compile-time args: TensorAccessorArgs of gated, history_next, chained from 0.  Runtime args: 0 gated address,
// 1 history_next address, 2 chunks (NC), 3 rows, 4 history, 5 units on this core, 6 first unit.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_PROJ = 5, CB_OUT = 16, CB_HIST = 17;
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t HEAD_TILES = 4;    // 128 / 32
constexpr uint32_t VALUE_TILES = 48;  // 1536 / 32
constexpr uint32_t HISTORY_HEADS = 10;
constexpr uint32_t HISTORY_COLUMNS = 8;
constexpr uint32_t HISTORY_ROWS = 3;
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t FACE_BYTES = 512;     // 16 x 16 bf16
constexpr uint32_t HALF_ROW_BYTES = 32;  // 16 bf16 of one tile row inside one face
constexpr uint32_t HALF_ROW_WORDS = HALF_ROW_BYTES / 4;

// byte offset of the left (segment 0) or right (segment 1) half of tile row r inside a bf16 tile
constexpr uint32_t row_half(uint32_t row, uint32_t segment) {
    return ((row >> 4) * 2 + segment) * FACE_BYTES + (row & 15) * HALF_ROW_BYTES;
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
    constexpr auto hist_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t hist_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t rows = get_arg_val<uint32_t>(arg++);
    const uint32_t history = get_arg_val<uint32_t>(arg++);
    const uint32_t units = get_arg_val<uint32_t>(arg++);
    const uint32_t first = get_arg_val<uint32_t>(arg++);

    const auto out = TensorAccessor(out_args, out_addr);
    const auto hist = TensorAccessor(hist_args, hist_addr);
    cb_reserve_back(CB_HIST, HISTORY_COLUMNS);
    const uint32_t hist_l1 = get_write_ptr(CB_HIST);

    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t unit = first + i;
        const uint32_t head = unit / chunks;
        const uint32_t chunk = unit - head * chunks;

        cb_wait_front(CB_OUT, HEAD_TILES);
        const uint32_t l1 = get_read_ptr(CB_OUT);
        const uint32_t start_row = chunk * TILE_ROWS;
        const uint32_t valid = rows <= start_row ? 0 : (rows - start_row < TILE_ROWS ? rows - start_row : TILE_ROWS);
        if (valid < TILE_ROWS) {
            FUSED_ZONE("fz_gpo_wn_mask");
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                for (uint32_t r = valid; r < TILE_ROWS; ++r) {
                    fill_words(l1 + d * BF16_TILE + row_half(r, 0), HALF_ROW_WORDS, 0);
                    fill_words(l1 + d * BF16_TILE + row_half(r, 1), HALF_ROW_WORDS, 0);
                }
            }
        }
        {
            FUSED_ZONE("fz_gpo_wn_write");
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_write_page(chunk * VALUE_TILES + head * HEAD_TILES + d, out, l1 + d * BF16_TILE);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(CB_OUT, HEAD_TILES);

        if (history != 0 && chunk + 1 == chunks && head < HISTORY_HEADS) {
            FUSED_ZONE("fz_gpo_wn_history");
            cb_wait_front(CB_PROJ, HISTORY_COLUMNS);
            const uint32_t p_l1 = get_read_ptr(CB_PROJ);
            for (uint32_t j = 0; j < HISTORY_COLUMNS; ++j) {
                const uint32_t tile = hist_l1 + j * BF16_TILE;
                fill_words(tile, BF16_TILE / 4, 0);
                for (uint32_t r = 0; r < HISTORY_ROWS; ++r) {
                    const uint32_t source = TILE_ROWS - HISTORY_ROWS + r;
                    for (uint32_t segment = 0; segment < 2; ++segment) {
                        copy_words(
                            tile + row_half(r, segment),
                            p_l1 + j * BF16_TILE + row_half(source, segment),
                            HALF_ROW_WORDS);
                    }
                }
                noc_async_write_page(head * HISTORY_COLUMNS + j, hist, tile);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_PROJ, HISTORY_COLUMNS);
        }
    }
}
