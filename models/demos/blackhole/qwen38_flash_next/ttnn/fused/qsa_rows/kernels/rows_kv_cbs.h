// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows KV core: the circular buffers beside main_tail's (its CB_PACK 24 and semaphores are shared with the
// key cores' writers), and the packed-row copy from the 16 pack tiles into a row-major cache row.
#pragma once
#include <cstdint>
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "../../qsa_block/kernels/tile_rows.h"

namespace rows_kv {
constexpr uint32_t CB_BLOCK = 29, CB_NEXT = 30, CB_KV_POS = 31;
constexpr uint32_t BLOCK_PAGES = tile_rows::TILE_ROWS;  // one 1 KB row-major page per cache row, 32 per block
// The CB_KV_POS page (128 B): the position page lands at +0 (word 0 = P; word 1 takes the block start for the writer),
// the row-count page at +64 -- each scalar's page read may cover up to 64 B, so the two never overlap (a 32 B offset
// let the position read overwrite the row count: R read as 0 or as garbage, the first device run).
constexpr uint32_t ROWS_WORD_OFFSET = 64;
#ifdef PAGED
// PAGED (compile-time define; the unpaged programs are unchanged): the cache is the page pool and a 32-row block at
// logical row `row` starts at pool row pt[row >> shift] * bs + (row & (bs - 1)) (pt = the user's int32 page table row,
// bs = 1 << shift).  CB_KV_PT (one page of KV_PT_BYTES) holds the 64-byte aligned page-table windows: 3 * 64 bytes for
// the reader (slack for the alignment) and 3 * 64 for the writer, which look up on different RISCs of the same core.
constexpr uint32_t CB_KV_PT = 32;
constexpr uint32_t PT_WINDOW = 64, PT_WRITER_OFFSET = 3 * PT_WINDOW;
inline uint32_t pt_window_l1(uint32_t cb_base, uint32_t index) { return ((cb_base + 63) & ~63u) + index * PT_WINDOW; }
// The window around the entry of logical row `row` -> NoC read of 64 bytes from the table's row 0 (page 0).
template <typename PT>
inline void pt_window_read(const PT& pt, uint32_t row, uint32_t shift, uint32_t window_l1) {
    noc_async_read(pt.get_noc_addr(0, ((row >> shift) * 4) & ~63u), window_l1, PT_WINDOW);
}
// The pool row of logical row `row` once its window has landed (after the barrier and invalidate_l1_cache).
inline uint32_t pt_base_row(uint32_t window_l1, uint32_t row, uint32_t shift, uint32_t bs) {
    const uint32_t entry_byte = (row >> shift) * 4;
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_l1)[(entry_byte & 63u) >> 2] * bs + (row & (bs - 1));
}
#endif

#ifdef QWEN38_ROWS_KV_FAST
// The same copy over the pack tiles c0 .. c1 - 1 (v tiles 0..7, k tiles 8..15), 32-bit loads and stores: two bf16 per
// word, each half with the -0 -> +0 rule.  Source chunks (32 B aligned) and cache rows (1 KB) are 4 B aligned.
inline void pack_row_to_rm_words(uint32_t pack_l1, uint32_t tile_row, uint32_t rm, uint32_t c0, uint32_t c1) {
    for (uint32_t c = c0; c < c1; ++c) {
        for (uint32_t half = 0; half < 2; ++half) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                pack_l1 + c * tile_rows::TILE_BYTES + tile_rows::chunk_offset(tile_row, half));
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                rm + c * 2 * tile_rows::ROW_BYTES + half * tile_rows::ROW_BYTES);
            for (uint32_t k = 0; k < 8; ++k) {
                uint32_t w = s[k];
                if ((w & 0xFFFFu) == 0x8000u) {
                    w &= 0xFFFF0000u;
                }
                if ((w >> 16) == 0x8000u) {
                    w &= 0x0000FFFFu;
                }
                d[k] = w;
            }
        }
    }
}
#endif
// Row `tile_row` of the 16 pack tiles ([v tiles 0..7 | k tiles 8..15], TILE layout: two 16-column faces per row) as
// the 1 KB row-major cache row [v (256) | k (256)] at `rm`; -0 -> +0 as the chain's one-hot placement in fp32
// accumulate leaves a placed element (a -0 source adds +0 products and reads +0).
inline void pack_row_to_rm(uint32_t pack_l1, uint32_t tile_row, uint32_t rm) {
    for (uint32_t c = 0; c < main_tail::PACK_TILES; ++c) {
        for (uint32_t half = 0; half < 2; ++half) {
            volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                pack_l1 + c * tile_rows::TILE_BYTES + tile_rows::chunk_offset(tile_row, half));
            volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                rm + c * 2 * tile_rows::ROW_BYTES + half * tile_rows::ROW_BYTES);
            for (uint32_t k = 0; k < 16; ++k) {
                const uint16_t value = s[k];
                d[k] = value == 0x8000 ? 0 : value;
            }
        }
    }
}
}  // namespace rows_kv
