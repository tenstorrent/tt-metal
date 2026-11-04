// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows KV core: the circular buffers beside main_tail's (its cb_pack 24 and semaphores are shared with the
// key cores' writers), and the packed-row copy from the 16 pack tiles into a row-major cache row.
#pragma once
#include <cstdint>
#include "api/core_local_mem.h"
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "../../qsa_block/kernels/tile_rows.h"

namespace rows_kv {
constexpr uint32_t cb_block = 29, cb_next = 30, cb_kv_pos = 31;
constexpr uint32_t block_pages = tile_rows::rows_per_tile;  // one 1 KB row-major page per cache row, 32 per block
// The cb_kv_pos page (128 B): the position page lands at +0 (word 0 = P, the first position; word 1 takes the block
// start for the writer), the row-count page at +64. Each scalar's page read may cover up to 64 B, so a smaller offset
// would overwrite R.
constexpr uint32_t row_count_offset_bytes = 64;

// Row `tile_row` of the 16 pack tiles ([v tiles 0..7 | k tiles 8..15], TILE layout: two 16-column faces per row) as
// the 1 KB row-major cache row [v (256) | k (256)] at `row_l1`; -0 -> +0, as a one-hot placement in fp32 accumulate
// leaves a placed element.
inline void pack_row_to_rm(uint32_t pack_l1, uint32_t tile_row, uint32_t row_l1) {
    for (uint32_t c = 0; c < main_tail::pack_tiles; ++c) {
        for (uint32_t half = 0; half < 2; ++half) {
            const CoreLocalMem<const volatile uint16_t> src_chunk(
                pack_l1 + c * tile_rows::tile_bytes + tile_rows::chunk_offset(tile_row, half));
            const CoreLocalMem<volatile uint16_t> dst_chunk(
                row_l1 + c * 2 * tile_rows::half_row_bytes + half * tile_rows::half_row_bytes);
            for (uint32_t k = 0; k < 16; ++k) {
                const uint16_t value = src_chunk[k];
                dst_chunk[k] = value == 0x8000 ? 0 : value;
            }
        }
    }
}
}  // namespace rows_kv
