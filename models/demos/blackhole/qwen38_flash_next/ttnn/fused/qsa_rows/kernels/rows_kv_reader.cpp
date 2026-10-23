// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows, the KV core's reader (one core): the verify chain's packed KV write on the 32-row tile at R real
// rows. The value tiles (8) come from the v projection window, the key's rotated tiles (8, 9) and normalized tiles
// (10..15) from the key cores (main_tail's writer_rope / writer_norm, ROLE 1, into CB_PACK with SEM_KROT / SEM_KNORM).
// The current cache block (the 32 row-major rows at P & ~31) is read back, row j < R with (P + j) in that block
// replaces row (P + j) & 31 with the packed row [v_j | k_j] (tile row j of the 16 pack tiles, -0 -> +0 as the chain's
// one-hot placement leaves it), its rows past P % 32 + R - 1 are zeroed as the chain leaves them, the 32 rows go to the
// writer (CB_BLOCK); then, unless single_row, the NEXT block: 32 zero rows with the rows crossing the block edge placed
// at their slots (CB_NEXT) -- the chain writes that block every pass (stage_b_select @ packed).  Rows past R (the
// tile's stale rows) are never read. The row count R is read from a uint32 [1, 1, 1, 1] page as well (the verify form's
// constant k + 1, the commit form's accepted count a*: one program for both), clamped to 32. Compile-time args:
// TensorAccessorArgs cache, v, position, rows. Runtime args: 0 cache, 1 v, 2 position, 3 rows addresses, 4 first tile
// of the value in v, 5 single_row.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "../../kernels/zones.h"
#include "rows_kv_cbs.h"
using namespace main_tail;
using namespace rows_kv;

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t rows_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_first = get_arg_val<uint32_t>(4);
    const uint32_t single_row = get_arg_val<uint32_t>(5);
    constexpr auto cache_args = TensorAccessorArgs<0>();
    constexpr auto v_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto rows_page = TensorAccessor(rows_args, rows_addr);
    Semaphore<> sem_krot(SEM_KROT), sem_knorm(SEM_KNORM);

    // the tile's first position P and the row count R (uint32 [1, 1, 1, 1] each) and the value tiles into pack slots
    // 0..7
    cb_reserve_back(CB_KV_POS, 1);
    cb_reserve_back(CB_PACK, PACK_TILES);
    const uint32_t pos_l1 = get_write_ptr(CB_KV_POS), pack_l1 = get_write_ptr(CB_PACK);
    noc_async_read_page(0, pos, pos_l1);
    noc_async_read_page(0, rows_page, pos_l1 + ROWS_WORD_OFFSET);
    {
        FUSED_ZONE("fz_qr_kv_value");
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(v_first + c, v, pack_l1 + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    const uint32_t P = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    const uint32_t rows_read = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1 + ROWS_WORD_OFFSET);
    const uint32_t rows = rows_read < tile_rows::TILE_ROWS ? rows_read : tile_rows::TILE_ROWS;
    const uint32_t block = P & ~31u;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1 + 4) = block;  // the writer reads the block start here
    cb_push_back(CB_KV_POS, 1);

    // the current block, read back as the chain's kv row lookup does
    cb_reserve_back(CB_BLOCK, BLOCK_PAGES);
    const uint32_t block_l1 = get_write_ptr(CB_BLOCK);
    {
        FUSED_ZONE("fz_qr_kv_block");
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_read(cache.get_noc_addr(block + k, 0), block_l1 + k * QUERY_ROW_BYTES, QUERY_ROW_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    {
        FUSED_ZONE("fz_qr_kv_key_wait");
        sem_krot.wait_min(1);
        sem_knorm.wait_min(1);
        invalidate_l1_cache();
    }
    // this pass's rows into the current block; the crossing rows into the next block's zero rows
    cb_reserve_back(CB_NEXT, BLOCK_PAGES);
    const uint32_t next_l1 = get_write_ptr(CB_NEXT);
    if (!single_row) {
        tile_rows::fill_words(next_l1, BLOCK_PAGES * QUERY_ROW_BYTES / 4, 0u);
    }
    {
        FUSED_ZONE("fz_qr_kv_scatter");
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t position = P + j;
            const uint32_t slot = position & 31u;
            if ((position & ~31u) == block) {
                pack_row_to_rm(pack_l1, j, block_l1 + slot * QUERY_ROW_BYTES);
            } else if (!single_row) {
                pack_row_to_rm(pack_l1, j, next_l1 + slot * QUERY_ROW_BYTES);
            }
        }
        // the chain keeps the block's rows below P % 32 and places rows P % 32 .. P % 32 + R - 1; the rows past them
        // (past P + R - 1, never named by a sparse row before a later pass rewrites them) read 0 * x + 0 = +0 there
        for (uint32_t slot = (P & 31u) + rows; slot < tile_rows::TILE_ROWS; ++slot) {
            tile_rows::fill_words(block_l1 + slot * QUERY_ROW_BYTES, QUERY_ROW_BYTES / 4, 0u);
        }
    }
    cb_push_back(CB_BLOCK, BLOCK_PAGES);
    cb_push_back(CB_NEXT, BLOCK_PAGES);
    sem_krot.set(0);
    sem_knorm.set(0);
}
