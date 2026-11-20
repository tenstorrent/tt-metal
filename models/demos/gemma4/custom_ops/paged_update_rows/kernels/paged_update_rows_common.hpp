// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// PagedUpdateRows: the compile-time argument layout shared by the reader, the compute and the writer kernel (the
// wrapper passes one list to all three; the TensorAccessorArgs of src, idx, page_table and cache follow at CT_END).
//
// A work unit is one (record row b, KV head h): its P packed rows (rows b*P .. b*P+P-1 of head h of the bf16 TILE
// source) go to cache positions idx_p[b] (wrapped into the ring when MODULO > 0, -1 = skip) through the page table
// row b. P <= 32 consecutive positions touch at most two cache tile rows (the ring's token count is a tile multiple,
// so a wrap lands on a tile boundary), so every unit runs exactly J tile-row JOBS (J = min(P, 2)); a job with no
// position is a null job (the reader pushes a stale block, the writer skips the write-back). The cache
// read-modify-write of one tile row is the stock paged_update_cache sequence (untilize the row's Wt tiles to bf16
// row-major, patch the rows, tilize back in the cache format), done by exactly one core per touched tile row.
#pragma once
#include <cstdint>

namespace pur {
constexpr uint32_t CB_SRC = get_compile_time_arg_val(0);       // the two source tile rows of the unit (bf16 TILE)
constexpr uint32_t CB_UNT_SRC = get_compile_time_arg_val(1);   // their untilized rows (bf16 row-major)
constexpr uint32_t CB_CACHE = get_compile_time_arg_val(2);     // one cache tile row per job (cache format)
constexpr uint32_t CB_UNT_A = get_compile_time_arg_val(3);     // the untilized cache tile row (compute -> writer)
constexpr uint32_t CB_UNT_B = get_compile_time_arg_val(4);     // the same L1 region, patched (writer -> compute)
constexpr uint32_t CB_OUT = get_compile_time_arg_val(5);       // the re-tilized tile row (cache format)
constexpr uint32_t CB_IDX = get_compile_time_arg_val(6);       // reader scratch: the P index sticks
constexpr uint32_t CB_PT = get_compile_time_arg_val(7);        // reader scratch: one page-table row
constexpr uint32_t CB_JOBS = get_compile_time_arg_val(8);      // the unit's job table (reader -> writer)
constexpr uint32_t B = get_compile_time_arg_val(9);            // record rows
constexpr uint32_t P = get_compile_time_arg_val(10);           // packed positions per row
constexpr uint32_t Wt = get_compile_time_arg_val(11);          // head_dim / 32
constexpr uint32_t St = get_compile_time_arg_val(12);          // block size in tiles
constexpr uint32_t NH = get_compile_time_arg_val(13);          // KV heads of the cache view
constexpr uint32_t BLOCK = get_compile_time_arg_val(14);       // block size in tokens
constexpr uint32_t MODULO = get_compile_time_arg_val(15);      // cache_position_modulo (0: unbounded)
constexpr uint32_t SRC_ST = get_compile_time_arg_val(16);      // source tile rows per head (padded rows / 32)
constexpr uint32_t J = get_compile_time_arg_val(17);           // tile-row jobs per unit
constexpr uint32_t IDX_STICK = get_compile_time_arg_val(18);   // aligned bytes of one index stick
constexpr uint32_t PT_STICK = get_compile_time_arg_val(19);    // aligned bytes of one page-table row
constexpr uint32_t PT_ENTRIES = get_compile_time_arg_val(20);  // blocks per page-table row
constexpr uint32_t CACHE_TILE = get_compile_time_arg_val(21);  // cache tile bytes (2048 bf16 / 1088 bfp8)
constexpr uint32_t CT_END = 22;

constexpr uint32_t SRC_TILE = 2048;       // bf16 tile
constexpr uint32_t WBYTES = Wt * 64;      // one untilized bf16 row of head_dim elements
constexpr uint32_t N_SRC = 2;             // source tile rows untilized per unit (see the header comment)
constexpr uint32_t NO_JOB = 0xFFFFFFFFu;  // a skipped position (idx == -1) / the stock's skip sentinel
// The job table of one unit, uint32 words: [J x (valid, first cache tile id)] then [P x (job, row in tile, row in
// the untilized source block)].
constexpr uint32_t JOB_WORDS = 2 * J + 3 * P;

static_assert(P >= 1 && P <= 32, "P packed positions per row: 1..32 (two tile rows per unit)");
static_assert(J >= 1 && J <= 2, "a unit runs one or two tile-row jobs");
static_assert(BLOCK % 32 == 0 && St * 32 == BLOCK, "the block size is a tile multiple");
static_assert(MODULO == 0 || MODULO % BLOCK == 0, "the ring is a whole number of blocks (the stock contract)");
}  // namespace pur
