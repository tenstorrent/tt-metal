// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routing of the tensor-parallel routed prefill (experts/tp_route.py): from a chunk's top-k expert ids, every expert's
// token count, its dispatch region (tile-aligned, the experts' regions back to back), the dispatch row of every
// (token, slot) pair and the token of every dispatch row. One program on E cores (core q = logical (q % GX, q / GX)):
//   1. token role (cores q < P): count the experts of token block q (TRC tile rows of 32 tokens) and send the counts
//      to every core (slot q of the shared count table), then wait for all P blocks' counts;
//   2. every core: totals, regions (exclusive prefix of the 32-rounded totals) and block q's first row per expert;
//      token cores write their tokens' rows (pair (t, j) of expert e: region[e] + #pairs of e from tokens < t) and
//      send each expert's run of their tokens to that expert's core (slot q of its run table);
//   3. expert role (core q = expert q): concatenate the P runs, zero-pad to 32 rows and write the region of the
//      row -> token table in one aligned write. Core 0 writes the counts, the regions and zeros from the end of the
//      last region through the spare tile.
// Every DRAM write is a whole page of its writer or a 128-byte aligned region of it: no two cores write the same bytes.
//
// Compile-time args: [C, K, E, P, TRC, SPARE, SLOT (run slot bytes), POS_PAGE (pos aligned page bytes), cb_idx,
//   cb_hist, cb_runs, cb_region, cb_pos, cb_stage, sem_hist, sem_runs, GX, accessors (indices, pos, tok, counts,
//   region)]. Common runtime args: [indices, pos, tok, counts, region addresses, noc x of cores 0..E-1, noc y ...].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TILE = 32;
constexpr uint32_t IDX_TILE_BYTES = TILE * TILE * 2;  // uint16 tile

// Expert id of slot j of token row r of a uint16 [32, 32] tile (faces of 16 x 16: r < 16 in face 0, else face 2).
inline uint32_t slot_id(const volatile tt_l1_ptr uint16_t* tile, uint32_t r, uint32_t j) {
    return tile[(r >> 4) * 512 + (r & 15) * 16 + j];
}

inline uint32_t round32(uint32_t n) { return (n + TILE - 1) / TILE * TILE; }
}  // namespace

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr uint32_t TRC = get_compile_time_arg_val(4);
    constexpr uint32_t SPARE = get_compile_time_arg_val(5);
    constexpr uint32_t SLOT = get_compile_time_arg_val(6);
    constexpr uint32_t POS_PAGE = get_compile_time_arg_val(7);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(8);
    constexpr uint32_t cb_hist = get_compile_time_arg_val(9);
    constexpr uint32_t cb_runs = get_compile_time_arg_val(10);
    constexpr uint32_t cb_region = get_compile_time_arg_val(11);
    constexpr uint32_t cb_pos = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(13);
    constexpr uint32_t sem_hist_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_runs_id = get_compile_time_arg_val(15);
    constexpr uint32_t GX = get_compile_time_arg_val(16);
    constexpr auto idx_args = TensorAccessorArgs<17>();
    constexpr auto pos_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto tok_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto cnt_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto reg_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();
    constexpr uint32_t TR = C / TILE;           // tile rows of tokens
    constexpr uint32_t BLOCK = TRC * TILE;      // tokens per token block
    constexpr uint32_t ROWS = SPARE + TILE;     // row -> token table length
    static_assert(K <= 16 && E <= 64 && P <= E && C % TILE == 0);

    const auto idx_acc = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(0), IDX_TILE_BYTES);
    const auto pos_acc = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(1), POS_PAGE);
    const auto tok_acc = TensorAccessor(tok_args, get_common_arg_val<uint32_t>(2), ROWS * 4);
    const auto cnt_acc = TensorAccessor(cnt_args, get_common_arg_val<uint32_t>(3), E * 4);
    const auto reg_acc = TensorAccessor(reg_args, get_common_arg_val<uint32_t>(4), E * 4);
    auto core_noc = [](uint32_t c, uint32_t l1) {
        return get_noc_addr(get_common_arg_val<uint32_t>(5 + c), get_common_arg_val<uint32_t>(5 + E + c), l1);
    };

    const uint32_t q = get_absolute_logical_y() * GX + get_absolute_logical_x();
    const uint32_t r0 = q * TRC;
    const uint32_t r1 = r0 + TRC < TR ? r0 + TRC : TR;
    const bool token_core = q < P;

    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const uint32_t hist_l1 = get_write_ptr(cb_hist);  // [P][E] uint32: every token block's counts
    const uint32_t runs_l1 = get_write_ptr(cb_runs);  // [P][SLOT]: block p's run of tokens of expert q
    const uint32_t region_l1 = get_write_ptr(cb_region);
    const uint32_t pos_l1 = get_write_ptr(cb_pos);      // [BLOCK][POS_PAGE]: block tokens' pair rows
    const uint32_t stage_l1 = get_write_ptr(cb_stage);  // [E][BLOCK] uint32: block tokens of each expert
    const uint32_t sem_hist = get_semaphore(sem_hist_id);
    const uint32_t sem_runs = get_semaphore(sem_runs_id);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_l1);

    // 1. this block's counts, to every core
    if (token_core) {
        for (uint32_t r = r0; r < r1; ++r) {
            noc_async_read(idx_acc.get_noc_addr(r), idx_l1 + (r - r0) * IDX_TILE_BYTES, IDX_TILE_BYTES);
        }
        volatile tt_l1_ptr uint32_t* mine = hist + q * E;
        for (uint32_t e = 0; e < E; ++e) {
            mine[e] = 0;
        }
        noc_async_read_barrier();
        for (uint32_t r = r0; r < r1; ++r) {
            const auto* tile = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(idx_l1 + (r - r0) * IDX_TILE_BYTES);
            for (uint32_t t = 0; t < TILE; ++t) {
                for (uint32_t j = 0; j < K; ++j) {
                    mine[slot_id(tile, t, j)] += 1;
                }
            }
        }
        for (uint32_t c = 0; c < E; ++c) {
            if (c != q) {
                noc_async_write(hist_l1 + q * E * 4, core_noc(c, hist_l1 + q * E * 4), E * 4);
            }
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < E; ++c) {
            noc_semaphore_inc(core_noc(c, sem_hist), 1);
        }
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_hist), P);

    // 2. totals, regions, this block's first row per expert
    uint32_t count[E], region[E], base[E];
    uint32_t end = 0;
    for (uint32_t e = 0; e < E; ++e) {
        uint32_t total = 0, before = 0;
        for (uint32_t p = 0; p < P; ++p) {
            const uint32_t h = hist[p * E + e];
            total += h;
            before += p < q ? h : 0;
        }
        count[e] = total;
        region[e] = end;
        base[e] = end + before;
        end += round32(total);
    }
    if (token_core) {
        uint32_t fill[E];
        for (uint32_t e = 0; e < E; ++e) {
            fill[e] = 0;
        }
        volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1);
        for (uint32_t r = r0; r < r1; ++r) {
            const auto* tile = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(idx_l1 + (r - r0) * IDX_TILE_BYTES);
            for (uint32_t t = 0; t < TILE; ++t) {
                const uint32_t i = (r - r0) * TILE + t;  // token within the block
                volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1 + i * POS_PAGE);
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t e = slot_id(tile, t, j);
                    pos[j] = base[e]++;
                    stage[e * BLOCK + fill[e]++] = r * TILE + t;
                }
                noc_async_write(pos_l1 + i * POS_PAGE, pos_acc.get_noc_addr(r * TILE + t), K * 4);
            }
        }
        for (uint32_t e = 0; e < E; ++e) {
            if (fill[e]) {
                noc_async_write(stage_l1 + e * BLOCK * 4, core_noc(e, runs_l1 + q * SLOT), (fill[e] * 4 + 15) / 16 * 16);
            }
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < E; ++c) {
            noc_semaphore_inc(core_noc(c, sem_runs), 1);
        }
    }

    // 3. expert q's region of the row -> token table
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_runs), P);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region_l1);
    uint32_t n = 0;
    for (uint32_t p = 0; p < P; ++p) {
        const volatile tt_l1_ptr uint32_t* run = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(runs_l1 + p * SLOT);
        const uint32_t h = hist[p * E + q];
        for (uint32_t i = 0; i < h; ++i) {
            out[n++] = run[i];
        }
    }
    const uint32_t padded = round32(count[q]);
    for (; n < padded; ++n) {
        out[n] = 0;
    }
    if (padded) {
        noc_async_write(region_l1, tok_acc.get_noc_addr(0, region[q] * 4), padded * 4);
    }
    if (q == 0) {
        // counts and regions, then zeros from the end of the last region through the spare tile (after the region
        // buffer, which this core may still be writing from)
        volatile tt_l1_ptr uint32_t* small =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region_l1 + round32(C) * 4);
        for (uint32_t e = 0; e < E; ++e) {
            small[e] = count[e];
            small[E + e] = region[e];
        }
        noc_async_write(region_l1 + round32(C) * 4, cnt_acc.get_noc_addr(0), E * 4);
        noc_async_write(region_l1 + round32(C) * 4 + E * 4, reg_acc.get_noc_addr(0), E * 4);
        const uint32_t zeros_l1 = region_l1 + round32(C) * 4 + 2 * E * 4;
        volatile tt_l1_ptr uint32_t* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros_l1);
        for (uint32_t i = 0; i < 1024; ++i) {
            zeros[i] = 0;
        }
        for (uint32_t row = end; row < ROWS; row += 1024) {
            const uint32_t len = ROWS - row < 1024 ? ROWS - row : 1024;
            noc_async_write(zeros_l1, tok_acc.get_noc_addr(0, row * 4), len * 4);
        }
    }
    noc_async_write_barrier();
}
