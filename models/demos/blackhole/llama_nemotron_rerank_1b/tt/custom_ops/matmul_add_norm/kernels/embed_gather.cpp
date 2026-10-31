// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// embed_rmsnorm's gather (RISCV_0; the norm's exchange and y writes stay on RISCV_1): fills this
// core's staging CB 14 with its rows x columns of the token embeddings, row-major, one 32-row
// tile-row at a time; the compute tilizes each tile-row into the residual shard.
//
// The n_peers cores holding the same rows split each tile-row's 32 tokens (token i to peer
// i mod n_peers). A core reads its tokens' whole embedding rows from DRAM (one contiguous read
// each, up to ROW_SLOTS in flight) and writes each peer's column slice into that peer's CB 14,
// which holds the whole shard so no slot is reused. Once a tile-row's writes are acked it increments
// that row's counter (CB 16, a 16-byte word per tile-row) on every peer; a core pushes staging
// tile-row r to the compute when its counter r reaches n_peers. The counters are zeroed before a
// start handshake on the embed semaphore, so no increment can precede the zeroing.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"  // fill_zeros_async

namespace {
constexpr uint32_t cb_ids_id = tt::CBIndex::c_13;
constexpr uint32_t cb_stage_id = tt::CBIndex::c_14;
constexpr uint32_t cb_bounce_id = tt::CBIndex::c_15;
constexpr uint32_t cb_cnt_id = tt::CBIndex::c_16;
constexpr uint32_t ROW_IDS_BYTES = 32 * sizeof(uint32_t);
constexpr uint32_t COL_BYTES = 32 * 2;  // one bf16 tile column of an embedding row
constexpr uint32_t CNT_STRIDE = 16;
}  // namespace

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t n_peers = get_compile_time_arg_val(3);
    constexpr uint32_t ROW_SLOTS = get_compile_time_arg_val(4);  // whole embedding rows in flight (CB 15)
    const uint32_t embed_sem_addr = get_semaphore(get_compile_time_arg_val(5));
    constexpr auto tok_args = TensorAccessorArgs<6>();
    constexpr auto table_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t tok_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t seq_len = get_arg_val<uint32_t>(argidx++);  // tokens per page of the [B, S] token ids
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t my_peer = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_xy_idx = argidx;  // n_peers (x, y) NOC coordinates follow

    const auto tok_reader = TensorAccessor(tok_args, tok_addr);
    const auto table_reader = TensorAccessor(table_args, table_addr);
    constexpr uint32_t stage_row_bytes = N_block_tiles * COL_BYTES;  // a staging row: one core's columns
    constexpr uint32_t stage_block_bytes = 32 * stage_row_bytes;     // a staging tile-row
    constexpr uint32_t table_row_bytes = N_tiles * COL_BYTES;
    CircularBuffer cb_ids(cb_ids_id), cb_stage(cb_stage_id), cb_bounce(cb_bounce_id), cb_cnt(cb_cnt_id);
    const uint32_t ids_base = cb_ids.get_write_ptr();
    const uint32_t stage_base = cb_stage.get_write_ptr();  // the same address on every core
    const uint32_t bounce_base = cb_bounce.get_write_ptr();
    const uint32_t cnt_base = cb_cnt.get_write_ptr();
    volatile tt_l1_ptr uint32_t* embed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(embed_sem_addr);
    const uint32_t n_rows = M_end_tile - M_start_tile;
    const uint32_t m_limit = M_end_tile < M_tiles ? M_end_tile : M_tiles;
    const uint32_t real_rows = m_limit > M_start_tile ? m_limit - M_start_tile : 0;
    const uint32_t n_real =
        N_tiles - my_peer * N_block_tiles < N_block_tiles ? N_tiles - my_peer * N_block_tiles : N_block_tiles;

    auto peer_addr = [&](uint32_t p, uint32_t addr) {
        return get_noc_addr(
            get_arg_val<uint32_t>(peer_xy_idx + 2 * p), get_arg_val<uint32_t>(peer_xy_idx + 2 * p + 1), addr);
    };

    for (uint32_t t = 0; t < real_rows; ++t) {
        const uint32_t r = (M_start_tile + t) * 32;
        noc.async_read(
            tok_reader,
            CoreLocalMem<uint32_t>(ids_base + t * ROW_IDS_BYTES),
            ROW_IDS_BYTES,
            {.page_id = r / seq_len, .offset_bytes = (r % seq_len) * (uint32_t)sizeof(uint32_t)},
            {});
    }
    for (uint32_t t = 0; t < n_rows; ++t) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_base + t * CNT_STRIDE) = 0;
    }
    if (n_real < N_block_tiles) {
        // Padding columns are never written: zero them so they tilize to zeros.
        fill_zeros_async(noc, cb_stage, n_rows * stage_block_bytes);
        noc.write_zeros_l1_barrier();
    }
    noc.async_read_barrier();
    cb_stage.reserve_back(n_rows * N_block_tiles);

    // Start handshake: every peer's counters are zero (and its staging cleared) before anyone writes.
    for (uint32_t p = 0; p < n_peers; ++p) {
        noc_semaphore_inc(peer_addr(p, embed_sem_addr), 1);
    }
    do {
        invalidate_l1_cache();
    } while (*embed_sem < n_peers);

    uint32_t pushed = 0;
    auto push_ready = [&]() {
        invalidate_l1_cache();
        while (pushed < n_rows &&
               *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_base + pushed * CNT_STRIDE) >= n_peers) {
            cb_stage.push_back(N_block_tiles);
            ++pushed;
        }
    };
    auto count_row = [&](uint32_t t) {
        for (uint32_t p = 0; p < n_peers; ++p) {
            noc_semaphore_inc(peer_addr(p, cnt_base + t * CNT_STRIDE), 1);
        }
    };

    // This core's tokens in order: (tile-row t, token i = my_peer + k * n_peers), read ROW_SLOTS ahead
    // of the slice writes, slot s under read transaction id 1 + s.
    const uint32_t per_row = (32 - my_peer + n_peers - 1) / n_peers;
    const uint32_t n_tok = real_rows * per_row;
    uint32_t issued = 0;
    auto issue = [&]() {
        const uint32_t t = issued / per_row, k = issued - t * per_row;
        const uint32_t id =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_base + t * ROW_IDS_BYTES)[my_peer + k * n_peers];
        const uint32_t slot = issued % ROW_SLOTS;
        noc.async_read<NocOptions::TXN_ID>(
            table_reader,
            CoreLocalMem<uint32_t>(bounce_base + slot * table_row_bytes),
            table_row_bytes,
            {.page_id = id},
            {},
            {.trid = 1 + slot});
        ++issued;
    };
    while (issued < n_tok && issued < ROW_SLOTS) {
        issue();
    }
    uint32_t done = 0;
    for (uint32_t t = 0; t < n_rows; ++t) {
        // Write transaction ids alternate by tile-row (apart from the read slots' ids), so tile-row
        // t - 1's writes can be awaited while tile-row t's are in flight.
        const uint32_t wtrid = 9 + (t & 1);
        for (uint32_t k = 0; t < real_rows && k < per_row; ++k, ++done) {
            const uint32_t slot = done % ROW_SLOTS;
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + slot});
            const uint32_t src = bounce_base + slot * table_row_bytes;
            const uint32_t dst = stage_base + t * stage_block_bytes + (my_peer + k * n_peers) * stage_row_bytes;
            noc_async_write_set_trid(wtrid);
            for (uint32_t p = 0; p < n_peers; ++p) {
                const uint32_t cols =
                    N_tiles - p * N_block_tiles < N_block_tiles ? N_tiles - p * N_block_tiles : N_block_tiles;
                noc_async_write(src + p * stage_row_bytes, peer_addr(p, dst), cols * COL_BYTES);
            }
            // The bounce slot is free once its writes have left this core.
            noc_async_writes_flushed();
            if (issued < n_tok) {
                issue();
            }
        }
        if (t > 0) {
            noc_async_write_barrier_with_trid(9 + ((t - 1) & 1));
            count_row(t - 1);
        }
        push_ready();
    }
    noc_async_write_barrier_with_trid(9 + ((n_rows - 1) & 1));
    noc_async_write_set_trid(0);
    count_row(n_rows - 1);
    while (pushed < n_rows) {
        push_ready();
    }
    noc.async_atomic_barrier();
}
