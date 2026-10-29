// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker reader. Reads the users' positions from the token state; per attention layer a (copy
// c = a % copies) and user u (its own KV cache):
//   chunk worker: reads its chunk of copy c's KV cache (complete position tiles; see attn_common.hpp) in
//     blocks of chunk_max tiles;
//   tail core: prefetches copy c's k | v row of the position tile holding the position and the
//     layer's k norm weight, waits for the streamers' projection row and places k and v into row 0 of
//     cb_kraw / cb_vraw;
// then releases the leader's multicast q and global row max, and a group head's children's partials,
// to compute.
//
// Runtime args: 0 role (1 chunk worker, 2 tail core), 1 chunk worker index, 2 k cache address,
// 3 v cache address, 4 per-copy stride (bytes per bank), 5 children, 6 optional timeline buffer
// (0 = off; per layer 4 wall-clock words: [0] KV read, [1] q arrived, [2] M arrived), 7 token state
// address, 8 rope offset in it, then (tail core) 9 row buffer address, 10 k norm weight address (DRAM,
// [copies][Dt] bf16 tiles).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

FORCE_INLINE void push_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

// copy a user's face rows of a batch x 32 tile (src: its face 0 row) into row 0 of a 32x32 bf16 tile
FORCE_INLINE void place_row0(uint32_t src, uint32_t tile) {
    noc_async_read(get_noc_addr(src), tile, kFaceRow);
    noc_async_read(get_noc_addr(src + batch * kFaceRow), tile + kFace, kFaceRow);
}

// a bf16 32x32 tile with `on` where one(row, col) holds and `off` elsewhere
FORCE_INLINE void fill_bf16(uint32_t addr, bool (*one)(uint32_t, uint32_t), uint16_t on, uint16_t off) {
    volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            m[((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16] = one(row, col) ? on : off;
        }
    }
}

uint32_t tail_r;

}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t w = get_arg_val<uint32_t>(1);
    const uint32_t k_cache = get_arg_val<uint32_t>(2);
    const uint32_t v_cache = get_arg_val<uint32_t>(3);
    const uint32_t copy_stride = get_arg_val<uint32_t>(4);
    const uint32_t children = get_arg_val<uint32_t>(5);
    const uint32_t ts_addr = get_arg_val<uint32_t>(6);
    const uint32_t tok_addr = get_arg_val<uint32_t>(7);
    const uint32_t rope_off = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_q));
    volatile tt_l1_ptr uint32_t* M_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_M));
    volatile tt_l1_ptr uint32_t* part_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 4 + i] = *clk;
        }
    };
    push_resident(cb_one, 1);

    // positions -> this core's KV rows per user
    const InterleavedAddrGen<true> tok{.bank_base_address = tok_addr, .page_size = rope_off + 3 * batch * kTile};
    cb_reserve_back(cb_ntiles, 1);
    const uint32_t scratch = get_write_ptr(cb_ntiles);
    noc_async_read(tok.get_noc_addr(0), scratch, 4 * (1 + batch));
    noc_async_read_barrier();
    Chunk chs[batch];
    uint32_t tail_rs[batch], tps[batch];
    for (uint32_t u = 0; u < batch; u++) {
        const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[1 + u];
        const uint32_t tp = pos / 32;
        tps[u] = tp;
        tail_rs[u] = pos % 32;
        chs[u] = role == 2 ? Chunk{tp % banks, tp / banks, 1, 1} : chunk_of(w, tp);
    }
    for (uint32_t u = 0; u < batch; u++) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[u] = chs[u].n;
    }
    cb_push_back(cb_ntiles, 1);

    uint32_t rows = 0;
    const InterleavedAddrGenFast<true> kw_dram{
        .bank_base_address = role == 2 ? get_arg_val<uint32_t>(10) : 0,
        .page_size = kTile,
        .data_format = DataFormat::Float16_b};
    if (role == 2) {
        rows = get_arg_val<uint32_t>(9);
        for (uint32_t cb : {cb_kraw, cb_vraw}) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb));
            for (uint32_t i = 0; i < Dt * kTile / 4; i++) {
                p[i] = 0;
            }
        }
        // per user: causal mask over the tail columns (0 up to r, -inf after); R = ones in row r, 1 - R
        cb_reserve_back(cb_mask, batch);
        cb_reserve_back(cb_rowsel, 2 * batch);
        for (uint32_t u = 0; u < batch; u++) {
            tail_r = tail_rs[u];
            fill_bf16(
                get_write_ptr(cb_mask) + u * kTile, [](uint32_t, uint32_t col) { return col <= tail_r; }, 0, 0xff80);
            const uint32_t sel = get_write_ptr(cb_rowsel) + 2 * u * kTile;
            fill_bf16(sel, [](uint32_t row, uint32_t) { return row == tail_r; }, 0x3f80, 0);
            fill_bf16(sel + kTile, [](uint32_t row, uint32_t) { return row == tail_r; }, 0, 0x3f80);
        }
        cb_push_back(cb_mask, batch);
        cb_push_back(cb_rowsel, 2 * batch);
        cb_reserve_back(cb_rope, 3 * batch);
        noc_async_read(tok.get_noc_addr(0, rope_off), get_write_ptr(cb_rope), 3 * batch * kTile);
        noc_async_read_barrier();
        cb_push_back(cb_rope, 3 * batch);
        push_resident(cb_mean, 1);
    }

    // rows from, from + 1, ... of chunk ch (shard rows j + k J of the bank), one kv_row each
    auto read_rows = [&](const Chunk& ch, uint32_t addr, uint32_t from, uint32_t n, uint32_t l1) {
        for (uint32_t k = from; k < from + n; k++, l1 += kKvRow) {
            noc_async_read(get_noc_addr_from_bank_id<true>(ch.bank, addr + (ch.j + k * ch.J) * kKvRow), l1, kKvRow);
        }
    };

    constexpr uint32_t k0 = 2 * heads * Dt, v0 = k0 + Dt;
    const uint32_t user_stride = verify ? 0 : copies * copy_stride;
    // verify, p the last row of its tile: whether row 1's chunk of this worker holds that tile
    bool waits_tail = false, same_tile = false;
    if constexpr (verify) {
        const Chunk& c1 = chs[1];
        const uint32_t row = tps[0] / banks;
        same_tile = tps[1] == tps[0];
        waits_tail = role == 1 && !same_tile && c1.n > 0 && c1.bank == tps[0] % banks && row >= c1.j &&
                     (row - c1.j) % c1.J == 0 && (row - c1.j) / c1.J < c1.n;
    }
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t l = i / batch, u = i % batch;
        const Chunk& ch = chs[u];
        const uint32_t k_addr = k_cache + u * user_stride + (l % copies) * copy_stride;
        const uint32_t v_addr = v_cache + u * user_stride + (l % copies) * copy_stride;
        // blocks of chunk_max rows; the ring holds two, so a block is read while the previous one is used
        auto read_blocks = [&](uint32_t from, uint32_t to) {
            for (uint32_t b0 = from; b0 < to; b0 += chunk_max) {
                const uint32_t nb = ch.n - b0 < chunk_max ? ch.n - b0 : chunk_max;
                cb_reserve_back(cb_k, chunk_max * Dt);
                cb_reserve_back(cb_v, chunk_max * Dt);
                read_rows(ch, k_addr, b0, nb, get_write_ptr(cb_k));
                read_rows(ch, v_addr, b0, nb, get_write_ptr(cb_v));
                noc_async_read_barrier();
                cb_push_back(cb_k, chunk_max * Dt);
                cb_push_back(cb_v, chunk_max * Dt);
            }
        };
        // the blocks that fit the ring are prefetched before q; the rest need compute to free slots
        const uint32_t early = ch.n < 2 * chunk_max ? ch.n : 2 * chunk_max;
        volatile tt_l1_ptr uint32_t* tail_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_tail));
        if (role == 1) {
            if (waits_tail && u == 1) {
                noc_semaphore_wait_min(tail_sem, l + 1);
            }
            read_blocks(0, early);
        } else {
            // the user's tail of this copy was written back by its previous use (verify: by row 0, same tile)
            if (same_tile && u == 1) {
                noc_semaphore_wait_min(tail_sem, i);
            } else if (l >= copies) {
                noc_semaphore_wait_min(tail_sem, i - copies * batch + 1);
            }
            cb_reserve_back(cb_tail8, 2 * Dt);
            read_rows(ch, k_addr, 0, 1, get_write_ptr(cb_tail8));
            read_rows(ch, v_addr, 0, 1, get_write_ptr(cb_tail8) + kKvRow);
            cb_reserve_back(cb_kw, Dt);
            for (uint32_t d = 0; d < Dt; d++) {
                noc_async_read_page((l % copies) * Dt + d, kw_dram, get_write_ptr(cb_kw) + d * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_tail8, 2 * Dt);
            cb_push_back(cb_kw, Dt);
            if (u == 0) {
                noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
            }
            cb_reserve_back(cb_kraw, Dt);
            cb_reserve_back(cb_vraw, Dt);
            const uint32_t k = get_write_ptr(cb_kraw), v = get_write_ptr(cb_vraw);
            for (uint32_t d = 0; d < Dt; d++) {
                place_row0(rows + (k0 + d) * kRowTile + user_offset(u), k + d * kTile);
                place_row0(rows + (v0 + d) * kRowTile + user_offset(u), v + d * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_kraw, Dt);
            cb_push_back(cb_vraw, Dt);
        }
        mark(l, 0);
        noc_semaphore_wait_min(q_sem, i + 1);
        mark(l, 1);
        push_resident(cb_q, Dt);
        if (role == 1) {
            read_blocks(early, ch.n);
        }
        noc_semaphore_wait_min(M_sem, i + 1);
        mark(l, 2);
        push_resident(cb_M, 1);
        if (children > 0) {
            noc_semaphore_wait_min(part_sem, children * (i + 1));
            push_resident(cb_parts, fanin * kPart);
        }
    }
}
