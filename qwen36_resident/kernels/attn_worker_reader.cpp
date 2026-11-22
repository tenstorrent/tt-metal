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
// [copies][Dt] bf16 tiles), then (paged KV, attn_paged.hpp; arg 9 on a chunk worker, 11 on the tail core) the
// page table address. Paged: 2 / 3 are the K / V pools and 4 is 0.
// Rows mode (active_rows.hpp): only the active users run (iteration i = layer * n + k for the k-th of the n);
// the active mask goes to the writer and compute in word `batch` of the cb_ntiles page.

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"
#include "attn_paged.hpp"
#include "active_rows.hpp"
#include "attn_rows.hpp"

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

// (row groups: the tail core reads its k / v rows from the leader's row buffer, which it has no copy of)
uint32_t rows_x = 0, rows_y = 0;
FORCE_INLINE uint64_t row_noc(uint32_t addr) {
    return groups > 1 ? get_noc_addr(rows_x, rows_y, addr) : get_noc_addr(addr);
}

// copy a user's face rows of a batch x 32 tile (src: its face 0 row) into row `row` of a 32x32 bf16 tile
FORCE_INLINE void place_row(uint32_t src, uint32_t tile, uint32_t row) {
    noc_async_read(row_noc(src), tile + row_offset(row), kFaceRow);
    noc_async_read(row_noc(src + batch * kFaceRow), tile + kFace + row_offset(row), kFaceRow);
}

FORCE_INLINE void place_row0(uint32_t src, uint32_t tile) { place_row(src, tile, 0); }

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
uint32_t tail_u;  // (kTailBatched: the user whose placement matrix is being filled)

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
    const InterleavedAddrGen<true> tok{
        .bank_base_address = tok_addr, .page_size = rope_off + 3 * batch * groups * kTile};
    cb_reserve_back(cb_ntiles, 1);
    const uint32_t scratch = get_write_ptr(cb_ntiles);
    noc_async_read(tok.get_noc_addr(0), scratch, rows_mode ? kTokHeadBytes : 4 * (1 + batch));
    // (row groups: the users' positions are the token state's positions region, staged past the -inf rows)
    // (verify rows: past the -inf rows of a 24-row pass, kMRows0 = 512)
    constexpr uint32_t kPosScratch = kVRows ? 768 : 512;
    if constexpr (groups > 1) {
        noc_async_read(tok.get_noc_addr(0, tok_pos_off), scratch + kPosScratch, (4 * kRows + 31) & ~31u);
    }
    noc_async_read_barrier();
    // (verify rows: a bit per pass, attn_common.hpp)
    const uint32_t mask =
        pass_mask(rows_mode ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[kTokActive] : ~0u);
    const ActiveRows<kUsers> act(mask);
    Chunk chs[kRows];
    uint32_t tail_rs[kRows], tps[kRows];
    // (kTails: whether this tail core takes row u — by its active index)
    bool takes[kRows];
    for (uint32_t u = 0; u < kRows; u++) {
        takes[u] = kTails == 1;
    }
    for (uint32_t k = 0; kTails > 1 && role >= 2 && k < act.n; k++) {
        takes[act.row[k]] = tail_takes(role, k);
    }
    for (uint32_t u = 0; u < kRows; u++) {
        const uint32_t pos = groups == 1 ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[1 + u]
                                         : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kPosScratch)[u];
        const uint32_t tp = pos / 32;
        tps[u] = tp;
        tail_rs[u] = pos % 32;
        chs[u] = role >= 2 ? (takes[u] ? Chunk{tp, 1, 1} : Chunk{0, 1, 0}) : chunk_of(w, tp);
    }
    // packed: every pass takes the tiles below its user's row 0's; the tail core that tile and the next one if a row
    // of the user reaches it (verify rows: user gr's rows gr * batch ..; pass u is user u / users's)
    auto n_tail_of = [&](uint32_t gr) -> uint32_t {
        return packed && tps[gr * batch + batch - 1] != tps[gr * batch] ? 2 : 1;
    };
    for (uint32_t u = 0; packed && u < kUsers; u++) {
        const uint32_t r0 = (u / users) * batch;
        chs[u] = role == 2 ? Chunk{tps[r0], 1, n_tail_of(u / users)} : chunk_of(w, tps[r0]);
    }
    if constexpr (kv_paged) {
        // the pages of this core's tiles (the writer's tail pages too: it reads them after the cb_ntiles push)
        const uint32_t pt_addr = get_arg_val<uint32_t>(role >= 2 ? 11 : 9);
        for (uint32_t k = 0; k < act.n; k++) {
            const uint32_t u = act.row[k];
            translate_chunk(pt_addr, kVRows ? (u / users) * batch : (verify ? 0 : u), u, chs[u], chs[u].n);
        }
    }
    for (uint32_t u = 0; u < kUsers; u++) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[u] = chs[u].n;
    }
    // (kSkipZeroParts, a group head: per active user the children with KV tiles — chunk worker c: its chunk; a tail:
    // one tile when it takes the user — children are the next workers w + 1 .. w + children)
    static_assert(!kSkipZeroParts || kChildMaskOff + 4 * kUsers <= (groups == 1 ? 640 : 1088), "cb_ntiles page");
    static_assert(groups == 1 || kNegInfOff + kMRows0 <= kPosScratch, "-inf rows overlap the positions");
    for (uint32_t k = 0; kSkipZeroParts && children > 0 && k < act.n; k++) {
        const uint32_t u = act.row[k];
        uint32_t cm = 0;
        for (uint32_t c = 0; c < children; c++) {
            const uint32_t cw = w + 1 + c;
            const bool tiles = cw < kChunkWorkers ? chunk_of(cw, tps[u]).n > 0 : tail_takes(2 + (workers - 1 - cw), k);
            cm |= (tiles ? 1u : 0u) << c;
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kChildMaskOff)[u] = cm;
    }
    static_assert((kMaskWord + 1) * 4 <= kNegInfOff, "the active mask overlaps the -inf rows");
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[kMaskWord] = mask;
    cb_push_back(cb_ntiles, 1);

    uint32_t rows = 0;
    const InterleavedAddrGenFast<true> kw_dram{
        .bank_base_address = role >= 2 ? get_arg_val<uint32_t>(10) : 0,
        .page_size = kTile,
        .data_format = DataFormat::Float16_b};
    if (role >= 2) {
        rows = get_arg_val<uint32_t>(9);
        if constexpr (groups > 1) {
            // row groups: the leader's row buffer (its address from the leader, its NOC coordinates the last args)
            rows_x = get_arg_val<uint32_t>(12);
            rows_y = get_arg_val<uint32_t>(13);
            volatile tt_l1_ptr uint32_t* raddr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_raddr));
            while (*raddr == 0) {
                invalidate_l1_cache();
            }
            rows = *raddr;
        }
        for (uint32_t cb : {cb_kraw, cb_vraw}) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb));
            for (uint32_t i = 0; i < Dt * kTile / 4; i++) {
                p[i] = 0;
            }
        }
        if constexpr (packed) {
            // per pass and tail tile j: row (r, h) sees column c up to its row's position; per tail tile j: the
            // placement matrix of the new rows on it (row pos % 32 <- row r) and the rows it keeps
            // (verify rows: masks 2 ug + j per pass ug, placement / keep 4 gr + 2 j (+ 1) per user gr)
            cb_reserve_back(cb_mask, 2 * kUsers);
            cb_reserve_back(cb_rowsel, 4 * groups);
            volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_mask));
            volatile tt_l1_ptr uint16_t* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_rowsel));
            for (uint32_t row = 0; row < 32; row++) {
                for (uint32_t col = 0; col < 32; col++) {
                    const uint32_t at = ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
                    for (uint32_t j = 0; j < 2; j++) {
                        for (uint32_t ug = 0; ug < kUsers; ug++) {
                            const uint32_t r0 = (ug / users) * batch, r = r0 + (ug % users) * group_rows + row / heads;
                            const bool seen = row >= qrows || (tps[r0] + j) * 32 + col <= tps[r] * 32 + tail_rs[r];
                            m[(2 * ug + j) * 1024 + at] = seen ? 0 : 0xff80;
                        }
                        for (uint32_t gr = 0; gr < groups; gr++) {
                            const uint32_t r0 = gr * batch;
                            bool placed = false, kept = true;
                            for (uint32_t r = 0; r < batch; r++) {
                                const bool here = tps[r0 + r] == tps[r0] + j && tail_rs[r0 + r] == row;
                                placed |= here && col == r;
                                kept &= !here;
                            }
                            sel[(4 * gr + 2 * j) * 1024 + at] = placed ? 0x3f80 : 0;
                            sel[(4 * gr + 2 * j + 1) * 1024 + at] = kept ? 0x3f80 : 0;
                        }
                    }
                }
            }
            cb_push_back(cb_mask, 2 * kUsers);
            cb_push_back(cb_rowsel, 4 * groups);
            cb_reserve_back(cb_rope, kRopePages);
            noc_async_read(tok.get_noc_addr(0, rope_off), get_write_ptr(cb_rope), 3 * batch * groups * kTile);
            noc_async_read_barrier();
            build_packed_rope(get_write_ptr(cb_rope));
            cb_push_back(cb_rope, kRopePages);
        }
        // per user: causal mask over the tail columns (0 up to r, -inf after); R = ones in row r, 1 - R
        cb_reserve_back(cb_mask, packed ? 0 : kRows);
        cb_reserve_back(cb_rowsel, packed ? 0 : 2 * kRows);
        for (uint32_t u = 0; !packed && u < kRows; u++) {
            tail_r = tail_rs[u];
            fill_bf16(
                get_write_ptr(cb_mask) + u * kTile, [](uint32_t, uint32_t col) { return col <= tail_r; }, 0, 0xff80);
            const uint32_t sel = get_write_ptr(cb_rowsel) + 2 * u * kTile;
            if constexpr (kTailBatched) {
                // P_u: row u of the batched new rows into row r of the tail tile; keep: every other row
                tail_u = u;
                fill_bf16(sel, [](uint32_t row, uint32_t col) { return row == tail_r && col == tail_u; }, 0x3f80, 0);
            } else {
                fill_bf16(sel, [](uint32_t row, uint32_t) { return row == tail_r; }, 0x3f80, 0);
            }
            fill_bf16(sel + kTile, [](uint32_t row, uint32_t) { return row == tail_r; }, 0, 0x3f80);
        }
        if constexpr (!packed) {
            cb_push_back(cb_mask, kRows);
            cb_push_back(cb_rowsel, 2 * kRows);
            cb_reserve_back(cb_rope, kRopeTail);
            noc_async_read(tok.get_noc_addr(0, rope_off), get_write_ptr(cb_rope), 3 * batch * groups * kTile);
            noc_async_read_barrier();
            if constexpr (kTailBatched) {
                build_rows_rope(get_write_ptr(cb_rope));
            }
            cb_push_back(cb_rope, kRopeTail);
        }
        push_resident(cb_mean, 1);
    }

    // two_nocs: a chunk worker's KV reads alternate between the NOCs (the other one's counters start here)
    const bool dual = two_nocs && role == 1;
    const uint8_t other_noc = 1 - noc_index;
    if (dual) {
        noc_local_state_init(other_noc);
    }
    // tiles from, from + 1, ... of chunk ch (tile t: row t / banks of bank t % banks; paged: page copy_page +
    // pages[k] of the pool at addr, attn_paged.hpp), one kv_row each
    auto read_rows = [&](const Chunk& ch,
                         uint32_t addr,
                         uint32_t from,
                         uint32_t n,
                         uint32_t l1,
                         volatile tt_l1_ptr uint32_t* pages,
                         uint32_t copy_page) {
        for (uint32_t k = from; k < from + n; k++, l1 += kKvRow) {
            const uint8_t noc = dual && (k & 1) ? other_noc : noc_index;
            uint64_t src;
            if constexpr (kv_paged) {
                src = page_noc_addr(addr, copy_page + pages[k], noc);
            } else {
                const uint32_t t = ch.t0 + k * ch.dt;
                src = get_noc_addr_from_bank_id<true>(t % banks, addr + (t / banks) * kKvRow, noc);
            }
            noc_async_read(src, l1, kKvRow, noc);
        }
    };

    constexpr uint32_t k0 = 2 * heads * Dt, v0 = k0 + Dt;
    const uint32_t user_stride = verify ? 0 : copies * copy_stride;
    // verify: rows from `cross` on (the first row past row 0's tile) have that tile among their complete tiles;
    // whether this worker's chunk of theirs holds it
    uint32_t cross = batch;
    bool waits_tail = false;
    if constexpr (verify) {
        for (uint32_t u = batch - 1; u > 0; u--) {
            cross = tps[u] != tps[0] ? u : cross;
        }
        waits_tail = !packed && role == 1 && cross < batch && holds(chs[cross], tps[0]);
    }
    const uint32_t n_act = act.n;
    for (uint32_t i = 0; i < layers * n_act; i++) {
        const uint32_t l = i / n_act, ka = i % n_act, u = act.row[ka];
        const Chunk& ch = chs[u];
        const uint32_t k_addr = k_cache + u * user_stride + (l % copies) * copy_stride;
        const uint32_t v_addr = v_cache + u * user_stride + (l % copies) * copy_stride;
        volatile tt_l1_ptr uint32_t* pages = kv_paged ? page_list(u) : nullptr;
        const uint32_t copy_page = (l % copies) * kv_copy_pages;
        // blocks of chunk_max rows; the ring holds two, so a block is read while the previous one is used
        auto read_blocks = [&](uint32_t from, uint32_t to) {
            for (uint32_t b0 = from; b0 < to; b0 += chunk_max) {
                const uint32_t nb = ch.n - b0 < chunk_max ? ch.n - b0 : chunk_max;
                cb_reserve_back(cb_k, chunk_max * Dt);
                cb_reserve_back(cb_v, chunk_max * Dt);
                read_rows(ch, k_addr, b0, nb, get_write_ptr(cb_k), pages, copy_page);
                read_rows(ch, v_addr, b0, nb, get_write_ptr(cb_v), pages, copy_page);
                noc_async_read_barrier();
                if (dual) {
                    noc_async_read_barrier(other_noc);
                }
                cb_push_back(cb_k, chunk_max * Dt);
                cb_push_back(cb_v, chunk_max * Dt);
            }
        };
        // the blocks that fit the ring are prefetched before q; the rest need compute to free slots
        const uint32_t early = ch.n < 2 * chunk_max ? ch.n : 2 * chunk_max;
        volatile tt_l1_ptr uint32_t* tail_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_tail));
        if (role == 1) {
            if (waits_tail && u == cross) {
                noc_semaphore_wait_min(tail_sem, l + 1);
            }
            read_blocks(0, early);
        } else if (packed) {
            // once per layer: the tail tiles, the k norm weight and every row's k / v (rows 0 .. batch - 1)
            // (verify rows: per user, at its first pass; its rows in its group's block of the row buffer)
            const uint32_t n_tail = n_tail_of(u / users);
            if (u % users == 0) {
                cb_reserve_back(cb_tail8, 2 * n_tail * Dt);
                read_rows(ch, k_addr, 0, n_tail, get_write_ptr(cb_tail8), pages, copy_page);
                read_rows(ch, v_addr, 0, n_tail, get_write_ptr(cb_tail8) + n_tail * kKvRow, pages, copy_page);
                cb_reserve_back(cb_kw, Dt);
                for (uint32_t d = 0; d < Dt; d++) {
                    noc_async_read_page((l % copies) * Dt + d, kw_dram, get_write_ptr(cb_kw) + d * kTile);
                }
                noc_async_read_barrier();
                cb_push_back(cb_tail8, 2 * n_tail * Dt);
                cb_push_back(cb_kw, Dt);
                noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
                cb_reserve_back(cb_kraw, Dt);
                cb_reserve_back(cb_vraw, Dt);
                const uint32_t k = get_write_ptr(cb_kraw), v = get_write_ptr(cb_vraw);
                const uint32_t grows = rows + (u / users) * row_tiles * kRowTile;
                for (uint32_t r = 0; r < batch; r++) {
                    for (uint32_t d = 0; d < Dt; d++) {
                        place_row(grows + (k0 + d) * kRowTile + user_offset(r), k + d * kTile, r);
                        place_row(grows + (v0 + d) * kRowTile + user_offset(r), v + d * kTile, r);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_kraw, Dt);
                cb_push_back(cb_vraw, Dt);
            }
        } else {
            // (kTails: only for the users this tail takes; the others run as a chunk worker without a chunk)
            const bool mine = takes[u];
            // the user's tail of this copy was written back by its previous use (verify: by row u - 1, same tile)
            if (!mine) {
            } else if (verify && u > 0 && tps[u] == tps[u - 1]) {
                noc_semaphore_wait_min(tail_sem, i);
            } else if (l >= copies) {
                noc_semaphore_wait_min(tail_sem, i - copies * n_act + 1);
            }
            if (mine) {
                cb_reserve_back(cb_tail8, 2 * Dt);
                read_rows(ch, k_addr, 0, 1, get_write_ptr(cb_tail8), pages, copy_page);
                read_rows(ch, v_addr, 0, 1, get_write_ptr(cb_tail8) + kKvRow, pages, copy_page);
            }
            // (kTailBatched: the k norm weight and every user's new rows once per layer)
            const bool per_user = !kTailBatched || ka == 0;
            if (per_user) {
                cb_reserve_back(cb_kw, Dt);
                for (uint32_t d = 0; d < Dt; d++) {
                    noc_async_read_page((l % copies) * Dt + d, kw_dram, get_write_ptr(cb_kw) + d * kTile);
                }
            }
            noc_async_read_barrier();
            if (mine) {
                cb_push_back(cb_tail8, 2 * Dt);
            }
            if (per_user) {
                cb_push_back(cb_kw, Dt);
            }
            if (ka == 0) {
                noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
            }
            if constexpr (kTailBatched) {
                if (ka == 0) {
                    // every active user's k / v row into row uu (its global row) of one tile set
                    cb_reserve_back(cb_kraw, Dt);
                    cb_reserve_back(cb_vraw, Dt);
                    const uint32_t k = get_write_ptr(cb_kraw), v = get_write_ptr(cb_vraw);
                    for (uint32_t a2 = 0; a2 < n_act; a2++) {
                        const uint32_t uu = act.row[a2];
                        const uint32_t grows = rows + (uu / batch) * row_tiles * kRowTile;
                        for (uint32_t d = 0; d < Dt; d++) {
                            place_row(grows + (k0 + d) * kRowTile + user_offset(uu % batch), k + d * kTile, uu);
                            place_row(grows + (v0 + d) * kRowTile + user_offset(uu % batch), v + d * kTile, uu);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_kraw, Dt);
                    cb_push_back(cb_vraw, Dt);
                }
            } else {
                cb_reserve_back(cb_kraw, Dt);
                cb_reserve_back(cb_vraw, Dt);
                const uint32_t k = get_write_ptr(cb_kraw), v = get_write_ptr(cb_vraw);
                // (row groups: user u's rows in block u / batch of the row buffer, row u % batch)
                const uint32_t grows = rows + (u / batch) * row_tiles * kRowTile;
                for (uint32_t d = 0; d < Dt; d++) {
                    place_row0(grows + (k0 + d) * kRowTile + user_offset(u % batch), k + d * kTile);
                    place_row0(grows + (v0 + d) * kRowTile + user_offset(u % batch), v + d * kTile);
                }
                noc_async_read_barrier();
                cb_push_back(cb_kraw, Dt);
                cb_push_back(cb_vraw, Dt);
            }
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
