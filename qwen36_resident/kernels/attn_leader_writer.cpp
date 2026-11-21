// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader writer. Per layer and user u:
//   multicasts q to the workers;
//   takes the row max over the active workers' (bf16 compared as sign-magnitude keys) and multicasts
//   it as the global row max M;
//   stages rows 0..heads-1 of the gated output into row u of batch x 32 tiles;
// after the last user writes the staged tiles into every streamer's attention-output buffer and bumps
// their head semaphore.
//
// Runtime args: 0 workers' q buffer address, 1 M buffer address (same on the workers), 2 row-max
// slots address, 3 row-max senders (active workers incl. the tail core), 4..7 worker rectangle NOC
// x0 y0 x1 y1, 8 streamers' o buffer address, then num_streamers x (x, y), then children (group heads),
// children x (x, y), then an optional timeline buffer (0 = off; per layer words [4] q sent, [6] M sent,
// [7] output sent, next to the reader's), then the token state address.
// Rows mode (active_rows.hpp): only the active users run; the stage starts zeroed, so inactive users' output rows
// are 0. With none active the leader still releases every layer's (zero) output, once the layer's rows are in.

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"
#include "active_rows.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE uint32_t key(uint16_t v) { return (v & 0x8000) ? (~v & 0xffff) : (v | 0x8000); }

}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t M_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_slots = get_arg_val<uint32_t>(2);
    const uint32_t active = get_arg_val<uint32_t>(3);
    const uint32_t x0 = get_arg_val<uint32_t>(4), y0 = get_arg_val<uint32_t>(5);
    const uint32_t x1 = get_arg_val<uint32_t>(6), y1 = get_arg_val<uint32_t>(7);
    uint32_t o_in = get_arg_val<uint32_t>(8);  // (row groups: from streamer 0, read after the rows publish below)
    constexpr uint32_t peers_base = 9;
    const uint64_t mc_q = get_noc_multicast_addr(x0, y0, x1, y1, q_addr);
    const uint64_t mc_M = get_noc_multicast_addr(x0, y0, x1, y1, M_addr);
    const uint64_t mc_q_sem = get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(sem_q));
    const uint64_t mc_M_sem = get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(sem_M));
    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_q));
    volatile tt_l1_ptr uint32_t* M_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_M));
    volatile tt_l1_ptr uint32_t* m_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_m));
    const uint32_t heads_sem = get_semaphore(sem_heads);
    const uint32_t mc_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    auto& stage_iface = get_local_cb_interface(cb_stage);
    const uint32_t stage = stage_iface.fifo_limit - stage_iface.fifo_size;
    if constexpr (groups > 1) {
        // row groups: the mixers' row buffer (CB kCbRows, the same address on every mixer core) to every streamer
        auto& rows_iface = get_local_cb_interface(kCbRows);
        const uint32_t rows = rows_iface.fifo_limit - rows_iface.fifo_size;
        for (uint32_t s = 0; s < num_streamers; s++) {
            noc_inline_dw_write(
                get_noc_addr(
                    get_arg_val<uint32_t>(peers_base + 2 * s),
                    get_arg_val<uint32_t>(peers_base + 2 * s + 1),
                    get_semaphore(sem_raddr)),
                rows);
        }
        // and to the tail core, which reads its k / v rows from here (args 6, 7: its coordinates)
        noc_inline_dw_write(
            get_noc_addr(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_semaphore(sem_raddr)), rows);
        noc_async_write_barrier();
        // the streamers' o memory (their activation CB), from streamer 0 (which publishes before it waits for ours)
        volatile tt_l1_ptr uint32_t* oaddr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_raddr));
        while (*oaddr == 0) {
            invalidate_l1_cache();
        }
        o_in = *oaddr;
    }
    constexpr uint32_t children_base = peers_base + 2 * num_streamers;
    const uint32_t children = get_arg_val<uint32_t>(children_base);
    const uint32_t ts_addr = get_arg_val<uint32_t>(children_base + 1 + 2 * children);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    // (row groups: the timeline is per iteration instead, mark_it: 8 words per (layer, user) iteration)
    auto mark_it = [&](uint32_t it, uint32_t k) {
        if (groups > 1 && ts_addr) {
            ts[it * 8 + k] = *clk;
        }
    };
    auto mark = [&](uint32_t l, uint32_t i) {
        if (groups == 1 && ts_addr) {
            ts[l * 8 + i] = *clk;
        }
    };
    {
        // the group heads write their partial sums straight into cb_parts
        auto& parts_iface = get_local_cb_interface(cb_parts);
        const uint32_t parts = parts_iface.fifo_limit - parts_iface.fifo_size;
        for (uint32_t w = 0; w < children; w++) {
            const uint32_t wx = get_arg_val<uint32_t>(children_base + 1 + 2 * w);
            const uint32_t wy = get_arg_val<uint32_t>(children_base + 2 + 2 * w);
            noc_inline_dw_write(get_noc_addr(wx, wy, get_semaphore(sem_addr)), parts);
        }
    }

    uint32_t mask = ~0u;
    if constexpr (rows_mode) {
        const uint32_t tok_addr = get_arg_val<uint32_t>(children_base + 2 + 2 * children);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, tok_addr), stage, kTokHeadBytes);
        noc_async_read_barrier();
        mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[kTokActive];
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
        for (uint32_t w = 0; w < groups * heads * Dt * kRowTile / 4; w++) {
            p[w] = 0;
        }
    }
    // (row groups: global rows; the stage holds one heads x Dt block per group, written to o_in block g)
    const ActiveRows<kUsers> act(mask);
    const uint32_t n_act = act.n;
    // row groups: only the groups with an active row go out (the streamers skip the others' out-projection)
    const ActiveGroups ag(rows_mode ? mask : 0, batch);

    // the staged output rows into every streamer's attention-output buffer, and their head semaphore bumped
    auto layer_out = [&](uint32_t l) {
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            for (uint32_t gr = 0; gr < groups; gr++) {
                if (!ag.has(gr)) {
                    continue;
                }
                noc_async_write(
                    stage + gr * heads * Dt * kRowTile,
                    get_noc_addr(px, py, o_in + gr * o_tiles * kRowTile),
                    heads * Dt * kRowTile);
            }
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, heads_sem), 1);
        }
        mark(l, 7);
    };

    // kLeadBatched: cb_q_mc holds a pass of kQPerPass users (user j in rows j * heads + h); the user's rows go into
    // the staging q (rows 0 .. heads - 1, the rest zero) and the pass is popped after its last user
    uint32_t qstage = 0;
    if constexpr (kLeadBatched) {
        auto& qs = get_local_cb_interface(cb_qstage);
        qstage = qs.fifo_limit - qs.fifo_size;
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(qstage);
        for (uint32_t w = 0; w < Dt * kTile / 4; w++) {
            z[w] = 0;
        }
    }
    auto send_q = [&](uint32_t i) {
        cb_wait_front(cb_q_mc, Dt);
        uint32_t src = get_read_ptr(cb_q_mc);
        const uint32_t ka = i % n_act;
        if constexpr (kLeadBatched) {
            const uint32_t j = ka % kQPerPass;
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    const uint32_t from = src + d * kTile + row_offset(j * heads + h);
                    const uint32_t to = qstage + d * kTile + row_offset(h);
                    noc_async_read(get_noc_addr(from), to, kFaceRow);
                    noc_async_read(get_noc_addr(from + kFace), to + kFace, kFaceRow);
                }
            }
            noc_async_read_barrier();
            src = qstage;
        }
        noc_async_write_multicast(src, mc_q, Dt * kTile, mc_dests);
        noc_async_write_barrier();
        *q_sem = i + 1;
        noc_semaphore_set_multicast(get_semaphore(sem_q), mc_q_sem, mc_dests);
        if (!kLeadBatched || ka % kQPerPass == kQPerPass - 1 || ka + 1 == n_act) {
            cb_pop_front(cb_q_mc, Dt);
        }
    };
    // rows 0..heads-1 of user (active index) k's output into row u of the staged tiles, then (the layer's last
    // user) the layer's output to the streamers
    auto stage_out = [&](uint32_t it) {
        const uint32_t k = it % n_act, u = act.row[k];
        cb_wait_front(cb_out, Dt);
        mark_it(it, 3);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                const uint32_t src = out + d * kTile + row_offset(h);
                const uint32_t dst =
                    stage + ((u / users) * heads * Dt + h * Dt + d) * kRowTile + user_offset(u % users);
                noc_async_read(get_noc_addr(src), dst, kFaceRow);
                noc_async_read(get_noc_addr(src + kFace), dst + batch * kFaceRow, kFaceRow);
            }
        }
        noc_async_read_barrier();
        mark_it(it, 4);
        cb_pop_front(cb_out, Dt);
        if (k + 1 == n_act) {
            layer_out(it / n_act);
        }
    };
    for (uint32_t i = 0; kLeadPipelined && i < layers * n_act; i++) {
        const uint32_t k = i % n_act;
        mark_it(i, 0);
        if (k == 0) {
            send_q(i);
        }
        noc_semaphore_wait_min(m_sem, active * (i + 1));
        mark_it(i, 1);
        invalidate_l1_cache();  // the slots are written by other cores
        {
            volatile tt_l1_ptr uint16_t* out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(M_addr);
            for (uint32_t h = 0; h < qrows; h++) {
                const uint32_t at = h < 16 ? h * 16 : 256 + (h - 16) * 16;
                uint16_t best = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots)[at];
                for (uint32_t w = 1; w < active; w++) {
                    const uint16_t v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots + w * kMSlot)[at];
                    if (key(v) > key(best)) {
                        best = v;
                    }
                }
                out[h < 16 ? h * 16 : 512 + (h - 16) * 16] = best;
            }
        }
        noc_async_write_multicast(M_addr, mc_M, top_faces ? heads * kFaceRow : kTile, mc_dests);
        noc_async_write_barrier();
        *M_sem = i + 1;
        noc_semaphore_set_multicast(get_semaphore(sem_M), mc_M_sem, mc_dests);
        mark_it(i, 2);
        if (k + 1 < n_act) {
            send_q(i + 1);
        }
        // the previous user's output (its reduction ran while this user scored), and the layer's last user's own
        if (k > 0) {
            stage_out(i - 1);
        }
        if (k + 1 == n_act) {
            stage_out(i);
        }
    }
    for (uint32_t i = 0; !kLeadPipelined && i < layers * n_act; i++) {
        const uint32_t l = i / n_act, k = i % n_act, u = act.row[k];
        mark_it(i, 0);
        // q to the workers (a later user's q goes out as soon as the workers have this user's scores)
        if (k == 0) {
            send_q(i);
        }
        mark(l, 4);

        // global row max
        noc_semaphore_wait_min(m_sem, active * (i + 1));
        mark_it(i, 1);
        invalidate_l1_cache();  // the slots are written by other cores
        {
            // row h's column 0: slot (face 0 rows, then face 2 rows at 512 B) -> tile (face 2 at 1024 B)
            volatile tt_l1_ptr uint16_t* out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(M_addr);
            for (uint32_t h = 0; h < qrows; h++) {
                const uint32_t at = h < 16 ? h * 16 : 256 + (h - 16) * 16;
                uint16_t best = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots)[at];
                for (uint32_t w = 1; w < active; w++) {
                    const uint16_t v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots + w * kMSlot)[at];
                    if (key(v) > key(best)) {
                        best = v;
                    }
                }
                out[h < 16 ? h * 16 : 512 + (h - 16) * 16] = best;
            }
        }
        noc_async_write_multicast(M_addr, mc_M, top_faces ? heads * kFaceRow : kTile, mc_dests);
        noc_async_write_barrier();
        *M_sem = i + 1;
        noc_semaphore_set_multicast(get_semaphore(sem_M), mc_M_sem, mc_dests);
        mark(l, 6);
        mark_it(i, 2);
        // every worker sent its row max, so none reads q any more: the next user's q can go
        if (k + 1 < n_act) {
            send_q(i + 1);
        }

        // rows 0..heads-1 of the output into row u of the staged tiles
        cb_wait_front(cb_out, Dt);
        mark_it(i, 3);
        const uint32_t out = get_read_ptr(cb_out);
        // (row (r, h) of a pass is head h of row u * group_rows + r)
        for (uint32_t r = 0; r < group_rows; r++) {
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    const uint32_t src = out + d * kTile + row_offset(r * heads + h);
                    const uint32_t dst = stage + ((u / users) * heads * Dt + h * Dt + d) * kRowTile +
                                         user_offset((u % users) * group_rows + r);
                    noc_async_read(get_noc_addr(src), dst, kFaceRow);
                    noc_async_read(get_noc_addr(src + kFace), dst + batch * kFaceRow, kFaceRow);
                }
            }
        }
        noc_async_read_barrier();
        mark_it(i, 4);
        cb_pop_front(cb_out, Dt);
        if (k + 1 < n_act) {
            continue;
        }
        layer_out(l);
    }
    if (n_act == 0) {
        volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
        for (uint32_t l = 0; l < layers; l++) {
            noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));  // the streamers are done with o of l - 1
            layer_out(l);
        }
    }
    noc_async_atomic_barrier();
}
