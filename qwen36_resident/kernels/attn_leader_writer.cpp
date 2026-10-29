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
// [7] output sent, next to the reader's).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

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
    const uint32_t o_in = get_arg_val<uint32_t>(8);
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
    constexpr uint32_t children_base = peers_base + 2 * num_streamers;
    const uint32_t children = get_arg_val<uint32_t>(children_base);
    const uint32_t ts_addr = get_arg_val<uint32_t>(children_base + 1 + 2 * children);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
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

    auto send_q = [&](uint32_t i) {
        cb_wait_front(cb_q_mc, Dt);
        noc_async_write_multicast(get_read_ptr(cb_q_mc), mc_q, Dt * kTile, mc_dests);
        noc_async_write_barrier();
        *q_sem = i + 1;
        noc_semaphore_set_multicast(get_semaphore(sem_q), mc_q_sem, mc_dests);
        cb_pop_front(cb_q_mc, Dt);
    };
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t l = i / batch, u = i % batch;
        // q to the workers (a later user's q goes out as soon as the workers have this user's scores)
        if (u == 0) {
            send_q(i);
        }
        mark(l, 4);

        // global row max
        noc_semaphore_wait_min(m_sem, active * (i + 1));
        invalidate_l1_cache();  // the slots are written by other cores
        {
            volatile tt_l1_ptr uint16_t* out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(M_addr);
            for (uint32_t h = 0; h < heads; h++) {
                uint16_t best = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots)[h * 16];
                for (uint32_t w = 1; w < active; w++) {
                    const uint16_t v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_slots + w * kMSlot)[h * 16];
                    if (key(v) > key(best)) {
                        best = v;
                    }
                }
                out[h * 16] = best;
            }
        }
        noc_async_write_multicast(M_addr, mc_M, heads * kFaceRow, mc_dests);
        noc_async_write_barrier();
        *M_sem = i + 1;
        noc_semaphore_set_multicast(get_semaphore(sem_M), mc_M_sem, mc_dests);
        mark(l, 6);
        // every worker sent its row max, so none reads q any more: the next user's q can go
        if (u + 1 < batch) {
            send_q(i + 1);
        }

        // rows 0..heads-1 of the output into row u of the staged tiles
        cb_wait_front(cb_out, Dt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                const uint32_t src = out + d * kTile + row_offset(h);
                const uint32_t dst = stage + (h * Dt + d) * kRowTile + user_offset(u);
                noc_async_read(get_noc_addr(src), dst, kFaceRow);
                noc_async_read(get_noc_addr(src + kFace), dst + batch * kFaceRow, kFaceRow);
            }
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Dt);
        if (u + 1 < batch) {
            continue;
        }
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_async_write(stage, get_noc_addr(px, py, o_in), heads * Dt * kRowTile);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, heads_sem), 1);
        }
        mark(l, 7);
    }
    noc_async_atomic_barrier();
}
