// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker writer: per layer and user, sends this core's row max (rows 0..heads-1 of column 0) to the
// leader, then its partial (a group head: its partial plus its children's) to slot `slot` of its
// parent's cb_parts (the leader or a group head), top two faces of each tile only. A parent's cb_parts
// address arrives in this core's sem_addr; a group head sends its own to its children at start. The
// tail core then writes the updated tail k | v tiles back to the set's DRAM cache (off the critical path).
//
// A worker without a chunk sends -inf as its row max, once q of the layer has arrived, and its zero
// partial once M has: its compute makes the zero partial on the math and pack threads alone, so nothing
// else orders it after the layer's q and M.
//
// Runtime args: 0 role (1 chunk worker, 2 tail core), 1 row-max slot index w, 2 leader x, 3 leader y
// (NOC coordinates), 4 leader row-max slots address, 5 parent x, 6 parent y, 7 slot at the parent,
// 8 children, then children x (x, y), then an optional timeline buffer (0 = off; word [3] of the
// reader's: partial sent), then (tail core) k cache address, v cache address, per-copy stride, token
// state address, then workers - 1 x (x, y) of the chunk workers (verify: see attn_common.hpp).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t w = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2);
    const uint32_t ly = get_arg_val<uint32_t>(3);
    const uint64_t m_dst = get_noc_addr(lx, ly, get_arg_val<uint32_t>(4) + w * kMSlot);
    const uint32_t px = get_arg_val<uint32_t>(5);
    const uint32_t py = get_arg_val<uint32_t>(6);
    const uint32_t slot = get_arg_val<uint32_t>(7);
    const uint32_t children = get_arg_val<uint32_t>(8);
    const uint32_t ts_addr = get_arg_val<uint32_t>(9 + 2 * children);
    const uint32_t tail_args = 10 + 2 * children;
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    {
        auto& parts_iface = get_local_cb_interface(cb_parts);
        const uint32_t parts = parts_iface.fifo_limit - parts_iface.fifo_size;
        for (uint32_t c = 0; c < children; c++) {
            const uint32_t cx = get_arg_val<uint32_t>(9 + 2 * c);
            const uint32_t cy = get_arg_val<uint32_t>(10 + 2 * c);
            noc_inline_dw_write(get_noc_addr(cx, cy, get_semaphore(sem_addr)), parts);
        }
    }
    volatile tt_l1_ptr uint32_t* addr_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_addr));
    while (*addr_sem == 0) {
    }
    const uint32_t part_dst = *addr_sem + slot * kPart * kTile;
    const uint64_t m_sem = get_noc_addr(lx, ly, get_semaphore(sem_m));
    const uint64_t part_sem = get_noc_addr(px, py, get_semaphore(sem_part));
    const uint32_t out_cb = children > 0 ? cb_osum : cb_o;
    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_q));
    volatile tt_l1_ptr uint32_t* M_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_M));
    cb_wait_front(cb_ntiles, 1);
    bool chunk_of_user[batch];
    bool any_chunkless = false;
    for (uint32_t u = 0; u < batch; u++) {
        chunk_of_user[u] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ntiles))[u] > 0;
        any_chunkless |= !chunk_of_user[u];
    }
    // -inf rows (column 0 of rows 0..heads-1) for a user this worker has no chunk of, next to the counts
    const uint32_t neg_inf = get_read_ptr(cb_ntiles) + kNegInfOff;
    if (any_chunkless) {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neg_inf);
        for (uint32_t i = 0; i < heads * kFaceRow / 2; i++) {
            p[i] = 0xff80;
        }
    }
    uint32_t tail_bank[batch], tail_row[batch];
    uint64_t owner_sem = 0;  // verify: the tail-semaphore of the chunk worker reading row 0's tile for row 1
    if (role == 2) {
        const InterleavedAddrGen<true> tok{
            .bank_base_address = get_arg_val<uint32_t>(tail_args + 3), .page_size = 4 * (1 + batch)};
        noc_async_read(tok.get_noc_addr(0), get_write_ptr(cb_flush), 4 * (1 + batch));
        noc_async_read_barrier();
        for (uint32_t u = 0; u < batch; u++) {
            const uint32_t tp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_flush))[1 + u] / 32;
            tail_bank[u] = tp % banks;
            tail_row[u] = tp / banks;
        }
        if constexpr (verify) {
            const uint32_t tp0 = tail_row[0] * banks + tail_bank[0], tp1 = tail_row[1] * banks + tail_bank[1];
            // chunk_of's worker for row 0's tile among row 1's complete tiles
            const uint32_t live = tp1 < banks ? tp1 : banks;
            const uint32_t b = tp0 % banks;
            const uint32_t J = (workers - 1 - b + live - 1) / live;
            const uint32_t owner = b + live * (tail_row[0] % J);
            if (tp1 != tp0) {
                owner_sem = get_noc_addr(
                    get_arg_val<uint32_t>(tail_args + 4 + 2 * owner),
                    get_arg_val<uint32_t>(tail_args + 5 + 2 * owner),
                    get_semaphore(sem_tail));
            }
        }
    }
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t l = i / batch, u = i % batch;
        const bool has_chunk = chunk_of_user[u];
        if (has_chunk) {
            cb_wait_front(cb_m, 1);
            noc_async_write(get_read_ptr(cb_m), m_dst, heads * kFaceRow);
        } else {
            noc_semaphore_wait_min(q_sem, i + 1);
            noc_async_write(neg_inf, m_dst, heads * kFaceRow);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(m_sem, 1);
        if (has_chunk) {
            cb_pop_front(cb_m, 1);
        }
        if (!has_chunk) {
            noc_semaphore_wait_min(M_sem, i + 1);
        }
        cb_wait_front(out_cb, kPart);
        const uint32_t src = get_read_ptr(out_cb);
        for (uint32_t d = 0; d < kPart; d++) {
            noc_async_write(src + d * kTile, get_noc_addr(px, py, part_dst + d * kTile), 2 * kFace);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(part_sem, 1);
        cb_pop_front(out_cb, kPart);
        if (ts_addr) {
            ts[l * 4 + 3] = *clk;
        }
        if (role == 2) {
            const uint32_t copy_stride = get_arg_val<uint32_t>(tail_args + 2);
            const uint32_t copy_off = ((verify ? 0 : u) * copies + l % copies) * copy_stride + tail_row[u] * kKvRow;
            cb_wait_front(cb_flush, 2 * Dt);
            for (uint32_t kv = 0; kv < 2; kv++) {
                const uint32_t addr = get_arg_val<uint32_t>(tail_args + kv) + copy_off;
                noc_async_write(
                    get_read_ptr(cb_flush) + kv * kKvRow, get_noc_addr_from_bank_id<true>(tail_bank[u], addr), kKvRow);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_flush, 2 * Dt);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_tail)) = i + 1;
            if (owner_sem && u == 0) {
                noc_semaphore_inc(owner_sem, 1);
            }
        }
    }
    noc_async_atomic_barrier();
}
