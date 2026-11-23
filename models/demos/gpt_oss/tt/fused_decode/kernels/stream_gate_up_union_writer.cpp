// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + SwiGLU output scatter (BRISC, NOC1) for the union-of-experts gate|up stream (multi-row decode
// MoE, DFlash verify; fused_decode/union.py ExpertGateUpUnion).
//
// 1. U (the router's union buffer [U, ids..]) goes to the three compute threads through the TRISC mailboxes.
// 2. x: the boundary's normed hidden in the row-tile layout ([rows, 32] tiles back to back, fused_decode/boundary.py),
//    read in one piece, plus a bias tile with `nbias` ones at the start of every row.
// 3. Routing weights: the router's [rows, 32] BF16 matrix (row r, entry i: row r's weight of union expert i) becomes
//    U [rows, 32] tiles, tile i holding w[r][i] in all 32 entries of row r (compute multiplies the SwiGLU rows by it
//    through a matmul with the identity tile, also loaded here).
// 4. Output tile (slot i, pair p) -> tile (act_tile0 + p) of slot i of the activation buffer (slot page: the expert's
//    [rows, I_pad] activation as [rows, 32] tiles back to back, the down stream's in0 segment).
//
// runtime args: [x_addr, act_addr, act_tile0, weights_addr, union_addr, ident_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// the fused boundary's x-ready count: 1 (one notifier) or the multi-core boundary's helper count (direct notify)
#ifndef BND_WAIT_N
#define BND_WAIT_N 1
#endif
#include "ckernel.h"
#include "ckernel_defs.h"
#ifdef MOE_PRED
#include "moe_order.hpp"
#endif

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_tile0 = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t uni_addr = get_arg_val<uint32_t>(4);
    const uint32_t ident_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t kx = get_compile_time_arg_val(2);
    constexpr uint32_t nbias = get_compile_time_arg_val(3);
    constexpr uint32_t pairs = get_compile_time_arg_val(4);
    constexpr uint32_t rows = get_compile_time_arg_val(5);
    constexpr uint32_t act_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(9);
    constexpr uint32_t ident_bytes = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto uni_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto id_args = TensorAccessorArgs<uni_args.next_compile_time_args_offset()>();

    constexpr uint32_t tiny_bytes = 64 * rows;
    constexpr uint32_t row_face = 32 * rows;

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    // one page per [rows, 32] tile: slot s, tile t = page s * slot_tiles + t (union.py act_buffer spread)
    constexpr uint32_t slot_tiles = act_page_bytes / tiny_bytes;
    const auto s_act = TensorAccessor(act_args, act_addr, tiny_bytes);
    const auto s_w = TensorAccessor(w_args, w_addr, 64 * rows);
    const auto s_uni = TensorAccessor(uni_args, uni_addr, 128);
    const auto s_id = TensorAccessor(id_args, ident_addr, ident_bytes);

    // scratch: [union (128 B) | weights (64 rows B) | prediction (16 B)]
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
#ifdef MOE_PRED
    constexpr uint32_t P = MOE_PRED;
    uint32_t pred[P];  // runtime args 9: this layer's prediction address (hub = runtime args 7, 8)
    moe_order::read_pred<P>(
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        scr + 128 + 64 * rows,
        MOE_PRED_SEM,
        pred);
#endif
    cb_reserve_back(cb_id, 1);
    noc_async_read(s_id.get_noc_addr(0), get_write_ptr(cb_id), ident_bytes);
    cb_reserve_back(cb_x, kx + 1);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    // Fused MoE program (dflash/moe_fused.py): x once the boundary packed it (program semaphore 0), the union and
    // weights once the router wrote them (semaphore 1).
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), BND_WAIT_N);
    for (uint32_t p = 0; p * 2048 < kx * tiny_bytes; ++p) {  // page by page: x may be sharded over the helpers
        const uint32_t n = kx * tiny_bytes - p * 2048 < 2048 ? kx * tiny_bytes - p * 2048 : 2048;
        noc_async_read(s_x.get_noc_addr(p), x_l1 + p * 2048, n);
    }
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x_l1 + kx * tiny_bytes);
    for (uint32_t i = 0; i < 32 * rows; ++i) {
        ones[i] = (i < 16 * rows && i % 16 < nbias) ? 0x3F80 : 0;  // BF16 1.0
    }
#ifndef MOE_NO_ROUTER
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
#endif
    noc_async_read(s_uni.get_noc_addr(0), scr, 128);
#ifndef MOE_WSEM
    noc_async_read(s_w.get_noc_addr(0), scr + 128, 64 * rows);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_push_back(cb_x, kx + 1);
    cb_push_back(cb_id, 1);

    volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    const uint32_t num_sel = uni[0];
    uint32_t order[32];
    uint32_t kept = 0;
#ifdef MOE_PRED
    uint32_t hits = 0;
    {
        uint32_t kept_pred[P];
        kept = moe_order::make_order<P>(pred, uni, order, kept_pred);
        for (uint32_t j = 0; j < kept; ++j) {
            hits |= 1u << kept_pred[j];
        }
    }
#else
    for (uint32_t e = 0; e < num_sel; ++e) {
        order[e] = e;
    }
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, num_sel);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, num_sel);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, num_sel);
#ifdef MOE_PRED
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, hits);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, hits);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, hits);
#endif

#ifdef MOE_WSEM
    // the router releases the union first, the routing weights after (semaphore MOE_WSEM)
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MOE_WSEM)), 1);
    noc_async_read(s_w.get_noc_addr(0), scr + 128, 64 * rows);
    noc_async_read_barrier();
    invalidate_l1_cache();
#endif
    volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 128);
    cb_reserve_back(cb_sc, num_sel);
    const uint32_t sc_l1 = get_write_ptr(cb_sc);
    // scale tile i: row r = w[r][order[i]] in all 32 entries; built one expert ahead of compute (below)
    auto scale_tile = [&](uint32_t i) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc_l1 + i * tiny_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t v = w[r * 32 + order[i]];
            const uint32_t pair = v | (v << 16);
            for (uint32_t q = 0; q < 8; ++q) {
                t[r * 8 + q] = pair;                    // face 0, row r
                t[(row_face + r * 32) / 4 + q] = pair;  // face 1, row r
            }
        }
        cb_push_back(cb_sc, 1);
    };
    if (num_sel > 0) {
        scale_tile(0);
    }

    for (uint32_t e = 0; e < num_sel; ++e) {
        if (e + 1 < num_sel) {
            scale_tile(e + 1);  // while compute works on expert e
        }
        for (uint32_t p = 0; p < pairs; ++p) {
            cb_wait_front(cb_act, 1);
            const uint32_t src = get_read_ptr(cb_act);
            noc_async_write(src, s_act.get_noc_addr(order[e] * slot_tiles + act_tile0 + p), tiny_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_act, 1);
        }
        // slot e's columns of this core are in DRAM: count them in the router core's counter e (runtime args 6..8:
        // counters address, router noc x, y); the down cores start on slot e once every gate|up core has.
        noc_async_write_barrier();
#ifdef MOE_CTR_SEM
        // counters = program semaphores MOE_CTR_SEM + e of the hub core (reset at every launch)
        const uint32_t ctr_e = get_semaphore(MOE_CTR_SEM + order[e]);
#else
        const uint32_t ctr_e = get_arg_val<uint32_t>(6) + 16 * order[e];
#endif
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8), ctr_e), 1);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
