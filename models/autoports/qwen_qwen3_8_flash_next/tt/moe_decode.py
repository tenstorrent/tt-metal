# SPDX-License-Identifier: Apache-2.0
"""Few-token routed experts: every selected local expert at once.

The prefill expert op walks one expert at a time over the whole grid. For a
decode batch of up to MAX_ROWS tokens each rank instead computes the union of
its selected experts in two generic programs, reading each selected expert's
weights once. Token t occupies row t of every tile:

1. gate/up: one (local expert, 32-column tile) item per core;
   H[t] = score[t] * SiLU(x[t] @ W_gate) * (x[t] @ W_up), where score[t] is 0
   for a token that did not select the expert.
2. down: one 32-column output tile per core; out[t] = sum_experts H[t] @ W_down.

Expert ids and scores stay on device. Each expert tensor keeps its own DRAM
buffer; a replicated table holds their (mesh-identical) base addresses. Only
rows below the token count are meaningful; phase 2 keeps its input padding rows
zero so its output is exactly zero below them. FP32 destination accumulation.
Measured against an FP32 host evaluation of the same BF4 experts: 0.992x
(packer truncation); the prefill expert op path measures 1.04x.

Diagnostics: MOE_EXP=noread,nomath compiles out weight reads / matmuls.
"""

import os

import numpy as np
import torch

import ttnn

_TILE_BF16 = 2048
_TILE_BFP4 = 576
MAX_ROWS = 4
_META = 4096  # ids, scores, rank and the 1536-byte address table for MAX_ROWS tokens
_HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Union of the ROWS tokens' selected local experts, first-seen order, with each
// token's score (0 when not selected). meta: ids[ROWS][32] u32 @0,
// scores[ROWS][32] bf16 @IDS_END, rank u32 @RANK, table[384] u32 @TABLE.
constexpr uint32_t MAXE = 10 * ROWS;
constexpr uint32_t SC_OFF = 128 * ROWS, RANK_OFF = SC_OFF + 64 * ROWS, TABLE_OFF = RANK_OFF + 64;
template <typename I, typename S, typename R, typename T>
uint32_t local_slots(const Noc& noc, const I& ids, const S& scores, const R& rank_tensor, const T& table,
                     uint32_t meta, uint32_t* expert, uint16_t (*score)[ROWS]) {
    for (uint32_t t = 0; t < ROWS; ++t) {
        noc.async_read(ids, CoreLocalMem<uint32_t>(meta + t * 128), 128, {.page_id=t}, {});
        noc.async_read(scores, CoreLocalMem<uint32_t>(meta + SC_OFF + t * 64), 64, {.page_id=t}, {});
    }
    noc.async_read(rank_tensor, CoreLocalMem<uint32_t>(meta + RANK_OFF), 4, {.page_id=0}, {});
    noc.async_read(table, CoreLocalMem<uint32_t>(meta + TABLE_OFF), 1536, {.page_id=0}, {});
    noc.async_read_barrier();
    auto id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    auto sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta + SC_OFF);
    const uint32_t rank = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + RANK_OFF);
    uint32_t count = 0;
    for (uint32_t t = 0; t < ROWS; ++t) {
        for (uint32_t k = 0; k < 10; ++k) {
            const uint32_t e = id[t * 32 + k];
            const uint16_t v = sc[t * 32 + k];
            if (e / 128 != rank || v == 0) continue;
            uint32_t slot = 0;
            while (slot < count && expert[slot] != e % 128) ++slot;
            if (slot == count) {
                expert[count] = e % 128;
                for (uint32_t r = 0; r < ROWS; ++r) score[count][r] = 0;
                ++count;
            }
            score[slot][t] = v;
        }
    }
    return count;
}
// Workers form a 10-wide rectangle at the grid origin; the index is device-derived
// so every core shares one common runtime-argument list.
inline uint32_t worker_index() { return get_absolute_logical_y() * 10 + get_absolute_logical_x(); }
inline uint32_t table_entry(uint32_t meta, uint32_t index) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + TABLE_OFF)[index];
}
// The shared expert (5 x 32 columns per rank, BF16 weights) runs as SH extra items;
// its per-row weight is sigmoid(shared gate logit), column 0 of each gate row.
#ifndef SHARED_ITEMS
#define SHARED_ITEMS 5
#endif
// SH items run; the weight layout always holds 5 gate then 5 up column tiles per K row.
constexpr uint32_t SH = SHARED_ITEMS, SHARED_COLS = 5, GATE_OFF = TABLE_OFF + 1536;
template <typename G>
void shared_gate(const Noc& noc, const G& gate, uint32_t meta, uint16_t* weight) {
    for (uint32_t t = 0; t < ROWS; ++t) {
        noc.async_read(gate, CoreLocalMem<uint32_t>(meta + GATE_OFF + t * 64), 64, {.page_id=t}, {});
    }
    noc.async_read_barrier();
    for (uint32_t t = 0; t < ROWS; ++t) weight[t] = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta + GATE_OFF + t * 64)[0];
}
"""

# ---------------------------------------------------------------- phase 1
_GU_READER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto xa = TensorAccessorArgs<4>();
    constexpr auto ia = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const auto gate_logit = TensorAccessor(ga, get_common_arg_val<uint32_t>(5));
    const auto shared_gu = TensorAccessor(qa, get_common_arg_val<uint32_t>(6));
    const uint32_t worker = worker_index();
    Noc noc;
    CircularBuffer cx(0), cw(1), cs(2), cc(3), cm(6), cq(8);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[MAXE];
    uint16_t score[MAXE][ROWS];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    uint16_t shared_weight[ROWS];
    if constexpr (SH > 0) shared_gate(noc, gate_logit, meta, shared_weight);
    uint32_t items[2] = {0, 0};  // routed items, then shared-expert items
    for (uint32_t i = worker; i < count * NT + SH; i += W) ++items[i >= count * NT];
    cc.reserve_back(1);
    auto ctrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr());
    ctrl[0] = items[0];
    ctrl[1] = items[1];
    cc.push_back(1);
    if (!items[0] && !items[1]) return;
    // Token rows of every activation tile (faces 0 and 1); padding rows are never consumed.
    cx.reserve_back(KT);
    const uint32_t xb = cx.get_write_ptr();
    for (uint32_t t = 0; t < KT; ++t) {
        noc.async_read(x, CoreLocalMem<uint32_t>(xb + t * 2048), 32 * ROWS, {.page_id=t, .offset_bytes=0}, {});
        noc.async_read(x, CoreLocalMem<uint32_t>(xb + t * 2048 + 512), 32 * ROWS, {.page_id=t, .offset_bytes=512}, {});
    }
    noc.async_read_barrier();
    cx.push_back(KT);
    for (uint32_t i = worker; i < count * NT + SH; i += W) {
        const bool shared = i >= count * NT;
        const uint32_t slot = i / NT, column = shared ? i - count * NT : i % NT;
        cs.reserve_back(1);
        auto st = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs.get_write_ptr());
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint16_t v = shared ? shared_weight[r] : score[slot][r];
            for (uint32_t c = 0; c < 16; ++c) { st[r * 16 + c] = v; st[256 + r * 16 + c] = v; }
        }
        cs.push_back(1);
        if (shared) {
            // Shared gate/up: BF16 [2560, 320] per rank, gate columns 0..4, up 5..9.
            cq.reserve_back(KT);
            const uint32_t qb = cq.get_write_ptr();
            for (uint32_t k = 0; k < KT; ++k) {
                noc.async_read(shared_gu, CoreLocalMem<uint32_t>(qb + k * 2048), 2048, {.page_id=k * 2 * SHARED_COLS + column}, {});
            }
            noc.async_read_barrier();
            cq.push_back(KT);
            continue;
        }
        // Gate pages on this RISC, up pages on the writer RISC: two independent
        // request streams; one barrier per item.
        const auto gate = TensorAccessor(wa, table_entry(meta, expert[slot]));
        cw.reserve_back(KT);
        const uint32_t wb = cw.get_write_ptr();
#ifndef EXP_NOREAD
        for (uint32_t k = 0; k < KT; ++k) {
            noc.async_read(gate, CoreLocalMem<uint32_t>(wb + k * 576), 576, {.page_id=k * NT + column}, {});
        }
#endif
        noc.async_read_barrier();
        cw.push_back(KT);
    }
}
"""

_GU_WRITER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto ia = TensorAccessorArgs<4>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(0));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(1));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(3));
    const auto hidden = TensorAccessor(ha, get_common_arg_val<uint32_t>(4));
    constexpr auto wa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto shared_gu = TensorAccessor(qa, get_common_arg_val<uint32_t>(5));
    const uint32_t worker = worker_index();
    Noc noc;
    CircularBuffer cu(5), co(4), cm(7), cr(9);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[MAXE];
    uint16_t score[MAXE][ROWS];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;  // previous item's output page, written after the next item's up reads
    for (uint32_t i = worker; i < count * NT + SH; i += W) {
        const bool shared = i >= count * NT;
        const uint32_t slot = i / NT, column = shared ? i - count * NT : i % NT;
        if (shared) {
            cr.reserve_back(KT);
            const uint32_t rb = cr.get_write_ptr();
            for (uint32_t k = 0; k < KT; ++k) {
                noc.async_read(shared_gu, CoreLocalMem<uint32_t>(rb + k * 2048), 2048, {.page_id=k * 2 * SHARED_COLS + SHARED_COLS + column}, {});
            }
            noc.async_read_barrier();
            cr.push_back(KT);
        } else {
            const auto up = TensorAccessor(wa, table_entry(meta, 128 + expert[slot]));
            cu.reserve_back(KT);
            const uint32_t ub = cu.get_write_ptr();
#ifndef EXP_NOREAD
            for (uint32_t k = 0; k < KT; ++k) {
                noc.async_read(up, CoreLocalMem<uint32_t>(ub + k * 576), 576, {.page_id=k * NT + column}, {});
            }
#endif
            noc.async_read_barrier();
            cu.push_back(KT);
        }
        if (pending != NONE) {
            co.wait_front(1);
            noc.async_write(co, hidden, 2048, {}, {.page_id=pending});
            noc.async_write_barrier();
            co.pop_front(1);
        }
        pending = shared ? MAXE * NT + column : i;
    }
    if (pending != NONE) {
        co.wait_front(1);
        noc.async_write(co, hidden, 2048, {}, {.page_id=pending});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}
"""

_COMPUTE_HEADER = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

// The trip count is device data: UNPACK reads the reader's control page and
// forwards it so MATH and PACK run the identical loop.
template <uint32_t N>
inline void trip_counts(uint32_t cb, uint32_t* n) {
    UNPACK(({
        cb_wait_front(cb, 1);
        auto ctrl = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(cb).fifo_rd_ptr << 4);
        for (uint32_t i = 0; i < N; ++i) {
            n[i] = ctrl[i];
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n[i]);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n[i]);
        }
        cb_pop_front(cb, 1);
    }));
    MATH(({ for (uint32_t i = 0; i < N; ++i) n[i] = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId); }));
    PACK(({ for (uint32_t i = 0; i < N; ++i) n[i] = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId); }));
}
"""

_GU_COMPUTE = _COMPUTE_HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t cx = 0, cs = 2, cc = 3, co = 4;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cx, 1, co);
    uint32_t items[2];
    trip_counts<2>(cc, items);
    if (!items[0] && !items[1]) return;
    cb_wait_front(cx, KT);
    for (uint32_t item = 0; item < items[0] + items[1]; ++item) {
        // Routed items use the BF4 gate/up CBs 1/5, shared items the BF16 CBs 8/9.
        const uint32_t cw = item < items[0] ? 1 : 8, cu = item < items[0] ? 5 : 9;
        reconfig_data_format_srca(cw);
        matmul_init(cx, cw);
        tile_regs_acquire();
        cb_wait_front(cw, KT);
        cb_wait_front(cu, KT);
#ifndef EXP_NOMATH
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_tiles(cx, cw, k, k, 0);
            matmul_tiles(cx, cu, k, k, 1);
        }
#endif
        cb_pop_front(cw, KT);
        cb_pop_front(cu, KT);
        cb_wait_front(cs, 1);
        reconfig_data_format_srca(cs);
        copy_init(cs);
        copy_tile(cs, 0, 2);
        cb_pop_front(cs, 1);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        cb_reserve_back(co, 1);
        tile_regs_wait();
        pack_tile(0, co);
        tile_regs_release();
        cb_push_back(co, 1);
    }
}
"""

# ---------------------------------------------------------------- phase 2
_DOWN_READER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr auto ha = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto qa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto hidden = TensorAccessor(ha, get_common_arg_val<uint32_t>(0));
    const auto shared_down = TensorAccessor(qa, get_common_arg_val<uint32_t>(5));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const uint32_t column = worker_index();
    Noc noc;
    CircularBuffer ch(0), cw(1), cc(3), cm(6), cq(8), cz(9);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[MAXE];
    uint16_t score[MAXE][ROWS];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    cc.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr()) = count;
    cc.push_back(1);
    // Shared-expert down weights (BF16 [160, 2560] per rank) and its zero-padded rows.
    if constexpr (SH > 0) {
        cq.reserve_back(SH);
        const uint32_t qb = cq.get_write_ptr();
        for (uint32_t k = 0; k < SH; ++k) {
            noc.async_read(shared_down, CoreLocalMem<uint32_t>(qb + k * 2048), 2048, {.page_id=k * NT + column}, {});
        }
    }
    // This RISC streams K tiles [0, KT/2) of every slot's down weights, the writer
    // RISC the rest (into its own CB); all DRAM reads are issued up front.
    constexpr uint32_t HALF = KT / 2;
    if (count) cw.reserve_back(count * HALF);
    const uint32_t wb = cw.get_write_ptr();
    for (uint32_t slot = 0; slot < count; ++slot) {
        const auto down = TensorAccessor(wa, table_entry(meta, 256 + expert[slot]));
#ifndef EXP_NOREAD
        for (uint32_t k = 0; k < HALF; ++k) {
            noc.async_read(down, CoreLocalMem<uint32_t>(wb + (slot * HALF + k) * 576), 576, {.page_id=k * NT + column}, {});
        }
#endif
    }
    // The hidden CB holds exactly 2*KT pages (double buffered), so reservations
    // alternate between two fixed L1 regions: zero both once; only row 0 is rewritten.
    noc.async_write_zeros(ch, 2 * KT * 2048);
    if constexpr (SH > 0) noc.async_write_zeros(cz, SH * 2048);
    noc.write_zeros_l1_barrier();
    for (uint32_t slot = 0; slot < count; ++slot) {
        ch.reserve_back(KT);
        const uint32_t hb = ch.get_write_ptr();
        for (uint32_t k = 0; k < KT; ++k) {
            const uint32_t page = slot * KT + k;
            noc.async_read(hidden, CoreLocalMem<uint32_t>(hb + k * 2048), 32 * ROWS, {.page_id=page, .offset_bytes=0}, {});
            noc.async_read(hidden, CoreLocalMem<uint32_t>(hb + k * 2048 + 512), 32 * ROWS, {.page_id=page, .offset_bytes=512}, {});
        }
        noc.async_read_barrier();
        if (slot == 0) cw.push_back(count * HALF);
        ch.push_back(KT);
    }
    if constexpr (SH > 0) {
        cz.reserve_back(SH);
        const uint32_t zb = cz.get_write_ptr();
        for (uint32_t k = 0; k < SH; ++k) {
            const uint32_t page = MAXE * KT + k;
            noc.async_read(hidden, CoreLocalMem<uint32_t>(zb + k * 2048), 32 * ROWS, {.page_id=page, .offset_bytes=0}, {});
            noc.async_read(hidden, CoreLocalMem<uint32_t>(zb + k * 2048 + 512), 32 * ROWS, {.page_id=page, .offset_bytes=512}, {});
        }
        noc.async_read_barrier();
        cq.push_back(SH);
        cz.push_back(SH);
    }
}
"""

_DOWN_WRITER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr auto oa = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const uint32_t column = worker_index();
    Noc noc;
    CircularBuffer cv(5), co(4), cm(7);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[MAXE];
    uint16_t score[MAXE][ROWS];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    constexpr uint32_t HALF = KT / 2;
    if (count) {
        cv.reserve_back(count * (KT - HALF));
        const uint32_t vb = cv.get_write_ptr();
        for (uint32_t slot = 0; slot < count; ++slot) {
            const auto down = TensorAccessor(wa, table_entry(meta, 256 + expert[slot]));
#ifndef EXP_NOREAD
            for (uint32_t k = HALF; k < KT; ++k) {
                noc.async_read(down, CoreLocalMem<uint32_t>(vb + (slot * (KT - HALF) + k - HALF) * 576), 576, {.page_id=k * NT + column}, {});
            }
#endif
        }
        noc.async_read_barrier();
        cv.push_back(count * (KT - HALF));
    }
    co.wait_front(1);
    noc.async_write(co, output, 2048, {}, {.page_id=column});
    noc.async_write_barrier();
    co.pop_front(1);
}
"""

_DOWN_COMPUTE = _COMPUTE_HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t ch = 0, cw = 1, cc = 3, co = 4, cv = 5, cq = 8, cz = 9, SH = SHARED_ITEMS;
    compute_kernel_hw_startup<SrcOrder::Reverse>(ch, cw, co);
    uint32_t slots;
    trip_counts<1>(cc, &slots);
    matmul_init(ch, cw);
    tile_regs_acquire();
    for (uint32_t slot = 0; slot < slots; ++slot) {
        cb_wait_front(ch, KT);
        cb_wait_front(cw, KT / 2);
        cb_wait_front(cv, KT - KT / 2);
#ifndef EXP_NOMATH
        for (uint32_t k = 0; k < KT / 2; ++k) matmul_tiles(ch, cw, k, k, 0);
        for (uint32_t k = KT / 2; k < KT; ++k) matmul_tiles(ch, cv, k, k - KT / 2, 0);
#endif
        cb_pop_front(ch, KT);
        cb_pop_front(cw, KT / 2);
        cb_pop_front(cv, KT - KT / 2);
    }
    if constexpr (SH > 0) {
        // Shared expert: BF16 weights, accumulated into the same DST tile.
        reconfig_data_format_srca(cq);
        matmul_init(cz, cq);
        cb_wait_front(cz, SH);
        cb_wait_front(cq, SH);
        for (uint32_t k = 0; k < SH; ++k) matmul_tiles(cz, cq, k, k, 0);
        cb_pop_front(cz, SH);
        cb_pop_front(cq, SH);
    }
    tile_regs_commit();
    cb_reserve_back(co, 1);
    tile_regs_wait();
    pack_tile(0, co);
    tile_regs_release();
    cb_push_back(co, 1);
}
"""


_EXP = "".join(f"#define EXP_{flag.upper()}\n" for flag in os.environ.get("MOE_EXP", "").split(",") if flag)


def _prelude(rows, shared):
    return _EXP + f"#define ROWS {rows}\n#define SHARED_ITEMS {5 if shared else 0}\n"


def _cb(index, pages, page_size, data_format, cores):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _accessor_args(tensors):
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    return args


class SingleTokenExperts:
    """Replaces dispatch + prefill expert op + compact + weighting for one token."""

    def __init__(self, moe, *, gate_up_workers=100, fidelity=ttnn.MathFidelity.HiFi2):
        self.moe, self.mesh = moe, moe.ops.mesh
        self.max_rows = MAX_ROWS
        grid = self.mesh.compute_with_storage_grid_size()
        if grid.x < 10 or grid.y < gate_up_workers // 10 or gate_up_workers % 10:
            raise ValueError("Single-token experts use 10-wide worker rectangles")
        self.gu_workers = gate_up_workers
        self.gu_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, gate_up_workers // 10 - 1))]
        )
        self.down_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))])
        addresses = [t.buffer_address() for t in (*moe.gates, *moe.ups, *moe.downs)]
        if len(addresses) != 384:
            raise ValueError("Expected 128 local experts per rank")
        table = torch.from_numpy(np.array(addresses, dtype=np.uint32).view(np.int32))
        self.table = moe.ops.upload(table, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
        )

    def __call__(self, x, ids, scores, gate=None):
        """x [1,1,B,2560] BF16 TILE, ids [1,1,B,32] UINT32 RM, scores [1,1,B,32] BF16 RM, B <= MAX_ROWS.

        gate [1,1,B,32] BF16 RM (column 0 = sigmoid shared-gate logit) also folds in the
        rank's shared-expert slice. Returns this rank's partial sums [1,1,B,2560] BF16 TILE.
        """
        moe = self.moe
        rows = x.shape[-2]
        if tuple(x.shape) != (1, 1, rows, 2560) or not 1 <= rows <= MAX_ROWS:
            raise ValueError("Few-token experts take [1,1,B<=4,2560] activations")
        shared = gate is not None
        prelude = _prelude(rows, shared)
        gate = gate if shared else scores  # never read without shared items
        # Every core reads the token rows of every x/hidden tile: keep both L1-interleaved
        # so those small reads spread over many cores instead of a few DRAM pages.
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        hidden = ttnn.empty(
            (1, 10 * rows + 1, 32, 640),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.empty(
            (1, 1, rows, 2560),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        meta = [ids, scores, moe.rank, self.table]
        # Phase 1: gate/up items.
        cores, workers = self.gu_cores, self.gu_workers
        reader = ttnn.KernelDescriptor(
            kernel_source=prelude + _GU_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[workers, 80, 20, 0] + _accessor_args([x, *meta, gate, moe.gates[0], moe.shared_gu]),
            common_runtime_args=[t.buffer_address() for t in (x, *meta, gate, moe.shared_gu)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=prelude + _GU_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[workers, 80, 20, 0] + _accessor_args([*meta, hidden, moe.ups[0], moe.shared_gu]),
            common_runtime_args=[t.buffer_address() for t in (*meta, hidden, moe.shared_gu)],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=prelude + _GU_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[80],
            config=self.compute,
        )
        cbs = [
            _cb(0, 80, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(1, 2 * 80, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(5, 2 * 80, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(2, 2, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(3, 1, 64, ttnn.uint32, cores),
            _cb(4, 2, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(6, 1, _META, ttnn.uint32, cores),
            _cb(7, 1, _META, ttnn.uint32, cores),
            _cb(8, 80, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(9, 80, _TILE_BF16, ttnn.bfloat16, cores),
        ]
        ttnn.generic_op([x, *meta, gate, hidden], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
        # Phase 2: down projection, one output column tile per core.
        cores = self.down_cores
        reader = ttnn.KernelDescriptor(
            kernel_source=prelude + _DOWN_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20, 80] + _accessor_args([hidden, *meta, moe.downs[0], moe.shared_down]),
            common_runtime_args=[t.buffer_address() for t in (hidden, *meta, moe.shared_down)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=prelude + _DOWN_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20, 80] + _accessor_args([output, *meta, moe.downs[0]]),
            common_runtime_args=[t.buffer_address() for t in (output, *meta)],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=prelude + _DOWN_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20],
            config=self.compute,
        )
        cbs = [
            _cb(0, 2 * 20, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(1, 10 * rows * 10, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(5, 10 * rows * 10, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(3, 1, 64, ttnn.uint32, cores),
            _cb(4, 1, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(6, 1, _META, ttnn.uint32, cores),
            _cb(7, 1, _META, ttnn.uint32, cores),
            _cb(8, 5, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(9, 5, _TILE_BF16, ttnn.bfloat16, cores),
        ]
        ttnn.generic_op([hidden, *meta, output], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
        return output


def group_single(moe, x):
    """ParallelMoE.group_native/group_packed for [1,1,B<=MAX_ROWS,2560] decode rows."""
    router = moe.ops.linear(x, moe.router, compute_kernel_config=moe.router_compute)
    values, indices = ttnn.topk(
        ttnn.pad(router[..., :512], [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
    )
    selected = ttnn.softmax(
        ttnn.add(values, moe.mask), dim=-1, numeric_stable=True, compute_kernel_config=moe.ops.compute
    )
    ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    mask = getattr(moe, "row_mask", None)
    # A zero score drops the expert for that row, so masked rows read no weights.
    scores = ttnn.to_layout(selected if mask is None else ttnn.multiply(selected, mask), ttnn.ROW_MAJOR_LAYOUT)
    if os.environ.get("QWEN_FLASH_DECODE_MOE_SHARED", "1") != "0":
        # Shared expert as five extra BF16 items of the same programs, gated per row.
        gate = ttnn.to_layout(ttnn.sigmoid(router[..., 512:544]), ttnn.ROW_MAJOR_LAYOUT)
        return moe.single_token(x, ids, scores, gate)
    result = moe.single_token(x, ids, scores)
    shared = moe.ops.linear(x, moe.shared_gu, compute_kernel_config=moe.shared_compute)
    shared = moe.ops.linear(
        ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
        moe.shared_down,
        compute_kernel_config=moe.shared_compute,
    )
    return ttnn.add(
        result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    )
