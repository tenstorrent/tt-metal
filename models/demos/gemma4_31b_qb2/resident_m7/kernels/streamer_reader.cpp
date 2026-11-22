// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident streamer reader (NCRISC): streams this core's columns of every layer's weights from its DRAM bank
// into the shared weight ring, running ahead of compute by up to the ring size (across the stub mixer phases and the
// all-reduces). Adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/kernels/streamer_reader.cpp
// (Apache-2.0): a bank's cores interleave their blocks in the bank (slot r * cores_per_bank + j is block r of the
// bank's core j) and each block's pages alternate between NOC0 and NOC1 (one NOC's reads cap at ~47 GB/s per bank,
// both reach the bank's ~64 GB/s, per the original's measurement).
//
// Runtime args: 0 bank_id, 1 vc, 2 slot j of this core in its bank, 3 cores_per_bank, then per entry e:
// 4 + 4e n_tiles, base address, per-copy stride (bytes per bank), copies; then 32 optional timeline buffer
// (0 = off): per layer the wall clock at the end of each entry's stream and the cycles each waited for ring room;
// 33 hub_x, 34 hub_y, 35 hub argmax records address, 36 hub sem_rec id, 37 nv (LM head tiles of this core), 38 vocab
// index of its first column, 39 this core's index, 40 control (serving: DRAM [1, 1, 1, 32] uint32 row; 0 = none),
// 41 sem_ctrl id. After its last weight block the reader scans logit rows [kMw, M) (the writer takes [0, kMw)) and
// writes their records to the hub. At launch the reader copies the control word's write_logits into sem_ctrl for the
// writer: the writer (NOC 1) must not read over NOC 1 itself, where this reader streams with its own read state and
// counters (a second reader's plain barrier on the same NOC never sees its count).

#include "api/dataflow/dataflow_api.h"
#include "streamer_common.hpp"

using namespace gres;

namespace {

constexpr uint32_t kInFlight = 2;

uint32_t ring_base;
volatile uint32_t* consumed;
RingCursor cursor;

volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
uint32_t stall_cycles;

FORCE_INLINE void wait_for_room() {
    constexpr uint32_t ring_units = ring_bytes / kUnit;
    if (((cursor.units - *consumed) & 0xffff) > ring_units) {
        const uint32_t t0 = *clk;
        while (((cursor.units - *consumed) & 0xffff) > ring_units) {
        }
        stall_cycles += *clk - t0;
    }
}

// copy k (of this entry's kind) of the weights
template <uint32_t E>
FORCE_INLINE void stream_entry(uint32_t bank_id, uint32_t vc, uint32_t slot, uint32_t slots, uint32_t k) {
    using En = Entry<E>;
    const uint32_t n_tiles = get_arg_val<uint32_t>(4 + 4 * E);
    const uint32_t copies = get_arg_val<uint32_t>(7 + 4 * E);
    const uint32_t weight_addr = get_arg_val<uint32_t>(5 + 4 * E) + (k % copies) * get_arg_val<uint32_t>(6 + 4 * E);
    const uint32_t num_blocks = n_tiles * En::nkb;
    const uint64_t base0 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 0);
    const uint64_t base1 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 1);
    noc_async_read_one_packet_set_state<true>(base0, En::page_size, vc, 0);
    noc_async_read_one_packet_set_state<true>(base1, En::page_size, vc, 1);

    const uint32_t stride = slots * En::block_bytes;
    uint32_t src = slot * En::block_bytes;
    uint32_t issued = 0;
    uint32_t next_trid = 1;
    uint32_t wait_trid = 1;
    auto retire = [&]() {
        noc_async_read_barrier_with_trid(wait_trid, 0);
        noc_async_read_barrier_with_trid(wait_trid, 1);
        cb_push_back(En::cb, En::sb);
        wait_trid = wait_trid == kInFlight + 1 ? 1 : wait_trid + 1;
        issued--;
    };
    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t l1 = ring_base + cursor.place<E>();
        wait_for_room();
        noc_async_read_set_trid(next_trid, 0);
        noc_async_read_set_trid(next_trid, 1);
        for (uint32_t p = 0; p < En::pages; p++) {
            const uint32_t off = p * En::page_size;
            const uint8_t noc = (b * En::pages + p) & 1;
            noc_async_read_one_packet_with_state_with_trid(noc ? base1 : base0, src + off, l1 + off, next_trid, noc);
        }
        src += stride;
        next_trid = next_trid == kInFlight + 1 ? 1 : next_trid + 1;
        if (++issued == kInFlight + 1) {
            retire();
        }
    }
    while (issued > 0) {
        retire();
    }
}

}  // namespace

void kernel_main() {
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t slots = get_arg_val<uint32_t>(3);
    const uint32_t ts_addr = get_arg_val<uint32_t>(32);
    {
        // serving: write_logits for the writer (the ring is still empty: its first 128 B take the control row)
        const uint32_t control = get_arg_val<uint32_t>(40);
        volatile tt_l1_ptr uint32_t* box =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(41)));
        uint32_t flag = 0;
        if (control) {
            auto& w = get_local_cb_interface(cb_w0);
            const uint32_t scratch = (w.fifo_limit - w.fifo_size) << cb_addr_shift;
            const InterleavedAddrGen<true> cg{.bank_base_address = control, .page_size = 128};
            noc_async_read(cg.get_noc_addr(0), scratch, 128);
            noc_async_read_barrier();
            flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[1];
        }
        *box = flag;
    }
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 1);
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = (w.fifo_limit - w.fifo_size) << cb_addr_shift;
        consumed = get_cb_tiles_acked_ptr(cb_consumed);
        *consumed = 0;
    }
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    auto mark = [&](uint32_t l, uint32_t e) {
        if (ts_addr) {
            ts[l * kTsWords + e] = *clk;
            ts[l * kTsWords + 4 + e] = stall_cycles;
        }
        stall_cycles = 0;
    };
    stall_cycles = 0;
    uint32_t s = 0, f = 0;  // sliding / full layers so far
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        if (full) {
            stream_entry<kQkvF>(bank_id, vc, slot, slots, f);
        } else {
            stream_entry<kQkvS>(bank_id, vc, slot, slots, s);
        }
        mark(l, 0);
        if (full) {
            stream_entry<kOF>(bank_id, vc, slot, slots, f++);
        } else {
            stream_entry<kOS>(bank_id, vc, slot, slots, s++);
        }
        mark(l, 1);
        stream_entry<kGU>(bank_id, vc, slot, slots, l);
        mark(l, 2);
        stream_entry<kDown>(bank_id, vc, slot, slots, l);
        mark(l, 3);
    }
    stream_entry<kHead>(bank_id, vc, slot, slots, 0);

    // argmax of logit rows [kMw, M): records [core][R rows][key, index] on the hub
    const uint32_t hub_x = get_arg_val<uint32_t>(33), hub_y = get_arg_val<uint32_t>(34);
    const uint32_t hub_rec = get_arg_val<uint32_t>(35);
    const uint32_t rec_sem = get_semaphore(get_arg_val<uint32_t>(36));
    const uint32_t nv = get_arg_val<uint32_t>(37), first_index = get_arg_val<uint32_t>(38);
    const uint32_t core = get_arg_val<uint32_t>(39);
    if constexpr (M > kMw) {
        cb_wait_front(cb_logits, nv_max);
        // staged in the ring (its weights are all consumed once the logits exist)
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_base);
        argmax_rows(get_read_ptr(cb_logits), nv, first_index, kMw, M, rec);
        noc_async_write(
            ring_base + kMw * 8, get_noc_addr(hub_x, hub_y, hub_rec + core * kRecStride + kMw * 8), (M - kMw) * 8, 0);
        noc_async_write_barrier(0);
    }
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, rec_sem, 0), 1, 0);
    noc_async_atomic_barrier(0);
}
