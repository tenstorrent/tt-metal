// Resident DFlash2 drafter streamer reader v4: v2 with the hub's top-16 record address from the streamers' sem_recaddr
// mailbox (runtime arg 35 holds its id; d_hub_dm_v6.cpp publishes its record CB there).
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter streamer reader v2 (NCRISC; M6b fix 4; d_streamer_reader.cpp is the verified v1): the head's
// per-row top-16 (d_topk.hpp) of rows [kMw, M) instead of the argmax (records: R rows x 32 words per core on the hub),
// and head marks at timeline words 16 L + 4 (head stream done) and 16 L + 5 (top-16 sent).
//
// v1 (NCRISC; M6b): streamer_reader.cpp (the target's) with the drafter's entry
// order (d_streamer_common.hpp): per layer Kp_attn, W_qkv, W_o, Kp_mlp, [G | U], W_down, then the LM head. The bank
// layout, the NOC alternation, the trid read state and the ring room accounting are the target's, unchanged, so the
// read shape on these multicast destinations is the validated one (K2a N2: own bank, pages alternating NOC 0 / 1).
//
// Runtime args: as streamer_reader.cpp: 0 bank_id, 1 vc, 2 slot j of this core in its bank, 3 cores_per_bank, then per
// entry e: 4 + 4e n_tiles, base address, per-copy stride (bytes per bank), copies; 32 timeline (0 = off), 33 hub_x,
// 34 hub_y, 35 hub argmax records address, 36 hub sem_rec id, 37 nv, 38 vocab index of its first column, 39 this
// core's index. (No serving control word: the D-program never writes the logits tensor.)

#include "api/dataflow/dataflow_api.h"
#include "d_streamer_common.hpp"
#include "d_topk.hpp"

using namespace dres;

namespace {

constexpr uint32_t kInFlight = 2;

uint32_t ring_base;
volatile uint32_t* consumed;
RingCursor cursor;

volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
constexpr uint32_t kTsWordsD = 8;  // timeline words per layer (reader [0, 8))
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
            ts[l * kTsWordsD + e] = *clk;
            ts[l * kTsWordsD + 4 + e] = stall_cycles;
        }
        stall_cycles = 0;
    };
    stall_cycles = 0;
    for (uint32_t l = 0; l < layers; l++) {
        stream_entry<kKpA>(bank_id, vc, slot, slots, l);
        stream_entry<kQkv>(bank_id, vc, slot, slots, l);
        mark(l, 0);
        stream_entry<kO>(bank_id, vc, slot, slots, l);
        mark(l, 1);
        stream_entry<kKpM>(bank_id, vc, slot, slots, l);
        stream_entry<kGU>(bank_id, vc, slot, slots, l);
        mark(l, 2);
        stream_entry<kDown>(bank_id, vc, slot, slots, l);
        mark(l, 3);
    }
    stream_entry<kHead>(bank_id, vc, slot, slots, 0);
    if (ts_addr) {
        ts[16 * layers + 4] = *clk;
    }

    // top-16 of logit rows [kMw, M): records [core][R rows][32 words] on the hub
    const uint32_t hub_x = get_arg_val<uint32_t>(33), hub_y = get_arg_val<uint32_t>(34);
    // (v4) the hub's record CB from the sem_recaddr mailbox (runtime arg 35 now holds its id; d_hub_dm_v6.cpp)
    volatile tt_l1_ptr uint32_t* recaddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(35)));
    uint32_t hub_rec = 0;
    while (hub_rec == 0) {
        invalidate_l1_cache();
        hub_rec = *recaddr_box;
    }
    const uint32_t rec_sem = get_semaphore(get_arg_val<uint32_t>(36));
    const uint32_t nv = get_arg_val<uint32_t>(37), first_index = get_arg_val<uint32_t>(38);
    const uint32_t core = get_arg_val<uint32_t>(39);
    constexpr uint32_t kCoreRec = R * dtop::kRowBytes;
    if constexpr (M > kMw) {
        cb_wait_front(cb_logits, nv_max);
        // staged in the ring (its weights are all consumed once the logits exist)
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_base);
        dtop::top16_rows(get_read_ptr(cb_logits), nv, first_index, kMw, M, rec);
        noc_async_write(ring_base + kMw * dtop::kRowBytes,
                        get_noc_addr(hub_x, hub_y, hub_rec + core * kCoreRec + kMw * dtop::kRowBytes, 0),
                        (M - kMw) * dtop::kRowBytes, 0);
        noc_async_write_barrier(0);
    }
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, rec_sem, 0), 1, 0);
    noc_async_atomic_barrier(0);
    if (ts_addr) {
        ts[16 * layers + 5] = *clk;
    }
}
