// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel reader core — weight streamer (BRISC, NoC0): the weight slices of EVERY matmul phase of the layer, back
// to back, from this reader's DRAM bank (the DS width-sharded tensors, read in place) into the phase's ring CB. The
// stream never waits for activations, only for ring space, so the next phase's first blocks are prefetched while the
// current phase computes and while the glue phases between run (the cross-phase overlap the op chain cannot have).
//
// Compile-time args: NPH, then per phase p 8 values at 1 + 8p: NB, KB, NR, TB, TPT, SHARD_W, CB (ring CB index),
// SLOTS (the ring's blocks: 3, or 2 for rings with large blocks, or the deep-ring count; the same for every phase of a
// ring), RP (readers per bank in this phase: reader indices >= RP sit it out — no reads, the ring untouched; 7a: the
// value is (R0 << 8) | RP and the phase runs on reader indices [R0, R0 + RP)).
// Phases sharing a ring CB must have the same block size; the ring's slot position carries over from one such phase
// to the next (one continuous stream per ring: the compute pops the blocks in the same order), so the explicit slot
// addressing stays in step with the CB's own pointers. No reads are in flight at a phase boundary (the last block is
// waited for), so transaction ids restart per phase.
// Runtime args: bank, reader index, VC, then per phase the weight tensor's bank base address.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"

// MK_NO_DRAM (timing only): no weight reads, the ring slots are pushed as they are -> the phase's compute-bound time
#if defined(MK_NO_DRAM)
constexpr bool MK_NO_DRAM_ROWS = true;
#else
constexpr bool MK_NO_DRAM_ROWS = false;
#endif

namespace {

uint32_t g_ring_slot[64];  // per ring CB: the slot the next block goes to (zero-initialised .bss)

#if defined(MK_PARK_ARG)
// Forwarding ring (MK_PARK_ARG = index of the park runtime args: n owners, then (x, y, slots) per owner, then the
// mask of the phases that park): when the phase's ring is full the streamer keeps reading DRAM — block by block into
// the staging CB, then over the NoC into a PARK slot in an owner core's L1 (the owners' L1 below the readers' static
// CB region end is otherwise unused: kernels/mk/owner_dm0.cpp publishes its park CB address in semaphore
// MK_PARK_SEM). Ring slots that free up are refilled from the park FIFO first (oldest block first: the stream order
// is kept), then from DRAM. Every parked block of a phase is back in the ring before the phase ends.
struct ParkState {
    uint32_t n, cap, head, count, mask;
    uint32_t x[4], y[4], slots[4], base[4];
};
ParkState g_park;

FORCE_INLINE void park_init(uint32_t scratch) {
    g_park.n = get_arg_val<uint32_t>(ph_rt0 + MK_PARK_ARG);
    g_park.cap = g_park.head = g_park.count = 0;
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t o = 0; o < g_park.n; ++o) {
        g_park.x[o] = get_arg_val<uint32_t>(ph_rt0 + MK_PARK_ARG + 1 + 3 * o);
        g_park.y[o] = get_arg_val<uint32_t>(ph_rt0 + MK_PARK_ARG + 2 + 3 * o);
        g_park.slots[o] = get_arg_val<uint32_t>(ph_rt0 + MK_PARK_ARG + 3 + 3 * o);
        uint32_t base = 0;
        while (base == 0) {  // the owner writes its park address at its kernel start
            s[0] = 0;
            noc_async_read(get_noc_addr(g_park.x[o], g_park.y[o], get_semaphore(MK_PARK_SEM)), scratch, 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            base = s[0];
        }
        g_park.base[o] = base;
        g_park.cap += g_park.slots[o];
    }
    g_park.mask = get_arg_val<uint32_t>(ph_rt0 + MK_PARK_ARG + 1 + 3 * g_park.n);
}

FORCE_INLINE uint64_t park_noc_addr(uint32_t i, uint32_t block_bytes) {
    uint32_t o = 0;
    while (i >= g_park.slots[o]) {
        i -= g_park.slots[o];
        ++o;
    }
    return get_noc_addr(g_park.x[o], g_park.y[o], g_park.base[o] + i * block_bytes);
}

// one weight block from DRAM into local L1 (synchronous: the block has landed on return)
template <uint32_t KB, uint32_t NR, uint32_t TB, uint32_t TPT, uint32_t SHARD_W>
FORCE_INLINE void dram_block(uint64_t dram, uint32_t w_addr, uint32_t ridx, uint32_t blk, uint32_t dst, uint32_t vc) {
    constexpr uint32_t TXN_BYTES = TPT * TB;
    noc_async_read_one_packet_set_state<true>(dram, TXN_BYTES, vc);  // a park pull reprograms the read cmd buffer
    for (uint32_t row = 0; row < KB; ++row) {
        uint32_t src = w_addr + ((blk * KB + row) * SHARD_W + ridx * NR) * TB;
        for (uint32_t t = 0; t < NR / TPT; ++t) {
            noc_async_read_one_packet_with_state(src, dst);
            src += TXN_BYTES;
            dst += TXN_BYTES;
        }
    }
    noc_async_read_barrier();
}

template <
    uint32_t NB,
    uint32_t KB,
    uint32_t NR,
    uint32_t TB,
    uint32_t TPT,
    uint32_t SHARD_W,
    uint32_t CB,
    uint32_t RING_SLOTS>
FORCE_INLINE void stream_phase_park(uint64_t dram, uint32_t ridx, uint32_t vc, uint32_t w_addr) {
    constexpr uint32_t BLOCK_TILES = KB * NR;
    constexpr uint32_t BLOCK_BYTES = BLOCK_TILES * TB;
    const uint32_t ring = get_local_cb_interface(CB).fifo_limit - get_local_cb_interface(CB).fifo_size;
    const uint32_t stage = get_write_ptr(MK_STAGE_CB);
    uint32_t slot = g_ring_slot[CB], fetch = 0, put = 0;  // invariant: fetch == put + g_park.count
    while (put < NB) {
        if (cb_pages_reservable_at_back(CB, BLOCK_TILES)) {
            cb_reserve_back(CB, BLOCK_TILES);
            const uint32_t dst = ring + slot * BLOCK_BYTES;
            if (g_park.count > 0) {  // the oldest parked block first
                noc_async_read(park_noc_addr(g_park.head, BLOCK_BYTES), dst, BLOCK_BYTES);
                noc_async_read_barrier();
                g_park.head = g_park.head + 1 == g_park.cap ? 0 : g_park.head + 1;
                --g_park.count;
            } else {
                dram_block<KB, NR, TB, TPT, SHARD_W>(dram, w_addr, ridx, fetch++, dst, vc);
            }
            cb_push_back(CB, BLOCK_TILES);
            slot = slot + 1 == RING_SLOTS ? 0 : slot + 1;
            ++put;
        } else if (fetch < NB && g_park.count < g_park.cap) {  // ring full: park the next block in an owner's L1
            dram_block<KB, NR, TB, TPT, SHARD_W>(dram, w_addr, ridx, fetch++, stage, vc);
            uint32_t tail = g_park.head + g_park.count;
            tail = tail >= g_park.cap ? tail - g_park.cap : tail;
            noc_async_write(stage, park_noc_addr(tail, BLOCK_BYTES), BLOCK_BYTES);
            noc_async_write_barrier();
            ++g_park.count;
        } else {
            cb_reserve_back(CB, BLOCK_TILES);  // nothing to park: wait for ring space (taken next iteration)
        }
    }
    g_ring_slot[CB] = slot;
}
#endif

template <uint32_t P>
FORCE_INLINE void stream_phase(uint64_t dram, uint32_t ridx, uint32_t vc, uint32_t w_addr) {
    constexpr uint32_t B = 1 + 9 * P;
    constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + B + 0);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + B + 1);
    constexpr uint32_t NR = get_compile_time_arg_val(PH_CT0 + B + 2);
    constexpr uint32_t TB = get_compile_time_arg_val(PH_CT0 + B + 3);
    constexpr uint32_t TPT = get_compile_time_arg_val(PH_CT0 + B + 4);
    constexpr uint32_t SHARD_W = get_compile_time_arg_val(PH_CT0 + B + 5);
    constexpr uint32_t CB = get_compile_time_arg_val(PH_CT0 + B + 6);
    constexpr uint32_t RING_SLOTS = get_compile_time_arg_val(PH_CT0 + B + 7);
    // (R0 << 8) | RP: 7a reader range per phase — the phase runs on readers [R0, R0 + RP) of every bank (R0 = 0: the
    // first RP); a reader's column slot is its index - R0
    constexpr uint32_t RPW = get_compile_time_arg_val(PH_CT0 + B + 8);
    constexpr uint32_t RP = RPW & 0xFFu, R0 = RPW >> 8;
    // transaction ids: at most two blocks are in flight (issue blk, then wait for blk - 1), so a short id cycle serves
    // any ring depth (NoC ids are 4-bit)
    constexpr uint32_t TRIDS = RING_SLOTS < 4 ? RING_SLOTS : 4;
    constexpr uint32_t TXN_BYTES = TPT * TB;
    constexpr uint32_t TXNS_PER_ROW = NR / TPT;
    constexpr uint32_t BLOCK_TILES = KB * NR;
    constexpr uint32_t BLOCK_BYTES = BLOCK_TILES * TB;
    static_assert(NR % TPT == 0 && TXN_BYTES <= NOC_MAX_BURST_SIZE, "transaction size");
    static_assert(RING_SLOTS >= 2, "the stream keeps one block in flight while the previous one is consumed");

    if (ridx < R0 || ridx >= R0 + RP) {
        return;
    }
    const uint32_t pidx = ridx - R0;
#if defined(MK_PARK_ARG)
    if (g_park.mask & (1u << P)) {
        stream_phase_park<NB, KB, NR, TB, TPT, SHARD_W, CB, RING_SLOTS>(dram, pidx, vc, w_addr);
        MK_MARK();
        return;
    }
#endif
    noc_async_read_one_packet_set_state<true>(dram, TXN_BYTES, vc);
    const uint32_t ring = get_local_cb_interface(CB).fifo_limit - get_local_cb_interface(CB).fifo_size;  // ring base
    uint32_t slot = g_ring_slot[CB], trid_issue = 1, trid_wait = 1;
    bool in_flight = false;
    // space for block 0 AND block 1 before either is issued: block 1 goes out before the loop's first reserve, and on a
    // ring shared with the previous phase its slot may still hold that phase's last block (pushed, not yet consumed;
    // the streamer never waits for activations) — reserving one block here let the next phase overwrite it (watcher
    // run 2026-10-19: o_proj's last K block clobbered by gate_up's block 1, R=3)
    cb_reserve_back(CB, (NB > 1 ? 2 : 1) * BLOCK_TILES);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        noc_async_read_set_trid(trid_issue);
        uint32_t dst = ring + slot * BLOCK_BYTES;
        for (uint32_t row = 0; row < (MK_NO_DRAM_ROWS ? 0 : KB); ++row) {
            uint32_t src = w_addr + ((blk * KB + row) * SHARD_W + pidx * NR) * TB;
            for (uint32_t t = 0; t < TXNS_PER_ROW; ++t) {
                noc_async_read_one_packet_with_state_with_trid<false, false>(dram, src, dst, trid_issue);
                src += TXN_BYTES;
                dst += TXN_BYTES;
            }
        }
        slot = slot + 1 == RING_SLOTS ? 0 : slot + 1;
        trid_issue = trid_issue == TRIDS ? 1 : trid_issue + 1;
        if (in_flight) {
            noc_async_read_barrier_with_trid(trid_wait);
            cb_push_back(CB, BLOCK_TILES);
            trid_wait = trid_wait == TRIDS ? 1 : trid_wait + 1;
            cb_reserve_back(CB, 2 * BLOCK_TILES);
        }
        in_flight = true;
    }
    noc_async_read_barrier_with_trid(trid_wait);
    cb_push_back(CB, BLOCK_TILES);
    g_ring_slot[CB] = slot;
    MK_MARK();
}

// phase sequence; a composite kernel may run another phase family between phase MK_SPLIT - 1 and MK_SPLIT (MK_BETWEEN)
template <uint32_t P, uint32_t NPH>
FORCE_INLINE void run_w(uint64_t dram, uint32_t ridx, uint32_t vc) {
#if defined(MK_SPLIT)
    if constexpr (P == MK_SPLIT) {
        MK_BETWEEN();
    }
#endif
    if constexpr (P < NPH) {
        stream_phase<P>(dram, ridx, vc, get_arg_val<uint32_t>(ph_rt0 + 3 + P));
        run_w<P + 1, NPH>(dram, ridx, vc);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
    const uint32_t bank = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t ridx = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t vc = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint64_t dram = get_noc_addr_from_bank_id<true>(bank, 0);
#if defined(MK_W_DELAY_CYCLES)  // Stage 14 knob: leave the DRAM to mhc_pre's critical-path reads first (layer 0)
    {
        const uint64_t t0 = get_timestamp();
        while (get_timestamp() - t0 < MK_W_DELAY_CYCLES) {
        }
    }
#endif
#if defined(MK_PARK_ARG)
    park_init(get_write_ptr(MK_STAGE_CB));
#endif
    // stack program (kernels/mk/stack.hpp): the weight stream runs on from layer l's last phase into layer l + 1's
    // first (the rings carry over, g_ring_slot); only the per-layer weight addresses change (a composite kernel's
    // MK_BETWEEN phases run inside every layer's phase sequence, with that layer's args)
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {
        MK_LAYER_BEGIN(layer);
        MK_MARK();
        if (layer > 0) {
            mkstack::load_args(layer);
        }
        run_w<0, NPH>(dram, ridx, vc);
#if defined(MK_PROBE_ARG)
        MK_FLUSH(get_write_ptr(MK_PROBE_CB));  // write cmd buffer: every stream_phase re-arms its read state anyway
#endif
    }
    noc_async_read_set_trid(0);
    noc_async_read_one_packet_set_state<true>(dram, 64, 1);  // firmware default VC (NOC_CTRL persists)
}
