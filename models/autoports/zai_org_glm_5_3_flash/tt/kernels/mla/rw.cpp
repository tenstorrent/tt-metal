// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel reader core — weight streamer with one-phase LOOKAHEAD (BRISC, NoC0). Same phase table, ring and
// block layout as 7a's kernels/mk/reader_w.cpp (the DS width-sharded weights read in place, one ring CB per (format,
// block size), the ring slot position carried across the phases of a ring), but the stream does not stall when the
// current phase's ring is full: it keeps issuing the NEXT phase's blocks into that phase's ring (if it is a different
// ring), so DRAM stays busy while the current phase waits for its activations — mhc_pre (q_a's ring full -> q_b's
// ring fills), the attention (W_uv's ring full -> o_proj's ring fills).
// Per ring, blocks are issued and pushed in stream order; a ring's slot is reserved before its block is issued (the
// CB's free space minus the blocks in flight), so a block never overwrites an unconsumed one. At most MAXF blocks per
// phase are in flight, each with its own NoC transaction id (per-ring id ranges; a ring's phases are never active
// together because lookahead needs a different ring).
// Compile-time args: NPH, then per phase p 9 values at 1 + 9p: NB, KB, NR, TB, TPT, SHARD_W, CB, SLOTS, RP.
// Runtime args: bank, reader index, VC, then per phase the weight tensor's bank base address (+ probe args).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t MAXP = 8;
constexpr uint32_t MAXF = 2;  // blocks in flight per phase
static_assert(NPH <= MAXP, "phases");

struct Phase {
    uint32_t nb, kb, nr, tb, tpt, shard_w, cb, slots, rp;
    uint32_t addr, issued, pushed, trid0;
    uint32_t block_tiles, block_bytes, txn_bytes;
};
Phase g_ph[MAXP];
uint32_t g_ring_slot[64];  // per ring CB: the slot the next block goes to

template <uint32_t P>
FORCE_INLINE void load_table() {
    if constexpr (P < NPH) {
        constexpr uint32_t B = 1 + 9 * P;
        Phase& f = g_ph[P];
        f.nb = get_compile_time_arg_val(PH_CT0 + B + 0);
        f.kb = get_compile_time_arg_val(PH_CT0 + B + 1);
        f.nr = get_compile_time_arg_val(PH_CT0 + B + 2);
        f.tb = get_compile_time_arg_val(PH_CT0 + B + 3);
        f.tpt = get_compile_time_arg_val(PH_CT0 + B + 4);
        f.shard_w = get_compile_time_arg_val(PH_CT0 + B + 5);
        f.cb = get_compile_time_arg_val(PH_CT0 + B + 6);
        f.slots = get_compile_time_arg_val(PH_CT0 + B + 7);
        f.rp = get_compile_time_arg_val(PH_CT0 + B + 8);
        f.addr = get_arg_val<uint32_t>(ph_rt0 + 3 + P);
        f.issued = f.pushed = 0;
        f.trid0 = 1 + 4 * (P % 3);  // per-phase id range (ids 1..12): a phase and its lookahead never share ids
        f.block_tiles = f.kb * f.nr;
        f.block_bytes = f.block_tiles * f.tb;
        f.txn_bytes = f.tpt * f.tb;
        load_table<P + 1>();
    }
}

FORCE_INLINE uint32_t trid_of(const Phase& f, uint32_t blk) { return f.trid0 + (blk & 3); }

// a block of phase p can be issued: blocks remain, < MAXF in flight, and the ring has room for every block issued but
// not yet pushed plus this one
FORCE_INLINE bool can_issue(const Phase& f) {
    const uint32_t inflight = f.issued - f.pushed;
    return f.issued < f.nb && inflight < MAXF &&
           cb_pages_reservable_at_back(f.cb, static_cast<int32_t>((inflight + 1) * f.block_tiles));
}

FORCE_INLINE void issue(Phase& f, uint64_t dram, uint32_t ridx, uint32_t vc, uint32_t& state_of) {
    const uint32_t me = static_cast<uint32_t>(&f - g_ph);
    if (state_of != me) {  // the read cmd buffer's size / VC state for this phase's transactions
        noc_async_read_one_packet_set_state<true>(dram, f.txn_bytes, vc);
        state_of = me;
    }
    const uint32_t trid = trid_of(f, f.issued);
    noc_async_read_set_trid(trid);
    const uint32_t ring = get_local_cb_interface(f.cb).fifo_limit - get_local_cb_interface(f.cb).fifo_size;
    uint32_t& slot = g_ring_slot[f.cb];
    uint32_t dst = ring + slot * f.block_bytes;
    const uint32_t txns = f.nr / f.tpt;
    for (uint32_t row = 0; row < f.kb; ++row) {
        uint32_t src = f.addr + ((f.issued * f.kb + row) * f.shard_w + ridx * f.nr) * f.tb;
        for (uint32_t t = 0; t < txns; ++t) {
            noc_async_read_one_packet_with_state_with_trid<false, false>(dram, src, dst, trid);
            src += f.txn_bytes;
            dst += f.txn_bytes;
        }
    }
    slot = slot + 1 == f.slots ? 0 : slot + 1;
    ++f.issued;
}

// push the oldest in-flight block of phase p if its reads landed (blocking = wait for them)
FORCE_INLINE bool complete(Phase& f, bool blocking) {
    if (f.pushed == f.issued) {
        return false;
    }
    const uint32_t trid = trid_of(f, f.pushed);
    if (!blocking && !ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
        return false;
    }
    noc_async_read_barrier_with_trid(trid);
    cb_reserve_back(f.cb, f.block_tiles);  // the space was checked at issue: returns at once
    cb_push_back(f.cb, f.block_tiles);
    ++f.pushed;
    return true;
}

}  // namespace

void kernel_main() {
    const uint32_t bank = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t ridx = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t vc = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint64_t dram = get_noc_addr_from_bank_id<true>(bank, 0);
#if defined(MK_W_DELAY_CYCLES)  // Stage 14's knob: leave the DRAM to mhc_pre's critical-path reads first
    {
        const uint64_t t0 = get_timestamp();
        while (get_timestamp() - t0 < MK_W_DELAY_CYCLES) {
        }
    }
#endif
    MK_MARK();
    load_table<0>();
    uint32_t state_of = MAXP;
    uint32_t cur = 0;
    while (cur < NPH) {
        Phase& f = g_ph[cur];
        if (ridx >= f.rp || f.pushed == f.nb) {  // sits this phase out / done
            MK_MARK();
            ++cur;
            continue;
        }
        bool progressed = complete(f, false);
        if (can_issue(f)) {
            issue(f, dram, ridx, vc, state_of);
            progressed = true;
        } else if (cur + 1 < NPH) {  // lookahead: the next phase's ring, if it is another ring
            Phase& n = g_ph[cur + 1];
            if (n.cb != f.cb && ridx < n.rp) {
                progressed = complete(n, false) || progressed;
                if (can_issue(n)) {
                    issue(n, dram, ridx, vc, state_of);
                    progressed = true;
                }
            }
        }
        if (!progressed && f.issued == f.nb) {
            complete(f, true);  // only completions left in this phase
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_one_packet_set_state<true>(dram, 64, 1);  // firmware default VC (NOC_CTRL persists)
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
