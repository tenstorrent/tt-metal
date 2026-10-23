// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-exit DFB finish (Quasar firmware).
//
//   Kernel DM (DM2-7), dfb_dm_kernel_exit:
//     1. For each implicit-sync DFB, wait until this DM's reads and writes on the DFB's txn ids have
//        completed (and so are counted in tiles_to_process), then hand its progress to DM0.
//     2. Set its run state to RUN_SYNC_MSG_DFB_FINISHING, which tells DM0 its records are written.
//     3. Wait for its implicit-sync writes to land, then drain: every counter it uses must reach
//        posted == acked.
//   DM0, dfb_dm0_flush_ready_sides, called while it waits for the kernels to finish:
//     Once every DM issuing on an implicit-sync DFB side is ready, credit the batches that never
//     reached the ISR threshold. DM0 is the hart that takes the ISR, so with it masked nothing can
//     race the flush, and no kernel DM waits for another.
//   Compute TRISC (pack/unpack), dfb_trisc_kernel_exit, from trisck.cc: drain.

#pragma once

#include <cstdint>

#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h"
#if defined(COMPILE_FOR_TRISC)
#include "ckernel_trisc_common.h"
#endif

#ifndef COMPILE_FOR_TRISC

// A tile counter read the way an implicit-sync side credits it: posted for a producer, acked for a
// consumer.
inline uint16_t dfb_read_side_counter(dfb::PackedTileCounter ptc, bool is_producer) {
    const uint8_t tensix_id = dfb::get_tensix_id(ptc);
    const uint8_t tc_id = dfb::get_counter_id(ptc);
    return static_cast<uint16_t>(
        is_producer ? overlay::fast_llk_intf_read_posted(tensix_id, tc_id)
                    : overlay::fast_llk_intf_read_acked(tensix_id, tc_id));
}

// Waits until every counter this DM uses for DFB `intf` has posted == acked.
inline void dfb_dm_drain(const LocalDFBInterface& intf) {
    bool all_acked = false;
    while (!all_acked) {
        all_acked = true;
        for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
            const dfb::PackedTileCounter ptc = intf.tc_slots[i].packed_tile_counter;
            const uint8_t tensix_id = dfb::get_tensix_id(ptc);
            const uint8_t tc_id = dfb::get_counter_id(ptc);
            if (overlay::fast_llk_intf_read_acked(tensix_id, tc_id) !=
                overlay::fast_llk_intf_read_posted(tensix_id, tc_id)) {
                all_acked = false;
            }
        }
    }
}

// Kernel DM (DM2-7), after the kernel returns and before it reports done. `run_state` is this DM's
// subordinate_sync byte.
inline void __attribute__((noinline)) dfb_dm_kernel_exit(uint32_t hartid, volatile uint8_t* run_state) {
    const uint32_t dfb_mask = g_dfb_bound_mask;

    // 1. For each implicit-sync DFB, wait until this DM's reads and writes on its txn ids have completed,
    // so tiles_to_process counts all of them, and gather what to hand to DM0. The uncached stores come
    // after the loop: an uncached store in flight blocks this core's cached misses, and the loop reads
    // cached TLS.
    WAYPOINT("DXW");
    uint32_t records[NUM_DFB_POOL_TXN_IDS];
    uint32_t record_mask = 0;    // bit per pool txn id that has a record
    uint32_t implicit_mask = 0;  // bit per implicit-sync DFB
    for (uint32_t m = dfb_mask; m != 0; m &= m - 1) {
        const uint32_t d = static_cast<uint32_t>(__builtin_ctz(m));
        const LocalDFBInterface& intf = g_dfb_interface[d];
        if (intf.num_txn_ids == 0) {
            continue;
        }
        implicit_mask |= 1u << d;
        for (uint8_t i = 0; i < intf.num_txn_ids; i++) {
            const uint32_t trid = intf.txn_ids[i];
            while (__builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(OVERLAY_RD_CMD_BUF, trid) != 0 ||
                   __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(OVERLAY_WR_CMD_BUF, trid) != 0) {
            }
        }
        DFBImplicitProgress& progress = g_dfb_progress[d];
        // An implicit-sync op was left half-issued: its credits can never balance.
        ASSERT(progress.share_pos == 0);
        // Exit records are indexed by a side's lowest txn id, which DM0 also finds (the side's lowest set
        // bit). The host allocates each side's txn ids as one ascending block (TxnIdAllocator::allocate),
        // so txn_ids[0] is the lowest.
        const uint32_t slot = intf.txn_ids[0] - DFB_TXN_ID_BASE;
        records[slot] = static_cast<uint32_t>(progress.loop) | (static_cast<uint32_t>(progress.pos) << 16) |
                        (static_cast<uint32_t>(progress.share) << 24);
        record_mask |= 1u << slot;
        progress = {};
    }

    // 2. Hand the records to DM0, then move to DFB_FINISHING. DM0 flushes a side once all its issuers have
    // reached it. This is separate from DONE because the drain below can wait on DM0's flush of a side.
    volatile uint32_t* record_out =
        dfb_l1_uncached_u32_ptr(reinterpret_cast<uintptr_t>(&g_dfb_exit_record[hartid - dfb::FIRST_KERNEL_DM_HART][0]));
    for (uint32_t m = record_mask; m != 0; m &= m - 1) {
        const uint32_t slot = static_cast<uint32_t>(__builtin_ctz(m));
        record_out[slot] = records[slot];
    }
    asm volatile("fence w, w" ::: "memory");
    *run_state = RUN_SYNC_MSG_DFB_FINISHING;
    WAYPOINT("DXD");

    // 3. Implicit sync credits a write once its data has left L1, so nothing else waits for it to land.
    // Wait here, per txn id (a full write barrier does not cover txn-id writes). On a txn id that only
    // carries reads this is already satisfied.
    for (uint32_t m = implicit_mask; m != 0; m &= m - 1) {
        const LocalDFBInterface& intf = g_dfb_interface[__builtin_ctz(m)];
        for (uint8_t i = 0; i < intf.num_txn_ids; i++) {
            while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, intf.txn_ids[i])) {
            }
        }
    }

    // 4. Drain.
    WAYPOINT("AAW");
    for (uint32_t m = dfb_mask; m != 0; m &= m - 1) {
        dfb_dm_drain(g_dfb_interface[__builtin_ctz(m)]);
    }
    WAYPOINT("AAD");
}

// DM0: credits one counter of an issuer is owed for everything the issuer did: `loop` full txn-id batches
// plus `pos` entries into the next, mod 2^16 like the counter. A batch covers whole round-robin rounds of
// the issuer's counters, so the partial one starts back at counter 0. Ops are spread the way
// commit_implicit_* spreads them: broadcast puts every op on all counters, split gives each counter
// 1/num_tcs of every op, otherwise ops go round-robin, `share` entries at a time.
inline uint16_t dfb_implicit_expected_for_slot(
    uint16_t loop, uint8_t pos, uint8_t share, uint8_t slot, uint8_t num_tcs, uint8_t flags, uint16_t per_tc) {
    if (flags & DFB_DM0_FINISH_FLAG_BROADCAST) {
        return static_cast<uint16_t>(loop * per_tc * num_tcs + pos);
    }
    if (flags & DFB_DM0_FINISH_FLAG_SPLIT) {
        return static_cast<uint16_t>(loop * per_tc + pos / num_tcs);
    }
    uint32_t tail = 0;
    if (pos != 0) {
        const uint32_t round = static_cast<uint32_t>(num_tcs) * share;
        const uint32_t rem = pos % round;
        const uint32_t start = static_cast<uint32_t>(slot) * share;
        tail = (pos / round) * share;
        if (rem > start) {
            tail += (rem - start > share) ? share : rem - start;
        }
    }
    return static_cast<uint16_t>(loop * per_tc + tail);
}

// DM0: the per-trid interrupt-pending bits of `trids`, producer txn ids in the upper half of IP_1,
// consumer txn ids in the lower half of IP_2.
inline uint32_t dfb_dm0_pending_trids(uint32_t trids, bool is_producer) {
    const uint64_t ip =
        is_producer ? __builtin_riscv_ttrocc_cmdbuf_rd_reg(
                          OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8) >>
                          32
                    : __builtin_riscv_ttrocc_cmdbuf_rd_reg(
                          OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return static_cast<uint32_t>(ip) & trids;
}

// DM0: finish one implicit-sync DFB side, the txn ids in `side_trids`, once all of its issuers are ready.
// Every issuer's transactions have completed, so each txn id's last batch is final: a full one has raised
// its interrupt, a partial one never will. Let the ISR take any full batch still pending, then with the
// ISR masked top up each issuer's counters to what it issued, and clear the partial counts so the ISR can
// never credit them again. The ISR runs on this hart, so it is never part-way through while this runs.
inline void __attribute__((noinline)) dfb_dm0_flush_side(uint32_t side_trids, bool is_producer) {
    const uint32_t first_trid = static_cast<uint32_t>(__builtin_ctz(side_trids));
    const volatile TxnDFBDescriptor& desc = g_txn_dfb_descriptor[first_trid];

    while (true) {
        disable_dfb_tile_isr();
        if (dfb_dm0_pending_trids(side_trids, is_producer) == 0) {
            break;
        }
        enable_dfb_tile_isr();  // the ISR takes the pending batch right here, then this checks again
    }
    asm volatile("fence" ::: "memory");

    const uint8_t issuer_mask = desc.finish.issuer_hart_mask;
    ASSERT(issuer_mask != 0);
    const uint8_t num_tcs = static_cast<uint8_t>(desc.num_counters / __builtin_popcount(issuer_mask));
    const uint16_t per_tc = desc.tiles_to_post;
    const uint8_t flags = desc.finish.flags;
    const uint32_t record_slot = first_trid - DFB_TXN_ID_BASE;
    uint8_t counter = 0;
    for (uint32_t h = issuer_mask; h != 0; h &= h - 1) {
        const uint32_t hart = static_cast<uint32_t>(__builtin_ctz(h));
        const uint32_t rec = *dfb_l1_uncached_u32_ptr(
            reinterpret_cast<uintptr_t>(&g_dfb_exit_record[hart - dfb::FIRST_KERNEL_DM_HART][record_slot]));
        const uint16_t loop = static_cast<uint16_t>(rec);
        const uint8_t pos = static_cast<uint8_t>(rec >> 16);
        const uint8_t share = static_cast<uint8_t>(rec >> 24);
        for (uint8_t slot = 0; slot < num_tcs; slot++, counter++) {
            const dfb::PackedTileCounter ptc = desc.tile_counters[counter];
            const uint16_t expected = dfb_implicit_expected_for_slot(loop, pos, share, slot, num_tcs, flags, per_tc);
            const uint16_t actual = dfb_read_side_counter(ptc, is_producer);
            // Modular: negative means the counter is behind what this issuer moved.
            if (static_cast<int16_t>(actual - expected) < 0) {
                const uint8_t tensix_id = dfb::get_tensix_id(ptc);
                const uint8_t tc_id = dfb::get_counter_id(ptc);
                if (is_producer) {
                    overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, static_cast<uint16_t>(expected - actual));
                } else {
                    overlay::fast_llk_intf_inc_acked(tensix_id, tc_id, static_cast<uint16_t>(expected - actual));
                }
            }
        }
    }

    for (uint32_t m = side_trids; m != 0; m &= m - 1) {
        const uint32_t trid = static_cast<uint32_t>(__builtin_ctz(m));
        if (is_producer) {
            __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_tr_ack(OVERLAY_RD_CMD_BUF, trid);
        } else {
            __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_wr_sent(OVERLAY_WR_CMD_BUF, trid);
        }
        asm volatile("nop");  // must give time for the clear to propagate (hw bug, this is needed)
    }
    enable_dfb_tile_isr();
}

// DM0: flushes every DFB side in `pending` (txn ids not yet flushed, of one role) whose issuers are all
// finishing, given `ready` (bit per DM hart), and drops those txn ids from `pending`. The txn ids of one side
// share a descriptor, so the side's first counter identifies them.
inline void __attribute__((noinline)) dfb_dm0_flush_ready_sides(uint32_t& pending, uint8_t ready, bool is_producer) {
    uint32_t unvisited = pending;
    while (unvisited != 0) {
        const uint32_t first_trid = static_cast<uint32_t>(__builtin_ctz(unvisited));
        const dfb::PackedTileCounter side_tc = g_txn_dfb_descriptor[first_trid].tile_counters[0];
        uint32_t side_trids = 0;
        for (uint32_t m = unvisited; m != 0; m &= m - 1) {
            const uint32_t trid = static_cast<uint32_t>(__builtin_ctz(m));
            if (g_txn_dfb_descriptor[trid].tile_counters[0] == side_tc) {
                side_trids |= 1u << trid;
            }
        }
        unvisited &= ~side_trids;
        const uint8_t issuers = g_txn_dfb_descriptor[first_trid].finish.issuer_hart_mask;
        if ((ready & issuers) == issuers) {
            dfb_dm0_flush_side(side_trids, is_producer);
            pending &= ~side_trids;
        }
    }
}

#endif  // !COMPILE_FOR_TRISC

#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK))
// Compute TRISC (pack/unpack), after the kernel returns: wait until every DFB this Neo is bound to has
// drained. TRISC tile_counters[].posted is live occupancy, not a running total, so drained is
// posted == 0. Unpack skips counters another TRISC of this Neo owns.
FORCE_INLINE void dfb_trisc_kernel_exit() {
    WAYPOINT("AAW");
    for (uint32_t m = g_dfb_bound_mask; m != 0; m &= m - 1) {
        const LocalDFBInterface& intf = get_local_dfb_interface(__builtin_ctz(m));
        bool drained = false;
        while (!drained) {
            drained = true;
            for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
#ifdef UCK_CHLKC_UNPACK
                if ((intf.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
                    continue;
                }
#endif
                const uint8_t tc_id = dfb::get_counter_id(intf.tc_slots[i].packed_tile_counter);
                if ((ckernel::trisc::tile_counters[tc_id].f.posted & 0xFFFFu) != 0) {
                    drained = false;
                }
            }
        }
    }
    WAYPOINT("AAD");
}
#endif
