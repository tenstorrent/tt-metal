// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-exit DFB finish
//
//   Kernel DM (DM2-7), dfb_dm_kernel_exit:
//     1. For each implicit-sync DFB, wait until this DM's reads and writes on the DFB's txn ids have
//        completed (and so are counted in tiles_to_process), then hand its progress to DM0.
//     2. Set its run state to RUN_SYNC_MSG_DFB_DRAINING, which tells DM0 its records are written.
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

// The DFBs hart `hart_u8` is bound to this launch, bit per logical id. A Metal 2.0 kernel's come from its
// bindings at compile time (kernel_bindings_generated.h). A legacy kernel reads its participation from the DFB
// config, whose address is 0 when the launch has no DFBs.
FORCE_INLINE uint32_t dfb_bound_mask([[maybe_unused]] uint8_t hart_u8) {
#ifdef KERNEL_BOUND_DFB_MASK
    // Every hart running a kernel participates in each DFB the kernel binds.
    ASSERT(
        KERNEL_BOUND_DFB_MASK == 0 ||
        (g_dfb_config_base_addr != 0 &&
         (KERNEL_BOUND_DFB_MASK & ~dfb_read_participation_mask(g_dfb_config_base_addr, hart_u8)) == 0));
    return KERNEL_BOUND_DFB_MASK;
#else
    const uintptr_t config_base = g_dfb_config_base_addr;
    return config_base == 0 ? 0 : dfb_read_participation_mask(config_base, hart_u8);
#endif
}

#ifndef COMPILE_FOR_TRISC

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

// Kernel DM (DM2-7), from dmk.cc after the kernel returns and before it reports done. `run_state` is
// this DM's subordinate_sync byte.
FORCE_INLINE void dfb_dm_kernel_exit(uint32_t hartid, volatile uint8_t* run_state) {
    const uint32_t dfb_mask = dfb_bound_mask(static_cast<uint8_t>(hartid));

    // 1. For each implicit-sync DFB, wait until this DM's reads and writes on its txn ids have completed,
    // so tiles_to_process counts all of them, and gather what to hand to DM0.
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

    // 2. Hand the records to DM0, then move to DFB_DRAINING. DM0 flushes a side once all its issuers have
    // reached it. This is separate from DONE because the drain below can wait on DM0's flush of a side.
    volatile uint32_t* record_out =
        dfb_l1_uncached_u32_ptr(reinterpret_cast<uintptr_t>(&g_dfb_exit_record[hartid - dfb::FIRST_KERNEL_DM_HART][0]));
    for (uint32_t m = record_mask; m != 0; m &= m - 1) {
        const uint32_t slot = static_cast<uint32_t>(__builtin_ctz(m));
        record_out[slot] = records[slot];
    }
    asm volatile("fence w, w" ::: "memory");
    *run_state = RUN_SYNC_MSG_DFB_DRAINING;
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

// DM0: finish one implicit-sync DFB side, the txn ids in `side_trids`, once all of its issuers are
// draining. Every issuer's transactions have completed, so each txn id's last batch is final: a full one
// has raised its interrupt, a partial one never will. Let the ISR take any full batch still pending, then
// with the ISR masked top up each issuer's counters to what it issued, and clear the partial counts so the
// ISR can never credit them again. The ISR runs on this hart, so it is never part-way through while this
// runs.
inline void dfb_dm0_flush_side(uint32_t side_trids, bool is_producer) {
    while (true) {
        disable_dfb_tile_isr();
        if (dfb_dm0_pending_trids(side_trids, is_producer) == 0) {
            break;
        }
        // The ISR takes the pending batch, possibly a few cycles after unmasking; until it has, the bit stays set
        // and this checks again.
        enable_dfb_tile_isr();
    }
    asm volatile("fence" ::: "memory");

    // Producers credit a counter's posted count, consumers its acked count.
    const uint64_t read_reg = is_producer ? TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR
                                          : TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_ADDR;
    const uint64_t inc_reg = is_producer ? TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_ADDR
                                         : TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__ACKED_REG_ADDR;
    const uint32_t first_trid = static_cast<uint32_t>(__builtin_ctz(side_trids));
    const volatile TxnDFBDescriptor& desc = g_txn_dfb_descriptor[first_trid];
    const uint8_t issuer_mask = desc.finish.issuer_hart_mask;
    ASSERT(issuer_mask != 0);
    const uint32_t num_tcs = desc.num_counters / static_cast<uint32_t>(__builtin_popcount(issuer_mask));
    const uint32_t per_tc = desc.tiles_to_post;
    const uint8_t flags = desc.finish.flags;
    const uint32_t record_slot = first_trid - DFB_TXN_ID_BASE;
    uint8_t counter = 0;
    for (uint32_t h = issuer_mask; h != 0; h &= h - 1) {
        const uint32_t hart = static_cast<uint32_t>(__builtin_ctz(h));
        const uint32_t rec = *dfb_l1_uncached_u32_ptr(
            reinterpret_cast<uintptr_t>(&g_dfb_exit_record[hart - dfb::FIRST_KERNEL_DM_HART][record_slot]));
        const uint32_t loop = rec & 0xFFFFu;
        const uint32_t pos = (rec >> 16) & 0xFFu;
        const uint32_t share = rec >> 24;
        // What each of the issuer's counters is owed: `loop` full txn-id batches plus `pos` entries into the
        // next, mod 2^16 like the counters. Ops are spread the way commit_implicit_* spreads them: broadcast
        // puts every op on all counters, split gives each counter 1/num_tcs of every op, otherwise ops go
        // round-robin `share` entries at a time. A batch covers whole rounds, so the last partial round
        // (`rr_rem` entries) starts back at the issuer's first counter.
        uint32_t owed = loop * per_tc;
        uint32_t rr_rem = 0;
        if (flags & DFB_DM0_FINISH_FLAG_BROADCAST) {
            owed = owed * num_tcs + pos;
        } else if (flags & DFB_DM0_FINISH_FLAG_SPLIT) {
            owed += pos / num_tcs;
        } else if (pos != 0) {
            const uint32_t round = num_tcs * share;
            owed += (pos / round) * share;
            rr_rem = pos % round;
        }
        for (uint32_t slot = 0; slot < num_tcs; slot++, counter++) {
            uint32_t expected = owed;
            if (rr_rem != 0) {
                const uint32_t n = rr_rem < share ? rr_rem : share;
                expected += n;
                rr_rem -= n;
            }
            const dfb::PackedTileCounter ptc = desc.tile_counters[counter];
            const uint64_t tc_base = dfb::get_tensix_id(ptc) * LLK_REG_SIZE +
                                     dfb::get_counter_id(ptc) * COUNTER_REG_SIZE;
            const uint16_t actual = static_cast<uint16_t>(overlay::fast_llk_reg_read(tc_base + read_reg));
            // Modular: negative means the counter is behind what this issuer moved.
            const uint16_t behind = static_cast<uint16_t>(expected - actual);
            if (static_cast<int16_t>(behind) > 0) {
                overlay::fast_llk_reg_write(tc_base + inc_reg, behind);
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
    // A batch may raise its pending bit after the check above. The top-up already credited it, and clearing
    // the count does not clear the bit, so clear the side's bits (W0C) here or the ISR would credit it again.
    if (is_producer) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            OVERLAY_RD_CMD_BUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8,
            ~(static_cast<uint64_t>(side_trids) << 32));
    } else {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            OVERLAY_WR_CMD_BUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8,
            ~static_cast<uint64_t>(side_trids));
    }
    enable_dfb_tile_isr();
}

// DM0: flushes every DFB side in `pending` (txn ids not yet flushed, of one role) whose issuers are all
// draining, given `ready` (bit per DM hart), and drops those txn ids from `pending`. A side's txn ids are one
// ascending block, so its lowest pending txn id and its descriptor's count give the whole side.
inline void __attribute__((noinline)) dfb_dm0_flush_ready_sides(uint32_t& pending, uint8_t ready, bool is_producer) {
    for (uint32_t unvisited = pending; unvisited != 0;) {
        const uint32_t first_trid = static_cast<uint32_t>(__builtin_ctz(unvisited));
        const volatile dfb_dm0_finish_info_t& finish = g_txn_dfb_descriptor[first_trid].finish;
        const uint8_t issuers = finish.issuer_hart_mask;
        const uint8_t num_txn_ids = finish.num_txn_ids;
        ASSERT(num_txn_ids != 0);
        const uint32_t side_trids = ((1u << num_txn_ids) - 1u) << first_trid;
        unvisited &= ~side_trids;
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
    const uint32_t dfb_mask =
        dfb_bound_mask(static_cast<uint8_t>(dfb::TENSIX_RISC_OFFSET + ckernel::csr_read<ckernel::CSR::NEO_ID>()));
    for (uint32_t m = dfb_mask; m != 0; m &= m - 1) {
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
