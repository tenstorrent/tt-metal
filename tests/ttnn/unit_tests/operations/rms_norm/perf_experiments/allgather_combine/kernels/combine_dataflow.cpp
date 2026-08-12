// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — the rms_norm cross-core statistics COMBINE, and nothing
// else. Not production code; it lives in perf_experiments/ and is never dispatched
// by the op.
//
// Every core of a reduction rectangle starts with `rows_t` bf16 stat tiles already
// resident in L1 (a pinned HEIGHT shard — no reader, no DRAM, no statistics
// pipeline), and the group must land `rsqrt(sum_of_partials + eps)` in every
// member's output shard. Three collectives are compiled from this one file:
//
//   MODE 0  baseline — the op's CURRENT approach, reconstructed faithfully:
//           level-1 gather (member -> row leader), level-2 gather (leader -> root),
//           root-only finalize, then the root multicasts the finished rstd back to
//           the rectangle (mcast_pipe SenderPipe/ReceiverPipe, INCLUDE_SRC loopback).
//           S2 == 1 is the op's FLAT tree: leader == root, one gather level.
//   MODE 1  candidate — mcast ALL-GATHER of the branch sums: each row leader
//           multicasts its level-1 result to the whole rectangle and EVERY core sums
//           the S2 branch tiles and finalizes its own rstd. Deletes the level-2
//           gather rendezvous, the level-2 tile-add, the root-only finalize and the
//           return multicast from the critical path.
//   MODE 2  flat all-gather — no level-1 gather at all: every member multicasts its
//           OWN partial and every core sums G tiles. Tells us whether level 1 is
//           worth keeping.
//
// -----------------------------------------------------------------------------
// RAW-LLK / RAW-NoC SUBSTITUTIONS AND WHY
// -----------------------------------------------------------------------------
//  * The level-1/level-2 GATHER legs are raw `noc_async_write` + semaphore, exactly
//    as the op has them: mcast_pipe is a one-to-many broadcast of ONE buffer to a
//    rectangle, the gather is many-to-one into DISJOINT SLOTS of one destination
//    (mcast_pipe.hpp:44-45 states its precondition as "single sender per receiver,
//    dst_l1 identical on all receivers" — the opposite direction).
//  * The ALL-GATHER leg (MODE 1/2) uses `Noc::async_write_multicast` + a
//    monotone FLAG-WORD arrival protocol instead of SenderPipe/ReceiverPipe.
//    SenderPipe cannot express it:
//      - it owns ONE data_ready cell per pipe and its precondition is a single
//        sender per receiver; here S2 (or G) senders broadcast CONCURRENTLY into
//        disjoint slots of the same landing CB, so every core needs to know that ALL
//        senders have arrived. Serializing them as ROTATING_SENDER rounds is the
//        `mcast_all_gather` of examples/tensix_all_reduce, which is S2x the
//        multicast latency — the opposite of the point.
//      - the Counter data-ready mode would be the natural fit, but its
//        `inc_multicast` has no loopback form ("the multicast sender cannot be part
//        of the multicast destinations", dataflow_api.h:2284) while the sender IS
//        inside its own reduction rectangle.
//    The flag protocol is one 4-byte multicast per sender per block into a
//    HOST-ZEROED, single-writer word (page 1 of the pinned zero-tile shard), read by
//    every core with a plain monotone poll. No atomics, no reset, no
//    sender-in-rect exclusion, and the arrival flag rides the SAME linked multicast
//    chain as the data, so data-before-signal holds without a barrier.
//    The rect/corner math still comes from `McastRect` and the transfer from the
//    `Noc` object — only the handshake is hand-rolled.
//
// -----------------------------------------------------------------------------
// WHY A SENDER'S BLOCK b+1 CANNOT RACE A PEER'S BLOCK b (the argument that
// replaces the baseline's "the return multicast IS the group barrier")
// -----------------------------------------------------------------------------
// cb_stat_gather (level-1 slots, written by a peer's raw NoC write):
//   a member reaches block b+1's gather write only after its block-b all-gather
//   wait returned, i.e. after ALL S2 leaders — including ITS OWN leader —
//   multicast their branch sums for block b. A leader only sends that after its
//   compute produced cb_branch_sum(b), which the combine chain only produces after
//   it has CONSUMED and POPPED cb_stat_gather(b). Same shape as the baseline's
//   argument, with "received the rstd multicast" replaced by "received every
//   branch sum".
// cb_ag (all-gather landing slots, written by S2 concurrent multicasts):
//   depth 1, so block b+1 reuses block b's slots. Explicit back-pressure: after its
//   compute has popped cb_ag(b) (observed as the finalize's output page), EVERY core
//   atomically increments a `ag_free` counter on every SENDER, and a sender waits
//   `b * (G - 1)` before broadcasting block b. A sender's own space is covered by
//   its own local cb_reserve_back. Block 0 needs no ack (the slots are
//   host-allocated and unwritten), which is why a single-block geometry pays nothing
//   for this. Non-leaders reach every leader with ONE `inc_multicast` down the
//   leaders' column (sender outside the rect — the sanctioned form); a leader acks
//   its S2-1 peers with unicast atomics down that same column.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_zero_tile = 4;     // pinned, host-zeroed: page 0 = identity tile, page 1 = flags
constexpr uint32_t cb_stat_partial = 7;  // pinned input shard: num_blocks * rows_t partials
constexpr uint32_t cb_stat_gather = 8;   // level-1 landing (row leader)
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;     // baseline: the root's multicast source
constexpr uint32_t cb_stat_gather2 = 15;  // level-2 landing (root)
constexpr uint32_t cb_out = 16;           // pinned output shard
constexpr uint32_t cb_branch_sum = 18;    // a leader's level-1 result
constexpr uint32_t cb_ag = 19;            // all-gather landing (MODE 1/2)
}  // namespace

using namespace dataflow_kernel_lib;

namespace {
// Monotone arrival poll: every sender's flag word is written by exactly ONE core, so
// no atomicity is involved. Invalidate once per sweep, then read the words back to
// back (the common case is one clean sweep).
FORCE_INLINE void wait_flags(volatile tt_l1_ptr uint32_t* flags, uint32_t n, uint32_t target) {
    while (true) {
        invalidate_l1_cache();
        bool all = true;
        for (uint32_t j = 0; j < n; ++j) {
            if (flags[j] < target) {
                all = false;
                break;
            }
        }
        if (all) {
            return;
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t S1 = get_compile_time_arg_val(1);  // level-1 fan-in (grid row width)
    constexpr uint32_t S2 = get_compile_time_arg_val(2);  // level-2 fan-in (grid rows); 1 == flat
    constexpr uint32_t G = get_compile_time_arg_val(3);   // group members
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(4);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(5);
    constexpr uint32_t SEM_AG_FREE = get_compile_time_arg_val(6);
    constexpr uint32_t MCAST_CT_BASE = 7;
    constexpr uint32_t MCAST_RT_BASE = 15;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr bool TWO_STAGE = S2 > 1;
    // Broadcast senders landing in cb_ag: G (flat all-gather), S2 (leader all-gather),
    // 1 (MODE 4 — the root alone broadcasts the SUM).
    constexpr uint32_t AG_SPAN = (MODE == 2) ? G : ((MODE == 4) ? 1 : S2);
    constexpr bool GATHER_TREE = (MODE == 0) || (MODE == 4);

    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t my_slot = get_arg_val<uint32_t>(2);  // level-1 slot inside my grid row
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    const uint32_t is_root = get_arg_val<uint32_t>(4);
    const uint32_t my_row_slot = get_arg_val<uint32_t>(5);  // level-2 slot == my grid row index
    const uint32_t my_ag_slot = get_arg_val<uint32_t>(6);   // all-gather slot / flag word index
    const uint32_t leader_x = get_arg_val<uint32_t>(7);
    const uint32_t leader_y = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t box_x0 = get_arg_val<uint32_t>(11);  // virtual, min corner
    const uint32_t box_y0 = get_arg_val<uint32_t>(12);
    const uint32_t box_x1 = get_arg_val<uint32_t>(13);  // virtual, max corner (the leaders' column)
    const uint32_t box_y1 = get_arg_val<uint32_t>(14);

    Noc noc;
    Semaphore<> gather_sem(SEM_GATHER);
    [[maybe_unused]] Semaphore<> gather2_sem(SEM_GATHER2);
    [[maybe_unused]] Semaphore<> ag_free(SEM_AG_FREE);

    const uint32_t tb = get_tile_size(cb_stat_partial);
    const uint32_t partial_base = get_read_ptr(cb_stat_partial);
    const uint32_t gather_base = get_write_ptr(cb_stat_gather);
    [[maybe_unused]] const uint32_t gather2_base = get_write_ptr(cb_stat_gather2);
    [[maybe_unused]] const uint32_t ag_base = get_write_ptr(cb_ag);
    // Page 1 of the pinned zero shard: the arrival flag words. HOST-zeroed, which is
    // what makes the monotone protocol safe — a fast sender may write a flag before
    // the receiving core's kernel has even started.
    const uint32_t flags_base = get_read_ptr(cb_zero_tile) + tb;
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base);

    // Hand the identity tile (page 0, host-zeroed) to the combine chain. The op fills
    // this with a 512-word store loop on the writer; the bench pins it over a zero
    // tensor so the fill cost — off the op's critical path by design — cannot
    // contaminate the collective's number, identically in every variant.
    cb_push_back(cb_zero_tile, 1);

    // The broadcast rectangle. McastRect owns the normalization and the per-NoC
    // routing-corner order; only the handshake below is hand-rolled.
    const McastRect<> box(box_x0, box_y0, box_x1, box_y1);

    auto mcast_bytes = [&](uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool linked) {
        const auto& r = box.bounds();
        UnicastEndpoint src_ep;
        MulticastEndpoint dst_ep;
        const typename noc_traits_t<UnicastEndpoint>::src_args_type src_args{.addr = src_l1};
        const typename noc_traits_t<MulticastEndpoint>::dst_args_mcast_type dst_args{r.sx, r.sy, r.ex, r.ey, dst_l1};
        // INCLUDE_SRC: the sender is inside its own reduction rectangle and needs its
        // own slot filled through the same path as everyone else's.
        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(src_ep, dst_ep, size, G, src_args, dst_args, linked);
    };

    // Constructed ONCE, before the loop (the ReceiverPipe ctor kernel-inits its
    // data_ready cell and must not re-run after a broadcast has started). Both faces
    // are built on every core: the ctors have no remote side effects, and only the
    // root ever calls send() / only a non-root ever calls receive().
    [[maybe_unused]] auto sender = mc.sender(noc);
    [[maybe_unused]] auto receiver = mc.receiver(noc);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t src = partial_base + b * rows_t * tb;

        // MODE 3 — ABLATION, not a candidate: the collective's PAYLOAD removed
        // entirely (no gather, no broadcast, no cross-core wait), leaving the program
        // launch + the finalize chain. Its answer is wrong by design; it exists to say
        // how much of a variant's wall is the collective at all.
        if constexpr (MODE == 3) {
            cb_reserve_back(cb_stat_partial, rows_t);
            cb_push_back(cb_stat_partial, rows_t);
            {
                cb_wait_front(cb_out, rows_t);
            }
            cb_pop_front(cb_out, rows_t);
            continue;
        }

        // ---- level 1: every member -> its grid row's leader --------------------
        if constexpr (MODE != 2) {
            {
                for (uint32_t r = 0; r < rows_t; ++r) {
                    const uint32_t dst = gather_base + (r * S1 + my_slot) * tb;
                    noc_async_write(src + r * tb, get_noc_addr(leader_x, leader_y, dst), tb);
                }
            }
            {
                noc_async_write_barrier();
            }
            if constexpr (S1 > 1) {
                if (!is_leader) {
                    gather_sem.up(noc, leader_x, leader_y, 1);
                }
            }
            if (is_leader) {
                cb_reserve_back(cb_stat_gather, rows_t * S1);
                if constexpr (S1 > 1) {
                    gather_sem.wait_min((b + 1) * (S1 - 1));
                }
                cb_push_back(cb_stat_gather, rows_t * S1);
            }
        }

        if constexpr (GATHER_TREE) {
            // ---- level 2: every leader -> the root ----------------------------
            if constexpr (TWO_STAGE) {
                if (is_leader) {
                    {
                        cb_wait_front(cb_branch_sum, rows_t);
                    }
                    const uint32_t bsrc = get_read_ptr(cb_branch_sum);
                    {
                        for (uint32_t r = 0; r < rows_t; ++r) {
                            const uint32_t dst = gather2_base + (r * S2 + my_row_slot) * tb;
                            noc_async_write(bsrc + r * tb, get_noc_addr(root_x, root_y, dst), tb);
                        }
                        noc_async_write_barrier();
                    }
                    cb_pop_front(cb_branch_sum, rows_t);
                    if (!is_root) {
                        gather2_sem.up(noc, root_x, root_y, 1);
                    }
                    if (is_root) {
                        cb_reserve_back(cb_stat_gather2, rows_t * S2);
                        gather2_sem.wait_min((b + 1) * (S2 - 1));
                        cb_push_back(cb_stat_gather2, rows_t * S2);
                    }
                }
            }
            // ---- the root's ONE broadcast ------------------------------------
            // MODE 0: the FINISHED rstd, straight into the output shard.
            // MODE 4: the raw SUM into cb_ag, and every core finalizes it itself.
            //         Same broadcast count and same tree as the baseline; what moves is
            //         only WHERE the finalize runs. The root's finalize (measured at
            //         ~720 ns per stat tile, and 11.9 us on the BLOCK-sharded geometry)
            //         leaves the critical path and runs concurrently on all G cores.
            constexpr uint32_t land_cb = (MODE == 0) ? cb_out : cb_ag;
            constexpr uint32_t send_cb = (MODE == 0) ? cb_rstd_send : cb_stat_sum;
            cb_reserve_back(land_cb, rows_t);
            const uint32_t land_dst = get_write_ptr(land_cb);
            if (is_root) {
                {
                    cb_wait_front(send_cb, rows_t);
                }
                {
                    sender.send(get_read_ptr(send_cb), land_dst, rows_t * tb);
                }
                cb_pop_front(send_cb, rows_t);
            } else {
                receiver.receive();
            }
            cb_push_back(land_cb, rows_t);
        } else {
            // ---- the all-gather: senders broadcast, everybody finalizes -------
            const bool i_send = (MODE == 2) ? true : (is_leader != 0);
            if (i_send) {
                uint32_t send_src = src;
                if constexpr (MODE == 1) {
                    cb_wait_front(cb_branch_sum, rows_t);
                    send_src = get_read_ptr(cb_branch_sum);
                }
                if (b) {
                    // Slot reuse back-pressure (see the header note). Never entered on
                    // a single-block geometry.
                    ag_free.wait_min(b * (G - 1));
                }
                {
                    for (uint32_t r = 0; r < rows_t; ++r) {
                        mcast_bytes(send_src + r * tb, ag_base + (r * AG_SPAN + my_ag_slot) * tb, tb, /*linked=*/true);
                    }
                    // The arrival flag rides the tail of the same linked chain, so it
                    // cannot overtake the data it announces.
                    flags[my_ag_slot] = b + 1;
                    mcast_bytes(flags_base + 4 * my_ag_slot, flags_base + 4 * my_ag_slot, 4, /*linked=*/false);
                }
                {
                    // A full write BARRIER, not `async_writes_flushed()`: a sender
                    // observes its OWN arrival through the same flag word it wrote
                    // LOCALLY (the flag multicast's source), so the linked
                    // data-before-flag ordering that protects every remote core does
                    // not protect the sender's own loopback slot. The barrier is what
                    // proves the loopback copy landed before wait_flags() lets this
                    // core's compute read the slot. It also frees the source
                    // (cb_branch_sum) for reuse.
                    noc_async_write_barrier();
                }
                if constexpr (MODE == 1) {
                    cb_pop_front(cb_branch_sum, rows_t);
                }
            }
            {
                cb_reserve_back(cb_ag, rows_t * AG_SPAN);
                wait_flags(flags, AG_SPAN, b + 1);
                cb_push_back(cb_ag, rows_t * AG_SPAN);
            }
        }

        // The op's writer drains compute's output block here (store_block); the bench
        // keeps the same dependency — it is what proves this core's compute is done
        // with the collective's buffers — but the output is already in its final home
        // (a pinned shard), so there is no NoC traffic.
        {
            cb_wait_front(cb_out, rows_t);
        }
        cb_pop_front(cb_out, rows_t);

        // ---- "my all-gather slots are free" ack ---------------------------------
        if constexpr (MODE == 1) {
            if (b + 1 < num_blocks) {
                if (is_leader) {
                    // The S2-1 other leaders, down the box's right-hand column.
                    const uint32_t my_y = box_y0 + my_row_slot;
                    for (uint32_t y = box_y0; y <= box_y1; ++y) {
                        if (y != my_y) {
                            ag_free.up(noc, box_x1, y, 1);
                        }
                    }
                } else if constexpr (S2 == 1) {
                    ag_free.up(noc, leader_x, leader_y, 1);
                } else {
                    // One multicast atomic down the leaders' column; the sender is a
                    // non-leader, hence outside the rect (the sanctioned form).
                    ag_free.inc_multicast(noc, box_x1, box_y1, box_x1, box_y0, 1, S2);
                }
                noc.async_atomic_barrier();
            }
        }
    }
}
