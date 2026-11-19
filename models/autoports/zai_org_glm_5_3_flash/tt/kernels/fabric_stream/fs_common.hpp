// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream (tt/fabric_stream.py): the persistent sync words on each chip's lane-0 core. Every counter is
// MONOTONIC across launches (zeroed once by the host): launch g waits for counter >= g x per-launch count. Every stream
// instance has its own sync tensor on its own lane-0 core (the 2x4 shard phase runs 4: AG rounds 1 / 2, RS rounds 1 / 2
// -- tt/tp_dsa_shard.py); ARR words are per (source position, lane) along the instance's axis, so a scatter's sources
// (FS_SCATTER) never share a counter. One `constexpr uint32_t NAME = N;` per line: tt/fabric_stream.py and
// tt/pmk_layer.py parse this header for the host's word indices / buffer sizes.
#pragma once
#include <cstdint>

namespace fs {
constexpr uint32_t WORD = 16;  // one counter per 16 B (NoC atomic / L1 alignment)
// ARR + q LANES + h: chunks landed here from lane h of axis position q, IN ORDER (< 16 words)
constexpr uint32_t ARR = 0;
constexpr uint32_t DONE = 16;  // this chip's lanes: local writes landed (one inc per lane per launch)
constexpr uint32_t GEN = 17;   // launches completed (the waiter's)
// C3 (PMK H0): members' XS-FREE credits, +1 per member per use (32 per lane per use at 64 members: member G credits
// lane G % 2); the lane forwards ONE per-chip credit to its peer
constexpr uint32_t XSFREE = 18;
// THE PER-HOP DRAIN (TP8 hang fix of record, coordinator 2026-10-23 21:55:17Z): on the FABRIC_1D 2x4 mesh a lane's
// slot frees on hop-1 COMPLETION (first-level ack off), so drain_close certifies hop 1 only and a multi-hop packet can
// still sit in a MIDDLE chip's onward router when that chip's program ends. The axis-1 lanes therefore end each launch
// with a row barrier of 1-HOP tokens (never in transit): LANDED = g once every row-incoming packet of this lane landed
// on this chip (the lane NCRISC); BAR_L / BAR_R = the low -> high / high -> low sweeps (+1 per launch from the lower /
// higher neighbour's lane). A lane passes only when every chip of its row has landed everything.
constexpr uint32_t BAR_L = 19;
constexpr uint32_t BAR_R = 20;
constexpr uint32_t LANDED = 21;
// C1 / C3 (FabricStreamC1, the AG into the MM cores' XS), per lane shard:
// LREADY + k: producer slices landed in the XS of this lane's local block k (+1 per slice)
constexpr uint32_t LREADY = 24;
constexpr uint32_t XSADDR = 40;   // the MM cores' XS (CB 35) base address (written by the producers)
constexpr uint32_t SEMADDR = 41;  // the MM cores' FS_ARRIVE semaphore address (written by the producers)
constexpr uint32_t GEN_NC = 42;   // launches done by this lane's NCRISC / BRISC
constexpr uint32_t GEN_B = 43;
constexpr uint32_t OUTADDR = 52;  // C2: the MM cores' OUT (CB 42) base address (written by the producers)
// RS into the members' POST 21 (FabricStreamRSR members=..., PMK H6), per REDUCER shard (reducer k = member group k):
constexpr uint32_t POSTADDR = 53;  // the members' POST (CB 21) base address (written by the members at launch start)
constexpr uint32_t RSEMADDR = 54;  // the members' FS_REDUCED semaphore address (written by the members at launch start)
// the group's POST slots freed: +1 per member at launch start and after each consumed block but the last
// (blocks_per_member per member per launch)
constexpr uint32_t POSTFREE = 55;
// CREDIT + q: launches whose program started on chip q (its lane h credits this lane h)
constexpr uint32_t CREDIT = 44;
// RS with the lane reduce (FabricStreamRSR):
constexpr uint32_t OWNRDY = 64;  // OWNRDY + i: the own partial block i of this lane's reduce list is complete (+1)
constexpr uint32_t GEN_C = 80;   // launches done by this lane's TRISC (unused: TRISC keeps no state)
// TP8 stage 2 (FabricStreamRSL, the row RS of the pair-summed fp32 r1): R1RDY + i = send i of this lane's list may go
// (the stage-1 reducer wrote that r1 block; +1 per launch), 12 sends per lane at most (3 peers x 4 blocks): 81..92
constexpr uint32_t R1RDY = 81;
constexpr uint32_t SYNC_WORDS = 96;
// THE RING-ADDRESS RULE (TP8 hang fix of record, coordinator 2026-10-23 21:14:24Z / 21:19:41Z): a +1 / SET into a
// ring-resident program semaphore of another core goes to the SENDER's own get_semaphore(id) -- one program has ONE
// config-ring layout on every tensix core of every chip (one WorkerConfigBufferMgr per mesh CQ) -- never to an address
// read from a published sync word (a stale publish = the PREVIOUS program's ring slot = a foreign +1 that overshoots
// the next program's equality waits). A published semaphore address is kept only as the FRESHNESS token of the CB
// base published before it (wait until it equals the sender's own address, then read the CB base). The ids are the
// host's (tt/fabric_stream.py SEM_FS_ARRIVE / SEM_FS_REDUCED, asserted against these lines).
constexpr uint32_t SEM_ARRIVE_ID = 15;
constexpr uint32_t SEM_REDUCED_ID = 14;
// rev 60 (review 2 item 4) + the stack45 x 1M merge (16 / 14b, 2026-10-23 16:25Z): the probe flush scratch of
// fs_send.cpp MK_PROBE_ARG (64 B, 64-B aligned from word PROBE_FLUSH: at most 63 + 64 B) in the FREE words 56..63
// between POSTFREE and OWNRDY -- it was 84, inside R1RDY's 81..92 (both fit SYNC_WORDS, nothing caught the overlap)
constexpr uint32_t PROBE_FLUSH = 56;
// every named counter as a word range [lo, lo + n): pairwise disjoint and inside the sync words (the merge lesson: the
// next overlap fails the compile). PROBE_FLUSH's range is 8 words: 64 B from a 64-B aligned start (+ up to 63 B).
namespace detail {
struct Range {
    uint32_t lo, n;
};
constexpr Range RANGES[] = {{ARR, 16},    {DONE, 1},    {GEN, 1},      {XSFREE, 1},   {BAR_L, 1},    {BAR_R, 1},
                            {LANDED, 1},  {LREADY, 16}, {XSADDR, 1},   {SEMADDR, 1},  {GEN_NC, 1},   {GEN_B, 1},
                            {CREDIT, 8},  {OUTADDR, 1}, {POSTADDR, 1}, {RSEMADDR, 1}, {POSTFREE, 1}, {PROBE_FLUSH, 8},
                            {OWNRDY, 16}, {GEN_C, 1},   {R1RDY, 12}};
constexpr bool disjoint() {
    constexpr auto N = sizeof(RANGES) / sizeof(RANGES[0]);
    for (auto i = decltype(N){0}; i < N; i++) {
        if (RANGES[i].lo + RANGES[i].n > SYNC_WORDS) {
            return false;
        }
        for (auto j = i + 1; j < N; j++) {
            if (RANGES[i].lo < RANGES[j].lo + RANGES[j].n && RANGES[j].lo < RANGES[i].lo + RANGES[i].n) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace detail
static_assert(detail::disjoint(), "fs_common.hpp: two named sync-word ranges overlap (or one passes SYNC_WORDS)");

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
inline volatile tt_l1_ptr uint32_t* word(uint32_t base, uint32_t idx) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + idx * WORD);
}
inline void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    WAYPOINT("FSWW");
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(*p - v) < 0);
    WAYPOINT("FSWD");
}
// DRAIN, then close (2026-10-23 18:04:51Z ruling, after three TP8 hangs in the first op fabric CCL after a PMK
// sequence): close() does not wait for the packets still in the router's sender channel -- close_start persists this
// producer's cursor into the EDM for the next opener and tears down. Spin until the router freed every slot this
// connection filled (get_num_free_write_slots() == the channel's buffer count, as the op writers do before they
// disconnect: ccl all_gather unicast_writer.cpp), then close.
// the freshness wait of THE RING-ADDRESS RULE: spin until the published word equals this program's own address
inline void wait_eq(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    WAYPOINT("FSEW");
    do {
        invalidate_l1_cache();
    } while (*p != v);
    WAYPOINT("FSED");
}
template <typename Conn>
inline void drain_close(Conn& conn) {
    WAYPOINT("FSDW");
    while (conn.get_num_free_write_slots() != conn.num_buffers_per_channel) {
    }
    WAYPOINT("FSDD");
    conn.close();
}
#endif
}  // namespace fs
