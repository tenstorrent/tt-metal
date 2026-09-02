// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ---- FABRIC ROUTER SYNC HOOK: in-router device-to-device wall-clock sampling ---------------------
//
// WHAT. A periodic callout in the fabric router's service loop that runs Cristian's algorithm over
// the router's own eth link: the initiator end sends a 16 B ping, the responder stamps its arrival
// and echoes, the initiator stamps the echo. Each stamp is emitted as a self-timestamping PP_SYNC
// packet into this erisc's streaming-profiler SPSC ring -- the packet's own timestamp IS the sample
// -- so the samples ride the ordinary drain (DRISC fillers sweep eth cores too) and a host-side
// aggregator joins T0/T2 (initiator's stream) with T1 (responder's stream) by (round, idx) and
// solves for the per-link clock offset. This is what makes ns-class cross-device alignment work in
// the DEFAULT fabric configuration, where fabric claims every eth channel and nothing else can run
// on the links.
//
// ZERO COST WHEN OFF, BY CONSTRUCTION. The entire hook -- prescaler increment, branch, body -- is
// reachable only through FABRIC_ROUTER_SYNC_INIT() / FABRIC_ROUTER_SYNC_POLL(), which expand to
// NOTHING unless the FABRIC_ROUTER_SYNC_HOOK JIT define is present (injected by
// compute_mesh_router_builder.cpp when the profiler is enabled and
// TT_METAL_PERF_DEBUG_FABRIC_SYNC_HZ is not explicitly 0 -- that var is the opt-OUT; unset means on
// at the 100 Hz default). With the define absent there is no token from this
// header in the router's translation unit: byte-identity of the stock router binary is by
// construction (and verified by disassembly in the prototype report).
//
// WHERE THE SHARED STATE LIVES. The config/message block (Blk) sits at sync_blk_address, carved
// from the EDM builder's own L1 walk and passed in as the SYNC_BLK_ADDR named compile-time arg.
// It must be real L1: the host writes the config over PCIe, the peer router's eth writes land in
// the ping/echo slots, and the local TXQ engine reads the tx staging slot -- kernel .bss lives in
// RISC-LOCAL memory, which none of those three agents can address. Only the hook's private state
// (prescaler, deadline, counters) stays in local memory, where it is fastest.
//
// COST WHEN ON, BETWEEN ROUNDS. Per loop iteration: one counter increment + mask test (the
// prescaler, mask = FABRIC_ROUTER_SYNC_PRESCALER_MASK, a JIT define). Per prescaler expiry: one
// wall-clock read + 64-bit compare against the next deadline. L1 is touched only AT a deadline,
// where the interval is re-read from the host-writable config word -- so the host can retune or
// disable the cadence live, and an unconfigured router settles into a ~ms-cadence one-load check.
//
// NEVER WEDGES, NEVER BLOCKS. Every wait is deadline-bounded (txq drain, echo, next ping); a lost
// packet or absent peer fails the ROUND (counted in the scratch status word), never the router --
// the hook returns to packet servicing and the next round starts clean, keyed by a fresh round
// number, so no stale state survives. Ring room for a whole round is checked up front against the
// drainer's published head; if the profiler is not draining, the round is SKIPPED, not blocked --
// the fabric must never stall on the profiler.
//
// WHO RUNS IT. Both eriscs execute the router main loop; the hook instantiates only on the erisc
// that services sender channel 0 (a compile-time per-risc constant), which is also the erisc that
// legitimately owns sender_txq_id -- so the hook's raw TXQ sends can never collide with the other
// erisc, and its PP_SYNC packets land in that erisc's own SPSC lane. The exchange uses the same
// primitives as the router's data path (bounded eth_txq_is_busy wait + eth_send_packet_bytes_unsafe),
// and the ring writes go through kernel_profiler's OWN producer state (wIndex/publish), so firmware
// markers and hook markers share one coherent ring.
#pragma once

#if defined(FABRIC_ROUTER_SYNC_HOOK) && defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL)

#include "hostdevcommon/fabric_router_sync.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/ethernet/tunneling.h"
#include "eth_l1_address_map.h"

// FABRIC_ROUTER_SYNC_NO_SEND (JIT define, host env TT_METAL_PERF_DEBUG_FABRIC_SYNC_NO_SEND):
// DIAGNOSTIC ONLY. The hook ticks on schedule and does everything it normally does -- deadline
// arithmetic, L1 config reads, state publication, timestamp reads, ring writes, the teardown drain --
// except it never puts a packet on the wire. Rounds then fail on the echo wait, which is expected and
// bounded. This exists to split "the hook's tick path" from "the hook's eth traffic" as the cause of
// the eth core failing to go active again for the NEXT profiler session.
// FABRIC_ROUTER_SYNC_NO_ECHO (host env TT_METAL_PERF_DEBUG_FABRIC_SYNC_NO_ECHO): DIAGNOSTIC ONLY.
// The INITIATOR still transmits -- so the TXQ is still used and the peer's L1 is still written -- but
// the RESPONDER only ever reads and never replies. Halves the traffic and makes it one-directional,
// to test whether one direction alone is enough to leave the eth core unable to go active again.
namespace fabric_router_sync {

using namespace tt::tt_fabric::router_sync;

// Private state: RISC-local memory (fast, and correctly NOT shared with anyone).
inline uint32_t g_prescaler = 0;
// Poll-cadence self-measurement (see GapStats in the shared header): local-memory accumulators,
// sampled at every prescaler expiry where now64() is already in hand, so the measurement is free.
inline uint64_t g_gap_last = 0;
inline uint64_t g_gap_first = 0;
inline uint32_t g_gap_min = 0xFFFFFFFFu;
inline uint32_t g_gap_max = 0;
inline uint32_t g_gap_cnt = 0;
// Latched sample count for a configured RESPONDER, 0 otherwise. Set from the host config on the
// deadline path; the doorbell fast path below reads only this, so it never touches cfg L1.
inline uint32_t g_responder_n = 0;
inline uint64_t g_deadline = 0;
// Initiator: last round STARTED. Responder: round of the last doorbell SERVED (compared against the
// 17-bit wire round, so a re-detected doorbell from an aborted round can never re-trigger).
inline uint32_t g_round = 0;
inline uint32_t g_ok = 0;
inline uint32_t g_fail = 0;
inline uint32_t g_expiries = 0;

// Unconfigured/disabled recheck cadence: ~0.8 ms at 1.35 GHz. One L1 load per recheck.
constexpr uint32_t kIdleRecheck = 1u << 20;

inline uint64_t now64() {
    uint32_t hi, lo;
    kernel_profiler::read_wall_clock(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

inline volatile uint32_t* scratch() {
    return reinterpret_cast<volatile uint32_t*>(eth_l1_mem::address_map::AERISC_FABRIC_SCRATCH_BASE);
}

inline void publish_stat() { scratch()[kScratchStatWord] = (g_ok << 8) | (g_fail & 0xFFu); }

// Deadline-expiry breadcrumb: state in the top nibble, expiry count below. One L1 store per expiry
// (~kHz at the idle recheck, ~the sync cadence once configured) -- a live "the hook is here, in
// this branch" surface the host can read without stopping anything.
inline void publish_state(uint32_t state) {
    scratch()[kScratchDbgWord] = (state << 28) | (++g_expiries & 0x0FFFFFFFu);
}

// The shared block, at the EDM-builder-carved L1 address (SYNC_BLK_ADDR named CT arg).
template <uint32_t BLK>
inline volatile Blk* blk() {
    return reinterpret_cast<volatile Blk*>(BLK);
}

// One PP_SYNC packet (2 words + at most one 1-word sticky) into this erisc's SPSC lane, through
// kernel_profiler's producer state. Callers reserve ring room for the whole round up front.
inline void emit_sample(uint32_t which, uint32_t round, uint32_t idx, uint32_t hi, uint32_t lo) {
#if defined(FABRIC_ROUTER_SYNC_NO_RING)
    // DIAGNOSTIC: send exactly as normal, but write NOTHING into the profiler ring. Every arm so far
    // has varied packets-on-the-wire and records-in-the-ring TOGETHER, so neither has been isolated.
    // The crash symptom points at the ring: the link is physically UP at the wedge (Rx link up 0x1,
    // retrain count 0), the erisc simply never returns to base FW to tick its heartbeat -- which is
    // what a BLOCKED PRODUCER looks like, not a broken link.
    (void)which;
    (void)round;
    (void)idx;
    (void)hi;
    (void)lo;
    return;
#endif
    kernel_profiler::ring_write_sticky_timer(hi);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::w0(
        kernel_profiler::ppfmt::T_SYNC, kernel_profiler::spsc_sync_low27(which, round, idx)));
    kernel_profiler::ring_write_word(lo);
    kernel_profiler::publish_tail();
}

// Ring room for one whole round, against the drainer's published head. Worst case: n samples times
// 2 words per packet (2 packets per sample on the initiator, 1 on the responder) + one sticky.
inline bool ring_has_room(uint32_t words_needed) {
    invalidate_l1_cache();
    const uint32_t head = kernel_profiler::profiler_control_buffer[kernel_profiler::HEAD_INDEX];
    return (kernel_profiler::wIndex - head) <= (kernel_profiler::RING_USABLE - words_needed);
}

// Bounded TXQ-idle wait. eth_send_packet_bytes_unsafe ASSERTS the TXQ command slot is free, so this
// wait is mandatory before every send; a deadline miss fails the round, never the router.
template <uint32_t TXQ>
inline bool txq_wait_idle(uint64_t deadline) {
    while (internal_::eth_txq_is_busy(TXQ)) {
        if (now64() > deadline) {
            return false;
        }
    }
    return true;
}

// Poll a local L1 tag word for an exact value, bounded.
inline bool wait_tag(volatile const uint32_t* w, uint32_t want, uint64_t deadline) {
    while (true) {
        invalidate_l1_cache();
        if (*w == want) {
            return true;
        }
        if (now64() > deadline) {
            return false;
        }
    }
}

// One initiator round: n back-to-back ping/echo trips. Per sample: wait txq idle, stamp t0
// IMMEDIATELY before the send command, send ping(round, i) into the peer's ping slot, emit T0
// (the ring writes overlap the wire flight), wait for echo(round, i), stamp t2, emit T2. A lost
// echo aborts the round -- the already-emitted T0 has no T2/T1 partners and the host join drops
// the key. Sample 0's wait is `first_wait` (it carries the responder's doorbell-notice latency,
// which the min-RTT filter then discards); later samples use the tight `next_wait`.
template <uint32_t TXQ, uint32_t BLK>
// NOINLINE ON PURPOSE. Inlined, this merges its frame into the router kernel_main, which on the
// 2D/Mesh build already sits near the limit: the hook pushed it to 4064 B against
// -Werror=stack-usage=1912 and the router FAILED TO COMPILE (1D fits, 2D does not). The frame is
// the cost, not the data -- the hook keeps its state in RISC-local globals and L1, not on the
// stack. Out of line it gets its own frame, live only during a round (20/s), while poll() stays
// inline so the per-iteration prescaler check keeps costing 6 instructions.
__attribute__((noinline)) void initiator_round(uint32_t n) {
    volatile Blk* b = blk<BLK>();
    if (!ring_has_room(4 * n + 1)) {
        publish_state(8);  // failed: no ring room (profiler not draining)
        g_fail++;
        publish_stat();
        return;
    }
    const uint32_t round = ++g_round & 0x1FFFFu;
    const uint32_t peer = b->cfg.peer_blk;
    const uint64_t first_wait = b->cfg.first_wait;
    const uint64_t nw = b->cfg.next_wait;
    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        // The TXQ engine reads the staged message from L1 as a SEPARATE agent once CMD is written;
        // the fence pins the tag store into L1 before the send command can issue.
        b->tx.tag = tag(kTagPing, round, i);
        asm volatile("fence" ::: "memory");
        if (!txq_wait_idle<TXQ>(now64() + nw)) {
            publish_state(6);  // failed: TXQ never went idle
            ok = false;
            break;
        }
        uint32_t hi, lo;
        kernel_profiler::read_wall_clock(hi, lo);  // t0: immediately before the send command
#if !defined(FABRIC_ROUTER_SYNC_NO_SEND)
        internal_::eth_send_packet_bytes_unsafe(
            TXQ, BLK + kTxOff, peer + kPingOff, sizeof(Msg));
#endif
        emit_sample(kernel_profiler::SPSC_SYNC_T0, round, i, hi, lo);
        const uint64_t dl = now64() + (i == 0 ? first_wait : nw);
        if (!wait_tag(&b->echo.tag, tag(kTagEcho, round, i), dl)) {
            publish_state(7);  // failed: echo never arrived
            ok = false;
#if defined(FABRIC_ROUTER_SYNC_NO_ECHO)
            // NO_ECHO is a VOLUME-MATCHED control, so it must not bail here. With the normal break,
            // an initiator whose echo never comes sends exactly ONE ping per round instead of n --
            // measured 103 packets/link against the enabled arm's 3168, ~31x less. A "clean" result
            // at 1/31 the traffic says nothing about direction, only about volume. Keep sending the
            // remaining samples so the comparison is one-directional-at-similar-volume.
            continue;
#else
            break;
#endif
        }
        kernel_profiler::read_wall_clock(hi, lo);  // t2: immediately after the echo is seen
        emit_sample(kernel_profiler::SPSC_SYNC_T2, round, i, hi, lo);
    }
    ok ? g_ok++ : g_fail++;
    publish_stat();
}

// Responder service: ping 0 of an unserved round IS the doorbell. Serve the whole round: for each
// sample, stamp t1 as close to the ping's arrival as the poll allows, emit T1, echo the tag back.
// The round is marked served the moment it is detected, so an aborted round's stragglers (idx > 0)
// can never re-trigger service.
template <uint32_t TXQ, uint32_t BLK>
// NOINLINE ON PURPOSE. Inlined, this merges its frame into the router kernel_main, which on the
// 2D/Mesh build already sits near the limit: the hook pushed it to 4064 B against
// -Werror=stack-usage=1912 and the router FAILED TO COMPILE (1D fits, 2D does not). The frame is
// the cost, not the data -- the hook keeps its state in RISC-local globals and L1, not on the
// stack. Out of line it gets its own frame, live only during a round (20/s), while poll() stays
// inline so the per-iteration prescaler check keeps costing 6 instructions.
__attribute__((noinline)) void responder_service(uint32_t n) {
    volatile Blk* b = blk<BLK>();
    invalidate_l1_cache();
    const uint32_t t0tag = b->ping.tag;
    if (tag_kind(t0tag) != kTagPing || tag_idx(t0tag) != 0 || tag_round(t0tag) == g_round) {
        return;  // no doorbell: nothing to do this deadline
    }
    g_round = tag_round(t0tag);
    if (!ring_has_room(2 * n + 1)) {
        g_fail++;
        publish_stat();
        return;  // initiator times out sample 0; a logged non-event on both ends
    }
    const uint32_t round = g_round;
    const uint32_t peer = b->cfg.peer_blk;
    const uint64_t nw = b->cfg.next_wait;
    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0 && !wait_tag(&b->ping.tag, tag(kTagPing, round, i), now64() + nw)) {
            publish_state(9);  // responder: ping i never arrived
            ok = false;
            break;
        }
        uint32_t hi, lo;
        kernel_profiler::read_wall_clock(hi, lo);  // t1: the ping's observation instant
        emit_sample(kernel_profiler::SPSC_SYNC_T1, round, i, hi, lo);
        b->tx.tag = tag(kTagEcho, round, i);
        asm volatile("fence" ::: "memory");  // pin the echo into L1 before the TXQ command (see above)
        if (!txq_wait_idle<TXQ>(now64() + nw)) {
            ok = false;
            break;
        }
#if !defined(FABRIC_ROUTER_SYNC_NO_SEND) && !defined(FABRIC_ROUTER_SYNC_NO_ECHO)
        internal_::eth_send_packet_bytes_unsafe(
            TXQ, BLK + kTxOff, peer + kEchoOff, sizeof(Msg));
#endif
    }
    ok ? g_ok++ : g_fail++;
    publish_stat();
}

// The per-iteration callout. kActive is the compile-time per-erisc gate (the erisc that services
// sender channel 0); on the other erisc this whole function is an empty inline.
template <bool kActive, uint32_t TXQ, uint32_t BLK>
inline void poll() {
    if constexpr (kActive) {
#if !defined(FABRIC_ROUTER_SYNC_SLOW_DOORBELL)
        // RESPONDER: EVERY iteration, doorbell only -- the receiver never answers on a timer.
        // The design rule: timers are for the INITIATOR (it schedules rounds); the responder's one
        // job is to answer doorbells, so the doorbell check is the only thing gating its response.
        // History, so nobody re-adds a grid: answering on the prescaler grid made sample-0's wait
        // the PHASE between the two ends' identical expiry periods -- and that phase is a
        // DETERMINISTIC property of topology + launch order (it survived a host reboot: the same
        // two links redrew ~175-254 us at mask 63), so bad links stay bad on every boot. Grid
        // period 4x'd under mask 15 and the "worst" link became the fastest -- phase, not silicon.
        // Answering every iteration removes the grid entirely: worst notice latency = one router
        // loop iteration (~3.2 us loaded, sub-us idle).
        // Cost: unarmed lanes (and initiators) pay one register test; an armed responder pays
        // responder_service()'s no-doorbell entry -- cache invalidate + one L1 tag read + compares.
        // The prescaler below now gates only the initiator machinery and the gap instrument.
        if (g_responder_n != 0) {
            responder_service<TXQ, BLK>(g_responder_n);
        }
#endif
        if (((++g_prescaler) & (FABRIC_ROUTER_SYNC_PRESCALER_MASK)) != 0) [[likely]] {
            return;
        }
        const uint64_t now = now64();
        // Gap tracking BEFORE the deadline early-return: every expiry contributes, not just the
        // ones that cross a deadline. now is already in hand; this is ~6 instructions.
        if (g_gap_last != 0) {
            const uint64_t gap = now - g_gap_last;
            const uint32_t g32 = gap > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(gap);
            if (g32 < g_gap_min) {
                g_gap_min = g32;
            }
            if (g32 > g_gap_max) {
                g_gap_max = g32;
            }
            ++g_gap_cnt;
        } else {
            g_gap_first = now;
        }
        g_gap_last = now;
        if (now < g_deadline) [[likely]] {
            return;
        }
        // Deadline: consult the host-writable L1 config -- the only L1 the hook reads between rounds.
        volatile Blk* b = blk<BLK>();
        invalidate_l1_cache();
        // Publish the poll-cadence stats once per deadline expiry (>= the sync interval apart):
        // 7 L1 stores, read by the host at teardown. Published even when unconfigured -- the host
        // only reads lanes it configured.
        b->gap.min_cy = g_gap_min == 0xFFFFFFFFu ? 0u : g_gap_min;
        b->gap.max_cy = g_gap_max;
        b->gap.cnt = g_gap_cnt;
        b->gap.first_lo = static_cast<uint32_t>(g_gap_first);
        b->gap.first_hi = static_cast<uint32_t>(g_gap_first >> 32);
        b->gap.last_lo = static_cast<uint32_t>(g_gap_last);
        b->gap.last_hi = static_cast<uint32_t>(g_gap_last >> 32);
        if (b->cfg.magic != kCfgMagic || (b->cfg.flags & kFlagEnabled) == 0) {
            publish_state(1);  // unconfigured / disabled by flags
            g_responder_n = 0;  // disarm the doorbell fast path with the rest of the hook
            g_deadline = now + kIdleRecheck;
            return;
        }
        const uint64_t interval = (static_cast<uint64_t>(b->cfg.interval_hi) << 32) | b->cfg.interval_lo;
        if (interval == 0) {
            publish_state(2);  // configured, interval 0 = disabled
            g_responder_n = 0;  // disarm the doorbell fast path with the rest of the hook
            g_deadline = now + kIdleRecheck;
            return;
        }
        // The router never launches again, so nothing re-arms the FW's publish gate; force it so
        // publish_tail() is never a silent no-op on this lane.
        kernel_profiler::zoneValid = true;
        uint32_t n = b->cfg.n_samples;
        if (n == 0 || n > kMaxSamples) {
            n = kMaxSamples;
        }
        if (b->cfg.flags & kFlagInitiator) {
            publish_state(4);
            g_responder_n = 0;
            initiator_round<TXQ, BLK>(n);
        } else {
            publish_state(5);
            g_responder_n = n;  // arm the doorbell fast path
            responder_service<TXQ, BLK>(n);
        }
        // `now +`, not `+=`: a long-idle or long-paused router must not replay a backlog of rounds.
        g_deadline = now + interval;
    }
}

// Kernel-start init on the hook erisc: zero the block and publish its address for host discovery.
// Runs before the host profiler exists; the hook stays inert until the host writes a config.
template <bool kActive, uint32_t BLK>
inline void init() {
    if constexpr (kActive) {
        volatile uint32_t* b = reinterpret_cast<volatile uint32_t*>(BLK);
        for (uint32_t i = 0; i < sizeof(Blk) / 4; i++) {
            b[i] = 0;
        }
        scratch()[kScratchBlkWord] = BLK;
        scratch()[kScratchStatWord] = 0;
        scratch()[kScratchDbgWord] = 0;
        scratch()[kScratchMagicWord] = kDiscMagic;  // magic LAST: address is valid when magic reads back
    }
}

}  // namespace fabric_router_sync

// clang-format off
#define FABRIC_ROUTER_SYNC_INIT() \
    fabric_router_sync::init<is_sender_channel_serviced[0], static_cast<uint32_t>(sync_blk_addr)>()
#define FABRIC_ROUTER_SYNC_POLL()                             \
    fabric_router_sync::poll<                                 \
        is_sender_channel_serviced[0],                        \
        static_cast<uint32_t>(sender_txq_id),                 \
        static_cast<uint32_t>(sync_blk_addr)>()
// clang-format on

#else

// Vanishing form: no token from the hook survives in the translation unit.
#define FABRIC_ROUTER_SYNC_INIT() \
    do {                          \
    } while (0)
#define FABRIC_ROUTER_SYNC_POLL() \
    do {                          \
    } while (0)

#endif  // FABRIC_ROUTER_SYNC_HOOK && PROFILE_KERNEL && !DISPATCH_KERNEL
