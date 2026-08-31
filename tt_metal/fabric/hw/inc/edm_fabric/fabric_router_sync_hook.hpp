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
// compute_mesh_router_builder.cpp only when the profiler is enabled and
// TT_METAL_PERF_DEBUG_FABRIC_SYNC_HZ is set). With the define absent there is no token from this
// header in the router's translation unit: byte-identity of the stock router binary is by
// construction (and still verified in the prototype report).
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

namespace fabric_router_sync {

using namespace tt::tt_fabric::router_sync;

inline Blk g_blk __attribute__((aligned(16)));
inline uint32_t g_prescaler = 0;
inline uint64_t g_deadline = 0;
// Initiator: last round STARTED. Responder: round of the last doorbell SERVED (compared against the
// 17-bit wire round, so a re-detected doorbell from an aborted round can never re-trigger).
inline uint32_t g_round = 0;
inline uint32_t g_ok = 0;
inline uint32_t g_fail = 0;

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

// One PP_SYNC packet (2 words + at most one 1-word sticky) into this erisc's SPSC lane, through
// kernel_profiler's producer state. Callers reserve ring room for the whole round up front.
inline void emit_sample(uint32_t which, uint32_t round, uint32_t idx, uint32_t hi, uint32_t lo) {
    kernel_profiler::ring_write_sticky_timer(hi);
    kernel_profiler::ring_write_word(
        kernel_profiler::ppfmt::w0(kernel_profiler::ppfmt::T_SYNC, kernel_profiler::spsc_sync_low27(which, round, idx)));
    kernel_profiler::ring_write_word(lo);
    kernel_profiler::publish_tail();
}

// Ring room for one whole round: worst case n * per-sample packets (2 words each) + one sticky.
// `packets_per_sample` is 2 on the initiator (T0+T2), 1 on the responder (T1).
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
template <uint32_t TXQ>
inline void initiator_round(uint32_t n) {
    if (!ring_has_room(4 * n + 1)) {
        g_fail++;
        publish_stat();
        return;
    }
    const uint32_t round = ++g_round & 0x1FFFFu;
    const uint32_t peer = g_blk.cfg.peer_blk;
    const uint64_t nw = g_blk.cfg.next_wait;
    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        // Volatile store + fence: the TXQ engine reads L1 as a SEPARATE agent once CMD is written,
        // and a plain store may sink past the volatile TXQ register writes -- the fence pins the
        // message into L1 before the send command can issue.
        *const_cast<volatile uint32_t*>(&g_blk.tx.tag) = tag(kTagPing, round, i);
        asm volatile("fence" ::: "memory");
        if (!txq_wait_idle<TXQ>(now64() + nw)) {
            ok = false;
            break;
        }
        uint32_t hi, lo;
        kernel_profiler::read_wall_clock(hi, lo);  // t0: immediately before the send command
        internal_::eth_send_packet_bytes_unsafe(
            TXQ, reinterpret_cast<uint32_t>(&g_blk.tx), peer + kPingOff, sizeof(Msg));
        emit_sample(kernel_profiler::SPSC_SYNC_T0, round, i, hi, lo);
        const uint64_t dl = now64() + (i == 0 ? g_blk.cfg.first_wait : nw);
        if (!wait_tag(&g_blk.echo.tag, tag(kTagEcho, round, i), dl)) {
            ok = false;
            break;
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
template <uint32_t TXQ>
inline void responder_service(uint32_t n) {
    invalidate_l1_cache();
    const uint32_t t0tag = g_blk.ping.tag;
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
    const uint32_t peer = g_blk.cfg.peer_blk;
    const uint64_t nw = g_blk.cfg.next_wait;
    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0 && !wait_tag(&g_blk.ping.tag, tag(kTagPing, round, i), now64() + nw)) {
            ok = false;
            break;
        }
        uint32_t hi, lo;
        kernel_profiler::read_wall_clock(hi, lo);  // t1: the ping's observation instant
        emit_sample(kernel_profiler::SPSC_SYNC_T1, round, i, hi, lo);
        *const_cast<volatile uint32_t*>(&g_blk.tx.tag) = tag(kTagEcho, round, i);
        asm volatile("fence" ::: "memory");  // pin the echo into L1 before the TXQ command (see above)
        if (!txq_wait_idle<TXQ>(now64() + nw)) {
            ok = false;
            break;
        }
        internal_::eth_send_packet_bytes_unsafe(
            TXQ, reinterpret_cast<uint32_t>(&g_blk.tx), peer + kEchoOff, sizeof(Msg));
    }
    ok ? g_ok++ : g_fail++;
    publish_stat();
}

// The per-iteration callout. kActive is the compile-time per-erisc gate (the erisc that services
// sender channel 0); on the other erisc this whole function is an empty inline.
template <bool kActive, uint32_t TXQ>
inline void poll() {
    if constexpr (kActive) {
        if (((++g_prescaler) & (FABRIC_ROUTER_SYNC_PRESCALER_MASK)) != 0) [[likely]] {
            return;
        }
        const uint64_t now = now64();
        if (now < g_deadline) [[likely]] {
            return;
        }
        // Deadline: consult the host-writable L1 config -- the only L1 the hook reads between rounds.
        invalidate_l1_cache();
        if (g_blk.cfg.magic != kCfgMagic || (g_blk.cfg.flags & kFlagEnabled) == 0) {
            g_deadline = now + kIdleRecheck;
            return;
        }
        const uint64_t interval =
            (static_cast<uint64_t>(g_blk.cfg.interval_hi) << 32) | g_blk.cfg.interval_lo;
        if (interval == 0) {
            g_deadline = now + kIdleRecheck;
            return;
        }
        // The router never launches again, so nothing re-arms the FW's publish gate; force it so
        // publish_tail() is never a silent no-op on this lane.
        kernel_profiler::zoneValid = true;
        uint32_t n = g_blk.cfg.n_samples;
        if (n == 0 || n > kMaxSamples) {
            n = kMaxSamples;
        }
        if (g_blk.cfg.flags & kFlagInitiator) {
            initiator_round<TXQ>(n);
        } else {
            responder_service<TXQ>(n);
        }
        // `now +`, not `+=`: a long-idle or long-paused router must not replay a backlog of rounds.
        g_deadline = now + interval;
    }
}

// Kernel-start init on the hook erisc: zero the block and publish its address for host discovery.
// Runs before the host profiler exists; the hook stays inert until the host writes a config.
template <bool kActive>
inline void init() {
    if constexpr (kActive) {
        volatile uint32_t* b = reinterpret_cast<volatile uint32_t*>(&g_blk);
        for (uint32_t i = 0; i < sizeof(Blk) / 4; i++) {
            b[i] = 0;
        }
        scratch()[kScratchBlkWord] = reinterpret_cast<uint32_t>(&g_blk);
        scratch()[kScratchStatWord] = 0;
        scratch()[kScratchMagicWord] = kDiscMagic;  // magic LAST: address is valid when magic reads back
    }
}

}  // namespace fabric_router_sync

// clang-format off
#define FABRIC_ROUTER_SYNC_INIT() fabric_router_sync::init<is_sender_channel_serviced[0]>()
#define FABRIC_ROUTER_SYNC_POLL() \
    fabric_router_sync::poll<is_sender_channel_serviced[0], static_cast<uint32_t>(sender_txq_id)>()
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
