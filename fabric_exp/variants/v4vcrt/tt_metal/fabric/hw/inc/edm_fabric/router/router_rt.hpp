// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Round-trip timing of the worker channel's credit loop (v4girt / v4vcrt only; never mainline). One packet at a time
// is followed, every interval measured on one chip's clock:
//   E0 (sending chip)    send -> E0 sees its completion (link out, all of E1, the frame, link back, E0's next look)
//                        -> E0's credit write to the worker done
//   E1 (receiving chip)  at the head of the ring with its data present -> retired (its writes issued)
//                        -> its group credited (the batching wait and the proof) -> the frame carrying it sent
// E0 samples every 61st worker packet (61: not a divisor or multiple of any burst length, so samples fall at every
// position in a burst; job 107 sampled every 64th, always the last packet of a burst of 8 or 32), when no sample is
// open; E1 samples the slot it watches, and after a sample waits
// 16 credits (~64 packets) before watching the slot 5 past the next one to retire, so every ring position is sampled.
// The hot paths pay one compare (E0: per send and per credit; E1: twice per packet, once per credit and per frame);
// the timing itself is out of line, each hook reads the clock first, and every L1 update waits until the sample's last
// timestamp is taken (a first version that timed every credit and sampled every ~5th packet added 24 to
// 47 cycles/packet, RESULTS §98). The results go into this ERISC's phase-profile block, which the builder
// allocates only with TT_FABRIC_TXQ_COUNTERS=1 (consolidation/rt_report.py decodes them); without it every hook
// compiles away. Slots (k = index into events[], phase_ticks[], phase_count[] as one array; each the low word):
//   E0  pkts  worker packets sent (absolute, at the last sample)
//       0 samples   1 send -> seen   2 (unused)   3 packets left in the worker channel at the send
//       4 max send -> seen   5-12 send -> seen histogram, 256-cycle buckets (the last open)
//   E1  pkts  samples
//       0 samples   1 head -> retire   2 retire -> credit   3 credit -> frame   4 packets present at the head
//       5 samples credited on a pass without a packet   6 their retire -> credit
//       7 samples credited in a partial credit (fewer than T slots)   8 their retire -> credit
//       9 sum of the sample's position in its credit (0 = oldest)   10-17 retire -> credit histogram, 128-cycle buckets

#include "router_common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"

#include <cstddef>

namespace fabric_router::rt {

#if defined(FABRIC_RT_ADDR)
constexpr uint32_t buffer = FABRIC_RT_ADDR;  // offline builds: the logged args have no profile buffer
#else
constexpr uint32_t buffer = NAMED_CT_ARG("PHASE_PROFILE_BUFFER_ADDR");
#endif
constexpr bool on = buffer != 0;
constexpr uint32_t block = buffer + MY_ERISC_ID * sizeof(FabricPhaseProfile);
constexpr uint32_t pkts_addr = block + offsetof(FabricPhaseProfile, pkts);
constexpr uint32_t slots_addr = block + offsetof(FabricPhaseProfile, events);
static_assert(offsetof(FabricPhaseProfile, phase_ticks) == offsetof(FabricPhaseProfile, events) + 8 * EVENT_COUNT);
static_assert(offsetof(FabricPhaseProfile, phase_count) == offsetof(FabricPhaseProfile, phase_ticks) + 8 * PHASE_COUNT);
constexpr uint32_t num_slots = EVENT_COUNT + 2 * PHASE_COUNT;

constexpr uint32_t E0_SAMPLE_EVERY = 61;
constexpr uint32_t E1_WATCH_STEP = 5;
constexpr uint32_t E1_COOLDOWN_CREDITS = 16;
constexpr uint32_t NOT_WATCHING = 0xFFFFFFFF;

FORCE_INLINE volatile tt_l1_ptr uint32_t* slot(uint32_t k) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_addr + 8 * k);
}
// Slot k += v as a 64-bit value (low word, then high word).
FORCE_INLINE void add(uint32_t k, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = slot(k);
    const uint32_t low = p[0] + v;
    p[0] = low;
    if (low < v) {
        p[1] = p[1] + 1;
    }
}

FORCE_INLINE uint32_t cycles() {
    uint32_t c = 0;
    if constexpr (on) {
        asm volatile("csrr %0, mcycle" : "=r"(c));
    }
    return c;
}

// E0's open sample: the worker packet's number (packets sent, counting it), 0 when none, its send time and the packets
// left in the worker channel then; and the number of the next packet to sample.
inline uint32_t e0_open = 0;
inline uint32_t e0_t_send = 0;
inline uint32_t e0_queued = 0;
inline uint32_t e0_next = E0_SAMPLE_EVERY;

// E1's sample: the watched slot index, the sample's state and times.
enum E1State : uint32_t { WATCHING = 0, AT_HEAD = 1, RETIRED = 2, CREDITED = 3, COOLING = 4 };
inline uint32_t e1_watch = 0;
inline uint32_t e1_state = WATCHING;
inline uint32_t e1_index = 0;
inline uint32_t e1_ring = 0;
inline uint32_t e1_t_head = 0;
inline uint32_t e1_t_retire = 0;
inline uint32_t e1_t_credit = 0;
inline uint32_t e1_cooldown = 0;
inline uint32_t e1_available = 0;
inline uint32_t e1_position = 0;
inline bool e1_idle = false;
inline bool e1_partial = false;

inline void init() {
    if constexpr (on) {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block);
        for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
            words[i] = 0;
        }
        e0_open = 0;
        e0_next = E0_SAMPLE_EVERY;
        e1_watch = 0;
        e1_state = WATCHING;
    }
}

// ---- E0 --------------------------------------------------------------------------------------------------------------

// The worker channel sent its packet number `sent` (= e0_next); `free_slots` is its free-slots register read now,
// `slots` its ring.
__attribute__((noinline)) inline void e0_sample(uint32_t sent, uint32_t free_slots, uint32_t slots) {
    const uint32_t now = cycles();
    e0_next = sent + E0_SAMPLE_EVERY;
    if (e0_open != 0) {
        return;
    }
    e0_open = sent;
    e0_t_send = now;
    e0_queued = slots - (free_slots & stream_count_mask);
}

// E0 has just read a completion count that covers the open sample (called before the worker's credit write).
__attribute__((noinline)) inline void e0_seen() {
    const uint32_t seen = cycles() - e0_t_send;
    add(0, 1);
    add(1, seen);
    add(3, e0_queued);
    if (seen > *slot(4)) {
        *slot(4) = seen;
    }
    add(5 + (seen / 256 < 7 ? seen / 256 : 7), 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkts_addr) = e0_open;
    e0_open = 0;
}

// ---- E1 --------------------------------------------------------------------------------------------------------------

// Slot `index` is at the head with its data present; `available` packets are present.
__attribute__((noinline)) inline void e1_head(uint32_t index, uint32_t available) {
    const uint32_t now = cycles();
    if (e1_state != WATCHING) {
        return;  // already at the head (a downstream had no room on an earlier pass)
    }
    e1_state = AT_HEAD;
    e1_index = index;
    e1_t_head = now;
    e1_available = available;
}

// The watched slot was retired.
__attribute__((noinline)) inline void e1_retired() {
    e1_t_retire = cycles();
    e1_state = RETIRED;
    e1_watch = NOT_WATCHING;
}

// Slots `first` .. `first + n - 1` of a ring of `ring` slots, groups of `t`, were credited.
__attribute__((noinline)) inline void e1_credited(uint32_t first, uint32_t n, bool progress, uint32_t t, uint32_t ring) {
    const uint32_t now = cycles();
    const uint32_t position = e1_index >= first ? e1_index - first : e1_index + ring - first;
    if (position >= n) {
        return;
    }
    e1_t_credit = now;
    e1_position = position;
    e1_idle = !progress;
    e1_partial = n < t;
    e1_ring = ring;
    e1_state = CREDITED;
}

// A credit frame was sent: the sample is complete; cool down before the next.
__attribute__((noinline)) inline void e1_frame() {
    const uint32_t to_frame = cycles() - e1_t_credit;
    const uint32_t waited = e1_t_credit - e1_t_retire;
    add(0, 1);
    add(1, e1_t_retire - e1_t_head);
    add(2, waited);
    add(3, to_frame);
    add(4, e1_available);
    add(9, e1_position);
    add(10 + (waited / 128 < 7 ? waited / 128 : 7), 1);
    if (e1_idle) {
        add(5, 1);
        add(6, waited);
    }
    if (e1_partial) {
        add(7, 1);
        add(8, waited);
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkts_addr) =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkts_addr) + 1;
    e1_cooldown = E1_COOLDOWN_CREDITS;
    e1_state = COOLING;
}

// The cooldown is over: watch the slot E1_WATCH_STEP past `next_index`, the next to retire.
__attribute__((noinline)) inline void e1_arm(uint32_t next_index, uint32_t ring) {
    e1_watch = (next_index + E1_WATCH_STEP) % ring;
    e1_state = WATCHING;
}

// Hot-path hooks: a compare each when the timing is built in, nothing otherwise.
FORCE_INLINE void e0_credit(uint32_t returned) {
    if constexpr (on) {
        if (e0_open != 0 && static_cast<int32_t>(returned - e0_open) >= 0) {
            e0_seen();
        }
    }
}
FORCE_INLINE void at_head(uint32_t index, uint32_t available) {
    if constexpr (on) {
        if (index == e1_watch) {
            e1_head(index, available);
        }
    }
}
FORCE_INLINE void retired(uint32_t index) {
    if constexpr (on) {
        if (index == e1_watch) {
            e1_retired();
        }
    }
}
template <uint32_t T, uint32_t N>
FORCE_INLINE void credited(uint32_t first, uint32_t n, bool progress, uint32_t next_index) {
    if constexpr (on) {
        const uint32_t state = e1_state;
        if (state == RETIRED) {
            e1_credited(first, n, progress, T, N);
        } else if (state == COOLING && --e1_cooldown == 0) {
            e1_arm(next_index, N);
        }
    }
}
FORCE_INLINE void frame_sent() {
    if constexpr (on) {
        if (e1_state == CREDITED) {
            e1_frame();
        }
    }
}

}  // namespace fabric_router::rt
