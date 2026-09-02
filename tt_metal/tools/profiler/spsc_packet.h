// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The relay's compact profiler packet wire format. Each (core, risc) lane is kept separate end to end, so
// identity is structural and packets carry no core/risc or framing bits. A packet is two 32-bit words:
// word0 = [31:27] type(5) | [26:0] low27, word1 = payload32. Markers carry the 27-bit structural zone id
// (tu_id(17) << 10 | local(10)) and timer_low; timer_hi rides the rare STICKY_TIMER. Plain C: included by the
// producer kernel and the host consumer.

#ifndef SPSC_PACKET_H
#define SPSC_PACKET_H

#include <stdint.h>

// The relay wire's own 5-bit type space, independent of hostdevcommon's PacketTypes: passing a PacketTypes value
// through once collided ZONE_TOTAL(2)/TS_DATA_16B(5) with unrelated codes here and desynchronized the walk.
// 
// Zone family. Both sides keep a per-lane 64-bit cursor = the end of the last S or ATOMIC zone; zones are emitted
// at close, so ends are monotonic per lane and start = end - dur may precede the cursor.
//   ZONE_ATOMIC (3 words): [0] type|id27  [1] end timer_low  [2] duration; re-anchors the cursor.
//   ZONE_S (2 words):      [0] type|id27  [1] end_delta16 << 16 | dur16; end = cursor + delta, cursor = end.
//   ZONE_L (5 words):      [0] type|id27  [1] end_lo [2] end_hi [3] dur_lo [4] dur_hi, for durations past 32 bits;
//                          the cursor is untouched.
//   ZONE_START/END (2 words each): [0] type|id27 [1] timer_low; only the stall zone and the >3.2 s fallback.
// Anchoring on the end is what lets one STICKY_TIMER cover everything after it. No sticky-lo exists: a re-anchor
// is never cheaper than an inline ZONE_ATOMIC, and a stale cursor is merely conservative.
// 
// Stickies (1 word unless noted), each reconstructed on the host from its last-seen value per lane:
//   STICKY_PROG  low27 = runtime host-id; ids >= 2^27 ship as the 2-word STICKY_PROG_EXT with the id in word1.
//   STICKY_TIMER low27 = timer_hi, emitted on each high-half tick (~3.2 s at 1.35 GHz).
//   STICKY_SRC   low27 = word1 = lane id (core*NRISC + risc), injected by the reader when it switches ring.
//   STICKY_META  2 words, timer_hi in low27 and prog_id in word1; the synthetic bench producer only.
#define PP_ZONE_START 0u
#define PP_ZONE_END 1u
#define PP_ZONE_S 3u
#define PP_ZONE_L 4u
#define PP_ZONE_ATOMIC 2u
#define PP_STICKY_META 6u
#define PP_STICKY_SRC 7u

#define PP_STICKY_PROG 8u
#define PP_STICKY_TIMER 9u

// DATA (3 + N words): [0] pp_data_w0(id) [1] timer_low [2] pp_data_w2(size) [3..] payload. word0 is exactly a
// zone marker's word0, so a point marker has the same ELF-resolved identity as a zone; the length has its own
// word so the host advances over any payload without a per-type table.
#define PP_DATA 10u

// EVENT (2 words): [0] pp_event_w0(id) [1] timer_low; a flag with no payload and no size word.
#define PP_EVENT 12u

// ZONE_TOTAL (2 words): word1 is an accumulated duration sum, not a timestamp.
#define PP_ZONE_TOTAL 11u

/* --- PP_DATA word2 sub-fields (word0 is type|id27, identical to a zone marker) --- */
#define PP_DATA_SIZE_SHIFT 25u
#define PP_DATA_SIZE_MASK 0x7Fu /* [31:25] payload length in 32-bit words, 0..127; [24:0] unused, zero */

// BULK_CORE: one bulk NoC read of all NRISC rings. [0] pp_bulkcore_w0(core_id) [1] raw_words
// [2..2+NRISC-1] pp_bulk_meta(head_mod, run) [pad to even], then NRISC contiguous RING_CAP-word ring blocks; the
// host takes ring r's circular [head_mod, head_mod+run) as lane core*NRISC+r.
#define PP_BULK_CORE 5u

// BULK_SPAN: the identity-free whole-core frame, carrying the worker's own control vector so nothing on the wire
// can disagree with the worker. Layout in hostdevcommon/profiler_common.h (SPSC_SPAN_*), which this plain-C
// header cannot include; spsc_marker_decode.hpp asserts the codes agree.
#define PP_BULK_SPAN 13u

/* 2-word STICKY_PROG escape for host-ids >= 2^27 -- see STICKY_PROG above. */
#define PP_STICKY_PROG_EXT 14u

/* --- word0 fields --- */
#define PP_TYPE_SHIFT 27
#define PP_TYPE_MASK 0x1Fu       /* 5 bits */
#define PP_LOW27_MASK 0x7FFFFFFu /* [26:0]: timer_hi (sticky) or the full 27-bit zone id (marker) */

/* --- word1 is a full 32-bit payload (prog_id or timer_low) --- */
#define PP_TIMER_HI_MASK 0x7FFFFFFu /* 27-bit high half (fits low27 of a sticky word0) */

static inline uint32_t pp_word0(uint32_t type, uint32_t low27) {
    return ((type & PP_TYPE_MASK) << PP_TYPE_SHIFT) | (low27 & PP_LOW27_MASK);
}

/* combined sticky, synthetic bench path only */
static inline uint32_t pp_sticky_w0(uint32_t timer_hi) { return pp_word0(PP_STICKY_META, timer_hi & PP_TIMER_HI_MASK); }
static inline uint32_t pp_sticky_w1(uint32_t prog_id) { return prog_id; }

static inline uint32_t pp_prog_w0(uint32_t prog_id) { return pp_word0(PP_STICKY_PROG, prog_id); }
static inline uint32_t pp_prog_ext_w0(void) { return pp_word0(PP_STICKY_PROG_EXT, 0u); }
static inline uint32_t pp_prog_ext_w1(uint32_t prog_id) { return prog_id; }
static inline uint32_t pp_timer_w0(uint32_t timer_hi) { return pp_word0(PP_STICKY_TIMER, timer_hi & PP_TIMER_HI_MASK); }
static inline uint32_t pp_timer_w1(void) { return 0u; }

static inline uint32_t pp_marker_w0(uint32_t type, uint32_t zone_id) { return pp_word0(type, zone_id & PP_LOW27_MASK); }
static inline uint32_t pp_marker_w1(uint32_t timer_low) { return timer_low; }

/* DATA header word0 (type | full 27-bit id) and its separate length word2. */
static inline uint32_t pp_data_w0(uint32_t id) { return pp_word0(PP_DATA, id & PP_LOW27_MASK); }
static inline uint32_t pp_data_w2(uint32_t size_words) {
    return (size_words & PP_DATA_SIZE_MASK) << PP_DATA_SIZE_SHIFT;
}

/* EVENT header: 2 words, no size word. */
static inline uint32_t pp_event_w0(uint32_t id) { return pp_word0(PP_EVENT, id & PP_LOW27_MASK); }

static inline uint32_t pp_type(uint32_t w0) { return (w0 >> PP_TYPE_SHIFT) & PP_TYPE_MASK; }
static inline uint32_t pp_low27(uint32_t w0) { return w0 & PP_LOW27_MASK; }
static inline uint32_t pp_payload32(uint32_t w1) { return w1; }
static inline int pp_is_sticky(uint32_t w0) { return pp_type(w0) == PP_STICKY_META; }
static inline int pp_is_src(uint32_t w0) { return pp_type(w0) == PP_STICKY_SRC; }
static inline int pp_is_prog(uint32_t w0) { return pp_type(w0) == PP_STICKY_PROG; }
static inline int pp_is_timer(uint32_t w0) { return pp_type(w0) == PP_STICKY_TIMER; }
static inline uint32_t pp_prog_id(uint32_t w1) { return w1; }
static inline uint32_t pp_timer_hi(uint32_t w0) { return pp_low27(w0); }
// No pp_is_point(): EVENT is 2 words and DATA is 3 + size, and a walk that advances an EVENT by 2 + size
// desynchronizes silently.
static inline int pp_is_data(uint32_t w0) { return pp_type(w0) == PP_DATA; }
static inline int pp_is_event(uint32_t w0) { return pp_type(w0) == PP_EVENT; }
static inline uint32_t pp_point_id(uint32_t w0) { return pp_low27(w0); }
static inline uint32_t pp_data_size(uint32_t w2) { return (w2 >> PP_DATA_SIZE_SHIFT) & PP_DATA_SIZE_MASK; }
static inline int pp_is_zone_total(uint32_t w0) { return pp_type(w0) == PP_ZONE_TOTAL; }
static inline int pp_is_zone_atomic(uint32_t w0) { return pp_type(w0) == PP_ZONE_ATOMIC; }
/* ZONE_S: word0 = type | id; word1 packs the end's cursor delta (hi16) and the duration (lo16). */
static inline uint32_t pp_zone_s_w0(uint32_t zone_id) { return pp_word0(PP_ZONE_S, zone_id); }
static inline uint32_t pp_zone_s_w1(uint32_t end_delta16, uint32_t dur16) {
    return (end_delta16 << 16) | (dur16 & 0xFFFFu);
}
static inline uint32_t pp_zone_s_delta(uint32_t w1) { return w1 >> 16; }
static inline uint32_t pp_zone_s_dur(uint32_t w1) { return w1 & 0xFFFFu; }

/* ZONE_L: word0 = type | id; then end_lo, end_hi, dur_lo, dur_hi. */
static inline uint32_t pp_zone_l_w0(uint32_t zone_id) { return pp_word0(PP_ZONE_L, zone_id); }

// Wire length in words. Pass 0 for w2 when the type is known not to be DATA; BULK_CORE has its own framing.
static inline uint32_t pp_packet_words(uint32_t w0, uint32_t w2) {
    uint32_t t = pp_type(w0);
    if (t == PP_STICKY_SRC || t == PP_STICKY_TIMER || t == PP_STICKY_PROG) {
        return 1u;
    }
    if (t == PP_DATA) {
        return 3u + pp_data_size(w2);  // word0 + timer_low + size word + payload (self-describing)
    }
    if (t == PP_ZONE_ATOMIC) {
        return 3u;  // word0 + end timer_low + duration
    }
    if (t == PP_ZONE_L) {
        return 5u;  // word0 + end_lo + end_hi + dur_lo + dur_hi
    }
    return 2u;  // ZONE_S, plain zone markers, PP_EVENT, STICKY_PROG_EXT, STICKY_META
}

/* reader-injected source sticky: lane_id = core*NRISC + risc, carried in both words. */
static inline uint32_t pp_src_w0(uint32_t lane_id) { return pp_word0(PP_STICKY_SRC, lane_id); }
static inline uint32_t pp_src_w1(uint32_t lane_id) { return lane_id; }
static inline uint32_t pp_src_lane(uint32_t w0) { return pp_low27(w0); }

static inline uint32_t pp_bulkcore_w0(uint32_t core) { return pp_word0(PP_BULK_CORE, core); }
static inline int pp_is_bulkcore(uint32_t w0) { return pp_type(w0) == PP_BULK_CORE; }
static inline uint32_t pp_bulkcore_core(uint32_t w0) { return pp_low27(w0); }
/* per-risc meta word: head_mod (ring start slot) in hi16, run (valid word count) in lo16 */
static inline uint32_t pp_bulk_meta(uint32_t head_mod, uint32_t run) { return (head_mod << 16) | (run & 0xFFFFu); }
static inline uint32_t pp_bulk_head(uint32_t m) { return (m >> 16) & 0xFFFFu; }
static inline uint32_t pp_bulk_run(uint32_t m) { return m & 0xFFFFu; }

static inline uint32_t pp_bulkspan_w0(void) { return pp_word0(PP_BULK_SPAN, 0u); }
static inline int pp_is_bulkspan(uint32_t w0) { return pp_type(w0) == PP_BULK_SPAN; }

/* reconstruct the 59-bit device timestamp from a marker's 32-bit low + the lane's sticky 27-bit high. */
static inline uint64_t pp_full_ts(uint32_t timer_hi, uint32_t timer_low) {
    return ((uint64_t)(timer_hi & PP_TIMER_HI_MASK) << 32) | (uint64_t)timer_low;
}
static inline uint32_t pp_ts_hi(uint64_t ts) { return (uint32_t)((ts >> 32) & PP_TIMER_HI_MASK); }
static inline uint32_t pp_ts_lo(uint64_t ts) { return (uint32_t)(ts & 0xFFFFFFFFu); }

#endif /* SPSC_PACKET_H */
