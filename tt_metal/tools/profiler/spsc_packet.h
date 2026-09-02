/*
 * spsc_packet.h - relay compact profiler packet wire format.
 *
 * Each (core,risc) lane is kept separate end to end (L1 ring -> per-lane SPSC -> per-lane host slot),
 * so identity is structural (the host slot position is the lane) and the packet carries no core/risc
 * and no header/framing bits.
 *
 * A packet is 2x 32-bit words = 8 B:
 *   word0:  [31:27] type(5)   [26:0] low27
 *   word1:  [31:0]  payload32
 *
 *   STICKY_META : low27 = timer_hi(27)             payload32 = prog_id(32)
 *   marker      : low27 = zone_id(27b structural: tu_id(17)<<10|local(10))  payload32 = timer_low(32)
 *
 * Timer split is at bit 32:
 *   timer_low = ts & 0xFFFFFFFF ;  timer_hi = ts >> 32 ;  full = (timer_hi<<32) | timer_low  (59-bit)
 * timer_hi is carried only by the (rare) sticky: the producer emits one at kernel start and a new one
 * whenever timer_hi ticks (~3.2 s at 1.35 GHz). The host keeps per-lane running (timer_hi, prog).
 *
 * Framing is positional: packets are 2-word aligned in each lane's stream and the host knows the exact
 * word count, so there is no valid/header bit and no pad slots to skip.
 *
 * Plain C header: included by the worker producer kernel (C++) and the host consumer (C++).
 */
#ifndef SPSC_PACKET_H
#define SPSC_PACKET_H

#include <stdint.h>

/* packet_type field (5 bits). This is the relay wire's own type space, independent of hostdevcommon's
 * PacketTypes (the DRAM readback path): the two sources never co-exist and their host decoders share no
 * code, so no value here needs to agree with a PacketTypes value. Do not pass a PacketTypes value
 * through onto this wire; that is what made ZONE_TOTAL(2) and TS_DATA_16B(5) collide with unrelated
 * types here (5 is PP_BULK_CORE) and silently desynchronize the whole packet walk. The producer maps its
 * marker kind to these codes explicitly (see ppfmt in kernel_profiler.hpp). */
#define PP_ZONE_START 0u
#define PP_ZONE_END 1u
/* ZONE_S: the small end of the variable-width zone family around ZONE_ATOMIC. Both sides keep a
 * per-lane 64-bit cursor = the end of the last S or ATOMIC zone on that lane. Ends are monotonic per
 * lane (zones are emitted at close, in end order), so an end-relative delta is unsigned; a zone's start
 * may freely precede the cursor (a closing parent), since start is always reconstructed as end - dur.
 *   ZONE_S (2 words): [0] type|id27  [1] end_delta16 << 16 | dur16
 *       end = cursor + delta ; start = end - dur ; cursor = end. The dense-zone hot case: end within
 *       ~48 us of the previous end and duration <= ~48 us (@1.35 GHz). The 64-bit cursor add crosses
 *       the 2^32 lo-wrap for free, so ZONE_S never needs a sticky.
 *   ZONE_L (5 words): [0] type|id27  [1] end_lo  [2] end_hi  [3] dur_lo  [4] dur_hi
 *       Two full 64-bit values, for the >3.2 s case.
 * There is deliberately no sticky-lo: a re-anchor is never cheaper as a separate packet than inline in a
 * 3-word ZONE_ATOMIC, which carries a zone as well. Only S and ATOMIC advance the cursor, on producer
 * and decoder identically, so a stale cursor is merely conservative: the next S falls back to ATOMIC
 * when its delta overflows 16 bits. */
#define PP_ZONE_S 3u
#define PP_ZONE_L 4u
/* 3-word complete zone: w0 = type|id27, w1 = end timer_low, w2 = duration in cycles. Anchored on the
 * end, not the start, because records leave the producer in completion order: ends are monotonic per
 * lane, so the STICKY_TIMER contract (one timer_hi covers everything after it) holds, while starts are
 * not monotonic and would break it. The host recovers start = full_end - duration. A duration past 32
 * bits (>~3.2 s at 1.35 GHz) is emitted as a START/END pair instead, so this word never needs widening. */
#define PP_ZONE_ATOMIC 2u
/* STICKY_META: combined sticky carrying both timer_hi (low27) and prog_id (payload32) in one packet.
 * Emitted only by the synthetic bench producer in producer_common.h; the kernel_profiler path splits
 * identity into the three separate stickies below. */
#define PP_STICKY_META 6u
/* STICKY_SRC: (core,risc) identity, injected by the reader into the linearized stream whenever it
 * switches to a new source ring. Everything up to the next STICKY_SRC belongs to this lane.
 * low27 = lane_id (core*NRISC + risc); payload32 = lane_id (redundant self-check). */
#define PP_STICKY_SRC 7u

/* kernel_profiler path: three separate stickies, each produced or injected at its own stage. The stream
 * is reconstructed on the host from the last-seen value of each; all three persist until the next update.
 *
 *   STICKY_PROG  : the runtime host-id (the per-program id ttnn assigns, the same value the DRAM profiler
 *                  uses). Emitted by every RISC at its launch point. 1 word: low27 = host-id. Ids that
 *                  outgrow 27 bits (134M launches) ship as STICKY_PROG_EXT instead: 2 words,
 *                  payload32 = full host-id, low27 unused (0).
 *   STICKY_TIMER : timer_hi, the high half of the device wall clock. Emitted by any risc at a marker
 *                  record whenever its high half ticks over. 1 word: low27 = timer_hi.
 *   STICKY_SRC   : (core,risc) lane identity, injected by the relay reader hart. 1 word: low27 = lane.
 *
 * The linearized stream is therefore variable-length: SRC/TIMER/PROG are 1 word, markers and PROG_EXT
 * are 2. The decoder advances by pp_packet_words(); SENT is always published on a packet boundary. (The
 * synthetic bench uses a fixed 2-word SRC and never calls pp_packet_words.)
 *
 * A marker (ZONE_START/END) is minimal: low27 = the full 27-bit structural zone id
 * (tu_id << TT_ZONE_LOCAL_BITS | local, hostdevcommon/profiler_zone_id.h), payload32 = timer_low. The
 * host binds each marker to the last-seen PROG, TIMER and SRC to reconstruct the full record. */
#define PP_STICKY_PROG 8u
#define PP_STICKY_TIMER 9u

/* DATA: a point-in-time marker carrying a payload. 3 + N words:
 *   [0] word0 = pp_data_w0(id)   [1] timer_low   [2] word2 = pp_data_w2(size)   [3 .. 3+size-1] payload
 * word0 is exactly a zone marker's word0, type(5) | the full 27-bit structural id, so a point marker has
 * the same compile-time source-location identity, and the same ELF-resolved name, as any zone; that is
 * why the payload length lives in its own word2 rather than sharing low27 with the id. The length still
 * travels in the packet, so the host advances over any payload without a per-type length table. */
#define PP_DATA 10u

/* EVENT: a point-in-time flag, no payload and therefore no size word. Exactly 2 words, the same shape as
 * a zone marker:
 *   [0] word0 = pp_event_w0(id)   [1] timer_low
 * Its id is a compile-time structural id like any other, so an EVENT is named from the ELF too. */
#define PP_EVENT 12u

/* ZONE_TOTAL: an accumulated-duration zone (DO_SUM / profileScopeAccumulate). 2 words, but word1 is the
 * accumulated sum, not a timer: the host must not treat it as a timestamp. */
#define PP_ZONE_TOTAL 11u

/* --- PP_DATA word2 sub-fields (word0 is type|id27, identical to a zone marker) --- */
#define PP_DATA_SIZE_SHIFT 25u
#define PP_DATA_SIZE_MASK 0x7Fu /* [31:25] payload length in 32-bit words, 0..127; [24:0] unused, zero */

/* BULK_CORE: raw-bulk frame for the whole core, emitted by the reader when it does one bulk NoC read of
 * all 5 rings. The host splits it into the 5 lanes. Frame layout in the stream (all 32-bit words):
 *   [0] word0 = pp_bulkcore_w0(core_id)   [1] word1 = raw_words (= NRISC*RING_CAP)
 *   [2..2+NRISC-1] per-risc meta: pp_bulk_meta(head_mod, run)   [pad to even]
 *   [prefix..prefix+raw_words) = NRISC contiguous RING_CAP-word ring blocks (ring r @ +r*RING_CAP), RAW
 * Host: for risc r, lane=core*NRISC+r, extract the circular [head_mod, head_mod+run) words of ring r (the
 * raw over-read past tail is present but the host takes only `run`), then decode as that lane's stream. */
#define PP_BULK_CORE 5u

/* BULK_SPAN: the identity-free whole-core frame. Where BULK_CORE has the reader tell the host which core
 * and how much (core_id in word0, per-risc head/run meta), a SPAN frame carries the worker's own profiler
 * control vector verbatim: identity from SPSC_CORE_XY, progress from the heads, extent from the tails.
 * The relay injects nothing, so nothing on the wire can disagree with the worker. Layout and the shared
 * slice geometry live in hostdevcommon/profiler_common.h (SPSC_SPAN_*), which this plain-C header cannot
 * include; spsc_marker_decode.hpp static_asserts that the codes agree. */
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

/* split stickies. STICKY_PROG is 1 word (host-id rides low27); ids past 2^27 go out as the 2-word
 * PP_STICKY_PROG_EXT with the full id in word1. */
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
static inline int pp_is_data(uint32_t w0) { return pp_type(w0) == PP_DATA; }
static inline int pp_is_event(uint32_t w0) { return pp_type(w0) == PP_EVENT; }
/* There is deliberately no pp_is_point(): the two point-marker types do not share a shape, so a single
 * "is it a point marker" test would invite sizing them identically. EVENT is always 2 words; DATA is 3 +
 * the size in its word2. A walk that advances an EVENT by 2 + size does not error, it desynchronizes from
 * that packet onward and produces plausible garbage. Branch on pp_is_event / pp_is_data separately. */
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

/* Wire length (32-bit words) of a packet: SRC/TIMER/PROG are 1 word (identity/timer_hi/host-id fit in
 * low27, no payload); zone markers, EVENT, PROG_EXT and META are 2; DATA is 3 + payload, whose length
 * lives in word2, which is why this takes w2 as well. Pass 0 for w2 when the type is known not to be
 * DATA. BULK_CORE has its own framing and must not be passed here. SENT is always published on a packet
 * boundary, so a decoder that advances by this length stays in sync. */
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
