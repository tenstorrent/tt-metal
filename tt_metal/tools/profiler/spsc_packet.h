/*
 * spsc_packet.h - drainer compact profiler packet wire format.
 *
 * Each (core,risc) lane is kept SEPARATE end to end (L1 ring -> per-lane SPSC -> per-lane host
 * slot), so identity is structural (the host slot position IS the lane) and the packet carries NO
 * core/risc and NO header/framing bits. That frees every bit for payload and drops the identity
 * stamping + header OR from the producer hot path.
 *
 * A packet is 2x 32-bit words = 8 B:
 *   word0:  [31:27] type(5)   [26:0] low27
 *   word1:  [31:0]  payload32
 *
 *   STICKY_META : low27 = timer_hi(27)             payload32 = prog_id(32)
 *   marker      : low27 = zone_srcloc(16b hash, 11 spare)   payload32 = timer_low(32)
 *
 * Timer split is at bit 32 (no header bit to dodge):
 *   timer_low = ts & 0xFFFFFFFF ;  timer_hi = ts >> 32 ;  full = (timer_hi<<32) | timer_low  (59-bit)
 * timer_hi is carried only by the (rare) sticky: producer emits one sticky at kernel start and a new
 * one whenever timer_hi ticks (~3.2 s at 1.35 GHz). Host keeps per-lane running (timer_hi, prog).
 *
 * Framing is positional: packets are 2-word aligned in each lane's stream and the host knows the exact
 * word count, so there is no valid/header bit -- there are no pad slots to skip.
 *
 * Plain C header: included by the worker producer kernel (C++) and the host
 * consumer (C++). No dependencies.
 */
#ifndef SPSC_PACKET_H
#define SPSC_PACKET_H

#include <stdint.h>

/* packet_type field (5 bits). This is the drainer wire's OWN type space -- deliberately INDEPENDENT of
 * hostdevcommon's PacketTypes, which belongs to the DRAM readback path. The two data sources never
 * co-exist and their host decoders share no code, so no value here needs to agree with a PacketTypes
 * value: ZONE_START/END coincide at 0/1 only by history. Do NOT reintroduce a 3-bit pass-through of a
 * PacketTypes value onto this wire -- that is what made ZONE_TOTAL(2) and TS_DATA_16B(5) collide with
 * unrelated types here (5 is PP_BULK_CORE), silently desynchronizing the whole packet walk. The producer maps
 * its logical marker kind to these codes explicitly (see ppfmt in kernel_profiler.hpp). */
#define PP_ZONE_START 0u
#define PP_ZONE_END 1u
/* STICKY_META (LEGACY / synthetic bench path only): combined sticky carrying BOTH timer_hi(low27) and
 * prog_id(payload32) in one packet. Emitted by the throwaway producer_common.h stand-in. The REAL
 * kernel_profiler path does NOT use this -- it splits identity into three separate stickies below
 * (PROG / TIMER produced at different stages, SRC injected by the reader). Kept so the untouched
 * synthetic drain benchmark keeps decoding. */
#define PP_STICKY_META 6u
/* STICKY_SRC: (core,risc) identity, injected by the READER into the LINEARIZED stream whenever it
 * switches to a new source ring. Everything after it (until the next STICKY_SRC) belongs to this lane.
 * low27 = lane_id (core*NRISC + risc); payload32 = lane_id (redundant/self-check). */
#define PP_STICKY_SRC 7u

/* --- REAL kernel_profiler path: three separate stickies, each produced/injected at its own stage ---
 * The producer no longer stamps identity into every marker; the stream is reconstructed on the host
 * from the last-seen value of each sticky (all three are "sticky": they persist until the next update).
 *
 *   STICKY_PROG  : the runtime host-id (per-program unique id ttnn assigns; same value the DRAM
 *                  profiler uses). Emitted ONCE at BRISC FW start (and on program change). 2 WORDS:
 *                  payload32 = host-id (needs a full 32 bits); low27 unused (0).
 *   STICKY_TIMER : timer_hi -- the high half of the device wall-clock. Emitted by ANY risc at a marker
 *                  record whenever its high half ticks over. 1 WORD: low27 = timer_hi (fits 27 bits, no
 *                  payload word).
 *   STICKY_SRC   : (core,risc) lane identity -- injected by the drainer reader hart. 1 WORD: low27 = lane.
 *
 * The real linearized stream is therefore VARIABLE-LENGTH: SRC/TIMER are 1 word, markers + PROG are 2.
 * The decoder advances by pp_packet_words(); SENT is always published on a packet boundary. (The frozen
 * synthetic bench predates this and uses a fixed 2-word SRC -- it never calls pp_packet_words.)
 *
 * A marker (ZONE_START/END) is minimal: low27 = zone srcloc (16-bit hash for now, room to 27),
 * payload32 = timer_low. Host binds each marker to the last-seen PROG (prog), TIMER (timer_hi) and
 * SRC (lane) to reconstruct the full record. */
#define PP_STICKY_PROG 8u
#define PP_STICKY_TIMER 9u

/* DATA: a point-in-time event carrying an OPTIONAL payload -- the unified EVENT/DATA packet. An "event"
 * is just size==0, so there is one code and one decode path for both. SELF-DESCRIBING LENGTH: the word
 * count lives in the header, so the host advances correctly over any payload without a per-type length
 * table, and a future payload shape needs no decoder change. 2 + N words:
 *   [0] word0 = pp_data_w0(id, size)   [1] timer_low   [2 .. 2+size-1] payload
 * low27 = size(7) << 20 | id(20). The id sits in the LOW bits so the existing `pp_low27(w0) & 0xFFFF`
 * still yields the 16-bit hash, and a plain 2-word marker reads as size==0.
 * Codes 10+ are unreachable by any zone marker, which is what keeps this out of the alias trap above. */
#define PP_DATA 10u

/* EVENT: same layout as PP_DATA, but its id is a RUNTIME value from the kernel rather than a compile-time
 * source-location hash. That is why it is a separate type and not "DATA with size 0": the host must NOT
 * name-resolve a runtime id -- the two share one 20-bit space, so a runtime id of 42 would otherwise
 * borrow the name of whatever zone hashes to 42. Carries the same size field so it can grow a payload. */
#define PP_EVENT 12u

/* ZONE_TOTAL: an accumulated-duration zone (DO_SUM / profileScopeAccumulate). 2 words, but word1 is the
 * accumulated SUM, not a timer -- the host must not treat it as a timestamp. Moved off the DRAM path's
 * value 2, which does not name a marker type on this wire. */
#define PP_ZONE_TOTAL 11u

/* --- PP_DATA low27 sub-fields --- */
#define PP_DATA_ID_MASK 0xFFFFFu /* [19:0]  20-bit id (currently populated with the 16-bit hash) */
#define PP_DATA_SIZE_SHIFT 20u
#define PP_DATA_SIZE_MASK 0x7Fu /* [26:20] payload length in 32-bit words, 0..127 */

/* BULK_CORE: raw-bulk frame for the whole core, emitted by the reader when it does ONE bulk NoC read of all
 * 5 rings (reclaims the single-read NoC amortization). The host splits it into the 5 lanes. Frame layout in
 * the stream (all 32-bit words):
 *   [0] word0 = pp_bulkcore_w0(core_id)   [1] word1 = raw_words (= NRISC*RING_CAP)
 *   [2..2+NRISC-1] per-risc meta: pp_bulk_meta(head_mod, run)   [pad to even]
 *   [prefix..prefix+raw_words) = NRISC contiguous RING_CAP-word ring blocks (ring r @ +r*RING_CAP), RAW
 * Host: for risc r, lane=core*NRISC+r, extract the circular [head_mod, head_mod+run) words of ring r (RAW
 * over-read past tail is present but the host takes only `run`), then decode as that lane's packet stream. */
#define PP_BULK_CORE 5u

/* BULK_SPAN: the IDENTITY-FREE whole-core frame. Where BULK_CORE has the reader tell the host which core
 * and how much (core_id in word0, per-risc head/run meta), a SPAN frame carries the worker's own profiler
 * control vector verbatim and lets the host read all of that out of it -- identity from SPSC_CORE_XY,
 * progress from the heads, extent from the tails. A drainer that has to poll the control vector anyway
 * therefore injects NOTHING, which is the whole point: nothing on the wire can disagree with the worker.
 * Layout and the shared slice geometry live in hostdevcommon/profiler_common.h (SPSC_SPAN_*), which this
 * plain-C header cannot include; spsc_marker_decode.hpp static_asserts that the codes agree. */
#define PP_BULK_SPAN 13u

/* --- word0 fields --- */
#define PP_TYPE_SHIFT 27
#define PP_TYPE_MASK 0x1Fu       /* 5 bits */
#define PP_LOW27_MASK 0x7FFFFFFu /* [26:0]: timer_hi (sticky) or zone_srcloc (marker) */

/* --- word1 is a full 32-bit payload (prog_id or timer_low) --- */
#define PP_TIMER_HI_MASK 0x7FFFFFFu /* 27-bit high half (fits low27 of a sticky word0) */

/* ----- encode ----- */

static inline uint32_t pp_word0(uint32_t type, uint32_t low27) {
    return ((type & PP_TYPE_MASK) << PP_TYPE_SHIFT) | (low27 & PP_LOW27_MASK);
}

/* legacy combined sticky (synthetic bench path) */
static inline uint32_t pp_sticky_w0(uint32_t timer_hi) { return pp_word0(PP_STICKY_META, timer_hi & PP_TIMER_HI_MASK); }
static inline uint32_t pp_sticky_w1(uint32_t prog_id) { return prog_id; }

/* real-path split stickies */
static inline uint32_t pp_prog_w0(void) { return pp_word0(PP_STICKY_PROG, 0u); }
static inline uint32_t pp_prog_w1(uint32_t prog_id) { return prog_id; }
static inline uint32_t pp_timer_w0(uint32_t timer_hi) { return pp_word0(PP_STICKY_TIMER, timer_hi & PP_TIMER_HI_MASK); }
static inline uint32_t pp_timer_w1(void) { return 0u; }

static inline uint32_t pp_marker_w0(uint32_t type, uint32_t zone_srcloc) {
    return pp_word0(type, zone_srcloc & PP_LOW27_MASK);
}
static inline uint32_t pp_marker_w1(uint32_t timer_low) { return timer_low; }

/* DATA/EVENT header: size is in 32-bit words (0 = a bare event). */
static inline uint32_t pp_data_w0(uint32_t id, uint32_t size_words) {
    return pp_word0(PP_DATA, ((size_words & PP_DATA_SIZE_MASK) << PP_DATA_SIZE_SHIFT) | (id & PP_DATA_ID_MASK));
}

/* Runtime-id event header; `id` is masked at runtime since it is not a constant. */
static inline uint32_t pp_event_w0(uint32_t runtime_id, uint32_t size_words) {
    return pp_word0(
        PP_EVENT, ((size_words & PP_DATA_SIZE_MASK) << PP_DATA_SIZE_SHIFT) | (runtime_id & PP_DATA_ID_MASK));
}

/* ----- decode (host) ----- */

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
/* Both point-marker types share the {size,id} low27 layout, so the walk can size them identically. */
static inline int pp_is_point(uint32_t w0) { return pp_is_data(w0) || pp_is_event(w0); }
static inline uint32_t pp_data_id(uint32_t w0) { return pp_low27(w0) & PP_DATA_ID_MASK; }
static inline uint32_t pp_data_size(uint32_t w0) { return (pp_low27(w0) >> PP_DATA_SIZE_SHIFT) & PP_DATA_SIZE_MASK; }
static inline int pp_is_zone_total(uint32_t w0) { return pp_type(w0) == PP_ZONE_TOTAL; }

/* Wire length (32-bit words) of a real-path packet from its type: SRC/TIMER are 1 word (identity/timer_hi
 * fit in low27, no payload), markers + PROG + META are 2. BULK_CORE has its own framing -- do NOT pass it
 * here (the decoder special-cases it first). SENT is always published on a packet boundary, so a decoder
 * that advances by this length stays in sync. */
static inline uint32_t pp_packet_words(uint32_t w0) {
    uint32_t t = pp_type(w0);
    if (t == PP_STICKY_SRC || t == PP_STICKY_TIMER) {
        return 1u;
    }
    if (t == PP_DATA || t == PP_EVENT) {
        return 2u + pp_data_size(w0);  // header + timer_low + payload (self-describing)
    }
    return 2u;
}

/* reader-injected source sticky: lane_id = core*NRISC + risc, carried in both words. */
static inline uint32_t pp_src_w0(uint32_t lane_id) { return pp_word0(PP_STICKY_SRC, lane_id); }
static inline uint32_t pp_src_w1(uint32_t lane_id) { return lane_id; }
static inline uint32_t pp_src_lane(uint32_t w0) { return pp_low27(w0); }

/* ----- BULK_CORE frame (raw-bulk) ----- */
static inline uint32_t pp_bulkcore_w0(uint32_t core) { return pp_word0(PP_BULK_CORE, core); }
static inline int pp_is_bulkcore(uint32_t w0) { return pp_type(w0) == PP_BULK_CORE; }
static inline uint32_t pp_bulkcore_core(uint32_t w0) { return pp_low27(w0); }
/* per-risc meta word: head_mod (ring start slot) in hi16, run (valid word count) in lo16 */
static inline uint32_t pp_bulk_meta(uint32_t head_mod, uint32_t run) { return (head_mod << 16) | (run & 0xFFFFu); }
static inline uint32_t pp_bulk_head(uint32_t m) { return (m >> 16) & 0xFFFFu; }
static inline uint32_t pp_bulk_run(uint32_t m) { return m & 0xFFFFu; }

/* ----- BULK_SPAN frame (identity-free raw span) ----- */
static inline uint32_t pp_bulkspan_w0(void) { return pp_word0(PP_BULK_SPAN, 0u); }
static inline int pp_is_bulkspan(uint32_t w0) { return pp_type(w0) == PP_BULK_SPAN; }

/* reconstruct the 59-bit device timestamp from a marker's 32-bit low + the lane's sticky 27-bit high. */
static inline uint64_t pp_full_ts(uint32_t timer_hi, uint32_t timer_low) {
    return ((uint64_t)(timer_hi & PP_TIMER_HI_MASK) << 32) | (uint64_t)timer_low;
}
static inline uint32_t pp_ts_hi(uint64_t ts) { return (uint32_t)((ts >> 32) & PP_TIMER_HI_MASK); }
static inline uint32_t pp_ts_lo(uint64_t ts) { return (uint32_t)(ts & 0xFFFFFFFFu); }

#endif /* SPSC_PACKET_H */
