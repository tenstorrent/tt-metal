/*
 * prof_packet.h - X280 compact profiler packet wire format (per-lane, never-linearized design).
 *
 * Each (core,risc) lane is kept SEPARATE end to end (L1 ring -> per-lane LIM SPSC -> per-lane host
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
 * Plain C header: included by the bare-metal FW (C), the worker producer kernel (C++), and the host
 * consumer (C++). No dependencies.
 */
#ifndef X280_PROF_PACKET_H
#define X280_PROF_PACKET_H

#include <stdint.h>

/* packet_type field (5 bits). Values match hostdevcommon PacketTypes where they overlap. */
#define PP_ZONE_START 0u
#define PP_ZONE_END 1u
#define PP_STICKY_META 6u
/* STICKY_SRC: (core,risc) identity, injected by the READER into the LINEARIZED stream whenever it
 * switches to a new source ring. Everything after it (until the next STICKY_SRC) belongs to this lane.
 * low27 = lane_id (core*NRISC + risc); payload32 = lane_id (redundant/self-check). */
#define PP_STICKY_SRC 7u

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

static inline uint32_t pp_sticky_w0(uint32_t timer_hi) { return pp_word0(PP_STICKY_META, timer_hi & PP_TIMER_HI_MASK); }
static inline uint32_t pp_sticky_w1(uint32_t prog_id) { return prog_id; }

static inline uint32_t pp_marker_w0(uint32_t type, uint32_t zone_srcloc) {
    return pp_word0(type, zone_srcloc & PP_LOW27_MASK);
}
static inline uint32_t pp_marker_w1(uint32_t timer_low) { return timer_low; }

/* ----- decode (host) ----- */

static inline uint32_t pp_type(uint32_t w0) { return (w0 >> PP_TYPE_SHIFT) & PP_TYPE_MASK; }
static inline uint32_t pp_low27(uint32_t w0) { return w0 & PP_LOW27_MASK; }
static inline uint32_t pp_payload32(uint32_t w1) { return w1; }
static inline int pp_is_sticky(uint32_t w0) { return pp_type(w0) == PP_STICKY_META; }
static inline int pp_is_src(uint32_t w0) { return pp_type(w0) == PP_STICKY_SRC; }

/* reader-injected source sticky: lane_id = core*NRISC + risc, carried in both words. */
static inline uint32_t pp_src_w0(uint32_t lane_id) { return pp_word0(PP_STICKY_SRC, lane_id); }
static inline uint32_t pp_src_w1(uint32_t lane_id) { return lane_id; }
static inline uint32_t pp_src_lane(uint32_t w0) { return pp_low27(w0); }

/* reconstruct the 59-bit device timestamp from a marker's 32-bit low + the lane's sticky 27-bit high. */
static inline uint64_t pp_full_ts(uint32_t timer_hi, uint32_t timer_low) {
    return ((uint64_t)(timer_hi & PP_TIMER_HI_MASK) << 32) | (uint64_t)timer_low;
}
static inline uint32_t pp_ts_hi(uint64_t ts) { return (uint32_t)((ts >> 32) & PP_TIMER_HI_MASK); }
static inline uint32_t pp_ts_lo(uint64_t ts) { return (uint32_t)(ts & 0xFFFFFFFFu); }

#endif /* X280_PROF_PACKET_H */
