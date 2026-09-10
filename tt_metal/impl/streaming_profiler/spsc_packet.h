// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The relay's compact profiler packet wire format. Each (core, risc) lane is kept separate end to end, so
// identity is structural and packets carry no core/risc or framing bits. A packet is two 32-bit words:
// word0 = [31:27] type(5) | [26:0] low27, word1 = payload32. Markers carry the 27-bit structural zone id
// (tu_id(13) << 14 | local(14), hostdevcommon/profiler_zone_id.h) and timer_low; timer_hi rides the rare STICKY_TIMER.
// Plain C, read by the host decoder; the producer keeps its own copy of the packer (ppfmt in
// kernel_profiler_streaming.hpp).

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
// Anchoring on the end is what lets one STICKY_TIMER cover everything after it. No sticky-lo exists: a re-anchor
// is never cheaper than an inline ZONE_ATOMIC, and a stale cursor is merely conservative.
//
// Stickies (1 word unless noted), each reconstructed on the host from its last-seen value per lane:
//   STICKY_PROG  low27 = runtime host-id; ids >= 2^27 ship as the 2-word STICKY_PROG_EXT with the id in word1.
//   STICKY_TIMER low27 = timer_hi, emitted on each high-half tick (~3.2 s at 1.35 GHz).
#define PP_ZONE_S 3u
#define PP_ZONE_L 4u
#define PP_ZONE_ATOMIC 2u

#define PP_STICKY_PROG 8u
#define PP_STICKY_TIMER 9u

// DATA (3 + N words): [0] type|id27 [1] timer_low [2] size << PP_DATA_SIZE_SHIFT [3..] payload. word0 is exactly a
// zone marker's word0, so a point marker has the same ELF-resolved identity as a zone; the length has its own
// word so the host advances over any payload without a per-type table.
#define PP_DATA 10u

// EVENT (2 words): [0] type|id27 [1] timer_low; a flag with no payload and no size word.
#define PP_EVENT 12u

// CLOCK (2 words): [0] type | kind<<PP_CLOCK_KIND_SHIFT | value24  [1] wall_lo. A local-refclk sample from the
// idle-eth clock tracker: value24 = this chip refclk low 24 bits, wall = lane sticky-timer hi | wall_lo. Routed
// to the clock sink at decode, never delivered as a record. Must match ppfmt::T_CLOCK (kernel_profiler_streaming.hpp).
#define PP_CLOCK 5u
#define PP_CLOCK_KIND_SHIFT 24u
#define PP_CLOCK_VALUE_MASK 0xFFFFFFu
#define PP_CLOCK_LOCAL_REFCLK 0u
#define PP_CLOCK_LINK_REFCLK 1u

/* 11 is retired (was ZONE_TOTAL); never reuse it. */

/* --- PP_DATA word2 sub-fields (word0 is type|id27, identical to a zone marker) --- */
#define PP_DATA_SIZE_SHIFT 25u
#define PP_DATA_SIZE_MASK 0x7Fu /* [31:25] payload length in 32-bit words, 0..127; [24:0] unused, zero */

// BULK_SPAN: the identity-free whole-core frame, carrying the worker's own control vector so nothing on the wire
// can disagree with the worker. Layout in hostdev/streaming_profiler_common.h (SPSC_SPAN_*), which this plain-C
// header cannot include; spsc_marker_decode.hpp asserts the codes agree.
#define PP_BULK_SPAN 13u

/* 2-word STICKY_PROG escape for host-ids >= 2^27 -- see STICKY_PROG above. */
#define PP_STICKY_PROG_EXT 14u

/* --- word0 fields --- */
#define PP_TYPE_SHIFT 27
#define PP_TYPE_MASK 0x1Fu       /* 5 bits */
#define PP_LOW27_MASK 0x7FFFFFFu /* [26:0]: timer_hi (sticky) or the full 27-bit zone id (marker) */

static inline uint32_t pp_type(uint32_t w0) { return (w0 >> PP_TYPE_SHIFT) & PP_TYPE_MASK; }
static inline uint32_t pp_low27(uint32_t w0) { return w0 & PP_LOW27_MASK; }
static inline int pp_is_bulkspan(uint32_t w0) { return pp_type(w0) == PP_BULK_SPAN; }

#endif /* SPSC_PACKET_H */
