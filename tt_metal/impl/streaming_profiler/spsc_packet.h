// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The relay's compact profiler packet wire format. Each (core, risc) lane is kept separate end to end, so
// identity is structural and packets carry no core/risc or framing bits. Word 0 of every packet is [31:27] type(5)
// | [26:0] low27. A record's low27 is its body's length and the call site's 16-bit id, which each image is assigned at
// load (hostdev/profiler_zone_id.h). A sticky's low27 is its value. timer_hi rides the rare STICKY_TIMER. Plain C, read
// by the host decoder; the producer keeps its own copy of the packer (ppfmt in hostdev/streaming_profiler_common.h).

#ifndef SPSC_PACKET_H
#define SPSC_PACKET_H

#include <stdint.h>

// The relay wire's own 5-bit type space, independent of hostdevcommon's PacketTypes: passing a PacketTypes value
// through once collided ZONE_TOTAL(2)/TS_DATA_16B(5) with unrelated codes here and desynchronized the walk.
//
// Stickies (1 word unless noted), each reconstructed on the host from its last-seen value per lane:
//   STICKY_PROG  low27 = runtime host-id; ids >= 2^27 ship as the 2-word STICKY_PROG_EXT with the id in word1.
//   STICKY_TIMER low27 = timer_hi, emitted on each high-half tick (~3.2 s at 1.35 GHz).
/* 2, 3, 4, 10, 11 and 12 are retired; never reuse them. */

#define PP_STICKY_PROG 8u
#define PP_STICKY_TIMER 9u

// A DevicePublish record, which every marker is. Its word 0 is type | N << PP_LENGTH_SHIFT | id, so its packet is 1 + N
// words, and its body is one of its call site's forms, which N tells apart (hostdev/record_site.h).
#define PP_RECORD 15u

#define PP_LENGTH_SHIFT 20u
#define PP_LENGTH_MASK 0x7Fu /* [26:20] */
#define PP_ID_MASK 0xFFFFu   /* [15:0] is the call site's id. [16] is its destination, [19:17] zero. */
/* A record's destination bit, which its call site fixes: the relays copy a record with PP_TO_EXTERNAL to the
   external process too. */
#define PP_TO_EXTERNAL (1u << 16)

// BULK_SPAN: the identity-free whole-core frame, carrying the worker's own control vector so nothing on the wire
// can disagree with the worker. Layout in hostdev/streaming_profiler_common.h (SPSC_SPAN_*), which this plain-C
// header cannot include; spsc_marker_decode.hpp asserts the codes agree.
#define PP_BULK_SPAN 13u

/* 2-word STICKY_PROG escape for host-ids >= 2^27 -- see STICKY_PROG above. */
#define PP_STICKY_PROG_EXT 14u

/* --- word0 fields --- */
#define PP_TYPE_SHIFT 27
#define PP_TYPE_MASK 0x1Fu       /* 5 bits */
#define PP_LOW27_MASK 0x7FFFFFFu /* [26:0]: a sticky's value */

static inline uint32_t pp_type(uint32_t w0) { return (w0 >> PP_TYPE_SHIFT) & PP_TYPE_MASK; }
static inline uint32_t pp_low27(uint32_t w0) { return w0 & PP_LOW27_MASK; }
static inline int pp_is_bulkspan(uint32_t w0) { return pp_type(w0) == PP_BULK_SPAN; }

#endif /* SPSC_PACKET_H */
