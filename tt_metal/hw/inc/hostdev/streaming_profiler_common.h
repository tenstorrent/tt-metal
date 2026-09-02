// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ---- STREAMING (perf_debug) profiler: host/device shared constants --------------------------------------
//
// Everything the streaming backend adds on top of the DRAM profiler's hostdev/profiler_common.h lives here,
// so that header stays byte-for-byte the DRAM profiler's own. The two backends are mutually exclusive at run
// time (TT_METAL_DEVICE_PROFILER vs TT_METAL_STREAMING_PROFILER, see llrt/rtoptions.cpp), and the device
// producer for this backend is tools/profiler/kernel_profiler_streaming.hpp, selected by -DPROFILE_STREAMING.
//
// Consumers: the SPSC producer (kernel_profiler_streaming.hpp), the DRISC filler kernels
// (tools/profiler/kernels/drisc_*.cpp), the host receiver/decoder (tools/profiler/perf_debug_*,
// spsc_marker_decode.hpp) and the DRISC test kernels.

#include <cstdint>

#include "hostdev/profiler_common.h"
#include "hostdevcommon/profiler_zone_id.h"

namespace kernel_profiler {

// ---- SPSC / drainer backend control-word layout ------------------------------------------------------
// The drainer backend overlays its OWN layout on the same profiler control vector. It deliberately does not
// reuse ControlBuffer's HOST_/DEVICE_BUFFER_END_INDEX_* slots, and deliberately derives nothing from
// PROFILER_MAX_RISC_COUNT: those are the DRAM profiler's DRAM-readout bookkeeping and its processor count,
// and this backend has no stake in either.
//
// That coupling was not hypothetical. Upstream raised PROFILER_MAX_RISC_COUNT from 5 to 24 for Quasar,
// which silently relocated this backend's ring tails from words 5..9 to 24..28 while the drainer firmware
// still read 5..9. Those became dead reserved slots reading 0, so tail always equalled head: nothing ever
// drained, the worker L1 rings filled, and every producing RISC blocked forever. A constant belonging to
// the other backend moved this one's flow-control words.
//
// Overlaying the same physical words is safe because the two backends are mutually exclusive: a process
// runs either the DRAM profiler (TT_METAL_DEVICE_PROFILER) or the streaming one (TT_METAL_STREAMING_PROFILER),
// never both (rtoptions TT_FATALs on the pair).
//
// Sized for the widest processor count -- including Quasar's 24 -- so the layout is arch-uniform and the
// host indexes it identically everywhere, whatever the DRAM side later does with its own count.
static constexpr std::uint32_t PROFILER_SPSC_MAX_RISC = 24;
// Tensix RISCs whose rings the relay sweeps and the host decodes (BRISC, NCRISC, TRISC0-2).
static constexpr std::uint32_t PROFILER_SPSC_TENSIX_RISC = 5;

enum SpscControlBuffer {
    // [0, PROFILER_SPSC_MAX_RISC): ring head per RISC, consumer-written (relay), monotonic word count.
    SPSC_RING_HEAD_0 = 0,
    // [PROFILER_SPSC_MAX_RISC, 2*): ring tail per RISC, producer-written, monotonic word count.
    SPSC_RING_TAIL_0 = PROFILER_SPSC_MAX_RISC,
    // Host->kernel terminate: while clear a producer blocks on a full ring; while set it proceeds, so a dispatch
    // core cannot wedge wait_until_cores_done() at device close.
    PROFILER_TERMINATE = 2 * PROFILER_SPSC_MAX_RISC,
    // NoC coords packed (y << 16) | x, written once by BRISC FW at init. Coords, not the flat id: the flat id is a
    // dense rank over a sorted core map with no positional formula, computable only host-side.
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of full-ring blocks, written in the stall path and read by the host from L1 at teardown;
    // counting decoded stall markers would undercount, since a marker can be dropped between the relay frame and
    // the BroadcastRing. 8 slots so SPSC_CONTROL_END stays inside the 64-word vector.
    SPSC_STALL_COUNT_0 = 2 * PROFILER_SPSC_MAX_RISC + 2,
    SPSC_STALL_COUNT_MAX = 8,
    SPSC_CONTROL_END = SPSC_STALL_COUNT_0 + SPSC_STALL_COUNT_MAX,  // first unused word; grow the layout here
};

// Bounds the SPSC backend's whole control block against the DRAM profiler's L1 control vector, which it
// overlays. Deliberately not asserted on DRAM_PROFILER_ADDRESS_T2_0: that entry is already out of bounds
// upstream (with PROFILER_MAX_RISC_COUNT = 24 it evaluates to 64), so asserting it would fail the build on a
// defect this backend neither introduced nor can fix here.
static_assert(
    SPSC_CONTROL_END <= PROFILER_L1_CONTROL_VECTOR_SIZE,
    "SPSC/drainer control layout overflows the profiler L1 control vector");

// Host->relay stop word: quiesce drains everything with every wait still holding, release is the kill
// switch that abandons the waits and hands the NIU back.
static constexpr std::uint32_t kRelayStopQuiesce = 1;
static constexpr std::uint32_t kRelayStopRelease = 2;
// Relay->host completion word, published only after the socket barrier; the host matches the high half.
static constexpr std::uint32_t kRelayDoneWord = 0xD09E0000u;
static constexpr std::uint32_t kRelayDoneMask = 0xFFFF0000u;
// Each relay control word owns a 64 B pad, so the words that share it (the sync rendezvous triple behind
// the stop word, the heartbeat behind done) travel in one host write.
static constexpr std::uint32_t kRelayCtrlWordStride = 64;

// STICKY_META (SPSC/drainer backend, legacy / synthetic bench path only): an 8B context packet whose high
// word carries (core_x, core_y, risc) + this type and whose low word is a 32-bit host-side ID. The host
// forward-fills that identity onto the following timing markers. Its type sits in the same bits as a
// marker's type. Value 6 == PP_STICKY_META in tools/profiler/spsc_packet.h (which is plain C and cannot
// include this header); spsc_marker_decode.hpp static_asserts the two agree. This used to be a trailing
// enumerator on the DRAM profiler's PacketTypes; it never belonged to that wire.
static constexpr std::uint32_t SPSC_TYPE_STICKY_META = 6;

// SPSC span frame, the relay wire format: a control block (identity from SPSC_CORE_XY, progress from the heads,
// extent from the tails) followed by the ring words it describes; the relay injects nothing of its own.
//
//   [0]                       w0 = SPSC_SPAN_PACKET_TYPE << PP_TYPE_SHIFT; low 27 bits are layout flags
//   [1]                       payload_words = control block + pack pads + shipped ring words
//   [2 .. PREFIX)             zero
//   [PREFIX .. +CONTROL)      packed frame: the SPSC_SPAN_WIRE_CTRL_WORDS control block; raw frame: the
//                             worker's whole 64-word control vector
//   [.. +payload)             per RISC in ascending order with a live run: spsc_span_pack_pad() skipped words,
//                             then the run, ring wrap resolved into a flat array
//   [.. frame_words)          skipped words up to a 64 B socket page
//
// The host recomputes the geometry from the control block (per RISC, run = spsc_span_live(head, tail) ending
// at word counter tail), so the wire carries no lane tags, run lengths or core id. The NIU gathers each live
// window straight into the host FIFO, and a NoC write mis-delivers a transfer whose destination is not
// congruent to the source modulo NOC_PCIE_WRITE_ALIGNMENT_BYTES (16 B), so each run is preceded by
// spsc_span_pack_pad() skipped words, never written; the host reads past them. The 16-word prefix puts the
// control block at 64 B and the payload at 320 B, both L1_ALIGNMENT multiples.
constexpr static std::uint32_t SPSC_SPAN_PREFIX_WORDS = 16;
// Wire type code. Must equal PP_BULK_SPAN in tt_metal/tools/profiler/spsc_packet.h, which is plain C and
// cannot include this header; spsc_marker_decode.hpp static_asserts that the two agree.
constexpr static std::uint32_t SPSC_SPAN_PACKET_TYPE = 13;
// Where the packet type sits in word0 of every packet in this stream (PP_TYPE_SHIFT in spsc_packet.h).
constexpr static std::uint32_t SPSC_SPAN_TYPE_SHIFT = 27;
// Socket page granularity in words; frames pad up to a whole number. Larger pages concentrate the same credit
// wait into stalls long enough to miss the ring-fill deadline.
constexpr static std::uint32_t SPSC_SPAN_PAGE_WORDS = 16;

// Live words one RISC contributes. Head and tail are monotonic, so the subtraction is wrap-safe; a run wider
// than the ring means a torn snapshot and is clamped here and counted by the caller.
constexpr std::uint32_t spsc_span_live(std::uint32_t head, std::uint32_t tail, std::uint32_t cap) {
    const std::uint32_t run = tail - head;
    return run > cap ? cap : run;
}

// NoC L1->PCIe write congruence quantum (NOC_PCIE_WRITE_ALIGNMENT_BYTES), in words.
constexpr static std::uint32_t SPSC_SPAN_PACK_ALIGN_WORDS = 4;
// Skipped words before a live run so the NIU gather lands src/dst congruent; frame start and staged span sit
// at alignment multiples, so only the run's ring phase and the wire offset decide it. Relay and host must
// agree or every later lane mis-walks.
constexpr std::uint32_t spsc_span_pack_pad(std::uint32_t start_counter, std::uint32_t frame_off_words) {
    return (start_counter - frame_off_words) & (SPSC_SPAN_PACK_ALIGN_WORDS - 1u);
}

// A wrapping run ships as its whole ring image only when the dead remainder is small, else as a two-piece
// split: the one-read image saves a NoC issue at the saturation boundary, but at sustained rates it inflates
// egress by the remainder. Both sides derive this from (start, extent) alone; the wire carries no flag.
constexpr static std::uint32_t SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS = 64;
constexpr bool spsc_span_wrap_image(std::uint32_t start, std::uint32_t extent, std::uint32_t ring_cap) {
    return (start & (ring_cap - 1u)) + extent > ring_cap && ring_cap - extent <= SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}

inline std::uint32_t spsc_span_w0() { return SPSC_SPAN_PACKET_TYPE << SPSC_SPAN_TYPE_SHIFT; }

// Control block of a packed frame: just the words the decoder walks. The L1 vector is 64 words laid out for
// 24 RISCs, ~50 of them dead on the wire. Raw frames carry the true vector; the w0 raw flag picks the geometry.
constexpr static std::uint32_t SPSC_SPAN_WIRE_CTRL_WORDS = 16;
enum SpscWireCtrl : std::uint32_t {
    SPSC_WIRE_HEAD_0 = 0,  // ..4
    SPSC_WIRE_TAIL_0 = 5,  // ..9
    SPSC_WIRE_XY = 10,
};

// w0 flag: the payload is the raw span (control vector plus five whole rings, wrap unresolved) rather than
// packed live runs. Packing trades bytes for ~10 extra NoC issues per frame, so above the fill threshold raw
// wins.
constexpr static std::uint32_t SPSC_SPAN_RAW_FLAG = 1u;

// ---- Wire codes shared with the producer and the host decoder --------------------------------------
//
// Codes MUST match spsc_packet.h's PP_* -- asserted in spsc_marker_decode.hpp, which is the one place that
// sees both headers -- and kernel_profiler_streaming.hpp's ppfmt (inlined there because the JIT build lacks the
// spsc_packet.h include path).

// Producer tail-publish batch (kernel_profiler_streaming.hpp publish_tail_batched): the published TAIL can lag
// true ring occupancy by up to this many words between fenced publishes -- drainer-invisible occupancy
// against the producer's 506-word bar. Must be a power of two. 16, not 64: the interleaved microbench
// (device reset between runs, 200k zones/RISC) measured 64 at 59.82/60.31 cycles/zone and 16 at
// 59.34/59.36 -- the "global producer-overhead knob" fear that once kept this at 64 has the sign
// wrong, and the recovered margin is what the knee needed: with the barrier-hoisted heads, delay 10
// goes 25-44 stalls to 0/0/0. NOT 8: it buys delay 9 (0-1 stalls) but measures 59.87 cycles/zone --
// a real producer cost over 16 -- and producer overhead outranks the knee here by policy.
static constexpr std::uint32_t SPSC_PUBLISH_BATCH_WORDS = 16;

static constexpr std::uint32_t SPSC_TYPE_ZONE_START = 0;  // legacy pair (workers: stall zone, >3.2s fallback)
static constexpr std::uint32_t SPSC_TYPE_ZONE_END = 1;    // legacy pair
static constexpr std::uint32_t SPSC_TYPE_ZONE_L = 4;      // >3.2 s zone: id | end_lo | end_hi | dur_lo | dur_hi
static constexpr std::uint32_t SPSC_TYPE_STICKY_TIMER = 9;
static constexpr std::uint32_t SPSC_TIMER_HI_MASK = 0x7FFFFFFu;  // the 27-bit low field of word0

// FULL 27 bits. This mask was 0xFFFF once, and that truncation was invisible: markers rendered
// perfectly and only their NAMES could not be resolved. Any change to the id width has to be made in
// EVERY copy of the packer at once -- this one, ppfmt in kernel_profiler_streaming.hpp, and pp_* in spsc_packet.h.
inline std::uint32_t spsc_marker_w0(std::uint32_t type, std::uint32_t zone_id) {
    return (type << SPSC_SPAN_TYPE_SHIFT) | (zone_id & TT_ZONE_ID_MASK);
}
inline std::uint32_t spsc_sticky_timer_w0(std::uint32_t timer_hi) {
    return (SPSC_TYPE_STICKY_TIMER << SPSC_SPAN_TYPE_SHIFT) | (timer_hi & SPSC_TIMER_HI_MASK);
}
// 3 + N words: word0 is shaped like a zone marker (type | 27-bit id), the length lives in word2. Mirrors
// spsc_packet.h's pp_data_w0/w2.
static constexpr std::uint32_t SPSC_TYPE_DATA = 10;
static constexpr std::uint32_t SPSC_DATA_SIZE_SHIFT = 25;
inline std::uint32_t spsc_data_w0(std::uint32_t id) {
    return (SPSC_TYPE_DATA << SPSC_SPAN_TYPE_SHIFT) | (id & TT_ZONE_ID_MASK);
}
inline std::uint32_t spsc_data_w2(std::uint32_t size_words) { return (size_words & 0x7Fu) << SPSC_DATA_SIZE_SHIFT; }

// Words in a staging/ring slot. A slot must hold the PACKED image of a span, which can be LARGER than the
// raw span it replaces: the raw layout needs no pads (lane r starts at prefix + ctrl + r*ring, inherently
// congruent), while the packed layout places extents back to back, so each of `num_risc` lanes can need up
// to SPSC_SPAN_PACK_ALIGN_WORDS-1 words of pad. Sizing a slot for the raw span alone let a nearly-full
// span's packed image overrun the slot by 16 words -- into the next slot, or past the last one into the
// drainer's head scratch -- which is why packing used to be gated behind a fill-fraction fallback. Sized
// for the worst case, packing needs no gate at all.
constexpr std::uint32_t spsc_span_slot_words(std::uint32_t num_risc) {
    const std::uint32_t span = PROFILER_L1_CONTROL_VECTOR_SIZE + num_risc * PROFILER_L1_VECTOR_SIZE;
    const std::uint32_t worst = SPSC_SPAN_PREFIX_WORDS + span + num_risc * (SPSC_SPAN_PACK_ALIGN_WORDS - 1u);
    return (worst + SPSC_SPAN_PAGE_WORDS - 1u) & ~(SPSC_SPAN_PAGE_WORDS - 1u);
}

// Total words a frame occupies on the wire, including the prefix and the pad up to a socket page.
constexpr std::uint32_t spsc_span_frame_words(std::uint32_t payload_words) {
    const std::uint32_t n = SPSC_SPAN_PREFIX_WORDS + payload_words;
    return (n + SPSC_SPAN_PAGE_WORDS - 1u) & ~(SPSC_SPAN_PAGE_WORDS - 1u);
}

static_assert(
    (SPSC_SPAN_PREFIX_WORDS + PROFILER_L1_CONTROL_VECTOR_SIZE) % SPSC_SPAN_PAGE_WORDS == 0,
    "the payload must start on a socket page boundary");

}  // namespace kernel_profiler
