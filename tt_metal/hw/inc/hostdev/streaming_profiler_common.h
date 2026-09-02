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
// Tensix RISCs whose rings the drainer sweeps and the host decodes (BRISC, NCRISC, TRISC0-2).
static constexpr std::uint32_t PROFILER_SPSC_TENSIX_RISC = 5;

enum SpscControlBuffer {
    // [0, PROFILER_SPSC_MAX_RISC): ring HEAD per RISC -- consumer-written (drainer), monotonic word count.
    SPSC_RING_HEAD_0 = 0,
    // [PROFILER_SPSC_MAX_RISC, 2*): ring TAIL per RISC -- producer-written, monotonic word count.
    SPSC_RING_TAIL_0 = PROFILER_SPSC_MAX_RISC,
    // Host->kernel terminate signal: set at teardown when the drainer consumer is stopping. While clear, a
    // producing RISC BLOCKS on a full ring (lossless). While set, the producer stops blocking and
    // proceeds, so a dispatch core cannot get stuck in ring_ensure_room and wedge wait_until_cores_done()
    // during device close.
    PROFILER_TERMINATE = 2 * PROFILER_SPSC_MAX_RISC,
    // Core identity: NoC coords packed as (y << 16) | x, written once by BRISC FW at init from
    // my_x[0]/my_y[0]. Deliberately NOT the push-to-DRAM profiler's ControlBuffer::NOC_X/NOC_Y or its
    // FLAT_ID -- that enum belongs to the other backend and the two never share a word.
    //
    // Coords rather than a flat id on purpose: the flat id is a dense rank over a sorted map of
    // Tensix+Eth cores, so it has no positional formula, shifts with harvesting, and can only be
    // computed host-side. (x,y) is what the core already knows about itself, needs no host round-trip,
    // and the host can map it back however it likes. It rides along free in every bulk read, since the
    // control vector is the first 256 B of the span -- a drainer never injects or constructs identity.
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of times this producer BLOCKED on a full ring. Written by the producer itself in the
    // stall path, read by the host directly out of L1 at teardown.
    //
    // This is the knee metric, and it lives here rather than being counted from decoded PROFILER_STALL_ZONE
    // markers because a marker has to survive the whole pipeline to be counted -- DRISC frame, PCIe FIFO,
    // buffer pool, decoder, BroadcastRing -- and every one of those can drop it (measured: 1.29 M records
    // dropped at the ring, ~27 K unaccounted elsewhere). A counter in the producer's own L1 cannot be lost,
    // and reading it needs no decode at all, so the measurement stops perturbing the thing it measures.
    //
    // 8 slots, indexed by processor id: enough for Tensix (5), Eth and DRAM, without pushing
    // SPSC_CONTROL_END past the 64-word control vector.
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

// Size of the drain kernel's results block, in words. Shared because the host both ZEROES and READS it and
// lays the handshake block out immediately behind it -- three places that silently disagreed would each fail
// differently (a stale counter, a short read, an overlapping handshake). Was 64 and exactly full; the DRISC
// self-profiling counters need out[64..87], and the NoC-footprint counters out[88..119].
//
// WHY 208 IS FREE, and why it must not be raised further. This block lives inside the drain kernel's
// `kMiscBytes` budget in perf_debug_profiler.cpp, which is 1024 B holding done(64) + stop(64) + results.
// 208 words is 832 B, so done + stop + results = 960 of 1024. Nothing else moves: `kMiscBytes` is
// unchanged, so `fixed` is unchanged, so the number of STAGING SLOTS the same L1 can hold is unchanged --
// which matters because nstage is 7 by a margin of well under one slot. Check that arithmetic, not just
// this constant, before raising it past the budget.
static constexpr std::uint32_t SPSC_DRAIN_RESULT_WORDS = 224;

// STICKY_META (SPSC/drainer backend, legacy / synthetic bench path only): an 8B context packet whose high
// word carries (core_x, core_y, risc) + this type and whose low word is a 32-bit host-side ID. The host
// forward-fills that identity onto the following timing markers. Its type sits in the same bits as a
// marker's type. Value 6 == PP_STICKY_META in tools/profiler/spsc_packet.h (which is plain C and cannot
// include this header); spsc_marker_decode.hpp static_asserts the two agree. This used to be a trailing
// enumerator on the DRAM profiler's PacketTypes; it never belonged to that wire.
static constexpr std::uint32_t SPSC_TYPE_STICKY_META = 6;

// ---- SPSC SPAN frame: the identity-free drain wire format ------------------------------------------
//
// A drainer should not know, or claim to know, who it is draining. Everything the host needs is already
// in the 256 B control vector the drainer had to poll anyway: identity in SPSC_CORE_XY, progress in the
// heads, extent in the tails. So the frame is that control vector, verbatim, followed by the ring bytes
// it describes -- and the drainer injects nothing.
//
//   [0]                       w0 = SPSC_SPAN_PACKET_TYPE << PP_TYPE_SHIFT; low 27 bits RESERVED, zero
//   [1]                       payload_words = PROFILER_L1_CONTROL_VECTOR_SIZE + pack pads + shipped ring words
//   [2 .. PREFIX)             zero
//   [PREFIX .. +CONTROL)      the worker's profiler control vector, verbatim
//   [.. +payload)             for each RISC in ascending order with a live run: spsc_span_pack_pad()
//                             skipped words, then the run, ring wrap resolved into a flat array
//   [.. frame_words)          skipped words up to a 64 B socket page
//
// The host recomputes the geometry from the control vector the frame carries -- per RISC, run =
// spsc_span_live(head, tail) starting at word counter tail - run -- so there is nothing on the wire to
// desynchronize: no lane tags, no run lengths, no core id.
//
// The payload is never CPU-copied: the NIU gathers each live window straight from the staged span into
// the host FIFO, one write per contiguous ring segment. A NoC write requires the destination to be
// CONGRUENT to the source modulo NOC_PCIE_WRITE_ALIGNMENT_BYTES (16 B -- the NoC MIS-DELIVERS a misaligned
// transfer rather than rejecting it), so each live run is preceded by spsc_span_pack_pad() skipped words
// bringing the wire offset to the ring phase of the run's first word. The wrap continuation needs no pad:
// the ring capacity is a multiple of the alignment, so it lands congruent by construction. Skipped words
// are never written -- the host derives every offset from the control vector and reads past them.
//
// Prefix is 16 words (64 B) so the control vector starts at 64 B and the payload at 320 B -- both
// multiples of L1_ALIGNMENT (16 B on Blackhole), which keeps the control-vector PCIe write aligned.
constexpr static std::uint32_t SPSC_SPAN_PREFIX_WORDS = 16;
// Wire type code. Must equal PP_BULK_SPAN in tt_metal/tools/profiler/spsc_packet.h, which is plain C and
// cannot include this header; spsc_marker_decode.hpp static_asserts that the two agree.
constexpr static std::uint32_t SPSC_SPAN_PACKET_TYPE = 13;
// Where the packet type sits in word0 of every packet in this stream (PP_TYPE_SHIFT in spsc_packet.h).
constexpr static std::uint32_t SPSC_SPAN_TYPE_SHIFT = 27;
// Socket page granularity, in words -- a frame is padded up to a whole number of these.
//
// MEASURED, do not "optimize" again without re-measuring: setting this to the whole frame (2,640 words)
// collapsed page operations 165x (2.5 M -> 8.9 K) and cut bytes shipped 41%, and bought NOTHING -- total
// busy time went 34.0 -> 34.3 ms. It also concentrated the same credit-wait into fewer, longer stalls
// (167 -> 339 us per busy sweep), pushing the worst sweep past the ring-fill deadline and turning 0
// producer stalls into 952. Socket page bookkeeping is not the host-egress wall.
constexpr static std::uint32_t SPSC_SPAN_PAGE_WORDS = 16;

// Live words one RISC contributes to a span frame. With exact packing this is the entire geometry: the
// drainer ships `run` words and the host consumes `run` words, both derived from the same control vector.
//
// Head and tail are monotonic word counters, so the subtraction is wrap-safe. A lossless producer blocks at
// capacity, so a run wider than the ring means the snapshot was read torn -- clamped here, and counted by
// the caller rather than trusted.
constexpr std::uint32_t spsc_span_live(std::uint32_t head, std::uint32_t tail, std::uint32_t cap) {
    const std::uint32_t run = tail - head;
    return run > cap ? cap : run;
}

// NoC L1->PCIe write congruence quantum (NOC_PCIE_WRITE_ALIGNMENT_BYTES), in words.
constexpr static std::uint32_t SPSC_SPAN_PACK_ALIGN_WORDS = 4;
// Skipped words before a live run so the NIU gather lands src/dst congruent: both the frame start on the
// wire and the staged span in L1 sit at alignment-multiple addresses, so only the run's ring phase and the
// current wire offset (in words from the frame start) decide the pad. The drainer sizes frames with this
// and the host walks them with it; a disagreement would desynchronize every lane after the first.
constexpr std::uint32_t spsc_span_pack_pad(std::uint32_t start_counter, std::uint32_t frame_off_words) {
    return (start_counter - frame_off_words) & (SPSC_SPAN_PACK_ALIGN_WORDS - 1u);
}

// A wrapping run ships as its whole ring image ONLY when the dead remainder is small. The one-read
// image saves a NoC issue on the drainer's critical path, which pays exactly at the saturation
// boundary -- where runs are near-full and the remainder is a few words. At sustained rates runs
// wrap at a few hundred words, and shipping the image inflates egress bytes by the remainder
// (measured +33% at delay 9, enough to push the drain past its equilibrium ceiling: sustained knee
// 9 -> 12). Below the threshold the run ships as the two-piece wrap split, byte-exact. Both sides
// derive the condition from (start, extent) alone, so the wire carries no flag -- but they MUST
// use this one predicate: a disagreement mis-walks every lane after the first.
constexpr static std::uint32_t SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS = 64;
constexpr bool spsc_span_wrap_image(std::uint32_t start, std::uint32_t extent, std::uint32_t ring_cap) {
    return (start & (ring_cap - 1u)) + extent > ring_cap && ring_cap - extent <= SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}

inline std::uint32_t spsc_span_w0() { return SPSC_SPAN_PACKET_TYPE << SPSC_SPAN_TYPE_SHIFT; }

// Compact on-wire control block for PACKED span frames: just the words the decoder walks. The L1
// control vector is 64 words laid out for 24 RISCs (heads at 0..4 but tails at 24..28 and XY at 49),
// and shipping it verbatim made ~50 dead words per frame in the loaded direction of the PCIe tile.
// RAW (self) frames still carry the true vector at its L1 layout -- the w0 raw flag picks the geometry.
constexpr static std::uint32_t SPSC_SPAN_WIRE_CTRL_WORDS = 16;
enum SpscWireCtrl : std::uint32_t {
    SPSC_WIRE_HEAD_0 = 0,  // ..4
    SPSC_WIRE_TAIL_0 = 5,  // ..9
    SPSC_WIRE_XY = 10,
};

// Layout flag in w0's reserved-zero low bits: set = the payload is the RAW span (control vector +
// five whole rings at fixed offsets, ring wrap NOT resolved) instead of packed live runs. The drainer
// ships whichever costs its egress less -- packing trades bytes for NoC write issues (~10 extra per
// frame), so above the kernel's fill threshold raw's single burst wins and the flag rides along so
// the host walks the right geometry.
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
// PP_DATA: a point-in-time packet with a self-describing payload length, 3 + N words. Mirrors
// spsc_packet.h's pp_data_w0 / pp_data_w2 and is asserted against them in spsc_marker_decode.hpp, the one
// translation unit that sees both headers. word0 is shaped exactly like a zone marker (type | the full
// 27-bit structural id); the length lives in its own word2.
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
