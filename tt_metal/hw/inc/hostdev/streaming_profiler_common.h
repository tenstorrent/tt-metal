// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host/device shared constants of the streaming profiler.
//
// Everything the streaming backend adds on top of the DRAM profiler's hostdev/profiler_common.h lives here,
// so that header stays byte-for-byte the DRAM profiler's own. The two backends are mutually exclusive at run
// time (TT_METAL_DEVICE_PROFILER vs TT_METAL_STREAMING_PROFILER, see llrt/rtoptions.cpp), and the device
// producer for this backend is tools/profiler/kernel_profiler_streaming.hpp, selected by -DPROFILE_STREAMING.
//
// Consumers: the SPSC producer (kernel_profiler_streaming.hpp), the relays and clock-sync kernels
// (tt_metal/impl/streaming_profiler/kernels/) and the host (impl/streaming_profiler/).

#include <cstddef>
#include <cstdint>

#include "hostdev/profiler_common.h"
#include "hostdev/profiler_zone_id.h"

namespace kernel_profiler {

template <typename T>
constexpr std::uint32_t word_of(const T& value) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(std::uint32_t, value);
}
template <typename T>
constexpr T word_as(std::uint32_t word) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(T, word);
}

// A core's NoC coordinates, as the relays and the clock kernels take them.
struct NocXy {
    std::uint32_t x : 16;
    std::uint32_t y : 16;
};

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
    // Per Tensix RISC, the timer high word and runtime id in effect at the published tail, for a decoder reseeding a
    // lane after a loss. The producer stores its tail, fences, then the state, so a reader that observes the state
    // no later than the tail never sees a value the frame's words do not already carry inline. They live in the
    // tails' 64 B block (words 16..31), in the words no Tensix RISC owns -- heads 16..23 and tails 29..30 -- so the
    // relay's one 64 B read takes state and tails together. Runtime ids: spsc_state_prog_word.
    SPSC_STATE_TIMER_0 = 16,
    SPSC_STATE_PROG_0 = 21,
    // Host->kernel arm: while set a producer blocks on a full ring, because a relay is draining this core; while
    // clear it proceeds and overwrites. The host clears it on every Tensix core of the device at session start and
    // sets it only on the cores its relays serve, so a core nobody drains (dispatch cores, whose rings fill one launch
    // at a time across processes) can never park in the stall path and wedge wait_until_cores_done() at device close.
    PROFILER_ARMED = 2 * PROFILER_SPSC_MAX_RISC,
    // Reserved; the core's NoC coordinate reaches the wire from the host's core list via the relay (SPSC_PREFIX_XY).
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of full-ring blocks, written in the stall path and read by the host from L1 at teardown;
    // counting decoded stall markers would undercount, since a marker can be dropped between the relay frame and
    // the BroadcastRing. 8 slots so SPSC_CONTROL_END stays inside the 64-word vector.
    SPSC_STALL_COUNT_0 = 2 * PROFILER_SPSC_MAX_RISC + 2,
    SPSC_STALL_COUNT_MAX = 8,
    // On an active eth core hosting a link end: the tail sits in the tails' 64 B block so the eth relay's one read of
    // that block takes it.
    SPSC_LINK_SYNC_TAIL = 31,
    SPSC_LINK_SYNC_HEAD = SPSC_STALL_COUNT_0 + SPSC_STALL_COUNT_MAX,
    SPSC_CONTROL_END = SPSC_LINK_SYNC_HEAD + 1,  // first unused word; grow the layout here
};
// Runtime-id slot of Tensix RISC `risc`: 21..23, then 29..30 past the tails.
constexpr std::uint32_t spsc_state_prog_word(std::uint32_t risc) {
    return risc < 3 ? SPSC_STATE_PROG_0 + risc : SPSC_RING_TAIL_0 + PROFILER_SPSC_TENSIX_RISC + (risc - 3);
}

// Bounds the SPSC backend's whole control block against the DRAM profiler's L1 control vector, which it
// overlays. Deliberately not asserted on DRAM_PROFILER_ADDRESS_T2_0: that entry is already out of bounds
// upstream (with PROFILER_MAX_RISC_COUNT = 24 it evaluates to 64), so asserting it would fail the build on a
// defect this backend neither introduced nor can fix here.
static_assert(
    SPSC_CONTROL_END <= PROFILER_L1_CONTROL_VECTOR_SIZE,
    "SPSC/drainer control layout overflows the profiler L1 control vector");
static_assert(
    PROFILER_SPSC_TENSIX_RISC == 5 && SPSC_STATE_TIMER_0 >= PROFILER_SPSC_TENSIX_RISC &&
        SPSC_STATE_TIMER_0 + PROFILER_SPSC_TENSIX_RISC <= SPSC_STATE_PROG_0 &&
        SPSC_STATE_PROG_0 + 3 == SPSC_RING_TAIL_0 &&
        spsc_state_prog_word(PROFILER_SPSC_TENSIX_RISC - 1) < SPSC_LINK_SYNC_TAIL && SPSC_LINK_SYNC_TAIL < 32,
    "lane state must fill the unowned words of the tails' 64 B block");

// Host->resident core stop word: quiesce drains everything with every wait still holding, then the core exits.
static constexpr std::uint32_t kResidentStopQuiesce = 1;
// Resident core->host completion words.
static constexpr std::uint32_t kResidentAwaitingAcksWord = 0xD09D0000u;
static constexpr std::uint32_t kResidentDoneWord = 0xD09E0000u;
// The control block at each resident core's ctrl address (relays, clock tracker, ruler). The core counts heartbeat from
// launch, the host writes stop (and go, on the eth cores), and the core writes done at the end. The sync fields are the
// eth cores' sync ring cursors and drop count.
struct ResidentCtrl {
    std::uint32_t done;
    std::uint32_t heartbeat;
    std::uint32_t go;
    std::uint32_t sync_tail;
    std::uint32_t sync_head;
    std::uint32_t dropped_sync;
    std::uint32_t stop;
};

// STICKY_META (SPSC/drainer backend, legacy / synthetic bench path only): an 8B context packet whose high
// word carries (core_x, core_y, risc) + this type and whose low word is a 32-bit host-side ID. The host
// forward-fills that identity onto the following timing markers. Its type sits in the same bits as a
// marker's type. Value 6 == PP_STICKY_META in impl/streaming_profiler/spsc_packet.h (which is plain C and cannot
// include this header); spsc_marker_decode.hpp static_asserts the two agree. This used to be a trailing
// enumerator on the DRAM profiler's PacketTypes; it never belonged to that wire.
static constexpr std::uint32_t SPSC_TYPE_STICKY_META = 6;

// SPSC span frame, the relay wire format: a control block (identity from the host-seeded coordinate, progress from
// the heads, extent from the tails) followed by the ring words it describes; the relay injects nothing of its own.
//
//   [0]                       w0 = SPSC_SPAN_PACKET_TYPE << PP_TYPE_SHIFT; low 27 bits are layout flags
//   [1]                       payload_words = control block + pack pads + shipped ring words
//   [2 .. 2+RISC)             ring head each RISC's run starts at, relay-written
//   [7]                       the core's NoC coordinate (y << 16 | x), relay-written
//   [PREFIX .. +CONTROL)      the SPSC_SPAN_WIRE_CTRL_WORDS control block
//   [.. +payload)             per RISC in ascending order with a live run: spsc_span_pack_pad() skipped words,
//                             then the run, ring wrap resolved into a flat array
//   [.. frame_words)          skipped words up to a 64 B socket page
//
// The host recomputes the geometry from the control block (per RISC, the run is head..tail), so the wire carries
// no lane tags, run lengths or core id. The NIU gathers each live
// window straight into the host FIFO, and a NoC write mis-delivers a transfer whose destination is not
// congruent to the source modulo NOC_PCIE_WRITE_ALIGNMENT_BYTES (16 B), so each run is preceded by
// spsc_span_pack_pad() skipped words, never written; the host reads past them. The 8-word prefix puts the
// control block at 32 B and the payload at 96 B, both NoC-alignment multiples.
constexpr static std::uint32_t SPSC_SPAN_PREFIX_WORDS = 8;
// Wire type code. Must equal PP_BULK_SPAN in tt_metal/impl/streaming_profiler/spsc_packet.h, which is plain C and
// cannot include this header; spsc_marker_decode.hpp static_asserts that the two agree.
constexpr static std::uint32_t SPSC_SPAN_PACKET_TYPE = 13;
// Where the packet type sits in word0 of every packet in this stream (PP_TYPE_SHIFT in spsc_packet.h).
constexpr static std::uint32_t SPSC_SPAN_TYPE_SHIFT = 27;
// Socket page granularity in words; frames pad up to a whole number. 64 B is the host socket's PCIe alignment (its
// smallest legal page); larger pages concentrate the same credit wait into stalls long enough to miss the ring-fill
// deadline.
constexpr static std::uint32_t SPSC_SPAN_PAGE_WORDS = 16;

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
static_assert(
    (PROFILER_L1_VECTOR_SIZE & (PROFILER_L1_VECTOR_SIZE - 1)) == 0,
    "relay and decoder mask ring offsets with PROFILER_L1_VECTOR_SIZE - 1");
constexpr bool spsc_span_wrap_image(std::uint32_t start, std::uint32_t extent, std::uint32_t ring_cap) {
    return (start & (ring_cap - 1u)) + extent > ring_cap && ring_cap - extent <= SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}

inline std::uint32_t spsc_span_w0() { return SPSC_SPAN_PACKET_TYPE << SPSC_SPAN_TYPE_SHIFT; }

// Control block of a packed frame: control-vector words SPSC_WIRE_CV_BASE..+16 exactly as the relay's one 64 B NoC
// read of them lands in the frame -- the lanes' state slots and their tails. The heads and XY the relay adds go in
// the prefix (SpscWirePrefix).
constexpr static std::uint32_t SPSC_SPAN_WIRE_CTRL_WORDS = 16;
constexpr static std::uint32_t SPSC_WIRE_CV_BASE = 16;
// Most bytes a relay pushes between two bytes_sent writes, so everything landed lies below the bytes_sent the host
// reads plus this: a host reader that copied a frame proves the device had not reached it.
constexpr static std::uint32_t SPSC_NOTIFY_CAP_BYTES = 128u * 1024u;
enum SpscWireCtrl : std::uint32_t {
    SPSC_WIRE_TIMER_0 = SPSC_STATE_TIMER_0 - SPSC_WIRE_CV_BASE,  // 0..4
    SPSC_WIRE_TAIL_0 = SPSC_RING_TAIL_0 - SPSC_WIRE_CV_BASE,     // 8..12
    SPSC_WIRE_LINK_SYNC_TAIL = SPSC_LINK_SYNC_TAIL - SPSC_WIRE_CV_BASE,
};
constexpr std::uint32_t spsc_wire_prog_word(std::uint32_t risc) {  // 5..7, 13..14
    return spsc_state_prog_word(risc) - SPSC_WIRE_CV_BASE;
}
enum SpscWirePrefix : std::uint32_t {
    SPSC_PREFIX_PAYLOAD_WORDS = 1,
    SPSC_PREFIX_HEAD_0 = 2,  // ..6
    SPSC_PREFIX_XY = 7,
    SPSC_PREFIX_SYNC_RECORD_COUNT = SPSC_PREFIX_HEAD_0,
};
static_assert(
    SPSC_WIRE_CV_BASE % 16 == 0 && SPSC_STATE_TIMER_0 >= SPSC_WIRE_CV_BASE &&
        SPSC_WIRE_TAIL_0 + PROFILER_SPSC_TENSIX_RISC <= SPSC_SPAN_WIRE_CTRL_WORDS &&
        spsc_wire_prog_word(PROFILER_SPSC_TENSIX_RISC - 1) < SPSC_SPAN_WIRE_CTRL_WORDS &&
        SPSC_WIRE_LINK_SYNC_TAIL < SPSC_SPAN_WIRE_CTRL_WORDS &&
        SPSC_PREFIX_HEAD_0 + PROFILER_SPSC_TENSIX_RISC == SPSC_PREFIX_XY && SPSC_PREFIX_XY < SPSC_SPAN_PREFIX_WORDS,
    "the control block is one 64 B window of the control vector; heads and XY fit the prefix");

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

static constexpr std::uint32_t SPSC_TYPE_ZONE_L = 4;      // >3.2 s zone: id | end_lo | end_hi | dur_lo | dur_hi
static constexpr std::uint32_t SPSC_TYPE_STICKY_TIMER = 9;
static constexpr std::uint32_t SPSC_TIMER_HI_MASK = 0x7FFFFFFu;  // the 27-bit low field of word0

// Marker ids are the FULL 27 bits. The mask was 0xFFFF once, and that truncation was invisible: markers rendered
// perfectly and only their NAMES could not be resolved. Any change to the id width has to be made in every packer
// at once: ppfmt in kernel_profiler_streaming.hpp and pp_* in spsc_packet.h.
// DATA is 3 + N words: word0 is shaped like a zone marker (type | 27-bit id), the length lives in word2.
static constexpr std::uint32_t SPSC_TYPE_DATA = 10;
static constexpr std::uint32_t SPSC_DATA_SIZE_SHIFT = 25;

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

// ---- Clock sync: records, the sample ring, link ends and the tile network ------------------------------------------

constexpr std::uint32_t kEthRefclkHz = 50'000'000u;

enum class SyncKind : std::uint32_t { Local, Link, Ruler };
// The stamps a link record averages. A round's forward frames go from the sender to the receiver and its return frames
// back: the receiver records the forward frames' egress and ingress, the sender the return frames'.
enum class SyncRole : std::uint32_t { ForwardEgress, ForwardIngress, ReturnEgress, ReturnIngress };

struct SyncMeta {
    std::uint32_t count : 2;  // Local, Ruler
    std::uint32_t dense : 1;  // Ruler: every update around a clock change, not one in kSyncRulerKeepEvery
    SyncRole role : 2;        // Link
    std::uint32_t rsvd0 : 3;
    SyncKind kind : 8;
    std::uint32_t rsvd1 : 16;
};

constexpr std::uint32_t kSyncLocalPoints = 3;
static_assert(kSyncLocalPoints < 4, "SyncMeta::count holds a record's point count in 2 bits");
// Blackhole's AICLK moves in 6.25 MHz PLL steps, an eighth of the refclk's 50 MHz, so the wall clock gains a whole
// number of eighths of a tick per refclk tick.
struct SyncLocalRates {
    // Each point's wall ticks per refclk tick, in eighths; 0 for a single sample or a group centroid.
    std::uint8_t wall_per_refclk_eighths[kSyncLocalPoints];
    std::uint8_t base_wall_per_refclk_eighths;  // the rate every step's wall_off is measured against
};
// A later point of a record, against its first.
struct SyncLocalStep {
    std::uint32_t refclk_from_first : 16;
    // Eighths of a tick, from the first point's wall plus base_wall_per_refclk_eighths times refclk_from_first.
    std::int32_t wall_off : 16;
};
static_assert(sizeof(SyncLocalRates) == sizeof(std::uint32_t) && sizeof(SyncLocalStep) == sizeof(std::uint32_t));
constexpr bool sync_local_step_fits(std::uint64_t refclk_from_first, std::int32_t wall_off) {
    return refclk_from_first <= 0xFFFFu && wall_off >= -32768 && wall_off <= 32767;
}
struct SyncLocalRecord {
    std::uint32_t meta;
    std::uint32_t rates;
    std::uint64_t first_refclk;
    std::uint64_t first_wall8;
    std::uint32_t from_first[kSyncLocalPoints - 1];  // SyncLocalSteps
};

// The average is first + sum_from_first_ns / count, in ns.
struct SyncLinkRecord {
    std::uint32_t meta;
    std::uint32_t round;
    std::uint64_t first;  // the first stamp less the stamping timer's PTP offset, ns
    std::uint64_t sum_from_first_ns;
    std::uint32_t count;
    std::uint32_t rsvd;
};

struct SyncHeader {
    std::uint32_t meta;
};
// A sync frame is the SPSC prefix, with the record count in SPSC_PREFIX_SYNC_RECORD_COUNT, and then its records, padded
// to at least SPSC_SPAN_WIRE_CTRL_WORDS words. The kind in header.meta says which member a record is.
union SyncRecord {
    SyncHeader header;
    SyncLocalRecord local;
    SyncLinkRecord link;
};
static_assert(sizeof(SyncRecord) == 32 && alignof(SyncRecord) == 8);
constexpr std::uint32_t kSyncRecordWords = sizeof(SyncRecord) / sizeof(std::uint32_t);

constexpr std::uint32_t kSyncRingRecords = 512;
constexpr std::uint32_t kSyncFrameRecords = 32;

// The model publishes its position as the ring's head, and the sampler never gets more than a ring ahead of it. To stop
// the sampler the model publishes its position plus kSyncHeadStop: the sampler's next reload of the head then reads it
// as ahead of its own tail and returns, so stopping costs the stream no load it doesn't already do.
constexpr std::uint32_t kSyncSampleRingSamples = 32768;
constexpr std::uint32_t kSyncHeadStop = 1u << 31;
// A refclk update: the refclk's new low word and the wall clock's low word at the step, in eighths.
struct SyncSample {
    std::uint32_t refclk, wall8;
};
struct SyncSampleRing {
    std::uint32_t tail;
    std::uint32_t done;
    // An instant read before the first sample, the base the low words widen against: its refclk and its wall clock in
    // eighths.
    std::uint64_t refclk, wall8;
    std::uint32_t head;
    std::uint32_t rsvd[9];
    SyncSample samples[kSyncSampleRingSamples];
};
static_assert(offsetof(SyncSampleRing, refclk) == 8 && offsetof(SyncSampleRing, samples) == 64);

// Away from a clock change, the sync check's ruler (eth_clock_ruler.cpp) keeps one update in this many, and the host
// weights each by it.
constexpr std::uint32_t kSyncRulerKeepEvery = 8;

constexpr std::uint32_t kLinkSyncPaceTicks = 500'000;  // a round every 10 ms
// With the sync check a link runs a round every 1 ms, and every kLinkSyncCheckSolveEvery-th round feeds the link solve.
constexpr std::uint32_t kLinkSyncCheckPaceTicks = 50'000;
constexpr std::uint32_t kLinkSyncCheckSolveEvery = kLinkSyncPaceTicks / kLinkSyncCheckPaceTicks;
enum class LinkSyncCtl : std::uint32_t { Idle, Run, Stop };
enum class LinkSyncRole : std::uint32_t { None, Sender, Receiver };
constexpr std::uint32_t kLinkSyncSlotWords = 96;  // kernels/link_sync.hpp's frames in flight
constexpr std::uint32_t kLinkSyncRingRecords = 8;
static_assert(
    (kSyncRingRecords & (kSyncRingRecords - 1)) == 0 && (kSyncSampleRingSamples & (kSyncSampleRingSamples - 1)) == 0 &&
    (kLinkSyncRingRecords & (kLinkSyncRingRecords - 1)) == 0);
static_assert(kLinkSyncRingRecords <= kSyncFrameRecords);

// A link end's L1, at the top of the active eth core's unreserved region, at the same address on both ends.
struct LinkSyncL1 {
    std::uint32_t slots[kLinkSyncSlotWords];
    LinkSyncCtl ctl;
    std::uint32_t done;
    alignas(32) SyncRecord ring[kLinkSyncRingRecords];
};

// The idle eth cores share one L1 layout, so one region is the tracker's sample ring and the eth relay's scratch.
constexpr std::uint32_t kEthSyncScratchBytes = sizeof(SyncSampleRing);
constexpr std::uint32_t kEthRelayMaxDrained = 16;  // eth cores the eth relay drains besides the tracker

constexpr std::uint32_t kTileNetMaxPartners = 40;
constexpr std::uint32_t kTileNetBins = 128;
enum class TileNetGo : std::uint32_t { Wait, Measure, Exit };
// The host zeroes the table before launch, so Launched is 0.
enum class TileNetReady : std::uint32_t { Launched, Up, Done };

// A tile's read of one partner, as its kernel argument.
struct TileNetRead {
    std::uint32_t x : 16;
    std::uint32_t y : 15;
    std::uint32_t noc : 1;
};
struct TileNetPartner {
    std::int64_t coarse;          // the whole-clock difference, partner minus this tile
    std::int32_t doubled_median;  // the median of 2 * (partner wall - bracket midpoint), in the clocks' low words
    std::uint32_t rsvd;
};
struct TileNetTable {
    TileNetGo go;
    TileNetReady ready;
    TileNetPartner partner[kTileNetMaxPartners];
};
struct TileNetScratch {
    std::uint32_t landing[16];
    TileNetTable table;
    std::uint32_t hist[kTileNetBins];
};

}  // namespace kernel_profiler
