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
// Consumers: the SPSC producer (kernel_profiler_streaming.hpp), the DRISC relay kernel
// (tools/profiler/kernels/streaming_profiler_relay.cpp) and the host receiver
// (impl/streaming_profiler/streaming_profiler_receiver.cpp, spsc_marker_decode.hpp).

#include <cstdint>

#include "hostdev/profiler_common.h"
#include "hostdev/profiler_zone_id.h"

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
    // Per Tensix RISC, the timer high word and runtime id in effect at the published tail, for a decoder reseeding a
    // lane after a loss. The producer stores its tail, fences, then the state, so a reader that observes the state
    // no later than the tail never sees a value the frame's words do not already carry inline. They live in the
    // tails' 64 B block (words 16..31), in the words no Tensix RISC owns -- heads 16..23 and tails 29..30 -- so the
    // relay's one 64 B read takes state and tails in a single L1 access. Runtime ids: spsc_state_prog_word.
    SPSC_STATE_TIMER_0 = 16,
    SPSC_STATE_PROG_0 = 21,
    // Host->kernel arm: while set a producer blocks on a full ring, because a relay is draining this core; while
    // clear it proceeds and overwrites. The host clears it on every Tensix core of the device at session start
    // (this backend does not touch the firmware) and sets it only on the cores its relays serve, so a core nobody
    // drains (dispatch cores, whose rings fill one launch at a time across processes) can never park in the stall
    // path and wedge wait_until_cores_done() at device close.
    PROFILER_ARMED = 2 * PROFILER_SPSC_MAX_RISC,
    // Reserved; the core's NoC coordinate reaches the wire from the host's core list via the relay (SPSC_PREFIX_XY).
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of full-ring blocks, written in the stall path and read by the host from L1 at teardown;
    // counting decoded stall markers would undercount, since a marker can be dropped between the relay frame and
    // the BroadcastRing. 8 slots so SPSC_CONTROL_END stays inside the 64-word vector.
    SPSC_STALL_COUNT_0 = 2 * PROFILER_SPSC_MAX_RISC + 2,
    SPSC_STALL_COUNT_MAX = 8,
    // On an active eth core hosting a link end: the tail of its sync ring (kLinkSyncRingOffset), records written.
    // Here rather than in the link L1 so the pusher's per-sweep read of this vector already carries it.
    SPSC_LINK_SYNC_TAIL = SPSC_STALL_COUNT_0 + SPSC_STALL_COUNT_MAX,
    SPSC_CONTROL_END = SPSC_LINK_SYNC_TAIL + 1,  // first unused word; grow the layout here
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
        SPSC_STATE_PROG_0 + 3 == SPSC_RING_TAIL_0 && spsc_state_prog_word(PROFILER_SPSC_TENSIX_RISC - 1) < 32,
    "lane state must fill the unowned words of the tails' 64 B block");

// Host->relay stop word: quiesce drains everything with every wait still holding, then the relay exits.
static constexpr std::uint32_t kRelayStopQuiesce = 1;
// Relay->host completion words; the host matches the high half. Drained: the relay's last page is out and the host
// may return every credit. Done follows the socket barrier.
static constexpr std::uint32_t kRelayDrainedWord = 0xD09D0000u;
static constexpr std::uint32_t kRelayDoneWord = 0xD09E0000u;
static constexpr std::uint32_t kRelayDoneMask = 0xFFFF0000u;
// Each relay control word owns a 64 B pad, so the words that share it (the sync rendezvous triple behind
// the stop word, the heartbeat behind done) travel in one host write.
static constexpr std::uint32_t kRelayCtrlWordStride = 64;

// Tile clock network scratch: the first 64 B take the landing word of the reads, the table follows at kTileNetTable,
// two histograms of kTileNetBins uint32 counts (offsets, then round trips) at kTileNetHist.
//   [TILE_NET_GO]        host-written: kTileNetGoMeasure when this tile's turn comes, kTileNetGoExit to release it
//   [TILE_NET_READY]     the host's nonce once the tile is up, its inverse once every partner is written
//   [TILE_NET_OUT_0 ..)  per partner: the median of 2 * (partner wall - bracket midpoint) in the clocks' low words,
//                        the spread between its quartiles, the median round trip (int32 ticks), then the coarse
//                        whole-clock difference partner - this tile (int64, low word first)
enum TileNetTable : std::uint32_t {
    TILE_NET_GO = 0,
    TILE_NET_READY = 1,
    TILE_NET_OUT_0 = 2,
    TILE_NET_OUT_WORDS = 5,
};
static constexpr std::uint32_t kTileNetGoMeasure = 1;
static constexpr std::uint32_t kTileNetGoExit = 2;
static constexpr std::uint32_t kTileNetTable = 64;
static constexpr std::uint32_t kTileNetMaxPartners = 32;
static constexpr std::uint32_t kTileNetHist =
    kTileNetTable + 4 * (TILE_NET_OUT_0 + TILE_NET_OUT_WORDS * kTileNetMaxPartners);
static constexpr std::uint32_t kTileNetBins = 128;
static constexpr std::uint32_t kTileNetScratchBytes = kTileNetHist + 2 * 4 * kTileNetBins;

// The device-to-device link sync's contract between the host, its resident kernels and the fabric routers: the eth
// tile's refclk, the unit a round's stamp averages are reported in, the L1 the two ends own at the top of the active
// eth core's unreserved region with the control word inside it (done at +4, diagnostics from +8), and the round
// period in refclk ticks.
static constexpr std::uint32_t kEthRefclkHz = 50'000'000u;
static constexpr std::uint32_t kLinkSyncStampUnitsPerNs = 4;
static constexpr std::uint32_t kLinkSyncL1Bytes = 800;
static constexpr std::uint32_t kLinkSyncCtlOffset = 480;
static constexpr std::uint32_t kLinkSyncCtlRun = 1, kLinkSyncCtlStop = 2;
static constexpr std::uint32_t kLinkSyncPaceTicks = 500'000;  // a round every 10 ms

// The sync's records, 8 words: [SYNC_META] kind << 8 | role; [SYNC_ROUND]; the reading and the wall clock at it as
// two words each; a link record also carries the refclk read with that wall clock. A link end writes a round's two
// stamp averages into the ring at the end of its link L1 (kLinkSyncRingRecords slots) and publishes the count in its
// control vector (SPSC_LINK_SYNC_TAIL); it never waits for a reader, so a pusher a whole ring behind loses the oldest.
// The pusher reads every linked core's control vector each sweep regardless, reads the ring when the tail moved, keeps
// its own clock model's points in a ring of kSyncRingRecords in its L1, and ships them all on its sync socket as sync
// frames: the SPSC frame prefix (w0, payload words, the source core's XY) with the record count at SPSC_PREFIX_HEAD_0,
// then the records, the payload padded to SPSC_SPAN_WIRE_CTRL_WORDS at least so the ingest's frame walk accepts it.
// Never a profiler record: the sync engine reads its socket itself.
static constexpr std::uint32_t kSyncRecordWords = 8;
enum SyncRecordWord : std::uint32_t {
    SYNC_META = 0,
    SYNC_ROUND,
    SYNC_VALUE_LO,
    SYNC_VALUE_HI,
    SYNC_WALL_LO,
    SYNC_WALL_HI,
    SYNC_REF_LO,  // link records: the refclk read together with the wall clock above
    SYNC_REF_HI,
};
static_assert(SYNC_REF_HI < kSyncRecordWords);
// LOCAL: a point of the chip's clock model, value the refclk, wall its line there, round = k8 | n << 8. LINK: a
// round's 1588 stamp average in kLinkSyncStampUnitsPerNs per ns.
static constexpr std::uint32_t kSyncKindLocal = 0, kSyncKindLink = 1;
static constexpr std::uint32_t kSyncLocalPoint = 0, kSyncLocalClose = 1;
static constexpr std::uint32_t kSyncRoleT0 = 0, kSyncRoleT1 = 1, kSyncRoleT1B = 2, kSyncRoleT2 = 3;
static constexpr std::uint32_t kLinkSyncRingOffset = 544;
static constexpr std::uint32_t kLinkSyncRingRecords = 8;
static constexpr std::uint32_t kSyncRingRecords = 128;
static constexpr std::uint32_t kSyncRingBytes = kSyncRingRecords * kSyncRecordWords * 4;
static constexpr std::uint32_t kSyncFrameRecords = 32;  // records per sync frame at most
static_assert(kLinkSyncRingOffset + kLinkSyncRingRecords * kSyncRecordWords * 4 <= kLinkSyncL1Bytes);
static_assert(kLinkSyncRingRecords <= kSyncFrameRecords);

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
};
constexpr std::uint32_t spsc_wire_prog_word(std::uint32_t risc) {  // 5..7, 13..14
    return spsc_state_prog_word(risc) - SPSC_WIRE_CV_BASE;
}
enum SpscWirePrefix : std::uint32_t {
    SPSC_PREFIX_HEAD_0 = 2,  // ..6
    SPSC_PREFIX_XY = 7,
};
static_assert(
    SPSC_WIRE_CV_BASE % 16 == 0 && SPSC_STATE_TIMER_0 >= SPSC_WIRE_CV_BASE &&
        SPSC_WIRE_TAIL_0 + PROFILER_SPSC_TENSIX_RISC <= SPSC_SPAN_WIRE_CTRL_WORDS &&
        spsc_wire_prog_word(PROFILER_SPSC_TENSIX_RISC - 1) < SPSC_SPAN_WIRE_CTRL_WORDS &&
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

static constexpr std::uint32_t SPSC_TYPE_ZONE_L = 4;  // >3.2 s zone: id | end_lo | end_hi | dur_lo | dur_hi
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

}  // namespace kernel_profiler
