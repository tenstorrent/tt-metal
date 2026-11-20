// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Drain job: the streaming profiler's drainer, which the DRISC service kernel runs in holds (abi::DrainJobBlock). It is
// the former drisc_relay.cpp metal kernel. It gathers the worker cores' profiler rings into staged frames, ships the
// frames to a GDDR spool with the seat's DMA engine, and drains the spool to the host through a D2H socket.
//
// While it holds the seat it owns the bounce arena (its staging overlays it), every command buffer and transaction ID
// on the read NoC, and write_cmd_buf on NoC0 for its host writes. It re-arms them at the start of every hold and leaves
// them idle before it hands the seat back. Only abi::DrainLayout survives between holds.
//
// The NoC1 seat gathers on NoC1 and writes to the host on NoC0, as the metal relay did. The NoC0 seat's NIU for NoC1
// is held in NOC2AXI mode, so it gathers and writes on NoC0: its self reads move to read_cmd_buf and its head writes to
// write_reg_cmd_buf, and its host writes carry their own transaction ID so they never count against the reads'.
//
// GDDR DMA streams stay as the relay had them (frames to the spool on TX stream 0's write queue, spool to bounce on TX
// stream 1's read queue); the service uses stream 0's reads and stream 1's writes, so the two never share a queue.

#include <cstdint>

#include "api/alignment.h"
#include "hostdev/socket.h"
#include "hostdev/streaming_profiler_common.h"
#include "kernel_utils.hpp"
#include "l1_layout.hpp"

namespace drain {

constexpr uint8_t kEgressNoc = 0;
constexpr uint8_t kReadNoc = NOC_INDEX;
constexpr bool kSingleNoc = kReadNoc == kEgressNoc;

constexpr uint32_t kStageBase = l1::BOUNCE_BUFFER_START;
constexpr uint32_t kNStage = 7;
constexpr uint32_t kCoreRecords = abi::DrainLayout::records_addr;
constexpr uint32_t kCtrlAddr = abi::DrainLayout::ctrl_addr;
constexpr uint32_t kSocketConfigAddr = abi::DrainLayout::socket_config_addr;
constexpr uint32_t kMaxCores = abi::DRAIN_MAX_CORES;
static_assert(kMaxCores <= 256, "the core lists index cores as bytes");

// ---- Frame packing, from relay_common.hpp ----
constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kPrefixWords = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireControlWords = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u;

FORCE_INLINE constexpr uint32_t page_round(uint32_t bytes) { return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u); }

constexpr bool wrap_image(uint32_t take) {
    return take - (kRingWords - kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS) <=
           kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}
static_assert(
    [] {
        for (uint32_t take = 1; take < 2 * kRingWords; take++) {
            if (wrap_image(take) != kernel_profiler::spsc_span_wrap_image(kRingWords - 1u, take, kRingWords)) {
                return false;
            }
        }
        return true;
    }(),
    "the inline image test drifted from spsc_span_wrap_image");

template <typename Read>
FORCE_INLINE uint32_t place_run(uint32_t start, uint32_t take, uint32_t frame_words, uint32_t frame, Read&& read) {
    const uint32_t ring_start = start & (kRingWords - 1u);
    if (ring_start + take <= kRingWords) {
        frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
        read(ring_start * 4u, frame + frame_words * 4u, take * 4u, true);
        return frame_words + take;
    }
    if (wrap_image(take)) {
        frame_words += kernel_profiler::spsc_span_pack_pad(0u, frame_words);
        read(0u, frame + frame_words * 4u, kRingWords * 4u, true);
        return frame_words + kRingWords;
    }
    frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
    const uint32_t first = kRingWords - ring_start;
    uint32_t dst = frame + frame_words * 4u;
    const uint32_t first_bytes = first * 4u;
    read(ring_start * 4u, dst, first_bytes, true);
    uint32_t rest = take;
    asm volatile("" : "+r"(dst), "+r"(rest), "+r"(frame_words));
    frame_words += rest;
    rest -= first;
    read(0u, dst + first_bytes, rest * 4u, false);
    return frame_words;
}

FORCE_INLINE void staged_store_fence() { asm volatile("fence" ::: "memory"); }

// ---- D2H socket sender, the subset of api/socket_api.h the job uses ----
constexpr uint32_t kSenderMdBytes = align_power_of_2(sizeof(sender_socket_md), 16u);
constexpr uint32_t kBytesAckedBytes = align_power_of_2(sizeof(uint32_t), 16u);
static_assert(
    kSenderMdBytes + kBytesAckedBytes + align_power_of_2(sizeof(d2h_sender_socket_md), 16u) <=
        abi::DRAIN_SOCKET_CONFIG_BYTES,
    "the D2H socket config outgrew its slot in abi::DrainLayout");

inline SocketSenderInterface create_sender(uint32_t config_addr) {
    volatile tt_l1_ptr sender_socket_md* md = reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(config_addr);
    SocketSenderInterface s{};
    s.config_addr = config_addr;
    s.write_ptr = md->write_ptr;
    s.bytes_sent = md->bytes_sent;
    s.bytes_acked_base_addr = config_addr + kSenderMdBytes;
    s.num_downstreams = md->num_downstreams;
    s.downstream_fifo_total_size = md->downstream_fifo_total_size;
    s.downstream_fifo_addr = md->downstream_fifo_addr;
    s.downstream_bytes_sent_addr = md->downstream_bytes_sent_addr;
    s.is_d2h = md->is_d2h;
    volatile tt_l1_ptr d2h_sender_socket_md* d2h =
        reinterpret_cast<volatile tt_l1_ptr d2h_sender_socket_md*>(config_addr + kSenderMdBytes + kBytesAckedBytes);
    s.d2h.bytes_sent_addr_hi = d2h->bytes_sent_addr_hi;
    s.d2h.data_addr_hi = d2h->data_addr_hi;
    s.d2h.pcie_xy_enc = d2h->pcie_xy_enc;
    return s;
}

inline void set_page_size(SocketSenderInterface& s, uint32_t page_size) {
    const uint32_t total = s.downstream_fifo_total_size;
    uint32_t next = align_power_of_2(s.write_ptr, page_size);
    const uint32_t aligned_size = total - total % page_size;
    if (next >= aligned_size) {
        s.bytes_sent += total - next;
        next = 0;
    }
    s.write_ptr = next;
    s.page_size = page_size;
    s.downstream_fifo_curr_size = aligned_size;
}

FORCE_INLINE void push_pages(SocketSenderInterface& s, uint32_t num_pages) {
    const uint32_t num_bytes = num_pages * s.page_size;
    if (s.write_ptr + num_bytes >= s.downstream_fifo_curr_size) {
        s.write_ptr = s.write_ptr + num_bytes - s.downstream_fifo_curr_size;
        s.bytes_sent += num_bytes + s.downstream_fifo_total_size - s.downstream_fifo_curr_size;
    } else {
        s.write_ptr += num_bytes;
        s.bytes_sent += num_bytes;
    }
}

inline void update_socket_config(const SocketSenderInterface& s) {
    volatile tt_l1_ptr sender_socket_md* md = reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(s.config_addr);
    md->bytes_sent = s.bytes_sent;
    md->write_ptr = s.write_ptr;
}

// ---- Layout, from drisc_relay.cpp ----
constexpr uint32_t kShipMinPct = 25;
constexpr uint32_t kNumRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
static_assert(kNumRisc == 5, "the control scans are unrolled for exactly five RISCs");
constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;
constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
constexpr uint32_t kSlotBytes = kSlotWords * 4u;
constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefixWords - kWireControlWords;
static_assert(
    kNumRisc * (kRingWords + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1u) <= kPayloadCapWords,
    "a full span no longer fits a slot");
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint8_t kDmaShip = 0;   // TX stream 0's write queue: staging -> spool
constexpr uint8_t kDmaDrain = 1;  // TX stream 1's read queue: spool -> bounce
constexpr uint32_t kGenSlots = 2;
constexpr uint32_t kNBounce = 2;
constexpr uint32_t kNGens = 2;
static_assert(kNStage >= kNBounce + kNGens * kGenSlots, "the staging arena must hold two generations and the bounces");
static_assert(kNGens * 4 < experimental::kMaxOutstandingWrites, "the ship stream could fill and drop an issue");
static_assert(kNBounce < experimental::kMaxOutstandingReads, "the drain stream could fill and drop an issue");
constexpr uint32_t kCvBaseWord = kernel_profiler::SPSC_WIRE_CV_BASE;
constexpr uint32_t kCvReadBytes = 64;
constexpr uint32_t kCvReadSrcOff = kCvBaseWord * 4u;
constexpr uint32_t kRecordBytes = kernel_profiler::kRelayCoreRecordBytes;
static_assert(kRecordBytes == abi::DRAIN_RECORD_BYTES, "abi::DrainLayout sizes the records for another record size");
static_assert(sizeof(kernel_profiler::ResidentCtrl) <= abi::DRAIN_CTRL_BYTES, "ResidentCtrl outgrew its slot");
constexpr uint32_t kTailWord = kernel_profiler::SPSC_RING_TAIL_0 - kCvBaseWord;
constexpr uint32_t kTimerWord = kernel_profiler::SPSC_STATE_TIMER_0 - kCvBaseWord;
constexpr uint32_t kHeadWord = kCvReadBytes / 4u;
constexpr uint32_t kXyWord = kHeadWord + kNumRisc;
constexpr uint32_t kPeakWord = kXyWord + 1;
static_assert((kPeakWord + 1u) * 4u <= kRecordBytes, "the core record overflows its bytes");
static_assert(
    kCvReadSrcOff % 64u == 0 && kCvReadBytes == 64u && kTimerWord < kTailWord && kTailWord + kNumRisc <= kHeadWord &&
        kernel_profiler::spsc_state_prog_word(kNumRisc - 1u) < kCvBaseWord + kHeadWord,
    "the control read is the one 64 B block holding the state slots and the tails");
constexpr uint32_t kBounceBase0 = kStageBase + kNGens * kGenSlots * kSlotBytes;
constexpr uint32_t kGenBytes = kGenSlots * kSlotBytes;
constexpr uint32_t kGenBase[kNGens] = {kStageBase, kStageBase + kGenBytes};
constexpr uint32_t kBounceBytes = ((kNStage - kNGens * kGenSlots) * kSlotBytes / 2u) & ~(kPageBytes - 1u);
static_assert(kBounceBase0 % kPageBytes == 0, "bounces start on a page");
static_assert(kBounceBytes <= NOC_MAX_BURST_SIZE && kSlotBytes <= NOC_MAX_BURST_SIZE, "every host write is one burst");
static_assert(
    kBounceBytes <= kernel_profiler::SPSC_NOTIFY_CAP_BYTES && 2u * kSlotBytes <= kernel_profiler::SPSC_NOTIFY_CAP_BYTES,
    "a single push must fit under the notify cap");
static_assert(
    kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
    "bounces must fit inside the staging arena");
static_assert(kNStage * kSlotBytes <= l1::BOUNCE_BUFFER_SIZE, "the drain job's staging must fit the bounce arena");
constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
constexpr uint32_t kMaxDeferSweeps = 2048;
constexpr uint32_t kIdlePumpStride = 8;
static_assert(kSpanWords * 4u <= NOC_MAX_BURST_SIZE, "a span read must fit one NoC burst");
static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a whole-ring gather must fit one NoC burst");
static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
static_assert(
    kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_PCIE_WRITE_ALIGNMENT_BYTES &&
        kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_L1_READ_ALIGNMENT_BYTES,
    "the shared pad rule no longer matches this part's NoC congruence");
static_assert(
    kRingWords % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        (kPrefixWords + kWireControlWords) % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        kStageBase % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0 &&
        kSlotBytes % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0,
    "packed-gather congruence broken");

FORCE_INLINE volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl() {
    return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
}
FORCE_INLINE volatile abi::DrainJobBlock* job_block() {
    return reinterpret_cast<volatile abi::DrainJobBlock*>(abi::DrainLayout::job_addr);
}

// Command buffers. The read NoC's four are programmed at every hold; a per-core command is then the coordinate, one
// address and the send.
constexpr uint32_t kCvCmdBuf = write_at_cmd_buf;
constexpr uint32_t kSelfCmdBuf = kSingleNoc ? read_cmd_buf : write_reg_cmd_buf;
constexpr uint32_t kHeadCmdBuf = kSingleNoc ? write_reg_cmd_buf : write_cmd_buf;
constexpr uint32_t kGatherTxn = 1;
constexpr uint32_t kProbeTxn = 2;
constexpr uint32_t kEgressTxn = 3;  // host writes; on the NoC0 seat they share the NIU with the reads

// ---- Job state; plain globals, since the service's crt1 runs no constructors ----
uint32_t write_vc;
uint32_t spool_base;
uint32_t spool_bytes;
uint32_t hold_turns;
uint32_t hold_sweeps;
uint32_t num_cores;
uint32_t ring_base;   // lane 0's ring on every worker: the control block plus its control words
uint32_t self_coord;  // this core's NOC_TARG_ADDR_COORDINATE word
uint32_t core_coord[kMaxCores];
SocketSenderInterface sender;

FORCE_INLINE uint32_t record(uint32_t c) { return kCoreRecords + c * kRecordBytes; }

// The relay's write_to_host: L1 to host memory through write_cmd_buf on NoC0, whose state the hold programmed.
inline void write_to_host(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<DM_DEDICATED_NOC, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        kEgressNoc, src_l1, pcie_xy_enc, dst_pcie, size, 1);
}

inline void push_fifo(const SocketSenderInterface& s, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = s.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = kernel_profiler::join_words(s.d2h.data_addr_hi, s.downstream_fifo_addr);
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host(s.d2h.pcie_xy_enc, src, base + dst, first);
    if (first < len) {
        write_to_host(s.d2h.pcie_xy_enc, src + first, base, len - first);
    }
}

// bytes_sent goes out once every push before it is acked: the PCIe tile orders nothing between NoC writes.
inline void notify_bytes_sent(const SocketSenderInterface& s) {
    while (!ncrisc_noc_nonposted_writes_flushed(kEgressNoc)) {
    }
    reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(s.config_addr)->bytes_sent = s.bytes_sent;
    staged_store_fence();
    const uint64_t dst = kernel_profiler::join_words(s.d2h.bytes_sent_addr_hi, s.downstream_bytes_sent_addr);
    write_to_host(s.d2h.pcie_xy_enc, s.config_addr, dst, 4u);
}

// ---- Spool pump, from drisc_relay.cpp; the spool is the job block's, so its bands are set at start ----
struct SpoolPump {
    enum : uint32_t { kEmpty = 0, kReading = 1, kReady = 2, kShipping = 3 };
    static constexpr uint32_t kNone = 2;
    enum : uint32_t { kLevelIdle = 0, kLevelEverySweep = 1, kLevelInline = 2 };
    struct Bounce {
        uint32_t state;
        uint32_t bytes;
        uint32_t off;
        uint32_t ack_target;
        uint32_t seq;
        uint32_t rd_mark;
        uint64_t rd_end;
    };

    uint64_t wr;
    uint64_t done;
    uint64_t rd_iss;
    uint64_t rd;
    uint32_t wr_off;
    uint32_t rd_iss_off;
    uint32_t dma_issued;
    uint32_t dma_rd_issued;
    uint32_t chunks;
    Bounce b[2];
    uint32_t notified;
    uint32_t acked_seen;
    uint32_t level;
    uint32_t band_every_sweep;
    uint32_t band_inline;
    volatile tt_l1_ptr uint32_t* acked_;

    void init() {
        wr = done = rd_iss = rd = 0;
        wr_off = rd_iss_off = dma_issued = dma_rd_issued = chunks = 0;
        b[0].state = kEmpty;
        b[1].state = kEmpty;
        notified = sender.bytes_sent;
        acked_seen = 0;
        level = kLevelIdle;
        band_every_sweep = spool_bytes / 2u;
        band_inline = spool_bytes / 2u + spool_bytes / 8u;
        acked_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr);
    }

    uint32_t occupancy() const { return static_cast<uint32_t>(wr - rd); }
    bool has_room(uint32_t bytes) const { return spool_bytes - occupancy() >= bytes; }
    bool both_empty() const { return b[0].state == kEmpty && b[1].state == kEmpty; }
    bool drained() const { return rd_iss == wr && both_empty(); }
    uint32_t first_empty() const { return b[0].state == kEmpty ? 0u : (b[1].state == kEmpty ? 1u : kNone); }
    uint32_t oldest_ready() const {
        const bool r0 = b[0].state == kReady;
        const bool r1 = b[1].state == kReady;
        if (r0 && r1) {
            return static_cast<int32_t>(b[0].seq - b[1].seq) < 0 ? 0u : 1u;
        }
        return r0 ? 0u : (r1 ? 1u : kNone);
    }

    FORCE_INLINE void rebalance() {
        const uint32_t occ = occupancy();
        level = occ >= band_inline ? kLevelInline : occ >= band_every_sweep ? kLevelEverySweep : kLevelIdle;
    }

    FORCE_INLINE void append(uint32_t src, uint32_t len) {
        while (len != 0) {
            const uint32_t piece = len > spool_bytes - wr_off ? spool_bytes - wr_off : len;
            experimental::dma_async_write<false>(kDmaShip, src, spool_base + wr_off, piece);
            dma_issued++;
            wr += piece;
            wr_off += piece;
            if (wr_off == spool_bytes) {
                wr_off = 0;
            }
            src += piece;
            len -= piece;
        }
    }

    FORCE_INLINE void reclaim_shipped(bool& did) {
        if (b[0].state == kShipping || b[1].state == kShipping) {
            const uint32_t acked = NOC_STATUS_READ_REG(kEgressNoc, NIU_MST_WR_ACK_RECEIVED);
            for (uint32_t i = 0; i < 2; i++) {
                if (b[i].state == kShipping && static_cast<int32_t>(acked - b[i].ack_target) >= 0) {
                    b[i].state = kEmpty;
                    did = true;
                }
            }
        }
    }

    FORCE_INLINE void retire_reads(bool& did) {
        if (b[0].state == kReading || b[1].state == kReading) {
            const uint32_t rd_out = experimental::dma_get_reads_outstanding(kDmaDrain);
            for (uint32_t i = 0; i < 2; i++) {
                if (b[i].state == kReading && rd_out <= dma_rd_issued - b[i].rd_mark) {
                    b[i].state = kReady;
                    if (b[i].rd_end > rd) {
                        rd = b[i].rd_end;
                    }
                    did = true;
                }
            }
        }
    }

    FORCE_INLINE void refill(bool& did) {
        const uint32_t emp = first_empty();
        const bool want_refill = emp != kNone && (level >= kLevelInline || b[emp ^ 1u].state != kReading);
        if (want_refill && done != wr && static_cast<uint32_t>(done - rd_iss) < kBounceBytes &&
            experimental::dma_get_writes_outstanding(kDmaShip) == 0) {
            done = wr;
        }
        if (want_refill && done != rd_iss) {
            uint32_t len = static_cast<uint32_t>(done - rd_iss);
            if (len > kBounceBytes) {
                len = kBounceBytes;
            }
            if (len > spool_bytes - rd_iss_off) {
                len = spool_bytes - rd_iss_off;
            }
            experimental::dma_async_read<false>(
                kDmaDrain, spool_base + rd_iss_off, kBounceBase0 + emp * kBounceBytes, len);
            rd_iss += len;
            rd_iss_off += len;
            if (rd_iss_off == spool_bytes) {
                rd_iss_off = 0;
            }
            b[emp].rd_mark = ++dma_rd_issued;
            b[emp].rd_end = rd_iss;
            b[emp].bytes = len;
            b[emp].off = 0;
            b[emp].seq = chunks++;
            b[emp].state = kReading;
            did = true;
        }
    }

    FORCE_INLINE void ship(bool& did) {
        const uint32_t rdy = oldest_ready();
        if (rdy != kNone) {
            uint32_t nb = b[rdy].bytes - b[rdy].off;
            uint32_t bytes_free = sender.downstream_fifo_total_size - (sender.bytes_sent - acked_seen);
            if (bytes_free < nb) {
                invalidate_l1_cache();
                acked_seen = *acked_;
                bytes_free = sender.downstream_fifo_total_size - (sender.bytes_sent - acked_seen);
                if (bytes_free < nb) {
                    nb = bytes_free & ~(kPageBytes - 1u);
                }
            }
            if (nb != 0) {
                if (sender.bytes_sent - notified + nb > kernel_profiler::SPSC_NOTIFY_CAP_BYTES) {
                    if (!ncrisc_noc_nonposted_writes_flushed(kEgressNoc)) {
                        return;
                    }
                    notify_now();
                }
                push_fifo(sender, kBounceBase0 + rdy * kBounceBytes + b[rdy].off, sender.write_ptr, nb);
                push_pages(sender, nb / kPageBytes);
                b[rdy].off += nb;
                if (b[rdy].off == b[rdy].bytes) {
                    b[rdy].state = kShipping;
                    b[rdy].off = 0;
                    b[rdy].ack_target = noc_nonposted_writes_acked[kEgressNoc];
                }
                did = true;
            }
        }
    }

    FORCE_INLINE void pass() {
        if (wr == rd && both_empty()) {
            return;
        }
        bool did = false;
        reclaim_shipped(did);
        retire_reads(did);
        refill(did);
        ship(did);
        if (did) {
            rebalance();
        }
    }

    __attribute__((noinline)) void pass_cold() { pass(); }

    // A pass with no refill: what a hold runs before handing the seat back, until both bounces are empty.
    __attribute__((noinline)) void flush_pass() {
        bool did = false;
        reclaim_shipped(did);
        retire_reads(did);
        ship(did);
        if (did) {
            rebalance();
        }
    }

    __attribute__((noinline)) void notify_now() {
        notify_bytes_sent(sender);
        notified = sender.bytes_sent;
    }
    FORCE_INLINE void notify() {
        if (sender.bytes_sent != notified) {
            notify_now();
        }
    }
    FORCE_INLINE void notify_if_cap(uint32_t nb) {
        if (sender.bytes_sent - notified + nb > kernel_profiler::SPSC_NOTIFY_CAP_BYTES) {
            notify_now();
        }
    }
};

SpoolPump pump;

// ---- Gather, from drisc_relay.cpp ----
struct GatherRegs {
    volatile uint32_t* base;
    uint32_t send;
};
FORCE_INLINE GatherRegs gather_regs() {
    volatile uint32_t* base =
        reinterpret_cast<volatile uint32_t*>(NOC_REGS_START_ADDR + NOC_CMD_BUF_INSTANCE_OFFSET(kReadNoc, read_cmd_buf));
    uint32_t send = NOC_CTRL_SEND_REQ;
    asm("" : "+r"(base), "+r"(send));
    return {base, send};
}
FORCE_INLINE void gather_reg(const GatherRegs& g, uint32_t reg, uint32_t val) {
    g.base[(reg - NOC_REGS_START_ADDR) / 4u] = val;
}
FORCE_INLINE bool gather_ready(const GatherRegs& g) {
    return g.base[(NOC_CMD_CTRL - NOC_REGS_START_ADDR) / 4u] == NOC_CTRL_STATUS_READY;
}
FORCE_INLINE void gather_read(const GatherRegs& g, bool poll, uint32_t src, uint32_t dst, uint32_t len_bytes) {
    if (poll) {
        while (!gather_ready(g)) {
        }
    }
    gather_reg(g, NOC_RET_ADDR_LO, dst);
    gather_reg(g, NOC_TARG_ADDR_LO, src);
    gather_reg(g, NOC_AT_LEN_BE, len_bytes);
    gather_reg(g, NOC_CMD_CTRL, g.send);
}
FORCE_INLINE uint32_t heads(uint32_t c) { return record(c) + kHeadWord * 4u; }

FORCE_INLINE void cv_issue(uint32_t c) {
    while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_COORDINATE, core_coord[c]);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_RET_ADDR_LO, record(c));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}
FORCE_INLINE void cv_issue(uint32_t lo, uint32_t hi) {
    for (uint32_t i = lo; i < hi; i++) {
        cv_issue(i);
    }
}
FORCE_INLINE void txn_wait(uint8_t noc, uint32_t id) {
    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(id)) != 0) {
    }
}
FORCE_INLINE void cv_wait(uint32_t id) {
    txn_wait(kReadNoc, id);
    invalidate_l1_cache();
}
FORCE_INLINE void post_heads(uint32_t c) {
    noc_wwrite_with_state<DM_DEDICATED_NOC, kHeadCmdBuf, CQ_NOC_SNdl, CQ_NOC_SEND, CQ_NOC_WAIT, true, true>(
        kReadNoc, heads(c), core_coord[c], 0);
}
FORCE_INLINE uint32_t frame_bytes(uint32_t slot) {
    return (reinterpret_cast<const tt_l1_ptr uint32_t*>(slot)[kernel_profiler::SPSC_PREFIX_PAYLOAD_WORDS] +
            kPrefixWords) *
           4u;
}

// One loopback read of each frame's record into its control block. On the NoC0 seat it goes on the gather buffer,
// whose coordinate and length the gathers set per read anyway.
FORCE_INLINE void self_read_batch(const uint8_t* cores, uint32_t n, uint32_t slot) {
    for (uint32_t i = 0; i < n; i++) {
        while (!noc_cmd_buf_ready(kReadNoc, kSelfCmdBuf)) {
        }
        if constexpr (kSingleNoc) {
            NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_COORDINATE, self_coord);
            NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_AT_LEN_BE, kCvReadBytes);
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_LO, record(cores[i]));
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_LO, slot + kPrefixWords * 4u);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        slot += kSlotBytes;
    }
}

__attribute__((noinline)) uint32_t issue_batch(const uint8_t* cores, uint32_t n, uint32_t slot, uint32_t rb) {
    uint32_t min_peak = ~0u;
    const uint8_t* end = cores + n;
    const GatherRegs g = gather_regs();
    do {
        const uint32_t c = *cores++;
        volatile tt_l1_ptr uint32_t* __restrict rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c));
        volatile tt_l1_ptr uint32_t* __restrict frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        uint32_t off = kPrefixWords + kWireControlWords;
        uint32_t peak = 0;
        while (!gather_ready(g)) {
        }
        gather_reg(g, NOC_TARG_ADDR_COORDINATE, core_coord[c]);
#pragma GCC unroll 5
        for (uint32_t r = 0; r < kNumRisc; r++) {
            const uint32_t tail = rec[kTailWord + r];
            const uint32_t start = rec[kHeadWord + r];
            const uint32_t take = tail - start;
            rec[kHeadWord + r] = start + take;
            if (take > peak) {
                peak = take;
            }
            frame[kernel_profiler::SPSC_PREFIX_HEAD_0 + r] = start;
            if (take == 0) {
                continue;
            }
            const uint32_t ring_src = rb + r * (kRingWords * 4u);
            const bool poll = r != 0;
            off = place_run(
                start,
                take,
                off,
                slot,
                [&](uint32_t src, uint32_t dst, uint32_t bytes, bool first_piece) __attribute__((always_inline)) {
                    gather_read(g, first_piece ? poll : true, ring_src + src, dst, bytes);
                });
        }
        frame[kernel_profiler::SPSC_PREFIX_XY] = rec[kXyWord];
        frame[kernel_profiler::SPSC_PREFIX_PAYLOAD_WORDS] = off - kPrefixWords;
        rec[kPeakWord] = peak;
        if (peak < min_peak) {
            min_peak = peak;
        }
        slot += kSlotBytes;
    } while (cores != end);
    return min_peak;
}

// At the start of every hold: the service used every command buffer on its NoC, and on the NoC1 seat noc_init also
// reset NoC0's.
static void program_command_buffers(uint32_t cv_src) {
    while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kGatherTxn));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_TARG_ADDR_MID, 0);
    self_coord = NOC_CMD_BUF_READ_REG(kReadNoc, read_cmd_buf, NOC_RET_ADDR_COORDINATE);
    while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
    }
    noc_read_init_state<kCvCmdBuf>(kReadNoc);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_PACKET_TAG, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_LO, cv_src + kCvReadSrcOff);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_RET_ADDR_COORDINATE, self_coord);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_AT_LEN_BE, kCvReadBytes);
    if constexpr (!kSingleNoc) {
        while (!noc_cmd_buf_ready(kReadNoc, kSelfCmdBuf)) {
        }
        noc_read_init_state<kSelfCmdBuf>(kReadNoc);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kGatherTxn));
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_COORDINATE, self_coord);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_COORDINATE, self_coord);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_AT_LEN_BE, kCvReadBytes);
    }
    while (!noc_cmd_buf_ready(kReadNoc, kHeadCmdBuf)) {
    }
    noc_write_init_state<kHeadCmdBuf, CQ_NOC_mkP>(kReadNoc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_PACKET_TAG, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_LO, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_AT_LEN_BE, kNumRisc * 4u);
    while (!noc_cmd_buf_ready(kEgressNoc, write_cmd_buf)) {
    }
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(kEgressNoc, write_vc);
    NOC_CMD_BUF_WRITE_REG(kEgressNoc, write_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kEgressTxn));
}

// The staging is the arena's between holds, so its prefixes are written again at every hold.
static void zero_stage_prefixes() {
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefixWords + kWireControlWords; k++) {
            pfx[k] = 0;
        }
    }
}

// ---- Core lists, from drisc_relay.cpp ----
uint8_t ship_list[kMaxCores];
uint8_t probe_list[kMaxCores];
uint8_t demote_pos[kMaxCores];
uint32_t tails_seen[kMaxCores];
uint32_t deferred[kMaxCores];
uint32_t n_list;
uint32_t n_probe;
uint32_t n_demote;
bool sweep_live;

__attribute__((noinline)) static void note_demotions(
    const uint8_t* cores, uint32_t n, uint32_t cur, uint32_t demote_below) {
    for (uint32_t i = 0; i < n; i++) {
        if (reinterpret_cast<const tt_l1_ptr uint32_t*>(record(cores[i]))[kPeakWord] < demote_below) {
            demote_pos[n_demote++] = static_cast<uint8_t>(cur + i);
        }
    }
}

__attribute__((noinline)) static void promote_probes(bool defer_ok) {
    for (uint32_t k = 0; k < n_probe;) {
        const uint32_t c = probe_list[k];
        const tt_l1_ptr uint32_t* __restrict rec = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c));
        const tt_l1_ptr uint32_t* __restrict tails = rec + kTailWord;
        const tt_l1_ptr uint32_t* __restrict mine = rec + kHeadWord;
        const uint32_t d0 = tails[0] - mine[0];
        const uint32_t d1 = tails[1] - mine[1];
        const uint32_t d2 = tails[2] - mine[2];
        const uint32_t d3 = tails[3] - mine[3];
        const uint32_t d4 = tails[4] - mine[4];
        const uint32_t live = d0 | d1 | d2 | d3 | d4;
        const uint32_t tsum = tails[0] + tails[1] + tails[2] + tails[3] + tails[4];
        const uint32_t grew = tsum - tails_seen[c];
        tails_seen[c] = tsum;
        uint32_t peak = d0;
        if (d1 > peak) {
            peak = d1;
        }
        if (d2 > peak) {
            peak = d2;
        }
        if (d3 > peak) {
            peak = d3;
        }
        if (d4 > peak) {
            peak = d4;
        }
        sweep_live |= live != 0;
        const bool below_gate = peak < kLaneShipWords && grew < kLaneShipWords;
        const bool defer = defer_ok && below_gate && deferred[c] < kMaxDeferSweeps;
        if (live != 0 && !defer) {
            deferred[c] = 0;
            ship_list[n_list++] = static_cast<uint8_t>(c);
            probe_list[k] = probe_list[--n_probe];
        } else {
            deferred[c] += below_gate ? 1u : 0u;
            k++;
        }
    }
}

__attribute__((noinline)) static void demote() {
    do {
        const uint32_t pos = demote_pos[--n_demote];
        const uint32_t c = ship_list[pos];
        ship_list[pos] = ship_list[--n_list];
        probe_list[n_probe++] = static_cast<uint8_t>(c);
        deferred[c] = 0;
        const tt_l1_ptr uint32_t* tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c)) + kTailWord;
        tails_seen[c] = tails[0] + tails[1] + tails[2] + tails[3] + tails[4];
    } while (n_demote != 0);
}

// ---- Sweep state carried between holds ----
uint32_t cv_src;
uint32_t sweeps;
uint32_t relieved;
uint32_t stop_sweeps;
uint32_t gen_shipped;  // bit g: generation g's last frame may still be leaving staging
uint32_t gen_dma_mark[kNGens];
bool seeded;

enum State : uint32_t { kIdle = 0, kRunning = 1, kFinished = 2 };
uint32_t state;  // kIdle until start()

// The owner's exit mailbox and the word the KMD writes there when the owner's process ends. Every wait on the host
// checks it, so a dead profiler cannot hold the seat: the job then drops what it owes the host instead of waiting.
volatile uint32_t* owner_exit;
uint32_t owner_exit_word;
FORCE_INLINE bool owner_gone() { return *owner_exit == owner_exit_word; }

// The service's heartbeat, which its slot scans bump and a joining client reads twice abi::ALIVE_PROBE_MS apart to
// tell a running kernel from a hung one. A hold keeps the service from scanning, and one that waits on the host can
// outlast the probe, so the job bumps it once per sweep and on every pass of a wait on the host.
volatile uint32_t* service_heartbeat;
FORCE_INLINE void beat() { *service_heartbeat = *service_heartbeat + 1; }
FORCE_INLINE void drop_ready_bounces() {
    for (uint32_t i = 0; i < 2; i++) {
        if (pump.b[i].state == SpoolPump::kReady) {
            pump.b[i].state = SpoolPump::kEmpty;
        }
    }
}

FORCE_INLINE void retire_gen(uint32_t g) {
    if ((gen_shipped >> g) & 1u) {
        const uint32_t since = pump.dma_issued - gen_dma_mark[g];
        const uint32_t cap = since > experimental::kMaxOutstandingWrites ? experimental::kMaxOutstandingWrites : since;
        while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
        }
        gen_shipped &= ~(1u << g);
    }
}

static FORCE_INLINE void emit_slots(uint32_t base, uint32_t count) {
    const uint32_t raw0 = frame_bytes(base);
    const uint32_t len0 = page_round(raw0);
    uint32_t len1 = 0;
    if (count == kGenSlots) {
        len1 = page_round(frame_bytes(base + kSlotBytes));
    }
    const uint32_t bytes = len0 + len1;
    while (!pump.has_room(bytes)) {
        if (owner_gone()) {
            return;
        }
        beat();
        pump.pass_cold();
        pump.notify();
    }
    staged_store_fence();
    if (len1 != 0 && len0 != kSlotBytes) {
        pump.append(base, len0);
        pump.append(base + kSlotBytes, len1);
    } else {
        pump.append(base, bytes);
    }
    pump.rebalance();
}

FORCE_INLINE void refresh_first_batch() {
    const uint32_t nn = n_list < kGenSlots ? n_list : kGenSlots;
    for (uint32_t i = 0; i < nn; i++) {
        cv_issue(ship_list[i]);
    }
}

// Heads seed from the current tails: everything published before the job started predates the capture.
static void seed_heads() {
    cv_issue(0, num_cores);
    cv_wait(0);
    volatile abi::DrainJobBlock* job = job_block();
    for (uint32_t c = 0; c < num_cores; c++) {
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c));
        uint32_t tsum = 0;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            rec[kHeadWord + r] = rec[kTailWord + r];
            tsum += rec[kTailWord + r];
        }
        rec[kXyWord] = job->core_xy[c];
        tails_seen[c] = tsum;
    }
}

// One sweep of the relay's loop, without its idle back-off: the service paces the holds.
__attribute__((noinline)) static void sweep() {
    sweeps++;
    ctrl()->heartbeat = sweeps;
    beat();
    const uint32_t relieved_at_sweep_start = relieved;
    sweep_live = false;
    n_demote = 0;
    uint32_t gen = 0;
    uint32_t pend_n = 0;
    constexpr uint32_t kDeferBelow = kLaneShipWords > 1 ? kLaneShipWords : 1u;
    const bool defer_ok = n_list != 0 && stop_sweeps == 0;
    const uint32_t demote_below = defer_ok ? kDeferBelow : 1u;

    bool probe_pending = n_probe != 0;
    if (probe_pending) {
        while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kProbeTxn));
        for (uint32_t i = 0; i < n_probe; i++) {
            cv_issue(probe_list[i]);
        }
        while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_PACKET_TAG, 0);
    }

    uint32_t cur = 0;
    uint32_t n = 0;
    while (true) {
        const uint32_t n_end = n_list;
        while (true) {
            const bool more = cur < n_end;
            uint32_t defer_ri = 0;
            uint32_t defer_nn = 0;
            if (more) {
                cv_wait(0);
                retire_gen(gen);
                n = (n_end - cur) < kGenSlots ? (n_end - cur) : kGenSlots;
                const uint32_t pk = issue_batch(&ship_list[cur], n, kGenBase[gen], ring_base);
                if (pk < demote_below) {
                    note_demotions(&ship_list[cur], n, cur, demote_below);
                }
                cur += n;
                uint32_t nn = n_end - cur;
                uint32_t ri = cur;
                if (nn > kGenSlots) {
                    nn = kGenSlots;
                } else if (nn == 0) {
                    nn = n_end < kGenSlots ? n_end : kGenSlots;
                    ri = 0;
                }
                if (ri < cur && ri + nn > cur - n) {
                    defer_ri = ri;
                    defer_nn = nn;
                } else {
                    for (uint32_t i = 0; i < nn; i++) {
                        cv_issue(ship_list[ri + i]);
                    }
                }
            }

            if (pend_n != 0) {
                const uint32_t pend_gen = gen ^ 1u;
                emit_slots(kGenBase[pend_gen], pend_n);
                gen_dma_mark[pend_gen] = pump.dma_issued;
                gen_shipped |= 1u << pend_gen;
                pend_n = 0;
            }
            if (!more) {
                break;
            }
            if (pump.level >= SpoolPump::kLevelInline) {
                pump.pass();
            }

            self_read_batch(&ship_list[cur - n], n, kGenBase[gen]);
            while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_REQS_OUTSTANDING_ID(kGatherTxn)) != 0) {
                if (pump.level >= SpoolPump::kLevelInline) {
                    pump.pass();
                }
            }
            for (uint32_t i = 0; i < defer_nn; i++) {
                cv_issue(ship_list[defer_ri + i]);
            }
            for (uint32_t i = 0; i < n; i++) {
                post_heads(ship_list[cur - n + i]);
                relieved++;
            }

            pend_n = n;
            gen ^= 1u;
        }
        if (!probe_pending) {
            break;
        }
        probe_pending = false;
        retire_gen(gen);
        cv_wait(kProbeTxn);
        promote_probes(defer_ok);
    }
    bool refresh_first = n_demote != 0;
    if (n_demote != 0) {
        demote();
    }
    const bool cadence = (sweeps & (kIdlePumpStride - 1u)) == 0;
    if (pump.level >= SpoolPump::kLevelEverySweep || cadence || relieved == relieved_at_sweep_start) {
        pump.pass();
        refresh_first_batch();
        refresh_first = false;
        pump.notify();
    }
    if (refresh_first) {
        refresh_first_batch();
    }
}

// The end of every hold: nothing of the job may be in flight or in staging once the service takes the seat back.
// Unsent bounces go to the host first (a slow host holds the seat that long).
static void leave_clean() {
    retire_gen(0);
    retire_gen(1);
    while (!pump.both_empty()) {
        if (owner_gone()) {
            drop_ready_bounces();
        }
        beat();
        pump.flush_pass();
        pump.notify();
    }
    pump.notify();
    cv_wait(0);
    cv_wait(kProbeTxn);
    txn_wait(kReadNoc, kGatherTxn);
    while (!ncrisc_noc_posted_writes_sent(kReadNoc)) {
    }
    while (!ncrisc_noc_nonposted_writes_flushed(kEgressNoc)) {
    }
}

// The spool drains, then the socket barrier holds until the host has acked every byte.
static void finish() {
    retire_gen(0);
    retire_gen(1);
    while (!pump.drained() && !owner_gone()) {
        beat();
        pump.pass_cold();
        pump.notify();
    }
    pump.notify();
    ctrl()->done = kernel_profiler::kResidentAwaitingAcksWord;
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr);
    while (sender.bytes_sent != *acked && !owner_gone()) {
        beat();
        invalidate_l1_cache();
    }
    while (!ncrisc_noc_nonposted_writes_flushed(kEgressNoc)) {
    }
    while (!(ncrisc_noc_posted_writes_sent(kEgressNoc) && ncrisc_noc_posted_writes_sent(kReadNoc))) {
    }
    update_socket_config(sender);
    ctrl()->done = kernel_profiler::kResidentDoneWord;
    state = kFinished;
}

// ---- The service's interface ----

// Reads the job block; the service calls it once the owner word names an attached client, passing that client's
// exit mailbox, the word that marks its exit, and the service heartbeat the job bumps during holds.
inline void start(volatile uint32_t* exit_mailbox, uint32_t exit_word, volatile uint32_t* heartbeat) {
    owner_exit = exit_mailbox;
    owner_exit_word = exit_word;
    service_heartbeat = heartbeat;
    volatile abi::DrainJobBlock* job = job_block();
    num_cores = job->num_cores < kMaxCores ? job->num_cores : kMaxCores;
    cv_src = job->cv_src;
    write_vc = job->write_vc;
    spool_base = job->spool_base;
    spool_bytes = job->spool_bytes;
    hold_turns = job->hold_turns != 0 ? job->hold_turns : 1;
    hold_sweeps = job->hold_sweeps != 0 ? job->hold_sweeps : 1;
    ring_base = cv_src + kCtrlWords * 4u;
    for (uint32_t i = 0; i < num_cores; i++) {
        const auto xy = kernel_profiler::word_as<kernel_profiler::NocXy>(job->core_xy[i]);
        core_coord[i] = static_cast<uint32_t>(NOC_XY_ADDR(xy.x, xy.y, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    }
    sender = create_sender(kSocketConfigAddr);
    set_page_size(sender, kPageBytes);
    pump.init();
    n_list = 0;
    n_probe = num_cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        probe_list[i] = static_cast<uint8_t>(i);
        deferred[i] = 0;
    }
    sweeps = 0;
    relieved = 0;
    stop_sweeps = 0;
    gen_shipped = 0;
    seeded = false;
    state = kRunning;
}

inline bool running() { return state == kRunning; }

// One hold: hold_sweeps sweeps, or the stop sweeps and the finish once the host asked to stop.
inline void hold() {
    noc_local_state_init(kEgressNoc);
    if constexpr (!kSingleNoc) {
        noc_local_state_init(kReadNoc);
    }
    program_command_buffers(cv_src);
    zero_stage_prefixes();
    if (!seeded) {
        seed_heads();
        seeded = true;
    } else {
        refresh_first_batch();
    }
    for (uint32_t m = 0; m < hold_sweeps; m++) {
        invalidate_l1_cache();
        if (ctrl()->stop != 0) {
            if (stop_sweeps == 2) {
                finish();
                break;
            }
            stop_sweeps++;
        }
        sweep();
    }
    leave_clean();
}

// The owner detached without stopping the job. Holds end with nothing of the job in flight (leave_clean), and the
// service calls this between holds, so there is nothing left to wait for.
inline void abort() { state = kIdle; }

}  // namespace drain
