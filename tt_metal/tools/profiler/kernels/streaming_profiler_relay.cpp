// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's relay: resident on a DRAM bank's spare DRISC, it polls its slice of the worker
// SPSC rings, gathers live runs into wire frames, spools them in its own GDDR bank and pumps them to the
// host FIFO over a D2H socket. Producers block on a full ring, so the pipeline is lossless end to end.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

// Caller runs noc_write_init_state<write_cmd_buf> once per push; nothing between the push's calls invalidates it.
inline void write_to_host_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    while (size) {
        const uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

// socket_push_pages only wraps the pointer, so a piece crossing the FIFO wrap splits here; fifo_size is whole
// pages, so the pads' NoC congruence survives the split.
inline void push_fifo(const SocketSenderInterface& sender, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = sender.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host_chunked(sender.d2h.pcie_xy_enc, src, base + dst, first);
    if (first < len) {
        write_to_host_chunked(sender.d2h.pcie_xy_enc, src + first, base, len - first);
    }
}

// Not socket_notify_receiver: it re-inits write_cmd_buf onto another VC, and the bytes_sent word can then
// overtake the data it announces.
inline void notify_bytes_sent(const SocketSenderInterface& sender) {
    volatile tt_l1_ptr sender_socket_md* cfg =
        reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr);
    cfg->bytes_sent = sender.bytes_sent;
    asm volatile("fence" ::: "memory");
    write_to_host_chunked(
        sender.d2h.pcie_xy_enc,
        sender.config_addr,
        (static_cast<uint64_t>(sender.d2h.bytes_sent_addr_hi) << 32) | sender.downstream_bytes_sent_addr,
        4u);
}

constexpr uint32_t kStageBase = get_named_compile_time_arg_val("stage_base");
constexpr uint32_t kNStage = get_named_compile_time_arg_val("n_stage");
constexpr uint32_t kCoreRecords = get_named_compile_time_arg_val("core_records");
constexpr uint32_t kDoneAddr = get_named_compile_time_arg_val("done_addr");
// 1 = quiesce (every wait holds), 2 = kill switch (abandon waits, free the NIU).
constexpr uint32_t kStopAddr = get_named_compile_time_arg_val("stop_addr");
constexpr uint32_t kSocketConfigAddr = get_named_compile_time_arg_val("socket_config_addr");
constexpr uint32_t kMaxCores = get_named_compile_time_arg_val("max_cores");
static_assert(kMaxCores <= 256, "ship_list and hot index cores as bytes");
// Static VC for PCIe pushes, spread across relays by the host.
constexpr uint32_t kWriteVc = get_named_compile_time_arg_val("write_vc");
// Per lane, not per span: the producer that blocks is always one lane.
constexpr uint32_t kShipMinPct = get_named_compile_time_arg_val("ship_min_pct");
// 0 selects direct push: frames go straight from staging to the host FIFO.
constexpr uint32_t kSpoolBase = get_named_compile_time_arg_val("spool_base");
constexpr uint32_t kSpoolBytes = get_named_compile_time_arg_val("spool_bytes");

constexpr uint32_t kNumRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
static_assert(kNumRisc == 5, "the control scans are unrolled for exactly five RISCs");
constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;
constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
// Slots hold a full span: a sub-span cap defers whole lanes at speed and starves their producers.
constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
constexpr uint32_t kSlotBytes = kSlotWords * 4u;
constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefix - kWireCtrl;
// The lane walk has no room gate: a frame of five full rings and their pads always fits the slot.
static_assert(
    kNumRisc * (kRingWords + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1u) <= kPayloadCapWords,
    "a full span no longer fits a slot");
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
// Reads take the NoC the writes do not: NOC_INDEX carries egress, the other NoC carries gathers.
constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
constexpr bool kSpool = kSpoolBytes != 0;
constexpr uint8_t kDmaShip = 0;   // TX stream 0: staging -> spool
constexpr uint8_t kDmaDrain = 1;  // TX stream 1: spool -> bounce
// The TX stream status register's num_writes_outstanding field is 4 bits wide (gddr_dma_regs.h).
constexpr uint32_t kDmaOutstandingMax = 15;
// Two-core batches in kNGens generations; in spool mode the leftover slots are the two drain bounces.
constexpr uint32_t kGenSlots = 2;
static_assert(kGenSlots == 2, "the frame emit is written for two-slot generations");
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kNGens = (kNStage - kNBounce) / kGenSlots;
static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
// 64 B per core: the tail read at +0, the head mirror at +32, the wire XY word behind the heads so the head
// write is 20 bytes.
constexpr uint32_t kCvReadBytes = 32;
constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
constexpr uint32_t kRecordBytes = 64;
constexpr uint32_t kHeadWord = kCvReadBytes / 4u;
constexpr uint32_t kXyWord = kHeadWord + kNumRisc;
static_assert((kXyWord + 1u) * 4u <= kRecordBytes, "the core record overflows its 64 bytes");
constexpr uint32_t kBounceBase0 = kStageBase + kNGens * kGenSlots * kSlotBytes;
constexpr uint32_t kBounceBytes = ((kNStage - kNGens * kGenSlots) * kSlotBytes / 2u) & ~(kPageBytes - 1u);
static_assert(kBounceBase0 % kPageBytes == 0, "bounces start on a page");
static_assert(
    !kSpool || kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
    "bounces must fit inside the mapped staging arena");
static_assert(!kSpool || kSpoolBytes % kPageBytes == 0, "spool wraps on pages");
constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
// A head only reaches a producer on a ship, so idle backoff must stop growing before lanes reach the trigger.
constexpr uint32_t kLaneTrigger = kRingWords / 2u;
constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
constexpr uint64_t kCyclesPerUs = 1350;  // DRISC wall clock at the 1.35 GHz AICLK
// Idle backoff ceiling. 20 us exceeded a lane's fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 5 * kCyclesPerUs;
constexpr uint32_t kCvIdleGapMinInc = 256;
// Worst-case host staleness for a workload too light to reach the occupancy bands.
constexpr uint64_t kSpoolFreshCycles = 50'000 * kCyclesPerUs;
constexpr uint64_t kStopDrainCycles = 1'000'000 * kCyclesPerUs;
// How long the exit lets the posted head writes stream out; small packets leave in nanoseconds.
constexpr uint64_t kPostedDrainCycles = 1000 * kCyclesPerUs;
// How long the exit waits for the host's NIU-restore word before restoring anyway.
constexpr uint64_t kNiuRestoreWaitCycles = 10'000'000 * kCyclesPerUs;

static_assert(kSpanWords * 4u <= NOC_MAX_BURST_SIZE, "a span read must fit one NoC burst");
static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a whole-ring gather must fit one NoC burst");
static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
// Pads bring each run to its ring phase, and slot base, payload base and wrap continuations land congruent,
// so one pad rule serves both the gather read and the PCIe write.
static_assert(
    kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_PCIE_WRITE_ALIGNMENT_BYTES &&
        kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_L1_READ_ALIGNMENT_BYTES,
    "the shared pad rule no longer matches this part's NoC congruence");
static_assert(
    kRingWords % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        (kPrefix + kWireCtrl) % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        kStageBase % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0 &&
        kSlotBytes % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0,
    "packed-gather congruence broken");

// pass() never blocks: every wait is a state a later pass observes, so the pump can delay host delivery but
// never the sweep. kSpoolBytes == 0 is the direct-push build, which never calls this.
struct SpoolPump {
    enum : uint32_t { kEmpty = 0, kReading = 1, kReady = 2, kShipping = 3 };
    static constexpr uint32_t kNone = 2;  // no bounce in the asked-for state
    // Pump effort by spool occupancy: idle sweeps only, every other sweep, every sweep, also per batch and
    // inside the read-wait spin.
    enum : uint32_t { kLevelIdle = 0, kLevelHalf = 1, kLevelEverySweep = 2, kLevelInline = 3 };
    static constexpr uint32_t kBandHalf = kSpoolBytes / 4u + kSpoolBytes / 8u;
    static constexpr uint32_t kBandEverySweep = kSpoolBytes / 2u;
    static constexpr uint32_t kBandInline = kSpoolBytes / 2u + kSpoolBytes / 8u;
    static constexpr uint32_t kFreshTickStride = 64;
    // At most one READING and one SHIPPING bounce at a time, so every pass is a poll.
    struct Bounce {
        uint32_t state;
        uint32_t bytes;       // spool bytes held
        uint32_t off;         // bytes already pushed to the host
        uint32_t ack_target;  // write-ack mirror at ship: this bounce's flush line
        uint32_t seq;         // refill order; a both-ready pass ships the older bytes first
        uint32_t rd_mark;     // drain-stream issue count at refill: this bounce's completion line
        uint64_t rd_end;      // what rd advances to when this bounce turns READY
    };

    uint64_t wr = 0;          // bytes appended by the ship DMA
    uint64_t done = 0;        // bytes whose ship writes completed (safe for the drain stream to read)
    uint64_t rd_iss = 0;      // bytes a bounce refill has been issued for
    uint64_t rd = 0;          // bytes whose refill reads completed (safe to overwrite)
    uint32_t wr_off = 0;      // wr % kSpoolBytes
    uint32_t rd_iss_off = 0;  // rd_iss % kSpoolBytes
    uint32_t dma_issued = 0;  // cumulative ship-stream writes: the caller's per-generation completion gate
    uint32_t dma_rd_issued = 0;
    uint32_t chunks = 0;  // refills so far; also the sequence number the oldest-first ship compares
    Bounce b[2] = {};
    bool notify_pending = false;  // pump ships owe the host a bytes_sent notify (batched per sweep)
    uint32_t level = kLevelIdle;
    // Latched trickle for a workload too light to reach the bands; cleared when the spool empties.
    bool fresh_boost = false;
    uint64_t oldest = 0;  // when the spool last went non-empty; 0 = empty
    uint32_t fresh_tick = 0;
    SocketSenderInterface& sender_;
    volatile tt_l1_ptr uint32_t* acked_;  // the downstream's bytes_acked word

    explicit SpoolPump(SocketSenderInterface& sender) :
        sender_(sender), acked_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr)) {}

    uint32_t occupancy() const { return static_cast<uint32_t>(wr - rd); }
    bool has_room(uint32_t bytes) const { return kSpoolBytes - occupancy() >= bytes; }
    bool both_empty() const { return b[0].state == kEmpty && b[1].state == kEmpty; }
    bool drained() const { return rd_iss == wr && both_empty(); }
    uint32_t first_empty() const { return b[0].state == kEmpty ? 0u : (b[1].state == kEmpty ? 1u : kNone); }
    // Oldest bounce first: the socket is a byte stream.
    uint32_t oldest_ready() const {
        const bool r0 = b[0].state == kReady;
        const bool r1 = b[1].state == kReady;
        if (r0 && r1) {
            return static_cast<int32_t>(b[0].seq - b[1].seq) < 0 ? 0u : 1u;
        }
        return r0 ? 0u : (r1 ? 1u : kNone);
    }

    // Call wherever wr or rd advance.
    __attribute__((always_inline)) void rebalance() {
        const uint32_t occ = occupancy();
        level = occ >= kBandInline       ? kLevelInline
                : occ >= kBandEverySweep ? kLevelEverySweep
                : occ >= kBandHalf       ? kLevelHalf
                                         : kLevelIdle;
    }

    __attribute__((always_inline)) void append(uint32_t src, uint32_t len) {
        while (len != 0) {
            const uint32_t piece = len > kSpoolBytes - wr_off ? kSpoolBytes - wr_off : len;
            experimental::dma_async_write<false>(kDmaShip, src, kSpoolBase + wr_off, piece);
            dma_issued++;
            wr += piece;
            wr_off += piece;
            if (wr_off == kSpoolBytes) {
                wr_off = 0;
            }
            src += piece;
            len -= piece;
        }
    }

    // SHIPPING -> EMPTY on write-ack, not on sent: the bounce's next writer is the DMA engine, which a
    // sent-only gate does not fence.
    __attribute__((always_inline)) void reclaim_shipped(bool& did) {
        if (b[0].state == kShipping || b[1].state == kShipping) {
            const uint32_t acked = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED);
            for (uint32_t i = 0; i < 2; i++) {
                if (b[i].state == kShipping && static_cast<int32_t>(acked - b[i].ack_target) >= 0) {
                    b[i].state = kEmpty;
                    did = true;
                }
            }
        }
    }

    // Stream completion is FIFO, so one outstanding count gives each bounce its own line.
    __attribute__((always_inline)) void retire_reads(bool& did) {
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

    // Refill before shipping so the read runs under the ship's NoC issue; a second concurrent refill only at
    // full pressure, where the deeper bank queue pays for itself.
    __attribute__((always_inline)) void refill(bool& did) {
        const uint32_t emp = first_empty();
        const bool want_refill = emp != kNone && (level >= kLevelInline || b[emp ^ 1u].state != kReading);
        // Only ship-completed bytes are readable: nothing short of a write's completion orders a read of the same
        // address behind it.
        if (want_refill && done != wr && static_cast<uint32_t>(done - rd_iss) < kBounceBytes &&
            experimental::dma_get_writes_outstanding(kDmaShip) == 0) {
            done = wr;
        }
        if (want_refill && done != rd_iss) {
            uint32_t len = static_cast<uint32_t>(done - rd_iss);
            if (len > kBounceBytes) {
                len = kBounceBytes;
            }
            if (len > kSpoolBytes - rd_iss_off) {
                len = kSpoolBytes - rd_iss_off;
            }
            experimental::dma_async_read<false>(
            kDmaDrain, kSpoolBase + rd_iss_off, kBounceBase0 + emp * kBounceBytes, len);
            rd_iss += len;
            rd_iss_off += len;
            if (rd_iss_off == kSpoolBytes) {
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

    __attribute__((always_inline)) void ship(bool& did) {
        const uint32_t rdy = oldest_ready();
        if (rdy != kNone) {
            invalidate_l1_cache();
            const uint32_t bytes_free = sender_.downstream_fifo_total_size - (sender_.bytes_sent - *acked_);
            uint32_t nb = b[rdy].bytes - b[rdy].off;
            if (bytes_free < nb) {
                nb = bytes_free & ~(kPageBytes - 1u);
            }
            if (nb != 0) {
                push_fifo(sender_, kBounceBase0 + rdy * kBounceBytes + b[rdy].off, sender_.write_ptr, nb);
                socket_push_pages(sender_, nb / kPageBytes);
                notify_pending = true;
                b[rdy].off += nb;
                if (b[rdy].off == b[rdy].bytes) {
                    b[rdy].state = kShipping;
                    b[rdy].off = 0;
                    // The ack mirror is cumulative, so earlier partial ships are covered too.
                    b[rdy].ack_target = noc_nonposted_writes_acked[NOC_INDEX];
                }
                did = true;
            }
        }
    }

    __attribute__((always_inline)) void pass() {
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

    // Out-of-line copy for the cold sites.
    __attribute__((noinline)) void pass_cold() { pass(); }

    // Once per sweep, not per chunk.
    __attribute__((always_inline)) void notify() {
        if (notify_pending) {
            notify_bytes_sent(sender_);
            notify_pending = false;
        }
    }

    // The wall clock is read every 64th call: one read per sweep stalls producers at saturation, and ~1 ms of
    // stride is noise against the 50 ms bound.
    __attribute__((always_inline)) void freshness_tick(uint64_t deadline_cycles) {
        if (++fresh_tick < kFreshTickStride) {
            return;
        }
        fresh_tick = 0;
        if (wr == rd) {
            oldest = 0;
            fresh_boost = false;
            return;
        }
        const uint64_t now = get_timestamp();
        if (oldest == 0 || level >= kLevelHalf || fresh_boost) {
            oldest = now;
            fresh_boost = level == kLevelIdle && fresh_boost;
        } else if (now - oldest > deadline_cycles) {
            fresh_boost = true;
        }
    }
};

// Tail reads and head writes each own a command buffer on the read NoC, programmed once; a per-core command
// is the coordinate, one address and the send.
constexpr uint32_t kCvCmdBuf = write_at_cmd_buf;
constexpr uint32_t kHeadCmdBuf = write_cmd_buf;

__attribute__((always_inline)) inline bool host_released(volatile tt_l1_ptr uint32_t* stop) {
    invalidate_l1_cache();
    return *stop == kernel_profiler::kRelayStopRelease;
}

__attribute__((always_inline)) inline uint32_t core_xy(uint64_t noc_addr) {
    return static_cast<uint32_t>(noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

__attribute__((always_inline)) inline uint32_t record(uint32_t c) { return kCoreRecords + c * kRecordBytes; }
__attribute__((always_inline)) inline uint32_t heads(uint32_t c) { return record(c) + kHeadWord * 4u; }

// Counted, not barriered: in-flight gather responses also bump the counter, which can only hand the scan
// stale-but-valid tails (tails are monotonic).
__attribute__((always_inline)) inline void cv_issue(const uint64_t* core_noc, uint32_t c) {
    while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_COORDINATE, core_xy(core_noc[c]));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_RET_ADDR_LO, record(c));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

__attribute__((always_inline)) inline void cv_issue(const uint64_t* core_noc, uint32_t lo, uint32_t hi) {
    for (uint32_t i = lo; i < hi; i++) {
        cv_issue(core_noc, i);
    }
}

__attribute__((always_inline)) inline void cv_wait(uint32_t rd0, uint32_t expect) {
    while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED) - rd0 < expect) {
    }
    invalidate_l1_cache();
}

// Posted (the barriers protect staging, which a head write never touches) and on the read NoC, where it does
// not queue behind frame data and the PCIe tile's acceptance jitter.
__attribute__((always_inline)) inline void post_heads(const uint64_t* core_noc, uint32_t c) {
    noc_wwrite_with_state<DM_DEDICATED_NOC, kHeadCmdBuf, CQ_NOC_SNdl, CQ_NOC_SEND, CQ_NOC_WAIT, true, true>(
        kReadNoc, heads(c), core_xy(core_noc[c]), 0);
}

// A frame occupies whole socket pages on the wire.
__attribute__((always_inline)) inline uint32_t page_round(uint32_t bytes) {
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

// Prefix word 1 is the payload length in words.
constexpr uint32_t kLenWord = 1;
__attribute__((always_inline)) inline uint32_t frame_bytes(uint32_t slot) {
    return (reinterpret_cast<const tt_l1_ptr uint32_t*>(slot)[kLenWord] + kPrefix) * 4u;
}

// Computed once: get_noc_addr's coordinate arithmetic would otherwise run at every issue of an
// instruction-bound sweep.
static uint64_t core_noc[kMaxCores];
static uint32_t ring_base;  // lane 0's ring on every worker: the control block plus its control words

// Gather-read each live run straight to its packed wire offset; the pads keep read src == dst (mod 16 B) for
// every piece, wrap continuations included. Returns the smallest per-core peak lane take.
__attribute__((noinline)) uint32_t issue_batch(const uint8_t* cores, uint32_t n, uint32_t slot, uint32_t rb) {
    uint32_t min_peak = ~0u;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = cores[i];
        const uint32_t rec = record(c);
        const tt_l1_ptr uint32_t* __restrict tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(rec);
        volatile tt_l1_ptr uint32_t* __restrict cv =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
        // The head advance hides behind the NIU's acceptance of the same lane's read; nothing reads the record
        // before the batch barrier.
        volatile tt_l1_ptr uint32_t* __restrict head =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec + kHeadWord * 4u);
        uint32_t off = kPrefix + kWireCtrl;
        uint32_t peak = 0;
        while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, core_xy(core_noc[c]));
        // A loop, not unrolled: lane r's bookkeeping hides behind lane r-1's NIU acceptance.
        for (uint32_t r = 0; r < kNumRisc; r++) {
            const uint32_t tail = tails[r];
            const uint32_t start = head[r];
            const uint32_t take = tail - start;
            head[r] = start + take;
            if (take > peak) {
                peak = take;
            }
            cv[kernel_profiler::SPSC_WIRE_HEAD_0 + r] = start;
            cv[kernel_profiler::SPSC_WIRE_TAIL_0 + r] = tail;
            if (take == 0) {
                continue;
            }
            const uint32_t ring_src = rb + r * (kRingWords * 4u);
            const uint32_t hm = start & (kRingWords - 1u);
            if (hm + take <= kRingWords) {
                off += kernel_profiler::spsc_span_pack_pad(start, off);
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                off += take;
            } else if (kernel_profiler::spsc_span_wrap_image(start, take, kRingWords)) {
                // A near-full wrapping run ships as its whole ring image in one read (the decoder linearises by head).
                // Coalescing adjacent images into one read starves the producer's L1 port ~70x.
                off += kernel_profiler::spsc_span_pack_pad(0u, off);
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src, slot + off * 4u, kRingWords * 4u);
                off += kRingWords;
            } else {
                // A small wrapping run ships as two byte-exact pieces: its dead remainder would be most of the ring.
                off += kernel_profiler::spsc_span_pack_pad(start, off);
                const uint32_t first = kRingWords - hm;
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, first * 4u);
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src, slot + (off + first) * 4u, (take - first) * 4u);
                off += take;
            }
        }
        cv[kernel_profiler::SPSC_WIRE_XY] = tails[kXyWord];
        // pfx[0] is staged once at init; only the payload word varies.
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        pfx[kLenWord] = off - kPrefix;
        if (peak < min_peak) {
            min_peak = peak;
        }
        slot += kSlotBytes;
    }
    return min_peak;
}

__attribute__((always_inline)) static inline void program_command_buffers(uint32_t cv_src) {
    // Both read buffers use transaction id 0: the NIU's outstanding count for that id is the batch barrier.
    while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_PACKET_TAG, 0);
    // Worker L1 addresses fit 32 bits, so the address-mid word is zero and set_state is the coordinate alone.
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_TARG_ADDR_MID, 0);
    while (!noc_cmd_buf_ready(kReadNoc, kCvCmdBuf)) {
    }
    noc_read_init_state<kCvCmdBuf>(kReadNoc);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_PACKET_TAG, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_LO, cv_src + kCvReadSrcOff);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(
        kReadNoc,
        kCvCmdBuf,
        NOC_RET_ADDR_COORDINATE,
        NOC_CMD_BUF_READ_REG(kReadNoc, read_cmd_buf, NOC_RET_ADDR_COORDINATE));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kCvCmdBuf, NOC_AT_LEN_BE, kCvReadBytes);
    while (!noc_cmd_buf_ready(kReadNoc, kHeadCmdBuf)) {
    }
    noc_write_init_state<kHeadCmdBuf, CQ_NOC_mkP>(kReadNoc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_LO, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_AT_LEN_BE, kNumRisc * 4u);
    // Programmed once: nothing else on this core touches write_cmd_buf on the egress NoC.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);
}

// Only heads, tails and the core identity are staged per frame; the rest must read zero on the wire.
__attribute__((always_inline)) static inline void zero_stage_prefixes() {
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix + kWireCtrl; k++) {
            pfx[k] = 0;
        }
    }
}

// Heads seed from the current tails: everything published before this launch predates the capture. The
// scratch is the only copy of the heads.
__attribute__((always_inline)) static inline void seed_heads(
    uint32_t num_cores, volatile tt_l1_ptr uint32_t* coords, uint32_t* tails_seen) {
    const uint32_t rd_seed = NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED);
    cv_issue(core_noc, 0, num_cores);
    cv_wait(rd_seed, num_cores);
    for (uint32_t c = 0; c < num_cores; c++) {
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c));
        uint32_t tsum = 0;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            rec[kHeadWord + r] = rec[r];
            tsum += rec[r];
        }
        rec[kXyWord] = coords[c];
        tails_seen[c] = tsum;
    }
}

// Drain the spool, barrier the socket, publish done, then hand the NIU back on the host's word.
__attribute__((always_inline)) static inline void finish(
    SpoolPump& pump, SocketSenderInterface& sender, volatile tt_l1_ptr uint32_t* stop, bool killed) {
    // Bounded: a consumer that stopped acking strands bytes instead of wedging teardown.
    if constexpr (kSpool) {
        while (!pump.drained()) {
            pump.pass_cold();
            // Notify per pass: with a FIFO smaller than the backlog, credit only returns after the host has seen the
            // bytes.
            pump.notify();
            // The host escalates stop to 2 after its own timeout.
            if (host_released(stop)) {
                killed = true;
                break;
            }
        }
        pump.notify();
    }

    // socket_barrier waits for the host to ack everything, so it would hang on a dead consumer.
    if (!killed) {
        socket_barrier(sender);
    }
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    // The posted head write-backs are outside that barrier's predicate; drain their sent counter too.
    const uint64_t t_ps = get_timestamp() + kPostedDrainCycles;
    while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
           get_timestamp() < t_ps) {
    }
    // Only for a live consumer: after an abandoned batch the socket's bytes_sent is already out of sync with
    // the host.
    if (!killed) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = kernel_profiler::kRelayDoneWord;

    // NIU_CFG_0 persists until chip reset, so whoever set stream mode restores it; it goes last because NOC2AXI
    // takes this L1 out of the host's view.
    const uint64_t t_end = get_timestamp() + kNiuRestoreWaitCycles;
    while (!host_released(stop) && get_timestamp() < t_end) {
    }
    experimental::drisc_set_noc2axi_mode_all();
}

void kernel_main() {
    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // profiler_msg_t base on every worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t xy = coords[i];
        core_noc[i] = get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src);
    }
    // The NoC counter mirrors persist across launches on this never-reset core; a previous run's unacked
    // writes would wedge the first barrier.
    noc_local_state_init(NOC_INDEX);
    noc_local_state_init(kReadNoc);
    ring_base = cv_src + kCtrlWords * 4u;
    program_command_buffers(cv_src);

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    // The host's launch check polls this; a DRISC that never leaves reset would otherwise wedge every producer
    // silently.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    *hb = 0;

    zero_stage_prefixes();

    // Statics persist across launches, so everything the loop trusts is re-initialised. tails_seen is the sum of
    // a core's tails at its last scan; tails are monotonic, so its delta is one interval's production.
    static uint32_t tails_seen[kMaxCores];
    static uint8_t hot[kMaxCores];        // shipped real words last scan; hot + empty scan = publish lag
    static uint8_t ship_list[kMaxCores];  // this sweep's ship set, dense core indices
    for (uint32_t i = 0; i < num_cores; i++) {
        hot[i] = 0;
    }
    seed_heads(num_cores, coords, tails_seen);

    uint32_t relieved = 0;
    uint32_t sweeps = 0;
    uint32_t gap = 0;
    // Deferral arms only after kBatchArmSweeps consecutive growing sweeps and flushes after kFlushQuietSweeps
    // dead ones: occupancy alone cannot tell pre-burst trickle from a light workload's steady lanes, and a
    // pre-loaded ring tips over during the detection latency.
    bool grid_busy = false;
    // Once a sweep shipped every core above the threshold the scan is pure overhead, so the grid ships in list
    // order off the per-batch refreshes until a core comes back empty or under threshold.
    bool all_live = false;
    uint32_t grow_streak = 0;
    uint32_t quiet_streak = 0;
    constexpr uint32_t kBatchArmSweeps = 3;
    constexpr uint32_t kFlushQuietSweeps = 8;
    // Re-posting every head on this cadence bounds a lost posted relief to a stall instead of a parked producer.
    constexpr uint32_t kHeadRefreshSweeps = 64;
    static_assert((kHeadRefreshSweeps & (kHeadRefreshSweeps - 1u)) == 0, "the refresh cadence is a mask");
    // Persists across sweeps so a sweep's final ship drains under the pace gap, not on its own critical path.
    bool gen_shipped[kNGens] = {};

    uint32_t gen_dma_mark[kNGens] = {};
    SpoolPump pump(sender);
    bool killed = false;  // the kill switch (stop=2) broke a wait: the consumer is gone, bytes are stranded

    // A staged slot is its frame's wire image, so a frame is one write or one DMA; the trailing page fill is
    // never written, the host reads past it.
    auto emit_slots = [&](uint32_t base, uint32_t count) __attribute__((always_inline)) {
        // Frames occupy whole pages on the wire, so the FIFO write pointer and the spool offset advance in lockstep.
        const uint32_t raw0 = frame_bytes(base);
        const uint32_t len0 = page_round(raw0);
        uint32_t raw1 = 0;
        uint32_t len1 = 0;
        if (count == kGenSlots) {
            raw1 = frame_bytes(base + kSlotBytes);
            len1 = page_round(raw1);
        }
        const uint32_t bytes = len0 + len1;
        if constexpr (kSpool) {
            // This wait is the spool's back-pressure: it holds through quiesce and only the kill switch breaks it.
            while (!pump.has_room(bytes)) {
                if (host_released(stop)) {
                    killed = true;
                    return;
                }
                pump.pass();
            }
            // Blackhole stores can reach SRAM out of order, and the DMA engine reads the words the scalar core staged.
            asm volatile("fence" ::: "memory");
            // A full-span frame fills its slot exactly, so it and a full neighbour are wire-contiguous and ship as one
            // DMA.
            if (len1 != 0 && len0 != kSlotBytes) {
                pump.append(base, len0);
                pump.append(base + kSlotBytes, len1);
            } else {
                pump.append(base, bytes);
            }
            pump.rebalance();
        } else {
            asm volatile("fence" ::: "memory");
            if (!socket_reserve_pages(sender, bytes / kPageBytes, stop, kernel_profiler::kRelayStopRelease)) {
                killed = true;
                return;
            }
            push_fifo(sender, base, sender.write_ptr, raw0);
            if (len1 != 0) {
                push_fifo(sender, base + kSlotBytes, sender.write_ptr + len0, raw1);
            }
            socket_push_pages(sender, bytes / kPageBytes);
            notify_bytes_sent(sender);
        }
    };

    // On stop=1, sweep until a whole sweep moves nothing, so no marker is stranded in a worker ring.
    uint64_t stop_seen_at = 0;
    uint32_t relieved_at_stop_check = 0;
    while (true) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_seen_at == 0) {
                stop_seen_at = get_timestamp();
            } else if (relieved == relieved_at_stop_check || get_timestamp() - stop_seen_at > kStopDrainCycles) {
                break;
            }
            relieved_at_stop_check = relieved;
        }
        sweeps++;
        *hb = sweeps;
        const uint32_t relieved_at_sweep_start = relieved;

        uint32_t sweep_peak = 0;
        bool sweep_grew = false;
        // Gather generation G on the read NoC while G^1 ships; tail reads issue up front and the rest of the grid is
        // scanned just in time as the ship list runs low. No lambda here: a by-reference capture costs sweep time
        // at the saturation boundary.
        uint32_t gen = 0;
        uint32_t gen_base = kStageBase;
        uint32_t pend_n = 0;
        uint32_t pend_gen = 0;
        uint32_t pend_base = kStageBase;
        bool have_pend = false;
        uint32_t n_ship = 0;
        uint32_t min_peak = ~0u;

        // Heads go out at the read barrier, not the frame emit: once the reads land the producer's ring slots are
        // free. A lambda on purpose: written inline, issue_batch picks up a spill per core.
        auto advance_heads = [&](uint32_t n, const uint8_t* cores) __attribute__((always_inline)) {
            for (uint32_t i = 0; i < n; i++) {
                post_heads(core_noc, cores[i]);
                relieved++;
            }
        };

        // Generation g's previous frame must be out of staging before its slots refill; gen_shipped persists so a
        // sweep never waits on its own last ship.
        auto retire_gen = [&](uint32_t g) __attribute__((always_inline)) {
            if (gen_shipped[g]) {
                // Both predicates complete on this device alone, so no consumer state can hang them.
                if constexpr (kSpool) {
                    // Stream completion is FIFO, so outstanding <= later-issues means this generation retired.
                    const uint32_t since = pump.dma_issued - gen_dma_mark[g];
                    const uint32_t cap = since > kDmaOutstandingMax ? kDmaOutstandingMax : since;
                    while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
                    }
                } else {
                    // Sent-only is legal: the slots' next writer is this core's own NIU read responses.
                    while (!ncrisc_noc_nonposted_writes_sent(NOC_INDEX)) {
                    }
                }
                gen_shipped[g] = false;
            }
        };

        if (all_live) {
            n_ship = num_cores;
            if constexpr (kLaneShipWords != 0) {
                sweep_grew = true;
            }
        } else {
            const uint32_t rd0 = NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED);
            // Responses can arrive out of order; a core counted early scans last sweep's tails (stale but valid),
            // under-ships, and catches up next visit.
            cv_issue(core_noc, 0, num_cores);
            // The previous sweep's last ship retires under the tail reads' round trip, not between the tails landing
            // and the first issue: the batch whose tails age most is the batch whose producers stall.
            retire_gen(gen);
            cv_wait(rd0, num_cores);
            for (uint32_t c = 0; c < num_cores; c++) {
                const tt_l1_ptr uint32_t* __restrict tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c));
                const tt_l1_ptr uint32_t* __restrict mine = tails + kHeadWord;
                // Unrolled into registers: an indexed-array loop spills, and each spilled word is an L1 round trip per
                // core per sweep.
                const uint32_t d0 = tails[0] - mine[0];
                const uint32_t d1 = tails[1] - mine[1];
                const uint32_t d2 = tails[2] - mine[2];
                const uint32_t d3 = tails[3] - mine[3];
                const uint32_t d4 = tails[4] - mine[4];
                // No clamp: a producer blocks 506 words past the head it sees, and the mirror is never behind that
                // head.
                const uint32_t live = d0 | d1 | d2 | d3 | d4;
                uint32_t grew = 0;
                uint32_t peak = 0;
                // With the ship gate open every live core ships; peak and growth are unused.
                if constexpr (kLaneShipWords != 0) {
                    const uint32_t tsum = tails[0] + tails[1] + tails[2] + tails[3] + tails[4];
                    grew = tsum - tails_seen[c];
                    tails_seen[c] = tsum;
                    sweep_grew |= grew != 0;
                    peak = d0;
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
                    if (peak > sweep_peak) {
                        sweep_peak = peak;
                    }
                }
                if (live != 0) {
                    // Deferral must survive one more interval of production, so `grew` (last interval's words) must
                    // be under the threshold too; that bounds a deferred core at ~2x threshold.
                    if (grid_busy && stop_seen_at == 0 && peak < kLaneShipWords && grew < kLaneShipWords &&
                        peak < kLaneTrigger) {
                        continue;
                    }
                    hot[c] = 1;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                } else if (hot[c] != 0) {
                    // A hot core scanning empty is almost always the producer's 64-word batched tail publish, not
                    // idleness; skipping it would double its service interval. One-shot: an idle core wastes one
                    // empty frame.
                    hot[c] = 0;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                }
            }
        }

        // One ship site: the last batch leaves through the same code, on the pass that finds nothing to issue.
        uint32_t cur = 0;
        uint32_t n = 0;
        const uint8_t* batch = ship_list;
        while (true) {
            const bool more = cur < n_ship;
            if (more) {
                retire_gen(gen);
                n = (n_ship - cur) < kGenSlots ? (n_ship - cur) : kGenSlots;
                batch = &ship_list[cur];
                const uint32_t pk = issue_batch(batch, n, gen_base, ring_base);
                if (pk < min_peak) {
                    min_peak = pk;
                }
                cur += n;
                // Refresh the next batch's tails in the same flight (this generation's read barrier covers them); a
                // full list's last batch refreshes the next sweep's first, so a saturated sweep opens on the issue with
                // no wave.
                uint32_t nn = n_ship - cur;
                uint32_t ri = cur;
                if (nn > kGenSlots) {
                    nn = kGenSlots;
                } else if (nn == 0 && n_ship == num_cores) {
                    nn = num_cores < kGenSlots ? num_cores : kGenSlots;
                    ri = 0;
                }
                for (uint32_t i = 0; i < nn; i++) {
                    cv_issue(core_noc, ship_list[ri + i]);
                }
            }

            if (have_pend) {
                emit_slots(pend_base, pend_n);
                if constexpr (kSpool) {
                    gen_dma_mark[pend_gen] = pump.dma_issued;
                }
                gen_shipped[pend_gen] = true;
                have_pend = false;
            }
            if (!more) {
                break;
            }
            if constexpr (kSpool) {
                if (pump.level >= SpoolPump::kLevelInline) {
                    pump.pass();
                }
            }

            // Hardware-counted read barrier: every read carries transaction id 0, so the NIU's per-id outstanding count
            // is the barrier. The spin doubles as the pump's slot only at full pressure, where the pump's GDDR reads no
            // longer contend with the gathers.
            while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_REQS_OUTSTANDING_ID(0)) != 0) {
                if constexpr (kSpool) {
                    // Inline level means occupancy is over the 5/8 line, so nonempty holds.
                    if (pump.level >= SpoolPump::kLevelInline) {
                        pump.pass();
                    }
                }
            }
            invalidate_l1_cache();
            advance_heads(n, batch);

            pend_n = n;
            pend_gen = gen;
            pend_base = gen_base;
            have_pend = true;
            gen = gen + 1u == kNGens ? 0u : gen + 1u;
            gen_base = gen == 0u ? kStageBase : gen_base + kGenSlots * kSlotBytes;
        }
        // Enter only when the scan would have shipped every core anyway; a core dropping below either bound leaves
        // the mode.
        const bool next = n_ship == num_cores && min_peak != 0 && min_peak >= kLaneShipWords;
        if (next && !all_live) {
            for (uint32_t c = 0; c < num_cores; c++) {
                ship_list[c] = static_cast<uint8_t>(c);
            }
        }
        all_live = next;

        // Every issued batch has passed its read barrier, so each scratch is exactly the head it was relieved to.
        if ((sweeps & (kHeadRefreshSweeps - 1u)) == 0) {
            for (uint32_t c = 0; c < num_cores; c++) {
                post_heads(core_noc, c);
            }
        }
        // Busy sweeps below the first band skip the post-sweep pump: a capture that fits the spool gets pure gather.
        if constexpr (kSpool) {
            const bool half_turn = pump.level == SpoolPump::kLevelHalf && (sweeps & 1u) != 0;
            if (pump.level >= SpoolPump::kLevelEverySweep || half_turn || pump.fresh_boost ||
                relieved == relieved_at_sweep_start) {
                pump.pass();
                pump.notify();
            }
        }

        if constexpr (kLaneShipWords != 0) {
            if (sweep_grew) {
                grow_streak++;
                quiet_streak = 0;
            } else if (++quiet_streak >= kFlushQuietSweeps) {
                grow_streak = 0;
            }
            grid_busy = grow_streak >= kBatchArmSweeps;
        }
        if constexpr (kSpool) {
            pump.freshness_tick(kSpoolFreshCycles);
        }
        // Collapse on work, creep toward the ceiling when idle; live-but-untriggered lanes count as work, since a
        // head only reaches a producer on a ship.
        if (relieved != relieved_at_sweep_start || sweep_peak >= kCvBusyPeak) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < kCvIdleGapMinInc) {
                inc = kCvIdleGapMinInc;
            }
            gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
        }
        if (gap != 0) {
            const uint64_t until = get_timestamp() + gap;
            while (get_timestamp() < until) {
                if constexpr (kSpool) {
                    pump.pass_cold();
                }
            }
        }
    }

    finish(pump, sender, stop, killed);
}
