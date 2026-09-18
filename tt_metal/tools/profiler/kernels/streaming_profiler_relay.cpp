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
#include "hostdev/streaming_profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

// write_cmd_buf is programmed once at init; nothing else on this core touches it.
inline void write_to_host(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, size, 1);
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
    write_to_host(sender.d2h.pcie_xy_enc, src, base + dst, first);
    if (first < len) {
        write_to_host(sender.d2h.pcie_xy_enc, src + first, base, len - first);
    }
}

// Blackhole stores can reach SRAM out of order, and the NIU and the DMA engine read the words the scalar core
// staged.
FORCE_INLINE void staged_store_fence() { asm volatile("fence" ::: "memory"); }

// Not socket_notify_receiver: it re-inits write_cmd_buf onto another VC, and the bytes_sent word can then
// overtake the data it announces. Same VC is not enough either: the PCIe tile turns each NoC write into its own
// AXI and PCIe transactions and keeps no order between packets on the way to host memory (a 4 B notify has been
// seen landing ahead of the 15 KB pushed before it), so bytes_sent goes out only once the tile has acknowledged
// every push.
inline void notify_bytes_sent(const SocketSenderInterface& sender) {
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    volatile tt_l1_ptr sender_socket_md* cfg =
        reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr);
    cfg->bytes_sent = sender.bytes_sent;
    staged_store_fence();
    write_to_host(
        sender.d2h.pcie_xy_enc,
        sender.config_addr,
        (static_cast<uint64_t>(sender.d2h.bytes_sent_addr_hi) << 32) | sender.downstream_bytes_sent_addr,
        4u);
}

constexpr uint32_t kStageBase = get_named_compile_time_arg_val("stage_base");
constexpr uint32_t kNStage = get_named_compile_time_arg_val("n_stage");
constexpr uint32_t kCoreRecords = get_named_compile_time_arg_val("core_records");
constexpr uint32_t kDoneAddr = get_named_compile_time_arg_val("done_addr");
// 1 = quiesce; 2 = the host has read everything it needs from this L1, restore the NIU.
constexpr uint32_t kStopAddr = get_named_compile_time_arg_val("stop_addr");
constexpr uint32_t kSocketConfigAddr = get_named_compile_time_arg_val("socket_config_addr");
constexpr uint32_t kMaxCores = get_named_compile_time_arg_val("max_cores");
static_assert(kMaxCores <= 256, "the core lists index cores as bytes");
// Static VC for PCIe pushes, spread across relays by the host.
constexpr uint32_t kWriteVc = get_named_compile_time_arg_val("write_vc");
// A core ships once its fullest lane holds this percent of its ring. Per lane, not per span: the producer that
// blocks is always one lane. The measured stall-free band ends between 30 and 35.
constexpr uint32_t kShipMinPct = 25;
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
// Two-core batches in kNGens generations; in spool mode the leftover slots are the two drain bounces.
constexpr uint32_t kGenSlots = 2;
static_assert(kGenSlots == 2, "the frame emit is written for two-slot generations");
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kNGens = 2;
static_assert(kNStage >= kNBounce + kNGens * kGenSlots, "the staging arena must hold two generations and the bounces");
// Both DMA streams are issued without the ready poll (dma_async_*<false>): a generation's emit is at most two
// appends of at most two pieces each (one split at the spool wrap), retire_gen keeps at most kNGens generations
// in flight, and each bounce holds at most one drain read.
static_assert(kNGens * 4 < experimental::kMaxOutstandingWrites, "the ship stream could fill and drop an issue");
static_assert(kNBounce < experimental::kMaxOutstandingReads, "the drain stream could fill and drop an issue");
// 112 B per core: control-vector words 12..31 at +0 (the lanes' state slots then their tails, in that address
// order so one read observes a slot no later than its tail), the head mirror at +80, the wire XY word behind the
// heads so the head write is 20 bytes.
constexpr uint32_t kCvBaseWord = 16;
constexpr uint32_t kCvReadBytes = 64;
constexpr uint32_t kCvReadSrcOff = kCvBaseWord * 4u;
constexpr uint32_t kRecordBytes = 128;
constexpr uint32_t kTailWord = kernel_profiler::SPSC_RING_TAIL_0 - kCvBaseWord;
constexpr uint32_t kTimerWord = kernel_profiler::SPSC_STATE_TIMER_0 - kCvBaseWord;
constexpr uint32_t kHeadWord = kCvReadBytes / 4u;
constexpr uint32_t kXyWord = kHeadWord + kNumRisc;
constexpr uint32_t kPeakWord = kXyWord + 1;  // the core's largest lane take at its last gather
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
    (!kSpool || kBounceBytes <= kernel_profiler::SPSC_NOTIFY_CAP_BYTES) &&
        2u * kSlotBytes <= kernel_profiler::SPSC_NOTIFY_CAP_BYTES,
    "a single push must fit under the notify cap");
static_assert(
    !kSpool || kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
    "bounces must fit inside the mapped staging arena");
static_assert(!kSpool || kSpoolBytes % kPageBytes == 0, "spool wraps on pages");
constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
// A probe below the ship gate waits at most this many sweeps (~5-10 ms at the idle gap), so a lane that trickles
// still reaches the host, and a sparse grid's frame headers are bounded to one frame per core per that long.
constexpr uint32_t kMaxDeferSweeps = 2048;
constexpr uint64_t kCyclesPerUs = 1350;  // DRISC wall clock at the 1.35 GHz AICLK
// Idle backoff ceiling, waited as a 32-bit low-word delta: a 64-bit wall-clock read on Blackhole can return the next
// epoch's high half with a pre-wrap low word (+2^32), which once parked a relay for 3.2 s. 20 us exceeded a lane's
// fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 5 * kCyclesPerUs;
constexpr uint32_t kCvIdleGapMinInc = 256;
// Below the first band the host is otherwise fed nothing until the spool fills that far; one pass every this many
// sweeps keeps it busy at a bounce per stride, and bounds host staleness to the stride.
constexpr uint32_t kIdlePumpStride = 8;

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
    // Pump effort by spool occupancy: the sweep cadence, every sweep, also per batch and inside the read-wait spin.
    enum : uint32_t { kLevelIdle = 0, kLevelEverySweep = 1, kLevelInline = 2 };
    static constexpr uint32_t kBandEverySweep = kSpoolBytes / 2u;
    static constexpr uint32_t kBandInline = kSpoolBytes / 2u + kSpoolBytes / 8u;
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
    uint32_t notified = 0;    // bytes_sent as the host last saw it
    uint32_t acked_seen = 0;  // the downstream's bytes_acked as last read
    uint32_t level = kLevelIdle;
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
    FORCE_INLINE void rebalance() {
        const uint32_t occ = occupancy();
        level = occ >= kBandInline ? kLevelInline : occ >= kBandEverySweep ? kLevelEverySweep : kLevelIdle;
    }

    FORCE_INLINE void append(uint32_t src, uint32_t len) {
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
    FORCE_INLINE void reclaim_shipped(bool& did) {
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

    // Refill before shipping so the read runs under the ship's NoC issue; a second concurrent refill only at
    // full pressure, where the deeper bank queue pays for itself.
    FORCE_INLINE void refill(bool& did) {
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

    FORCE_INLINE void ship(bool& did) {
        const uint32_t rdy = oldest_ready();
        if (rdy != kNone) {
            // bytes_acked only advances, so a copy that shows room is still right; it is re-read (an L1 invalidate
            // the next record reads pay for) only when it does not.
            uint32_t nb = b[rdy].bytes - b[rdy].off;
            uint32_t bytes_free = sender_.downstream_fifo_total_size - (sender_.bytes_sent - acked_seen);
            if (bytes_free < nb) {
                invalidate_l1_cache();
                acked_seen = *acked_;
                bytes_free = sender_.downstream_fifo_total_size - (sender_.bytes_sent - acked_seen);
                if (bytes_free < nb) {
                    nb = bytes_free & ~(kPageBytes - 1u);
                }
            }
            if (nb != 0) {
                // A push past the notify cap first announces what is acked so far; while those acks are still out
                // the bounce waits a pass instead of the sweep waiting on PCIe.
                if (sender_.bytes_sent - notified + nb > kernel_profiler::SPSC_NOTIFY_CAP_BYTES) {
                    if (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
                        return;
                    }
                    notify_now();
                }
                push_fifo(sender_, kBounceBase0 + rdy * kBounceBytes + b[rdy].off, sender_.write_ptr, nb);
                socket_push_pages(sender_, nb / kPageBytes);
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

    __attribute__((noinline)) void notify_now() {
        notify_bytes_sent(sender_);
        notified = sender_.bytes_sent;
    }
    // At the end of a sweep; notify_if_cap adds one before a push that would exceed the cap.
    FORCE_INLINE void notify() {
        if (sender_.bytes_sent != notified) {
            notify_now();
        }
    }
    // Before a push of nb bytes: keeps every byte landed within SPSC_NOTIFY_CAP_BYTES of a bytes_sent the host holds.
    FORCE_INLINE void notify_if_cap(uint32_t nb) {
        if (sender_.bytes_sent - notified + nb > kernel_profiler::SPSC_NOTIFY_CAP_BYTES) {
            notify_now();
        }
    }
};

// Tail reads and head writes each own a command buffer on the read NoC, programmed once; a per-core command
// is the coordinate, one address and the send.
constexpr uint32_t kCvCmdBuf = write_at_cmd_buf;
constexpr uint32_t kHeadCmdBuf = write_cmd_buf;
constexpr uint32_t kSelfCmdBuf = write_reg_cmd_buf;
constexpr uint32_t kGatherTxn = 1;
constexpr uint32_t kProbeTxn = 2;

FORCE_INLINE uint32_t record(uint32_t c) { return kCoreRecords + c * kRecordBytes; }

// Computed once: get_noc_addr's coordinate arithmetic would otherwise run at every issue of an
// instruction-bound sweep.
// NOC_TARG_ADDR_COORDINATE field of each worker's profiler block, ready to write.
static uint32_t core_coord[kMaxCores];
static uint32_t ring_base;  // lane 0's ring on every worker: the control block plus its control words

// The gather command buffer's register block and its send word. Laundered through an empty asm: as constants
// the allocator rematerialises both before every poll of the unrolled lane walk instead of keeping them in the
// registers it has free.
struct GatherRegs {
    volatile uint32_t* base;
    uint32_t send;
};
FORCE_INLINE GatherRegs gather_regs() {
    volatile uint32_t* base = reinterpret_cast<volatile uint32_t*>(
        NOC_REGS_START_ADDR + NOC_CMD_BUF_INSTANCE_OFFSET(kReadNoc, read_cmd_buf));
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

// Counted, not barriered: in-flight gather responses also bump the counter, which can only hand the scan
// stale-but-valid tails (tails are monotonic).
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

// Lands every control read issued under transaction id `id`: 0 for tail refreshes and the seed, kProbeTxn for probes.
FORCE_INLINE void cv_wait(uint32_t id) {
    while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_REQS_OUTSTANDING_ID(id)) != 0) {
    }
    invalidate_l1_cache();
}

// Posted (the barriers protect staging, which a head write never touches) and on the read NoC, where it does
// not queue behind frame data and the PCIe tile's acceptance jitter.
FORCE_INLINE void post_heads(uint32_t c) {
    noc_wwrite_with_state<DM_DEDICATED_NOC, kHeadCmdBuf, CQ_NOC_SNdl, CQ_NOC_SEND, CQ_NOC_WAIT, true, true>(
        kReadNoc, heads(c), core_coord[c], 0);
}

// A frame occupies whole socket pages on the wire.
FORCE_INLINE uint32_t page_round(uint32_t bytes) { return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u); }

// Prefix word 1 is the payload length in words.
constexpr uint32_t kLenWord = 1;
FORCE_INLINE uint32_t frame_bytes(uint32_t slot) {
    return (reinterpret_cast<const tt_l1_ptr uint32_t*>(slot)[kLenWord] + kPrefix) * 4u;
}

// The wrap-image rule as one subtraction on the take already in hand (spsc_span_wrap_image's form costs two
// more instructions per wrapping lane); the check pins it to the shared rule.
constexpr uint32_t kImageMinTake = kRingWords - kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
constexpr bool image_rule_matches() {
    for (uint32_t take = 1; take < 2 * kRingWords; take++) {
        const bool mine = take - kImageMinTake <= kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
        if (mine != kernel_profiler::spsc_span_wrap_image(kRingWords - 1u, take, kRingWords)) {
            return false;
        }
    }
    return true;
}
static_assert(image_rule_matches(), "the inline image test drifted from spsc_span_wrap_image");

// The record's first 64 B are the frame's control block, state slots and tails, in the frame's own layout: one
// loopback read per frame, issued where the sweep waits on the gathers so the same barrier covers it and no
// instruction of it lands in the lane walk.
FORCE_INLINE void self_read_batch(const uint8_t* cores, uint32_t n, uint32_t slot) {
    for (uint32_t i = 0; i < n; i++) {
        while (!noc_cmd_buf_ready(kReadNoc, kSelfCmdBuf)) {
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_LO, record(cores[i]));
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_LO, slot + kPrefix * 4u);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        slot += kSlotBytes;
    }
}

// Gather-read each live run straight to its packed wire offset; the pads keep read src == dst (mod 16 B) for
// every piece, wrap continuations included. Returns the smallest per-core peak lane take.
__attribute__((noinline)) uint32_t issue_batch(const uint8_t* cores, uint32_t n, uint32_t slot, uint32_t rb) {
    uint32_t min_peak = ~0u;
    const uint8_t* end = cores + n;
    const GatherRegs g = gather_regs();
    do {
        const uint32_t c = *cores++;
        // The head advance hides behind the NIU's acceptance of the same lane's read; nothing reads the record
        // before the batch barrier.
        volatile tt_l1_ptr uint32_t* __restrict rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c));
        volatile tt_l1_ptr uint32_t* __restrict frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        uint32_t off = kPrefix + kWireCtrl;
        uint32_t peak = 0;
        while (!gather_ready(g)) {
        }
        gather_reg(g, NOC_TARG_ADDR_COORDINATE, core_coord[c]);
        // Unrolled: the three induction pointers and the split +2048 stride were 5 of the lane's 29 instructions.
        // All three read shapes stay inline: at the knee most runs wrap (P ~ take/512), so none of them is rare.
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
            const uint32_t hm = start & (kRingWords - 1u);
            // Lane 0's first read needs no poll: the buffer was polled before the coordinate write above and
            // nothing has been sent since. Every later read follows a send.
            const bool poll = r != 0;
            if (hm + take <= kRingWords) {
                off += kernel_profiler::spsc_span_pack_pad(start, off);
                gather_read(g, poll, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                off += take;
            } else if (take - kImageMinTake <= kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS) {
                // A near-full wrapping run ships as its whole ring image in one read (the decoder linearises by head).
                // Coalescing adjacent images into one read starves the producer's L1 port ~70x.
                off += kernel_profiler::spsc_span_pack_pad(0u, off);
                gather_read(g, poll, ring_src, slot + off * 4u, kRingWords * 4u);
                off += kRingWords;
            } else {
                // A small wrapping run ships as two byte-exact pieces: its dead remainder would be most of the ring.
                off += kernel_profiler::spsc_span_pack_pad(start, off);
                const uint32_t first = kRingWords - hm;
                uint32_t dst = slot + off * 4u;
                const uint32_t first_bytes = first * 4u;
                gather_read(g, poll, ring_src + hm * 4u, dst, first_bytes);
                // The second piece's operands are derived only after the first send: the poll that follows must
                // trail the send by at least this much work, or it lands before the NIU has assigned the VC and
                // costs a full extra spin per command.
                uint32_t rest = take;
                asm volatile("" : "+r"(dst), "+r"(rest), "+r"(off));
                off += rest;
                rest -= first;
                gather_read(g, true, ring_src, dst + first_bytes, rest * 4u);
            }
        }
        frame[kernel_profiler::SPSC_PREFIX_XY] = rec[kXyWord];
        // frame[0] is staged once at init; only the payload word varies.
        frame[kLenWord] = off - kPrefix;
        rec[kPeakWord] = peak;
        if (peak < min_peak) {
            min_peak = peak;
        }
        slot += kSlotBytes;
    } while (cores != end);
    return min_peak;
}

static FORCE_INLINE void program_command_buffers(uint32_t cv_src) {
    // Gathers carry transaction id kGatherTxn, tail refreshes id 0 and probes kProbeTxn: the NIU's outstanding count
    // per id lets each wait cover exactly the reads it consumes.
    while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kGatherTxn));
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
    // Loopback reads of this core's own records: the return coordinate the gathers use is this core, and in stream
    // mode a plain local address reaches DRISC L1 (drisc_mode.h).
    while (!noc_cmd_buf_ready(kReadNoc, kSelfCmdBuf)) {
    }
    noc_read_init_state<kSelfCmdBuf>(kReadNoc);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(kGatherTxn));
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_MID, 0);
    {
        const uint32_t self = NOC_CMD_BUF_READ_REG(kReadNoc, read_cmd_buf, NOC_RET_ADDR_COORDINATE);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_TARG_ADDR_COORDINATE, self);
        NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_RET_ADDR_COORDINATE, self);
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kSelfCmdBuf, NOC_AT_LEN_BE, kCvReadBytes);
    while (!noc_cmd_buf_ready(kReadNoc, kHeadCmdBuf)) {
    }
    noc_write_init_state<kHeadCmdBuf, CQ_NOC_mkP>(kReadNoc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_LO, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(kReadNoc, kHeadCmdBuf, NOC_AT_LEN_BE, kNumRisc * 4u);
    // Programmed once: nothing else on this core touches write_cmd_buf on the egress NoC.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);
}

// Only the heads and the core identity are staged per frame (the control block arrives with the loopback read); the
// rest must read zero on the wire.
static FORCE_INLINE void zero_stage_prefixes() {
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
static FORCE_INLINE void seed_heads(uint32_t num_cores, volatile tt_l1_ptr uint32_t* coords, uint32_t* tails_seen) {
    cv_issue(0, num_cores);
    cv_wait(0);
    for (uint32_t c = 0; c < num_cores; c++) {
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c));
        uint32_t tsum = 0;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            rec[kHeadWord + r] = rec[kTailWord + r];
            tsum += rec[kTailWord + r];
        }
        rec[kXyWord] = coords[c];
        tails_seen[c] = tsum;
    }
}

// The spool drains, then the socket barrier holds until the host has acked every byte: done means nothing of the
// capture is in flight anywhere.
static FORCE_INLINE void finish(SpoolPump& pump, SocketSenderInterface& sender, volatile tt_l1_ptr uint32_t* stop) {
    if constexpr (kSpool) {
        while (!pump.drained()) {
            pump.pass_cold();
            // Notify per pass: with a FIFO smaller than the backlog, credit only returns after the host has seen the
            // bytes.
            pump.notify();
        }
    }
    pump.notify();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr) = kernel_profiler::kRelayDrainedWord;
    socket_barrier(sender);
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    // The posted head write-backs are outside that barrier's predicate.
    while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc))) {
    }
    update_socket_config(sender);
    // After the socket barrier, so the host only sees `done` once every page is out.
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr) = kernel_profiler::kRelayDoneWord;
    // NIU_CFG_0 persists until chip reset, so whoever set stream mode restores it; NOC2AXI takes this L1 out of the
    // host's view, so it waits for the host to say it has read everything.
    do {
        invalidate_l1_cache();
    } while (*stop != kernel_profiler::kRelayStopRelease);
    experimental::drisc_set_noc2axi_mode_all();
}

// Every core is on exactly one list. The ship list persists across sweeps and is gathered in order off the
// per-batch tail refreshes; the probe list is read once per sweep, and a core that comes back live joins the
// sweep in progress. A gathered core whose peak comes back under the sweep's demote bound moves to the probes.
// tails_seen is the sum of a core's tails at its last read; tails are monotonic, so its delta is one interval's
// production.
static uint8_t ship_list[kMaxCores];
static uint8_t probe_list[kMaxCores];
static uint8_t demote_pos[kMaxCores];  // ship-list positions demoted this sweep, ascending
static uint32_t tails_seen[kMaxCores];
static uint32_t deferred[kMaxCores];  // sweeps a probe has waited below the ship gate
static uint32_t n_list;
static uint32_t n_probe;
static uint32_t acked_seen;  // direct push: the downstream's bytes_acked as last read
static uint32_t n_demote;
static bool sweep_live;  // a probe had unshipped words this sweep

// Out of line: these run at most once per sweep, and inline they cost the batch loop registers.
__attribute__((noinline)) static void note_demotions(
    const uint8_t* cores, uint32_t n, uint32_t cur, uint32_t demote_below) {
    for (uint32_t i = 0; i < n; i++) {
        if (reinterpret_cast<const tt_l1_ptr uint32_t*>(record(cores[i]))[kPeakWord] < demote_below) {
            demote_pos[n_demote++] = static_cast<uint8_t>(cur + i);
        }
    }
}

// The probes' tails have landed; every live, undeferred probe joins the ship list.
__attribute__((noinline)) static void promote_probes(bool defer_ok) {
    for (uint32_t k = 0; k < n_probe;) {
        const uint32_t c = probe_list[k];
        const tt_l1_ptr uint32_t* __restrict rec = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c));
        const tt_l1_ptr uint32_t* __restrict tails = rec + kTailWord;
        const tt_l1_ptr uint32_t* __restrict mine = rec + kHeadWord;
        // Unrolled into registers: an indexed-array loop spills, and each spilled word is an L1 round trip per
        // core per sweep.
        const uint32_t d0 = tails[0] - mine[0];
        const uint32_t d1 = tails[1] - mine[1];
        const uint32_t d2 = tails[2] - mine[2];
        const uint32_t d3 = tails[3] - mine[3];
        const uint32_t d4 = tails[4] - mine[4];
        // No clamp: a producer blocks 506 words past the head it sees, and the mirror is never behind that head.
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
        // Deferral must survive one more interval of production, so `grew` (last interval's words) must be under
        // the threshold too; that bounds a deferred core at ~2x threshold.
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

// Descending, so each swap-remove leaves the earlier positions valid.
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

// A staged slot is its frame's wire image, so a frame is one write or one DMA; the trailing page fill is
// never written, the host reads past it.
static FORCE_INLINE void emit_slots(SpoolPump& pump, SocketSenderInterface& sender, uint32_t base, uint32_t count) {
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
        // The spool's back-pressure; it holds through quiesce.
        while (!pump.has_room(bytes)) {
            pump.pass_cold();
        }
        staged_store_fence();
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
        staged_store_fence();
        // bytes_acked only advances, so a copy that shows room is still right; it is re-read (an L1 invalidate the
        // next batch's record reads pay for) only when it does not. Credit only returns for bytes the host has been
        // told about, so a FIFO without room is notified before the wait; otherwise bytes_sent goes out once per
        // sweep, as the pump's does.
        if (sender.downstream_fifo_total_size - (sender.bytes_sent - acked_seen) < bytes) {
            pump.notify();
            socket_reserve_pages(sender, bytes / kPageBytes);
            acked_seen = *pump.acked_;
        }
        pump.notify_if_cap(bytes);
        push_fifo(sender, base, sender.write_ptr, raw0);
        if (len1 != 0) {
            push_fifo(sender, base + kSlotBytes, sender.write_ptr + len0, raw1);
        }
        socket_push_pages(sender, bytes / kPageBytes);
    }
}

void kernel_main() {
    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // profiler_msg_t base on every worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t xy = coords[i];
        const uint64_t noc_addr = get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src);
        core_coord[i] = static_cast<uint32_t>(noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
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

    // Statics persist across launches, so everything the loop trusts is re-initialised.
    n_list = 0;
    n_probe = num_cores;
    acked_seen = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        probe_list[i] = static_cast<uint8_t>(i);
        deferred[i] = 0;
    }
    seed_heads(num_cores, coords, tails_seen);

    uint32_t relieved = 0;
    uint32_t sweeps = 0;
    uint32_t gap = 0;
    // An empty core always leaves the ship list; so does one below the ship gate while deferral holds.
    constexpr uint32_t kDeferBelow = kLaneShipWords > 1 ? kLaneShipWords : 1u;
    // Persists across sweeps so a sweep's final ship drains under the pace gap, not on its own critical path.
    uint32_t gen_shipped = 0;  // bit g: generation g's last frame may still be leaving staging

    uint32_t gen_dma_mark[kNGens] = {};
    SpoolPump pump(sender);

    // Two more sweeps once stop is seen: the first ships every live core (deferral is off) and reads every tail it
    // will use afterwards, so the second gathers everything published before the stop. Anything a producer publishes
    // after that is outside the capture.
    uint32_t stop_sweeps = 0;
    while (true) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_sweeps == 2) {
                break;
            }
            stop_sweeps++;
        }
        sweeps++;
        *hb = sweeps;
        const uint32_t relieved_at_sweep_start = relieved;

        sweep_live = false;
        n_demote = 0;
        // Gather generation G on the read NoC while G^1 ships. No lambda here: a by-reference capture costs sweep
        // time at the saturation boundary.
        uint32_t gen = 0;
        uint32_t pend_n = 0;  // frames staged for the previous generation and not yet shipped; 0 = none pending
        static_assert(kNGens == 2, "the pending generation is derived as the other one");
        // A core below the ship gate waits only while the relay has other cores to gather: with an empty ship list
        // nothing is lost by shipping it now, and at the ingress knee the wait was the stall margin.
        const bool defer_ok = n_list != 0 && stop_sweeps == 0;
        const uint32_t demote_below = defer_ok ? kDeferBelow : 1u;

        // Heads go out at the read barrier, not the frame emit: once the reads land the producer's ring slots are
        // free. A lambda on purpose: written inline, issue_batch picks up a spill per core.
        auto advance_heads = [&](uint32_t n, const uint8_t* cores) __attribute__((always_inline)) {
            for (uint32_t i = 0; i < n; i++) {
                post_heads(cores[i]);
                relieved++;
            }
        };

        // Generation g's previous frame must be out of staging before its slots refill; gen_shipped persists so a
        // sweep never waits on its own last ship.
        auto retire_gen = [&](uint32_t g) __attribute__((always_inline)) {
            if ((gen_shipped >> g) & 1u) {
                // Both predicates complete on this device alone, so no consumer state can hang them.
                if constexpr (kSpool) {
                    // Stream completion is FIFO, so outstanding <= later-issues means this generation retired.
                    const uint32_t since = pump.dma_issued - gen_dma_mark[g];
                    const uint32_t cap = since > experimental::kMaxOutstandingWrites ? experimental::kMaxOutstandingWrites : since;
                    while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
                    }
                } else {
                    // Sent-only is legal: the slots' next writer is this core's own NIU read responses.
                    while (!ncrisc_noc_nonposted_writes_sent(NOC_INDEX)) {
                    }
                }
                gen_shipped &= ~(1u << g);
            }
        };

        // Probes fly under their own transaction id, so the first batch waits for its refreshed tails alone; they are
        // waited for after the loop.
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

        // One ship site: the last batch leaves through the same code, on the pass that finds nothing to issue. The
        // outer loop runs only for probes: they landed under the first batch's barrier (an empty list waits for
        // them here), and the cores they promote are gathered by a second pass over the appended tail.
        uint32_t cur = 0;
        uint32_t n = 0;
        while (true) {
            const uint32_t n_end = n_list;
            while (true) {
                const bool more = cur < n_end;
                uint32_t defer_ri = 0;
                uint32_t defer_nn = 0;
                if (more) {
                    // The tails this batch consumes were refreshed a batch ago (or probed at sweep start); they are
                    // waited for here, not at the gather barrier. retire_gen comes after them: with full frames its
                    // DMA wait is what binds here, and the poll and fence then run under it instead of after it.
                    cv_wait(0);
                    retire_gen(gen);
                    n = (n_end - cur) < kGenSlots ? (n_end - cur) : kGenSlots;
                    const uint32_t pk = issue_batch(&ship_list[cur], n, kGenBase[gen], ring_base);
                    if (pk < demote_below) {
                        note_demotions(&ship_list[cur], n, cur, demote_below);
                    }
                    cur += n;
                    // Refresh the next batch's tails in the same flight (this generation's read barrier covers them);
                    // the last batch refreshes the next sweep's first, so a sweep opens on the issue with no wave.
                    uint32_t nn = n_end - cur;
                    uint32_t ri = cur;
                    if (nn > kGenSlots) {
                        nn = kGenSlots;
                    } else if (nn == 0) {
                        nn = n_end < kGenSlots ? n_end : kGenSlots;
                        ri = 0;
                    }
                    // A refresh covering this batch's own cores (a list no longer than a batch) would land in records
                    // the frames' loopback reads are still sourcing; it goes out after the gather barrier instead.
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
                    emit_slots(pump, sender, kGenBase[pend_gen], pend_n);
                    if constexpr (kSpool) {
                        gen_dma_mark[pend_gen] = pump.dma_issued;
                    }
                    gen_shipped |= 1u << pend_gen;
                    pend_n = 0;
                }
                if (!more) {
                    break;
                }
                if constexpr (kSpool) {
                    if (pump.level >= SpoolPump::kLevelInline) {
                        pump.pass();
                    }
                }

                self_read_batch(&ship_list[cur - n], n, kGenBase[gen]);
                // Hardware-counted read barrier on the gathers' transaction id. The spin doubles as the pump's slot
                // only at full pressure, where the pump's GDDR reads no longer contend with the gathers. The heads are
                // NIU-sourced from L1, so no cache invalidate is needed before posting them.
                while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_REQS_OUTSTANDING_ID(kGatherTxn)) != 0) {
                    if constexpr (kSpool) {
                        // Inline level means occupancy is over the 5/8 line, so nonempty holds.
                        if (pump.level >= SpoolPump::kLevelInline) {
                            pump.pass();
                        }
                    }
                }
                for (uint32_t i = 0; i < defer_nn; i++) {
                    cv_issue(ship_list[defer_ri + i]);
                }
                advance_heads(n, &ship_list[cur - n]);

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
        // The next sweep's first batch had its tails read by this sweep's last batch. demote() may have moved other
        // cores into it, and a pump pass plus the notify's PCIe ack wait would leave those tails several us staler than
        // any other batch's; in either case the first batch is read again, under the ack wait when there is one.
        bool refresh_first = n_demote != 0;
        if (n_demote != 0) {
            demote();
        }
        const auto refresh_first_batch = [&]() __attribute__((always_inline)) {
            const uint32_t nn = n_list < kGenSlots ? n_list : kGenSlots;
            for (uint32_t i = 0; i < nn; i++) {
                cv_issue(ship_list[i]);
            }
            refresh_first = false;
        };
        // Busy sweeps below the first band skip the post-sweep pump: a capture that fits the spool gets pure gather.
        if constexpr (kSpool) {
            const bool cadence = (sweeps & (kIdlePumpStride - 1u)) == 0;
            if (pump.level >= SpoolPump::kLevelEverySweep || cadence || relieved == relieved_at_sweep_start) {
                pump.pass();
                refresh_first_batch();
                pump.notify();
            }
        } else {
            refresh_first_batch();
            pump.notify();
        }
        if (refresh_first) {
            refresh_first_batch();
        }

        // Collapse on work, creep toward the ceiling only when nothing is live: a lane waiting below the ship gate is
        // work too, since a head only reaches a producer on a ship.
        if (relieved != relieved_at_sweep_start || sweep_live) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < kCvIdleGapMinInc) {
                inc = kCvIdleGapMinInc;
            }
            gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
        }
        if (gap != 0) {
            const uint32_t t0 = get_timestamp_32b();
            while (get_timestamp_32b() - t0 < gap) {
                if constexpr (kSpool) {
                    pump.pass_cold();
                }
            }
        }
    }

    finish(pump, sender, stop);
}
