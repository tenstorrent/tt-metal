// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's relay. Each relay is resident on one DRAM bank's free DRISC and owns a
// slice of the worker grid. It polls each worker's per-RISC SPSC ring tails, gather-reads the live
// runs into packed wire frames in L1 staging, DMAs the frames into a spool ring in its own GDDR
// bank, and a non-blocking pump forwards spool bytes to the host FIFO through its own D2H socket.
// Producers are lossless: a full ring blocks the worker, so the whole pipeline is flow-controlled
// end to end.

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

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. The caller owns the one
// noc_write_init_state<write_cmd_buf>(NOC_INDEX, vc) per push; nothing between a push's calls
// invalidates that state.
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

// Write `len` bytes of L1 at `src` to FIFO offset `dst`, splitting a piece that crosses the FIFO wrap:
// socket_push_pages only wraps the pointer. fifo_size is a whole number of pages, so the split preserves
// the pack pads' NoC congruence.
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

// The bytes_sent notify. Not socket_notify_receiver: that re-inits write_cmd_buf onto a different VC, and
// the mesh can then deliver the bytes_sent word ahead of the data it announces.
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

// Replacement for socket_reserve_pages (socket_api.h), which spins on `bytes_free < num_bytes` with no
// escape. The wait holds through quiesce (stop=1): the receiver is still acking then, and returning would
// lose frames whose heads were already relieved. Only the kill switch (stop=2) returns false.
inline bool reserve_pages(const SocketSenderInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* stop) {
    const uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_base_addr);
    const uint32_t acked_end = socket.bytes_acked_base_addr + socket.num_downstreams * bytes_acked_size_bytes;
    while (reinterpret_cast<uint32_t>(acked) < acked_end) {
        while (true) {
            invalidate_l1_cache();
            // bytes_acked is never ahead of bytes_sent, so this cannot underflow
            const uint32_t bytes_free = socket.downstream_fifo_total_size - (socket.bytes_sent - *acked);
            if (bytes_free >= num_bytes) {
                break;
            }
            if (*stop == kernel_profiler::kRelayStopRelease) {
                return false;
            }
        }
        acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
    }
    return true;
}

// dma_async_write/read (gddr_dma.h) poll the engine's ready status before every issue; these variants
// are the same register programming with the poll gone. The poll guards queue-full, which this pipeline
// makes unreachable: the generation gate keeps stream-0 writes at most ~10 of the queue's 15, and the
// pump keeps stream-1 reads at most 2 of 255.
__attribute__((always_inline)) inline void dma_write_unchecked(
    uint8_t stream, uint32_t src_l1, uint64_t dst_gddr, uint32_t size_bytes) {
    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.start_of_packet = 0;
    attrs.f.end_of_packet = 0;
    attrs.f.transfer_start_raw = 1;
    experimental::program_dma_write_addresses_(stream, src_l1, dst_gddr);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

__attribute__((always_inline)) inline void dma_read_unchecked(
    uint8_t stream, uint64_t src_gddr, uint32_t dst_l1, uint32_t size_bytes) {
    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.transfer_start_read = 1;
    experimental::program_dma_read_addresses_(stream, src_gddr, dst_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

// The ship DMA (TX stream kShip) appends page-rounded frames to a ring in this DRISC's own GDDR bank, and
// pass() forwards them to the host FIFO through two L1 bounce buffers refilled by TX stream kDrain. A pass
// never spins: every wait this stage could have is a state a later pass observes, so the pump can delay
// host delivery but never the sweep. Byte counters are 64-bit because a long capture exceeds 4 GiB.
// kBytes == 0 selects the direct-push build, which never calls any of this.
template <
    uint32_t kBase,
    uint32_t kBytes,
    uint32_t kBounceBase,
    uint32_t kBounceBytes,
    uint32_t kPageBytes,
    uint8_t kShip,
    uint8_t kDrain>
struct SpoolPump {
    enum : uint32_t { kEmpty = 0, kReading = 1, kReady = 2, kShipping = 3 };
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
    uint32_t wr_off = 0;      // wr % kBytes
    uint32_t rd_iss_off = 0;  // rd_iss % kBytes
    uint32_t dma_issued = 0;  // cumulative ship-stream writes: the caller's per-generation completion gate
    uint32_t dma_rd_issued = 0;
    uint32_t chunks = 0;  // refills so far; also the sequence number the oldest-first ship compares
    Bounce b[2] = {};
    bool notify_pending = false;  // pump ships owe the host a bytes_sent notify (batched per sweep)
    // Pump effort, a pure function of spool occupancy. 0: idle sweeps and the pace gap only. 1: one
    // post-sweep pass every other sweep. 2: post-sweep every sweep. 3: also inline per-batch and in the
    // read-wait spin.
    uint32_t level = 0;
    // Latched per-sweep trickle for a workload too light to reach the occupancy bands, so the host
    // stream stays roughly real-time; cleared when the spool empties.
    bool fresh_boost = false;
    uint64_t oldest = 0;  // when the spool last went non-empty; 0 = empty
    uint32_t fresh_tick = 0;
    SocketSenderInterface& sender_;
    volatile tt_l1_ptr uint32_t* acked_;  // the downstream's bytes_acked word

    explicit SpoolPump(SocketSenderInterface& sender) :
        sender_(sender), acked_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr)) {}

    uint32_t occupancy() const { return static_cast<uint32_t>(wr - rd); }
    bool has_room(uint32_t bytes) const { return kBytes - occupancy() >= bytes; }
    bool drained() const { return rd_iss == wr && b[0].state == kEmpty && b[1].state == kEmpty; }

    // Must be called wherever wr or rd advance; nothing else recomputes the level.
    __attribute__((always_inline)) void rebalance() {
        const uint32_t occ = occupancy();
        level = occ >= kBytes / 2u + kBytes / 8u   ? 3u
                : occ >= kBytes / 2u               ? 2u
                : occ >= kBytes / 4u + kBytes / 8u ? 1u
                                                   : 0u;
    }

    // Append `len` bytes of staging at `src`, split at the ring wrap.
    __attribute__((always_inline)) void append(uint32_t src, uint32_t len) {
        while (len != 0) {
            const uint32_t piece = len > kBytes - wr_off ? kBytes - wr_off : len;
            dma_write_unchecked(kShip, src, kBase + wr_off, piece);
            dma_issued++;
            wr += piece;
            wr_off += piece;
            if (wr_off == kBytes) {
                wr_off = 0;
            }
            src += piece;
            len -= piece;
        }
    }

    __attribute__((always_inline)) void pass() {
        if (wr == rd && b[0].state == kEmpty && b[1].state == kEmpty) {
            return;
        }
        bool did = false;
        // SHIPPING -> EMPTY on write-ack, not on "sent": the bounce's next writer is the DMA engine,
        // which a sent-only gate does not fence. Every NIU register poll below is gated on the L1 state
        // that could consume it, so an idle pass reads no registers.
        if (b[0].state == kShipping || b[1].state == kShipping) {
            const uint32_t acked = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED);
            for (uint32_t i = 0; i < 2; i++) {
                if (b[i].state == kShipping && static_cast<int32_t>(acked - b[i].ack_target) >= 0) {
                    b[i].state = kEmpty;
                    did = true;
                }
            }
        }
        // READING -> READY when a bounce's drain reads retire. Stream completion is FIFO, so one
        // outstanding count gives each bounce its own line and both can fill at once.
        if (b[0].state == kReading || b[1].state == kReading) {
            const uint32_t rd_out = experimental::dma_get_reads_outstanding(kDrain);
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
        // Refill an empty bounce before shipping the ready one, so the read runs under the ship's NoC
        // issue. A second concurrent refill only at full pressure: the extra in-flight GDDR read deepens
        // the bank queue exactly when the ship DMA needs it.
        const uint32_t emp = b[0].state == kEmpty ? 0u : (b[1].state == kEmpty ? 1u : 2u);
        const bool want_refill = emp != 2u && (level >= 3u || b[emp ^ 1u].state != kReading);
        // Only ship-completed bytes are readable: nothing short of a ship write's completion orders a
        // drain read of the same address behind it.
        if (want_refill && done != wr && static_cast<uint32_t>(done - rd_iss) < kBounceBytes &&
            experimental::dma_get_writes_outstanding(kShip) == 0) {
            done = wr;
        }
        if (want_refill && done != rd_iss) {
            uint32_t len = static_cast<uint32_t>(done - rd_iss);
            if (len > kBounceBytes) {
                len = kBounceBytes;
            }
            if (len > kBytes - rd_iss_off) {
                len = kBytes - rd_iss_off;
            }
            dma_read_unchecked(kDrain, kBase + rd_iss_off, kBounceBase + emp * kBounceBytes, len);
            rd_iss += len;
            rd_iss_off += len;
            if (rd_iss_off == kBytes) {
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
        // Ship a READY bounce, as much as the host FIFO has credit for right now. Oldest first when
        // both are ready: the socket is a byte stream, so the younger bounce would reorder the wire.
        uint32_t rdy = 2u;
        if (b[0].state == kReady && b[1].state == kReady) {
            rdy = static_cast<int32_t>(b[0].seq - b[1].seq) < 0 ? 0u : 1u;
        } else if (b[0].state == kReady) {
            rdy = 0u;
        } else if (b[1].state == kReady) {
            rdy = 1u;
        }
        if (rdy != 2u) {
            invalidate_l1_cache();
            const uint32_t bytes_free = sender_.downstream_fifo_total_size - (sender_.bytes_sent - *acked_);
            uint32_t nb = b[rdy].bytes - b[rdy].off;
            if (bytes_free < nb) {
                nb = bytes_free & ~(kPageBytes - 1u);
            }
            if (nb != 0) {
                push_fifo(sender_, kBounceBase + rdy * kBounceBytes + b[rdy].off, sender_.write_ptr, nb);
                socket_push_pages(sender_, nb / kPageBytes);
                notify_pending = true;
                b[rdy].off += nb;
                if (b[rdy].off == b[rdy].bytes) {
                    b[rdy].state = kShipping;
                    b[rdy].off = 0;
                    // The ack mirror is cumulative, so this also covers earlier partial ships.
                    b[rdy].ack_target = noc_nonposted_writes_acked[NOC_INDEX];
                }
                did = true;
            }
        }
        if (did) {
            rebalance();
        }
    }

    // Out-of-line copy for the sites outside a loaded sweep, where the inlined copy is not worth its
    // code size.
    __attribute__((noinline)) void pass_cold() { pass(); }

    // Batched bytes_sent notify for pump ships: once per sweep instead of once per chunk.
    __attribute__((always_inline)) void notify() {
        if (notify_pending) {
            notify_bytes_sent(sender_);
            notify_pending = false;
        }
    }

    // Once per sweep, but the wall clock is read only every 64th call: even one read per sweep stalls
    // producers at the saturation boundary, and ~1 ms of stride is noise against the 50 ms bound. A
    // workload too light to reach the occupancy bands escalates to the latched trickle, never to inline
    // pumping, because freshness is a latency bound and not a pressure emergency.
    __attribute__((always_inline)) void freshness_tick(uint64_t deadline_cycles) {
        if (++fresh_tick < 64u) {
            return;
        }
        fresh_tick = 0;
        if (wr == rd) {
            oldest = 0;
            fresh_boost = false;
            return;
        }
        const uint64_t now = get_timestamp();
        if (oldest == 0 || level >= 1u || fresh_boost) {
            oldest = now;
            fresh_boost = level == 0u && fresh_boost;
        } else if (now - oldest > deadline_cycles) {
            fresh_boost = true;
        }
    }
};

constexpr uint32_t kStageBase = get_named_compile_time_arg_val("stage_base");
constexpr uint32_t kNStage = get_named_compile_time_arg_val("n_stage");
constexpr uint32_t kCoreRecords = get_named_compile_time_arg_val("core_records");
constexpr uint32_t kDoneAddr = get_named_compile_time_arg_val("done_addr");
// Host writes 1 = quiesce (drain everything, every wait holds), 2 = kill switch (abandon waits, free the NIU).
constexpr uint32_t kStopAddr = get_named_compile_time_arg_val("stop_addr");
constexpr uint32_t kSocketConfigAddr = get_named_compile_time_arg_val("socket_config_addr");
constexpr uint32_t kMaxCores = get_named_compile_time_arg_val("max_cores");
static_assert(kMaxCores <= 256, "ship_list and hot index cores as bytes");
// Static VC for PCIe pushes, spread across relays by the host.
constexpr uint32_t kWriteVc = get_named_compile_time_arg_val("write_vc");
// Ship threshold, percent of one ring. Binds on the core's fullest lane, not its span: the producer
// that blocks is always a single lane, and a span-percent under-reads the binding ring.
constexpr uint32_t kShipMinPct = get_named_compile_time_arg_val("ship_min_pct");
// GDDR spool ring in this DRISC's own bank; 0 bytes selects the direct-push path, where frames go
// straight from staging to the host FIFO.
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
// Staging layout: two-core batches in kNGens generations and, in spool mode, every slot the generations
// leave split into two drain bounce buffers.
constexpr uint32_t kGenSlots = 2;
static_assert(kGenSlots == 2, "the frame emit is written for two-slot generations");
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kNGens = (kNStage - kNBounce) / kGenSlots;
static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
// One 64-byte record per core in the host-provided region: the 32-byte tail read lands at +0, the head
// mirror follows at +32, and the core's wire XY word rides behind the five heads, so the head write sends
// only those 20 bytes.
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
// Per-lane ship trigger, and the fill level at which idle backoff must stop growing: a head only reaches
// a producer on a ship, so backing off while lanes fill toward the trigger blinds the relay.
constexpr uint32_t kLaneTrigger = kRingWords / 2u;
constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
constexpr uint64_t kCyclesPerUs = 1350;  // DRISC wall clock at the 1.35 GHz AICLK
// Idle backoff ceiling. 20 us exceeded a lane's fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 5 * kCyclesPerUs;
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
// Packed-gather congruence: pads bring each run to its ring phase, and slot base, payload base and wrap
// continuations must land congruent with no pad. One pad rule serves both the gather read into staging
// and the frame's PCIe write.
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

// Tail reads and head writes each own a command buffer on the read NoC (the gathers hold read_cmd_buf;
// nothing else on this core issues atomics or writes there), programmed once with everything common to
// every core, so a per-core command is the coordinate, one address and the send.
constexpr uint32_t kCvCmdBuf = write_at_cmd_buf;
constexpr uint32_t kHeadCmdBuf = write_cmd_buf;

__attribute__((always_inline)) inline uint32_t core_xy(uint64_t noc_addr) {
    return static_cast<uint32_t>(noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

__attribute__((always_inline)) inline uint32_t record(uint32_t c) { return kCoreRecords + c * kRecordBytes; }

// Control-vector wave: read cores [lo, hi)'s tails into CV staging, then wait until `expect` responses
// have landed since `rd0`. Counted, not barriered: gather responses in flight also bump the counter,
// which can only hand a scan stale-but-valid tails, since tails are monotonic.
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

// Posted, since the barriers protect staging reuse and a head write never touches staging, and on the
// read NoC: on the egress NoC this small packet queues behind frame data and head visibility then
// inherits the PCIe tile's acceptance jitter.
__attribute__((always_inline)) inline void post_heads(const uint64_t* core_noc, uint32_t c) {
    noc_wwrite_with_state<DM_DEDICATED_NOC, kHeadCmdBuf, CQ_NOC_SNdl, CQ_NOC_SEND, CQ_NOC_WAIT, true, true>(
        kReadNoc, record(c) + kHeadWord * 4u, core_xy(core_noc[c]), 0);
}

// A frame occupies whole socket pages on the wire.
__attribute__((always_inline)) inline uint32_t page_round(uint32_t bytes) {
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

// Per-core NoC address of the profiler control block, computed once: the sweep is instruction-stream
// bound, and get_noc_addr's coordinate arithmetic would otherwise run at every issue site.
static uint64_t core_noc[kMaxCores];
static uint32_t ring_base;  // lane 0's ring on every worker: the control block plus its control words

// Stage a batch's frames, one core per staging slot from `sl`: write the prefix and control words
// locally, then gather-read each live run straight to its packed wire offset. The pads bring each
// destination to its ring phase, so read src == dst (mod 16 B) holds for every piece, a wrap split's
// continuation included. Returns the smallest per-core peak lane take.
__attribute__((noinline)) uint32_t issue_batch(const uint8_t* cores, uint32_t n, uint32_t slot, uint32_t rb) {
    uint32_t min_peak = ~0u;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = cores[i];
        const tt_l1_ptr uint32_t* __restrict tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c));
        volatile tt_l1_ptr uint32_t* __restrict cv =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
        // The head advance is staged here, hidden behind the NIU's acceptance of the same lane's
        // gather read; nothing reads the record between issue and the batch barrier.
        volatile tt_l1_ptr uint32_t* __restrict heads =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record(c) + kHeadWord * 4u);
        uint32_t off = kPrefix + kWireCtrl;
        uint32_t peak = 0;
        while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, core_xy(core_noc[c]));
        // A loop, unlike the scan: lane r's bookkeeping hides behind lane r-1's NIU acceptance,
        // which unrolling front-loads against every issue.
        for (uint32_t r = 0; r < kNumRisc; r++) {
            const uint32_t tail = tails[r];
            const uint32_t head = heads[r];
            const uint32_t take = tail - head;
            const uint32_t start = head;
            heads[r] = head + take;
            if (take > peak) {
                peak = take;
            }
            cv[kernel_profiler::SPSC_WIRE_HEAD_0 + r] = start;
            cv[kernel_profiler::SPSC_WIRE_TAIL_0 + r] = tail;
            if (take == 0) {
                continue;
            }
            const bool img = kernel_profiler::spsc_span_wrap_image(start, take, kRingWords);
            off += kernel_profiler::spsc_span_pack_pad(img ? 0u : start, off);
            const uint32_t ring_src = rb + r * (kRingWords * 4u);
            const uint32_t hm = start & (kRingWords - 1u);
            if (img) {
                // A near-full wrapping run ships as its whole ring image in one read; the decoder
                // linearises by head with the same predicate. Never coalesce adjacent ring images
                // into one read: at five rings per read it starves the producer's L1 port to ~70x
                // the stall floor.
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src, slot + off * 4u, kRingWords * 4u);
                off += kRingWords;
            } else if (hm + take > kRingWords) {
                // A small wrapping run ships as a byte-exact two-piece split: the image's dead
                // remainder would be most of the ring, and at sustained rates the drain, not the
                // sweep, binds.
                const uint32_t first = kRingWords - hm;
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, first * 4u);
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src, slot + (off + first) * 4u, (take - first) * 4u);
                off += take;
            } else {
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, false, false>(
                    kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                off += take;
            }
        }
        cv[kernel_profiler::SPSC_WIRE_XY] = tails[kXyWord];
        // pfx[0] is constant and staged once at init; only the payload word varies.
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        pfx[1] = off - kPrefix;
        if (peak < min_peak) {
            min_peak = peak;
        }
        slot += kSlotBytes;
    }
    return min_peak;
}

void kernel_main() {
    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // profiler_msg_t base on every worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t xy = coords[i];
        core_noc[i] = get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src);
    }
    // Resync the software NoC counter mirrors from hardware: they persist across launches on this
    // never-reset core and firmware only initialises them at boot, so a previous run that ended with
    // unacked writes would wedge this run's first barrier.
    noc_local_state_init(NOC_INDEX);
    noc_local_state_init(kReadNoc);
    ring_base = cv_src + kCtrlWords * 4u;
    // Both read buffers tag their requests with transaction id 0: the batch barrier is the NIU's
    // outstanding count for that id. The return coordinate is this NIU's own, programmed into
    // read_cmd_buf by firmware.
    while (!noc_cmd_buf_ready(kReadNoc, read_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(kReadNoc, read_cmd_buf, NOC_PACKET_TAG, 0);
    // Worker L1 addresses have no bits above 32, so the gather's address-mid word is zero for every
    // core and a core's set_state is its coordinate alone.
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

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);
    // Egress write command state, programmed once: nothing else on this core touches write_cmd_buf on
    // the egress NoC.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    // The host's launch check polls this: a DRISC that never leaves reset would otherwise wedge every
    // producer on a full ring with no error anywhere.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    *hb = 0;

    // Every frame's prefix is identical, and of the control words only heads, tails and the core
    // identity are staged per frame; the rest must read zero on the wire.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix + kWireCtrl; k++) {
            pfx[k] = 0;
        }
    }

    // Statics persist across launches on this core, so everything the loop trusts is re-initialised
    // explicitly. tails_seen is the sum of a core's five tails at its last scan; tails are monotonic,
    // so its delta is the words produced in one service interval.
    static uint32_t tails_seen[kMaxCores];
    static uint8_t hot[kMaxCores];        // shipped real words last scan; hot + empty scan = publish lag
    static uint8_t ship_list[kMaxCores];  // this sweep's ship set, dense core indices
    for (uint32_t i = 0; i < num_cores; i++) {
        hot[i] = 0;
    }
    // Seed the heads from the tails as they stand now: everything published before this launch predates
    // the capture. The scratch is the only copy of the heads: the scan reads it, the issue advances it,
    // and the posted head write ships it.
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

    uint32_t relieved = 0;
    uint32_t sweeps = 0;
    uint32_t gap = 0;
    // Ship-threshold arming. Batching must never hold pre-burst trickle across a burst onset, since a
    // pre-loaded ring tips over during the detection latency, and occupancy alone cannot tell one-shot
    // trickle from a light workload's steady sub-threshold lanes. Growth persistence can: defer only
    // after kBatchArmSweeps consecutive growing sweeps, flush after kFlushQuietSweeps dead ones.
    bool grid_busy = false;
    // Saturation shortcut: once a sweep shipped every core with real words above the ship threshold, the
    // scan is pure overhead and the first batch's tails age through it, so the whole grid ships in list
    // order off the per-batch refreshes until a core comes back empty or under the threshold.
    bool all_live = false;
    uint32_t grow_streak = 0;
    uint32_t quiet_streak = 0;
    constexpr uint32_t kBatchArmSweeps = 3;
    constexpr uint32_t kFlushQuietSweeps = 8;
    // A head relief is one posted write; re-posting every core's head on this cadence bounds a
    // lost one to a stall instead of parking the producer for the rest of the run.
    constexpr uint32_t kHeadRefreshSweeps = 64;
    static_assert((kHeadRefreshSweeps & (kHeadRefreshSweeps - 1u)) == 0, "the refresh cadence is a mask");
    // Which staging generations may still have a ship in flight. Persists across sweeps so a sweep's
    // final ship drains under the pace gap or the next CV pass, not on its own critical path.
    bool gen_shipped[kNGens] = {};

    uint32_t gen_dma_mark[kNGens] = {};
    SpoolPump<kSpoolBase, kSpoolBytes, kBounceBase0, kBounceBytes, kPageBytes, kDmaShip, kDmaDrain> pump(sender);
    bool killed = false;  // the kill switch (stop=2) broke a wait: the consumer is gone, bytes are stranded

    // Ship `count` adjacent staged slots. A staged slot is already its frame's wire image, so a frame is
    // one write (or one DMA), and the trailing page fill is never written: the host derives every offset
    // from the control vector and reads past it.
    auto emit_slots = [&](uint32_t base, uint32_t count) __attribute__((always_inline)) {
        // A frame's length is its own prefix word: payload words behind the fixed prefix. Frames occupy
        // whole pages on the wire, dead tail bytes included, so the FIFO write pointer and the spool
        // offset advance in lockstep and the drain needs no frame geometry.
        const uint32_t raw0 = (reinterpret_cast<const tt_l1_ptr uint32_t*>(base)[1] + kPrefix) * 4u;
        const uint32_t len0 = page_round(raw0);
        uint32_t raw1 = 0;
        uint32_t len1 = 0;
        if (count == kGenSlots) {
            raw1 = (reinterpret_cast<const tt_l1_ptr uint32_t*>(base + kSlotBytes)[1] + kPrefix) * 4u;
            len1 = page_round(raw1);
        }
        const uint32_t bytes = len0 + len1;
        if constexpr (kSpool) {
            // Full spool: pump until there is room. This wait, not a drop, is the spool's
            // back-pressure; it holds through quiesce and only the kill switch breaks it.
            while (!pump.has_room(bytes)) {
                if (*stop == kernel_profiler::kRelayStopRelease) {
                    killed = true;
                    return;
                }
                invalidate_l1_cache();
                pump.pass();
            }
            // The DMA engine reads the control and length words the scalar core staged; Blackhole
            // stores can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            // A full-span frame fills its slot exactly, so a full first frame and its neighbour are
            // wire-contiguous in staging and ship as one DMA write.
            if (len1 != 0 && len0 != kSlotBytes) {
                pump.append(base, len0);
                pump.append(base + kSlotBytes, len1);
            } else {
                pump.append(base, bytes);
            }
            pump.rebalance();
        } else {
            asm volatile("fence" ::: "memory");
            if (!reserve_pages(sender, bytes / kPageBytes, stop)) {
                killed = true;
                return;
            }
            const uint32_t fifo_size = sender.downstream_fifo_curr_size;
            uint32_t wr = sender.write_ptr;
            push_fifo(sender, base, wr, raw0);
            if (len1 != 0) {
                wr += len0;
                if (wr >= fifo_size) {
                    wr -= fifo_size;
                }
                push_fifo(sender, base + kSlotBytes, wr, raw1);
            }
            socket_push_pages(sender, bytes / kPageBytes);
            notify_bytes_sent(sender);
        }
    };

    // Main loop. On stop=1, keep sweeping until one whole sweep moves nothing, so markers still in
    // worker rings ship instead of being stranded.
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
        // Software pipeline: gather generation G on the read NoC while G^1 ships on the egress side.
        // The CV pass is pipelined into the batch flights: all tail reads issue up front, the wait
        // covers only the first chunk's responses, and the rest of the grid is scanned just in time
        // when the ship list runs low. Keep the scan out of a lambda: a by-reference capture costs
        // sweep time that the saturation boundary amplifies.
        uint32_t gen = 0;
        uint32_t gen_base = kStageBase;
        uint32_t pend_n = 0;
        uint32_t pend_gen = 0;
        uint32_t pend_base = kStageBase;
        bool have_pend = false;
        uint32_t n_ship = 0;
        uint32_t min_peak = ~0u;

        // Heads go out the moment the batch's read barrier passes, not with the frame
        // emit: the payload is resident in staging once the reads land, so the producer's
        // ring slots are free regardless of when the frame reaches the host.
        auto advance_heads = [&](uint32_t n, const uint8_t* cores) __attribute__((always_inline)) {
            for (uint32_t i = 0; i < n; i++) {
                post_heads(core_noc, cores[i]);
                relieved++;
            }
        };

        auto ship_frames = [&](uint32_t n, uint32_t g, uint32_t base) __attribute__((always_inline)) {
            emit_slots(base, n);
            if constexpr (kSpool) {
                gen_dma_mark[g] = pump.dma_issued;
            }
            gen_shipped[g] = true;
        };

        // Staging reuse: generation g's previous frame must be out of staging before its slots
        // refill. gen_shipped persists across sweeps, so a sweep's last ship is never waited on
        // inside its own sweep.
        auto retire_gen = [&](uint32_t g) __attribute__((always_inline)) {
            if (gen_shipped[g]) {
                // Bare waits: both predicates complete on this device alone (the DMA engine's
                // writes to GDDR, the NIU's sent counter), so no consumer state can hang them.
                if constexpr (kSpool) {
                    // This generation's ship writes only: stream completion is FIFO, so
                    // outstanding <= later-issues means this generation retired.
                    const uint32_t since = pump.dma_issued - gen_dma_mark[g];
                    const uint32_t cap = since > kDmaOutstandingMax ? kDmaOutstandingMax : since;
                    while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
                    }
                } else {
                    // Sent-only is legal here because the staging slots' next writer is this
                    // core's own NIU read responses.
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
            // Responses can arrive out of order, so a counted response may belong to a later core;
            // that core then scans last sweep's tails, stale but valid because tails are monotonic,
            // under-ships, and catches up next visit.
            cv_issue(core_noc, 0, num_cores);
            // The previous sweep's last ship is still retiring out of the generation this sweep's first
            // batch refills. Its wait rides under the tail reads' round trip here, not between those
            // tails landing and the first batch's issue: the batch whose tails age the most is the batch
            // whose producers stall.
            retire_gen(gen);
            cv_wait(rd0, num_cores);
            for (uint32_t c = 0; c < num_cores; c++) {
                const tt_l1_ptr uint32_t* __restrict tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(record(c));
                const tt_l1_ptr uint32_t* __restrict mine = tails + kHeadWord;
                // Unrolled into registers: a loop over indexed arrays spills on this core, and each
                // spilled word is an L1 round trip per core per sweep.
                const uint32_t d0 = tails[0] - mine[0];
                const uint32_t d1 = tails[1] - mine[1];
                const uint32_t d2 = tails[2] - mine[2];
                const uint32_t d3 = tails[3] - mine[3];
                const uint32_t d4 = tails[4] - mine[4];
                // No clamp: a producer blocks 506 words past the head it sees, and the mirror is
                // never behind that head, so no lane's diff can exceed the ring.
                const uint32_t live = d0 | d1 | d2 | d3 | d4;
                uint32_t grew = 0;
                uint32_t peak = 0;
                // With the ship gate open every live core ships, so nothing downstream reads the peak
                // or the growth.
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
                if (live == 0) {
                    // A hot core scanning empty is almost always the producer's 64-word batched tail
                    // publish, not idleness, and skipping it would hand the core a two-sweep service
                    // interval. One-shot: a genuinely idle core wastes at most one empty frame before
                    // going cold.
                    if (hot[c] == 0) {
                        continue;
                    }
                    hot[c] = 0;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                    continue;
                }
                // Deferral must be safe against one more service interval of production, which the
                // level alone cannot promise: a core scanned just under the threshold at a high rate
                // blows the ring-fill margin two sweeps later. Requiring `grew`, the words produced in
                // the last interval, under the threshold too bounds a deferred core at ~2x threshold.
                if (grid_busy && stop_seen_at == 0 && peak < kLaneShipWords && grew < kLaneShipWords &&
                    peak < kLaneTrigger) {
                    continue;
                }
                hot[c] = 1;
                ship_list[n_ship++] = static_cast<uint8_t>(c);
            }
        }

        // One ship site: the last batch's frames leave through the same code as every other batch's,
        // on the pass that finds nothing left to issue.
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
                // Refresh the next batch's tails in the same flight: on the sweep-start snapshot
                // alone the last cores would be served a sweep stale. This generation's read barrier
                // covers these reads too. A full list's last batch refreshes the next sweep's first
                // batch instead, so a saturated sweep opens on the issue with no wave of its own.
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
                ship_frames(pend_n, pend_gen, pend_base);
                have_pend = false;
            }
            if (!more) {
                break;
            }
            if constexpr (kSpool) {
                if (pump.level >= 3u) {
                    pump.pass();
                }
            }

            // Read barrier before the heads go out, hardware-counted: every read this core issues
            // carries transaction id 0, so the NIU's per-id outstanding count is the barrier and no
            // software count rides in the lane loop. The spin doubles as the pump's slot, but only at
            // full pressure: below it the pump's GDDR reads contend with the ship DMA and the gathers.
            while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_REQS_OUTSTANDING_ID(0)) != 0) {
                if constexpr (kSpool) {
                    // Level 3 means occupancy is over the 5/8 line, so nonempty holds.
                    if (pump.level >= 3u) {
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
        // Enter only when the scan would have shipped every core anyway: all on the list, none empty,
        // none under the ship threshold. A core dropping below either leaves the mode.
        const bool next = n_ship == num_cores && min_peak != 0 && min_peak >= kLaneShipWords;
        if (next && !all_live) {
            for (uint32_t c = 0; c < num_cores; c++) {
                ship_list[c] = static_cast<uint8_t>(c);
            }
        }
        all_live = next;

        // Every issued batch has passed its read barrier by here, so each core's scratch is exactly
        // the head it was last relieved to.
        if ((sweeps & (kHeadRefreshSweeps - 1u)) == 0) {
            for (uint32_t c = 0; c < num_cores; c++) {
                post_heads(core_noc, c);
            }
        }
        // Busy sweeps below the first band skip the post-sweep pump entirely: the spool is the burst
        // absorber, and a capture that fits in it deserves pure gather.
        if constexpr (kSpool) {
            if (pump.level >= 2u || (pump.level == 1u && (sweeps & 1u) != 0) || pump.fresh_boost ||
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
        // Idle pacing: collapse on work, creep toward the ceiling when idle. Live-but-untriggered lanes
        // count as work: a head only reaches a producer on a ship, so sleeping while lanes fill toward
        // the trigger blinds the relay.
        if (relieved != relieved_at_sweep_start || sweep_peak >= kCvBusyPeak) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < 256u) {
                inc = 256u;
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

    // Everything the run spooled must reach the host FIFO before the socket barrier can pass; bounded,
    // so a consumer that stopped acking strands bytes instead of wedging teardown.
    if constexpr (kSpool) {
        while (!pump.drained()) {
            pump.pass_cold();
            // Notify per pass, not per sweep: with a host FIFO smaller than the backlog, the
            // acks that free credit only come after the host has seen the bytes.
            pump.notify();
            // The host's teardown escalates stop to 2 after its own timeout: the close path's kill
            // switch for a drain whose consumer will never finish it.
            invalidate_l1_cache();
            if (*stop == kernel_profiler::kRelayStopRelease) {
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
    // The posted head write-backs are outside that barrier's predicate, so drain their sent counter
    // too and leave no unstreamed head behind.
    const uint64_t t_ps = get_timestamp() + kPostedDrainCycles;
    while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
           get_timestamp() < t_ps) {
    }
    // Written back only for a live consumer: after an abandoned batch the socket's view of bytes_sent is
    // already out of sync with the host's, and the socket is being torn down either way.
    if (!killed) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = kernel_profiler::kRelayDoneWord;

    // NIU restore, on the host's word. NIU_CFG_0 persists until chip reset, so whoever set stream mode
    // owns putting it back, and it goes last because the flip to NOC2AXI takes this L1 (`done`, the
    // results, bytes_acked) out of the host's view.
    const uint64_t t_end = get_timestamp() + kNiuRestoreWaitCycles;
    while (*stop != kernel_profiler::kRelayStopRelease && get_timestamp() < t_end) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
