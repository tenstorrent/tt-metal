// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Support spine of the DRISC drain kernel (drisc_profiler_filler.cpp): the D2H write/credit primitives,
// the poll-free GDDR DMA issue variants and the GDDR spool pump.
#pragma once

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

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. The caller runs
// noc_write_init_state<write_cmd_buf>(NOC_INDEX, vc) once per push -- a push makes several calls (one
// per frame plus the notify), so a per-call init would repeat command-buffer setup that nothing between
// the calls invalidates. Alternating two command buffers to pipeline NIU acceptance was measured
// stall-neutral at the delay-16 saturation wall and deleted.
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

// Write `len` bytes of L1 at `src` to FIFO offset `dst`, splitting a piece that crosses the FIFO wrap --
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
// the mesh can then deliver the bytes_sent word ahead of the data it announces. Same command state, VC
// and route as the data makes delivery order the issue order again.
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
// escape. Keeps waiting through quiesce (stop=1): the receiver is still acking then, and returning would
// lose frames whose heads were already relieved. Only the host's kill switch (stop=2, written after its
// own teardown timeout) returns false.
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
            if (*stop == 2u) {
                return false;
            }
        }
        acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
    }
    return true;
}

// dma_async_write/read (gddr_dma.h) poll the engine's ready status before every issue -- an MMIO round
// trip plus a per-iteration stack bounce (the volatile union: lw/sw/lw/andi/beqz in the compiled ELF).
// The poll guards the queue-full case, which the filler's pipeline makes unreachable by construction:
// the generation gate keeps stream-0 writes at most ~10 of the 15 the queue holds, and the pump keeps
// stream-1 reads at most 2 of 255. These variants are the same register programming with the poll gone.
// Forced inline: at three issue sites -Os outlines these four register writes into a call.
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

// ---- GDDR spool ---------------------------------------------------------------------------------------
//
// The ship DMA (TX stream kShip) appends page-rounded frames to a ring in this DRISC's own GDDR bank, and
// pass() forwards them to the host FIFO through two L1 bounce buffers refilled by TX stream kDrain. A pass
// never spins: every wait this stage could have is a state a later pass observes, so the pump can delay
// host delivery but never the sweep. Byte counters are monotonic 64-bit (long captures exceed 4 GiB); ring
// offsets are kept incrementally so the hot path never takes a runtime modulo. The direct-push build
// instantiates it with kBytes == 0 and never calls it.
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
        uint32_t off;         // bytes already pushed to the host (partial ships under credit)
        uint32_t ack_target;  // write-ack mirror at ship: this bounce's flush line
        uint32_t seq;         // refill order, so a both-ready pass ships the older bytes first
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
    // Pump effort, a pure function of spool occupancy. 0: idle sweeps and the pace gap only.
    // 1: one post-sweep pass every other sweep. 2: post-sweep every sweep. 3: also inline per-batch
    // and in the read-wait spin. Graduated, not bang-bang: one step is well under a microsecond,
    // where a single engage/release threshold added +2.4 us to sweeps already near ring-full.
    uint32_t level = 0;
    // The freshness deadline wants at least a per-sweep trickle. Latched: a light workload that
    // never reaches the occupancy bands degrades into a permanent trickle after the first deadline,
    // which is the "reasonably real-time" host stream; cleared when the spool empties.
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

    // Occupancy changes only where wr or rd advance (append and pass), so the level is maintained at
    // those two sites instead of being recomputed from two u64s per batch.
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

    // One pass. Inlined at every call site on purpose: one out-of-line copy measured 15% more d1
    // stalls (233k vs 198k), and the six copies fit the DRISC code region with ~50 B to spare --
    // kernel growth pays here first.
    __attribute__((always_inline)) void pass() {
        // L1-only early-out so idle passes cost no DMA or NIU register reads.
        if (wr == rd && b[0].state == kEmpty && b[1].state == kEmpty) {
            return;
        }
        bool did = false;
        // SHIPPING -> EMPTY once the egress writes are acked. Full flush, not "sent": the
        // bounce's next writer is the DMA engine, which a sent-only gate does not fence. Most
        // passes make no progress, so every NIU register poll below is gated on the L1 state
        // that could consume it -- an idle pass costs loads the core already has in hand.
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
        // Refill an empty bounce before shipping the ready one, so the read runs under the
        // ship's NoC issue. The second concurrent refill only at full pressure: at a burst the
        // extra in-flight GDDR read deepens the bank queue exactly when the ship DMA needs it.
        const uint32_t emp = b[0].state == kEmpty ? 0u : (b[1].state == kEmpty ? 1u : 2u);
        const bool want_refill = emp != 2u && (level >= 3u || b[emp ^ 1u].state != kReading);
        // Only ship-completed bytes are readable: nothing short of a ship write's completion
        // orders a drain read of the same address behind it. Advanced lazily: polled only when a
        // refill could consume more than the window it already sees.
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
        // Ship a READY bounce, as much as the host FIFO has credit for right now. Partial ships
        // keep the FIFO fed under credit pressure. Oldest first when both are ready: the socket
        // is a byte stream and the younger bounce would reorder the wire. No per-ship command
        // init and no per-ship notify -- together most of a shipping pass's cost.
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

    // One out-of-line copy of the pass for the two sites that never run inside a loaded sweep (the
    // idle gap and the exit drain); the inlined copies at the other sites are worth 15% of d1 stalls.
    __attribute__((noinline)) void pass_cold() { pass(); }

    // The batched bytes_sent notify for pump ships: once per sweep instead of once per chunk.
    __attribute__((always_inline)) void notify() {
        if (notify_pending) {
            notify_bytes_sent(sender_);
            notify_pending = false;
        }
    }

    // Once per sweep. The wall clock is read only every 64th call, because even one read per sweep
    // measurably stalls producers at the saturation boundary (~1 ms of stride is noise against the
    // 50 ms bound). A workload too light to reach the occupancy bands would otherwise sit in the
    // spool for seconds; the deadline escalates it to the latched per-sweep trickle, never to
    // inline pumping -- freshness is a latency bound, not a pressure emergency.
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
