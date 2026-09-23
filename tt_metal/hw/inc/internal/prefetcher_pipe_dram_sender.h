// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe sender helpers for programmable DRAM cores (Blackhole DRISCs).
//
// The device-side experimental::PrefetcherPipe class cannot be used here: its constructor reads the
// launch message to find its config slot, and DRAM cores are never dispatched to, so they have no
// launch message and never binds a Program slot. Instead the host stamps a complete sender config
// page into DRISC L1 (see impl/buffers/prefetcher_pipe.cpp, initialize_dram_sender_config_page) and
// this header works directly against that page. Deliberately depends only on headers a DRISC kernel already builds
// with -- notably NOT prefetcher_pipe.h or prefetcher_pipe_init.h.
//
// The wire protocol is identical to the worker-sender PrefetcherPipe, so an ordinary receiver
// (a ProgramSpec/ProgramRunArgs binding + the device PrefetcherPipe class) is the consumer:
//
//   * Credits are counted in L1_ALIGNMENT-byte units. A config page holds two credit blocks --
//     SENT (word[7]) and ACKED (word[8]) -- each one L1_ALIGNMENT slot per receiver, on both the
//     DRISC side and the receiver side. The two blocks are kept in separate cache lines so a
//     core's cached stores to its own counters can never write back over the peer's NoC-written
//     ones; see remote_dfb_config_layout.h.
//   * A receiver's write cursor is stored in the padding word of its SENT slot
//     (PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD) and advanced whenever that receiver is credited, the
//     same durable field the worker-sender path keeps it in. It lives in the config page, so it
//     survives across programs; it is not derived from entries_sent because that counter's 2^32
//     wrap only preserves a (sent % ring_units) derivation for a power-of-two ring.
//   * The receivers' pages sit at their own L1 address, not this page's, so the base of their SENT
//     block travels as a page-relative delta in word[9] (PREFETCHER_PIPE_CFG_PEER_COUNTER_OFFSET).
//     Their acks come back to this page's ACKED block, which each receiver page names the same way.
//   * Credit increments and payload writes ride the same NOC VC, so a drained NIU implies the
//     payload landed before the credit the receiver observes.
//
// An entry size need not divide the ring. When it does not, the last `ring_bytes % entry_bytes`
// bytes are a trailing gap that holds no entry: writes stop at the page-aligned usable limit, and
// the wrap credits the gap along with the entry that reaches it. That keeps a lap worth exactly
// ring_bytes of credit, which is what lets a cursor advanced by credited units come back to the
// ring base on the same lap as the receiver's read pointer. This is the same trailing-gap term the
// worker-sender path uses.
//
// The one rule a caller must keep: a write of n entries must not straddle the usable limit,
// matching the contiguous-write rule the worker-sender path also enforces.

#pragma once

#include <cstdint>

#include "hostdev/remote_dfb_config_layout.h"
#include "internal/dram_sender_credit_counters.h"
#include "internal/risc_attribs.h"

namespace experimental {

// Working copy of a DRAM-sender PrefetcherPipe endpoint, loaded from its DRISC-L1 config page.
struct PipeSenderCtx {
    uint32_t config_ptr;           // the config page itself, in DRISC L1
    uint32_t fifo_start_addr;      // ring base, in receiver (worker) L1
    uint32_t ring_bytes;           // ring size in bytes; need not be a multiple of entry_bytes
    uint32_t entry_bytes;          // push granularity
    uint32_t num_receivers;        // receivers this sender core drives
    uint32_t receiver_noc_xy_ptr;  // -> 2 * num_receivers words of receiver NOC XY
    uint32_t local_sent_base;      // DRISC-side SENT block (word[7])
    uint32_t local_acked_base;     // DRISC-side ACKED block (word[8]), the receivers' ack target
    uint32_t remote_sent_base;     // receiver-side SENT block (word[9])
};

// Base of the ACKED block on a sender's config page: where this pipe's receivers aim their ack
// atomics, and so what the sender reads them from. Takes the page address rather than a
// PipeSenderCtx so a caller holding only that -- the prefetcher's stop-sentinel drain, which reads
// it off the last loaded interface -- does not have to rebuild a whole context.
FORCE_INLINE uint32_t pipe_local_acked_base(uint32_t config_page_addr) {
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
    return config_page_addr + cfg[PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET];
}

FORCE_INLINE void pipe_load_sender_ctx(PipeSenderCtx& ctx, uint32_t config_page_addr) {
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
    ASSERT(static_cast<bool>(cfg[REMOTE_DFB_CFG_IS_SENDER]));

    ctx.config_ptr = config_page_addr;
    ctx.num_receivers = cfg[REMOTE_DFB_CFG_NUM_RECEIVERS];
    ctx.fifo_start_addr = cfg[REMOTE_DFB_CFG_FIFO_START];
    ctx.ring_bytes = cfg[REMOTE_DFB_CFG_FIFO_SIZE];
    ctx.entry_bytes = cfg[PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE];
    ctx.receiver_noc_xy_ptr = config_page_addr + cfg[PREFETCHER_PIPE_CFG_NOC_XY_OFFSET];
    ctx.local_sent_base = config_page_addr + cfg[PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET];
    ctx.local_acked_base = pipe_local_acked_base(config_page_addr);
    ctx.remote_sent_base = config_page_addr + cfg[PREFETCHER_PIPE_CFG_PEER_COUNTER_OFFSET];

    ASSERT(ctx.entry_bytes != 0);
    ASSERT(ctx.entry_bytes % L1_ALIGNMENT == 0);
    ASSERT(ctx.entry_bytes <= ctx.ring_bytes);
}

FORCE_INLINE uint32_t pipe_ring_units(const PipeSenderCtx& ctx) { return ctx.ring_bytes / L1_ALIGNMENT; }

FORCE_INLINE uint32_t pipe_units_per_entry(const PipeSenderCtx& ctx) { return ctx.entry_bytes / L1_ALIGNMENT; }

// Bytes of the ring that hold whole entries. The remainder is the trailing gap: no entry starts
// there, and it is credited as padding by whichever write reaches this limit.
FORCE_INLINE uint32_t pipe_usable_bytes(const PipeSenderCtx& ctx) {
    return ctx.ring_bytes - ctx.ring_bytes % ctx.entry_bytes;
}

// This sender's entries_sent counter for receiver r. A DRAM-sender pipe is single-lane, so each
// block holds exactly one L1_ALIGNMENT slot per receiver.
FORCE_INLINE volatile tt_l1_ptr uint32_t* pipe_local_sent_ptr(const PipeSenderCtx& ctx, uint32_t r) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.local_sent_base + r * L1_ALIGNMENT);
}

// Receiver r's NOC address encoding, decoded from the config page's XY table.
FORCE_INLINE uint32_t pipe_receiver_noc_xy(const PipeSenderCtx& ctx, uint32_t r, uint8_t noc) {
    volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.receiver_noc_xy_ptr);
    return uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, xy[2 * r]), DYNAMIC_NOC_Y(noc, xy[2 * r + 1])));
}

// Receiver r's write cursor, in the padding word of its SENT slot. A credit slot is a whole
// L1_ALIGNMENT because it is a NOC-atomic target; the cursor rides the padding that alignment
// already reserves, which is both where the worker-sender path keeps it and inside the range a
// credit reset zeroes, so credits and cursors can never reset out of step.
FORCE_INLINE volatile tt_l1_ptr uint32_t* pipe_local_wr_offset_ptr(const PipeSenderCtx& ctx, uint32_t r) {
    return pipe_local_sent_ptr(ctx, r) + PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD;
}

// Byte offset into the ring where receiver r's next entry goes. Matches the worker-sender path's
// sender_wr_offset().
FORCE_INLINE uint32_t pipe_sender_wr_offset(const PipeSenderCtx& ctx, uint32_t r) {
    return *pipe_local_wr_offset_ptr(ctx, r);
}

// Advance receiver r's cursor by the units just credited to it. Payload and any trailing gap are
// both in `units` (pipe_units_for_write), so a lap is exactly the full ring and the contiguous-write
// rule caps one credit at a lap: one conditional subtract is enough. Mirrors the worker-sender
// path's advance_wr_offset().
FORCE_INLINE void pipe_advance_wr_offset(const PipeSenderCtx& ctx, uint32_t r, uint32_t units) {
    volatile tt_l1_ptr uint32_t* offset_ptr = pipe_local_wr_offset_ptr(ctx, r);
    uint32_t next = *offset_ptr + units * L1_ALIGNMENT;
    ASSERT(next <= ctx.ring_bytes);
    if (next >= ctx.ring_bytes) {
        next -= ctx.ring_bytes;
    }
    *offset_ptr = next;
}

// Publish `wr_offset` as every receiver's cursor, given the base of a sender's SENT block. Takes
// that base rather than a PipeSenderCtx so the prefetcher's receiver-contiguous loop -- which
// keeps one working cursor for a whole round in its RemoteSenderCBInterface, because it credits
// every receiver the same bytes -- can put the round's result back without rebuilding the context.
FORCE_INLINE void pipe_store_wr_offset(uint32_t local_sent_base, uint32_t num_receivers, uint32_t wr_offset) {
    volatile tt_l1_ptr uint32_t* offset_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sent_base) + PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD;
    for (uint32_t r = 0; r < num_receivers; ++r) {
        *offset_ptr = wr_offset;
        offset_ptr += L1_ALIGNMENT / sizeof(uint32_t);
    }
}

// Free credit units (L1_ALIGNMENT-sized) at the most-backed-up receiver, without blocking. Lets a
// batching caller size its next round.
FORCE_INLINE uint32_t pipe_poll_min_free_units(const PipeSenderCtx& ctx) {
    return dram_sender_min_free_units(
        ctx.local_sent_base, ctx.local_acked_base, L1_ALIGNMENT, ctx.num_receivers, pipe_ring_units(ctx));
}

// Spin until every receiver can take num_entries more entries. The requirement is converted to
// credit units once so the spin body carries no division, and includes the trailing gap because a
// write that reaches the usable limit credits it too -- reserving the worst case here rather than
// each receiver's actual cursor keeps the spin a single comparison.
FORCE_INLINE void pipe_reserve_back(const PipeSenderCtx& ctx, uint32_t num_entries) {
    const uint32_t needed_units =
        num_entries * pipe_units_per_entry(ctx) + (ctx.ring_bytes - pipe_usable_bytes(ctx)) / L1_ALIGNMENT;
    while (pipe_poll_min_free_units(ctx) < needed_units) {
    }
}

// Publish `units` L1_ALIGNMENT-sized credit units to one receiver: bump this core's local counter,
// advance that receiver's write cursor by the same units, and NOC-inc the receiver's mirror of the
// counter. Crediting is what moves the cursor, so call this only after the payload writes are
// flushed. Every write path publishes through here, which is what keeps the cursor in step with the
// credits -- a resize's pad credits move it by exactly the bytes they skip.
FORCE_INLINE void pipe_credit_receiver(const PipeSenderCtx& ctx, uint32_t r, uint32_t units, uint8_t noc) {
    if (units == 0) {
        return;
    }
    const uint32_t remote_noc_xy = pipe_receiver_noc_xy(ctx, r, noc);
    const uint32_t remote_sent_ptr = ctx.remote_sent_base + r * L1_ALIGNMENT;
    *pipe_local_sent_ptr(ctx, r) += units;
    pipe_advance_wr_offset(ctx, r, units);
    // Posted, matching the worker-sender path: receivers discover credit by polling, and this
    // core observes their acks the same way.
    noc_semaphore_inc</*skip_ptr_update=*/true>(get_noc_addr_helper(remote_noc_xy, remote_sent_ptr), units, noc);
}

// Credit units a write of `num_entries` starting at `wr_offset` publishes: the payload, plus the
// trailing gap when the write reaches the usable limit, so a full lap credits exactly ring_bytes
// and the cursor wraps to zero. Mirrors the worker-sender path's units_for_write().
FORCE_INLINE uint32_t pipe_units_for_write(const PipeSenderCtx& ctx, uint32_t wr_offset, uint32_t num_entries) {
    const uint32_t payload_bytes = num_entries * ctx.entry_bytes;
    const uint32_t usable = pipe_usable_bytes(ctx);
    uint32_t credited_bytes = payload_bytes;
    if (wr_offset + payload_bytes >= usable) {
        credited_bytes += ctx.ring_bytes - usable;
    }
    return credited_bytes / L1_ALIGNMENT;
}

// Publish num_entries to every receiver. Each receiver's own cursor decides whether this write
// reaches the wrap, so the gap credit is computed per receiver rather than once.
FORCE_INLINE void pipe_push_credits(const PipeSenderCtx& ctx, uint32_t num_entries, uint8_t noc) {
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        pipe_credit_receiver(ctx, r, pipe_units_for_write(ctx, pipe_sender_wr_offset(ctx, r), num_entries), noc);
    }
}

// Switch this sender to `entry_bytes` for subsequent pushes: snap every receiver's stored write
// cursor onto the new entry grid and publish the bytes it skips as pad credits. A receiver runs the
// matching snap -- PrefetcherPipe's constructor, when the Attach entry size differs from the one
// last applied -- and waits for exactly these credits, so the two endpoints stay on one grid.
// Mirrors PrefetcherPipe::resize_sender_interface<true>() step for step, including the case where
// aligning up lands in the new size's trailing gap: there the cursor wraps to zero instead, and the
// credit covers everything from the old cursor to the end of the full ring.
//
// A cursor already on the grid takes no credit, so this is idempotent: a caller pushing a run of
// same-sized tensors can call it before each one.
FORCE_INLINE void pipe_set_entry_size(PipeSenderCtx& ctx, uint32_t entry_bytes, uint8_t noc) {
    ASSERT(entry_bytes != 0);
    ASSERT(entry_bytes % L1_ALIGNMENT == 0);
    ASSERT(entry_bytes <= ctx.ring_bytes);
    ctx.entry_bytes = entry_bytes;
    const uint32_t usable = pipe_usable_bytes(ctx);
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        const uint32_t current_offset = pipe_sender_wr_offset(ctx, r);
        const uint32_t offset_into_entry = current_offset % entry_bytes;
        uint32_t adjustment_bytes = offset_into_entry == 0 ? 0u : entry_bytes - offset_into_entry;
        if (current_offset + adjustment_bytes >= usable) {
            // Aligning up reaches the usable limit: wrap to the ring base instead, crediting the
            // trailing gap along with the skipped bytes so the cursor lands back on zero.
            adjustment_bytes = ctx.ring_bytes - current_offset;
        }
        if (adjustment_bytes != 0) {
            pipe_credit_receiver(ctx, r, adjustment_bytes / L1_ALIGNMENT, noc);
        }
    }
}

// Post num_entries' worth of payload to one receiver at its stored write position, as packets of
// at most NOC_MAX_BURST_SIZE bytes. Does not touch credits, so it does not move that cursor:
// repeating a write before crediting overwrites the same slots.
FORCE_INLINE void pipe_write_to_receiver(
    const PipeSenderCtx& ctx, uint32_t r, uint32_t src_l1_addr, uint32_t num_entries, uint8_t noc) {
    const uint32_t wr_offset = pipe_sender_wr_offset(ctx, r);
    uint32_t bytes = num_entries * ctx.entry_bytes;
    // Contiguous-write rule: a write must not straddle the usable limit, past which the ring holds
    // only the trailing gap.
    ASSERT(wr_offset + bytes <= pipe_usable_bytes(ctx));

    const uint32_t remote_noc_xy = pipe_receiver_noc_xy(ctx, r, noc);
    uint64_t dst = get_noc_addr_helper(remote_noc_xy, ctx.fifo_start_addr + wr_offset);
    while (bytes != 0) {
        const uint32_t packet_bytes = bytes < NOC_MAX_BURST_SIZE ? bytes : NOC_MAX_BURST_SIZE;
        noc_async_write_one_packet</*enable_noc_tracing=*/false, /*posted=*/true>(src_l1_addr, dst, packet_bytes, noc);
        src_l1_addr += packet_bytes;
        dst += packet_bytes;
        bytes -= packet_bytes;
    }
}

// Spin until every receiver has acked everything this core has sent.
FORCE_INLINE void pipe_sender_barrier(const PipeSenderCtx& ctx) {
    dram_sender_barrier(ctx.local_sent_base, ctx.local_acked_base, L1_ALIGNMENT, ctx.num_receivers);
}

}  // namespace experimental
