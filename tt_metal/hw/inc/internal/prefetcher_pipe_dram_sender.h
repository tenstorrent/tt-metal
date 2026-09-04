// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe sender helpers for programmable DRAM cores (Blackhole DRISCs).
//
// The device-side experimental::PrefetcherPipe class cannot be used here: its constructor reads the
// launch message to find its config slot, and DRAM cores are never dispatched to, so they have no
// launch message and never Attach. Instead the host stamps a complete PrefetcherPipe sender config
// page into DRISC L1 (see impl/buffers/prefetcher_pipe.cpp, initialize_dram_sender_config_page) and
// this header works directly against that page. Deliberately depends only on headers a DRISC kernel already builds
// with -- notably NOT prefetcher_pipe.h or prefetcher_pipe_init.h.
//
// The wire protocol is identical to the worker-sender PrefetcherPipe, so an ordinary receiver
// (AttachPrefetcherPipe + the device PrefetcherPipe class) is the consumer:
//
//   * Credits are counted in L1_ALIGNMENT-byte units. Each receiver owns a counter pair --
//     entries_sent at +0, entries_acked at +L1_ALIGNMENT -- and pairs are strided by
//     2 * L1_ALIGNMENT, both locally in DRISC L1 and on the receiver side.
//   * A receiver's write cursor is *derived* from its entries_sent counter rather than stored,
//     which is what makes the cursor durable across programs: (sent % ring_units) * L1_ALIGNMENT.
//   * Credit increments and payload writes ride the same NOC VC, so a drained NIU implies the
//     payload landed before the credit the receiver observes.
//
// Simplifications this sender relies on, both asserted at load time and again on every
// pipe_set_entry_size:
//   * ring_bytes % entry_bytes == 0, so the page-aligned usable region is the whole ring and no
//     end-of-ring padding credit is ever needed (the worker-sender path's trailing-gap term).
//   * A write of n entries must not straddle the ring wrap, matching the contiguous-write rule the
//     worker-sender path also enforces.

#pragma once

#include <cstdint>

#include "hostdev/remote_dfb_config_layout.h"
#include "internal/risc_attribs.h"

namespace experimental {

// Working copy of a DRAM-sender PrefetcherPipe endpoint, loaded from its DRISC-L1 config page.
struct PipeSenderCtx {
    uint32_t config_ptr;            // the config page itself, in DRISC L1
    uint32_t fifo_start_addr;       // ring base, in receiver (worker) L1
    uint32_t ring_bytes;            // entry_bytes * num_entries
    uint32_t entry_bytes;           // push granularity
    uint32_t num_receivers;         // receivers this sender core drives
    uint32_t receiver_noc_xy_ptr;   // -> 2 * num_receivers words of receiver NOC XY
    uint32_t local_counters_ptr;    // DRISC-side counter pairs
    uint32_t remote_counters_base;  // receiver-side counter pairs (see word[8])
};

FORCE_INLINE void pipe_load_sender_ctx(PipeSenderCtx& ctx, uint32_t config_page_addr) {
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
    ASSERT(static_cast<bool>(cfg[REMOTE_DFB_CFG_IS_SENDER]));

    ctx.config_ptr = config_page_addr;
    ctx.num_receivers = cfg[REMOTE_DFB_CFG_NUM_RECEIVERS];
    ctx.fifo_start_addr = cfg[REMOTE_DFB_CFG_FIFO_START];
    ctx.ring_bytes = cfg[REMOTE_DFB_CFG_FIFO_SIZE];
    ctx.entry_bytes = cfg[PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE];
    ctx.receiver_noc_xy_ptr = config_page_addr + cfg[PREFETCHER_PIPE_CFG_NOC_XY_OFFSET];
    ctx.local_counters_ptr = config_page_addr + cfg[PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET];
    ctx.remote_counters_base = config_page_addr + cfg[PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET];

    ASSERT(ctx.entry_bytes != 0);
    ASSERT(ctx.entry_bytes % L1_ALIGNMENT == 0);
    // No trailing-gap credit anywhere in this sender: see the header comment.
    ASSERT(ctx.ring_bytes % ctx.entry_bytes == 0);
}

FORCE_INLINE uint32_t pipe_ring_units(const PipeSenderCtx& ctx) { return ctx.ring_bytes / L1_ALIGNMENT; }

FORCE_INLINE uint32_t pipe_units_per_entry(const PipeSenderCtx& ctx) { return ctx.entry_bytes / L1_ALIGNMENT; }

// This sender's entries_sent counter for receiver r. entries_acked sits one L1_ALIGNMENT above it.
FORCE_INLINE volatile tt_l1_ptr uint32_t* pipe_local_sent_ptr(const PipeSenderCtx& ctx, uint32_t r) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.local_counters_ptr) +
           (2 * r * L1_ALIGNMENT / sizeof(uint32_t));
}

// Byte offset into the ring where receiver r's next entry goes, derived from its credit counter.
// Matches the worker-sender path's wr_offset_from_sent().
FORCE_INLINE uint32_t pipe_derived_wr_offset(const PipeSenderCtx& ctx, uint32_t r) {
    const uint32_t sent_units = *pipe_local_sent_ptr(ctx, r);
    return (sent_units % pipe_ring_units(ctx)) * L1_ALIGNMENT;
}

// Free entries at the most-backed-up receiver, without blocking. Lets a batching caller size its
// next round the way poll_min_free_aligned_pages() does for a GlobalCircularBuffer.
FORCE_INLINE uint32_t pipe_poll_min_free_entries(const PipeSenderCtx& ctx) {
    const uint32_t ring_units = pipe_ring_units(ctx);
    const uint32_t units_per_entry = pipe_units_per_entry(ctx);
    uint32_t min_free_units = ring_units;
    invalidate_l1_cache();
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        volatile tt_l1_ptr uint32_t* sent_ptr = pipe_local_sent_ptr(ctx, r);
        volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
        const uint32_t outstanding = *sent_ptr - *acked_ptr;
        // Clamp rather than subtract blindly: a resize padding credit can transiently push sent
        // further ahead of acked than the ring holds, and an underflow here would wrap to a huge
        // value and defeat receiver backpressure.
        const uint32_t free_units = outstanding >= ring_units ? 0u : ring_units - outstanding;
        if (free_units < min_free_units) {
            min_free_units = free_units;
        }
    }
    return min_free_units / units_per_entry;
}

// Spin until every receiver can take num_entries more entries.
FORCE_INLINE void pipe_reserve_back(const PipeSenderCtx& ctx, uint32_t num_entries) {
    while (pipe_poll_min_free_entries(ctx) < num_entries) {
    }
}

// Publish `units` L1_ALIGNMENT-sized credit units to one receiver: bump this core's local counter
// and NOC-inc the receiver's mirror of it. Advancing entries_sent is also what moves that
// receiver's derived write cursor forward, so call this only after the payload writes are flushed.
FORCE_INLINE void pipe_credit_receiver(const PipeSenderCtx& ctx, uint32_t r, uint32_t units, uint8_t noc) {
    if (units == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.receiver_noc_xy_ptr);
    const uint32_t remote_noc_xy =
        uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, xy[2 * r]), DYNAMIC_NOC_Y(noc, xy[2 * r + 1])));
    const uint32_t remote_sent_ptr = ctx.remote_counters_base + 2 * r * L1_ALIGNMENT;
    *pipe_local_sent_ptr(ctx, r) += units;
    // Posted, matching the worker-sender path: receivers discover credit by polling, and this
    // core observes their acks the same way.
    noc_semaphore_inc</*skip_ptr_update=*/true>(get_noc_addr_helper(remote_noc_xy, remote_sent_ptr), units, noc);
}

// Publish num_entries to every receiver.
FORCE_INLINE void pipe_push_credits(const PipeSenderCtx& ctx, uint32_t num_entries, uint8_t noc) {
    const uint32_t units = num_entries * pipe_units_per_entry(ctx);
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        pipe_credit_receiver(ctx, r, units, noc);
    }
}

// Switch this sender to `entry_bytes` for subsequent pushes: snap every receiver's derived write
// cursor onto the new entry grid and publish the bytes it skips as pad credits. A receiver runs the
// matching snap -- PrefetcherPipe's constructor, when the Attach entry size differs from the one
// last applied -- and waits for exactly these credits, so the two endpoints stay on one grid.
// Mirrors PrefetcherPipe::resize_sender_interface<true>(): because ring_bytes % entry_bytes == 0
// the usable region is the whole ring, so its align-then-wrap step reduces to the one remainder
// below (a cursor that would land on the ring end wraps to zero after crediting the same bytes).
//
// A cursor already on the grid takes no credit, so this is idempotent: a caller pushing a run of
// same-sized tensors can call it before each one.
FORCE_INLINE void pipe_set_entry_size(PipeSenderCtx& ctx, uint32_t entry_bytes, uint8_t noc) {
    ASSERT(entry_bytes != 0);
    ASSERT(entry_bytes % L1_ALIGNMENT == 0);
    ASSERT(ctx.ring_bytes % entry_bytes == 0);
    ctx.entry_bytes = entry_bytes;
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        const uint32_t offset_into_entry = pipe_derived_wr_offset(ctx, r) % entry_bytes;
        if (offset_into_entry == 0) {
            continue;
        }
        pipe_credit_receiver(ctx, r, (entry_bytes - offset_into_entry) / L1_ALIGNMENT, noc);
    }
}

// Post num_entries' worth of payload to one receiver at its derived write position, as
// entries_per_packet-sized packets. Does not touch credits.
FORCE_INLINE void pipe_write_to_receiver(
    const PipeSenderCtx& ctx, uint32_t r, uint32_t src_l1_addr, uint32_t num_entries, uint8_t noc) {
    const uint32_t wr_offset = pipe_derived_wr_offset(ctx, r);
    const uint32_t bytes = num_entries * ctx.entry_bytes;
    // Contiguous-write rule: a write must not straddle the ring wrap.
    ASSERT(wr_offset + bytes <= ctx.ring_bytes);

    volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.receiver_noc_xy_ptr);
    const uint32_t remote_noc_xy =
        uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, xy[2 * r]), DYNAMIC_NOC_Y(noc, xy[2 * r + 1])));
    const uint64_t dst = get_noc_addr_helper(remote_noc_xy, ctx.fifo_start_addr + wr_offset);
    noc_async_write_one_packet</*enable_noc_tracing=*/false, /*posted=*/true>(src_l1_addr, dst, bytes, noc);
}

// Spin until every receiver has acked everything this core has sent.
FORCE_INLINE void pipe_sender_barrier(const PipeSenderCtx& ctx) {
    for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
        volatile tt_l1_ptr uint32_t* sent_ptr = pipe_local_sent_ptr(ctx, r);
        volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
        while (true) {
            invalidate_l1_cache();
            if (*acked_ptr == *sent_ptr) {
                break;
            }
        }
    }
}

}  // namespace experimental
