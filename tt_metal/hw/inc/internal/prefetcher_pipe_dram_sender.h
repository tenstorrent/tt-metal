// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Loading a DRAM-sender PrefetcherPipe's config page into plain values, for the Tensor prefetcher's
// receiver-contiguous loop (impl/buffers/kernels/tensor_prefetcher.cpp). That loop drives both of
// its transports -- DRAM-sender GlobalCircularBuffers and PrefetcherPipes -- through one
// RemoteSenderCBInterface and one copy of its credit code, so it reads a pipe's page into that
// interface rather than pushing through the device PrefetcherPipe class. Everything else a DRAM
// sender does with a pipe -- an entry-size change, a drain, a plain push -- goes through the class
// (PrefetcherPipe's DramSenderConfigPage constructor).
//
// The page is the one the host stamps into DRISC L1 (impl/buffers/prefetcher_pipe.cpp,
// build_dram_sender_config_pages); its layout is remote_dfb_config_layout.h's. What this header
// relies on:
//
//   * Credits are counted in L1_ALIGNMENT-byte units, one L1_ALIGNMENT slot per receiver in each of
//     the SENT (word[7]) and ACKED (word[8]) blocks. A DRAM-sender pipe is single-lane.
//   * A receiver's write cursor is stored in the padding word of its SENT slot
//     (PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD), as a byte offset from the ring base.
//   * The receivers' SENT block base is page-relative to this page, in word[9]
//     (PREFETCHER_PIPE_CFG_PEER_COUNTER_OFFSET).
//   * An entry size need not divide the ring: the last `ring_bytes % entry_bytes` bytes are a
//     trailing gap that holds no entry and is credited at the wrap.

#pragma once

#include <cstdint>

#include "hostdev/remote_dfb_config_layout.h"
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
// PipeSenderCtx so the prefetcher can find it per request without building a context.
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

// Bytes of the ring that hold whole entries. The remainder is the trailing gap: no entry starts
// there, and it is credited as padding by whichever write reaches this limit.
FORCE_INLINE uint32_t pipe_usable_bytes(const PipeSenderCtx& ctx) {
    return ctx.ring_bytes - ctx.ring_bytes % ctx.entry_bytes;
}

// Receiver r's write cursor, in the padding word of its SENT slot. A credit slot is a whole
// L1_ALIGNMENT because it is a NOC-atomic target; the cursor rides the padding that alignment
// already reserves, which is both where the device PrefetcherPipe class keeps it and inside the
// range a credit reset zeroes, so credits and cursors can never reset out of step.
FORCE_INLINE volatile tt_l1_ptr uint32_t* pipe_local_wr_offset_ptr(const PipeSenderCtx& ctx, uint32_t r) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.local_sent_base + r * L1_ALIGNMENT) +
           PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD;
}

// Byte offset into the ring where receiver r's next entry goes.
FORCE_INLINE uint32_t pipe_sender_wr_offset(const PipeSenderCtx& ctx, uint32_t r) {
    return *pipe_local_wr_offset_ptr(ctx, r);
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

}  // namespace experimental
