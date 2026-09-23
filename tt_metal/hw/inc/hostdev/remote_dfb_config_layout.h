// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Remote-DFB config page layouts used by CrossNodeDFB and PrefetcherPipe.
//
// Shared prefix:
//   word[0]  is_sender (1) | is_receiver (0)
//   word[1]  num_receivers
//   word[2]  fifo_start_addr
//   word[3]  fifo_size (CrossNode: entry_size * num_entries; PrefetcherPipe: ring bytes)
//
// CrossNode:
//   word[4]  fifo_ptr_checkpoint   // reserved / ignored; ctor resets ptrs to word[2]
//   word[5]  noc_xy_offset         // page-relative → word[8]
//   word[6]  pages_sent_offset     // page-relative
//   word[7]  pages_acked_offset    // page-relative
//   + sender NOC XY table + L1-aligned sent/acked pairs
//   + receiver sender XY after header
//
// PrefetcherPipe (one config page per core — not per thread):
//
//   Config page
//   ├── header
//   │     word[4]  fifo_ptr_checkpoint   // sender wr / receiver rd; commit stores; ctor loads
//   │     word[5]  applied_entry_size    // epoch + last successful resize
//   │     word[6]  noc_xy_offset         // page-relative → after header
//   │     word[7]  pages_sent_offset
//   │     word[8]  pages_acked_offset
//   │     word[9]  reserved (0)          // active lane count P travels in the per-program
//   │                                    // kernel-config slot, see remote_dfb_constants.h
//   ├── NOC XY table
//   ├── pad → PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN
//   ├── SENT block   (word[7]) — one L1_ALIGNMENT slot per (receiver, lane)
//   │     receiver 0: lane 0: [sent | wr_cursor]   lane 1: [sent]   lane 2: …   lane 3: …
//   │     receiver 1: …
//   ├── pad → PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN
//   └── ACKED block  (word[8]) — one L1_ALIGNMENT slot per (receiver, lane)
//         receiver 0: lane 0: [acked]   lane 1: [acked]   lane 2: …   lane 3: …
//         receiver 1: …
//
//   The two blocks are kept in separate cache lines on purpose. On the sender core the SENT
//   block is written locally (cached stores) and the ACKED block is written by receivers'
//   NoC atomics; on a receiver core it is the reverse. Quasar's DM L2 writes back whole 64B
//   lines, so a line holding both kinds of word would, on eviction, overwrite the peer's
//   NoC-written counter with a stale copy. With the split, a dirty line can only ever hold
//   words the same core wrote. The sender page and every receiver page share this layout, so
//   the same slot offset addresses the mirror counter on the peer.
//
//   Active lane count P (receiver kernel num_threads / relay num_producers) is not in
//   the page: it is packed into the program's kernel-config slot so it arrives in CQ order
//   with the program that uses it. It is armed once per pipe lifetime (1 -> P) because the
//   persistent credit block below is interpreted through it. Layout stride per receiver is
//   PREFETCHER_PIPE_MAX_CREDIT_LANES on Quasar (1 on WH/BH), so activating more consumers
//   does not resize persistent L1. Sender stripes pages_sent to lane (entry_idx % P);
//   receiver hart tid binds to lane tid. Still one ring and one page.
//   The sender page's word[7]/word[8] are the block bases; a receiver page's are that
//   receiver's lane-0 slot within each block.

inline constexpr uint32_t REMOTE_DFB_CFG_IS_SENDER = 0;
inline constexpr uint32_t REMOTE_DFB_CFG_NUM_RECEIVERS = 1;
inline constexpr uint32_t REMOTE_DFB_CFG_FIFO_START = 2;
inline constexpr uint32_t REMOTE_DFB_CFG_FIFO_SIZE = 3;

// --- CrossNodeDFB header ---
inline constexpr uint32_t CROSS_NODE_DFB_CONFIG_HEADER_WORDS = 8;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_FIFO_PTR_CHECKPOINT = 4;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_NOC_XY_OFFSET = 5;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_PAGES_SENT_OFFSET = 6;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_PAGES_ACKED_OFFSET = 7;

constexpr uint32_t cross_node_dfb_noc_xy_byte_offset() {
    return CROSS_NODE_DFB_CONFIG_HEADER_WORDS * static_cast<uint32_t>(sizeof(uint32_t));
}

// --- PrefetcherPipe header ---
inline constexpr uint32_t PREFETCHER_PIPE_CONFIG_HEADER_WORDS = 10;
inline constexpr uint32_t PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT = 4;
inline constexpr uint32_t PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE = 5;
inline constexpr uint32_t PREFETCHER_PIPE_CFG_NOC_XY_OFFSET = 6;
inline constexpr uint32_t PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET = 7;
inline constexpr uint32_t PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET = 8;
// word[9] reserved.

// Quasar: config pages reserve this many lane (sent,acked) slots per receiver so
// consumer relay bind can activate up to this many producers without resizing
// persistent L1 (same role as max_receivers_per_pipe for receiver fanout). WH/BH
// use 1. The active count is per program (kernel-config slot), not this capacity.
inline constexpr uint32_t PREFETCHER_PIPE_MAX_CREDIT_LANES = 4;

// The SENT and ACKED blocks each start on this boundary (and the config page itself is
// allocated at it), so no cache line holds words from both blocks. Sized for the Quasar DM
// L2 line; WH/BH have no write-back cache and simply pay the (≤ 2 × 64B) padding.
inline constexpr uint32_t PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN = 64;

// Word index, within a lane-0 SENT slot, of that receiver's sender-side write cursor.
//
// A SENT slot is L1_ALIGNMENT bytes with entries_sent at word 0; the cursor lives in the
// padding that alignment already reserves, so it costs no page growth and -- the reason it
// belongs here rather than in a separate array -- it sits inside the range
// PrefetcherPipe::credit_reset_offset() / credit_reset_size() covers. Zeroing a pipe's credits
// therefore returns its cursors to the ring start in the same store, and the two can never be
// reset out of step. On the sender core both words are locally written; on a receiver core the
// slot is a NoC-atomic target and the cursor word is unused.
//
// The cursor is a byte offset from the ring base that wraps at the full allocation. It is kept
// separately from entries_sent -- rather than derived as (entries_sent % ring_units) -- because
// entries_sent is a free-running uint32 whose 2^32 wrap only preserves that modulus when
// ring_units is a power of two. Only lane 0's sent slot carries this cursor.
inline constexpr uint32_t PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD = 1;

constexpr uint32_t prefetcher_pipe_noc_xy_byte_offset() {
    return PREFETCHER_PIPE_CONFIG_HEADER_WORDS * static_cast<uint32_t>(sizeof(uint32_t));
}
