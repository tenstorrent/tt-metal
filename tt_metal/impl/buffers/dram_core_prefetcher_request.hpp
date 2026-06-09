// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wire format of one DRAM-core prefetcher request page, shared by the host
// (DramCorePrefetcherManager composes the bytes) and the DRISC kernel (reads them
// out of its H2D socket page via the same structs). Keep all structs packed so the
// L1 byte layout matches on both sides.
//
// One request page is a fixed kRequestPageBytes payload region whose two halves grow
// toward each other:
//
//   [Header][Entry 0][Entry 1] ... [Entry K-1]  ... free ...  [Layout L-1] ... [Layout 0]
//    ^offset 0, entries grow forward                          ^layouts grow backward
//                                                               from kRequestPageBytes
//
// - Entries (one per prefetched tensor) carry the tensor's bank-local address plus an
//   index into the layout table. Entry k lives at byte offset
//   sizeof(DramCorePrefetcherRequestHeader) + k * sizeof(DramCorePrefetcherEntry).
// - The layout table deduplicates the address-independent geometry: tensors that share
//   a shape/dtype/ring topology share one DramCorePrefetcherTensorLayout. Layout i lives
//   at byte offset kRequestPageBytes - (i + 1) * sizeof(DramCorePrefetcherTensorLayout)
//   (i.e. layout 0 is flush against the end of the payload).
//
// The kernel walks the entries in order; for each it reads the address from the entry
// and the geometry from the referenced layout, then runs the per-tensor chunk loop. A
// header with num_entries == 0 is the stop sentinel.
//
// When one Queue call has more tensors than fit in a single page, the host emits
// multiple pages (each an independent request); the per-GCB fifo_wr_ptr persists in the
// sender state block across requests, so the page split is invisible to the receiver.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Fixed usable payload size of one request page, in bytes. Both the host (layout/entry
// placement) and the kernel (backward layout indexing) must agree on this exact value —
// it is the *unaligned* payload size, distinct from the pcie-aligned socket page size.
// Larger packs more tensors per page but grows the per-socket DRISC L1 FIFO by
// kSocketFifoPages × this.
inline constexpr uint32_t kRequestPageBytes = 1024;

// Address-independent per-tensor geometry handed to the DRAM-core prefetcher kernel.
// All values are derived from the tensor shape + dtype + GCB ring topology + DRISC L1
// stage budget; the host (compute_tensor_layout) picks (rows_per_sub, M) by the fit
// ladder documented in tt_metal/impl/buffers/prefetcher_matmul_design.md §6. Tensors
// that produce identical layouts share a single table entry (deduplicated per page).
//
// Invariant: rows_per_sub > 1 implies M == 1 (the kernel cannot row-stride DMA).
struct DramCorePrefetcherTensorLayout {
    uint32_t num_sub = 0;              // sub-bands per ring-block
    uint32_t M = 0;                    // N-chunks per sub-band (divides num_receivers)
    uint32_t rows_per_sub = 0;         // K-rows per sub-band
    uint32_t coalesced_page_size = 0;  // bytes per K-row per receiver per coalesced page
    uint32_t coalesced_num_pages = 0;  // coalesced pages per K-row per receiver
    uint32_t sub_chunk_bytes = 0;      // bytes per DMA into one ring half
    uint32_t sub_stride_bytes = 0;     // DRAM byte stride between sub-bands within a block
    uint32_t block_stride_bytes = 0;   // DRAM byte stride between ring-blocks
    uint32_t page_bytes_per_recv = 0;  // bytes per receiver per full block (fifo_page_size)
    uint32_t block_count = 0;          // K-blocks for this tensor (per-tensor; was the shared GCB ring size)
} __attribute__((packed));

// One prefetched tensor: its bank-local address plus an index into the page's layout
// table. The kernel resolves the layout via layout_index (see header comment for the
// offset formula).
struct DramCorePrefetcherEntry {
    uint32_t bank_local_base = 0;  // GDDR offset where this tensor starts in the bank
    uint32_t layout_index = 0;     // index into the page's DramCorePrefetcherTensorLayout table
} __attribute__((packed));

// Header at the start of each request page.
struct DramCorePrefetcherRequestHeader {
    uint32_t num_entries = 0;     // number of valid DramCorePrefetcherEntry entries; 0 = stop sentinel
    uint32_t num_layouts = 0;     // number of valid DramCorePrefetcherTensorLayout table entries
    uint32_t gcb_state_addr = 0;  // DRISC L1 base of the target GCB's sender state block
} __attribute__((packed));

// A single tensor must fit in an otherwise-empty page (header + one layout + one entry).
static_assert(
    sizeof(DramCorePrefetcherRequestHeader) + sizeof(DramCorePrefetcherTensorLayout) +
            sizeof(DramCorePrefetcherEntry) <=
        kRequestPageBytes,
    "kRequestPageBytes too small to hold a single tensor's header + layout + entry");

}  // namespace tt::tt_metal
