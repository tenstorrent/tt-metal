// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wire format of one DRAM-core prefetcher request page, shared by the host
// (DramCorePrefetcherManager composes the bytes) and the DRISC kernel (reads them
// out of its H2D socket page via the same structs). Keep both structs packed so the
// L1 byte layout matches on both sides.
//
// One request page is a DramCorePrefetcherRequestHeader followed by
// kMaxTensorsPerRequest DramCorePrefetcherTensorGeom entries (only the first
// header.num_tensors are valid). A header with num_tensors == 0 is the stop sentinel.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Per-tensor geometry handed to the DRAM-core prefetcher kernel. All values are
// derived from the tensor shape + dtype + GCB ring topology + DRISC L1 stage budget;
// the host (compute_tensor_geom) picks (rows_per_sub, M) by the fit ladder documented
// in tt_metal/impl/buffers/prefetcher_matmul_design.md §6.
//
// Invariant: rows_per_sub > 1 implies M == 1 (the kernel cannot row-stride DMA).
struct DramCorePrefetcherTensorGeom {
    uint32_t bank_local_base = 0;      // GDDR offset where this tensor starts in the bank
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

// Header at the start of each request page.
struct DramCorePrefetcherRequestHeader {
    uint32_t num_tensors = 0;     // number of valid DramCorePrefetcherTensorGeom entries; 0 = stop sentinel
    uint32_t num_layers = 0;      // outer loop count: the kernel replays the tensor list this many times
    uint32_t gcb_state_addr = 0;  // DRISC L1 base of the target GCB's sender state block
} __attribute__((packed));

}  // namespace tt::tt_metal
