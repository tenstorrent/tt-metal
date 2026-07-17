// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wire format of one Tensor prefetcher request page, shared by the host
// (TensorPrefetcherManager composes the bytes) and the DRISC kernel (reads them
// out of its H2D socket page via the same structs). Keep all structs packed so the
// L1 byte layout matches on both sides.
//
// Every page starts with a TensorPrefetcherRequestHeader: a one-byte command id
// (TensorPrefetcherBaseCmd) followed by a union of the per-command payloads,
// modeled on the dispatch CQPrefetchCmd / CQDispatchCmd encoding in
// tt_metal/impl/dispatch/kernels/cq_commands.hpp. The three commands are:
//   * STOP      — no payload; the kernel exits its request loop. STOP == 0, so an
//                 all-zero page is a valid stop sentinel.
//   * PREFETCH  — the rest of the page holds the entry + layout tables described
//                 below; the kernel streams those tensors into the target GCB.
//   * WAIT_CQ   — no tables; the kernel spins until its per-CQ signal slot
//                 [wait_cq.cq_index] reaches wait_cq.cq_wait_value (wrap-safe).
//
// Request pages are per-sender: the host serializes one page per DRAM sender core. The
// pages share identical header/entry/geometry bytes, but each carries only that sender's
// slice of the per-receiver streaming rotation table (see below), so worker_loop sends
// sender s's page to that sender's socket rather than broadcasting one page to all.
//
// For a PREFETCH page the payload region (kRequestPageBytes) has two halves that grow
// toward each other:
//
//   [Header][Entry 0][Entry 1] ... [Entry K-1]  ... free ...  [Layout slot L-1] ... [Layout slot 0]
//    ^offset 0, entries grow forward                          ^layout slots grow backward
//                                                               from kRequestPageBytes
//
// - Entries (one per prefetched tensor) carry the tensor's bank-local address plus an
//   index into the layout table. Entry k lives at byte offset
//   sizeof(TensorPrefetcherRequestHeader) + k * sizeof(TensorPrefetcherEntry).
// - The layout table deduplicates the address-independent geometry: tensors that share a
//   shape/dtype/ring topology — and, for streaming, the same per-receiver rotation slice —
//   share one layout slot. A layout slot is sizeof(TensorPrefetcherTensorLayout) bytes of
//   geometry immediately followed by this sender's per-receiver streaming rotation table
//   (num_receivers uint32s; zeroed and ignored for batched tensors). Each slot is therefore
//   layout_stride = sizeof(TensorPrefetcherTensorLayout) + num_receivers * sizeof(uint32_t)
//   bytes. num_receivers is constant within a page (one GCB per request), so the stride is
//   uniform: layout slot i starts at kRequestPageBytes - (i + 1) * layout_stride, with the
//   geometry at the slot start and the rotation table at slot_start +
//   sizeof(TensorPrefetcherTensorLayout) (layout slot 0 is flush against the end of the
//   payload). The kernel reconstructs num_receivers (and thus the stride) from the GCB
//   sender state block.
//
// The kernel walks header.prefetch.num_entries entries in order; for each it reads the
// address from the entry and the geometry from the referenced layout, then runs the
// per-tensor chunk loop.
//
// When one Queue call has more tensors than fit in a single page, the host emits
// multiple PREFETCH pages (each an independent request); the per-GCB fifo_wr_ptr persists
// in the sender state block across requests, so the page split is invisible to the receiver.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Fixed usable payload size of one request page, in bytes. Both the host (layout/entry
// placement) and the kernel (backward layout indexing) must agree on this exact value —
// it is the *unaligned* payload size, distinct from the pcie-aligned socket page size.
// Larger packs more tensors per page but grows the per-socket DRISC L1 FIFO by
// kSocketFifoPages × this.
//
// Sized for fine-grained queueing (one matmul per request → one entry per page): a single
// tensor needs header + one layout + one entry = 72 B, so 128 B holds one comfortably with
// room for a few entries that share a layout. The per-socket L1 FIFO is held constant by
// scaling kSocketFifoPages inversely (see tensor_prefetcher_manager.hpp).
inline constexpr uint32_t kRequestPageBytes = 128;

// Number of per-DRAM-core CQ signal slots (one uint32 counter per command queue).
// WaitForCqOnTensorPrefetcher writes an incrementing value into slot[cq_id] via
// the dispatcher; a WAIT_CQ request makes the kernel spin until it is reached.
constexpr uint32_t kNumCqSignalSlots = 2;

// Address-independent per-tensor geometry handed to the Tensor prefetcher kernel.
// All values are derived from the tensor shape + dtype + GCB ring topology + DRISC L1
// stage budget; the host (compute_tensor_layout) picks (rows_per_sub, M) by the fit
// ladder documented in tt_metal/impl/buffers/prefetcher_matmul_design.md §6. Tensors
// that produce identical layouts share a single table entry (deduplicated per page).
//
// Invariant: rows_per_sub > 1 implies M == 1 (the kernel cannot row-stride DMA).
struct TensorPrefetcherTensorLayout {
    uint32_t num_sub = 0;              // sub-bands per ring-block
    uint32_t M = 0;                    // N-chunks per sub-band (divides num_receivers)
    uint32_t rows_per_sub = 0;         // K-rows per sub-band
    uint32_t coalesced_page_size = 0;  // bytes per K-row per receiver per coalesced page
    uint32_t coalesced_num_pages = 0;  // coalesced pages per K-row per receiver
    uint32_t sub_chunk_bytes = 0;      // bytes per DMA into one ring half
    uint32_t sub_stride_bytes = 0;     // DRAM byte stride between sub-bands within a block
    uint32_t block_stride_bytes = 0;   // DRAM byte stride between ring-blocks
    uint32_t page_bytes_per_recv = 0;  // bytes per receiver per full block (fifo_page_size)
    // Receiver-contiguous-layout fields. Zero/unused under KRowMajor.
    uint32_t layout_mode = 0;             // 0=KRowMajor, 1=ReceiverContiguous (matches LayoutMode)
    uint32_t target_per_visit_pages = 1;  // recv-contig per-receiver visit size ceiling (blocks)
    uint32_t recv_stride_bytes = 0;       // GDDR byte stride between receiver slabs in a bank
    uint32_t block_count = 0;             // K-blocks for this tensor (per-tensor; was the shared GCB ring size)
    // Streaming mode (receiver-contiguous only): when nonzero, the kernel delivers this
    // tensor's receiver slabs in host-specified ring-rotated order — at push step p it sources
    // physical block (rotation[r] + p) mod block_count for local receiver r, where rotation[]
    // is the per-receiver lead-block table the host appends right after this layout in the page
    // (num_receivers uint32s; see the page-format comment above). This delivers blocks in the
    // order the matmul consumes them, so the matmul can stream them FIFO instead of waiting for
    // the whole tensor. 0 = batched (the appended rotation table is zeroed and ignored). The
    // streaming flag is part of the layout so it participates in per-page layout dedup; the
    // appended rotation bytes extend each layout slot's stride and are deduped together with
    // the geometry, so tensors that differ only in rotation get distinct slots.
    uint32_t streaming = 0;
} __attribute__((packed));

// One prefetched tensor: its bank-local address plus an index into the page's layout
// table. The kernel resolves the layout via layout_index (see header comment for the
// offset formula).
struct TensorPrefetcherEntry {
    uint32_t bank_local_base = 0;  // GDDR offset where this tensor starts in the bank
    uint32_t layout_index = 0;     // index into the page's TensorPrefetcherTensorLayout table
} __attribute__((packed));

// One-byte command id at the front of every request page.
enum TensorPrefetcherCmdId : uint8_t {
    DRAM_PREFETCHER_CMD_STOP = 0,      // exit the request loop (no payload; all-zero page)
    DRAM_PREFETCHER_CMD_PREFETCH = 1,  // entry + layout tables follow the header
    DRAM_PREFETCHER_CMD_WAIT_CQ = 2,   // spin until cq slot[cq_index] >= cq_wait_value
};

struct TensorPrefetcherBaseCmd {
    TensorPrefetcherCmdId cmd_id;  // 1 byte
} __attribute__((packed));

// PREFETCH payload. The leading pad keeps the 32-bit fields 4-byte aligned past the
// one-byte base (mirrors the pad fields in cq_commands.hpp commands); the resulting
// 12-byte header then keeps the entry table 4-byte aligned.
struct TensorPrefetcherPrefetchCmd {
    uint8_t pad1;
    uint16_t num_entries;     // number of valid TensorPrefetcherEntry entries
    uint32_t num_layouts;     // number of valid TensorPrefetcherTensorLayout table entries
    uint32_t gcb_state_addr;  // DRISC L1 base of the target GCB's sender state block
} __attribute__((packed));

// WAIT_CQ payload.
struct TensorPrefetcherWaitCqCmd {
    uint8_t cq_index;  // which per-core CQ signal slot to wait on (0/1)
    uint16_t pad1;
    uint32_t cq_wait_value;  // wait until slot >= this value (wrap-safe int32 compare)
} __attribute__((packed));

// Header at the start of each request page: command id + per-command payload union.
struct TensorPrefetcherRequestHeader {
    TensorPrefetcherBaseCmd base;
    union {
        TensorPrefetcherPrefetchCmd prefetch;
        TensorPrefetcherWaitCqCmd wait_cq;
    } __attribute__((packed));
} __attribute__((packed));

// The host fills this header and the kernel parses it field-by-field, so its layout is a
// host↔kernel wire contract. Pin the size (and pack the struct above) so a difference in
// padding/alignment between host and JIT compiler settings can't silently shift cmd_id,
// the payload union, or the layout-table offsets.
static_assert(
    sizeof(TensorPrefetcherRequestHeader) == 12,
    "TensorPrefetcherRequestHeader must be 12 bytes (host↔kernel wire contract)");

// A single tensor must fit in an otherwise-empty PREFETCH page (header + one layout + one entry).
// This is a compile-time floor; a streaming tensor's layout slot additionally carries
// num_receivers * sizeof(uint32_t) rotation bytes (runtime, bounded by recv_per_bank since a
// page is per-sender), which serialize_request_pages validates against kRequestPageBytes.
static_assert(
    sizeof(TensorPrefetcherRequestHeader) + sizeof(TensorPrefetcherTensorLayout) + sizeof(TensorPrefetcherEntry) <=
        kRequestPageBytes,
    "kRequestPageBytes too small to hold a single tensor's header + layout + entry");

}  // namespace tt::tt_metal
