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
// tt_metal/impl/dispatch/kernels/cq_commands.hpp. The commands are:
//   * STOP        — no payload; the kernel exits its request loop. STOP == 0, so an
//                   all-zero page is a valid stop sentinel.
//   * PREFETCH    — the rest of the page holds the entry + layout tables described
//                   below; the kernel streams those tensors into the target GCB.
//   * WAIT_SIGNAL — no tables; the kernel spins until its signal slot
//                   [wait_signal.slot_index] reaches wait_signal.wait_value (wrap-safe).
//   * DRAIN       — the target words that follow the header; see TensorPrefetcherDrainCmd.
//
// Request pages are per-sender: the host serializes one page per DRAM sender core. The
// entry and geometry bytes are identical across senders, but the header carries that
// sender's target state address, every layout slot carries that sender's bank-local slab
// base, and a streaming page carries only that sender's slice of the per-receiver rotation
// table (see below) -- so worker_loop sends sender s's page to that sender's socket rather
// than broadcasting one page to all.
//
// For a PREFETCH page the payload region (kRequestPageBytes) has two halves that grow
// toward each other:
//
//   [Header][Entry 0][Entry 1] ... [Entry K-1]  ... free ...  [Layout slot L-1] ... [Layout slot 0]
//    ^offset 0, entries grow forward                          ^layout slots grow backward
//                                                               from kRequestPageBytes
//
// - Entries (one per prefetched tensor) carry the tensor's bank-local address, the byte offset of
//   its layout slot from the page start, and its group selector (how many groups the tensor stacks
//   per receiver slab, and optionally the NoC location of a mask choosing which of them to
//   stream). Entry k lives at byte offset sizeof(TensorPrefetcherRequestHeader) +
//   k * sizeof(TensorPrefetcherEntry).
// - The layout table deduplicates the address-independent geometry: tensors that share a
//   shape/dtype/ring topology — and, for streaming, the same per-receiver rotation slice —
//   share one layout slot. A layout slot is sizeof(TensorPrefetcherTensorLayout) bytes of
//   geometry immediately followed by this sender's per-receiver streaming rotation table
//   (num_receivers uint32s; zeroed and ignored for batched tensors), so its useful length is
//   sizeof(TensorPrefetcherTensorLayout) + num_receivers * sizeof(uint32_t). The host packs
//   the slots at one uniform stride sized for the largest receiver count over the request's
//   senders, which is what lets a single template page serve every sender: layout slot i
//   starts at kRequestPageBytes - (i + 1) * layout_stride, with the geometry at the slot start
//   and the rotation table at slot_start + sizeof(TensorPrefetcherTensorLayout) (layout slot 0
//   is flush against the end of the payload). That stride is host-only packing bookkeeping —
//   the kernel never reconstructs it, because each entry names its slot by byte offset.
//
// The kernel walks header.prefetch.num_entries entries in order; for each it reads the
// address from the entry and the geometry from the referenced layout, then runs the
// per-tensor chunk loop once per streamed group (once for an ungrouped tensor).
//
// Budget: header (12) + one entry (20) + one layout (64) leaves 32 B of a 128 B page for the
// streaming rotation table, so a streaming tensor fits for up to 8 receivers per sender.
// serialize_request_pages rejects anything larger.
//
// When one Queue call has more tensors than fit in a single page, the host emits
// multiple PREFETCH pages (each an independent request); the target's per-sender write cursor
// persists across requests, so the page split is invisible to the receiver.

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
// tensor needs header + one layout + one entry = 96 B, so 128 B holds one with room for one
// more entry that shares its layout, or a rotation table for up to 8 receivers. The per-socket
// L1 FIFO is held constant by scaling kSocketFifoPages inversely (see
// tensor_prefetcher_manager.hpp).
inline constexpr uint32_t kRequestPageBytes = 128;

// Every DRAM sender core keeps one table of signal slots, each a uint32 counter that only grows, and a
// WAIT_SIGNAL request names one slot and the count to wait for. The table holds kNumCqSignalSlots CQ
// fence slots followed by the op signal slots:
//   * CQ fence slot c: WaitForCqOnTensorPrefetcher has the dispatcher write an incrementing value
//     into it, ordered after the work already on command queue c.
//   * Op signal slot kNumCqSignalSlots + s: device kernels atomically increment it by one per signal
//     on op signal s (QueueTensorPrefetcherWaitForSignal), and the host counts the waits it queued
//     to know which count each new wait needs. Signals reach only each bank's free sender, which
//     copies its count into the same slot of the bank's NOC1-endpoint sender each time it passes a
//     wait on it.
// The host zeroes the table before launching the kernels.
constexpr uint32_t kNumCqSignalSlots = 2;

// How a DRAIN request names a target: its target_state_addr, which is L1-aligned, with the low bit
// set for a PrefetcherPipe and clear for a GlobalCircularBuffer.
inline constexpr uint32_t kDrainTargetPipeBit = 1;

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
    // Bank-local slab index of this sender's first receiver, so a bank's two senders can split its
    // receiver set: local receiver r reads slab recv_index_base + r. 0 for a single sender.
    // Receiver-contiguous only. The one per-sender field in this struct: the template page holds 0
    // and each sender's copy is patched in every slot, the way the rotation table is. It is
    // therefore excluded from layout dedup by construction — dedup compares the template geometry,
    // where this is always 0.
    uint32_t recv_index_base = 0;
    // Grouped weight (receiver-contiguous only): each receiver's slab stacks the tensor's groups
    // (experts) one after another, and this is the byte stride from one group's block to the next.
    // Group g of local receiver r starts at
    // bank_local_base + g * group_stride_bytes + (recv_index_base + r) * recv_stride_bytes.
    // 0 for an ungrouped tensor. Geometry, so it takes part in layout dedup.
    uint32_t group_stride_bytes = 0;
} __attribute__((packed));

// Delivery transport for a request, carried in the page header. Per-request rather than per-tensor
// because a page names exactly one target (target_state_addr), so every tensor in it is delivered
// the same way. It selects how the kernel interprets target_state_addr: a DramSenderStateBlock for
// a GlobalCircularBuffer, a PrefetcherPipe sender config page for a pipe.
enum TensorPrefetcherTransport : uint8_t {
    TENSOR_PREFETCHER_TRANSPORT_GLOBAL_CB = 0,
    TENSOR_PREFETCHER_TRANSPORT_PREFETCHER_PIPE = 1,
};

// How an entry picks which of its groups to stream (TensorPrefetcherEntry::selector_mode).
enum TensorPrefetcherSelectorMode : uint8_t {
    // Stream every group, in ascending order.
    TENSOR_PREFETCHER_SELECTOR_NONE = 0,
    // Read num_groups 16-bit words from the selector page and stream, in ascending order, only the
    // groups whose word is non-zero. The consumer scans the same mask in the same order.
    TENSOR_PREFETCHER_SELECTOR_NONZERO_MASK = 1,
};

// Most groups one selector can name: the kernel reads the selector page (one 16-bit word per group)
// into a scratch buffer of this many words.
inline constexpr uint32_t kTensorPrefetcherSelectorMaxEntries = 256;
inline constexpr uint32_t kTensorPrefetcherSelectorScratchBytes =
    kTensorPrefetcherSelectorMaxEntries * sizeof(uint16_t);

// One prefetched tensor: its bank-local address, the position of its layout slot, and which of its
// groups to stream. The kernel resolves the layout as page start + layout_offset, so it needs to know
// nothing about how the host packed the slots.
struct TensorPrefetcherEntry {
    uint32_t bank_local_base = 0;  // GDDR offset where this tensor starts in the bank
    // Byte offset from the page start of this tensor's TensorPrefetcherTensorLayout; its
    // per-receiver rotation table follows the struct.
    uint32_t layout_offset = 0;
    // Where the selector page lives when selector_mode is not NONE: a NoC endpoint as virtual
    // coordinates, x in the low 16 bits and y in the high 16, which the kernel maps onto its own NoC
    // the way it maps the receiver table; and the local address of the page at that endpoint.
    uint32_t selector_noc_xy = 0;
    uint32_t selector_addr = 0;
    // Groups the tensor stacks per receiver slab: 1 for an ungrouped tensor. With a selector this is
    // also how many selector words the kernel reads.
    uint16_t num_groups = 1;
    TensorPrefetcherSelectorMode selector_mode = TENSOR_PREFETCHER_SELECTOR_NONE;
    uint8_t pad = 0;
} __attribute__((packed));

static_assert(
    sizeof(TensorPrefetcherEntry) == 20, "TensorPrefetcherEntry must be 20 bytes (host↔kernel wire contract)");

// One-byte command id at the front of every request page.
enum TensorPrefetcherCmdId : uint8_t {
    DRAM_PREFETCHER_CMD_STOP = 0,         // exit the request loop (no payload; all-zero page)
    DRAM_PREFETCHER_CMD_PREFETCH = 1,     // entry + layout tables follow the header
    DRAM_PREFETCHER_CMD_WAIT_SIGNAL = 2,  // spin until signal slot[slot_index] >= wait_value
    DRAM_PREFETCHER_CMD_DRAIN = 3,        // wait for every receiver of the listed targets to ack
};

struct TensorPrefetcherBaseCmd {
    TensorPrefetcherCmdId cmd_id;  // 1 byte
} __attribute__((packed));

// PREFETCH payload. Field order and widths keep every field naturally aligned past the one-byte
// base (u16 at offsets 2 and 4, u32 at offset 8, mirroring the pad fields in cq_commands.hpp
// commands); the resulting 12-byte header then keeps the entry table 4-byte aligned.
//
// The header carries only what holds for the whole request: how many entries and layout slots
// follow, and which target endpoint they are delivered to. Nothing here describes how the host
// packed the page — entries name their slots by byte offset — and nothing here is per-tensor.
// Carrying the target address per request rather than reading it from the target's DRISC L1 state
// keeps a page self-describing for both transports.
struct TensorPrefetcherPrefetchCmd {
    // Fits in the byte that pads cmd_id out to the 16-bit fields, so the header stays 12 bytes and
    // the entry table that follows stays 4-byte aligned.
    TensorPrefetcherTransport transport;
    uint16_t num_entries;  // number of valid TensorPrefetcherEntry entries
    // Number of valid TensorPrefetcherTensorLayout table entries. uint16 is ample: the table is
    // bounded by kRequestPageBytes / layout_stride, well under 300 even at the smallest stride.
    uint16_t num_layouts;
    // Reserved, and the reason target_state_addr below lands on offset 8: that keeps the header 12
    // bytes (see the static_assert below) and so the entry table 4-byte aligned. Zeroed by the host.
    uint16_t pad1;
    // DRISC L1 base of this sender's target state, whose meaning follows the `transport` above: a
    // DramSenderStateBlock for TENSOR_PREFETCHER_TRANSPORT_GLOBAL_CB, or a PrefetcherPipe sender
    // config page for TENSOR_PREFETCHER_TRANSPORT_PREFETCHER_PIPE. One address per request, so all
    // tensors in a request target the same object.
    uint32_t target_state_addr;
} __attribute__((packed));

// WAIT_SIGNAL payload.
struct TensorPrefetcherWaitSignalCmd {
    uint8_t slot_index;  // which signal slot to wait on: a CQ fence slot, then the op signal slots
    uint16_t pad1;
    uint32_t wait_value;  // wait until slot >= this value (wrap-safe int32 compare)
} __attribute__((packed));

// DRAIN payload. A request returns without waiting for its receivers, so an earlier target's
// receivers may still be acking into this sender's L1 after it has moved on to another target. The
// host sends DRAIN pages naming every target this sender was queued for, and still holds the memory
// of, right before STOP, so the kernel exits only once those acks have all landed. The target words
// (kDrainTargetPipeBit encoding) follow the header.
struct TensorPrefetcherDrainCmd {
    uint8_t pad0;
    uint16_t num_targets;
} __attribute__((packed));

// Header at the start of each request page: command id + per-command payload union.
struct TensorPrefetcherRequestHeader {
    TensorPrefetcherBaseCmd base;
    union {
        TensorPrefetcherPrefetchCmd prefetch;
        TensorPrefetcherWaitSignalCmd wait_signal;
        TensorPrefetcherDrainCmd drain;
    } __attribute__((packed));
} __attribute__((packed));

// The host fills this header and the kernel parses it field-by-field, so its layout is a
// host↔kernel wire contract. Pin the size (and pack the struct above) so a difference in
// padding/alignment between host and JIT compiler settings can't silently shift cmd_id,
// the payload union, or the layout-table offsets.
static_assert(
    sizeof(TensorPrefetcherRequestHeader) == 12,
    "TensorPrefetcherRequestHeader must be 12 bytes (host↔kernel wire contract)");

// Target words one DRAIN page carries after its header.
inline constexpr uint32_t kDrainTargetsPerPage =
    (kRequestPageBytes - sizeof(TensorPrefetcherRequestHeader)) / sizeof(uint32_t);

// A single tensor must fit in an otherwise-empty PREFETCH page (header + one layout + one entry).
// This is a compile-time floor; a streaming tensor's layout slot additionally carries
// num_receivers * sizeof(uint32_t) rotation bytes (runtime, bounded by recv_per_bank since a
// page is per-sender), which serialize_request_pages validates against kRequestPageBytes.
static_assert(sizeof(TensorPrefetcherTensorLayout) == 64, "TensorPrefetcherTensorLayout must be 64 bytes");
static_assert(
    sizeof(TensorPrefetcherRequestHeader) + sizeof(TensorPrefetcherTensorLayout) + sizeof(TensorPrefetcherEntry) <=
        kRequestPageBytes,
    "kRequestPageBytes too small to hold a single tensor's header + layout + entry");

}  // namespace tt::tt_metal
