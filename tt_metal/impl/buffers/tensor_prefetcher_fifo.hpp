// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the host and the Tensor prefetcher kernels that fast dispatch runs on a dispatch engine
// (tensor_prefetcher_kernels(): tt_metal/impl/buffers/kernels/tensor_prefetcher_{reader,pusher}.cpp).
//
// The kernels share one L1 region: the part of the dispatch engine's L1 that the dispatch memory map leaves
// above every CQ zone. Offsets below are from the base of that region. The host writes requests into a FIFO
// there and every kernel reads each slot in place; the slot is free again once every kernel has retired it.
//
// Every word that another agent reads or writes sits alone on a cache line and is accessed through
// the uncached L1 alias: host writes and NoC traffic do not snoop the data-movement caches.

#include <stdint.h>

#include "tensor_prefetcher_request.hpp"

namespace tt::tt_metal::tensor_prefetcher_fifo {

inline constexpr uint32_t kLineBytes = 64;

// The most Tensor prefetcher kernels a dispatch engine runs: it has 8 data-movement cores, and the kernels of
// the one CQ it serves take 3.
inline constexpr uint32_t kMaxKernels = 5;

// Control block, one word per line.
// Non-zero tells every kernel to exit. Written by the host when fast dispatch tears down.
inline constexpr uint32_t kTerminateOffset = 0 * kLineBytes;
// Slots the host has written so far (monotonic).
inline constexpr uint32_t kFifoWrOffset = 1 * kLineBytes;
// Slots kernel `k` (its index in tensor_prefetcher_kernels()) has retired so far (monotonic).
constexpr uint32_t fifo_rd_offset(uint32_t k) { return (2 + k) * kLineBytes; }
// Signal slot `slot_index` (tensor_prefetcher_request.hpp), which a WAIT_SIGNAL request names: the dispatcher
// writes a CQ fence slot and device kernels increment an op signal slot. A line apiece, which also keeps each
// slot L1-aligned, as a dispatcher write must be. Zeroed by the host at Start.
constexpr uint32_t signal_slot_offset(uint32_t slot_index) {
    return fifo_rd_offset(kMaxKernels) + slot_index * kLineBytes;
}
inline constexpr uint32_t kControlBytes = signal_slot_offset(kNumSignalSlots);

// Request slots in the FIFO.
inline constexpr uint32_t kFifoSlots = 16;

// Whether FIFO count `a` is behind count `b`. The counts wrap, and every kernel's count stays within a
// FIFO's depth of the host's, so their signed difference orders them; the numeric order is wrong once one
// has wrapped and the other has not.
constexpr bool count_behind(uint32_t a, uint32_t b) { return static_cast<int32_t>(a - b) < 0; }
static_assert(count_behind(0xFFFFFFFFu, 0), "A count about to wrap is behind one that has wrapped");
static_assert(!count_behind(0, 0xFFFFFFFFu), "A count that has wrapped is ahead of one about to wrap");
static_assert(!count_behind(7, 7), "A count is not behind itself");
// Before the host has written a full FIFO, the count it waits for to reuse a slot is below zero, and no
// kernel is behind it.
static_assert(!count_behind(0, 0 - kFifoSlots + 1), "Nothing to wait for before the FIFO first fills");

// The rest of the region, in order: a FIFO of kFifoSlots request slots, the per-receiver progress
// counters, the PrefetcherPipe sender state of the DRAM-sender endpoints placed on this dispatch engine,
// the kernels' own state, and the stage ring. Its size depends on how many lanes a slot carries and on how
// much state the kernels keep, so RegionLayout below gives the offsets for those.
//
// One slot = a header naming the DRAM bank behind each lane, then one request page per lane (the Tensor
// prefetcher request page, tensor_prefetcher_request.hpp). A lane is one DRAM bank's sender endpoint; the
// dispatch engine holds one endpoint per bank. A bank's endpoint is implied by the H2D socket a page arrives
// on when the sender is a DRAM core, so here the bank travels in the header instead.
inline constexpr uint32_t kLanePageBytes = kRequestPageBytes;
inline constexpr uint32_t kSlotHeaderBytes = kLineBytes;

// Per-receiver progress between the reader and the pusher: blocks staged by the reader, and blocks pushed
// (so their stage slots are free again) by the pusher. One monotonic word per line, one writer each.
inline constexpr uint32_t kMaxReceivers = 32;

// Sender config pages of the endpoints' pipes. A bank's one endpoint serves all of its receivers, so a page
// is sized for kMaxReceivers of them (remote_dfb_config_layout.h): a 10-word header and a 2-word NoC XY entry
// per receiver, rounded up to a line, then a SENT and an ACKED block with one 16 B (L1-aligned) slot per
// receiver for each of Quasar's 4 credit lanes. Each endpoint gets room for two pages, so a re-carve or a
// second pipe space can hold one while another is live. The zone is fixed for the region, so carving a pipe
// never moves the stage ring under a running kernel.
inline constexpr uint32_t kSenderStatePageBytes =
    ((10 + 2 * kMaxReceivers) * 4 + kLineBytes - 1) / kLineBytes * kLineBytes + 2 * kMaxReceivers * 4 * 16;
inline constexpr uint32_t kSenderStateBytesPerEndpoint = 2 * kSenderStatePageBytes;

// The stage ring is split evenly between a request's receivers, kStageSlotsPerReceiver slots each, so the
// reader can keep reading for every receiver whichever one the pusher is waiting on. One slot per receiver
// is the minimum that cannot deadlock; the second overlaps a read with a push. Slot size depends on the
// receiver count alone, not on the block size, so the reader stages the next tensor or request while the
// pusher is still sending the last; it waits for the pusher to empty the ring only when the receiver count
// changes.
inline constexpr uint32_t kStageSlotsPerReceiver = 2;
// The smallest stage ring the kernels are run with: both stage slots of every receiver a slot can name,
// at 1 KiB per block.
inline constexpr uint32_t kMinStageRingBytes = kMaxReceivers * kStageSlotsPerReceiver * 1024;

enum SlotCommand : uint32_t {
    // Retired by every kernel with no other effect. Lets the host check that the FIFO is serviced.
    kSlotNop = 1,
    // num_lanes request pages follow the header, one per lane, all PREFETCH or all DRAIN. A PREFETCH
    // slot's lanes carry the same tensors for different receivers, and its receivers are served
    // together rather than lane by lane.
    kSlotRequest = 2,
};

// The most lanes a slot can carry: their bank ids fill the rest of the header line.
inline constexpr uint32_t kMaxLanes = kSlotHeaderBytes - 2 * sizeof(uint32_t);

struct SlotHeader {
    uint32_t command = 0;
    uint32_t num_lanes = 0;
    uint8_t lane_bank_id[kMaxLanes] = {};
};
static_assert(sizeof(SlotHeader) == kSlotHeaderBytes, "SlotHeader must fill exactly one header line");

// The region's layout for slots of up to `lanes` lanes and `kernel_state_bytes` of the kernels' own state.
// The host and the kernels build it from the same values -- the device's DRAM bank count, one endpoint per
// bank, and the kernels' state; the kernels get them as TENSOR_PREFETCHER_LANES and
// TENSOR_PREFETCHER_KERNEL_STATE_BYTES -- so a single-bank device pays for one lane per slot and one
// endpoint's sender state.
struct RegionLayout {
    uint32_t lanes = 1;
    // Whole cache lines.
    uint32_t kernel_state_bytes = 0;

    constexpr uint32_t slot_bytes() const { return kSlotHeaderBytes + lanes * kLanePageBytes; }
    constexpr uint32_t slot_offset(uint32_t slot_count) const {
        return kControlBytes + (slot_count % kFifoSlots) * slot_bytes();
    }
    constexpr uint32_t staged_offset() const { return kControlBytes + kFifoSlots * slot_bytes(); }
    constexpr uint32_t pushed_offset() const { return staged_offset() + kMaxReceivers * kLineBytes; }
    constexpr uint32_t sender_state_zone_offset() const { return pushed_offset() + kMaxReceivers * kLineBytes; }
    constexpr uint32_t sender_state_zone_bytes() const { return lanes * kSenderStateBytesPerEndpoint; }
    // The kernels' state, each kernel's in whole cache lines, in kernel order.
    constexpr uint32_t kernel_state_zone_offset() const {
        return sender_state_zone_offset() + sender_state_zone_bytes();
    }
    constexpr uint32_t stage_ring_offset() const { return kernel_state_zone_offset() + kernel_state_bytes; }
    constexpr uint32_t min_region_bytes() const { return stage_ring_offset() + kMinStageRingBytes; }
    // Stage ring bytes in a region of `region_bytes`, whole cache lines.
    constexpr uint32_t stage_ring_bytes(uint32_t region_bytes) const {
        return region_bytes < stage_ring_offset() ? 0 : (region_bytes - stage_ring_offset()) / kLineBytes * kLineBytes;
    }
};
static_assert(RegionLayout{kMaxLanes}.slot_bytes() % kLineBytes == 0, "FIFO slots must be whole cache lines");

// Bytes of one stage slot when a ring of `ring_bytes` serves `num_receivers` receivers, whole cache lines.
// A block must fit in one slot.
constexpr uint32_t stage_slot_bytes(uint32_t ring_bytes, uint32_t num_receivers) {
    return ring_bytes / (num_receivers * kStageSlotsPerReceiver) / kLineBytes * kLineBytes;
}
// The kernels run only where the ring holds kMinStageRingBytes, and the host caps a request at
// kMaxReceivers, so no stage slot is smaller than this.
static_assert(stage_slot_bytes(kMinStageRingBytes, kMaxReceivers) == 1024, "The smallest stage slot is 1 KiB");

}  // namespace tt::tt_metal::tensor_prefetcher_fifo
