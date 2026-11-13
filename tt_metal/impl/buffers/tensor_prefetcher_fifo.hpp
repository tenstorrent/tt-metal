// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the host and the Tensor prefetcher kernels that fast dispatch runs on a dispatch engine
// (tt_metal/impl/buffers/kernels/tensor_prefetcher_{reader,pusher}.cpp).
//
// Those two kernels share one L1 region: the part of the dispatch engine's L1 that the dispatch memory
// map leaves above every CQ zone. Offsets below are from the base of that region. The host writes
// requests into a FIFO there and both kernels read each slot in place; the slot is free again once
// both kernels have retired it.
//
// Every word that another agent reads or writes sits alone on a cache line and is accessed through
// the uncached L1 alias: host writes and NoC traffic do not snoop the data-movement caches.

#include <stdint.h>

#include "tensor_prefetcher_request.hpp"

namespace tt::tt_metal::tensor_prefetcher_fifo {

inline constexpr uint32_t kLineBytes = 64;

// Control block, one word per line.
// Non-zero tells both kernels to exit. Written by the host when fast dispatch tears down.
inline constexpr uint32_t kTerminateOffset = 0 * kLineBytes;
// Slots the host has written so far (monotonic).
inline constexpr uint32_t kFifoWrOffset = 1 * kLineBytes;
// Slots the reader / the pusher has retired so far (monotonic).
inline constexpr uint32_t kReaderRdOffset = 2 * kLineBytes;
inline constexpr uint32_t kPusherRdOffset = 3 * kLineBytes;
inline constexpr uint32_t kControlBytes = 4 * kLineBytes;

// The rest of the region, in order: a FIFO of kFifoSlots request slots, the per-receiver progress
// counters, the PrefetcherPipe sender state of the DRAM-sender endpoints placed on this dispatch engine,
// and the stage ring. Its size depends on how many lanes a slot carries, so RegionLayout below gives the
// offsets for a lane count.
//
// One slot = a header naming the DRAM bank behind each lane, then one request page per lane (the Tensor
// prefetcher request page, tensor_prefetcher_request.hpp). A lane is one DRAM bank's sender endpoint; the
// dispatch engine holds one endpoint per bank. A bank's endpoint is implied by the H2D socket a page arrives
// on when the sender is a DRAM core, so here the bank travels in the header instead.
inline constexpr uint32_t kLanePageBytes = kRequestPageBytes;
inline constexpr uint32_t kSlotHeaderBytes = kLineBytes;
inline constexpr uint32_t kFifoSlots = 16;

// Per-receiver progress between the two kernels: blocks staged by the reader, and blocks pushed (so
// their stage slots are free again) by the pusher. One monotonic word per line, one writer each.
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
    // Retired by both kernels with no other effect. Lets the host check that the FIFO is serviced.
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

// The region's layout for slots of up to `lanes` lanes. The host and the kernels build it from the same
// count -- the device's DRAM bank count, one endpoint per bank; the kernels get it as TENSOR_PREFETCHER_LANES
// -- so a single-bank device pays for one lane per slot and one endpoint's sender state.
struct RegionLayout {
    uint32_t lanes = 1;

    constexpr uint32_t slot_bytes() const { return kSlotHeaderBytes + lanes * kLanePageBytes; }
    constexpr uint32_t slot_offset(uint32_t slot_count) const {
        return kControlBytes + (slot_count % kFifoSlots) * slot_bytes();
    }
    constexpr uint32_t staged_offset() const { return kControlBytes + kFifoSlots * slot_bytes(); }
    constexpr uint32_t pushed_offset() const { return staged_offset() + kMaxReceivers * kLineBytes; }
    constexpr uint32_t sender_state_zone_offset() const { return pushed_offset() + kMaxReceivers * kLineBytes; }
    constexpr uint32_t sender_state_zone_bytes() const { return lanes * kSenderStateBytesPerEndpoint; }
    constexpr uint32_t stage_ring_offset() const { return sender_state_zone_offset() + sender_state_zone_bytes(); }
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

}  // namespace tt::tt_metal::tensor_prefetcher_fifo
