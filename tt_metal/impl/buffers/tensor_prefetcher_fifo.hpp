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

// One slot = a header naming the DRAM bank behind each lane, then one 128 B request page per lane
// (the Tensor prefetcher request page, tensor_prefetcher_request.hpp). A lane is one DRAM-bank sender
// endpoint; a bank's endpoint is implied by the H2D socket a page arrives on when the sender is a DRAM
// core, so here the bank travels in the header instead.
inline constexpr uint32_t kMaxLanes = 8;
inline constexpr uint32_t kLanePageBytes = 128;
inline constexpr uint32_t kSlotHeaderBytes = kLineBytes;
inline constexpr uint32_t kSlotBytes = kSlotHeaderBytes + kMaxLanes * kLanePageBytes;
inline constexpr uint32_t kFifoSlots = 16;
inline constexpr uint32_t kFifoOffset = kControlBytes;
inline constexpr uint32_t kFifoBytes = kFifoSlots * kSlotBytes;

// Per-receiver progress between the two kernels: blocks staged by the reader, and blocks pushed (so
// their stage slots are free again) by the pusher. One monotonic word per line, one writer each.
inline constexpr uint32_t kMaxReceivers = 32;
inline constexpr uint32_t kStagedOffset = kFifoOffset + kFifoBytes;
inline constexpr uint32_t kPushedOffset = kStagedOffset + kMaxReceivers * kLineBytes;

// PrefetcherPipe sender config pages of the DRAM-bank endpoints placed on this dispatch engine. Fixed
// size, so carving a pipe never moves the stage ring under a running kernel.
inline constexpr uint32_t kSenderStateZoneOffset = kPushedOffset + kMaxReceivers * kLineBytes;
inline constexpr uint32_t kSenderStateZoneBytes = 4096;

// The rest of the region stages blocks between the reader and the pusher. While a tensor streams,
// the ring is split evenly between that tensor's receivers, kStageSlotsPerReceiver blocks each, so
// the reader can keep reading for every receiver whichever one the pusher is waiting on. One slot
// per receiver is the minimum that cannot deadlock; the second overlaps a read with a push.
inline constexpr uint32_t kStageRingOffset = kSenderStateZoneOffset + kSenderStateZoneBytes;
inline constexpr uint32_t kMinStageRingBytes = 64 * 1024;
inline constexpr uint32_t kMinRegionBytes = kStageRingOffset + kMinStageRingBytes;
inline constexpr uint32_t kStageSlotsPerReceiver = 2;

enum SlotCommand : uint32_t {
    // Retired by both kernels with no other effect. Lets the host check that the FIFO is serviced.
    kSlotNop = 1,
    // num_lanes request pages follow the header, one per lane, all PREFETCH or all DRAIN. A PREFETCH
    // slot's lanes carry the same tensors for different receivers, and its receivers are served
    // together rather than lane by lane.
    kSlotRequest = 2,
};

struct SlotHeader {
    uint32_t command;
    uint32_t num_lanes;
    uint32_t lane_bank_id[kMaxLanes];
    uint32_t reserved[(kSlotHeaderBytes / sizeof(uint32_t)) - 2 - kMaxLanes];
};
static_assert(sizeof(SlotHeader) == kSlotHeaderBytes, "SlotHeader must fill exactly one header line");
static_assert(kSlotBytes % kLineBytes == 0, "FIFO slots must be whole cache lines");

constexpr uint32_t slot_offset(uint32_t slot_count) { return kFifoOffset + (slot_count % kFifoSlots) * kSlotBytes; }

// Stage ring bytes in a region of `region_bytes`, whole cache lines.
constexpr uint32_t stage_ring_bytes(uint32_t region_bytes) {
    return region_bytes < kStageRingOffset ? 0 : (region_bytes - kStageRingOffset) / kLineBytes * kLineBytes;
}

// TensorPrefetcherTensorLayout::layout_mode of a receiver-contiguous weight, the only layout these kernels serve.
inline constexpr uint32_t kReceiverContiguousLayout = 1;

}  // namespace tt::tt_metal::tensor_prefetcher_fifo
