// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TL1 layout and status of a Quasar CCE Tensor prefetcher sender, shared by the host
// (TensorPrefetcherManager, Tl1RingTransport) and the CCE sender kernel
// (kernels/tensor_prefetcher_cce.cpp).
//
// The sender's control block sits at the front of its kernel region, at the same TL1 offset on every
// sender CCE. Offsets below are from that base:
//   CQ signal slots: kNumCqSignalSlots slots, one line each; the dispatcher writes, the kernel polls
//   ring meta:       RingMeta, a host line (bytes_sent) then a kernel line (bytes_acked)
//   status:          CceSenderStatus on its own line; the kernel writes, the host reads
//   request ring:    kCceRequestRingSlots slots of kRequestPageBytes
// The stage ring follows the control block: stage_ring_bytes (a runtime arg) in three equal slots, each on whole
// 64 B lines. A PREFETCH copies each chunk from GDDR into one slot and from there into the receivers' FIFOs.
//
// A CCE cannot invalidate its L1 cache, so every word one agent writes and another polls sits alone on a
// 64 B line, and the kernel reads and writes all of them through the uncached alias.
//
// The host zeroes the CQ slots, ring meta and status before each launch; the kernel never zeroes them.
// The host writes a request page into slot (bytes_sent / kRequestPageBytes) % kCceRequestRingSlots, then
// stores the new running byte count to bytes_sent. The kernel copies the page out, handles it, and stores
// its own running byte count to bytes_acked. Both counts wrap at 2^32, which a whole number of rings
// divides.
//
// Shared by host and CCE kernels; C++17.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

namespace tt::tt_metal {

inline constexpr uint32_t kCceSenderLineBytes = 64;
inline constexpr uint32_t kCceRequestRingSlots = 128;
inline constexpr uint32_t kCceRequestRingBytes = kCceRequestRingSlots * kRequestPageBytes;

struct RingMeta {
    uint32_t bytes_sent;
    uint32_t host_line_pad[(kCceSenderLineBytes / sizeof(uint32_t)) - 1];
    uint32_t bytes_acked;
    uint32_t kernel_line_pad[(kCceSenderLineBytes / sizeof(uint32_t)) - 1];
};

static_assert(offsetof(RingMeta, bytes_acked) == kCceSenderLineBytes);
static_assert(sizeof(RingMeta) == 2 * kCceSenderLineBytes);

// Where the sender is. NotStarted is what the host's zeroing leaves.
enum class CceSenderState : uint32_t {
    NotStarted = 0,
    Running = 1,
    // In WAIT_CQ, polling a CQ signal slot.
    WaitingCq = 2,
    // In DRAIN, waiting for a target's receivers to ack.
    Draining = 3,
    // Exited on STOP.
    Stopped = 4,
    // Exited on an error; code says which.
    Failed = 5,
};

enum class CceSenderStatusCode : uint32_t {
    Ok = 0,
    // A PREFETCH free-space wait expired; receiver, sent and acked name the short receiver.
    CreditTimeout = 1,
    // A DRAIN barrier expired; target_state_addr is the target, and receiver, sent and acked its first receiver
    // that had not acked everything.
    DrainTimeout = 2,
    // A WAIT_CQ poll expired; receiver is the CQ index, sent the awaited value, acked the slot's last value.
    CqTimeout = 3,
    // An unknown command, a command this sender does not handle, or a DRAIN naming a PrefetcherPipe.
    BadCmd = 4,
    // A PREFETCH layout this sender cannot deliver.
    BadLayout = 5,
};

// Written by the kernel at every state change; state is stored last, behind a fence.
struct CceSenderStatus {
    uint32_t state;
    uint32_t code;
    // Requests the sender had consumed before the one this status is about.
    uint32_t request_index;
    uint32_t target_state_addr;
    uint32_t receiver;
    uint32_t sent;
    uint32_t acked;
};

static_assert(sizeof(CceSenderStatus) <= kCceSenderLineBytes);

inline constexpr uint32_t kCceCqSignalSlotsOffset = 0;
inline constexpr uint32_t kCceCqSignalSlotStrideBytes = kCceSenderLineBytes;
inline constexpr uint32_t kCceRingMetaOffset =
    kCceCqSignalSlotsOffset + kNumCqSignalSlots * kCceCqSignalSlotStrideBytes;
inline constexpr uint32_t kCceSenderStatusOffset = kCceRingMetaOffset + sizeof(RingMeta);
inline constexpr uint32_t kCceRequestRingOffset = kCceSenderStatusOffset + kCceSenderLineBytes;
// The part of the control block the host zeroes before launch: CQ slots, ring meta and status.
inline constexpr uint32_t kCceSenderZeroedBytes = kCceRequestRingOffset;
inline constexpr uint32_t kCceSenderControlBytes = kCceRequestRingOffset + kCceRequestRingBytes;

static_assert(kCceRingMetaOffset % kCceSenderLineBytes == 0);
static_assert(kCceRequestRingOffset % kCceSenderLineBytes == 0);
static_assert(kCceSenderControlBytes % kCceSenderLineBytes == 0);
static_assert((uint64_t{1} << 32) % kCceRequestRingBytes == 0);

// TensorPrefetcherTensorLayout::layout_mode of a receiver-contiguous tensor, the only layout a CCE sender delivers.
inline constexpr uint32_t kCceReceiverContiguousLayoutMode = 1;

// The CCE sender kernel's runtime args. The budget is rdcycle cycles per bounded wait, 0 = infinite. The sender reads
// tensors from its own DRAM view only: GDDR offset view offset + bank-local address on its own Mimir. The view offset
// is 64-bit because a view can start 4 GiB or more into its Mimir.
inline constexpr uint32_t kCceSenderArgControlBase = 0, kCceSenderArgBudgetLo = 1, kCceSenderArgBudgetHi = 2,
                          kCceSenderArgStageRingBytes = 3, kCceSenderArgMimir = 4, kCceSenderArgViewOffsetLo = 5,
                          kCceSenderArgViewOffsetHi = 6, kCceSenderArgs = 7;

}  // namespace tt::tt_metal
