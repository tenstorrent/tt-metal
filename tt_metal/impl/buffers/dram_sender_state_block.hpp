// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// L1 byte layout of the per-GCB "sender state block" that lives in DRISC L1
// for DRAM-sender GlobalCircularBuffers. The GCB constructor pre-initializes
// one block per sender core (constants + initial mutable state) so the
// long-running DRAM-core prefetcher kernel can switch between multiple GCBs
// across successive requests, preserving each GCB's ring-buffer state
// (fifo_wr_ptr, pages_sent, stage_slot, has_next) across the gap.
//
// The first 32 bytes are byte-compatible with hw/inc/internal/circular_buffer_interface.h
// RemoteSenderCBInterface so the kernel can memcpy this block into the static
// cb_interface[] slot on each request, run the existing chunk-loop logic
// unchanged, and memcpy the mutable fields back at request end.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// ----- byte offsets, shared with the DRISC kernel (must match) -----
inline constexpr uint32_t kDramSenderStateBlockConfigPtrOffset = 0x00;            // const: ptr to config_block below
inline constexpr uint32_t kDramSenderStateBlockFifoStartAddrOffset = 0x04;        // const: receiver buffer base
inline constexpr uint32_t kDramSenderStateBlockFifoLimitOffset = 0x08;            // per-tensor mutable
inline constexpr uint32_t kDramSenderStateBlockFifoPageSizeOffset = 0x0C;         // per-tensor mutable
inline constexpr uint32_t kDramSenderStateBlockFifoWrPtrOffset = 0x10;            // persistent mutable
inline constexpr uint32_t kDramSenderStateBlockReceiverNocXyPtrOffset = 0x14;     // const: ptr to noc_xy table below
inline constexpr uint32_t kDramSenderStateBlockAlignedPagesSentPtrOffset = 0x18;  // const: pages_sent slot base
inline constexpr uint32_t kDramSenderStateBlockNumRecvAndRemotePtrOffset = 0x1C;  // const: packed
// ----- 32 B mark: RemoteSenderCBInterface byte-compatible region ends here -----
inline constexpr uint32_t kDramSenderStateBlockStageSlotOffset = 0x20;  // persistent mutable
inline constexpr uint32_t kDramSenderStateBlockHasNextOffset = 0x24;    // persistent mutable
inline constexpr uint32_t kDramSenderStateBlockPagesSentOffset = 0x28;  // persistent mutable (sender-local counter)
inline constexpr uint32_t kDramSenderStateBlockReservedOffset = 0x2C;   // padding/future use
// ----- 48 B mark: prefetcher-extra mutable state ends here -----
inline constexpr uint32_t kDramSenderStateBlockConfigBlockOffset =
    0x30;  // 4 uint32s: is_sender, num_receivers, buffer_address, fifo_size
inline constexpr uint32_t kDramSenderStateBlockReceiverNocXyTableOffset = 0x40;  // 2 * num_receivers uint32s

inline constexpr uint32_t kDramSenderStateBlockHeaderBytes = 0x40;  // bytes before the variable-size noc_xy table

// Total size of one block: header + 2 * num_receivers * sizeof(uint32_t).
inline constexpr uint32_t dram_sender_state_block_size(uint32_t num_receivers) {
    return kDramSenderStateBlockHeaderBytes + 2u * num_receivers * sizeof(uint32_t);
}

// Host-side POD view of the block. Used by GlobalCircularBuffer ctor to compose
// the initial bytes before writing to L1 via WriteToDeviceL1. Trailing
// receiver_noc_xy[] entries are appended by the caller after this struct's
// bytes since their count is dynamic.
struct DramSenderStateBlockHeader {
    // RemoteSenderCBInterface byte-compatible region (32 B).
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;
    uint32_t fifo_wr_ptr;
    uint32_t receiver_noc_xy_ptr;
    uint32_t aligned_pages_sent_ptr;
    uint32_t num_receivers_and_remote_pages_sent_ptr;
    // Prefetcher-extra mutable state (16 B).
    uint32_t stage_slot;
    uint32_t has_next;
    uint32_t pages_sent;
    uint32_t reserved0;
    // Sender-side config block read by kernel helpers (16 B).
    uint32_t is_sender;
    uint32_t num_receivers;
    uint32_t buffer_address;
    uint32_t fifo_size_per_receiver;
};

static_assert(sizeof(DramSenderStateBlockHeader) == kDramSenderStateBlockHeaderBytes, "layout drift");
static_assert(offsetof(DramSenderStateBlockHeader, fifo_wr_ptr) == kDramSenderStateBlockFifoWrPtrOffset);
static_assert(offsetof(DramSenderStateBlockHeader, pages_sent) == kDramSenderStateBlockPagesSentOffset);
static_assert(offsetof(DramSenderStateBlockHeader, is_sender) == kDramSenderStateBlockConfigBlockOffset);

}  // namespace tt::tt_metal
