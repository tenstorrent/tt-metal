// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TL1 layout of a DRAM-sender GlobalCircularBuffer whose receivers are DRAM cores (Quasar CCEs):
// each receiver's config page, and the per-receiver credit counters in the sender's state.
//
// A CCE cannot invalidate its L1 cache, so every word one agent writes and another polls sits alone
// on a 64 B line, and each side reaches it through the uncached alias.
//
// Receiver config page, one per receiver CCE, at the same TL1 offset on every receiver:
//   line 0 (host-written once): fifo_start_addr, fifo_size, sender_index, remote_pages_acked_addr
//   line 1 (credit line):       pages_sent, the absolute count the sender stores; the receiver polls it
//   line 2 (receiver-written):  fifo_rd_ptr, pages_acked (this receiver's own absolute ack count)
// The receiver FIFO follows the page. Every receiver's credit line is at the same offset, so the sender
// stores pages_sent to that one offset on each receiver; it does not step per receiver as a Blackhole
// sender does.
//
// Sender credit counters, at the sender state's aligned_pages_sent_ptr:
//   [pages_sent block: num_receivers x kDramReceiverCreditStrideBytes]
//   [pages_acked block: num_receivers x kDramReceiverCreditStrideBytes]
// Receiver r's counters are at r x stride in each block. The sender writes pages_sent; receiver r
// stores its absolute pages_acked to remote_pages_acked_addr, which names its slot in the acked block.
// num_receivers is the one in the sender's DramSenderStateBlock.
// The host also writes the sender's state block and counters before launch, so the sender reads them
// through the uncached alias too.
//
// Shared by host and CCE kernels; C++17.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

inline constexpr uint32_t kDramReceiverLineBytes = 64;
inline constexpr uint32_t kDramReceiverCreditStrideBytes = kDramReceiverLineBytes;
// pages_sent and pages_acked count FIFO bytes in these units, as a Blackhole remote CB counts
// REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE units. Page sizes and the FIFO size are multiples of it.
inline constexpr uint32_t kDramReceiverCreditUnitBytes = 16;

inline constexpr uint32_t kDramReceiverFifoStartAddrOffset = 0;
inline constexpr uint32_t kDramReceiverFifoSizeOffset = 4;
// Package CCE index of the sender, as cce_sram_address takes it.
inline constexpr uint32_t kDramReceiverSenderIndexOffset = 8;
// TL1 offset, on the sender, of this receiver's pages_acked slot.
inline constexpr uint32_t kDramReceiverRemotePagesAckedAddrOffset = 12;
inline constexpr uint32_t kDramReceiverCreditLineOffset = kDramReceiverLineBytes;
inline constexpr uint32_t kDramReceiverStateLineOffset = 2 * kDramReceiverLineBytes;
inline constexpr uint32_t kDramReceiverFifoRdPtrOffset = kDramReceiverStateLineOffset;
inline constexpr uint32_t kDramReceiverPagesAckedOffset = kDramReceiverStateLineOffset + 4;
inline constexpr uint32_t kDramReceiverConfigPageBytes = 3 * kDramReceiverLineBytes;

static_assert(kDramReceiverRemotePagesAckedAddrOffset + 4 <= kDramReceiverCreditLineOffset);
static_assert(kDramReceiverCreditLineOffset % kDramReceiverLineBytes == 0);
static_assert(kDramReceiverStateLineOffset % kDramReceiverLineBytes == 0);
static_assert(kDramReceiverPagesAckedOffset + 4 <= kDramReceiverConfigPageBytes);

// Offset of the pages_acked block from aligned_pages_sent_ptr.
constexpr uint32_t dram_receiver_sender_pages_acked_offset(uint32_t num_receivers) {
    return num_receivers * kDramReceiverCreditStrideBytes;
}

// Bytes of both counter blocks.
constexpr uint32_t dram_receiver_sender_counters_bytes(uint32_t num_receivers) {
    return 2 * num_receivers * kDramReceiverCreditStrideBytes;
}

}  // namespace tt::tt_metal
