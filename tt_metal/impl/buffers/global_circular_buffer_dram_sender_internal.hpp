// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Impl-internal DRAM-sender accessors for GlobalCircularBuffer. These read back the L1 layout
// that the DRAM-sender GCB constructor stamps out; they are consumed only inside tt_metal/ (the
// Tensor prefetcher manager and the DRAM-sender GCB tests), so they live here rather than on the
// public experimental surface in tt-metalium/experimental/global_circular_buffer.hpp. The public
// header keeps only what ttnn consumes: SenderCoreType, sender_core_type(), and
// CreateGlobalCircularBufferForTensorPrefetcher().

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace tt::tt_metal::experimental {

// DRISC unreserved-L1 base where the sender's per-receiver pages_sent/acked counters
// live. Zero for worker-sender GCBs.
DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb);

// Receiver-L1 offset where the receiver's local pages_sent counter lives: inside the config buffer page in
// worker L1 for worker receivers, the credit line of the receiver config page in DRISC L1 for DRAM receivers.
// Zero for worker-sender GCBs.
DeviceAddr pages_sent_receiver_l1_base(const GlobalCircularBuffer& gcb);

// DRISC L1 base of the per-GCB "sender state block" — the RemoteSenderCBInterface
// bytes (including the fifo_wr_ptr that persists across requests), the sender config
// block, and the receiver NOC XY table. Pre-written by the GCB constructor on every
// (device, sender_core). The Tensor prefetcher kernel loads this block into its
// static cb_interface slot on each request that targets this GCB, runs the chunk loop,
// and writes fifo_wr_ptr back so the ring offset survives multi-GCB request switching.
// Layout: tt_metal/impl/buffers/dram_sender_state_block.hpp. Zero for worker-sender GCBs.
DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb);
}  // namespace tt::tt_metal::experimental

namespace tt::tt_metal {
class DriscL1Allocation;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {
// The DRISC L1 range holding that block and its pages_sent counters, released when the last copy of
// the GCB goes. The prefetcher keeps a weak reference to tell whether the block is still this GCB's.
// Null for worker-sender GCBs.
std::shared_ptr<DriscL1Allocation> sender_state_drisc_l1_allocation(const GlobalCircularBuffer& gcb);

// The receiver-zone range holding each DRAM receiver's config page and FIFO, which the senders store data and
// credits into. Null unless the GCB has DRAM receivers.
std::shared_ptr<DriscL1Allocation> receiver_drisc_l1_allocation(const GlobalCircularBuffer& gcb);

// Each sender's receivers as logical coords, row-wise (the order the DRISC kernel's slab
// indices follow). Logical rather than physical because the physical worker coord depends on
// which device of the mesh is being addressed. Empty for worker-sender GCBs.
const std::vector<std::vector<CoreCoord>>& receiver_logical_cores_per_sender(const GlobalCircularBuffer& gcb);

}  // namespace tt::tt_metal::experimental
