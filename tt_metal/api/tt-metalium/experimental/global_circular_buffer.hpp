// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental DRAM-sender extension to GlobalCircularBuffer. This is the opt-in API for
// constructing a GCB whose senders are programmable DRAM cores (Blackhole DRISCs) rather
// than worker cores. The public surface of `GlobalCircularBuffer` is unchanged; the
// DRAM-sender mode is accessed through the free functions declared here.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// Sender domain for a GlobalCircularBuffer. Worker = standard sharded GCB where senders
// are worker cores hosting their own slice of the cb_buffer in L1. Dram = senders are
// programmable DRAM cores (Blackhole DRISCs) that own their staging L1 separately; the
// cb_buffer is sharded over receivers only, and the receiver-side config_buffer
// remote_pages_addr_override slot points at DRISC L1 so the receiver's pages_acked
// NoC-inc lands on the DRISC side.
enum class SenderCoreType : uint8_t {
    Worker = 0,
    Dram = 1,
};

// Construct a GlobalCircularBuffer where senders are programmable DRAM cores identified
// by DRAM bank id. Each bank id is mapped internally to an unused DRAM subchannel (one
// that the SOC descriptor does not already reserve as a worker/eth endpoint). Receiver
// sets across senders must be disjoint and must not collide with the resolved DRAM-sender
// physical NOC coords.
//
// MeshDevice-only: the arena that backs this GCB's pages_sent allocation lives on
// MeshDeviceImpl, so a bare IDevice cannot construct one.
GlobalCircularBuffer CreateGlobalCircularBufferWithDramSenders(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Read-only accessors for the DRAM-sender state inside a GlobalCircularBuffer. For
// GCBs created via the worker-sender path these return SenderCoreType::Worker / 0 /
// empty respectively.
SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb);

// DRISC unreserved-L1 base where the sender's per-receiver pages_sent/acked counters
// live. Zero for worker-sender GCBs.
DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb);

// Worker-L1 offset (inside the receiver's config buffer page) where the receiver's
// local pages_sent counter lives. Zero for worker-sender GCBs.
DeviceAddr pages_sent_worker_l1_base(const GlobalCircularBuffer& gcb);

// DRISC L1 base of the per-GCB "sender state block" — RemoteSenderCBInterface bytes,
// prefetcher mutable state (fifo_wr_ptr, pages_sent, stage_slot, has_next), the sender
// config block, and the receiver NOC XY table. Pre-written by the GCB constructor on
// every (device, sender_core). The DRAM-core prefetcher kernel memcpys this block
// into its static cb_interface slot on each request that targets this GCB, runs the
// chunk loop, and writes the mutable fields back so state survives multi-GCB request
// switching. Layout: tt_metal/impl/buffers/dram_sender_state_block.hpp. Zero for
// worker-sender GCBs.
DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb);

// Physical worker NOC XY for each sender's receivers. The DRISC kernel uses these as
// runtime args. Empty for worker-sender GCBs.
const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender(const GlobalCircularBuffer& gcb);

}  // namespace experimental
}  // namespace tt::tt_metal
