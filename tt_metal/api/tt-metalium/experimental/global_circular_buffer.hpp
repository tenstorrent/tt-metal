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
// `support_multi_receiver_shards=true` declares that a bank's shard may be consumed by
// more than one receiver — the legacy interleaved DRAM layout, where a single read pulls data for
// every receiver on the bank. Because the receivers share that data, the bank must be driven by a
// single sender core (the free non-endpoint subchannel on NOC0).
//
// The default, false, promises the opposite: each receiver owns a disjoint, contiguous shard (the
// receiver-contiguous layout). With no shared data between receivers, a bank holding two or more
// receivers can then be driven by two DRISC sender cores (the free subchannel plus the bank's
// NOC1-endpoint subchannel, both on NOC0), splitting the bank's receivers ceil/floor across them
// to roughly double per-bank bandwidth. A single-receiver bank cannot split one receiver across
// two senders, so it falls back to a single sender with its secondary core left parked —
// single- and dual-sender banks may therefore coexist in one GCB. The Tensor prefetcher always
// provisions both cores and routes PREFETCH requests only to this GCB's mapped sender subset.
//
// MeshDevice-only: the arena that backs this GCB's pages_sent allocation lives on
// MeshDeviceImpl, so a bare IDevice cannot construct one.
GlobalCircularBuffer CreateGlobalCircularBufferForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = false);

// Sender domain of a GlobalCircularBuffer. Returns SenderCoreType::Worker for GCBs created
// via the worker-sender path, SenderCoreType::Dram for those from
// CreateGlobalCircularBufferForTensorPrefetcher.
SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb);

// The impl-internal DRAM-sender L1-layout accessors (pages_sent / sender-state / receiver
// coords / slab indices) are consumed only inside tt_metal/ and live in
// tt_metal/impl/buffers/global_circular_buffer_dram_sender_internal.hpp.

}  // namespace experimental
}  // namespace tt::tt_metal
