// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared DRAM-sender topology helpers: turning a caller's (bank id -> receivers) request into
// the (DRAM-logical sender core -> receivers) mapping the remote-buffer objects are built from,
// and deriving each sender's bank-local slab base from that mapping.
//
// Used by both DRAM-sender transports -- GlobalCircularBuffer
// (CreateGlobalCircularBufferForTensorPrefetcher) and PrefetcherPipe
// (CreatePrefetcherPipesForTensorPrefetcher) -- so the two agree on sender placement, the
// dual-sender receiver split, and slab numbering. The Tensor prefetcher's receiver-contiguous
// contract depends on all three matching what the consumer op computes independently.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

class IDevice;

namespace distributed {
class MeshDevice;
}  // namespace distributed

// How many DRISC sender cores drive one DRAM bank. TwoPerBank splits a bank's receivers across
// both of its senders; OnePerBank is what a shard feeding more than one receiver requires.
enum class DramSenderSplit : uint8_t { OnePerBank, TwoPerBank };

// Map (bank_id, receivers) pairs to (DRAM-logical CoreCoord, receivers) pairs. With
// DramSenderSplit::TwoPerBank each bank is driven by two DRISC sender cores (a free non-endpoint
// subchannel on NOC0 and the NOC1-endpoint subchannel, also running on NOC0): the bank's ordered
// receiver list is split ceil/floor across them, so each core delivers roughly half. Receiver
// order is preserved (receiver-table order == bank-local slab order, the recv-contig contract);
// the second sender's slabs start where the first sender's receivers end (tracked host-side via
// recv_index_base, whose per-bank reset assumes a bank's senders are contiguous in this mapping --
// hence the no-duplicate-bank guard).
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    DramSenderSplit split);

// One sender mapping serves the whole mesh because a logical DRAM coord names an endpoint role
// (see metal_SocDescriptor::dram_bank_endpoint_coords). Check that once here rather than trusting
// it: a descriptor whose endpoint layout didn't reproduce the role would silently drive the wrong
// DRISC core, and a sender that isn't provisioned for its bank would take credits nobody returns.
void validate_dram_senders_across_mesh(
    distributed::MeshDevice* mesh_device, const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping);

// Per-sender bank-local recv_index_base. Senders are ordered [bank b s0, bank b s1, bank b+1 s0,
// ...] (sender_logical.x == bank_id); recv_index_base resets to 0 on a bank change and accumulates
// within a bank (dual senders share a bank). Returns one value per sender in mapping order. Single
// source for the request-header stamping and for a consumer's own slab accounting.
std::vector<uint32_t> recv_index_bases_per_sender(const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping);

// Write `bytes` into DRAM-logical sender core `sender_logical`'s own L1 on `device`, at DRISC-L1
// address `local_addr`.
//
// Host writes to a DRAM core's L1 go over NOC and need the DRAM-L1 NOC offset added on top of the
// local address -- worker L1 has local == NOC space, so a worker-side write does not, and forgetting
// it here lands the bytes somewhere harmless-looking instead of failing. The sender's virtual coord
// is resolved per device because DRAM harvesting can place it differently on each.
void write_dram_sender_l1(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& sender_logical,
    DeviceAddr local_addr,
    std::span<const std::byte> bytes);

}  // namespace tt::tt_metal
