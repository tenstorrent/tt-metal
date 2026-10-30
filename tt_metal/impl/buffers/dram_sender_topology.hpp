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
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-umd/types/io_window_config.hpp>

struct metal_SocDescriptor;

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
// hence the no-duplicate-bank guard). Each transport reads the bases off this mapping once, when
// it builds its senders, so the resulting order is a convention of what the factory returns and
// not something a later caller has to reproduce.
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    const distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    DramSenderSplit split);

// One sender mapping serves the whole mesh because a logical DRAM coord names an endpoint role
// (see metal_SocDescriptor::dram_bank_endpoint_coords). Check that once here rather than trusting
// it: a descriptor whose endpoint layout didn't reproduce the role would silently drive the wrong
// DRISC core, and a sender that isn't provisioned for its bank would take credits nobody returns.
void validate_dram_senders_across_mesh(
    const distributed::MeshDevice* mesh_device, const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping);

// Per-sender bank-local recv_index_base. Senders are ordered [bank b s0, bank b s1, bank b+1 s0,
// ...] (sender_logical.x == bank_id); recv_index_base resets to 0 on a bank change and accumulates
// within a bank (dual senders share a bank). Returns one value per sender in mapping order. Called
// once per target, where the mapping is built: a GCB derives its bases from its own immutable
// mapping, and CreatePrefetcherPipesForTensorPrefetcher hands each pipe the base it keeps. Single
// source for those and for a consumer's own slab accounting.
std::vector<uint32_t> recv_index_bases_per_sender(const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping);

struct DramCoreL1Address {
    CoreCoord virtual_core;
    uint64_t noc_addr;
};

// Where DRAM-logical core `dram_logical`'s own L1 on `device`, at DRISC-L1 address `local_addr`, is
// reached over NOC.
//
// Access to a DRAM core's L1 goes over NOC and needs the DRAM-L1 NOC offset added on top of the
// local address -- worker L1 has local == NOC space, so a worker-side access does not, and forgetting
// it here lands the bytes somewhere harmless-looking instead of failing. The core's virtual coord
// is resolved per device because DRAM harvesting can place it differently on each.
DramCoreL1Address dram_core_l1_address(
    const distributed::MeshDevice& mesh_device, IDevice* device, const CoreCoord& dram_logical, DeviceAddr local_addr);

// Write `bytes` into DRAM-logical core `dram_logical`'s own L1 on `device`, at DRISC-L1 address
// `local_addr`: a sender's state, or a DRAM receiver's config page. An unset `ordering` takes the
// Cluster's default for a DRAM core.
void write_dram_core_l1(
    const distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& dram_logical,
    DeviceAddr local_addr,
    std::span<const std::byte> bytes,
    std::optional<tt::umd::IoOrdering> ordering = std::nullopt);

// Read `bytes.size()` bytes of DRAM-logical core `dram_logical`'s own L1 on `device`, from DRISC-L1
// address `local_addr`.
void read_dram_core_l1(
    const distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& dram_logical,
    DeviceAddr local_addr,
    std::span<std::byte> bytes);

// Package index of a DRAM core: its channel times the DRAM cores per channel, plus its subchannel. On
// Quasar this is the CCE index cce_sram_address takes. Two logical coords can name one core.
uint32_t package_dram_core_index(const metal_SocDescriptor& soc_desc, const CoreCoord& dram_logical);

}  // namespace tt::tt_metal
