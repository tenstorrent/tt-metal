// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sender topology helpers: turning a caller's (bank id -> receivers) request into the
// (DRAM-logical sender core -> receivers) mapping DRAM-sender PrefetcherPipes are built from
// (CreatePrefetcherPipesForTensorPrefetcher), and deriving each sender's bank-local slab base from
// that mapping. The Tensor prefetcher's delivery contract depends on sender placement, the
// dual-sender receiver split, and slab numbering matching what the consumer op computes
// independently.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/context/context_types.hpp"

struct metal_SocDescriptor;

namespace tt::tt_metal {

class IDevice;

namespace distributed {
class MeshDevice;
}  // namespace distributed

// Where the runtime runs the sender endpoints of DRAM-sender PrefetcherPipes. Either way an endpoint is
// named by its DRAM-logical coord (x = bank, y = role), so callers never see the difference; only the
// core holding the endpoint's state and running its kernel changes.
enum class DramSenderPlacement : uint8_t {
    // Each endpoint on its own programmable DRAM core (Blackhole DRISCs).
    DramCore,
    // Every endpoint on the dispatch engine that runs fast dispatch, beside the CQ kernels (Quasar). The
    // Tensor prefetcher's reader and pusher, launched with fast dispatch, serve all of them.
    DispatchEngine,
};

// Placement for a device of context `context_id` running `num_hw_cqs` CQs, or nullopt when it can run no
// DRAM-sender endpoints.
std::optional<DramSenderPlacement> dram_sender_placement(ContextId context_id, uint8_t num_hw_cqs);

// Under DramSenderPlacement::DispatchEngine: the logical dispatch-engine core of device `device_id` that
// holds every DRAM-sender endpoint's state.
CoreCoord dram_sender_dispatch_engine(ContextId context_id, ChipId device_id);

// Virtual NoC coord of the core holding endpoint `sender_logical`'s state on `device`; receivers return
// their credits there.
CoreCoord dram_sender_virtual_core(
    const distributed::MeshDevice& mesh_device, const IDevice* device, const CoreCoord& sender_logical);

// The logical DRAM core of DRAM view `bank_id` whose physical NoC coord the SoC descriptor does not
// already claim as a worker_endpoint or eth_endpoint -- i.e. one safe for a DRISC kernel to occupy.
// Resolved against `soc_desc`'s harvested DRAM topology; `device_id` only names the device in errors.
// Throws if no free subchannel exists, or TT_FATALs if bank_id is out of range. Depends only on the SoC
// descriptor, so device init can name this core in the firmware's tensor_prefetcher_signal_noc_xy table before
// any MeshDevice exists.
CoreCoord pick_unused_dram_logical_core(const metal_SocDescriptor& soc_desc, uint32_t bank_id, ChipId device_id);

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
// once, where the mapping is built: CreatePrefetcherPipesForTensorPrefetcher hands each pipe the
// base it keeps. Single source for those and for a consumer's own slab accounting.
std::vector<uint32_t> recv_index_bases_per_sender(const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping);

// Write `bytes` into the L1 that holds DRAM-logical sender endpoint `sender_logical`'s state on
// `device`, at local address `local_addr`: the DRAM core's own L1, or the dispatch engine's.
//
// Host writes to a DRAM core's L1 go over NOC and need the DRAM-L1 NOC offset added on top of the
// local address -- worker L1 has local == NOC space, so a worker-side write does not, and forgetting
// it here lands the bytes somewhere harmless-looking instead of failing. The sender's virtual coord
// is resolved per device because DRAM harvesting can place it differently on each.
void write_dram_sender_l1(
    const distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& sender_logical,
    DeviceAddr local_addr,
    std::span<const std::byte> bytes);

}  // namespace tt::tt_metal
