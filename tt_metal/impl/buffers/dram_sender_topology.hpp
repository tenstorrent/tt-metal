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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <core_coord.hpp>
#include <device.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal {

// Map (bank_id, receivers) pairs to (DRAM-logical CoreCoord, receivers) pairs. In dual mode each
// bank is driven by two DRISC sender cores (a free non-endpoint subchannel on NOC0 and the
// NOC1-endpoint subchannel, also running on NOC0): the bank's ordered receiver list is split
// ceil/floor across them, so each core delivers roughly half. Receiver order is preserved
// (receiver-table order == bank-local slab order, the recv-contig contract); the second sender's
// slabs start where the first sender's receivers end (tracked host-side via recv_index_base,
// whose per-bank reset assumes a bank's senders are contiguous in this mapping -- hence the
// no-duplicate-bank guard below).
inline std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank) {
    // Sender coords name endpoint roles, so resolving them against any one device gives the
    // mapping for the whole mesh; validate_dram_senders_across_mesh rechecks that per device.
    const auto& devices = mesh_device->get_devices();
    TT_FATAL(
        !devices.empty(),
        "Cannot build a DRAM sender mapping for a mesh with no local devices (shape {}); a submesh whose slots are "
        "all owned by another host/rank has no device to resolve the senders' endpoint roles against.",
        mesh_device->shape());
    const IDevice* reference_device = devices.front();
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve((dual_senders_per_bank ? 2 : 1) * bank_to_receivers.size());
    std::unordered_set<uint32_t> seen_banks;
    for (const auto& [bank_id, receivers] : bank_to_receivers) {
        const uint32_t n = receivers.num_cores();
        TT_FATAL(n > 0, "DRAM bank {} has no receivers", bank_id);
        TT_FATAL(
            seen_banks.insert(bank_id).second,
            "DRAM bank {} appears more than once in bank_to_receivers; each bank must be listed exactly once "
            "(the per-bank recv_index_base / slab assignment assumes one contiguous group of senders per bank).",
            bank_id);

        if (!dual_senders_per_bank) {
            // Single sender per bank (the free non-endpoint subchannel).
            mapping.emplace_back(
                mesh_device->impl().pick_unused_dram_logical_core(reference_device, bank_id), receivers);
            continue;
        }

        // A single receiver cannot be split across two senders. Since the prefetcher always
        // provisions both senders per bank and routes PREFETCH only to the senders this target
        // actually maps, we can map just the primary sender for such a bank and leave the
        // secondary parked -- same as the single-sender path. Dual- and single-sender banks may
        // therefore coexist in one dual-mode target.
        const std::vector<CoreCoord> sender_cores =
            mesh_device->impl().dram_sender_logical_cores(reference_device, bank_id);
        if (n == 1) {
            mapping.emplace_back(sender_cores.at(0), receivers);
            continue;
        }

        // Two sender cores per bank: split the bank's ordered receivers ceil/floor.
        // select_from_corerangeset indices are inclusive and traverse row-wise (matching
        // corerange_to_cores used elsewhere), so the receiver-table / bank-local slab
        // order is preserved.
        const uint32_t first_count = (n + 1) / 2;
        mapping.emplace_back(sender_cores.at(0), select_from_corerangeset(receivers, 0, first_count - 1, true));
        mapping.emplace_back(sender_cores.at(1), select_from_corerangeset(receivers, first_count, n - 1, true));
    }
    return mapping;
}

// One sender mapping serves the whole mesh because a logical DRAM coord names an endpoint role
// (see metal_SocDescriptor::dram_bank_endpoint_coords). Check that once here rather than trusting
// it: a descriptor whose endpoint layout didn't reproduce the role would silently drive the wrong
// DRISC core, and a sender that isn't provisioned for its bank would take credits nobody returns.
inline void validate_dram_senders_across_mesh(
    distributed::MeshDevice* mesh_device, const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
    std::unordered_map<uint32_t, std::vector<CoreCoord>> senders_by_bank;
    for (const IDevice* device : mesh_device->get_devices()) {
        senders_by_bank.clear();
        for (const auto& [sender_logical, _receivers] : mapping) {
            const auto bank_id = static_cast<uint32_t>(sender_logical.x);
            auto [it, inserted] = senders_by_bank.try_emplace(bank_id);
            if (inserted) {
                it->second = mesh_device->impl().dram_sender_logical_cores(device, bank_id);
            }
            TT_FATAL(
                std::find(it->second.begin(), it->second.end(), sender_logical) != it->second.end(),
                "DRAM sender ({}, {}) is not a provisioned sender for bank {} on device {}",
                sender_logical.x,
                sender_logical.y,
                bank_id,
                device->id());
        }
    }
}

// Per-sender bank-local recv_index_base. Senders are ordered [bank b s0, bank b s1, bank b+1 s0,
// ...] (sender_logical.x == bank_id); recv_index_base resets to 0 on a bank change and accumulates
// within a bank (dual senders share a bank). Returns one value per sender in mapping order. Single
// source for both the request-header stamping and the experimental slab-index accessors.
inline std::vector<uint32_t> recv_index_bases_per_sender(
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
    std::vector<uint32_t> bases(mapping.size(), 0);
    uint32_t recv_index_base = 0;
    uint32_t prev_bank = std::numeric_limits<uint32_t>::max();
    for (size_t s = 0; s < mapping.size(); ++s) {
        const uint32_t bank = static_cast<uint32_t>(mapping[s].first.x);
        recv_index_base = (bank == prev_bank) ? recv_index_base : 0u;
        prev_bank = bank;
        bases[s] = recv_index_base;
        recv_index_base += mapping[s].second.num_cores();
    }
    return bases;
}

// Per-sender bank-local slab indices: entry [s][r] is the slab index (recv_index_base + r) that
// sender s's local receiver r reads, in mapping order.
//
// Order-agnostic on purpose: mapping a receiver's (bank, slab index) to a global position depends on
// the tensor's shard distribution (ROUND_ROBIN_1D strided vs CONTIGUOUS_1D contiguous), which a
// ring-matmul consumer treats as a "ring position". That is the caller's concept, not the
// transport's, so callers read the raw slab indices and permute them themselves. Well-defined
// regardless of bank density or per-bank uniformity.
inline std::vector<std::vector<uint32_t>> receiver_slab_indices_per_sender(
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
    const std::vector<uint32_t> bases = recv_index_bases_per_sender(mapping);
    std::vector<std::vector<uint32_t>> slab(mapping.size());
    for (size_t s = 0; s < mapping.size(); ++s) {
        const uint32_t n = mapping[s].second.num_cores();
        slab[s].resize(n);
        for (uint32_t r = 0; r < n; ++r) {
            slab[s][r] = bases[s] + r;
        }
    }
    return slab;
}

}  // namespace tt::tt_metal
