// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/dram_sender_topology.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <device.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal {

std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    DramSenderSplit split) {
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
    const bool dual_senders_per_bank = split == DramSenderSplit::TwoPerBank;
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

void validate_dram_senders_across_mesh(
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

std::vector<uint32_t> recv_index_bases_per_sender(const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
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

void write_dram_sender_l1(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& sender_logical,
    DeviceAddr local_addr,
    std::span<const std::byte> bytes) {
    auto& metal_ctx = MetalContext::instance(mesh_device.impl().get_context_id());
    const uint64_t write_addr =
        metal_ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + static_cast<uint64_t>(local_addr);
    const CoreCoord virtual_core = device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    metal_ctx.get_cluster().write_core(device->id(), tt_cxy_pair(device->id(), virtual_core), bytes, write_addr);
}

}  // namespace tt::tt_metal
