// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/dram_sender_topology.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <device.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal {

CoreCoord pick_unused_dram_logical_core(const metal_SocDescriptor& soc_desc, uint32_t bank_id, ChipId device_id) {
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    TT_FATAL(
        bank_id < num_banks, "bank_id={} out of range for device {} (num_banks={})", bank_id, device_id, num_banks);

    std::set<std::pair<size_t, size_t>> reserved;
    for (const auto& c : soc_desc.dram_view_worker_cores.at(bank_id)) {
        reserved.emplace(c.x, c.y);
    }
    for (const auto& c : soc_desc.dram_view_eth_cores.at(bank_id)) {
        reserved.emplace(c.x, c.y);
    }

    const uint32_t num_subchannels = soc_desc.get_grid_size(tt::CoreType::DRAM).y;
    const size_t channel = soc_desc.get_channel_for_dram_view(static_cast<int>(bank_id));
    for (uint32_t sub = 0; sub < num_subchannels; ++sub) {
        tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
            static_cast<int>(channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
        if (!reserved.contains({coord.x, coord.y})) {
            return soc_desc.get_logical_dram_core_for_subchannel(static_cast<int>(bank_id), static_cast<int>(sub));
        }
    }
    TT_THROW(
        "No unused DRAM subchannel found for bank_id={} on device {}; all {} subchannels are reserved as "
        "worker/eth endpoints",
        bank_id,
        device_id,
        num_subchannels);
}

std::vector<CoreCoord> dram_sender_logical_cores(
    const metal_SocDescriptor& soc_desc, uint32_t bank_id, ChipId device_id) {
    // Sender 0: the free non-endpoint subchannel.
    const CoreCoord free_core = pick_unused_dram_logical_core(soc_desc, bank_id, device_id);

    // Sender 1: the NOC1 worker-endpoint subchannel. Resolve its subchannel index by
    // matching the endpoint's physical (TRANSLATED) coord against this bank's
    // subchannels (mirrors the loop in pick_unused_dram_logical_core).
    const CoreCoord noc1_endpoint_phys =
        soc_desc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank_id), /*noc=*/static_cast<uint8_t>(1));
    const uint32_t num_subchannels = soc_desc.get_grid_size(tt::CoreType::DRAM).y;
    const size_t channel = soc_desc.get_channel_for_dram_view(static_cast<int>(bank_id));
    for (uint32_t sub = 0; sub < num_subchannels; ++sub) {
        const tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
            static_cast<int>(channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
        if (coord.x == noc1_endpoint_phys.x && coord.y == noc1_endpoint_phys.y) {
            const CoreCoord noc1_core =
                soc_desc.get_logical_dram_core_for_subchannel(static_cast<int>(bank_id), static_cast<int>(sub));
            // A sender mapping resolves logical DRAM coords against one reference device and is then
            // reused mesh-wide (see build_dram_sender_mapping), so a role must land on the same
            // logical y on every device. dram_bank_endpoint_coords orders a bank as [NOC0 endpoint,
            // NOC1 endpoint, remaining subchannels ascending], which pins the NOC1 endpoint to y == 1
            // and every free subchannel above it -- but only while the bank's two worker endpoints are
            // distinct subchannels. A descriptor that names one subchannel for both NOCs (Wormhole
            // does) collapses that ordering and would let the free and NOC1 roles swap logical y
            // between devices, which the per-device membership check cannot see. Wormhole has no DRAM
            // senders, so require the split here rather than in the descriptor.
            TT_FATAL(
                noc1_core.y == 1 && free_core.y > noc1_core.y,
                "DRAM bank {} on device {}: expected the NOC1 worker endpoint at logical y=1 with the free "
                "subchannel above it, but found NOC1 at ({}, {}) and free at ({}, {}). A bank's NOC0 and NOC1 "
                "worker endpoints must be distinct subchannels so that a sender's logical coord names the same "
                "endpoint role on every device in the mesh; check this view's worker_endpoint pair in the SoC "
                "descriptor.",
                bank_id,
                device_id,
                noc1_core.x,
                noc1_core.y,
                free_core.x,
                free_core.y);
            return {free_core, noc1_core};
        }
    }
    TT_THROW(
        "Could not resolve the NOC1 worker-endpoint subchannel for DRAM bank_id={} on device {}", bank_id, device_id);
}

std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    const distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    DramSenderSplit split) {
    // Sender coords name endpoint roles, so resolving them against any one device gives the
    // mapping for the whole mesh; validate_dram_senders_across_mesh rechecks that per device.
    const auto& devices = mesh_device->impl().get_devices();
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
            "(the slab bases this mapping is read for reset per contiguous run of one bank's senders, so a bank "
            "split across two runs would be numbered from 0 twice).",
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
    const distributed::MeshDevice* mesh_device, const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
    std::unordered_map<uint32_t, std::vector<CoreCoord>> senders_by_bank;
    for (const auto* device : mesh_device->impl().get_devices()) {
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
    const distributed::MeshDevice& mesh_device,
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
