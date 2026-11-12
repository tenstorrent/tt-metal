// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "hostdev/dram_receiver_config_page.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/global_circular_buffer_impl.hpp"
#include "impl/context/context_types.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/experimental/global_circular_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mesh_device.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental {

namespace {

// A DRAM receiver must be its own core: one CCE runs one Program at a time, so a receiver that is also
// a sender, or two entries naming one core, would lose a launch. Two logical coords can name one core,
// so compare package indices on every device.
void reject_conflicting_dram_receivers(
    const distributed::MeshDevice& mesh_device,
    const std::vector<CoreCoord>& senders,
    const std::vector<std::vector<CoreCoord>>& receivers_per_sender) {
    const Cluster& cluster = MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster();
    for (const IDevice* device : mesh_device.impl().get_devices()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        std::unordered_map<uint32_t, CoreCoord> sender_by_index;
        for (const CoreCoord& sender_logical : senders) {
            sender_by_index.emplace(package_dram_core_index(soc_desc, sender_logical), sender_logical);
        }
        std::unordered_map<uint32_t, CoreCoord> receiver_by_index;
        for (const std::vector<CoreCoord>& receivers : receivers_per_sender) {
            for (const CoreCoord& receiver_logical : receivers) {
                const uint32_t core_index = package_dram_core_index(soc_desc, receiver_logical);
                const std::unordered_map<uint32_t, CoreCoord>::const_iterator sender_it =
                    sender_by_index.find(core_index);
                TT_FATAL(
                    sender_it == sender_by_index.end(),
                    "DRAM receiver {} is core {} on device {}, which is also this GlobalCircularBuffer's sender {}",
                    receiver_logical.str(),
                    core_index,
                    device->id(),
                    sender_it->second.str());
                const auto [receiver_it, inserted] = receiver_by_index.try_emplace(core_index, receiver_logical);
                TT_FATAL(
                    inserted || receiver_it->second != receiver_logical,
                    "DRAM receiver {} is listed more than once",
                    receiver_logical.str());
                TT_FATAL(
                    inserted,
                    "DRAM receivers {} and {} are logical aliases of core {} on device {}",
                    receiver_it->second.str(),
                    receiver_logical.str(),
                    core_index,
                    device->id());
            }
        }
    }
}

}  // namespace

void GlobalCircularBufferImpl::allocate_dram_receiver_zone() {
    TT_FATAL(
        size_ > 0 && size_ % kDramReceiverCreditUnitBytes == 0,
        "DRAM-receiver GlobalCircularBuffer size {} B must be a non-zero multiple of the {} B credit unit",
        size_,
        kDramReceiverCreditUnitBytes);
    const uint32_t receiver_zone_size = device_->impl().drisc_l1_arena().receiver_zone_size();
    TT_FATAL(
        static_cast<uint64_t>(kDramReceiverConfigPageBytes) + size_ <= receiver_zone_size,
        "DRAM-receiver GlobalCircularBuffer size {} B and its {} B config page exceed the {} B DRISC L1 receiver "
        "zone",
        size_,
        kDramReceiverConfigPageBytes,
        receiver_zone_size);
    drisc_receiver_alloc_ = device_->impl().drisc_l1_arena().allocate_receiver(
        kDramReceiverConfigPageBytes + size_, kDramReceiverLineBytes);
    pages_sent_receiver_l1_base_ = drisc_receiver_alloc_->addr() + kDramReceiverCreditLineOffset;
    this->write_dram_receiver_config_pages();
}

void GlobalCircularBufferImpl::write_dram_receiver_config_pages() {
    const Cluster& cluster = MetalContext::instance(extract_context_id(device_)).get_cluster();
    const uint32_t buffer_address = static_cast<uint32_t>(this->buffer_address());
    std::vector<uint32_t> page_words(kDramReceiverConfigPageBytes / sizeof(uint32_t), 0);
    page_words[kDramReceiverFifoStartAddrOffset / sizeof(uint32_t)] = buffer_address;
    page_words[kDramReceiverFifoSizeOffset / sizeof(uint32_t)] = size_;
    page_words[kDramReceiverFifoRdPtrOffset / sizeof(uint32_t)] = buffer_address;
    for (IDevice* dev : device_->impl().get_devices()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(dev->id());
        for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
            const std::vector<CoreCoord>& receivers_vec = receiver_logical_cores_per_sender_[s];
            const uint32_t this_num_receivers = static_cast<uint32_t>(receivers_vec.size());
            const DeviceAddr pages_acked_base =
                pages_sent_drisc_l1_base_ + dram_receiver_sender_pages_acked_offset(this_num_receivers);
            page_words[kDramReceiverSenderIndexOffset / sizeof(uint32_t)] =
                package_dram_core_index(soc_desc, sender_receiver_core_mapping_[s].first);
            for (uint32_t i = 0; i < this_num_receivers; ++i) {
                page_words[kDramReceiverRemotePagesAckedAddrOffset / sizeof(uint32_t)] =
                    static_cast<uint32_t>(pages_acked_base + (i * kDramReceiverCreditStrideBytes));
                write_dram_core_l1(
                    *device_, dev, receivers_vec[i], config_address(), std::as_bytes(std::span(page_words)));
            }
        }
    }
}

GlobalCircularBuffer CreateGlobalCircularBufferForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>& bank_to_dram_receivers,
    uint32_t size) {
    if (!MetalContext::instance(mesh_device.impl().get_context_id()).hal().has_cce_dram_cores()) {
        TT_THROW(
            "DRAM receivers need CCE DRAM cores, which this device ({}) does not have; pass worker receivers as a "
            "CoreRangeSet per bank",
            mesh_device.arch());
    }
    const std::vector<IDevice*> devices = mesh_device.impl().get_devices();
    TT_FATAL(!devices.empty(), "Cannot build a DRAM-receiver GlobalCircularBuffer for a mesh with no local devices");
    std::vector<CoreCoord> senders;
    std::vector<std::vector<CoreCoord>> receivers_per_sender;
    std::unordered_set<uint32_t> seen_banks;
    for (const auto& [bank_id, receivers] : bank_to_dram_receivers) {
        TT_FATAL(!receivers.empty(), "DRAM bank {} has no receivers", bank_id);
        TT_FATAL(seen_banks.insert(bank_id).second, "DRAM bank {} appears more than once", bank_id);
        senders.push_back(mesh_device.impl().dram_sender_logical_cores(devices.front(), bank_id).front());
        receivers_per_sender.push_back(receivers);
    }
    reject_conflicting_dram_receivers(mesh_device, senders, receivers_per_sender);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(senders.size());
    for (size_t s = 0; s < senders.size(); ++s) {
        mapping.emplace_back(senders[s], CoreRangeSet(ttsl::Span<const CoreCoord>(receivers_per_sender[s])));
    }
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        mesh_device, mapping, std::move(receivers_per_sender), size, BufferType::L1, CoreType::DRAM);
}

}  // namespace tt::tt_metal::experimental
