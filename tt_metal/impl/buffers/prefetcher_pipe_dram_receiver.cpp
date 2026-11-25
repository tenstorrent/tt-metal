// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipes whose receivers are DRAM cores (Quasar CCEs) rather than worker cores: the factory
// that carves them from a ReceiverCoreType::Dram space, and the pages a carve
// writes. Layouts are in dram_receiver_config_page.hpp and dram_sender_state_block.hpp.

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "hostdev/dram_receiver_config_page.hpp"
#include "impl/buffers/dram_sender_state_block.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe_dram_sender_internal.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal::experimental {

namespace {

// A DRAM receiver must be its own core: one CCE runs one Program at a time, so a receiver that is also
// a sender, or two entries naming one core, would lose a launch. Two logical coords can name one core,
// so compare package indices on every device.
void reject_conflicting_dram_receivers(
    const distributed::MeshDevice& mesh_device,
    const std::vector<CoreCoord>& senders,
    const std::vector<std::vector<CoreCoord>>& receivers_per_sender) {
    const Cluster& cluster = mesh_device.impl().metal_env().get_cluster();
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
                    "DRAM receiver {} is core {} on device {}, which is also this pipe set's sender {}",
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

std::vector<PrefetcherPipe> CreatePrefetcherPipesForTensorPrefetcher(
    PrefetcherPipeSpace& space,
    const std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>& bank_to_dram_receivers) {
    TT_FATAL(
        !bank_to_dram_receivers.empty(), "CreatePrefetcherPipesForTensorPrefetcher requires at least one DRAM bank");
    const distributed::MeshDevice* mesh_device = space.get_device();
    TT_FATAL(mesh_device != nullptr, "CreatePrefetcherPipesForTensorPrefetcher requires a live space");
    if (!space.impl().has_dram_receivers()) {
        TT_THROW(
            "DRAM receivers are Quasar CCEs and need a space whose receiver_core_type is ReceiverCoreType::Dram; for "
            "worker receivers pass a CoreRangeSet per bank");
    }
    const std::vector<::tt::tt_metal::Device*> devices = mesh_device->impl().get_devices();
    TT_FATAL(!devices.empty(), "Cannot carve DRAM-receiver PrefetcherPipes for a mesh with no local devices");
    std::vector<CoreCoord> senders;
    std::vector<std::vector<CoreCoord>> receivers_per_sender;
    std::unordered_set<uint32_t> seen_banks;
    for (const auto& [bank_id, receivers] : bank_to_dram_receivers) {
        TT_FATAL(!receivers.empty(), "DRAM bank {} has no receivers", bank_id);
        TT_FATAL(seen_banks.insert(bank_id).second, "DRAM bank {} appears more than once", bank_id);
        senders.push_back(mesh_device->impl().dram_sender_logical_cores(devices.front(), bank_id).front());
        receivers_per_sender.push_back(receivers);
    }
    reject_conflicting_dram_receivers(*mesh_device, senders, receivers_per_sender);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(senders.size());
    for (size_t s = 0; s < senders.size(); ++s) {
        mapping.emplace_back(senders[s], CoreRangeSet(ttsl::Span<const CoreCoord>(receivers_per_sender[s])));
    }
    // Capacity, receiver domain and claim state are checked before the space is mutated.
    space.impl().validate_dram_carves(mapping);
    set_dram_sender_cores(space, senders);

    std::vector<PrefetcherPipe> pipes;
    pipes.reserve(senders.size());
    const uint64_t factory_id = next_tensor_prefetcher_factory_id();
    for (size_t s = 0; s < senders.size(); ++s) {
        pipes.push_back(space.impl().create_dram_receiver_pipe(
            senders[s], std::move(receivers_per_sender[s]), factory_id, static_cast<uint32_t>(senders.size())));
    }
    return pipes;
}

void PrefetcherPipeImpl::build_dram_receiver_config_pages() {
    const distributed::MeshDevice* mesh_device = space_->get_device();
    const Cluster& cluster = mesh_device->impl().metal_env().get_cluster();
    const std::vector<CoreCoord>& receivers = dram_sender_receiver_table_;
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    const uint32_t max_receivers = space_->max_receivers_per_pipe();
    const uint32_t buffer_address = space_->buffer_address();
    const uint32_t config_address = space_->config_address();
    const uint32_t pages_sent_base = static_cast<uint32_t>(pages_sent_drisc_l1_base());
    const uint32_t state_base = static_cast<uint32_t>(sender_state_drisc_l1_base());

    // Receiver config page: line 0 is host-written once, the credit line and the receiver's own
    // state line start zeroed apart from fifo_rd_ptr.
    std::vector<uint32_t> page_words(kDramReceiverConfigPageBytes / sizeof(uint32_t), 0);
    page_words[kDramReceiverFifoStartAddrOffset / sizeof(uint32_t)] = buffer_address;
    page_words[kDramReceiverFifoSizeOffset / sizeof(uint32_t)] = space_->ring_size();
    page_words[kDramReceiverFifoRdPtrOffset / sizeof(uint32_t)] = buffer_address;
    const uint32_t pages_acked_base = pages_sent_base + dram_receiver_sender_pages_acked_offset(num_receivers);

    // Sender state block. The receiver table follows the fixed struct; config_ptr points at the
    // sender config block embedded inside it. The table is sized for the space's largest pipe and
    // holds package DRAM-core indices, which is how one CCE addresses another's L1.
    std::vector<uint8_t> block_bytes(dram_sender_state_block_size(max_receivers), 0);
    DramSenderStateBlock* block = reinterpret_cast<DramSenderStateBlock*>(block_bytes.data());
    block->config_ptr = state_base + offsetof(DramSenderStateBlock, is_sender);
    block->fifo_start_addr = buffer_address;
    block->fifo_wr_ptr = buffer_address;
    block->receiver_noc_xy_ptr = state_base + sizeof(DramSenderStateBlock);
    block->aligned_pages_sent_ptr = pages_sent_base;
    // Every receiver's credit line is at the same offset, so one address names all of them.
    block->num_receivers_and_remote_pages_sent_ptr =
        (num_receivers << dev_msgs::REMOTE_CB_PACKED_COUNT_SHIFT) |
        ((config_address + kDramReceiverCreditLineOffset) & dev_msgs::REMOTE_CB_PACKED_ADDR_MASK);
    block->is_sender = 1;
    block->num_receivers = num_receivers;
    block->buffer_address = buffer_address;
    block->fifo_size_per_receiver = space_->ring_size();
    uint32_t* receiver_index_words = reinterpret_cast<uint32_t*>(block_bytes.data() + sizeof(DramSenderStateBlock));
    const std::vector<uint8_t> counters_zero_bytes(dram_receiver_sender_counters_bytes(max_receivers), 0);

    config_pages_.clear();
    for (IDevice* target_device : mesh_device->impl().get_devices()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(target_device->id());
        page_words[kDramReceiverSenderIndexOffset / sizeof(uint32_t)] = package_dram_core_index(soc_desc, sender_core_);
        for (uint32_t r = 0; r < num_receivers; ++r) {
            page_words[kDramReceiverRemotePagesAckedAddrOffset / sizeof(uint32_t)] =
                pages_acked_base + (r * kDramReceiverCreditStrideBytes);
            write_dram_core_l1(
                *mesh_device, target_device, receivers[r], config_address, std::as_bytes(std::span(page_words)));
            config_pages_.try_emplace(receivers[r], page_words);
            receiver_index_words[2 * r + 0] = package_dram_core_index(soc_desc, receivers[r]);
            receiver_index_words[2 * r + 1] = 0;
        }
        write_dram_core_l1(
            *mesh_device, target_device, sender_core_, pages_sent_base, std::as_bytes(std::span(counters_zero_bytes)));
        write_dram_core_l1(
            *mesh_device, target_device, sender_core_, state_base, std::as_bytes(std::span(block_bytes)));
    }
}

}  // namespace tt::tt_metal::experimental
