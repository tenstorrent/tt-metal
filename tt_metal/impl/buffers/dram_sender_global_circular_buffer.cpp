// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <dram_sender_global_circular_buffer.hpp>
#include <host_api.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "distributed.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::experimental {

DramSenderGlobalCircularBuffer::DramSenderGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& dram_sender_to_worker_receivers,
    uint32_t size,
    BufferType buffer_type) :
    device_(device), sender_receiver_core_mapping_(dram_sender_to_worker_receivers), size_(size) {
    TT_FATAL(device_ != nullptr, "Device cannot be null");

    const auto& hal = MetalContext::instance().hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DramSenderGlobalCircularBuffer requires programmable DRAM cores; set "
        "TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1");

    const uint32_t num_sender_cores = dram_sender_to_worker_receivers.size();
    TT_FATAL(num_sender_cores > 0, "At least one DRAM sender required");

    uint32_t num_receiver_cores = 0;
    uint32_t max_num_receivers_per_sender = 0;
    std::vector<CoreRange> sender_cores;
    sender_cores.reserve(num_sender_cores);
    std::unordered_set<CoreCoord> sender_set;
    for (const auto& [sender_core, receiver_cores] : dram_sender_to_worker_receivers) {
        TT_FATAL(
            sender_set.insert(sender_core).second,
            "Duplicate DRAM sender core ({}, {}) in sender_receiver_core_mapping",
            sender_core.x,
            sender_core.y);
        num_receiver_cores += receiver_cores.num_cores();
        sender_cores.emplace_back(sender_core);
        receiver_cores_ = receiver_cores_.merge(receiver_cores);
        max_num_receivers_per_sender = std::max(max_num_receivers_per_sender, receiver_cores.num_cores());
    }
    sender_cores_ = CoreRangeSet(sender_cores);
    TT_FATAL(num_sender_cores == sender_cores_.num_cores(), "Duplicate sender cores found");
    TT_FATAL(num_receiver_cores == receiver_cores_.num_cores(), "Duplicate receiver cores found");

    // Validate sender DRAM coords and forbid sender DRAM physical core appearing as a receiver worker core.
    std::unordered_set<CoreCoord> sender_dram_phys;
    for (const auto& [sender_core, _receiver_cores] : dram_sender_to_worker_receivers) {
        CoreCoord phys = device_->virtual_core_from_logical_core(sender_core, CoreType::DRAM);
        sender_dram_phys.insert(phys);
    }
    for (const auto& receiver_logical : corerange_to_cores(receiver_cores_)) {
        CoreCoord receiver_phys = device_->worker_core_from_logical_core(receiver_logical);
        TT_FATAL(
            sender_dram_phys.find(receiver_phys) == sender_dram_phys.end(),
            "Receiver worker core physical ({}, {}) collides with a sender DRAM physical coord",
            receiver_phys.x,
            receiver_phys.y);
    }

    // Pre-compute physical worker NOC XY for each sender's receivers.
    receiver_coords_per_sender_.reserve(num_sender_cores);
    for (const auto& [_sender_core, receivers] : dram_sender_to_worker_receivers) {
        const auto& receivers_vec = corerange_to_cores(receivers);
        std::vector<CoreCoord> phys;
        phys.reserve(receivers_vec.size());
        for (const auto& r : receivers_vec) {
            phys.emplace_back(device_->worker_core_from_logical_core(r));
        }
        receiver_coords_per_sender_.push_back(std::move(phys));
    }

    setup_receiver_buffers(buffer_type, max_num_receivers_per_sender);
}

void DramSenderGlobalCircularBuffer::setup_receiver_buffers(
    BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL,
        "DramSenderGlobalCircularBuffer can only be created for L1 buffer types");
    const uint32_t num_recv = receiver_cores_.num_cores();
    TT_FATAL(num_recv > 0, "DramSenderGlobalCircularBuffer requires at least one receiver");

    auto shard_parameters =
        ShardSpecBuffer(receiver_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_recv, 1});

    const uint32_t cb_buffer_size = size_ * num_recv;
    ShardedBufferConfig cb_buffer_shard_config = {
        .device = device_,
        .size = cb_buffer_size,
        .page_size = size_,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_parameters,
    };
    cb_buffer_ = distributed::AnyBuffer::create(cb_buffer_shard_config);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // is_sender, num_receivers, fifo_start_addr, fifo_size, fifo_ptr, noc_xy_addr, pages_sent_addr
    constexpr uint32_t num_config_elements = 7;
    const uint32_t num_noc_xy_words = 2 * max_num_receivers_per_sender;
    auto cb_config_page_size = tt::align((num_config_elements + num_noc_xy_words) * sizeof(uint32_t), l1_alignment) +
                               (2 * max_num_receivers_per_sender * l1_alignment);
    const uint32_t cb_config_size = cb_config_page_size * num_recv;
    ShardedBufferConfig cb_config_buffer_shard_config = {
        .device = device_,
        .size = cb_config_size,
        .page_size = cb_config_page_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    cb_config_buffer_ = distributed::AnyBuffer::create(cb_config_buffer_shard_config);

    auto config_buffer_address = cb_config_buffer_.get_buffer()->address();
    const auto& core_to_core_id = cb_config_buffer_.get_buffer()->get_buffer_page_mapping()->core_to_core_id;
    std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
    const uint32_t noc_xy_address = config_buffer_address + (num_config_elements * sizeof(uint32_t));
    const auto buffer_address = cb_buffer().address();

    // Receiver acks (`pages_acked`) and sender's view of `pages_sent` live in DRISC L1, not the
    // receiver L1, because we deliberately don't shard a buffer onto DRAM cores. Use DRISC L1
    // UNRESERVED as the agreed base — the DRISC kernel reserves the same region.
    const auto& hal_inst = MetalContext::instance().hal();
    pages_sent_drisc_l1_base_ = hal_inst.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping_) {
        const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
        const uint32_t num_receivers = receiver_cores.num_cores();
        // Sender DRAM virtual coord — receiver ack semaphores write back to this.
        auto sender_physical_coord = device_->virtual_core_from_logical_core(sender_core, CoreType::DRAM);
        for (uint32_t i = 0; i < receiver_cores_vec.size(); ++i) {
            uint32_t receiver_idx = core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[receiver_idx++] = 0;  // is_sender
            cb_config_host_buffer[receiver_idx++] = num_receivers;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = size_;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = noc_xy_address;
            cb_config_host_buffer[receiver_idx++] = pages_sent_drisc_l1_base_ + 2 * i * l1_alignment;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.x;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.y;
        }
    }
    auto mesh_buffer = cb_config_buffer_.get_mesh_buffer();
    distributed::EnqueueWriteMeshBuffer(
        mesh_buffer->device()->mesh_command_queue(), mesh_buffer, cb_config_host_buffer, false);
}

const Buffer& DramSenderGlobalCircularBuffer::cb_buffer() const { return *cb_buffer_.get_buffer(); }
const CoreRangeSet& DramSenderGlobalCircularBuffer::sender_cores() const { return sender_cores_; }
const CoreRangeSet& DramSenderGlobalCircularBuffer::receiver_cores() const { return receiver_cores_; }
DeviceAddr DramSenderGlobalCircularBuffer::buffer_address() const { return cb_buffer().address(); }
DeviceAddr DramSenderGlobalCircularBuffer::config_address() const { return cb_config_buffer_.get_buffer()->address(); }
uint32_t DramSenderGlobalCircularBuffer::size() const { return size_; }
const std::vector<std::pair<CoreCoord, CoreRangeSet>>& DramSenderGlobalCircularBuffer::sender_receiver_core_mapping()
    const {
    return sender_receiver_core_mapping_;
}
const std::vector<std::vector<CoreCoord>>& DramSenderGlobalCircularBuffer::receiver_coords_per_sender() const {
    return receiver_coords_per_sender_;
}
DeviceAddr DramSenderGlobalCircularBuffer::pages_sent_drisc_l1_base() const { return pages_sent_drisc_l1_base_; }

DramSenderGlobalCircularBuffer CreateDramSenderGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& dram_sender_to_worker_receivers,
    uint32_t size,
    BufferType buffer_type) {
    return DramSenderGlobalCircularBuffer(device, dram_sender_to_worker_receivers, size, buffer_type);
}

std::ostream& operator<<(std::ostream& os, const DramSenderGlobalCircularBuffer& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::DramSenderGlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::DramSenderGlobalCircularBuffer& cb) const {
    return tt::stl::hash::hash_objects_with_default_seed(cb.attribute_values());
}

}  // namespace std
