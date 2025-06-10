// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_circular_buffer.hpp>
#include <host_api.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "distributed.hpp"
#include "hal_types.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.h>

namespace tt::tt_metal {
namespace experimental {

GlobalCircularBuffer::GlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    device_(device), sender_receiver_core_mapping_(sender_receiver_core_mapping), size_(size) {
    TT_FATAL(device_ != nullptr, "Device cannot be null");
    uint32_t num_sender_cores = sender_receiver_core_mapping.size();
    uint32_t num_receiver_cores = 0;
    uint32_t max_num_receivers_per_sender = 0;
    std::vector<CoreRange> sender_cores;
    sender_cores.reserve(num_sender_cores);
    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
        num_receiver_cores += receiver_cores.num_cores();
        sender_cores.emplace_back(sender_core);
        receiver_cores_ = receiver_cores_.merge(receiver_cores);
        max_num_receivers_per_sender = std::max(max_num_receivers_per_sender, receiver_cores.num_cores());
    }
    sender_cores_ = CoreRangeSet(sender_cores);
    TT_FATAL(num_sender_cores == sender_cores_.num_cores(), "Duplicate sender cores found");
    TT_FATAL(num_receiver_cores == receiver_cores_.num_cores(), "Duplicate receiver cores found");
    all_cores_ = sender_cores_.merge(receiver_cores_);
    TT_FATAL(all_cores_.num_cores() == num_sender_cores + num_receiver_cores, "Duplicate cores found");
    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

void GlobalCircularBuffer::setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global circular buffer can only be created for L1 buffer types");
    uint32_t num_cores = all_cores_.num_cores();

    auto shard_parameters = ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    uint32_t cb_buffer_size = size_ * num_cores;
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
    // is_sender, receiver_val, fifo_start_addr, fifo_size, fifo_ptr, noc_xy coords, and pages_sent
    constexpr uint32_t num_config_elements = 7;
    uint32_t num_noc_xy_words = 2 * max_num_receivers_per_sender;
    auto cb_config_page_size = tt::align((num_config_elements + num_noc_xy_words) * sizeof(uint32_t), l1_alignment) +
                               2 * max_num_receivers_per_sender * l1_alignment;
    uint32_t cb_config_size = cb_config_page_size * num_cores;
    ShardedBufferConfig cb_config_buffer_shard_config = {
        .device = device_,
        .size = cb_config_size,
        .page_size = cb_config_page_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    cb_config_buffer_ = distributed::AnyBuffer::create(cb_config_buffer_shard_config);

    // Write the config buffer to the device
    // Only block for the slow dispatch case
    auto config_buffer_address = cb_config_buffer_.get_buffer()->address();
    const auto& core_to_core_id = cb_config_buffer_.get_buffer()->get_buffer_page_mapping()->core_to_core_id;
    std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
    uint32_t noc_xy_address = config_buffer_address + num_config_elements * sizeof(uint32_t);
    uint32_t pages_sent_address = tt::align(noc_xy_address + num_noc_xy_words * sizeof(uint32_t), l1_alignment);
    auto buffer_address = cb_buffer().address();
    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping_) {
        const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
        uint32_t sender_idx = core_to_core_id.at(sender_core) * cb_config_page_size / sizeof(uint32_t);
        uint32_t num_receivers = receiver_cores.num_cores();
        uint32_t pages_acked_address = pages_sent_address + num_receivers * l1_alignment;
        cb_config_host_buffer[sender_idx++] = 1;
        cb_config_host_buffer[sender_idx++] = receiver_cores.num_cores();
        cb_config_host_buffer[sender_idx++] = buffer_address;
        cb_config_host_buffer[sender_idx++] = size_;
        cb_config_host_buffer[sender_idx++] = buffer_address;
        cb_config_host_buffer[sender_idx++] = noc_xy_address;
        cb_config_host_buffer[sender_idx++] = pages_sent_address;

        auto sender_physical_coord = device_->worker_core_from_logical_core(sender_core);
        for (uint32_t i = 0; i < receiver_cores_vec.size(); i++) {
            auto receiver_physical_coord = device_->worker_core_from_logical_core(receiver_cores_vec[i]);
            cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
            cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;

            uint32_t receiver_idx = core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[receiver_idx++] = 0;
            cb_config_host_buffer[receiver_idx++] = num_receivers;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = size_;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = noc_xy_address;
            cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.x;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.y;
        }
    }
    if (auto mesh_buffer = cb_config_buffer_.get_mesh_buffer()) {
        distributed::EnqueueWriteMeshBuffer(
            mesh_buffer->device()->mesh_command_queue(), mesh_buffer, cb_config_host_buffer, false);
    } else {
        if (device_->using_slow_dispatch()) {
            detail::WriteToBuffer(*cb_config_buffer_.get_buffer(), cb_config_host_buffer);
            tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_->id());
        } else {
            EnqueueWriteBuffer(
                device_->command_queue(), *cb_config_buffer_.get_buffer(), cb_config_host_buffer.data(), false);
        }
    }
}

const Buffer& GlobalCircularBuffer::cb_buffer() const { return *cb_buffer_.get_buffer(); }

const CoreRangeSet& GlobalCircularBuffer::sender_cores() const { return sender_cores_; }

const CoreRangeSet& GlobalCircularBuffer::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& GlobalCircularBuffer::all_cores() const { return all_cores_; }

DeviceAddr GlobalCircularBuffer::buffer_address() const { return cb_buffer().address(); }

DeviceAddr GlobalCircularBuffer::config_address() const { return cb_config_buffer_.get_buffer()->address(); }

uint32_t GlobalCircularBuffer::size() const { return size_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBuffer::sender_receiver_core_mapping() const {
    return sender_receiver_core_mapping_;
}

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
