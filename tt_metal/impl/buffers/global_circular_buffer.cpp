// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <circular_buffer_constants.h>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_circular_buffer.hpp>
#include <host_api.hpp>
#include "impl/buffers/global_circular_buffer_impl.hpp"
#include "impl/context/context_types.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distributed.hpp"
#include "hal_types.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-umd/types/xy_pair.hpp>

namespace tt::tt_metal::experimental {

GlobalCircularBufferImpl::GlobalCircularBufferImpl(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    device_(&device), sender_receiver_core_mapping_(sender_receiver_core_mapping), size_(size) {
    const uint32_t num_sender_cores = sender_receiver_core_mapping.size();
    TT_FATAL(num_sender_cores > 0, "At least one sender required");
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

void GlobalCircularBufferImpl::setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global circular buffer can only be created for L1 buffer types");
    uint32_t num_cores = all_cores_.num_cores();

    auto shard_parameters = ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    uint32_t cb_buffer_size = size_ * num_cores;
    cb_buffer_ = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = cb_buffer_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = size_,
            .buffer_type = buffer_type,
            .sharding_args = BufferShardingArgs(shard_parameters, TensorMemoryLayout::HEIGHT_SHARDED),
        },
        device_);

    auto l1_alignment = device_->impl().metal_env().get_hal().get_alignment(HalMemType::L1);
    // [0] is_sender, [1] num_receivers, [2] fifo_start_addr, [3] fifo_size, [4] fifo_ptr,
    // [5] noc_xy_addr, [6] aligned_pages_sent_addr (local), [7] remote_pages_addr_override
    // (the remote NoC target; sender and receivers share an L1 layout, so it is the peer's copy of
    // slot 6's counter).
    constexpr uint32_t num_config_elements = 8;
    uint32_t num_noc_xy_words = 2 * max_num_receivers_per_sender;
    auto cb_config_page_size = tt::align((num_config_elements + num_noc_xy_words) * sizeof(uint32_t), l1_alignment) +
                               (2 * max_num_receivers_per_sender * l1_alignment);
    uint32_t cb_config_size = cb_config_page_size * num_cores;
    cb_config_buffer_ = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = cb_config_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = cb_config_page_size,
            .buffer_type = buffer_type,
            .sharding_args = BufferShardingArgs(std::move(shard_parameters), TensorMemoryLayout::HEIGHT_SHARDED),
        },
        device_);

    // Write the config buffer to the device
    // Only block for the slow dispatch case
    auto config_buffer_address = cb_config_buffer_->address();
    const auto& core_to_core_id = cb_config_buffer_->get_reference_buffer()->get_buffer_page_mapping()->core_to_core_id;
    uint32_t noc_xy_address = config_buffer_address + (num_config_elements * sizeof(uint32_t));
    uint32_t pages_sent_address = tt::align(noc_xy_address + (num_noc_xy_words * sizeof(uint32_t)), l1_alignment);
    auto buffer_address = cb_buffer().address();

    std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
    for (const auto& [sender_logical, receiver_cores] : sender_receiver_core_mapping_) {
        // Row-wise: this vector's index is the receiver's credit slot -- it picks both the
        // sender's NOC XY table order and each receiver's pages_sent/pages_acked offsets below,
        // and the sender walks the two in lockstep.
        const auto& receiver_cores_vec =
            corerange_to_cores(receiver_cores, /*max_cores=*/std::nullopt, /*row_wise=*/true);
        uint32_t num_receivers = receiver_cores.num_cores();

        uint32_t sender_idx = core_to_core_id.at(sender_logical) * cb_config_page_size / sizeof(uint32_t);
        cb_config_host_buffer[sender_idx++] = 1;
        cb_config_host_buffer[sender_idx++] = num_receivers;
        cb_config_host_buffer[sender_idx++] = buffer_address;
        cb_config_host_buffer[sender_idx++] = size_;
        cb_config_host_buffer[sender_idx++] = buffer_address;
        cb_config_host_buffer[sender_idx++] = noc_xy_address;
        cb_config_host_buffer[sender_idx++] = pages_sent_address;
        // The sender's NOC inc lands at the same L1 offset on the receiver's side, so the remote
        // address equals aligned_pages_sent_ptr.
        cb_config_host_buffer[sender_idx++] = pages_sent_address;
        for (const auto& receiver_logical : receiver_cores_vec) {
            auto receiver_physical_coord = device_->worker_core_from_logical_core(receiver_logical);
            cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
            cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;
        }

        // Sender's physical NOC coord -- where the receiver's pages_acked NOC-inc lands.
        CoreCoord sender_physical_coord = device_->worker_core_from_logical_core(sender_logical);

        for (uint32_t i = 0; i < receiver_cores_vec.size(); i++) {
            uint32_t receiver_idx = core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[receiver_idx++] = 0;  // is_sender
            cb_config_host_buffer[receiver_idx++] = num_receivers;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = size_;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = noc_xy_address;
            // aligned_pages_sent_ptr; pages_acked for this receiver lives at +L1_ALIGNMENT.
            cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment;
            // remote_pages_addr_override: the address on the SENDER's L1 where this receiver's
            // NoC-inc for pages_acked lands. Sender and receiver share an L1 layout, so this equals
            // aligned_pages_acked_ptr.
            cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment + l1_alignment;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.x;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.y;
        }
    }

    // Every device gets the same config page, so one broadcast write covers the mesh.
    distributed::EnqueueWriteMeshBuffer(device_->mesh_command_queue(), cb_config_buffer_, cb_config_host_buffer, false);
}

const Buffer& GlobalCircularBufferImpl::cb_buffer() const { return *cb_buffer_->get_reference_buffer(); }

const CoreRangeSet& GlobalCircularBufferImpl::sender_cores() const { return sender_cores_; }

const CoreRangeSet& GlobalCircularBufferImpl::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& GlobalCircularBufferImpl::all_cores() const { return all_cores_; }

DeviceAddr GlobalCircularBufferImpl::buffer_address() const { return cb_buffer().address(); }

DeviceAddr GlobalCircularBufferImpl::config_address() const { return cb_config_buffer_->address(); }

uint32_t GlobalCircularBufferImpl::size() const { return size_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBufferImpl::sender_receiver_core_mapping() const {
    return sender_receiver_core_mapping_;
}

// GlobalCircularBuffer (public pimpl wrapper) implementation

GlobalCircularBuffer::GlobalCircularBuffer(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    GlobalCircularBuffer(GlobalCircularBufferImpl(device, sender_receiver_core_mapping, size, buffer_type)) {}

GlobalCircularBuffer CreateGlobalCircularBuffer(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(GlobalCircularBufferImpl(device, sender_receiver_core_mapping, size, buffer_type));
}

GlobalCircularBuffer::GlobalCircularBuffer(GlobalCircularBufferImpl impl) :
    impl_(std::make_unique<GlobalCircularBufferImpl>(std::move(impl))) {}

GlobalCircularBuffer::GlobalCircularBuffer(const GlobalCircularBuffer& other) :
    impl_(other.impl_ ? std::make_unique<GlobalCircularBufferImpl>(*other.impl_) : nullptr) {}

GlobalCircularBuffer& GlobalCircularBuffer::operator=(const GlobalCircularBuffer& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<GlobalCircularBufferImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

GlobalCircularBuffer::GlobalCircularBuffer(GlobalCircularBuffer&& other) noexcept = default;
GlobalCircularBuffer& GlobalCircularBuffer::operator=(GlobalCircularBuffer&& other) noexcept = default;
GlobalCircularBuffer::~GlobalCircularBuffer() = default;

GlobalCircularBufferImpl& GlobalCircularBuffer::impl() {
    TT_FATAL(impl_ != nullptr, "GlobalCircularBuffer is in a moved-from state.");
    return *impl_;
}

const GlobalCircularBufferImpl& GlobalCircularBuffer::impl() const {
    TT_FATAL(impl_ != nullptr, "GlobalCircularBuffer is in a moved-from state.");
    return *impl_;
}

const Buffer& GlobalCircularBuffer::cb_buffer() const { return impl().cb_buffer(); }

const CoreRangeSet& GlobalCircularBuffer::sender_cores() const { return impl().sender_cores(); }

const CoreRangeSet& GlobalCircularBuffer::receiver_cores() const { return impl().receiver_cores(); }

DeviceAddr GlobalCircularBuffer::buffer_address() const { return impl().buffer_address(); }

DeviceAddr GlobalCircularBuffer::config_address() const { return impl().config_address(); }

uint32_t GlobalCircularBuffer::size() const { return impl().size(); }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBuffer::sender_receiver_core_mapping() const {
    return impl().sender_receiver_core_mapping();
}

std::tuple<std::vector<std::pair<CoreCoord, CoreRangeSet>>, uint32_t, BufferType>
GlobalCircularBuffer::attribute_values() const {
    return impl().attribute_values();
}

std::ostream& operator<<(std::ostream& os, const GlobalCircularBuffer& value) {
    ttsl::reflection::operator<<(os, value);
    return os;
}

SenderCoreType sender_core_type(const GlobalCircularBuffer& /*gcb*/) { return SenderCoreType::Worker; }

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return ttsl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
