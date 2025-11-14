// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>

#include "impl/buffers/buffer.hpp"

namespace tt::tt_metal {

// Static factory methods
std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    std::optional<bool> bottom_up,
    std::optional<SubDeviceId> sub_device_id) {
    return BufferImpl::create(device, size, page_size, buffer_type, sharding_args, bottom_up, sub_device_id);
}

std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr address,
    DeviceAddr size,
    DeviceAddr page_size,
    BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    std::optional<bool> bottom_up,
    std::optional<SubDeviceId> sub_device_id) {
    return BufferImpl::create(device, address, size, page_size, buffer_type, sharding_args, bottom_up, sub_device_id);
}

// Destructor
Buffer::~Buffer() = default;

// Getter methods
IDevice* Buffer::device() const { return pimpl_->device(); }

Allocator* Buffer::allocator() const { return pimpl_->allocator(); }

DeviceAddr Buffer::size() const { return pimpl_->size(); }

uint32_t Buffer::address() const { return pimpl_->address(); }

DeviceAddr Buffer::page_size() const { return pimpl_->page_size(); }

uint32_t Buffer::num_pages() const { return pimpl_->num_pages(); }

BufferType Buffer::buffer_type() const { return pimpl_->buffer_type(); }

CoreType Buffer::core_type() const { return pimpl_->core_type(); }

bool Buffer::is_l1() const { return pimpl_->is_l1(); }

bool Buffer::is_dram() const { return pimpl_->is_dram(); }

TensorMemoryLayout Buffer::buffer_layout() const { return pimpl_->buffer_layout(); }

DeviceAddr Buffer::page_address(DeviceAddr bank_id, DeviceAddr page_index) const {
    return pimpl_->page_address(bank_id, page_index);
}

uint32_t Buffer::alignment() const { return pimpl_->alignment(); }

DeviceAddr Buffer::aligned_page_size() const { return pimpl_->aligned_page_size(); }

DeviceAddr Buffer::aligned_size_per_bank() const { return pimpl_->aligned_size_per_bank(); }

// Setter methods
void Buffer::set_page_size(DeviceAddr page_size) { pimpl_->set_page_size(page_size); }

void Buffer::set_shard_spec(const ShardSpecBuffer& shard_spec) { pimpl_->set_shard_spec(shard_spec); }

// Sharded API methods
const std::optional<BufferDistributionSpec>& Buffer::buffer_distribution_spec() const {
    return pimpl_->buffer_distribution_spec();
}

ShardSpecBuffer Buffer::shard_spec() const { return pimpl_->shard_spec(); }

std::optional<uint32_t> Buffer::num_cores() const { return pimpl_->num_cores(); }

const std::shared_ptr<const BufferPageMapping>& Buffer::get_buffer_page_mapping() {
    return pimpl_->get_buffer_page_mapping();
}

// Unique ID
size_t Buffer::unique_id() const { return pimpl_->unique_id(); }

// Implementation accessors
BufferImpl* Buffer::impl() { return pimpl_.get(); }

const BufferImpl* Buffer::impl() const { return pimpl_.get(); }

// Constructor
Buffer::Buffer(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    std::optional<bool> bottom_up,
    std::optional<SubDeviceId> sub_device_id,
    bool owns_data) :
    pimpl_(std::make_unique<BufferImpl>(
        device, size, page_size, buffer_type, sharding_args, bottom_up, sub_device_id, owns_data, this)) {}

}  // namespace tt::tt_metal
