// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"

#include "tt_metal/buffer.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/types.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <utility>
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "umd/device/tt_soc_descriptor.h"
#include "fmt/base.h"
#include "tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

std::atomic<size_t> Buffer::next_unique_id = 0;

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec) {
    tt::stl::reflection::operator<<(os, spec);
    return os;
}

bool is_sharded(const TensorMemoryLayout &layout) {
    return (
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED ||
        layout == TensorMemoryLayout::BLOCK_SHARDED);
}

bool is_l1(BufferType buffer_type) {
    return buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL;
}

void validate_buffer_size_and_page_size(
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType &buffer_type,
    const TensorMemoryLayout &buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters) {
    if (size == 0) {
        return;
    }

    bool valid_page_size = (size % page_size == 0);
    TT_FATAL(
        valid_page_size,
        "For valid non-interleaved buffers page size {} must equal buffer size {}. For interleaved-buffers page size "
        "should be divisible by buffer size",
        page_size,
        size);

    if (is_sharded(buffer_layout)) {
        TT_FATAL(
            shard_parameters != std::nullopt,
            "Buffer was specified as sharded but does not have shard_parameters specified");
    } else {
        TT_FATAL(
            shard_parameters == std::nullopt, "Buffer was specified as not sharded but has shard_parameters specified");
        if (buffer_layout == TensorMemoryLayout::SINGLE_BANK) {
            TT_FATAL(page_size == size, "Contiguous buffer must be one contiguous page");
        }
    }
}

inline std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::array<uint32_t, 2>>> core_to_host_pages(
    const uint32_t &total_pages,
    const uint32_t &pages_per_shard,
    const uint32_t &num_shards,
    const TensorMemoryLayout &layout,
    const std::array<uint32_t, 2> &page_shape,
    const std::array<uint32_t, 2> &shard_shape,
    const std::array<uint32_t, 2> &tensor2d_size) {
    std::array<uint32_t, 2> shard_in_pages = {page_shape[0] == 0 ? 0 : shard_shape[0] / page_shape[0], page_shape[1] == 0 ? 0 : shard_shape[1] / page_shape[1]};
    std::vector<std::vector<uint32_t>> ret_vec(num_shards);
    std::vector<std::array<uint32_t, 2>> ret_shard_shape(num_shards, shard_in_pages);

    if (layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t rem_pages = tensor2d_size[0] * tensor2d_size[1];
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < num_shards; i++) {
            if (rem_pages == 0) {
                ret_shard_shape[i] = {0, 0};
            } else {
                uint32_t num_cols = std::min(pages_per_shard, rem_pages);
                if (pages_per_shard > rem_pages) {
                    ret_shard_shape[i] = {rem_pages / ret_shard_shape[i][1], ret_shard_shape[i][1]};
                }
                ret_vec[i] = std::vector<uint32_t>(num_cols);
                for (uint32_t j = 0; j < num_cols; j++) {
                    ret_vec[i][j] = page_id++;
                }
                rem_pages -= num_cols;
            }
        }
    } else if (layout == TensorMemoryLayout::WIDTH_SHARDED or layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t i_offset = 0;
        uint32_t j_offset = 0;
        uint32_t num_shard_columns = shard_in_pages[1] == 0 ? 0 : div_up(tensor2d_size[1], shard_in_pages[1]);
        uint32_t shard_in_row = 0;

        for (uint32_t shard_idx = 0; shard_idx < num_shards; shard_idx++) {
            ret_vec[shard_idx].reserve(pages_per_shard);

            uint32_t host_idx = 0;
            uint32_t i = 0;
            uint32_t j = 0;
            for (i = i_offset; i < (shard_in_pages[0] + i_offset); i++) {
                if (i >= tensor2d_size[0]) {
                    break;
                }
                for (j = j_offset; j < (shard_in_pages[1] + j_offset) and (j < (tensor2d_size[1])); j++) {
                    uint32_t host_page = i * tensor2d_size[1] + j;
                    ret_vec[shard_idx].push_back(host_page);
                    host_idx++;
                }
            }
            ret_shard_shape[shard_idx] = {i - i_offset, j - j_offset};
            if (((shard_in_row + 1) == (num_shard_columns))) {
                shard_in_row = 0;
                j_offset = 0;
                i_offset += shard_in_pages[0];
            } else {
                shard_in_row++;
                j_offset += shard_in_pages[1];
            }
        }
    }
    return {ret_vec, ret_shard_shape};
}

BufferPageMapping generate_buffer_page_mapping(const Buffer& buffer) {
    BufferPageMapping buffer_page_mapping;

    if (buffer.size() == 0) {
        return buffer_page_mapping;
    }
    auto shard_spec = buffer.shard_spec();

    bool row_major = shard_spec.orientation() == ShardOrientation::ROW_MAJOR;
    uint32_t num_cores = buffer.num_cores().value();

    buffer_page_mapping.all_cores_ = corerange_to_cores(shard_spec.grid(), num_cores, row_major);
    TT_FATAL(num_cores == buffer_page_mapping.all_cores_.size(), "Buffer has {} cores, but page mapping expects {} cores", num_cores, buffer_page_mapping.all_cores_.size());
    uint32_t core_id = 0;
    for (const auto &core : buffer_page_mapping.all_cores_) {
        buffer_page_mapping.core_to_core_id_.insert({core, core_id});
        core_id++;
    }

    uint32_t num_dev_pages = buffer.num_dev_pages();
    auto [core_host_page_indices, shard_shape] = core_to_host_pages(
        num_dev_pages,
        shard_spec.size(),
        num_cores,
        buffer.buffer_layout(),
        shard_spec.page_shape,
        shard_spec.shape(),
        shard_spec.tensor2d_shape);

    buffer_page_mapping.core_host_page_indices_ = std::vector<std::vector<uint32_t>>(num_cores);

    buffer_page_mapping.dev_page_to_host_page_mapping_ =
        std::vector<std::optional<uint32_t>>(num_dev_pages, std::nullopt);
    buffer_page_mapping.dev_page_to_core_mapping_ = std::vector<uint32_t>(num_dev_pages);

    buffer_page_mapping.host_page_to_local_shard_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.host_page_to_dev_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.core_shard_shape_ = std::move(shard_shape);
    uint32_t dev_page_index = 0;

    auto shape_in_pages = shard_spec.shape_in_pages();
    for (uint32_t core_index = 0; core_index < core_host_page_indices.size(); core_index++) {
        uint32_t valid_shard_page = 0;
        buffer_page_mapping.core_host_page_indices_[core_index].reserve(shard_spec.size());
        uint32_t shard_page_id = 0;
        for (uint32_t shard_page_x = 0; shard_page_x < shape_in_pages[0]; shard_page_x++) {
            for (uint32_t shard_page_y = 0; shard_page_y < shape_in_pages[1]; shard_page_y++) {
                buffer_page_mapping.dev_page_to_core_mapping_[dev_page_index] = core_index;
                if (shard_page_x < buffer_page_mapping.core_shard_shape_[core_index][0] and
                    shard_page_y < buffer_page_mapping.core_shard_shape_[core_index][1]) {
                    uint32_t host_page = core_host_page_indices[core_index][valid_shard_page];
                    buffer_page_mapping.dev_page_to_host_page_mapping_[dev_page_index] = host_page;
                    buffer_page_mapping.core_host_page_indices_[core_index].push_back(host_page);
                    buffer_page_mapping.host_page_to_local_shard_page_mapping_[host_page] = shard_page_id;
                    buffer_page_mapping.host_page_to_dev_page_mapping_[host_page] = dev_page_index;
                    valid_shard_page++;
                }
                dev_page_index++;
                shard_page_id++;
            }
        }
    }

    return buffer_page_mapping;
}

void validate_sub_device_id(std::optional<SubDeviceId> sub_device_id, Device *device, BufferType buffer_type, const std::optional<ShardSpecBuffer>& shard_parameters) {
    // No need to validate if we're using the global allocator or not sharding
    if (!sub_device_id.has_value()) {
        return;
    }
    TT_FATAL(shard_parameters.has_value(), "Specifying sub-device for buffer requires buffer to be sharded");
    TT_FATAL(is_l1(buffer_type), "Specifying sub-device for buffer requires buffer to be L1");
    const auto &sub_device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id.value());
    const auto &shard_cores = shard_parameters->grid();
    TT_FATAL(sub_device_cores.contains(shard_cores), "Shard cores specified {} do not match sub-device cores {}", shard_cores, sub_device_cores);
}

Buffer::Buffer(
    Device *device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id,
    const bool owns_data,
    Private) :
    device_(device),
    size_(size),
    page_size_(page_size),
    buffer_type_(buffer_type),
    buffer_layout_(buffer_layout),
    shard_parameters_(shard_parameters),
    bottom_up_(bottom_up.value_or(this->is_dram())),
    sub_device_id_(sub_device_id),
    owns_data_(owns_data),
    buffer_page_mapping_(nullptr) {
    TT_FATAL(this->device_ != nullptr, "Device needs to not be null.");
    if (this->sub_device_id_.has_value()) {
        validate_sub_device_id(this->sub_device_id_, this->device_, buffer_type, shard_parameters);
        this->sub_device_manager_id_ = this->device_->get_active_sub_device_manager_id();
        this->allocator_ = device->get_initialized_allocator(*this->sub_device_id_).get();
    } else {
        this->allocator_ = device->get_initialized_allocator().get();
    }
    if (size != 0) {
        validate_buffer_size_and_page_size(size, page_size, buffer_type, buffer_layout, shard_parameters);
    }
    unique_id_ = next_unique_id.fetch_add(1);
}

std::shared_ptr<Buffer> Buffer::create(
    Device *device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {
    auto* bufferPtr = new Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up, sub_device_id, true /* owns data */, Private());
    // Using a custom deleter to properly clean up the owned datas
    auto buffer = std::shared_ptr<Buffer>(bufferPtr, deleter);
    buffer->weak_self = buffer;

    if (buffer->size_ == 0) {
        buffer->allocation_status_.store(AllocationStatus::ALLOCATED, std::memory_order::relaxed);
        return buffer;
    }

    buffer->device_->push_work([buffer] {
        try {
            buffer->address_ = detail::AllocateBuffer(buffer.get());
        } catch(...) {
            std::unique_lock lock(buffer->allocation_mutex_);
            buffer->allocation_status_.store(AllocationStatus::ALLOCATION_FAILED, std::memory_order::relaxed);
            lock.unlock();
            buffer->allocation_cv_.notify_all();

            throw;
        }

        std::unique_lock lock(buffer->allocation_mutex_);
        buffer->allocation_status_.store(AllocationStatus::ALLOCATED, std::memory_order::release);
        lock.unlock();
        buffer->allocation_cv_.notify_all();
    });

    return buffer;
}

std::shared_ptr<Buffer> Buffer::create(
    Device *device,
    DeviceAddr address,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {
    // Not using a custom deleter, because it doesn't own any data to cleanup
    auto buffer = std::make_shared<Buffer>(device, size, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up, sub_device_id, false /* owns data */, Private());
    buffer->weak_self = buffer;

    buffer->address_ = address;
    buffer->allocation_status_.store(AllocationStatus::ALLOCATED, std::memory_order::relaxed);

    return buffer;
}

void Buffer::deallocate() {
    deallocation_requested_.store(true, std::memory_order::relaxed);
    if (!owns_data_) {
        return;
    }
    device_->push_work([self = weak_self.lock()] {
        self->deallocate_impl();
    });
}

void Buffer::deleter(Buffer* buffer) {
    buffer->device_->push_work([buffer] {
        std::unique_ptr<Buffer> unique_buffer = std::unique_ptr<Buffer>(buffer);
        buffer->deallocate_impl();
    });
}

void Buffer::deallocate_impl() {
    if (allocation_status_.load(std::memory_order::relaxed) != AllocationStatus::ALLOCATED) {
        return;
    }

    if (device_->is_initialized() && size_ != 0) {
        // address_ is only modified from this thread, no sync required
        detail::DeallocateBuffer(this);
    }

    allocation_status_.store(AllocationStatus::DEALLOCATED, std::memory_order::relaxed);
}

bool Buffer::is_allocated() const {
    auto allocation_status = allocation_status_.load(std::memory_order::relaxed);

    if (device_->can_use_passthrough_scheduling()) {
        return allocation_status == AllocationStatus::ALLOCATED;
    }

    // For calls from different threads we consider buffer to be allocated even if it's just ALLOCATION_REQUESTED,
    // because once the caller will try to access it, the buffer will already be fully allocated. For the same reason we need to check deallocation_requested_ too.
    bool deallocation_requested = deallocation_requested_.load(std::memory_order::relaxed);
    return (allocation_status == AllocationStatus::ALLOCATION_REQUESTED || allocation_status == AllocationStatus::ALLOCATED) && !deallocation_requested;
}

uint32_t Buffer::address() const {
    if (allocation_status_.load(std::memory_order::acquire) != AllocationStatus::ALLOCATION_REQUESTED) {
        return address_;
    }

    if (device_->can_use_passthrough_scheduling()) {
        return address_;
    }

    std::unique_lock lock(allocation_mutex_);
    allocation_cv_.wait(lock, [this] { return this->allocation_status_.load(std::memory_order::relaxed) != AllocationStatus::ALLOCATION_REQUESTED; });
    return address_;
}

DeviceAddr Buffer::page_size() const {
    return page_size_;
}

void Buffer::set_page_size(DeviceAddr page_size) {
    TT_FATAL(page_size == 0 ? size_ == 0 : size_ % page_size == 0, "buffer size must be divisible by new page size");
    page_size_ = page_size;
    this->buffer_page_mapping_ = nullptr;
}

uint32_t Buffer::num_pages() const {
    return page_size() == 0 ? 0 : size() / page_size();
}

uint32_t Buffer::num_dev_pages() const {
    if (!is_sharded(this->buffer_layout_)) {
        return this->num_pages();
    }

    return this->shard_spec().size() * this->num_cores().value();
}

CoreType Buffer::core_type() const {
    switch (this->buffer_type_) {
        case BufferType::DRAM:
            return CoreType::DRAM;
        case BufferType::L1:
        case BufferType::L1_SMALL:
            return CoreType::WORKER;
        default:
            TT_THROW("Unknown CoreType {} for buffer", this->buffer_type_);
    }
}

bool Buffer::is_l1() const {
    return tt::tt_metal::is_l1(buffer_type());
}
bool Buffer::is_dram() const {
    return buffer_type() == BufferType::DRAM || buffer_type() == BufferType::TRACE;
}
bool Buffer::is_trace() const {
    return buffer_type() == BufferType::TRACE;

}

bool Buffer::is_valid_region(const BufferRegion& region) const { return region.offset + region.size <= this->size(); }

bool Buffer::is_partial_region(const BufferRegion& region) const {
    TT_FATAL(
        this->is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid!",
        region.offset,
        region.size);
    return region.offset > 0 || region.size != this->size();
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_FATAL(this->is_dram(), "Expected DRAM buffer!");
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_FATAL(this->is_l1(), "Expected L1 buffer!");
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

DeviceAddr Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    uint32_t num_banks = allocator::num_banks(*this->allocator_, this->buffer_type_);
    TT_FATAL(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    int pages_offset_within_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

DeviceAddr Buffer::bank_local_page_address(uint32_t bank_id, uint32_t page_index) const {
    uint32_t num_banks = allocator::num_banks(*this->allocator_, this->buffer_type_);
    TT_FATAL(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    uint32_t offset;
    if (is_sharded(this->buffer_layout())) {
        auto shard_spec = this->shard_spec();
        uint32_t pages_offset_within_bank = page_index % shard_spec.size();
        offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    } else {
        uint32_t pages_offset_within_bank = page_index / num_banks;
        offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    }
    return this->address() + offset;
}

uint32_t Buffer::alignment() const {
    return this->allocator_->config.alignment;
}

DeviceAddr Buffer::aligned_page_size() const {
    return align(page_size(), this->alignment());
}
DeviceAddr Buffer::aligned_size() const {
    return this->num_dev_pages() * this->aligned_page_size();
}

DeviceAddr Buffer::aligned_size_per_bank() const {
    uint32_t num_banks = is_sharded(this->buffer_layout_) ? this->num_cores().value() : this->device_->num_banks(this->buffer_type());
    return tt::tt_metal::detail::SizeBytesPerBank(this->aligned_size(), this->aligned_page_size(), num_banks, this->alignment());
}

DeviceAddr Buffer::sharded_page_address(uint32_t bank_id, uint32_t page_index) const {
    TT_FATAL(is_sharded(this->buffer_layout()), "Buffer not sharded");
    auto shard_spec = this->shard_spec();
    uint32_t pages_offset_within_bank = page_index % shard_spec.size();
    auto offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

ShardSpecBuffer Buffer::shard_spec() const {
    TT_FATAL(is_sharded(this->buffer_layout_), "Buffer not sharded");
    TT_FATAL(shard_parameters_.has_value(), "Buffer is sharded, but no shard parameters specified");
    return this->shard_parameters_.value();
}

void Buffer::set_shard_spec(const ShardSpecBuffer& shard_spec) {
    this->shard_parameters_ = shard_spec;
    this->buffer_page_mapping_ = nullptr;
}

std::optional<uint32_t> Buffer::num_cores() const {
    if (!is_sharded(this->buffer_layout_))
        return std::nullopt;

    return this->shard_spec().tensor_shard_spec.grid.num_cores();
}

DeviceAddr Buffer::translate_page_address(uint64_t offset, uint32_t bank_id) const {
    allocator::bank_offset(*this->allocator_, this->buffer_type_, bank_id);
    DeviceAddr base_page_address = this->address() + allocator::bank_offset(*this->allocator_, this->buffer_type_, bank_id);
    return base_page_address + offset;
}

const std::shared_ptr<const BufferPageMapping>& Buffer::get_buffer_page_mapping() {
    TT_FATAL(is_sharded(this->buffer_layout_), "Buffer not sharded");
    if (!this->buffer_page_mapping_) {
        this->buffer_page_mapping_ = std::make_shared<const BufferPageMapping>(generate_buffer_page_mapping(*this));
    }
    return this->buffer_page_mapping_;
}

bool ShardSpec::operator==(const ShardSpec&) const = default;
bool ShardSpec::operator!=(const ShardSpec&) const = default;

std::array<uint32_t, 2> ShardSpecBuffer::shape_in_pages() const {
    auto width_in_pages = page_shape[0] == 0 ? 0 : tensor_shard_spec.shape[0] / page_shape[0];
    auto height_in_pages = page_shape[1] == 0 ? 0 : tensor_shard_spec.shape[1] / page_shape[1];
    return {width_in_pages, height_in_pages};
}

DeviceAddr ShardSpecBuffer::size() const {
    auto shape_in_pages_ = this->shape_in_pages();
    return shape_in_pages_[0] * shape_in_pages_[1];
}

v1::BufferHandle v1::CreateBuffer(InterleavedBufferConfig config) { return v1::BufferHandle{v0::CreateBuffer(config)}; }

void v1::DeallocateBuffer(const BufferHandle& buffer) { v0::DeallocateBuffer(*buffer); }

std::size_t v1::GetId(const BufferHandle& buffer) { return buffer->unique_id(); }

void v1::WriteToBuffer(const BufferHandle& buffer, stl::Span<const std::byte> host_buffer) {
    detail::WriteToBuffer(*buffer, stl::Span<const uint8_t>{reinterpret_cast<const std::uint8_t *>(host_buffer.data()), host_buffer.size()});
}

void v1::ReadFromBuffer(const BufferHandle& buffer, stl::Span<std::byte> host_buffer, bool shard_order) {
    detail::ReadFromBuffer(*buffer, reinterpret_cast<std::uint8_t *>(host_buffer.data()), shard_order);
}

void v1::ReadFromShard(const BufferHandle& buffer, stl::Span<std::byte> host_buffer, std::uint32_t core_id) {
    detail::ReadShard(*buffer, reinterpret_cast<std::uint8_t *>(host_buffer.data()), core_id);
}

}  // namespace tt_metal
}  // namespace tt

namespace tt::stl::json {
tt_metal::ShardSpec from_json_t<tt_metal::ShardSpec>::operator()(const nlohmann::json &json_object) const {
    const auto& shard_mode = from_json<tt_metal::ShardMode>(json_object.at("mode"));
    const auto& physical_shard_shape = from_json<std::optional<std::array<uint32_t, 2>>>(json_object.at("physical_shard_shape"));
    if (physical_shard_shape.has_value()) {
        TT_FATAL(shard_mode == tt::tt_metal::ShardMode::LOGICAL, "Physical shard shape can only be provided in logical sharding mode!");
        return tt_metal::ShardSpec{
            from_json<CoreRangeSet>(json_object.at("grid")),
            from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
            physical_shard_shape.value(),
            from_json<tt_metal::ShardOrientation>(json_object.at("orientation")),
            from_json<bool>(json_object.at("halo"))};
    }
    return tt_metal::ShardSpec{
        from_json<CoreRangeSet>(json_object.at("grid")),
        from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
        from_json<tt_metal::ShardOrientation>(json_object.at("orientation")),
        from_json<bool>(json_object.at("halo")),
        shard_mode};
}
}
