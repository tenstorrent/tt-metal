// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

bool is_sharded(const TensorMemoryLayout &layout) {
    return (
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED ||
        layout == TensorMemoryLayout::BLOCK_SHARDED);
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
    TT_FATAL(
        page_size % sizeof(uint32_t) == 0,
        "Page size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values");

    if (is_sharded(buffer_layout)) {
        TT_FATAL(shard_parameters != std::nullopt, "Sharded buffers must have a core grid assigned");
    } else if (buffer_layout == TensorMemoryLayout::SINGLE_BANK) {
        TT_FATAL(page_size == size, "Contiguous buffer must be one contiguous page");
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
    uint32_t num_cores = buffer.num_cores();

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

Buffer::Buffer(
    Device *device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up) :
    device_(device),
    size_(size),
    page_size_(page_size),
    buffer_type_(buffer_type),
    buffer_layout_(buffer_layout),
    shard_parameters_(shard_parameters),
    bottom_up_(bottom_up),
    buffer_page_mapping_(nullptr) {
    TT_FATAL(this->device_ != nullptr && this->device_->allocator_ != nullptr, "Device and allocator need to not be null.");

    if (size != 0) {
        validate_buffer_size_and_page_size(size, page_size, buffer_type, buffer_layout, shard_parameters);
    }
}

std::shared_ptr<Buffer> Buffer::create(
    Device *device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up) {
    auto* bufferPtr = new Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up);
    auto buffer = std::shared_ptr<Buffer>(bufferPtr, deleter);
    buffer->weak_self = buffer;

    if (buffer->size_ == 0) {
        buffer->allocation_status_.store(AllocationStatus::ALLOCATED, std::memory_order::relaxed);
        return buffer;
    }

    buffer->device_->push_work([buffer] {
        bool bottom_up = buffer->bottom_up_.value_or(buffer->is_dram());
        buffer->address_ = detail::AllocateBuffer(buffer.get(), bottom_up);
        detail::BUFFER_MAP.insert({buffer->device_->id(), buffer->address_}, buffer.get());

        buffer->allocation_status_.store(AllocationStatus::ALLOCATED, std::memory_order::release);
        buffer->allocation_status_.notify_all();
    });

    return buffer;
}

void Buffer::deallocate() {
    deallocation_requested_.store(true, std::memory_order::relaxed);
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
    if (allocation_status_.load(std::memory_order::relaxed) == AllocationStatus::DEALLOCATED) {
        return;
    }

    if (device_->initialized_ && size_ != 0) {
        // address_ is only modified from this thread, no sync required
        detail::BUFFER_MAP.erase({device_->id(), address_});
        detail::DeallocateBuffer(this);
    }

    allocation_status_.store(AllocationStatus::DEALLOCATED, std::memory_order::relaxed);
}

bool Buffer::is_allocated() const {
    if (deallocation_requested_.load(std::memory_order::relaxed)) {
        return false;
    }

    auto allocation_status = allocation_status_.load(std::memory_order::relaxed);

    if (device_->can_use_passthrough_scheduling()) {
        return allocation_status == AllocationStatus::ALLOCATED;
    }

    // For calls from different threads we consider buffer to be allocated even if it's just ALLOCATION_REQUESTED,
    // because once the caller will try to access it, the buffer will already be fully allocated
    return allocation_status == AllocationStatus::ALLOCATION_REQUESTED || allocation_status == AllocationStatus::ALLOCATED;
}

uint32_t Buffer::address() const {
    allocation_status_.wait(AllocationStatus::ALLOCATION_REQUESTED, std::memory_order::acquire);
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

    return this->shard_spec().size() * this->num_cores();
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
    return buffer_type() == BufferType::L1 or buffer_type() == BufferType::L1_SMALL;
}
bool Buffer::is_dram() const {
    return buffer_type() == BufferType::DRAM || buffer_type() == BufferType::TRACE;
}
bool Buffer::is_trace() const {
    return buffer_type() == BufferType::TRACE;
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_FATAL(this->is_dram(), "Expected DRAM buffer!");
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_FATAL(this->is_l1(), "Expected L1 buffer!");
    return this->device_->logical_core_from_bank_id(bank_id);
}

CoreCoord Buffer::noc_coordinates(uint32_t bank_id) const {
    switch (this->buffer_type_) {
        case BufferType::DRAM:
        case BufferType::TRACE: {
            auto dram_channel = this->dram_channel_from_bank_id(bank_id);
            return this->device_->dram_core_from_dram_channel(dram_channel);
        }
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            auto logical_core = this->logical_core_from_bank_id(bank_id);
            return this->device_->worker_core_from_logical_core(logical_core);
        }
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Host buffer is located in system memory! Cannot retrieve NoC coordinates for it");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

CoreCoord Buffer::noc_coordinates() const { return this->noc_coordinates(0); }

DeviceAddr Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    auto num_banks = this->device_->num_banks(this->buffer_type_);
    TT_FATAL(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    int pages_offset_within_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

uint32_t Buffer::alignment() const {
    return this->device_->get_allocator_alignment();
}
DeviceAddr Buffer::aligned_page_size() const {
    return align(page_size(), this->alignment());
}
DeviceAddr Buffer::aligned_size() const {
    return this->num_dev_pages() * this->aligned_page_size();
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

uint32_t Buffer::num_cores() const {
    if (!is_sharded(this->buffer_layout_))
        return 1;

    return this->shard_spec().tensor_shard_spec.grid.num_cores();
}

DeviceAddr Buffer::translate_page_address(uint64_t offset, uint32_t bank_id) const {
    DeviceAddr base_page_address = this->address() + this->device_->bank_offset(this->buffer_type_, bank_id);
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

namespace detail {
buffer_map_t BUFFER_MAP = {};
}

}  // namespace tt_metal
}  // namespace tt

namespace tt::stl::json {
tt_metal::ShardSpec from_json_t<tt_metal::ShardSpec>::operator()(const nlohmann::json &json_object) const {
    return tt_metal::ShardSpec{
        from_json<CoreRangeSet>(json_object.at("grid")),
        from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
        from_json<tt_metal::ShardOrientation>(json_object.at("orientation")),
        from_json<bool>(json_object.at("halo"))};
}
}
