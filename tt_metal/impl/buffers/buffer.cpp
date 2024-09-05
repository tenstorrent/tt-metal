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
    TT_FATAL(size != 0 and page_size != 0, "Buffer size and page size should be larger than 0 bytes!");
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
    std::array<uint32_t, 2> shard_in_pages = {shard_shape[0] / page_shape[0], shard_shape[1] / page_shape[1]};
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
        uint32_t num_shard_columns = div_up(tensor2d_size[1], shard_in_pages[1]);
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

Buffer::Buffer(
    Device *device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<bool> bottom_up,
    bool allocate) :
    device_(device),
    size_(size),
    page_size_(page_size),
    buffer_type_(buffer_type),
    buffer_layout_(buffer_layout),
    shard_parameters_(shard_parameters),
    bottom_up_(bottom_up),
    buffer_page_mapping_(nullptr),
    allocate_(allocate) {
    TT_FATAL(this->device_ != nullptr and this->device_->allocator_ != nullptr);
    validate_buffer_size_and_page_size(size, page_size, buffer_type, buffer_layout, shard_parameters);
    if (allocate) {
        this->allocate();
    }
}

BufferPageMapping generate_buffer_page_mapping(const Buffer &buffer) {
    BufferPageMapping buffer_page_mapping;
    bool row_major = buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR;
    uint32_t num_cores = buffer.num_cores();

    buffer_page_mapping.all_cores_ = corerange_to_cores(buffer.shard_spec().grid(), num_cores, row_major);
    TT_ASSERT(num_cores == buffer_page_mapping.all_cores_.size());
    uint32_t core_id = 0;
    for (const auto &core : buffer_page_mapping.all_cores_) {
        buffer_page_mapping.core_to_core_id_.insert({core, core_id});
        core_id++;
    }

    uint32_t num_dev_pages = buffer.num_dev_pages();
    auto [core_host_page_indices, shard_shape] = core_to_host_pages(
        num_dev_pages,
        buffer.shard_spec().size(),
        num_cores,
        buffer.buffer_layout(),
        buffer.shard_spec().page_shape,
        buffer.shard_spec().shape(),
        buffer.shard_spec().tensor2d_shape);

    buffer_page_mapping.core_host_page_indices_ = std::vector<std::vector<uint32_t>>(num_cores);

    buffer_page_mapping.dev_page_to_host_page_mapping_ =
        std::vector<std::optional<uint32_t>>(num_dev_pages, std::nullopt);
    buffer_page_mapping.dev_page_to_core_mapping_ = std::vector<uint32_t>(num_dev_pages);

    buffer_page_mapping.host_page_to_local_shard_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.host_page_to_dev_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.core_shard_shape_ = std::move(shard_shape);
    uint32_t dev_page_index = 0;

    auto shape_in_pages = buffer.shard_spec().shape_in_pages();
    for (uint32_t core_index = 0; core_index < core_host_page_indices.size(); core_index++) {
        uint32_t valid_shard_page = 0;
        buffer_page_mapping.core_host_page_indices_[core_index].reserve(buffer.shard_spec().size());
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

Buffer::Buffer(const Buffer &other) :
    device_(other.device_),
    size_(other.size_),
    page_size_(other.page_size_),
    buffer_type_(other.buffer_type_),
    buffer_layout_(other.buffer_layout_),
    shard_parameters_(other.shard_parameters_),
    bottom_up_(other.bottom_up_),
    buffer_page_mapping_(other.buffer_page_mapping_),
    allocate_(other.allocate_) {
    if (this->allocate_) {
        this->allocate();
    }
}

Buffer &Buffer::operator=(const Buffer &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = other.shard_parameters_;
        this->bottom_up_ = other.bottom_up_;
        this->buffer_page_mapping_ = other.buffer_page_mapping_;
        this->allocate_ = other.allocate_;
        if (this->allocate_) {
            this->allocate();
        }
    }
    return *this;
}

Buffer::Buffer(Buffer &&other) :
    device_(other.device_),
    size_(other.size_),
    address_(other.address_),
    page_size_(other.page_size_),
    buffer_type_(other.buffer_type_),
    buffer_layout_(other.buffer_layout_),
    shard_parameters_(std::move(other.shard_parameters_)),
    bottom_up_(other.bottom_up_),
    buffer_page_mapping_(std::move(other.buffer_page_mapping_)),
    allocate_(other.allocate_) {
    // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is
    // transferred to `this`
    other.device_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = std::move(other.shard_parameters_);
        this->bottom_up_ = other.bottom_up_;
        this->buffer_page_mapping_ = std::move(other.buffer_page_mapping_);
        this->allocate_ = other.allocate_;
        // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is
        // transferred to `this`
        other.device_ = nullptr;
    }
    return *this;
}

void Buffer::allocate() {
    TT_ASSERT(this->device_ != nullptr);
    // L1 and Trace buffers (which live in DRAM) are allocated top down!
    bool bottom_up = this->bottom_up_.value_or(this->is_dram());
    detail::AllocateBuffer(this, bottom_up);
    detail::BUFFER_MAP.insert({this->device_->id(), this->address_}, this);
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->is_dram(), "Expected DRAM buffer!");
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->is_l1(), "Expected L1 buffer!");
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
    TT_ASSERT(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    int pages_offset_within_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size_, this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

DeviceAddr Buffer::sharded_page_address(uint32_t bank_id, uint32_t page_index) const {
    TT_ASSERT(is_sharded(this->buffer_layout()));
    int pages_offset_within_bank = page_index % shard_spec().size();
    auto offset = (round_up(this->page_size_, this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

DeviceAddr Buffer::translate_page_address(uint64_t offset, uint32_t bank_id) const {
    DeviceAddr base_page_address = this->address_ + this->device_->bank_offset(this->buffer_type_, bank_id);
    return base_page_address + offset;
}

const std::shared_ptr<const BufferPageMapping>& Buffer::get_buffer_page_mapping() {
    TT_ASSERT(is_sharded(this->buffer_layout_), "Buffer not sharded");
    if (!this->buffer_page_mapping_) {
        this->buffer_page_mapping_ = std::make_shared<const BufferPageMapping>(generate_buffer_page_mapping(*this));
    }
    return this->buffer_page_mapping_;
}

void Buffer::deallocate() {

    if (this->device_ == nullptr or not this->device_->initialized_ or this->size_ == 0 or not this->allocate_) {
        return;
    }
    // Mark as deallocated
    this->size_ = 0;
    TT_ASSERT(this->device_->allocator_ != nullptr, "Expected allocator to be initialized!");
    detail::BUFFER_MAP.erase({this->device_->id(), this->address_});
    detail::DeallocateBuffer(this);
}

Buffer::~Buffer() { this->deallocate(); }

bool operator==(const ShardSpec &spec_a, const ShardSpec &spec_b) {
    if (spec_a.grid != spec_b.grid) {
        return false;
    }
    if (spec_a.shape != spec_b.shape) {
        return false;
    }
    if (spec_a.orientation != spec_b.orientation) {
        return false;
    }
    if (spec_a.halo != spec_b.halo) {
        return false;
    }
    return true;
}

bool operator!=(const ShardSpec &spec_a, const ShardSpec &spec_b) { return not(spec_a == spec_b); }

namespace detail {
buffer_map_t BUFFER_MAP = {};
}

}  // namespace tt_metal
}  // namespace tt
