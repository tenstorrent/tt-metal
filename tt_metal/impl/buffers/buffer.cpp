// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/common/math.hpp"
#include "common/assert.hpp"
#include "tt_metal/impl/device/device.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

bool is_sharded(const TensorMemoryLayout & layout){
    return (layout==TensorMemoryLayout::SINGLE_BANK ||
        layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        layout == TensorMemoryLayout::WIDTH_SHARDED);
}

void validate_buffer_size_and_page_size(u64 size, u64 page_size, const BufferStorage &buffer_storage, const TensorMemoryLayout &buffer_layout, std::optional<ShardSpec > shard_parameters) {
    TT_ASSERT(size != 0 and page_size != 0, "Buffer size and page size should be larger than 0 bytes!");
    bool valid_page_size = (size % page_size == 0);
    TT_ASSERT(valid_page_size, "For valid non-interleaved buffers page size {} must equal buffer size {}. For interleaved-buffers page size should be divisible by buffer size", page_size, size);
    TT_ASSERT(page_size % sizeof(u32) == 0, "Page size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values");
    if(buffer_layout == TensorMemoryLayout::SINGLE_BANK){
        TT_ASSERT(page_size == size , "Continguous buffer must be one contiguous page");
    }
    else if(is_sharded(buffer_layout)){
        TT_ASSERT(shard_parameters != std::nullopt , "Sharded buffers must have a core grid assigned");
        TT_ASSERT(shard_parameters.value().element_size != std::nullopt, "Sharded buffers must specifiy element size");
    }
}

Buffer::Buffer(Device *device, u64 size, u64 page_size,
                const BufferStorage buffer_storage, const TensorMemoryLayout buffer_layout,
                std::optional< ShardSpec> shard_parameters)
    : device_(device), size_(size), page_size_(page_size), buffer_storage_(buffer_storage), buffer_layout_(buffer_layout) , shard_parameters_(shard_parameters){
    TT_ASSERT(this->device_ != nullptr and this->device_->allocator_ != nullptr);
    validate_buffer_size_and_page_size(size, page_size, buffer_storage, buffer_layout, shard_parameters);
    this->allocate();
}

Buffer::Buffer(const Buffer &other)
    : device_(other.device_), size_(other.size_), page_size_(other.page_size_), buffer_storage_(other.buffer_storage_) {
    this->allocate();
}

Buffer &Buffer::operator=(const Buffer &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->page_size_ = other.page_size_;
        this->buffer_storage_ = other.buffer_storage_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = other.shard_parameters_;
        this->allocate();
    }
    return *this;
}

Buffer::Buffer(Buffer &&other) : device_(other.device_), size_(other.size_), address_(other.address_), page_size_(other.page_size_), buffer_storage_(other.buffer_storage_) {
    // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
    other.device_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->page_size_ = other.page_size_;
        this->buffer_storage_ = other.buffer_storage_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = other.shard_parameters_;
        // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
        other.device_ = nullptr;
    }
    return *this;
}

void Buffer::allocate() {
    TT_ASSERT(this->device_ != nullptr);
    // L1 buffers are allocated top down!
    bool bottom_up = this->buffer_storage_ == BufferStorage::DRAM;

    if(is_sharded(this->buffer_layout_)){
        this->address_ = allocator::allocate_buffer(*this->device_->allocator_, this->size_, this->page_size_, this->buffer_storage_, bottom_up);
    }
    else{
        this->address_ = allocator::allocate_buffer(*this->device_->allocator_, this->size_, this->page_size_, this->buffer_storage_, bottom_up);
    }
}

u32 Buffer::dram_channel_from_bank_id(u32 bank_id) const {
    TT_ASSERT(this->buffer_storage_ == BufferStorage::DRAM, "Expected DRAM buffer!");
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(u32 bank_id) const {
    TT_ASSERT(this->buffer_storage_ == BufferStorage::L1, "Expected L1 buffer!");
    return this->device_->logical_core_from_bank_id(bank_id);
}

CoreCoord Buffer::noc_coordinates(u32 bank_id) const {
    switch (this->buffer_storage_) {
        case BufferStorage::DRAM: {
            auto dram_channel = this->dram_channel_from_bank_id(bank_id);
            return llrt::get_core_for_dram_channel(dram_channel, this->device_->id());
        }
        case BufferStorage::L1: {
            auto logical_core = this->logical_core_from_bank_id(bank_id);
            return this->device_->worker_core_from_logical_core(logical_core);
        }
        break;
        case BufferStorage::SYSTEM_MEMORY: {
            TT_THROW("Host buffer is located in system memory! Cannot retrieve NoC coordinates for it");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
    return CoreCoord{.x=0, .y=0};
}

CoreCoord Buffer::noc_coordinates() const {
    return this->noc_coordinates(0);
}

u64 Buffer::page_address(u32 bank_id, u32 page_index) const {
    auto num_banks = this->device_->num_banks(this->buffer_storage_);
    TT_ASSERT(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);


    // DRAM readers and writers in Cluster add DRAM bank offset before doing a read but L1 readers and writers do not

    //TODO: investigate why second page on each bank is not working
    // Added offset to bypass this
    u64 base_page_address = this->buffer_storage_ == BufferStorage::DRAM ?
        this->address_ :
        this->address_ + this->device_->l1_bank_offset_from_bank_id(bank_id);

    int pages_offset_within_bank = (int)page_index / num_banks;
    if( is_sharded(this->buffer_layout_)){
        pages_offset_within_bank = page_index % shard_size();

    }
    auto offset = (round_up(this->page_size_, ADDRESS_ALIGNMENT) * pages_offset_within_bank);
    return base_page_address + offset;
}

void Buffer::deallocate() {
    if (this->device_ == nullptr or not this->device_->initialized_) {
        return;
    }
    TT_ASSERT(this->device_->allocator_ != nullptr, "Expected allocator to be initialized!");
    allocator::deallocate_buffer(*this->device_->allocator_, this->address_, this->buffer_storage_);
}

Buffer::~Buffer() {
    this->deallocate();
}

tt::stl::reflection::Attributes ShardSpec::attributes() const {
    return {
        {"shard_grid", this->shard_grid.str()},
        {"shard_shape", this->shard_shape},
        {"shard_orientation", this->shard_orientation},
        {"halo", this->halo},
    };
}

}  // namespace tt_metal

}  // namespace tt
