#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

Buffer::Buffer(Device *device, uint32_t size, uint32_t address, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type)
    : device_(device), size_(size), address_(address), starting_bank_id_(starting_bank_id), page_size_(page_size), buffer_type_(buffer_type) {
    this->bank_id_to_relative_address_ = this->device_->allocator_->allocate_buffer(starting_bank_id, size, page_size, address, buffer_type);
}

Buffer::Buffer(Device *device, uint32_t size, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type)
    : device_(device), size_(size), address_(std::numeric_limits<uint32_t>::max()), starting_bank_id_(starting_bank_id), page_size_(page_size), buffer_type_(buffer_type) {
    this->bank_id_to_relative_address_ = this->device_->allocator_->allocate_buffer(starting_bank_id, size, page_size, buffer_type);
    TT_ASSERT(this->bank_id_to_relative_address_.find(starting_bank_id) != this->bank_id_to_relative_address_.end());
    this->address_ = this->bank_id_to_relative_address_.at(starting_bank_id).absolute_address();
    TT_ASSERT(this->address_ != std::numeric_limits<uint32_t>::max());
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    TT_ASSERT(this->buffer_type_ == BufferType::DRAM);
    return this->device_->dram_channel_from_bank_id(bank_id);
}

tt_xy_pair Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    TT_ASSERT(this->buffer_type_ == BufferType::L1);
    return this->device_->logical_core_from_bank_id(bank_id);
}

tt_xy_pair Buffer::noc_coordinates(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    switch (this->buffer_type_) {
        case BufferType::DRAM: {
            auto dram_channel = this->dram_channel_from_bank_id(bank_id);
            return llrt::get_core_for_dram_channel(this->device_->cluster(), dram_channel, this->device_->pcie_slot());
        }
        case BufferType::L1: {
            auto logical_core = this->logical_core_from_bank_id(bank_id);
            return this->device_->worker_core_from_logical_core(logical_core);
        }
        break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Host buffer is located in system memory! Cannot retrieve NoC coordinates for it");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
    return tt_xy_pair(0, 0);
}

tt_xy_pair Buffer::noc_coordinates() const {
    return this->noc_coordinates(this->starting_bank_id_);
}

uint32_t Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    auto num_banks = this->device_->num_banks(this->buffer_type_);
    if (bank_id >= num_banks) {
        TT_THROW("Bank index " + std::to_string(bank_id) + " exceeds number of banks!");
    }
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    auto relative_address = this->bank_id_to_relative_address_.at(bank_id);
    auto absolute_address = relative_address.absolute_address();
    int pages_handled_in_bank = (int)page_index / num_banks;
    uint32_t offset = (this->page_size_ * pages_handled_in_bank);
    return absolute_address + offset;
}

void Buffer::deallocate() {
    if (this->device_ == nullptr or this->device_->closed_) {
        return;
    }
    for (auto &[bank_id, relative_address] : this->bank_id_to_relative_address_) {
        uint32_t abs_addr = relative_address.offset_bytes + relative_address.relative_address;
        this->device_->allocator_->deallocate_buffer(bank_id, abs_addr, this->buffer_type_);
    }
    this->bank_id_to_relative_address_.clear();
}

Buffer::~Buffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
