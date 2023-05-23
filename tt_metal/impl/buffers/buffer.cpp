#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

void validate_buffer_size_and_page_size(uint32_t size, uint32_t page_size, const BufferType &buffer_type) {
    if (size == 0) {
        TT_THROW("Buffer size should be larger than 0 bytes!");
    }
    bool valid_page_size = (size == page_size) or (page_size > 0 and size % page_size == 0);
    if (not valid_page_size) {
        TT_THROW("For non-interleaved buffers page size of " + std::to_string(page_size) + " bytes should be equal to buffer size of " + std::to_string(size) + ". " +
                 "For interleaved buffers page size should be divisible by total buffer size." );
    }
}

Buffer::Buffer(Device *device, uint32_t size, uint32_t address, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type)
    : device_(device), size_(size), address_(address), starting_bank_id_(starting_bank_id), page_size_(page_size), buffer_type_(buffer_type) {
    TT_ASSERT(this->device_ != nullptr and this->device_->allocator_ != nullptr);
    validate_buffer_size_and_page_size(size, page_size, buffer_type);
    this->bank_id_to_relative_address_ = allocator::allocate_buffer_at_address(*this->device_->allocator_, starting_bank_id, size, page_size, address, buffer_type);
    TT_ASSERT(this->bank_id_to_relative_address_.find(this->starting_bank_id_) != this->bank_id_to_relative_address_.end());
}

Buffer::Buffer(Device *device, uint32_t size, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type)
    : device_(device), size_(size), address_(std::numeric_limits<uint32_t>::max()), starting_bank_id_(starting_bank_id), page_size_(page_size), buffer_type_(buffer_type) {
    TT_ASSERT(this->device_ != nullptr and this->device_->allocator_ != nullptr);
    validate_buffer_size_and_page_size(size, page_size, buffer_type);
    this->allocate();
    TT_ASSERT(this->address_ != std::numeric_limits<uint32_t>::max());
}

Buffer::Buffer(const Buffer &other)
    : device_(other.device_), size_(other.size_), address_(std::numeric_limits<uint32_t>::max()), starting_bank_id_(other.starting_bank_id_), page_size_(other.page_size_), buffer_type_(other.buffer_type_) {
    this->allocate();
    TT_ASSERT(this->address_ != std::numeric_limits<uint32_t>::max());
}

Buffer &Buffer::operator=(const Buffer &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->starting_bank_id_ = other.starting_bank_id_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->allocate();
    }
    return *this;
}

Buffer::Buffer(Buffer &&other)
    : device_(other.device_), size_(other.size_), address_(other.address_), starting_bank_id_(other.starting_bank_id_), page_size_(other.page_size_), buffer_type_(other.buffer_type_), bank_id_to_relative_address_(other.bank_id_to_relative_address_) {
    other.bank_id_to_relative_address_.clear();
    other.device_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->starting_bank_id_ = other.starting_bank_id_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->bank_id_to_relative_address_ = other.bank_id_to_relative_address_;
        other.bank_id_to_relative_address_.clear();
        other.device_ = nullptr;
    }
    return *this;
}

void Buffer::allocate() {
    TT_ASSERT(this->device_ != nullptr);
    bool bottom_up = true;
    if (this->device_->allocator_scheme() == MemoryAllocator::L1_BANKING and this->buffer_type_ == BufferType::L1) {
        bottom_up = false;
    }
    this->bank_id_to_relative_address_ = allocator::allocate_buffer(*this->device_->allocator_, this->starting_bank_id_, this->size_, this->page_size_, this->buffer_type_, bottom_up);
    TT_ASSERT(this->bank_id_to_relative_address_.find(this->starting_bank_id_) != this->bank_id_to_relative_address_.end());
    this->address_ = this->bank_id_to_relative_address_.at(this->starting_bank_id_).absolute_address();
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    TT_ASSERT(this->buffer_type_ == BufferType::DRAM);
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_relative_address_.find(bank_id) != this->bank_id_to_relative_address_.end());
    TT_ASSERT(this->buffer_type_ == BufferType::L1);
    return this->device_->logical_core_from_bank_id(bank_id);
}

CoreCoord Buffer::noc_coordinates(uint32_t bank_id) const {
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
    return CoreCoord(0, 0);
}

CoreCoord Buffer::noc_coordinates() const {
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
    TT_ASSERT(this->device_->allocator_ != nullptr);
    for (auto &[bank_id, relative_address] : this->bank_id_to_relative_address_) {
        uint32_t abs_addr = relative_address.offset_bytes + relative_address.relative_address;
        allocator::deallocate_buffer(*this->device_->allocator_, bank_id, abs_addr, this->buffer_type_);
    }
    this->bank_id_to_relative_address_.clear();
}

Buffer::~Buffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
