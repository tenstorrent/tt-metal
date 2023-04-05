#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

DramBuffer::DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes) : dram_channel_(dram_channel), Buffer(device, size_in_bytes, -1, true) {
    this->address_ = device->allocate_dram_buffer(dram_channel, size_in_bytes);
}

DramBuffer::DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address) : dram_channel_(dram_channel), Buffer(device, size_in_bytes, address, true) {
    device->reserve_dram_buffer(dram_channel, size_in_bytes, address);
}

Buffer *DramBuffer::clone() {
    return new DramBuffer(this->device_, this->dram_channel_, this->size_in_bytes_);
}

tt_xy_pair DramBuffer::noc_coordinates() const {
    return llrt::get_core_for_dram_channel(this->device_->cluster(), this->dram_channel_, this->device_->pcie_slot());
}

void DramBuffer::free() {
    if (this->allocated_on_device_) {
        this->device_->free_dram_buffer(this->dram_channel_, this->address_);
        this->allocated_on_device_ = false;
    }
}

DramBuffer::~DramBuffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

L1Buffer::L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes) : logical_core_(logical_core), Buffer(device, size_in_bytes, -1, true) {
    this->address_ = device->allocate_l1_buffer(logical_core, size_in_bytes);
}

L1Buffer::L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address) : logical_core_(logical_core), Buffer(device, size_in_bytes, address, false) {
    // TODO (abhullar): Enable this when we have a spec for overlapping buffers in L1
    //device->reserve_l1_buffer(logical_core, size_in_bytes, address);
    TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
}

Buffer *L1Buffer::clone() {
    return new L1Buffer(this->device_, this->logical_core_, this->size_in_bytes_);
}

tt_xy_pair L1Buffer::noc_coordinates() const {
    return this->device_->worker_core_from_logical_core(this->logical_core_);
}

void L1Buffer::reserve() {
    auto address = this->device_->reserve_l1_buffer(this->logical_core_, this->size_in_bytes_, this->address_);
    TT_ASSERT(address == this->address_);
}

void L1Buffer::free() {
    if (this->allocated_on_device_) {
        this->device_->free_l1_buffer(this->logical_core_, this->address_);
        this->allocated_on_device_ = false;
    }
}

L1Buffer::~L1Buffer() {
    this->free();
}

}  // namespace tt_metal

}  // namespace tt
