#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

DramBuffer::DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes) : dram_channel_(dram_channel), Buffer(device, size_in_bytes, -1, true) {
    this->address_ = device->allocate_buffer(dram_channel, size_in_bytes);
}

DramBuffer::DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address) : dram_channel_(dram_channel), Buffer(device, size_in_bytes, address, true) {
    device->allocate_buffer(dram_channel, size_in_bytes, address);
}

Buffer *DramBuffer::clone() {
    return new DramBuffer(this->device_, this->dram_channel_, this->size_in_bytes_);
}

tt_xy_pair DramBuffer::noc_coordinates() const {
    return llrt::get_core_for_dram_channel(this->device_->cluster(), this->dram_channel_, this->device_->pcie_slot());
}

void DramBuffer::free() {
    if (this->allocated_on_device_) {
        this->device_->free_buffer(this->dram_channel_, this->size_in_bytes_, this->address_);
        this->allocated_on_device_ = false;
    }
}

DramBuffer::~DramBuffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

Buffer *L1Buffer::clone() {
    TT_ASSERT(false && "L1 buffer clone is not currently supported");
    return new L1Buffer(this->device_, this->logical_core_, this->size_in_bytes_, this->address_);
}

tt_xy_pair L1Buffer::noc_coordinates() const {
    return this->device_->worker_core_from_logical_core(this->logical_core_);
}

}  // namespace tt_metal

}  // namespace tt
