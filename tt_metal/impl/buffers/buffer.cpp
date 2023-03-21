#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

tt_xy_pair DramBuffer::noc_coordinates() const {
    return llrt::get_core_for_dram_channel(this->device_->cluster(), this->dram_channel_, this->device_->pcie_slot());
}

tt_xy_pair L1Buffer::noc_coordinates() const {
    return this->device_->worker_core_from_logical_core(this->logical_core_);
}

}  // namespace tt_metal

}  // namespace tt
