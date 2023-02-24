#include "ll_buda/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace ll_buda {

tt_xy_pair DramBuffer::noc_coordinates() const {
    return llrt::get_core_for_dram_channel(device_->cluster(), dram_channel_, device_->pcie_slot());
}

}  // namespace ll_buda

}  // namespace tt
