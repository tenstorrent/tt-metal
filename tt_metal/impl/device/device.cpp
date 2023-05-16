#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

void Device::initialize_cluster() {
    std::set<chip_id_t> target_device_ids = {pcie_slot_};
    tt_device_params default_params;
    const std::string sdesc_file = get_soc_description_file(arch_, target_type_);

    this->cluster_ = new tt_cluster();
    this->cluster_->open_device(arch_, target_type_, target_device_ids, sdesc_file);
    this->cluster_->start_device(default_params);

    llrt::utils::log_current_ai_clk(cluster_);
    llrt::assert_reset_for_all_chips(cluster_);
}

void Device::initialize_allocator(const MemoryAllocator &memory_allocator) {
    TT_ASSERT(cluster_is_initialized() && "Cluster needs to be initialized!");
    auto soc_desc = this->cluster_->get_soc_desc(this->pcie_slot_);
    switch (memory_allocator) {
        case MemoryAllocator::BASIC: {
            this->allocator_ = std::make_unique<BasicAllocator>(soc_desc);
        }
        break;
        case MemoryAllocator::L1_BANKING: {
            this->allocator_ = std::make_unique<L1BankingAllocator>(soc_desc);
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported memory allocator");
    }
    this->allocator_scheme_ = memory_allocator;
}

bool Device::initialize(const MemoryAllocator &memory_allocator) {
    this->initialize_cluster();
    this->initialize_allocator(memory_allocator);
    this->closed_ = false;
    return true;
}

bool Device::close() {
    llrt::assert_reset_for_all_chips(cluster_);
    cluster_->close_device();
    cluster_ = nullptr;
    allocator::clear(*this->allocator_);
    this->closed_ = true;
    return true;
}

Device::~Device() {
    if (this->cluster_is_initialized()) {
        this->close();
    }
    delete this->cluster_;
}

tt_cluster *Device::cluster() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return cluster_;
}

int Device::num_dram_channels() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).get_num_dram_channels();
}

uint32_t Device::l1_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).worker_l1_size;
}

tt_xy_pair Device::logical_grid_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).worker_grid_size;
}

tt_xy_pair Device::compute_and_storage_grid_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).compute_and_storage_grid_size;
}

tt_xy_pair Device::worker_core_from_logical_core(const tt_xy_pair &logical_core) const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto worker_core_x = cluster_->get_soc_desc(pcie_slot_).worker_log_to_routing_x.at(logical_core.x);
    auto worker_core_y = cluster_->get_soc_desc(pcie_slot_).worker_log_to_routing_y.at(logical_core.y);
    tt_xy_pair worker_core = tt_xy_pair(worker_core_x, worker_core_y);
    return worker_core;
}

std::vector<tt_xy_pair> Device::worker_cores_from_logical_cores(const std::vector<tt_xy_pair> &logical_cores) {
    std::vector<tt_xy_pair> worker_cores;
    for (auto logical_core : logical_cores) {
        worker_cores.push_back(worker_core_from_logical_core(logical_core));
    }
    return worker_cores;
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

tt_xy_pair Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

std::vector<uint32_t> Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

std::vector<uint32_t> Device::bank_ids_from_logical_core(const tt_xy_pair &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, logical_core);
}

}  // namespace tt_metal

}  // namespace tt
