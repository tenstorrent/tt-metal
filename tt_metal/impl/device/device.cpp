#include "tt_metal/impl/device/device.hpp"

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

void Device::initialize_banked_dram_manager() {
    TT_ASSERT(cluster_is_initialized() && "Cluster needs to be initialized!");

    for (int dram_bank = 0; dram_bank < num_dram_banks(); dram_bank++) {
        this->banked_dram_manager_.insert(
            {
                dram_bank,
                new MemoryManager(this->cluster_->get_soc_desc(this->pcie_slot_).dram_bank_size)
            }
        );
    }
}

bool Device::initialize() {
    initialize_cluster();
    initialize_banked_dram_manager();
    this->closed_ = false;
    return true;
}

bool Device::close() {
    llrt::assert_reset_for_all_chips(cluster_);
    cluster_->close_device();
    cluster_ = nullptr;
    for (auto const& [dram_bank, memory_manager] : this->banked_dram_manager_) {
        memory_manager->clear();
    }
    this->closed_ = true;
    return true;
}

Device::~Device() {
    if (this->cluster_is_initialized()) {
        this->close();
    }
    for (auto const& [dram_bank, memory_manager] : this->banked_dram_manager_) {
        delete memory_manager;
    }
    delete this->cluster_;
}

tt_cluster *Device::cluster() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return cluster_;
}

int Device::num_dram_banks() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).get_num_dram_channels();
}

tt_xy_pair Device::logical_grid_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster_->get_soc_desc(pcie_slot_).worker_grid_size;
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

uint32_t Device::allocate_buffer(int dram_channel, uint32_t size_in_bytes) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    auto buffer_address = memory_manager->allocate(size_in_bytes);
    return buffer_address;
}

uint32_t Device::allocate_buffer(int dram_channel, uint32_t size_in_bytes, uint32_t address) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    auto buffer_address = memory_manager->reserve(address, size_in_bytes);
    TT_ASSERT(buffer_address == address);
    return buffer_address;
}

void Device::free_buffer(int dram_channel, uint32_t size_in_bytes, uint32_t address) {
    if (this->closed_) {
        return;
    }

    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    memory_manager->deallocate(address);
}

}  // namespace tt_metal

}  // namespace tt
