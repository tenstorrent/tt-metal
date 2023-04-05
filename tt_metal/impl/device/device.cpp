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

void Device::initialize_banked_dram_manager() {
    TT_ASSERT(cluster_is_initialized() && "Cluster needs to be initialized!");

    for (int dram_bank = 0; dram_bank < this->num_dram_banks(); dram_bank++) {
        this->banked_dram_manager_.insert(
            {
                dram_bank,
                new MemoryManager(this->cluster_->get_soc_desc(this->pcie_slot_).dram_bank_size)
            }
        );
    }
}

void Device::initialize_l1_manager() {
    TT_ASSERT(cluster_is_initialized() && "Cluster needs to be initialized!");

    auto soc_desc = cluster_->get_soc_desc(pcie_slot_);
    for (auto worker_core : soc_desc.workers) {
        auto logical_core_x = soc_desc.routing_x_to_worker_x.at(worker_core.x);
        auto logical_core_y = soc_desc.routing_y_to_worker_y.at(worker_core.y);
        tt_xy_pair logical_core = tt_xy_pair(logical_core_x, logical_core_y);
        auto l1_mem_manager = new MemoryManager(soc_desc.worker_l1_size);
        // Space up to UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
        l1_mem_manager->reserve(0, UNRESERVED_BASE);
        this->l1_manager_.insert({logical_core, l1_mem_manager});
    }
}

bool Device::initialize() {
    this->initialize_cluster();
    this->initialize_banked_dram_manager();
    this->initialize_l1_manager();
    this->closed_ = false;
    return true;
}

bool Device::close() {
    llrt::assert_reset_for_all_chips(cluster_);
    cluster_->close_device();
    cluster_ = nullptr;
    for (auto const& [dram_bank, dram_memory_manager] : this->banked_dram_manager_) {
        dram_memory_manager->clear();
    }
    for (auto const& [logical_core, l1_memory_manager] : this->l1_manager_) {
        l1_memory_manager->clear();
    }
    this->closed_ = true;
    return true;
}

Device::~Device() {
    if (this->cluster_is_initialized()) {
        this->close();
    }
    for (auto const& [dram_bank, dram_memory_manager] : this->banked_dram_manager_) {
        delete dram_memory_manager;
    }
    for (auto const& [logical_core, l1_memory_manager] : this->l1_manager_) {
        delete l1_memory_manager;
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

uint32_t Device::allocate_dram_buffer(int dram_channel, uint32_t size_in_bytes) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    auto buffer_address = memory_manager->allocate(size_in_bytes);
    return buffer_address;
}

uint32_t Device::reserve_dram_buffer(int dram_channel, uint32_t size_in_bytes, uint32_t address) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    auto buffer_address = memory_manager->reserve(address, size_in_bytes);
    TT_ASSERT(buffer_address == address);
    return buffer_address;
}

void Device::free_dram_buffer(int dram_channel, uint32_t address) {
    if (this->closed_) {
        return;
    }

    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->banked_dram_manager_.at(dram_channel);
    memory_manager->deallocate(address);
}

uint32_t Device::allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_in_bytes) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->l1_manager_.at(logical_core);
    auto buffer_addess = memory_manager->allocate(size_in_bytes);
    return buffer_addess;
}

uint32_t Device::reserve_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address) {
    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->l1_manager_.at(logical_core);
    auto buffer_address = memory_manager->reserve(address, size_in_bytes);
    TT_ASSERT(buffer_address == address);
    return buffer_address;
}

void Device::free_l1_buffer(const tt_xy_pair &logical_core, uint32_t address) {
    if (this->closed_) {
        return;
    }

    if (not is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto memory_manager = this->l1_manager_.at(logical_core);
    memory_manager->deallocate(address);
}

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges) {
    uint32_t smallest_chunk = candidate_addr_ranges[0].second - candidate_addr_ranges[0].first;
    uint32_t address = candidate_addr_ranges[0].first;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        uint32_t range_size = candidate_addr_range.second - candidate_addr_range.first;
        if (range_size < smallest_chunk) {
            smallest_chunk = range_size;
            address = candidate_addr_range.first;
        }
    }
    return address;
}

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges) {
    if (candidate_addr_ranges.empty()) {
        candidate_addr_ranges = potential_addr_ranges;
        return;
    }
    int i = 0;
    int j = 0;
    std::vector<std::pair<uint32_t, uint32_t>> intersecting_addr_ranges;
    while (i < candidate_addr_ranges.size() and j < potential_addr_ranges.size()) {
        uint32_t lower_addr = std::max(candidate_addr_ranges[i].first, potential_addr_ranges[j].first);
        uint32_t upper_addr = std::min(candidate_addr_ranges[i].second, potential_addr_ranges[j].second);
        if (lower_addr <= upper_addr) {
            intersecting_addr_ranges.push_back({lower_addr, upper_addr});
        }
        if (candidate_addr_ranges[i].second < potential_addr_ranges[j].second) {
            i++;
        } else {
            j++;
        }
    }
    candidate_addr_ranges = intersecting_addr_ranges;
}

uint32_t Device::address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank) {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    uint32_t total_size_bytes = 0;
    for (auto &[dram_bank, required_size_bytes] : size_in_bytes_per_bank) {
        auto potential_addr_ranges = this->banked_dram_manager_.at(dram_bank)->available_addresses(required_size_bytes);
        populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        total_size_bytes += required_size_bytes;
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space to hold interleave " + std::to_string(total_size_bytes) + " bytes across DRAM channels");
    }

    return find_address_of_smallest_chunk(candidate_addr_ranges);
}

uint32_t Device::address_for_l1_buffers_across_core_range(const CoreRange &logical_core_range, uint32_t size_in_bytes) {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    auto start_core = logical_core_range.first;
    auto end_core = logical_core_range.second;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            auto potential_addr_ranges = this->l1_manager_.at(logical_core)->available_addresses(size_in_bytes);
            populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        }
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space for " + std::to_string(size_in_bytes) +
            " byte CircularBuffers in cores ranging from " + start_core.str() + " to " + end_core.str());
    }

    return find_address_of_smallest_chunk(candidate_addr_ranges);
}

}  // namespace tt_metal

}  // namespace tt
