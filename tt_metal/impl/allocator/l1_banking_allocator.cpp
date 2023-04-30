#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

#include <cmath>

namespace tt {

namespace tt_metal {

L1BankingAllocator::L1BankingAllocator(const tt_SocDescriptor &soc_desc) : Allocator() {
    this->init_dram_manager(soc_desc);
    this->init_compute_and_storage_cores_l1_manager(soc_desc);
    this->init_storage_cores_l1_manager(soc_desc);
}

void L1BankingAllocator::init_dram_manager(const tt_SocDescriptor &soc_desc) {
    // Initialize DRAM manager
    for (int dram_channel = 0; dram_channel < soc_desc.get_num_dram_channels(); dram_channel++) {
        auto allocator = std::make_unique<allocator::FreeList>(
            soc_desc.dram_bank_size,
            this->min_allocation_size_bytes_,
            this->alignment_,
            allocator::FreeList::SearchPolicy::FIRST
        );
        this->dram_manager_.insert({dram_channel, std::move(allocator)});
    }
}

void L1BankingAllocator::init_compute_and_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc) {
    uint32_t compute_and_storage_core_bank_size_bytes = soc_desc.worker_l1_size;
    TT_ASSERT(compute_and_storage_core_bank_size_bytes / this->num_banks_per_storage_core_ == this->storage_core_bank_size_bytes_);
    // Initialize L1 manager for compute and storage cores
    for (auto compute_and_storage_core : soc_desc.compute_and_storage_cores) {
        auto logical_core_x = soc_desc.routing_x_to_worker_x.at(compute_and_storage_core.x);
        auto logical_core_y = soc_desc.routing_y_to_worker_y.at(compute_and_storage_core.y);
        tt_xy_pair logical_core = tt_xy_pair(logical_core_x, logical_core_y);
        auto allocator = std::make_unique<allocator::FreeList>(
            compute_and_storage_core_bank_size_bytes,
            this->min_allocation_size_bytes_,
            this->alignment_,
            allocator::FreeList::SearchPolicy::FIRST
        );
        // Space up to UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
        allocator->allocate_at_address(0, UNRESERVED_BASE);
        auto compute_and_storage_bank = std::make_unique<Bank>(std::move(allocator), 0);
        this->compute_and_storage_cores_l1_manager_.emplace(logical_core, std::move(compute_and_storage_bank));
    }
}

void L1BankingAllocator::init_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc) {
    // Initialize L1 manager for storage only cores
    for (auto storage_core : soc_desc.storage_cores) {
        auto logical_core_x = soc_desc.routing_x_to_worker_x.at(storage_core.x);
        auto logical_core_y = soc_desc.routing_y_to_worker_y.at(storage_core.y);
        tt_xy_pair logical_core = tt_xy_pair(logical_core_x, logical_core_y);
        UniqueBanks banks(this->num_banks_per_storage_core_);
        for (int bank_idx = 0; bank_idx < this->num_banks_per_storage_core_; bank_idx++) {
            auto allocator = std::make_unique<allocator::FreeList>(
                this->storage_core_bank_size_bytes_,
                this->min_allocation_size_bytes_,
                this->alignment_,
                allocator::FreeList::SearchPolicy::FIRST
            );
            auto storage_bank = std::make_unique<Bank>(std::move(allocator), bank_idx * this->storage_core_bank_size_bytes_);
            banks.at(bank_idx) = std::move(storage_bank);
        }
        this->storage_cores_l1_manager_.emplace(logical_core, std::move(banks));
    }
}

// L1BankingAllocator::L1BankingAllocator(const L1BankingAllocator &other);
// L1BankingAllocator& operator=(const L1BankingAllocator &other);

// L1BankingAllocator(L1BankingAllocator &&other);
// L1BankingAllocator& operator=(L1BankingAllocator &&other);

allocator::Algorithm &L1BankingAllocator::allocator_for_dram_channel(int dram_channel) const {
    if (this->dram_manager_.find(dram_channel) == this->dram_manager_.end()) {
        TT_THROW("Allocator for DRAM channel " + std::to_string(dram_channel) + " does not exist!");
    }
    return *this->dram_manager_.at(dram_channel);
}

bool L1BankingAllocator::is_compute_and_storage_core(const tt_xy_pair &logical_core) const {
    if (this->compute_and_storage_cores_l1_manager_.find(logical_core) != this->compute_and_storage_cores_l1_manager_.end()) {
        return true;
    }
    return false;
}

bool L1BankingAllocator::is_storage_only_core(const tt_xy_pair &logical_core) const {
    if (this->storage_cores_l1_manager_.find(logical_core) != this->storage_cores_l1_manager_.end()) {
        return true;
    }
    return false;
}

L1BankingAllocator::Bank &L1BankingAllocator::bank_for_logical_compute_and_storage_core(const tt_xy_pair &logical_core) const {
    if (not this->is_compute_and_storage_core(logical_core)) {
        TT_THROW(logical_core.str() + " is not a compute and storage core!");
    }
    return *this->compute_and_storage_cores_l1_manager_.at(logical_core);
}

L1BankingAllocator::UniqueBanks &L1BankingAllocator::banks_for_storage_only_cores(const tt_xy_pair &logical_core) {
    if (not this->is_storage_only_core(logical_core)) {
        TT_THROW(logical_core.str() + " is not a storage core!");
    }
    return this->storage_cores_l1_manager_.at(logical_core);
}

L1BankingAllocator::Bank &L1BankingAllocator::bank_for_logical_core(const tt_xy_pair &logical_core, uint32_t absolute_address) const {
    if (this->is_compute_and_storage_core(logical_core)) {
        return *this->compute_and_storage_cores_l1_manager_.at(logical_core);
    } else {
        TT_ASSERT(this->is_storage_only_core(logical_core));
        int bank_index = absolute_address < this->storage_core_bank_size_bytes_ ? 0 : 1;
        return *this->storage_cores_l1_manager_.at(logical_core).at(bank_index);
    }
}

uint32_t L1BankingAllocator::allocate_dram_buffer(int dram_channel, uint32_t size_bytes) {
    auto address = this->allocator_for_dram_channel(dram_channel).allocate(size_bytes, true);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for DRAM buffer in channel " + std::to_string(dram_channel));
    }
    return address.value();
}

uint32_t L1BankingAllocator::allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes) {
    auto address = this->allocator_for_dram_channel(dram_channel).allocate_at_address(start_address, size_bytes);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for DRAM buffer in channel " + std::to_string(dram_channel) + " at " + std::to_string(start_address));
    }
    return address.value();
}

uint32_t L1BankingAllocator::get_address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank) const {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    uint32_t total_size_bytes = 0;
    for (auto &[dram_channel, required_size_bytes] : size_in_bytes_per_bank) {
        auto potential_addr_ranges = this->allocator_for_dram_channel(dram_channel).available_addresses(required_size_bytes);
        allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        total_size_bytes += required_size_bytes;
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space to hold interleave " + std::to_string(total_size_bytes) + " bytes across DRAM channels");
    }

    return allocator::find_address_of_smallest_chunk(candidate_addr_ranges);
}

void L1BankingAllocator::deallocate_dram_buffer(int dram_channel, uint32_t address) {
    this->allocator_for_dram_channel(dram_channel).deallocate(address);
}

uint32_t L1BankingAllocator::allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) {
    auto &compute_and_storage_bank = this->bank_for_logical_compute_and_storage_core(logical_core);
    auto address = compute_and_storage_bank.allocator_algo->allocate(size_bytes, true);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for circular buffer on core " + logical_core.str());
    }
    return compute_and_storage_bank.offset_bytes + address.value();
}

uint32_t L1BankingAllocator::allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) {
    auto &compute_and_storage_bank = this->bank_for_logical_compute_and_storage_core(logical_core);
    auto address = compute_and_storage_bank.allocator_algo->allocate_at_address(start_address, size_bytes);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for circular buffer on core " + logical_core.str() + " at " + std::to_string(start_address));
    }
    return compute_and_storage_bank.offset_bytes + address.value();
}

uint32_t L1BankingAllocator::allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) {
    std::optional<uint32_t> address = std::nullopt;
    int bank_offset_bytes = 0;
    if (this->is_compute_and_storage_core(logical_core)) {
        auto &compute_and_storage_bank = this->bank_for_logical_compute_and_storage_core(logical_core);
        address = compute_and_storage_bank.allocator_algo->allocate(size_bytes, false);
        // In this case L1 buffer space cannot grow past 512 KB
        if (address.has_value() and address.value() < this->storage_core_bank_size_bytes_) {
            compute_and_storage_bank.allocator_algo->deallocate(address.value());
            TT_THROW(std::to_string(size_bytes / 1024) + " KB L1 buffer allocated at " + std::to_string(address.value() / 1024) + " grows past " + std::to_string(this->storage_core_bank_size_bytes_ / 1024) + " KB");
        }
    } else {
        TT_ASSERT(this->is_storage_only_core(logical_core));
        auto &storage_banks = this->banks_for_storage_only_cores(logical_core);
        for (auto &bank : storage_banks) {
            address = bank->allocator_algo->allocate(size_bytes, false);
            if (address.has_value()) {
                bank_offset_bytes = bank->offset_bytes;
                break;
            }
        }
    }

    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for l1 buffer on core " + logical_core.str());
    }

    return bank_offset_bytes + address.value();
}

uint32_t L1BankingAllocator::allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) {
    auto &storage_bank = this->bank_for_logical_core(logical_core, start_address);
    auto address = storage_bank.allocator_algo->allocate_at_address(start_address, size_bytes);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for l1 buffer on core " + logical_core.str() + " at " + std::to_string(start_address));
    }
    if (this->is_compute_and_storage_core(logical_core)) {
        // L1 buffers in compute and store cores cannot grow past 512 KB
        if (address.value() < this->storage_core_bank_size_bytes_) {
            storage_bank.allocator_algo->deallocate(address.value());
            TT_THROW(std::to_string(size_bytes / 1024) + " KB L1 buffer allocated at " + std::to_string(address.value() / 1024) + " grows past " + std::to_string(this->storage_core_bank_size_bytes_ / 1024) + " KB");
        }
    }
    TT_ASSERT(storage_bank.offset_bytes + address.value() == start_address);
    return start_address;
}

void L1BankingAllocator::deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address) {
    auto &bank = this->bank_for_logical_core(logical_core, address);
    auto relative_address = address - bank.offset_bytes;
    bank.allocator_algo->deallocate(relative_address);
}

uint32_t L1BankingAllocator::get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    auto start_core = logical_core_range.first;
    auto end_core = logical_core_range.second;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            auto &bank = this->bank_for_logical_compute_and_storage_core(logical_core);
            auto potential_addr_ranges = bank.allocator_algo->available_addresses(size_in_bytes);
            allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        }
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space for " + std::to_string(size_in_bytes) +
            " byte CircularBuffers in cores ranging from " + start_core.str() + " to " + end_core.str());
    }

    return allocator::find_address_of_smallest_chunk(candidate_addr_ranges);
}

void L1BankingAllocator::clear_dram() {
    for (auto &[dram_channel, dram_allocator] : this->dram_manager_) {
        dram_allocator->clear();
    }
}

void L1BankingAllocator::clear_l1() {
    for (auto &[logical_core, bank] : this->compute_and_storage_cores_l1_manager_) {
        bank->allocator_algo->clear();
    }
    for (auto &[logical_core, banks] : this->storage_cores_l1_manager_) {
        for (auto &bank : banks) {
            bank->allocator_algo->clear();
        }
    }
}

void L1BankingAllocator::clear() {
    this->clear_dram();
    this->clear_l1();
}

}  // namespace tt_metal

}  // namespace tt
