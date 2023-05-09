#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

BasicAllocator::BasicAllocator(const tt_SocDescriptor &soc_desc) : logical_grid_size_(soc_desc.worker_grid_size), Allocator() {
    constexpr static uint32_t min_allocation_size_bytes = 32;
    // DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
    constexpr static uint32_t alignment = 32;

    // Initialize DRAM manager
    for (int dram_channel = 0; dram_channel < soc_desc.get_num_dram_channels(); dram_channel++) {
        auto allocator = std::make_unique<allocator::FreeList>(
            soc_desc.dram_bank_size,
            min_allocation_size_bytes,
            alignment,
            allocator::FreeList::SearchPolicy::FIRST
        );
        this->dram_manager_.insert({dram_channel, std::move(allocator)});
    }

    for (auto worker_core : soc_desc.workers) {
        auto logical_core_x = soc_desc.routing_x_to_worker_x.at(worker_core.x);
        auto logical_core_y = soc_desc.routing_y_to_worker_y.at(worker_core.y);
        tt_xy_pair logical_core = tt_xy_pair(logical_core_x, logical_core_y);
        auto allocator = std::make_unique<allocator::FreeList>(
            soc_desc.worker_l1_size,
            min_allocation_size_bytes,
            alignment,
            allocator::FreeList::SearchPolicy::FIRST
        );
        // Space up to UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
        allocator->allocate_at_address(0, UNRESERVED_BASE);
        this->l1_manager_.insert({logical_core, std::move(allocator)});
    }
}

// BasicAllocator::BasicAllocator(const BasicAllocator &other);
// BasicAllocator& operator=(const BasicAllocator &other);

// BasicAllocator(BasicAllocator &&other);
// BasicAllocator& operator=(BasicAllocator &&other);

uint32_t BasicAllocator::num_banks(const BufferType &buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return this->dram_manager_.size();
        case BufferType::L1: return this->l1_manager_.size();
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return 0;
}

// 1:1 mapping between DRAM bank ID and DRAM channel
uint32_t BasicAllocator::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(bank_id >= 0 and bank_id <= (this->dram_manager_.size() - 1) && "Bank ID exceeds number of DRAM banks");
    return bank_id;
}

tt_xy_pair BasicAllocator::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(bank_id < this->l1_manager_.size() && "Bank ID exceeds number of L1 banks");
    uint32_t x = bank_id % this->logical_grid_size_.x;
    uint32_t y = bank_id / this->logical_grid_size_.x;
    return tt_xy_pair(x, y);
}

std::vector<uint32_t> BasicAllocator::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    TT_ASSERT(dram_channel >= 0 and dram_channel <= (this->dram_manager_.size() - 1) && "There is one bank per DRAM channel");
    return {dram_channel};
}

std::vector<uint32_t> BasicAllocator::bank_ids_from_logical_core(const tt_xy_pair &logical_core) const {
    uint32_t bank_index = logical_core.x + (this->logical_grid_size_.x * logical_core.y);
    return {bank_index};
}

allocator::Algorithm &BasicAllocator::allocator_for_dram_channel(uint32_t bank_id) const {
    int dram_channel = this->dram_channel_from_bank_id(bank_id);
    if (this->dram_manager_.find(dram_channel) == this->dram_manager_.end()) {
        TT_THROW("Allocator for DRAM channel " + std::to_string(dram_channel) + " does not exist!");
    }
    return *this->dram_manager_.at(dram_channel);
}

allocator::Algorithm &BasicAllocator::allocator_for_logical_core(uint32_t bank_id) const {
    auto logical_core = this->logical_core_from_bank_id(bank_id);
    if (this->l1_manager_.find(logical_core) == this->l1_manager_.end()) {
        TT_THROW("Allocator for logical core " + logical_core.str() + " does not exist!");
    }
    return *this->l1_manager_.at(logical_core);
}

allocator::Algorithm &BasicAllocator::get_allocator(uint32_t bank_id, const BufferType &buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return this->allocator_for_dram_channel(bank_id);
        case BufferType::L1: return this->allocator_for_logical_core(bank_id);
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return this->allocator_for_dram_channel(0);
}

std::string BasicAllocator::generate_bank_identifier_str(uint32_t bank_id, uint32_t size_bytes, const BufferType &buffer_type) const {
    std::string bank_identifier;
    switch (buffer_type) {
        case BufferType::DRAM: {
            bank_identifier = "DRAM channel ";
            bank_identifier += std::to_string(this->dram_channel_from_bank_id(bank_id));
        }
        break;
        case BufferType::L1: {
            bank_identifier = "L1 of logical core ";
            bank_identifier += this->logical_core_from_bank_id(bank_id).str();
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
    return bank_identifier;
}

BankIdToRelativeAddress BasicAllocator::allocate_contiguous_buffer(uint32_t bank_id, uint32_t size_bytes, const BufferType &buffer_type) {
    BankIdToRelativeAddress bank_to_address;
    auto address = this->get_allocator(bank_id, buffer_type).allocate(size_bytes, this->allocate_bottom_up_);
    if (not address.has_value()) {
        auto bank_identifier = this->generate_bank_identifier_str(bank_id, size_bytes, buffer_type);
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for buffer in  " + bank_identifier);
    }
    bank_to_address.insert({bank_id, {.offset_bytes = 0, .relative_address = address.value()}});
    return bank_to_address;
}

BankIdToRelativeAddress BasicAllocator::allocate_contiguous_buffer(uint32_t bank_id, uint32_t address, uint32_t size_bytes, const BufferType &buffer_type) {
    BankIdToRelativeAddress bank_to_address;
    auto allocated_address = this->get_allocator(bank_id, buffer_type).allocate_at_address(address, size_bytes);
    if (not allocated_address.has_value()) {
        auto bank_identifier = this->generate_bank_identifier_str(bank_id, size_bytes, buffer_type);
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for buffer in  " + bank_identifier + " at " + std::to_string(address));
    }
    bank_to_address.insert({bank_id, {.offset_bytes = 0, .relative_address = allocated_address.value()}});
    return bank_to_address;
}

BankIdToRelativeAddress BasicAllocator::allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, const BufferType &buffer_type) {
    TT_ASSERT(size % page_size == 0);
    uint32_t num_pages = size / page_size;
    if (num_pages == 1) {
        return this->allocate_contiguous_buffer(starting_bank_id, size, buffer_type);
    }

    int num_banks = this->num_banks(buffer_type);
    int num_equally_distributed_pages = num_pages / num_banks;
    int remaining_pages_after_equally_distributing = num_pages % num_banks;
    uint32_t total_size_bytes = num_pages * page_size;
    std::unordered_map<uint32_t, uint32_t> bank_id_to_size;
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;

    uint32_t total_accounted = 0;
    uint32_t bank_index = starting_bank_id;
    while (total_accounted < total_size_bytes) {
        int num_pages_in_bank = num_equally_distributed_pages;
        if (remaining_pages_after_equally_distributing > 0) {
            num_pages_in_bank += 1;
            remaining_pages_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_pages_in_bank * page_size;
        bank_id_to_size.emplace(bank_index, buffer_size);
        auto potential_addr_ranges = this->get_allocator(bank_index, buffer_type).available_addresses(buffer_size);
        allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        total_accounted += buffer_size;
        bank_index = (bank_index + 1) % num_banks;
    }

    if (candidate_addr_ranges.empty()) {
        std::string interleave_target = buffer_type == BufferType::DRAM ? "DRAM channels" : "L1 of cores";
        TT_THROW("Not enough space to hold interleave " + std::to_string(total_size_bytes) + " bytes across" + interleave_target);
    }

    auto address = allocator::find_address_of_smallest_chunk(candidate_addr_ranges);

    BankIdToRelativeAddress bank_to_address;
    for (auto &[bank_id, buffer_size] : bank_id_to_size) {
        this->allocate_contiguous_buffer(bank_id, address, buffer_size, buffer_type);
        bank_to_address.insert({bank_id, {.offset_bytes = 0, .relative_address = address}});
    }
    return bank_to_address;
}

BankIdToRelativeAddress BasicAllocator::allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t address, const BufferType &buffer_type) {
    TT_ASSERT(size % page_size == 0);
    uint32_t num_pages = size / page_size;
    if (num_pages == 1) {
        return this->allocate_contiguous_buffer(starting_bank_id, address, size, buffer_type);
    }
    int num_banks = this->num_banks(buffer_type);
    int num_equally_distributed_pages = num_pages / num_banks;
    int remaining_pages_after_equally_distributing = num_pages % num_banks;
    uint32_t total_size_bytes = num_pages * page_size;

    BankIdToRelativeAddress bank_to_address;
    uint32_t total_accounted = 0;
    uint32_t bank_index = starting_bank_id;
    while (total_accounted < total_size_bytes) {
        int num_pages_in_bank = num_equally_distributed_pages;
        if (remaining_pages_after_equally_distributing > 0) {
            num_pages_in_bank += 1;
            remaining_pages_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_pages_in_bank * page_size;
        this->allocate_contiguous_buffer(bank_index, address, buffer_size, buffer_type);
        bank_to_address.insert({bank_index, {.offset_bytes = 0, .relative_address = address}});

        total_accounted += buffer_size;
        bank_index = (bank_index + 1) % num_banks;
    }
    return bank_to_address;
}

void BasicAllocator::deallocate_buffer(uint32_t bank_id, uint32_t address, const BufferType &buffer_type) {
    this->get_allocator(bank_id, buffer_type).deallocate(address);
}

uint32_t BasicAllocator::allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) {
    auto bank_indices = this->bank_ids_from_logical_core(logical_core);
    TT_ASSERT(bank_indices.size() == 1);
    auto address = this->allocator_for_logical_core(bank_indices.at(0)).allocate(size_bytes, this->allocate_bottom_up_);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for circular buffer on core " + logical_core.str());
    }
    return address.value();
}

uint32_t BasicAllocator::allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) {
    auto bank_indices = this->bank_ids_from_logical_core(logical_core);
    TT_ASSERT(bank_indices.size() == 1);
    auto address = this->allocator_for_logical_core(bank_indices.at(0)).allocate_at_address(start_address, size_bytes);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for circular buffer on core " + logical_core.str() + " at " + std::to_string(start_address));
    }
    return address.value();
}

uint32_t BasicAllocator::get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    auto start_core = logical_core_range.first;
    auto end_core = logical_core_range.second;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            auto bank_indices = this->bank_ids_from_logical_core(logical_core);
            TT_ASSERT(bank_indices.size() == 1);
            auto potential_addr_ranges = this->allocator_for_logical_core(bank_indices.at(0)).available_addresses(size_in_bytes);
            allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        }
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space for " + std::to_string(size_in_bytes) +
            " byte CircularBuffers in cores ranging from " + start_core.str() + " to " + end_core.str());
    }

    return allocator::find_address_of_smallest_chunk(candidate_addr_ranges);
}

void BasicAllocator::clear_dram() {
    for (auto &[dram_channel, dram_allocator] : this->dram_manager_) {
        dram_allocator->clear();
    }
}

void BasicAllocator::clear_l1() {
    for (auto &[logical_core, l1_allocator] : this->l1_manager_) {
        l1_allocator->clear();
    }
}

void BasicAllocator::clear() {
    this->clear_dram();
    this->clear_l1();
}

}  // namespace tt_metal

}  // namespace tt
