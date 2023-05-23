#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

namespace tt {

namespace tt_metal {

namespace allocator {

BankManager::BankManager(uint32_t num_banks, uint32_t offset_bytes, uint32_t bank_size_bytes) : initialized_(true) {
    for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
        this->bank_id_to_offset_.insert({bank_id, offset_bytes});
        auto allocator = std::make_unique<FreeList>(
            bank_size_bytes,
            this->min_allocation_size_bytes_,
            this->alignment_,
            FreeList::SearchPolicy::FIRST
        );
        this->bank_id_to_allocator_.insert({bank_id, std::move(allocator)});
    }
}

BankManager::BankManager(const std::unordered_map<uint32_t, BankDescriptor> &bank_id_to_descriptor) : initialized_(true) {
    for (const auto &[bank_id, bank_descriptor] : bank_id_to_descriptor) {
        this->bank_id_to_offset_.insert({bank_id, bank_descriptor.offset_bytes});
        auto allocator = std::make_unique<FreeList>(
            bank_descriptor.size_bytes,
            this->min_allocation_size_bytes_,
            this->alignment_,
            FreeList::SearchPolicy::FIRST
        );
        this->bank_id_to_allocator_.insert({bank_id, std::move(allocator)});
    }
}

uint32_t BankManager::num_banks() const {
    TT_ASSERT(bank_id_to_offset_.size() == bank_id_to_allocator_.size());
    return bank_id_to_allocator_.size();
}

uint32_t BankManager::size(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return this->bank_id_to_allocator_.at(bank_id)->max_size_bytes();
}

uint32_t BankManager::offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return this->bank_id_to_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->bank_id_to_offset_.find(bank_id) != this->bank_id_to_offset_.end());
    TT_ASSERT(this->bank_id_to_allocator_.find(bank_id) != this->bank_id_to_allocator_.end());
}

BankIdToRelativeAddress BankManager::allocate_contiguous_buffer(uint32_t bank_id, uint32_t size_bytes, bool bottom_up) {
    BankIdToRelativeAddress bank_to_address;
    this->validate_bank_id(bank_id);
    auto bank_offset = this->bank_id_to_offset_.at(bank_id);
    auto address = this->bank_id_to_allocator_.at(bank_id)->allocate(size_bytes, bottom_up);
    if (not address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for buffer in bank " + std::to_string(bank_id));
    }
    bank_to_address.insert({bank_id, {.offset_bytes = bank_offset, .relative_address = address.value()}});
    return bank_to_address;
}

BankIdToRelativeAddress BankManager::allocate_contiguous_buffer_at_address(uint32_t bank_id, uint32_t size_bytes, uint32_t address) {
    BankIdToRelativeAddress bank_to_address;
    this->validate_bank_id(bank_id);
    auto bank_offset = this->bank_id_to_offset_.at(bank_id);
    auto bank_address = address - bank_offset;
    auto allocated_address = this->bank_id_to_allocator_.at(bank_id)->allocate_at_address(bank_address, size_bytes);
    if (not allocated_address.has_value()) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes for buffer in bank " + std::to_string(bank_id) + " at " + std::to_string(address));
    }
    TT_ASSERT(bank_offset + allocated_address.value() == address);
    bank_to_address.insert({bank_id, {.offset_bytes = bank_offset, .relative_address = allocated_address.value()}});
    return bank_to_address;
}

BankIdToRelativeAddress BankManager::allocate_buffer(
    uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up,
    std::function<void(uint32_t, std::vector<std::pair<uint32_t, uint32_t>> &)> adjust_potential_addresses,
    std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter,
    std::function<uint32_t(uint32_t, uint32_t)> adjust_relative_address) {
    TT_ASSERT(page_size > 0 and size % page_size == 0);
    uint32_t num_pages = size / page_size;
    if (num_pages == 1) {
        return this->allocate_contiguous_buffer(starting_bank_id, size, bottom_up);
    }

    uint32_t num_banks = this->num_banks();
    int num_equally_distributed_pages = num_pages / num_banks;
    int remaining_pages_after_equally_distributing = num_pages % num_banks;
    uint32_t total_size_bytes = num_pages * page_size;

    std::unordered_map<uint32_t, uint32_t> bank_id_to_size;
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;

    uint32_t total_accounted = 0;
    uint32_t bank_id = starting_bank_id;
    while (total_accounted < total_size_bytes) {
        this->validate_bank_id(bank_id);
        int num_pages_in_bank = num_equally_distributed_pages;
        if (remaining_pages_after_equally_distributing > 0) {
            num_pages_in_bank += 1;
            remaining_pages_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_pages_in_bank * page_size;
        bank_id_to_size.emplace(bank_id, buffer_size);
        auto potential_addr_ranges = this->bank_id_to_allocator_.at(bank_id)->available_addresses(buffer_size);
        adjust_potential_addresses(bank_id, potential_addr_ranges);
        allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges, filter);
        total_accounted += buffer_size;
        bank_id = (bank_id + 1) % num_banks;
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space to interleave " + std::to_string(total_size_bytes) + " bytes across" + std::to_string(num_banks) + " banks");
    }

    uint32_t relative_address;
    if (bottom_up) {
        relative_address = allocator::find_address_of_smallest_chunk(candidate_addr_ranges);
    } else {
        relative_address = allocator::find_max_address(candidate_addr_ranges);
    }

    BankIdToRelativeAddress bank_to_address;
    for (auto &[bank_id, buffer_size] : bank_id_to_size) {
        uint32_t bank_offset = this->bank_id_to_offset_.at(bank_id);
        uint32_t adjusted_relative_address = adjust_relative_address(relative_address, bank_id);
        uint32_t absolute_address = adjusted_relative_address + bank_offset;
        this->allocate_contiguous_buffer_at_address(bank_id, buffer_size, absolute_address);
        bank_to_address.insert({bank_id, {.offset_bytes = bank_offset, .relative_address = adjusted_relative_address}});
    }
    return bank_to_address;
}

BankIdToRelativeAddress BankManager::allocate_buffer_at_address(
    uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address,
    std::function<uint32_t(uint32_t, uint32_t)> adjust_absolute_address) {
    TT_ASSERT(page_size > 0 and size % page_size == 0);

    uint32_t num_pages = size / page_size;
    if (num_pages == 1) {
        return this->allocate_contiguous_buffer_at_address(starting_bank_id, size, absolute_address);
    }

    uint32_t num_banks = this->num_banks();
    int num_equally_distributed_pages = num_pages / num_banks;
    int remaining_pages_after_equally_distributing = num_pages % num_banks;
    uint32_t total_size_bytes = num_pages * page_size;

    BankIdToRelativeAddress bank_to_address;
    uint32_t total_accounted = 0;
    uint32_t bank_id = starting_bank_id;
    while (total_accounted < total_size_bytes) {
        this->validate_bank_id(bank_id);
        int num_pages_in_bank = num_equally_distributed_pages;
        if (remaining_pages_after_equally_distributing > 0) {
            num_pages_in_bank += 1;
            remaining_pages_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_pages_in_bank * page_size;
        uint32_t adjusted_absolute_address = adjust_absolute_address(absolute_address, bank_id);
        this->allocate_contiguous_buffer_at_address(bank_id, buffer_size, adjusted_absolute_address);
        uint32_t bank_offset = this->bank_id_to_offset_.at(bank_id);
        uint32_t relative_address = adjusted_absolute_address - bank_offset;
        bank_to_address.insert({bank_id, {.offset_bytes = bank_offset, .relative_address = relative_address}});

        total_accounted += buffer_size;
        bank_id = (bank_id + 1) % num_banks;
    }
    return bank_to_address;
}

void BankManager::deallocate_buffer(uint32_t bank_id, uint32_t absolute_address) {
    this->validate_bank_id(bank_id);
    auto bank_offset = this->bank_id_to_offset_.at(bank_id);
    auto relative_address = absolute_address - bank_offset;
    this->bank_id_to_allocator_.at(bank_id)->deallocate(relative_address);
}

std::vector<std::pair<uint32_t, uint32_t>> BankManager::available_addresses(uint32_t bank_id, uint32_t size_bytes, bool return_absolute_addresses) const {
    this->validate_bank_id(bank_id);
    auto addresses = this->bank_id_to_allocator_.at(bank_id)->available_addresses(size_bytes);
    auto offset = this->bank_id_to_offset_.at(bank_id);
    if (return_absolute_addresses) {
        for (auto &[start_addr, end_addr] : addresses) {
            start_addr += offset;
            end_addr += offset;
        }
    }
    return addresses;
}

void BankManager::clear() {
    for (auto &[bank_id, allocator] : this->bank_id_to_allocator_) {
        allocator->clear();
    }
}

uint32_t find_max_address(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges) {
    uint32_t max_address = candidate_addr_ranges[0].second;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        max_address = std::max(max_address, candidate_addr_range.second);
    }
    return max_address;
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
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges,
    std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter) {
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
            std::pair<uint32_t, uint32_t> address_range = {lower_addr, upper_addr};
            if (filter(address_range)) {
                intersecting_addr_ranges.push_back(address_range);
            }
        }
        if (candidate_addr_ranges[i].second < potential_addr_ranges[j].second) {
            i++;
        } else {
            j++;
        }
    }
    candidate_addr_ranges = intersecting_addr_ranges;
}

void init_one_bank_per_channel(Allocator &allocator, const tt_SocDescriptor &soc_desc) {
    uint32_t bank_offset = 0;
    allocator.dram_manager = BankManager(soc_desc.get_num_dram_channels(), bank_offset, soc_desc.dram_bank_size);
    for (uint32_t bank_id = 0; bank_id < soc_desc.get_num_dram_channels(); bank_id++) {
        allocator.bank_id_to_dram_channel.insert({bank_id, bank_id});
        allocator.dram_channel_to_bank_ids.insert({bank_id, {bank_id}});
    }
}

void init_one_bank_per_l1(Allocator &allocator, const tt_SocDescriptor &soc_desc) {
    uint32_t num_l1_banks = soc_desc.worker_grid_size.y * soc_desc.worker_grid_size.x;
    // Space up to UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
    uint32_t offset_bytes = UNRESERVED_BASE;
    uint32_t l1_bank_size = soc_desc.worker_l1_size - UNRESERVED_BASE;
    allocator.l1_manager = BankManager(num_l1_banks, offset_bytes, l1_bank_size);

    uint32_t bank_id = 0;
    for (uint32_t y = 0; y < soc_desc.worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < soc_desc.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord(x, y);
            allocator.bank_id_to_logical_core.insert({bank_id, logical_core});
            allocator.logical_core_to_bank_ids.insert({logical_core, {bank_id}});
            bank_id++;
        }
    }
}

uint32_t num_banks(const Allocator &allocator, const BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.dram_manager.num_banks();
        case BufferType::L1: return allocator.l1_manager.num_banks();
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return 0;
}

uint32_t dram_channel_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    TT_ASSERT(allocator.bank_id_to_dram_channel.find(bank_id) != allocator.bank_id_to_dram_channel.end());
    return allocator.bank_id_to_dram_channel.at(bank_id);
}

CoreCoord logical_core_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    TT_ASSERT(allocator.bank_id_to_logical_core.find(bank_id) != allocator.bank_id_to_logical_core.end());
    return allocator.bank_id_to_logical_core.at(bank_id);
}

std::vector<uint32_t> bank_ids_from_dram_channel(const Allocator &allocator, uint32_t dram_channel) {
    TT_ASSERT(allocator.dram_channel_to_bank_ids.find(dram_channel) != allocator.dram_channel_to_bank_ids.end());
    return allocator.dram_channel_to_bank_ids.at(dram_channel);
}

std::vector<uint32_t> bank_ids_from_logical_core(const Allocator &allocator, const CoreCoord &logical_core) {
    TT_ASSERT(allocator.logical_core_to_bank_ids.find(logical_core) != allocator.logical_core_to_bank_ids.end());
    return allocator.logical_core_to_bank_ids.at(logical_core);
}

BankIdToRelativeAddress alloc_one_bank_per_storage_unit(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up) {
    return bank_manager.allocate_buffer(starting_bank_id, size, page_size, bottom_up);
}

BankIdToRelativeAddress alloc_at_addr_one_bank_per_storage_unit(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address) {
    return bank_manager.allocate_buffer_at_address(starting_bank_id, size, page_size, absolute_address);
}

BankIdToRelativeAddress allocate_buffer(Allocator &allocator, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, const BufferType &buffer_type, bool bottom_up) {
    BankIdToRelativeAddress bank_to_address;
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.descriptor.dram.alloc(allocator.dram_manager, starting_bank_id, size, page_size, bottom_up);
        case BufferType::L1: return allocator.descriptor.l1.alloc(allocator.l1_manager, starting_bank_id, size, page_size, bottom_up);
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return bank_to_address;
}

BankIdToRelativeAddress allocate_buffer_at_address(Allocator &allocator, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address, const BufferType &buffer_type) {
    BankIdToRelativeAddress bank_to_address;
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.descriptor.dram.alloc_at_addr(allocator.dram_manager, starting_bank_id, size, page_size, absolute_address);
        case BufferType::L1: return allocator.descriptor.l1.alloc_at_addr(allocator.l1_manager, starting_bank_id, size, page_size, absolute_address);
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return bank_to_address;
}

void deallocate_buffer(Allocator &allocator, uint32_t bank_id, uint32_t absolute_address, const BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM:
            allocator.dram_manager.deallocate_buffer(bank_id, absolute_address);
        break;
        case BufferType::L1:
            allocator.l1_manager.deallocate_buffer(bank_id, absolute_address);
        break;
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
}

void clear(Allocator &allocator) {
    allocator.dram_manager.clear();
    allocator.l1_manager.clear();
}

uint32_t allocate_circular_buffer(Allocator &allocator, const CoreCoord &logical_core, uint32_t size_bytes) {
    auto bank_indices = bank_ids_from_logical_core(allocator, logical_core);
    TT_ASSERT(bank_indices.size() == 1);
    auto bank_id = bank_indices.at(0);
    static constexpr bool bottom_up = true;
    auto bank_id_to_rel_address = allocator.l1_manager.allocate_buffer(bank_id, size_bytes, size_bytes, bottom_up);
    TT_ASSERT(bank_id_to_rel_address.find(bank_id) != bank_id_to_rel_address.end());
    auto rel_address_desc = bank_id_to_rel_address.at(bank_id);
    // Return absolute address for circular buffers
    return rel_address_desc.absolute_address();
}

uint32_t allocate_circular_buffer(Allocator &allocator, const CoreCoord &logical_core, uint32_t start_address, uint32_t size_bytes) {
    auto bank_indices = bank_ids_from_logical_core(allocator, logical_core);
    TT_ASSERT(bank_indices.size() == 1);
    auto bank_id = bank_indices.at(0);
    auto bank_id_to_rel_address = allocator.l1_manager.allocate_buffer_at_address(bank_id, size_bytes, size_bytes, start_address);
    TT_ASSERT(bank_id_to_rel_address.find(bank_id) != bank_id_to_rel_address.end());
    auto rel_address_desc = bank_id_to_rel_address.at(bank_id);
    // Return absolute address for circular buffers
    return rel_address_desc.absolute_address();
}

uint32_t get_address_for_circular_buffers_across_core_range(Allocator &allocator, const std::pair<CoreCoord, CoreCoord> &logical_core_range, uint32_t size_in_bytes) {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    bool get_absolute_addresses = true;
    auto start_core = logical_core_range.first;
    auto end_core = logical_core_range.second;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = CoreCoord(x, y);
            auto bank_indices = bank_ids_from_logical_core(allocator, logical_core);
            TT_ASSERT(bank_indices.size() == 1);
            auto bank_id = bank_indices.at(0);
            auto potential_addr_ranges = allocator.l1_manager.available_addresses(bank_id, size_in_bytes, get_absolute_addresses);
            allocator::populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        }
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space for " + std::to_string(size_in_bytes) +
            " byte CircularBuffers in cores ranging from " + start_core.str() + " to " + end_core.str());
    }

    return allocator::find_address_of_smallest_chunk(candidate_addr_ranges);
}

}  // namespace allocator

Allocator::Allocator(const tt_SocDescriptor &soc_desc, const allocator::AllocDescriptor &alloc_descriptor) : descriptor(alloc_descriptor) {
    // TODO: add validation for allocator_descriptor?
    this->descriptor.dram.init(*this, soc_desc);
    this->descriptor.l1.init(*this, soc_desc);
    // assert that bank managers have been initialized?
    TT_ASSERT(not bank_id_to_dram_channel.empty() and not dram_channel_to_bank_ids.empty());
    TT_ASSERT(not bank_id_to_logical_core.empty() and not bank_id_to_logical_core.empty());
}

}  // namespace tt_metal

}  // namespace tt
