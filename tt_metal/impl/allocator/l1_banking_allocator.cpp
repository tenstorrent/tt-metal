#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/hostdevcommon/bank_to_noc_coord_mapping.h"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

namespace allocator {

void init_compute_and_storage_l1_bank_manager(Allocator &allocator, const tt_SocDescriptor &soc_desc) {
    auto in_core_category = [](const std::vector<CoreCoord> &core_category, const CoreCoord &noc_core){
        return std::find(core_category.begin(), core_category.end(), noc_core) != core_category.end();
    };

    std::unordered_map<uint32_t, BankDescriptor> bank_id_to_descriptor;

    uint32_t compute_core_bank_size = soc_desc.worker_l1_size - UNRESERVED_BASE;

    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    static constexpr uint32_t num_banks_per_storage_core = 2;
    int expected_num_l1_banks = soc_desc.compute_and_storage_cores.size() + (num_banks_per_storage_core * soc_desc.storage_cores.size());
    uint8_t shuffled_l1_bank_ids[expected_num_l1_banks];
    init_shuffled_l1_bank_id_mapping(shuffled_l1_bank_ids);

    uint32_t bank_id = 0;
    for (uint32_t y = 0; y < soc_desc.worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < soc_desc.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord(x, y);
            uint32_t noc_x = soc_desc.worker_log_to_routing_x.at(x);
            uint32_t noc_y = soc_desc.worker_log_to_routing_y.at(y);
            CoreCoord noc_core = CoreCoord(noc_x, noc_y);
            if (in_core_category(soc_desc.compute_and_storage_cores, noc_core)) {
                uint32_t remapped_bank_id = shuffled_l1_bank_ids[bank_id];
                allocator.logical_core_to_bank_ids.insert({logical_core, {remapped_bank_id}});
                allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                bank_id_to_descriptor.insert({remapped_bank_id, {.offset_bytes = UNRESERVED_BASE, .size_bytes = compute_core_bank_size}});
                bank_id++;
            } else if (in_core_category(soc_desc.storage_cores, noc_core)) {
                std::vector<uint32_t> bank_ids;
                for (int storage_bank_index = 0; storage_bank_index < num_banks_per_storage_core; storage_bank_index++) {
                    uint32_t remapped_bank_id = shuffled_l1_bank_ids[bank_id];
                    bank_ids.push_back(remapped_bank_id);
                    allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                    uint32_t storage_core_offset = storage_bank_index * storage_core_bank_size;
                    bank_id_to_descriptor.insert({remapped_bank_id, {.offset_bytes = storage_core_offset, .size_bytes = storage_core_bank_size}});

                    bank_id++;
                }
                allocator.logical_core_to_bank_ids.insert({logical_core, bank_ids});
            }
        }
    }
    TT_ASSERT(bank_id_to_descriptor.size() == expected_num_l1_banks);
    allocator.l1_manager = BankManager(bank_id_to_descriptor);
}

bool is_compute_and_storage_bank(BankManager &bank_manager, uint32_t bank_id) {
    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    return bank_manager.size(bank_id) > storage_core_bank_size;
}

bool is_storage_only_bank(BankManager &bank_manager, uint32_t bank_id) {
    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    return bank_manager.size(bank_id) == storage_core_bank_size or bank_manager.size(bank_id) == (storage_core_bank_size - UNRESERVED_BASE);
}

void validate_l1_addresses(BankManager &bank_manager, BankIdToRelativeAddress &bank_id_to_address, bool bottom_up) {
    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    for (const auto &[bank_id, address_descriptor] : bank_id_to_address) {
        if (is_compute_and_storage_bank(bank_manager, bank_id) and not bottom_up) {
            if (address_descriptor.absolute_address() < storage_core_bank_size) {
                bank_manager.deallocate_buffer(bank_id, address_descriptor.absolute_address());
                TT_THROW("L1 buffer allocated at " + std::to_string(address_descriptor.absolute_address() / 1024) + " grows past " + std::to_string(storage_core_bank_size / 1024) + " KB");
            }
        }
    }
}

BankIdToRelativeAddress alloc_in_compute_and_storage_l1(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up) {
    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    static constexpr uint32_t num_banks_per_storage_core = 2;
    static constexpr uint32_t total_storage_size = num_banks_per_storage_core * storage_core_bank_size;

    auto adjust_address_ranges = [&](uint32_t bank_id, std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges) {
        if (bottom_up) {
            return;
        }
        uint32_t bank_offset = bank_manager.offset(bank_id);
        for (auto &addr_range : potential_addr_ranges) {
            uint32_t offset;
            auto compute_and_storage_bank = is_compute_and_storage_bank(bank_manager, bank_id);
            if (compute_and_storage_bank) {
                offset = bank_offset;
            } else {
                TT_ASSERT(is_storage_only_bank(bank_manager, bank_id));
                offset = bank_offset == (total_storage_size - storage_core_bank_size) ? bank_offset : (total_storage_size - storage_core_bank_size);
            }
            addr_range.first = addr_range.first + offset;
            addr_range.second = addr_range.second + offset;
            if (compute_and_storage_bank) { // snap up
                if (addr_range.first <= storage_core_bank_size and storage_core_bank_size <= addr_range.second) {
                    addr_range.first = storage_core_bank_size;
                }
            }
        }
    };

    auto filter_addresses = [&](const std::pair<uint32_t, uint32_t> &range){
        if (bottom_up) { return true; }
        return range.first >= storage_core_bank_size and range.second >= storage_core_bank_size;
    };

    auto get_relative_address = [&](uint32_t relative_address, uint32_t bank_id) {
        uint32_t adjusted_relative_addr = relative_address;
        uint32_t bank_offset = bank_manager.offset(bank_id);
        if (is_compute_and_storage_bank(bank_manager, bank_id)) {
            adjusted_relative_addr -= bank_offset;
        } else if (is_storage_only_bank(bank_manager, bank_id)) {
            uint32_t relative_offset = bank_offset == (total_storage_size - storage_core_bank_size) ? bank_offset : (total_storage_size - storage_core_bank_size);
            adjusted_relative_addr = relative_address - relative_offset;
        }
        return adjusted_relative_addr;
    };

    auto bank_to_address = bank_manager.allocate_buffer(starting_bank_id, size, page_size, bottom_up, adjust_address_ranges, filter_addresses, get_relative_address);
    validate_l1_addresses(bank_manager, bank_to_address, bottom_up);
    return bank_to_address;
}

BankIdToRelativeAddress alloc_at_addr_in_compute_and_storage(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address) {
    static constexpr uint32_t storage_core_bank_size = 512 * 1024;
    static constexpr uint32_t num_banks_per_storage_core = 2;

    auto adjust_address_to_allocate = [&](uint32_t address, uint32_t bank_id) {
        uint32_t adjusted_address = address;
        if (is_storage_only_bank(bank_manager, bank_id)) {
            uint32_t bank_offset = bank_manager.offset(bank_id);
            TT_ASSERT(bank_offset % storage_core_bank_size == 0);
            uint32_t index_of_bank_in_storage_core = bank_offset / storage_core_bank_size;
            uint32_t rev_index = num_banks_per_storage_core - (index_of_bank_in_storage_core + 1);
            adjusted_address = adjusted_address - (rev_index * storage_core_bank_size);
        }
        return adjusted_address;
    };

    auto bank_to_address = bank_manager.allocate_buffer_at_address(starting_bank_id, size, page_size, absolute_address, adjust_address_to_allocate);
    bool bottom_up = false; // TODO (abhullar): uplift this when CBs are treated as L1 buffers
    validate_l1_addresses(bank_manager, bank_to_address, bottom_up);
    return bank_to_address;
}

}   // namespace allocator

L1BankingAllocator::L1BankingAllocator(const tt_SocDescriptor &soc_desc)
    : Allocator(
        soc_desc,
        {
            .dram = {
                .init=allocator::init_one_bank_per_channel,
                .alloc=allocator::alloc_one_bank_per_storage_unit,
                .alloc_at_addr=allocator::alloc_at_addr_one_bank_per_storage_unit
            },
            .l1 = {
                .init=allocator::init_compute_and_storage_l1_bank_manager,
                .alloc=allocator::alloc_in_compute_and_storage_l1,
                .alloc_at_addr=allocator::alloc_at_addr_in_compute_and_storage
            }
        }
    ) {}

}  // namespace tt_metal

}  // namespace tt
