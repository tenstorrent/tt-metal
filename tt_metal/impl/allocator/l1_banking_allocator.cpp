#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/hostdevcommon/bank_to_noc_coord_mapping.h"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

namespace allocator {

void init_compute_and_storage_l1_bank_manager(Allocator &allocator, const AllocatorConfig &alloc_config) {
    auto num_in_category = [](const std::unordered_map<CoreCoord, AllocCoreType> &core_allocation_types, const AllocCoreType &alloc_type){
        int num_cores = 0;
        for (const auto& core_allocation_type: core_allocation_types) {
            if (core_allocation_type.second == alloc_type) {
                num_cores++;
            }
        }
        return num_cores;
    };

    std::unordered_map<uint32_t, BankDescriptor> bank_id_to_descriptor;

    uint32_t compute_core_bank_size = alloc_config.worker_l1_size - UNRESERVED_BASE;

    TT_ASSERT(alloc_config.worker_l1_size % alloc_config.storage_core_l1_bank_size == 0);
    uint32_t num_banks_per_storage_core = alloc_config.worker_l1_size / alloc_config.storage_core_l1_bank_size;
    int expected_num_l1_banks =
        num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::ComputeAndStore) +
        (num_banks_per_storage_core * num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::StorageOnly));
    uint8_t shuffled_l1_bank_ids[expected_num_l1_banks];
    init_shuffled_l1_bank_id_mapping(shuffled_l1_bank_ids);

    uint32_t bank_id = 0;
    for (uint32_t y = 0; y < alloc_config.worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < alloc_config.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord(x, y);
            log_assert (
                alloc_config.logical_to_routing_coord_lookup_table.find(logical_core) != alloc_config.logical_to_routing_coord_lookup_table.end(),
                "Cannot find log_coord=[.y={}, .x={}] in logical_to_routing_coord_lookup_table... invalid AllocatorConfig setup",
                logical_core.y, logical_core.x
            );
            CoreCoord noc_core = alloc_config.logical_to_routing_coord_lookup_table.at(logical_core);
            log_assert (
                alloc_config.core_type_from_noc_coord_table.find(noc_core) != alloc_config.core_type_from_noc_coord_table.end(),
                "Cannot find noc-coord=[.y={}, .x={}] in core_type_from_noc_coord_table... invalid AllocatorConfig setup",
                noc_core.y, noc_core.x
            );
            if (alloc_config.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::ComputeAndStore) {
                uint32_t remapped_bank_id = shuffled_l1_bank_ids[bank_id];
                allocator.logical_core_to_bank_ids.insert({logical_core, {remapped_bank_id}});
                allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                bank_id_to_descriptor.insert({remapped_bank_id, {.offset_bytes = UNRESERVED_BASE, .size_bytes = compute_core_bank_size}});
                bank_id++;
            } else if (alloc_config.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::StorageOnly) {
                std::vector<uint32_t> bank_ids;
                for (int storage_bank_index = 0; storage_bank_index < num_banks_per_storage_core; storage_bank_index++) {
                    uint32_t remapped_bank_id = shuffled_l1_bank_ids[bank_id];
                    bank_ids.push_back(remapped_bank_id);
                    allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                    uint32_t storage_core_offset = storage_bank_index * alloc_config.storage_core_l1_bank_size;
                    bank_id_to_descriptor.insert({remapped_bank_id, {.offset_bytes = storage_core_offset, .size_bytes = (uint32_t)alloc_config.storage_core_l1_bank_size}});

                    bank_id++;
                }
                allocator.logical_core_to_bank_ids.insert({logical_core, bank_ids});
            }
        }
    }

    log_assert(
        bank_id_to_descriptor.size() == expected_num_l1_banks,
        "init_compute_and_storage_l1_bank_manager() -- banks setup={} must be equal to the number of bankes expected={}",
        bank_id_to_descriptor.size(),
        expected_num_l1_banks
    );
    allocator.l1_manager = BankManager(bank_id_to_descriptor);
}

bool is_compute_and_storage_bank(BankManager &bank_manager, size_t storage_core_bank_size, uint32_t bank_id) {
    return bank_manager.size(bank_id) > storage_core_bank_size;
}

bool is_storage_only_bank(BankManager &bank_manager, size_t storage_core_bank_size, uint32_t bank_id) {
    return bank_manager.size(bank_id) == storage_core_bank_size or bank_manager.size(bank_id) == (storage_core_bank_size - UNRESERVED_BASE);
}

void validate_l1_addresses(BankManager &bank_manager, size_t storage_core_bank_size, BankIdToRelativeAddress &bank_id_to_address, bool bottom_up) {
    for (const auto &[bank_id, address_descriptor] : bank_id_to_address) {
        if (is_compute_and_storage_bank(bank_manager, storage_core_bank_size, bank_id) and not bottom_up) {
            if (address_descriptor.absolute_address() < storage_core_bank_size) {
                bank_manager.deallocate_buffer(bank_id, address_descriptor.absolute_address());
                TT_THROW("L1 buffer allocated at " + std::to_string(address_descriptor.absolute_address() / 1024) + " grows past " + std::to_string(storage_core_bank_size / 1024) + " KB");
            }
        }
    }
}

BankIdToRelativeAddress alloc_in_compute_and_storage_l1(const AllocatorConfig &config, BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up) {
    auto adjust_address_ranges = [&](uint32_t bank_id, std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges) {
        if (bottom_up) {
            return;
        }
        uint32_t bank_offset = bank_manager.offset(bank_id);
        for (auto &addr_range : potential_addr_ranges) {
            uint32_t offset;
            auto compute_and_storage_bank = is_compute_and_storage_bank(bank_manager, config.storage_core_l1_bank_size, bank_id);
            if (compute_and_storage_bank) {
                offset = bank_offset;
            } else {
                TT_ASSERT(is_storage_only_bank(bank_manager, config.storage_core_l1_bank_size, bank_id));
                offset = bank_offset == (config.worker_l1_size - config.storage_core_l1_bank_size) ? bank_offset : (config.worker_l1_size - config.storage_core_l1_bank_size);
            }
            addr_range.first = addr_range.first + offset;
            addr_range.second = addr_range.second + offset;
            if (compute_and_storage_bank) { // snap up
                if (addr_range.first <= config.storage_core_l1_bank_size and config.storage_core_l1_bank_size <= addr_range.second) {
                    addr_range.first = config.storage_core_l1_bank_size;
                }
            }
        }
    };

    auto filter_addresses = [&](const std::pair<uint32_t, uint32_t> &range){
        if (bottom_up) { return true; }
        return range.first >= config.storage_core_l1_bank_size and range.second >= config.storage_core_l1_bank_size;
    };

    auto get_relative_address = [&](uint32_t relative_address, uint32_t bank_id) {
        uint32_t adjusted_relative_addr = relative_address;
        uint32_t bank_offset = bank_manager.offset(bank_id);
        if (is_compute_and_storage_bank(bank_manager, config.storage_core_l1_bank_size, bank_id)) {
            adjusted_relative_addr -= bank_offset;
        } else if (is_storage_only_bank(bank_manager, config.storage_core_l1_bank_size, bank_id)) {
            uint32_t relative_offset = bank_offset == (config.worker_l1_size - config.storage_core_l1_bank_size) ? bank_offset : (config.worker_l1_size - config.storage_core_l1_bank_size);
            adjusted_relative_addr = relative_address - relative_offset;
        }
        return adjusted_relative_addr;
    };

    auto bank_to_address = bank_manager.allocate_buffer(starting_bank_id, size, page_size, bottom_up, adjust_address_ranges, filter_addresses, get_relative_address);
    validate_l1_addresses(bank_manager, config.storage_core_l1_bank_size, bank_to_address, bottom_up);
    return bank_to_address;
}

BankIdToRelativeAddress alloc_at_addr_in_compute_and_storage(const AllocatorConfig &config, BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address) {
    TT_ASSERT(config.worker_l1_size % config.storage_core_l1_bank_size == 0);
    uint32_t num_banks_per_storage_core = config.worker_l1_size / config.storage_core_l1_bank_size;

    auto adjust_address_to_allocate = [&](uint32_t address, uint32_t bank_id) {
        uint32_t adjusted_address = address;
        if (is_storage_only_bank(bank_manager, config.storage_core_l1_bank_size, bank_id)) {
            uint32_t bank_offset = bank_manager.offset(bank_id);
            TT_ASSERT(bank_offset % config.storage_core_l1_bank_size == 0);
            uint32_t index_of_bank_in_storage_core = bank_offset / config.storage_core_l1_bank_size;
            uint32_t rev_index = num_banks_per_storage_core - (index_of_bank_in_storage_core + 1);
            adjusted_address = adjusted_address - (rev_index * config.storage_core_l1_bank_size);
        }
        return adjusted_address;
    };

    auto bank_to_address = bank_manager.allocate_buffer_at_address(starting_bank_id, size, page_size, absolute_address, adjust_address_to_allocate);
    bool bottom_up = false; // TODO (abhullar): uplift this when CBs are treated as L1 buffers
    validate_l1_addresses(bank_manager, config.storage_core_l1_bank_size, bank_to_address, bottom_up);
    return bank_to_address;
}

}   // namespace allocator

L1BankingAllocator::L1BankingAllocator(const AllocatorConfig &alloc_config)
    : Allocator(
        alloc_config,
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
