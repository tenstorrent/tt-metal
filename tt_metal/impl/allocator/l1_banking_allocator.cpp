#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
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
    std::unordered_map<u32, i64> bank_id_to_bank_offset;

    TT_ASSERT(alloc_config.worker_l1_size % alloc_config.storage_core_l1_bank_size == 0);

    u32 num_banks_per_storage_core = alloc_config.worker_l1_size / alloc_config.storage_core_l1_bank_size;
    int expected_num_l1_banks =
        num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::ComputeAndStore) +
        (num_banks_per_storage_core * num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::StorageOnly));

    // Define the bank assignment here.
    std::vector<u32> shuffled_bank_id = {};
    if (not alloc_config.l1_bank_remap.empty()) {
        log_assert(
            expected_num_l1_banks == alloc_config.l1_bank_remap.size(),
            "override l1_bank_remap.size()={} which is not equal to the expected expected_num_l1_banks={} from soc-desc",
            alloc_config.l1_bank_remap.size(), expected_num_l1_banks
        );
        std::copy(alloc_config.l1_bank_remap.begin(),alloc_config.l1_bank_remap.end(), std::back_inserter(shuffled_bank_id));
    } else {
        // randomize remap
        for (u32 id = 0; id < expected_num_l1_banks; id++) {
            shuffled_bank_id.push_back(id);
        }
        auto rng = std::default_random_engine(0);
        std::shuffle(std::begin(shuffled_bank_id), std::end(shuffled_bank_id), rng);
    }

    u32 bank_id = 0;
    for (u32 y = 0; y < alloc_config.worker_grid_size.y; y++) {
        for (u32 x = 0; x < alloc_config.worker_grid_size.x; x++) {
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
                u32 remapped_bank_id = shuffled_bank_id[bank_id];
                allocator.logical_core_to_bank_ids.insert({logical_core, {remapped_bank_id}});
                allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                bank_id_to_bank_offset.insert({remapped_bank_id, 0});
                bank_id++;
            } else if (alloc_config.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::StorageOnly) {
                std::vector<u32> bank_ids;
                for (int storage_bank_index = 0; storage_bank_index < num_banks_per_storage_core; storage_bank_index++) {
                    u32 remapped_bank_id = shuffled_bank_id[bank_id];
                    bank_ids.push_back(remapped_bank_id);
                    allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                    u64 storage_core_offset = storage_bank_index * alloc_config.storage_core_l1_bank_size;
                    i64 bank_offset_bytes = static_cast<i64>(storage_core_offset) - alloc_config.storage_core_l1_bank_size; // Assuming top-down here --  Not sure if this is hacky... need to specialize based off top-down cofnig flag or not?
                    bank_id_to_bank_offset.insert({remapped_bank_id, bank_offset_bytes});
                    bank_id++;
                }
                allocator.logical_core_to_bank_ids.insert({logical_core, bank_ids});
            }
        }
    }

    log_assert(
        bank_id_to_bank_offset.size() == expected_num_l1_banks,
        "init_compute_and_storage_l1_bank_manager() -- banks setup={} must be equal to the number of bankes expected={}",
        bank_id_to_bank_offset.size(),
        expected_num_l1_banks
    );

    // There is only alloc_config.storage_core_l1_bank_size bytes available for L1 buffers to be allocated in
    u64 allocatable_l1_size = static_cast<u64>(alloc_config.storage_core_l1_bank_size);
    // Assuming top down allocation for L1 buffers so the allocatable memory space is the top alloc_config.storage_core_l1_bank_size bytes of L1
    u64 alloc_offset = static_cast<u64>(alloc_config.worker_l1_size) - allocatable_l1_size;
    allocator.l1_manager = BankManager(bank_id_to_bank_offset, allocatable_l1_size, alloc_offset);
}

u64 alloc_at_addr_in_compute_and_storage(const AllocatorConfig &config, BankManager &bank_manager, u64 size, u64 page_size, u64 relative_address) {
    u64 allocatable_l1_size = static_cast<u64>(config.storage_core_l1_bank_size);
    u64 alloc_offset = static_cast<u64>(config.worker_l1_size - config.storage_core_l1_bank_size);

    auto adjust_address = [&](u32 rel_addr){
        auto starting_bank_offset = bank_manager.bank_offset(0);
        auto offset_to_cancel_starting_bank_offset = 0 - starting_bank_offset;
        auto adjusted_addr = relative_address + offset_to_cancel_starting_bank_offset;

        log_assert(adjusted_addr >= alloc_offset, "Invalid address specified: L1 buffers cannot grow past {}, specified address {} does not meet this criteria!", alloc_offset, relative_address);
        return adjusted_addr;
    };

    return bank_manager.allocate_buffer_at_address(size, page_size, relative_address, adjust_address);
}

}   // namespace allocator

L1BankingAllocator::L1BankingAllocator(const AllocatorConfig &alloc_config)
    : Allocator(
        alloc_config,
        allocator::AllocDescriptor{
            .dram = {
                .init=allocator::init_one_bank_per_channel,
                .alloc=allocator::base_alloc,
                .alloc_at_addr=allocator::base_alloc_at_addr
            },
            .l1 = {
                .init=allocator::init_compute_and_storage_l1_bank_manager,
                .alloc=allocator::base_alloc,
                .alloc_at_addr=allocator::alloc_at_addr_in_compute_and_storage
            }
        }
    ) {}

}  // namespace tt_metal

}  // namespace tt
