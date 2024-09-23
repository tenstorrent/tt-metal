// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"
namespace tt {

namespace tt_metal {

namespace allocator {

struct num_banks_t {
    uint32_t total;
    uint32_t num_l1_banks;
    uint32_t num_l1_small_banks;
    uint32_t per_storage_core;
};

num_banks_t compute_total_and_storage_only_num_l1_banks(const AllocatorConfig &alloc_config) {
    auto num_in_category = [](const std::unordered_map<CoreCoord, AllocCoreType> &core_allocation_types, const AllocCoreType &alloc_type){
        int num_cores = 0;
        for (const auto& core_allocation_type: core_allocation_types) {
            if (core_allocation_type.second == alloc_type) {
                num_cores++;
            }
        }
        return num_cores;
    };

    auto num_compute_and_storage_cores = num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::ComputeAndStore);
    auto num_storage_only_cores =
        num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::StorageOnly);
    uint32_t num_banks_per_storage_core = 0;
    if (num_storage_only_cores > 0) {
        TT_ASSERT(alloc_config.storage_core_bank_size.has_value());
        TT_ASSERT(alloc_config.worker_l1_size % alloc_config.storage_core_bank_size.value() == 0);
        num_banks_per_storage_core = alloc_config.worker_l1_size / alloc_config.storage_core_bank_size.value();
    }
    // L1 small region carve out is only for compute cores
    uint32_t num_l1_small_banks = (alloc_config.l1_small_size > 0) ? num_compute_and_storage_cores : 0;
    uint32_t num_l1_banks =
        num_compute_and_storage_cores + (num_banks_per_storage_core * num_storage_only_cores);
    return num_banks_t{
        .total = num_l1_banks + num_l1_small_banks,
        .num_l1_banks = num_l1_banks,
        .num_l1_small_banks = num_l1_small_banks,
        .per_storage_core = num_banks_per_storage_core,
    };
}

void init_compute_and_storage_l1_bank_manager(Allocator &allocator, const AllocatorConfig &alloc_config) {
    num_banks_t num_banks = compute_total_and_storage_only_num_l1_banks(alloc_config);

    auto logical_to_noc_coord = [&alloc_config](CoreCoord logical_core) {
        TT_ASSERT(
            alloc_config.worker_log_to_physical_routing_x.find(logical_core.x) !=
                    alloc_config.worker_log_to_physical_routing_x.end() and
                alloc_config.worker_log_to_physical_routing_y.find(logical_core.y) !=
                    alloc_config.worker_log_to_physical_routing_y.end(),
            "Cannot find log_coord=[.y={}, .x={}] in logical to routing coord lookup tables... invalid AllocatorConfig "
            "setup",
            logical_core.y,
            logical_core.x);
        CoreCoord noc_core({
            static_cast<size_t>(alloc_config.worker_log_to_physical_routing_x.at(logical_core.x)),
            static_cast<size_t>(alloc_config.worker_log_to_physical_routing_y.at(logical_core.y)),
            });
        TT_ASSERT (
            alloc_config.core_type_from_noc_coord_table.find(noc_core) != alloc_config.core_type_from_noc_coord_table.end(),
            "Cannot find noc-coord=[.y={}, .x={}] in core_type_from_noc_coord_table... invalid AllocatorConfig setup",
            noc_core.y, noc_core.x
            );
        return noc_core;
    };

    // Define the bank assignment here.
    // Only l1_banks are shuffled, l1_small_banks are just contiguous and don't
    // need remapping since they are only for compute cores
    std::vector<uint32_t> shuffled_bank_id = {};
    if (not alloc_config.l1_bank_remap.empty()) {
        TT_ASSERT(
            num_banks.num_l1_banks == alloc_config.l1_bank_remap.size(),
            "override l1_bank_remap.size()={} which is not equal to the expected expected_num_l1_banks={} from "
            "soc-desc",
            alloc_config.l1_bank_remap.size(),
            num_banks.num_l1_banks);
        std::copy(alloc_config.l1_bank_remap.begin(),alloc_config.l1_bank_remap.end(), std::back_inserter(shuffled_bank_id));
    } else {
        // randomize remap
        for (uint32_t id = 0; id < num_banks.num_l1_banks; id++) {
            shuffled_bank_id.push_back(id);
        }
        auto rng = std::default_random_engine(0);
        std::shuffle(std::begin(shuffled_bank_id), std::end(shuffled_bank_id), rng);
    }

    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset;
    // If l1_small_size exists, then it gets the top of L1 (offset 0)
    // and the regular L1 region is offset just below it
    uint32_t bank_id = 0;
    for (uint32_t y = 0; y < alloc_config.worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < alloc_config.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord(x, y);
            CoreCoord noc_core = logical_to_noc_coord(logical_core);

            if (alloc_config.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::ComputeAndStore) {
                uint32_t remapped_bank_id = shuffled_bank_id[bank_id];
                allocator.logical_core_to_bank_ids[BufferType::L1].insert({logical_core, {remapped_bank_id}});
                allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                bank_id_to_bank_offset.insert({remapped_bank_id, 0});
                bank_id++;
            } else if (alloc_config.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::StorageOnly) {
                std::vector<uint32_t> bank_ids;
                for (int storage_bank_index = 0; storage_bank_index < num_banks.per_storage_core; storage_bank_index++) {
                    uint32_t remapped_bank_id = shuffled_bank_id[bank_id];
                    bank_ids.push_back(remapped_bank_id);
                    allocator.bank_id_to_logical_core.insert({remapped_bank_id, logical_core});
                    int64_t bank_offset_bytes = 0;
                    if (alloc_config.storage_core_bank_size.value() != alloc_config.worker_l1_size) {
                        uint64_t storage_core_offset = storage_bank_index * alloc_config.storage_core_bank_size.value();
                        bank_offset_bytes = static_cast<int64_t>(storage_core_offset) - alloc_config.storage_core_bank_size.value(); // Assuming top-down here --  Not sure if this is hacky... need to specialize based off top-down cofnig flag or not?
                    } else if (num_banks.per_storage_core != 1) {
                        TT_THROW("Expected 1 bank per storage core if L1 bank size equals total worker L1 size but have {} banks", num_banks.per_storage_core);
                    }
                    bank_id_to_bank_offset.insert({remapped_bank_id, bank_offset_bytes});
                    bank_id++;
                }
                allocator.logical_core_to_bank_ids[BufferType::L1].insert({logical_core, bank_ids});
            }
        }
    }
    TT_ASSERT(bank_id == shuffled_bank_id.size());

    std::unordered_map<uint32_t, int64_t> small_bank_id_to_bank_offset;
    if (alloc_config.l1_small_size > 0) {
        TT_ASSERT(num_banks.num_l1_small_banks > 0);
        for (uint32_t y = 0; y < alloc_config.worker_grid_size.y; y++) {
            for (uint32_t x = 0; x < alloc_config.worker_grid_size.x; x++) {
                CoreCoord logical_core = CoreCoord(x, y);
                CoreCoord noc_core = logical_to_noc_coord(logical_core);

                if (alloc_config.core_type_from_noc_coord_table.at(noc_core) != AllocCoreType::ComputeAndStore) {
                    continue;
                }

                allocator.logical_core_to_bank_ids[BufferType::L1_SMALL].insert({logical_core, {bank_id}});
                allocator.bank_id_to_logical_core.insert({bank_id, logical_core});
                small_bank_id_to_bank_offset.insert({bank_id, 0});
                bank_id++;
            }
        }
    }

    TT_ASSERT(bank_id_to_bank_offset.size() == num_banks.num_l1_banks);
    TT_ASSERT(small_bank_id_to_bank_offset.size() == num_banks.num_l1_small_banks);
    TT_ASSERT(
        (bank_id_to_bank_offset.size() + small_bank_id_to_bank_offset.size()) == num_banks.total,
        "init_compute_and_storage_l1_bank_manager() -- banks setup={} must be equal to the number of banks "
        "expected={}",
        bank_id_to_bank_offset.size(),
        small_bank_id_to_bank_offset.size(),
        num_banks.total);

    // Storage only cores only need to reserve mailbox space to hold barriers
    constexpr uint32_t storage_core_unreserved_base = ((MEM_MAILBOX_BASE + ALLOCATOR_ALIGNMENT - 1) / ALLOCATOR_ALIGNMENT) * ALLOCATOR_ALIGNMENT;

    // There is only l1_bank_size bytes available for L1 buffers to be allocated in
    uint64_t l1_bank_size = alloc_config.storage_core_bank_size.has_value()
                                ? alloc_config.storage_core_bank_size.value()
                                : (alloc_config.worker_l1_size - alloc_config.l1_unreserved_base);
    uint64_t interleaved_address_limit = static_cast<uint64_t>(alloc_config.worker_l1_size - l1_bank_size) + storage_core_unreserved_base;
    uint64_t allocatable_l1_size =
        static_cast<uint64_t>(alloc_config.worker_l1_size) - alloc_config.l1_unreserved_base - alloc_config.l1_small_size;
    // Assuming top down allocation for L1 buffers so the allocatable memory space is the top l1_bank_size bytes of L1
    allocator.l1_manager = BankManager(BufferType::L1, bank_id_to_bank_offset, allocatable_l1_size, interleaved_address_limit, ALLOCATOR_ALIGNMENT, alloc_config.l1_unreserved_base);

    uint64_t small_interleaved_address_limit = alloc_config.worker_l1_size - alloc_config.l1_small_size;
    uint64_t small_alloc_offset = alloc_config.l1_unreserved_base + allocatable_l1_size;
    TT_ASSERT(
        (alloc_config.l1_unreserved_base + alloc_config.l1_small_size) <= alloc_config.worker_l1_size,
        "L1 small region extends past L1 size");
    allocator.l1_small_manager = BankManager(
        BufferType::L1_SMALL,
        small_bank_id_to_bank_offset,
        alloc_config.l1_small_size,
        small_interleaved_address_limit,
        ALLOCATOR_ALIGNMENT,
        small_alloc_offset);
}

}   // namespace allocator

L1BankingAllocator::L1BankingAllocator(const AllocatorConfig &alloc_config)
    : Allocator(
        alloc_config,
        allocator::AllocDescriptor{
            .dram = {
                .init=allocator::init_one_bank_per_channel,
                .alloc=allocator::base_alloc
            },
            .l1 = {
                .init=allocator::init_compute_and_storage_l1_bank_manager,
                .alloc=allocator::base_alloc
            }
        }
    ) {}
    L1BankingAllocator::~L1BankingAllocator() {
        Allocator::reset();
    }
}  // namespace tt_metal

}  // namespace tt
