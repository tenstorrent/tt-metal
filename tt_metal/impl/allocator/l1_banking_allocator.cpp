// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "l1_banking_allocator.hpp"

#include <allocator.hpp>
#include <allocator_types.hpp>
#include <assert.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "bank_manager.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.h>

namespace tt {

namespace tt_metal {

struct num_banks_t {
    uint32_t total;
    uint32_t num_l1_banks;
    uint32_t num_l1_small_banks;
    uint32_t per_storage_core;
};

num_banks_t compute_total_and_storage_only_num_l1_banks(const AllocatorConfig& alloc_config) {
    auto num_in_category = [](const std::unordered_map<CoreCoord, AllocCoreType>& core_allocation_types,
                              const AllocCoreType& alloc_type) {
        int num_cores = 0;
        for (const auto& core_allocation_type : core_allocation_types) {
            if (core_allocation_type.second == alloc_type) {
                num_cores++;
            }
        }
        return num_cores;
    };

    auto num_compute_and_storage_cores =
        num_in_category(alloc_config.core_type_from_noc_coord_table, AllocCoreType::ComputeAndStore);
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
    uint32_t num_l1_banks = num_compute_and_storage_cores + (num_banks_per_storage_core * num_storage_only_cores);
    return num_banks_t{
        .total = num_l1_banks + num_l1_small_banks,
        .num_l1_banks = num_l1_banks,
        .num_l1_small_banks = num_l1_small_banks,
        .per_storage_core = num_banks_per_storage_core,
    };
}

void Allocator::init_compute_and_storage_l1_bank_manager() {
    TT_FATAL(config_.worker_grid.contains(config_.compute_grid), "Compute grid must be a subset of worker grid");
    num_banks_t num_banks = compute_total_and_storage_only_num_l1_banks(config_);
    auto logical_to_noc_coord = [this](CoreCoord logical_core) {
        TT_ASSERT(
            config_.worker_log_to_virtual_routing_x.find(logical_core.x) !=
                    config_.worker_log_to_virtual_routing_x.end() and
                config_.worker_log_to_virtual_routing_y.find(logical_core.y) !=
                    config_.worker_log_to_virtual_routing_y.end(),
            "Cannot find log_coord=[.y={}, .x={}] in logical to routing coord lookup tables... invalid AllocatorConfig "
            "setup",
            logical_core.y,
            logical_core.x);
        CoreCoord noc_core({
            static_cast<std::size_t>(config_.worker_log_to_virtual_routing_x.at(logical_core.x)),
            static_cast<std::size_t>(config_.worker_log_to_virtual_routing_y.at(logical_core.y)),
        });
        TT_ASSERT(
            config_.core_type_from_noc_coord_table.find(noc_core) != config_.core_type_from_noc_coord_table.end(),
            "Cannot find noc-coord=[.y={}, .x={}] in core_type_from_noc_coord_table... invalid AllocatorConfig setup",
            noc_core.y,
            noc_core.x);
        return noc_core;
    };

    // Define the bank assignment here.
    std::vector<uint32_t> shuffled_bank_id = {};
    if (not config_.l1_bank_remap.empty()) {
        TT_ASSERT(
            num_banks.num_l1_banks == config_.l1_bank_remap.size(),
            "override l1_bank_remap.size()={} which is not equal to the expected expected_num_l1_banks={} from "
            "soc-desc",
            config_.l1_bank_remap.size(),
            num_banks.num_l1_banks);
        std::copy(config_.l1_bank_remap.begin(), config_.l1_bank_remap.end(), std::back_inserter(shuffled_bank_id));
    } else {
        // randomize remap
        for (uint32_t id = 0; id < num_banks.num_l1_banks; id++) {
            shuffled_bank_id.push_back(id);
        }
        auto rng = std::default_random_engine(0);
        std::shuffle(std::begin(shuffled_bank_id), std::end(shuffled_bank_id), rng);
    }

    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset;
    std::unordered_map<uint32_t, int64_t> small_bank_id_to_bank_offset;
    if (config_.l1_small_size > 0) {
        TT_ASSERT(num_banks.num_l1_small_banks > 0);
    }

    // If l1_small_size exists, then it gets the top of L1 (offset 0)
    // and the regular L1 region is offset just below it
    uint32_t bank_id = 0;
    const auto& cores = corerange_to_cores(config_.worker_grid, std::nullopt, true);
    for (const auto& logical_core : cores) {
        CoreCoord noc_core = logical_to_noc_coord(logical_core);

        if (config_.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::ComputeAndStore) {
            uint32_t remapped_bank_id = shuffled_bank_id[bank_id];
            logical_core_to_bank_ids_[BufferType::L1].insert({logical_core, {remapped_bank_id}});
            bank_id_to_logical_core_.insert({remapped_bank_id, logical_core});
            bank_id_to_bank_offset.insert({remapped_bank_id, 0});
            if (config_.l1_small_size > 0) {
                // Now map the L1-small bank id
                logical_core_to_bank_ids_[BufferType::L1_SMALL].insert({logical_core, {remapped_bank_id}});
                small_bank_id_to_bank_offset.insert({remapped_bank_id, 0});
            }
            bank_id++;
        } else if (config_.core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::StorageOnly) {
            std::vector<uint32_t> bank_ids;
            for (int storage_bank_index = 0; storage_bank_index < num_banks.per_storage_core; storage_bank_index++) {
                uint32_t remapped_bank_id = shuffled_bank_id[bank_id];
                bank_ids.push_back(remapped_bank_id);
                bank_id_to_logical_core_.insert({remapped_bank_id, logical_core});
                int64_t bank_offset_bytes = 0;
                if (config_.storage_core_bank_size.value() != config_.worker_l1_size) {
                    uint64_t storage_core_offset = storage_bank_index * config_.storage_core_bank_size.value();
                    bank_offset_bytes = static_cast<int64_t>(storage_core_offset) -
                                        config_.storage_core_bank_size
                                            .value();  // Assuming top-down here --  Not sure if this is hacky... need
                                                       // to specialize based off top-down cofnig flag or not?
                } else if (num_banks.per_storage_core != 1) {
                    TT_THROW(
                        "Expected 1 bank per storage core if L1 bank size equals total worker L1 size but have {} "
                        "banks",
                        num_banks.per_storage_core);
                }
                bank_id_to_bank_offset.insert({remapped_bank_id, bank_offset_bytes});
                bank_id++;
            }
            logical_core_to_bank_ids_[BufferType::L1].insert({logical_core, bank_ids});
        }
    }
    TT_ASSERT(bank_id == shuffled_bank_id.size());
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
    uint32_t mem_mailbox_base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX);
    uint32_t storage_core_unreserved_base =
        ((mem_mailbox_base + config_.l1_alignment - 1) / config_.l1_alignment) * config_.l1_alignment;

    // There is only l1_bank_size bytes available for L1 buffers to be allocated in
    uint64_t l1_bank_size = config_.storage_core_bank_size.has_value()
                                ? config_.storage_core_bank_size.value()
                                : (config_.worker_l1_size - config_.l1_unreserved_base);
    uint64_t interleaved_address_limit =
        static_cast<uint64_t>(config_.worker_l1_size - l1_bank_size) + storage_core_unreserved_base;
    uint64_t allocatable_l1_size =
        static_cast<uint64_t>(config_.worker_l1_size) - config_.l1_unreserved_base - config_.l1_small_size;
    // Assuming top down allocation for L1 buffers so the allocatable memory space is the top l1_bank_size bytes of L1
    l1_manager_ = std::make_unique<BankManager>(
        BufferType::L1,
        bank_id_to_bank_offset,
        allocatable_l1_size,
        interleaved_address_limit,
        config_.l1_alignment,
        config_.l1_unreserved_base,
        config_.disable_interleaved);
    log_debug(
        tt::LogMetal,
        "Configured partition params: mem_mailbox_base:0x{:X}, storage_core_bank_size:0x{:X}, worker_l1_size:0x{:X}, "
        "l1_unreserved_base:0x{:X}, l1_small_size:0x{:X}, disable_interleaved:{}, l1_alignment:{}",
        mem_mailbox_base,
        config_.storage_core_bank_size.has_value() ? config_.storage_core_bank_size.value() : 0,
        config_.worker_l1_size,
        config_.l1_unreserved_base,
        config_.l1_small_size,
        config_.disable_interleaved,
        config_.l1_alignment);

    uint64_t small_interleaved_address_limit = config_.worker_l1_size - config_.l1_small_size;
    uint64_t small_alloc_offset = config_.l1_unreserved_base + allocatable_l1_size;
    log_debug(
        tt::LogMetal,
        "Derived partition params: storage_core_unreserved_base:0x{:X}, l1_bank_size:0x{:X}, "
        "interleaved_address_limit:0x{:X}, "
        "allocatable_l1_size:0x{:X}, small_interleaved_address_limit:0x{:X}, small_alloc_offset:0x{:X}",
        storage_core_unreserved_base,
        l1_bank_size,
        interleaved_address_limit,
        allocatable_l1_size,
        small_interleaved_address_limit,
        small_alloc_offset);
    TT_ASSERT(
        (config_.l1_unreserved_base + config_.l1_small_size) <= config_.worker_l1_size,
        "L1 small region extends past L1 size");
    l1_small_manager_ = std::make_unique<BankManager>(
        BufferType::L1_SMALL,
        small_bank_id_to_bank_offset,
        config_.l1_small_size,
        small_interleaved_address_limit,
        config_.l1_alignment,
        small_alloc_offset,
        config_.disable_interleaved);
}

L1BankingAllocator::L1BankingAllocator(const AllocatorConfig& alloc_config) : Allocator(alloc_config) {
    this->init_one_bank_per_channel();
    this->init_compute_and_storage_l1_bank_manager();
    this->validate_bank_assignments();
}

}  // namespace tt_metal

}  // namespace tt
