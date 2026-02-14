// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "l1_banking_allocator.hpp"

#include <allocator.hpp>
#include <tt_stl/assert.hpp>
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
#include <utility>
#include <vector>

#include "bank_manager.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/allocator/allocator_types.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/tt_align.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "impl/dispatch/dispatch_core_common.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

void AllocatorImpl::init_compute_and_storage_l1_bank_manager() {
    TT_FATAL(config_->worker_grid.contains(config_->compute_grid), "Compute grid must be a subset of worker grid");

    uint32_t num_l1_banks = 0;
    for (const auto& core_allocation_type : config_->core_type_from_noc_coord_table) {
        if (core_allocation_type.second == AllocCoreType::ComputeAndStore) {
            num_l1_banks++;
        }
    }

    uint32_t num_l1_small_banks = (config_->l1_small_size > 0) ? num_l1_banks : 0;

    auto logical_to_noc_coord = [this](CoreCoord logical_core) {
        TT_ASSERT(
            config_->worker_log_to_virtual_routing_x.contains(logical_core.x) and
                config_->worker_log_to_virtual_routing_y.contains(logical_core.y),
            "Cannot find log_coord=[.y={}, .x={}] in logical to routing coord lookup tables... invalid AllocatorConfig "
            "setup",
            logical_core.y,
            logical_core.x);
        CoreCoord noc_core({
            static_cast<std::size_t>(config_->worker_log_to_virtual_routing_x.at(logical_core.x)),
            static_cast<std::size_t>(config_->worker_log_to_virtual_routing_y.at(logical_core.y)),
        });
        TT_ASSERT(
            config_->core_type_from_noc_coord_table.contains(noc_core),
            "Cannot find noc-coord=[.y={}, .x={}] in core_type_from_noc_coord_table... invalid AllocatorConfig setup",
            noc_core.y,
            noc_core.x);
        return noc_core;
    };

    // Define the bank assignment here.
    std::vector<uint32_t> shuffled_bank_id = {};
    if (not config_->l1_bank_remap.empty()) {
        TT_ASSERT(
            num_l1_banks == config_->l1_bank_remap.size(),
            "override l1_bank_remap.size()={} which is not equal to the expected expected_num_l1_banks={} from "
            "soc-desc",
            config_->l1_bank_remap.size(),
            num_l1_banks);
        std::copy(config_->l1_bank_remap.begin(), config_->l1_bank_remap.end(), std::back_inserter(shuffled_bank_id));
    } else {
        // randomize remap
        for (uint32_t id = 0; id < num_l1_banks; id++) {
            shuffled_bank_id.push_back(id);
        }
        auto rng = std::default_random_engine(0);
        std::shuffle(std::begin(shuffled_bank_id), std::end(shuffled_bank_id), rng);
    }

    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset;
    std::unordered_map<uint32_t, int64_t> small_bank_id_to_bank_offset;
    if (config_->l1_small_size > 0) {
        TT_ASSERT(num_l1_small_banks > 0);
    }

    // If l1_small_size exists, then it gets the top of L1 (offset 0)
    // and the regular L1 region is offset just below it
    uint32_t bank_id = 0;
    const auto& cores = corerange_to_cores(config_->worker_grid, std::nullopt, true);
    for (const auto& logical_core : cores) {
        CoreCoord noc_core = logical_to_noc_coord(logical_core);

        if (config_->core_type_from_noc_coord_table.at(noc_core) == AllocCoreType::ComputeAndStore) {
            uint32_t remapped_bank_id = shuffled_bank_id[bank_id];
            logical_core_to_bank_ids_[BufferType::L1].insert({logical_core, {remapped_bank_id}});
            bank_id_to_logical_core_.insert({remapped_bank_id, logical_core});
            bank_id_to_bank_offset.insert({remapped_bank_id, 0});
            if (config_->l1_small_size > 0) {
                // Now map the L1-small bank id
                logical_core_to_bank_ids_[BufferType::L1_SMALL].insert({logical_core, {remapped_bank_id}});
                small_bank_id_to_bank_offset.insert({remapped_bank_id, 0});
            }
            bank_id++;
        }
    }
    TT_ASSERT(bank_id == shuffled_bank_id.size());
    TT_ASSERT(bank_id_to_bank_offset.size() == num_l1_banks);
    TT_ASSERT(small_bank_id_to_bank_offset.size() == num_l1_small_banks);

    // There is only l1_bank_size bytes available for L1 buffers to be allocated in
    uint64_t l1_bank_size = config_->worker_l1_size - config_->l1_unreserved_base;
    uint64_t interleaved_address_limit = static_cast<uint64_t>(config_->worker_l1_size - l1_bank_size);
    uint64_t allocatable_l1_size =
        static_cast<uint64_t>(config_->worker_l1_size) - config_->l1_unreserved_base - config_->l1_small_size;
    // Assuming top down allocation for L1 buffers so the allocatable memory space is the top l1_bank_size bytes of L1
    l1_manager_ = std::make_unique<BankManager>(
        BufferType::L1,
        bank_id_to_bank_offset,
        allocatable_l1_size,
        interleaved_address_limit,
        config_->l1_alignment,
        config_->l1_unreserved_base,
        config_->disable_interleaved);
    log_debug(
        tt::LogMetal,
        "Configured partition params: worker_l1_size:0x{:X}, "
        "l1_unreserved_base:0x{:X}, l1_small_size:0x{:X}, disable_interleaved:{}, l1_alignment:{}",
        config_->worker_l1_size,
        config_->l1_unreserved_base,
        config_->l1_small_size,
        config_->disable_interleaved,
        config_->l1_alignment);

    uint64_t small_interleaved_address_limit = config_->worker_l1_size - config_->l1_small_size;
    uint64_t small_alloc_offset = config_->l1_unreserved_base + allocatable_l1_size;
    log_debug(
        tt::LogMetal,
        "Derived partition params: l1_bank_size:0x{:X}, "
        "interleaved_address_limit:0x{:X}, "
        "allocatable_l1_size:0x{:X}, small_interleaved_address_limit:0x{:X}, small_alloc_offset:0x{:X}",
        l1_bank_size,
        interleaved_address_limit,
        allocatable_l1_size,
        small_interleaved_address_limit,
        small_alloc_offset);
    TT_ASSERT(
        (config_->l1_unreserved_base + config_->l1_small_size) <= config_->worker_l1_size,
        "L1 small region extends past L1 size");
    l1_small_manager_ = std::make_unique<BankManager>(
        BufferType::L1_SMALL,
        small_bank_id_to_bank_offset,
        config_->l1_small_size,
        small_interleaved_address_limit,
        config_->l1_alignment,
        small_alloc_offset,
        config_->disable_interleaved);
}

L1BankingAllocator::L1BankingAllocator(const AllocatorConfig& alloc_config) : AllocatorImpl(alloc_config) {
    this->init_one_bank_per_channel();
    this->init_compute_and_storage_l1_bank_manager();
    this->validate_bank_assignments();
}

AllocatorConfig L1BankingAllocator::generate_config(
    ChipId device_id,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    BankMapping l1_bank_remap) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device_id);
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    // Construct allocator config from soc_desc
    // Take max alignment to satisfy NoC rd/wr constraints
    // Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
    // PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
    // Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
    const auto& logical_size = soc_desc.get_grid_size(CoreType::TENSIX);
    const auto& compute_size = tt::get_compute_grid_size(device_id, num_hw_cqs, dispatch_core_config);
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_views()),
         .dram_bank_size = soc_desc.dram_view_size,
         .dram_bank_offsets = {},
         .dram_unreserved_base = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::UNRESERVED)),
         .dram_alignment = hal.get_alignment(HalMemType::DRAM),
         .l1_unreserved_base =
             static_cast<uint32_t>(align(worker_l1_unreserved_start, hal.get_alignment(HalMemType::DRAM))),
         .worker_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(logical_size.x - 1, logical_size.y - 1))),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .l1_small_size = align(l1_small_size, hal.get_alignment(HalMemType::DRAM)),
         .trace_region_size = align(trace_region_size, hal.get_alignment(HalMemType::DRAM)),
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_virtual_routing_x = cluster.get_worker_logical_to_virtual_x(device_id),
         .worker_log_to_virtual_routing_y = cluster.get_worker_logical_to_virtual_y(device_id),
         .l1_bank_remap = std::move(l1_bank_remap),
         .compute_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(compute_size.x - 1, compute_size.y - 1))),
         .l1_alignment = hal.get_alignment(HalMemType::L1),
         .disable_interleaved = false});
    TT_FATAL(
        config.l1_small_size < config.worker_l1_size - config.l1_unreserved_base,
        "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % config.l1_alignment == 0,
        "Reserved size must be aligned to L1 allocator alignment {}",
        config.l1_alignment);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_views(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const CoreCoord& core : soc_desc.get_all_cores(CoordSystem::NOC0)) {
        config.core_type_from_noc_coord_table.insert(
            {cluster.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y}),
             AllocCoreType::Invalid});
    }
    for (const CoreCoord& core : soc_desc.get_all_harvested_cores(CoordSystem::NOC0)) {
        config.core_type_from_noc_coord_table.insert(
            {cluster.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y}),
             AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        const auto noc_coord =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, core, CoreType::WORKER);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        const auto noc_coord =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    return config;
}

}  // namespace tt::tt_metal
