// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>
#include <functional>
#include "common/core_coord.hpp"
#include "hostdevcommon/common_values.hpp"
#include "dev_mem_map.h"

namespace tt::tt_metal {

// Fwd declares
struct Allocator;
namespace allocator {
    class BankManager;
}

// Setup what each core-type is
enum class AllocCoreType {
    Dispatch,
    StorageOnly,
    ComputeOnly,
    ComputeAndStore,
    Invalid,
};

using BankMapping = std::vector<uint32_t>;

//! Allocator configuration -- decouples allocation from soc-desc - Up to user to populate from soc_desc
struct AllocatorConfig {
    //! DRAM specific configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    std::vector<size_t> dram_bank_offsets = {};
    uint32_t dram_unreserved_base = 0;
    //! worker specific configuration
    uint32_t l1_unreserved_base = 0;
    CoreCoord worker_grid_size = {};
    size_t worker_l1_size = 0;
    std::optional<uint32_t> storage_core_bank_size = 0;
    size_t l1_small_size = 0;
    size_t trace_region_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<int, int> worker_log_to_physical_routing_x = {};
    std::unordered_map<int, int> worker_log_to_physical_routing_y = {};
    BankMapping l1_bank_remap = {}; // for remapping which l1 bank points to which bank if we assume normal row-major assignment
    CoreCoord compute_grid_size = {};
    uint32_t alignment = 0;
    void reset();
    ~AllocatorConfig() { reset(); }
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

namespace allocator {

struct InitAndAllocFuncs {
    std::function<void(Allocator &, const AllocatorConfig &)> init;
    std::function<uint64_t(const AllocatorConfig &, BankManager &, uint64_t, uint64_t, bool, std::optional<uint32_t> )> alloc;
};

// Holds callback functions required by allocators that specify how to initialize the bank managers and what the allocation scheme
// is for a given storage substrate
struct AllocDescriptor {
    InitAndAllocFuncs dram;
    InitAndAllocFuncs l1;
};

struct Statistics {
    size_t total_allocatable_size_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    std::vector<uint32_t> largest_free_block_addrs;  // addresses (relative to bank) that can hold the largest_free_block_bytes
};

}

}
