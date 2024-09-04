// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>
#include <functional>
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
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
    //! worker specific configuration
    CoreCoord worker_grid_size = {};
    size_t worker_l1_size = 0;
    size_t l1_bank_size = 0;
    size_t l1_small_size = 0;
    size_t trace_region_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<int, int> worker_log_to_physical_routing_x = {};
    std::unordered_map<int, int> worker_log_to_physical_routing_y = {};
    BankMapping l1_bank_remap = {}; // for remapping which l1 bank points to which bank if we assume normal row-major assignment
    CoreCoord compute_grid_size = {};
    void reset();
    ~AllocatorConfig() { reset(); }
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

constexpr static std::uint32_t STORAGE_ONLY_RESERVED_SIZE = ((MEM_MAILBOX_END + ALLOCATOR_ALIGNMENT - 1) / ALLOCATOR_ALIGNMENT) * ALLOCATOR_ALIGNMENT;
// Storage only cores only need to reserve mailbox space to hold barriers
constexpr static std::uint32_t STORAGE_ONLY_UNRESERVED_BASE = STORAGE_ONLY_RESERVED_SIZE;

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
