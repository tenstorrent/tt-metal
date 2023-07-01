#pragma once

#include <vector>
#include <cstdlib>
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"

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

using BankMapping = std::vector<u32>;

//! Allocator configuration -- decouples allocation from soc-desc - Up to user to populate from soc_desc
struct AllocatorConfig {
    //! DRAM specific configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    std::vector<size_t> dram_bank_offsets = {};
    //! worker specific configuration
    CoreCoord worker_grid_size = {};
    size_t worker_l1_size = 0;
    size_t storage_core_l1_bank_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<CoreCoord, CoreCoord> logical_to_routing_coord_lookup_table = {};
    BankMapping l1_bank_remap = {}; // for remapping which l1 bank points to which bank if we assume normal row-major assignment
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

namespace allocator {

struct InitAndAllocFuncs {
    std::function<void(Allocator &, const AllocatorConfig &)> init;
    std::function<u64(const AllocatorConfig &, BankManager &, u64, u64, bool)> alloc;
    std::function<u64(const AllocatorConfig &, BankManager &, u64, u64, u64)> alloc_at_addr;
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
    std::vector<u32> largest_free_block_addrs;  // addresses (relative to bank) that can hold the largest_free_block_bytes
};

}

}
